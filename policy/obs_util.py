"""
Converts raw sim observations (uint8 HWC images, float joints) into Diffusion Policy inputs.
Shared by the training dataset and the eval env so both see identical inputs.

Besides the images it can produce two things the fusion models need:

* a depth validity channel: with `mask_channel: true` on the depth key, the 3 channels are [depth, depth > 0, depth]
  instead of the depth repeated three times, so the encoder can tell "no reading" (0) from "very close" (also 0);
* reliability cues (`cue` key, type low_dim): per-frame scalars computed from the *raw* sensor data before any
  normalization, two per modality, that say how much each sensor can be trusted right now (see compute_cues).
  The cue is computed here, not stored in the zarr, so train and eval use the same code path.
  Optional 7th cue (`depth_object_ref: <.npy>` on the cue key, cue shape [7]): depth_object = fraction of ROI pixels
  with a reading more than `depth_object_thr` (8-bit depth units, 7.8 mm each) closer than the empty-table reference
  depth of the fixed camera, i.e. "does the depth sensor see an object"; it drops when the sensor sees through the
  glass. Without the key the cue is the 6-vector above, unchanged.
  Optional target-visibility entries (`vis_thr: [depth, thermal]` on the cue key, 4 more entries at the end):
  [depth step, thermal step, depth visible, thermal visible] measured where a glass mask says the glass is
  (visibility_cues); the last two are 0/1 flags, step > threshold. The mask is `glass_mask` in the raw dict when the
  dataset supplies it (policy/make_glass_masks.py), else predicted from the frame with `seg_ckpt` (policy/glass_seg.py).
  `mask_noise: {drop: p, shift: k}` on the cue key degrades the predicted mask (degrade_mask; evaluation only, set by
  eval.py --mask_noise, never in a training config).
  `seg_mods: [rgb, depth]` on the cue key: the sensors the segmentation network may use for a predicted mask (default:
  all it was trained with; evaluation only, set by eval.py --seg_mods).
* the glass location (`glass_loc` key, type low_dim, shape [5]): [found, centre x, centre y, width, height] of the glass
  mask of the frame as fractions of the frame (glass_location), so the policy is told where the segmentation network
  sees the glass. The dataset computes it from stored masks (`glass_mask: <.npy>` on the glass_loc key, else the cue
  key's masks); the eval env predicts the mask with `seg_ckpt` and optionally `seg_mods` of the glass_loc key.
"""
import warnings

import numpy as np

MODALITIES = ("rgb", "depth", "thermal")          # fixed order of the cue vector
CUES_PER_MODALITY = 2
CUE_DIM = CUES_PER_MODALITY * len(MODALITIES)     # 6
CUE_NAMES = ("rgb_brightness", "rgb_sharpness",   # mean gray level; log variance of the Laplacian (blur / noise)
             "depth_valid", "depth_rough",        # fraction of pixels with a reading; mean |Laplacian| on them
             "thermal_contrast", "thermal_extent")  # log((p99 - p1) / noise floor); fraction of pixels far from median

# Rows / columns of the 96x96 frame the cues are computed on: the lower part where the table is, so the robot arm
# (always 5-12 degC warmer than ambient, entering from the top) does not make every thermal frame look high-contrast.
# Check against results/hard_scenario/levels_grid.png when the camera changes.
DEFAULT_ROI = (32, 96, 0, 96)
# Noise std of the simulated thermal camera in raw counts (sim/configs/thermal/calibrated_v1.yaml, 160x120); the
# bilinear resize to 96x96 lowers it a little, which only shifts the log scale of the contrast cue.
THERMAL_NOISE_COUNTS = 2.73
# When only the 8-bit normalized thermal image is available (old zarr without thermal_raw), the cue is computed on it
# with this noise floor in gray levels. It is a weak proxy for min-max normalized frames, which stretch any noise to
# the full range; prefer thermal_raw.
THERMAL_NOISE_UINT8 = 2.0
NOISE_P99_P1 = 4.65                 # p99 - p1 of a unit Gaussian
# Optional cue after the fixed six (enabled per run, see the module docstring)
DEPTH_OBJECT_THR = 8                # 8-bit depth units (255 = 2 m): 6 cm above the table

_warned = set()


def _warn_once(key, msg):
    if key not in _warned:
        _warned.add(key)
        warnings.warn(msg)


def _gray(img):
    """(..., H, W, C) -> (..., H, W) float in [0, 1]."""
    x = np.asarray(img, dtype=np.float32)
    if x.ndim >= 3 and x.shape[-1] in (1, 3):
        x = x.mean(-1)
    return x / 255.0


def _laplacian(g):
    """4-neighbour Laplacian on the interior of (..., H, W)."""
    return (g[..., :-2, 1:-1] + g[..., 2:, 1:-1] + g[..., 1:-1, :-2] + g[..., 1:-1, 2:] - 4 * g[..., 1:-1, 1:-1])


def _roi(x, roi):
    y0, y1, x0, x1 = roi
    return x[..., y0:y1, x0:x1]


def rgb_cues(rgb, roi=DEFAULT_ROI):
    """(..., H, W, 3) uint8 -> (..., 2): brightness, sharpness."""
    g = _roi(_gray(rgb), roi)
    brightness = g.mean((-2, -1))
    sharpness = np.log1p(1e3 * _laplacian(g).var((-2, -1)))
    return np.stack([brightness, sharpness], -1)


def depth_cues(depth8, roi=DEFAULT_ROI):
    """(..., H, W, 1) uint8 (0 = no reading) -> (..., 2): valid fraction, roughness of the valid readings."""
    d = _roi(_gray(depth8), roi)
    valid = d > 0
    valid_frac = valid.mean((-2, -1))
    lap = np.abs(_laplacian(d))
    v = valid[..., 1:-1, 1:-1]
    rough = (lap * v).sum((-2, -1)) / np.maximum(v.sum((-2, -1)), 1)
    return np.stack([valid_frac, np.log1p(10 * rough)], -1)


def thermal_cues(raw, roi=DEFAULT_ROI, noise=THERMAL_NOISE_COUNTS):
    """(..., H, W, 1) raw counts (uint16) or 8-bit image -> (..., 2): contrast, extent.
    contrast = log(1 + (p99 - p1) / (4.65 noise)): the p99 - p1 spread of pure Gaussian noise is 4.65 sigma, so the
    ratio is ~1 (log 2 = 0.69) at thermal crossover and grows with the object's temperature contrast (>2.5 for 30 degC).
    extent = fraction of pixels more than 3 noise stds from the frame median (how much of the view is "an object")."""
    x = np.asarray(raw, dtype=np.float32)
    if x.shape[-1] == 1:
        x = x[..., 0]
    x = _roi(x, roi)
    flat = x.reshape(x.shape[:-2] + (-1,))
    p1, p99 = np.percentile(flat, [1, 99], axis=-1)
    med = np.median(flat, axis=-1)
    contrast = np.log1p((p99 - p1) / (NOISE_P99_P1 * noise))
    extent = (np.abs(flat - med[..., None]) > 3 * noise).mean(-1)
    return np.stack([contrast, extent], -1)


def depth_object_cue(depth8, ref8, roi=DEFAULT_ROI, thr=DEPTH_OBJECT_THR):
    """(..., H, W, 1) uint8 depth, (H, W) uint8 empty-table reference -> (...,): fraction of ROI pixels with a reading
    more than thr units closer than the table."""
    d = _roi(np.asarray(depth8, np.float32)[..., 0], roi)
    r = _roi(np.asarray(ref8, np.float32), roi)
    return ((d > 0) & (r - d > thr)).mean((-2, -1))


def visibility_cues(depth8, thermal, mask, noise=THERMAL_NOISE_COUNTS, ring=5, min_pixels=5):
    """Does each sensor show the target? Measured where a glass mask says the glass is, against a ring around it.
    depth8 (..., H, W, 1) uint8, thermal (..., H, W, 1) raw counts (or the 8-bit image with its noise floor), mask
    (..., H, W) bool or (..., H, W, 1) uint8 -> (..., 3):
      depth step       median depth of the ring minus median depth on the mask, 8-bit units (7.8 mm each): > 0 when
                       the sensor sees something standing there, ~0 when it sees through the glass to the table;
      thermal step     |median on the mask - median of the ring| / noise: ~1-10 at thermal crossover, > 40 otherwise;
      found            1 if the mask has at least min_pixels pixels, else 0 (the two steps are then 0)."""
    from scipy.ndimage import maximum_filter
    m = np.asarray(mask)
    if m.ndim == np.ndim(depth8):
        m = m[..., 0]
    m = m.astype(bool)
    lead = m.shape[:-2]
    m2 = m.reshape((-1,) + m.shape[-2:])
    d = np.asarray(depth8, np.float32)[..., 0].reshape(m2.shape)
    t = np.asarray(thermal, np.float32)[..., 0].reshape(m2.shape)
    around = maximum_filter(m2, size=(1, 2 * ring + 1, 2 * ring + 1)) & ~m2
    out = np.zeros((len(m2), 3), np.float32)
    for i in range(len(m2)):
        if m2[i].sum() < min_pixels or not around[i].any():
            continue
        di, do = d[i][m2[i]], d[i][around[i]]
        di, do = di[di > 0], do[do > 0]
        step = float(np.median(do) - np.median(di)) if len(di) and len(do) else 0.0
        out[i] = (step, abs(float(np.median(t[i][m2[i]]) - np.median(t[i][around[i]]))) / noise, 1.0)
    return out.reshape(lead + (3,))


def degrade_mask(mask, depth8, drop=0.0, shift=0, seed=0):
    """Errors of the glass mask for robustness tests: each frame's mask is emptied with probability `drop` (the cues
    then count the target as visible in both sensors) and otherwise moved by an offset drawn uniformly from
    [-shift, shift]^2 pixels, vacated pixels empty. mask (..., H, W) bool, depth8 (..., H, W, 1). The draw is seeded
    by the frame's depth image, so a rollout is reproducible and parallel envs draw independently."""
    import zlib
    m = np.array(mask, dtype=bool)
    flat = m.reshape((-1,) + m.shape[-2:])
    d = np.asarray(depth8).reshape((len(flat), -1))
    H, W = flat.shape[-2:]
    for i in range(len(flat)):
        rng = np.random.default_rng([int(seed), zlib.crc32(np.ascontiguousarray(d[i]).tobytes())])
        if rng.random() < drop:
            flat[i] = False
        elif shift:
            dy, dx = (int(v) for v in rng.integers(-int(shift), int(shift) + 1, 2))
            moved = np.zeros_like(flat[i])
            moved[max(dy, 0):H + min(dy, 0), max(dx, 0):W + min(dx, 0)] = \
                flat[i][max(-dy, 0):H + min(-dy, 0), max(-dx, 0):W + min(-dx, 0)]
            flat[i] = moved
    return flat.reshape(m.shape)


VIS_DIM = 4                         # depth step, thermal step, depth visible, thermal visible
LOC_DIM = 5                         # found, centre x, centre y, width, height of the glass mask


def glass_location(mask, min_pixels=5):
    """Where the glass mask is. mask (..., H, W) bool or (..., H, W, 1) uint8 -> (..., LOC_DIM) float32:
    [found, centre x, centre y, width, height]; the centre is the centroid and width / height the bounding box, all as
    fractions of the frame (x to the right, y down). Zeros when the mask has fewer than min_pixels pixels."""
    m = np.asarray(mask)
    if m.shape[-1] == 1:
        m = m[..., 0]
    m = m.astype(bool)
    lead, (H, W) = m.shape[:-2], m.shape[-2:]
    f = m.reshape((-1, H, W))
    out = np.zeros((len(f), LOC_DIM), np.float32)
    cols, rows = f.sum(1), f.sum(2)                   # pixels per column (n, W) and per row (n, H)
    px = cols.sum(1)
    ok = px >= min_pixels
    if ok.any():
        c, r, n = cols[ok], rows[ok], px[ok].astype(np.float32)
        out[ok, 0] = 1.0
        out[ok, 1] = (c * (np.arange(W) + 0.5)).sum(1) / n / W
        out[ok, 2] = (r * (np.arange(H) + 0.5)).sum(1) / n / H
        out[ok, 3] = (W - np.argmax(c[:, ::-1] > 0, 1) - np.argmax(c > 0, 1)) / W
        out[ok, 4] = (H - np.argmax(r[:, ::-1] > 0, 1) - np.argmax(r > 0, 1)) / H
    return out.reshape(lead + (LOC_DIM,))


def compute_cues(raw, roi=DEFAULT_ROI, depth_ref=None, depth_thr=DEPTH_OBJECT_THR, vis=None):
    """Reliability cue vector (..., CUE_DIM) from a raw observation dict (arrays (..., H, W, C)).
    Modalities missing from `raw` get zeros, so the layout is fixed whatever sensors a run uses.
    Uses thermal_raw (counts, before normalization) when present, else the 8-bit thermal image.
    With depth_ref ((H, W) uint8), the vector has a 7th entry, depth_object_cue.
    With vis = dict(thr=(depth, thermal), seg_ckpt=path or None) the last VIS_DIM entries are the target-visibility
    cues; a frame whose mask is empty counts as visible in both sensors (nothing is switched off). vis["mask_noise"]
    (degrade_mask keyword arguments) is applied to a predicted mask only, never to a mask stored with the demos."""
    ref = next(v for k, v in raw.items() if k in ("rgb", "depth", "thermal", "thermal_raw"))
    lead = np.shape(ref)[:-3]
    n_base = CUE_DIM + (depth_ref is not None)
    out = np.zeros(lead + (n_base + (VIS_DIM if vis else 0),), np.float32)
    if "rgb" in raw:
        out[..., 0:2] = rgb_cues(raw["rgb"], roi)
    if "depth" in raw:
        out[..., 2:4] = depth_cues(raw["depth"], roi)
    if "thermal_raw" in raw:
        out[..., 4:6] = thermal_cues(raw["thermal_raw"], roi, THERMAL_NOISE_COUNTS)
    elif "thermal" in raw:
        _warn_once("thermal8", "compute_cues: no thermal_raw, computing the thermal cue on the normalized 8-bit image")
        out[..., 4:6] = thermal_cues(raw["thermal"], roi, THERMAL_NOISE_UINT8)
    if depth_ref is not None and "depth" in raw:
        out[..., CUE_DIM] = depth_object_cue(raw["depth"], depth_ref, roi, depth_thr)
    if vis:
        mask = raw.get("glass_mask")
        if mask is None:
            from policy.glass_seg import predict_glass_mask
            mask = predict_glass_mask(raw, vis["seg_ckpt"], mods=vis.get("seg_mods"))
            if vis.get("mask_noise"):
                mask = degrade_mask(mask, raw["depth"], **vis["mask_noise"])
        if "thermal_raw" in raw:
            v = visibility_cues(raw["depth"], raw["thermal_raw"], mask, THERMAL_NOISE_COUNTS)
        elif "thermal" in raw:
            v = visibility_cues(raw["depth"], raw["thermal"], mask, THERMAL_NOISE_UINT8)
        else:       # no thermal image in the sample (a policy trained on rgb + depth): only the depth entries count
            v = visibility_cues(raw["depth"], np.zeros_like(raw["depth"]), mask, THERMAL_NOISE_UINT8)
        found = v[..., 2] > 0
        out[..., n_base:n_base + 2] = v[..., :2]
        out[..., n_base + 2] = np.where(found, v[..., 0] > vis["thr"][0], True)
        out[..., n_base + 3] = np.where(found, v[..., 1] > vis["thr"][1], True)
    return np.nan_to_num(out)


def cue_roi(obs_meta):
    """ROI for the cues: `roi` on the cue key of shape_meta, else DEFAULT_ROI."""
    attr = obs_meta.get("cue")
    roi = attr.get("roi") if attr is not None and hasattr(attr, "get") else None
    return tuple(int(v) for v in roi) if roi is not None else DEFAULT_ROI


_depth_refs = {}


def cue_opts(obs_meta):
    """compute_cues keyword arguments beyond the ROI from the cue key of shape_meta: depth_object_ref / _thr, and
    vis_thr / seg_ckpt for the target-visibility entries."""
    attr = obs_meta.get("cue")
    if attr is None or not hasattr(attr, "get"):
        return {}
    opts = {}
    path = attr.get("depth_object_ref")
    if path is not None:
        if path not in _depth_refs:
            _depth_refs[path] = np.load(path)
        opts.update(depth_ref=_depth_refs[path], depth_thr=int(attr.get("depth_object_thr", DEPTH_OBJECT_THR)))
    thr = attr.get("vis_thr")
    if thr is not None:
        opts["vis"] = {"thr": (float(thr[0]), float(thr[1])), "seg_ckpt": attr.get("seg_ckpt")}
        if attr.get("vis_raw"):      # the two measurements reach the encoder unnormalized (route=soft)
            opts["vis"]["raw"] = True
        if attr.get("mask_noise"):   # robustness tests only (eval.py --mask_noise)
            opts["vis"]["mask_noise"] = {k: float(v) for k, v in dict(attr.get("mask_noise")).items()}
        if attr.get("seg_mods") is not None:   # sensors the segmentation network may use (eval.py --seg_mods)
            opts["vis"]["seg_mods"] = [str(m) for m in attr.get("seg_mods")]
    return opts


def _loc_mask(raw, obs_meta):
    """Predicted glass mask for the glass_loc observation, and whether the cue can use the same one (same checkpoint
    and sensors, no mask noise), so that the eval env runs the segmentation network once per frame."""
    from policy.glass_seg import predict_glass_mask
    attr = obs_meta["glass_loc"]
    mods = [str(m) for m in attr.get("seg_mods")] if attr.get("seg_mods") is not None else None
    mask = predict_glass_mask(raw, attr["seg_ckpt"], mods=mods)
    vis = cue_opts(obs_meta).get("vis") if "cue" in obs_meta else None
    shared = bool(vis) and not vis.get("mask_noise") and str(vis.get("seg_ckpt")) == str(attr["seg_ckpt"]) \
        and vis.get("seg_mods") == mods
    return mask, shared


def to_policy_obs(raw, obs_meta):
    """raw: dict of (..., H, W, C) uint8 images / (..., D) vectors. obs_meta: shape_meta["obs"]."""
    out = {}
    if "glass_loc" in obs_meta and "glass_loc" not in raw:    # eval env: from the predicted mask of the frame
        mask, shared = _loc_mask(raw, obs_meta)
        raw = dict(raw, glass_loc=glass_location(mask))
        if shared and raw.get("glass_mask") is None:
            raw["glass_mask"] = mask
    for key, attr in obs_meta.items():
        if key == "cue":
            out[key] = compute_cues(raw, cue_roi(obs_meta), **cue_opts(obs_meta))
            continue
        if key == "glass_loc":
            out[key] = np.asarray(raw[key], np.float32)
            continue
        x = raw[key]
        if attr.get("type", "low_dim") == "rgb":
            if x.shape[-1] == 1:
                if key == "depth" and attr.get("mask_channel", False):
                    mask = (x > 0).astype(x.dtype) * 255           # 1 channel of "has a reading"
                    x = np.concatenate([x, mask, x], axis=-1)
                else:
                    x = np.repeat(x, 3, axis=-1)                    # depth / thermal -> 3 channels for the ResNet encoder
            x = np.moveaxis(x, -1, -3).astype(np.float32) / 255
        else:
            x = x.astype(np.float32)
        out[key] = x
    return out
