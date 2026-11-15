"""
Converts raw sim observations (uint8 HWC images, float joints) into Diffusion Policy inputs.
Shared by the training dataset and the eval env so both see identical inputs.

Besides the images it can produce two things the fusion models need:

* a depth validity channel: with `mask_channel: true` on the depth key, the 3 channels are [depth, depth > 0, depth]
  instead of the depth repeated three times, so the encoder can tell "no reading" (0) from "very close" (also 0);
* reliability cues (`cue` key, type low_dim): per-frame scalars computed from the *raw* sensor data before any
  normalization, two per modality, that say how much each sensor can be trusted right now (see compute_cues).
  The cue is computed here, not stored in the zarr, so train and eval use the same code path.
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


def compute_cues(raw, roi=DEFAULT_ROI):
    """Reliability cue vector (..., CUE_DIM) from a raw observation dict (arrays (..., H, W, C)).
    Modalities missing from `raw` get zeros, so the layout is fixed whatever sensors a run uses.
    Uses thermal_raw (counts, before normalization) when present, else the 8-bit thermal image."""
    ref = next(v for k, v in raw.items() if k in ("rgb", "depth", "thermal", "thermal_raw"))
    lead = np.shape(ref)[:-3]
    out = np.zeros(lead + (CUE_DIM,), np.float32)
    if "rgb" in raw:
        out[..., 0:2] = rgb_cues(raw["rgb"], roi)
    if "depth" in raw:
        out[..., 2:4] = depth_cues(raw["depth"], roi)
    if "thermal_raw" in raw:
        out[..., 4:6] = thermal_cues(raw["thermal_raw"], roi, THERMAL_NOISE_COUNTS)
    elif "thermal" in raw:
        _warn_once("thermal8", "compute_cues: no thermal_raw, computing the thermal cue on the normalized 8-bit image")
        out[..., 4:6] = thermal_cues(raw["thermal"], roi, THERMAL_NOISE_UINT8)
    return np.nan_to_num(out)


def cue_roi(obs_meta):
    """ROI for the cues: `roi` on the cue key of shape_meta, else DEFAULT_ROI."""
    attr = obs_meta.get("cue")
    roi = attr.get("roi") if attr is not None and hasattr(attr, "get") else None
    return tuple(int(v) for v in roi) if roi is not None else DEFAULT_ROI


def to_policy_obs(raw, obs_meta):
    """raw: dict of (..., H, W, C) uint8 images / (..., D) vectors. obs_meta: shape_meta["obs"]."""
    out = {}
    for key, attr in obs_meta.items():
        if key == "cue":
            out[key] = compute_cues(raw, cue_roi(obs_meta))
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
