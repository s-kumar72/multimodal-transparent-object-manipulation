"""
Is glass of a given visibility as hard to find in the simulator as in real sensor images?
(results/hard_scenario/bench_v2/README.md, section 11; everything below is fixed there before it was run.)

  python real_data/findability.py real    [--root ~/datasets/transparent/unified]   windows of real glass regions
  python real_data/findability.py sim     [--level V2_place] [-n 3000]              the same from first frames of the simulator
  python real_data/findability.py fit     [--device cuda]                           classifiers: out of fold and across the domains
  python real_data/findability.py readout                                           the tables

Second design (section 11, after the first failed its control: in the simulator the surroundings of the two windows
differ and give the answer away). In the simulator the pair is the same window with the glass and with the glass
moved out of the scene:
  python real_data/findability.py simpair [--level V2_place] [-n 3000]
  python real_data/findability.py fit --sim simpair
  python real_data/findability.py readout2

Section 12, the optical colour model of the glass (sim/glass_optics.py):
  python real_data/findability.py realrgb                      real colour pairs of all TRansPose frames, with the region's mask
  python real_data/findability.py look                         k_s, w_rim, rho of the model from real images (not from a judge)
  python real_data/findability.py simpair --config sim/configs/V2_place_opt.yaml --out data/real_check/simpair_opt.npz
  python real_data/findability.py judge --which A --sims simpair_opt     the judge of the development rounds
  python real_data/findability.py readout3                     the final read-out (judge B, once)

Unit: a glass region of a frame, a square window of 1.5 times the longer side of its bounding box around it and a
background window of the same size in the same frame (centre 1.0 to 2.5 window sides away, at most 5% glass), both at
32 x 32. Per sensor a small network tells the two apart; the AUC is read per bin of the region's visibility number
(real_data/glass_visibility.py). Inputs, the same for both domains:
  rgb      three channels in 0..1
  depth    (reading - median reading of the window) / 10 cm clipped to +-1 over the pixels with a reading, and the share
           of pixels without one
  thermal  (raw count - median of the window) / camera noise (0.89 counts) / 100 clipped to +-1
"""
import json
import os
import sys
from pathlib import Path

import click
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from real_data.glass_visibility import (BEHIND_MM, TRANSPOSE_NOISE, depth_stats, glass_regions, luminance,  # noqa: E402
                                        rgb_contrast, thermal_contrast)

SIZE, K, LO, HI, COVER, TRIES = 32, 1.5, 1.0, 2.5, 0.05, 60
SENSORS = {"rgb": 3, "depth": 2, "thermal": 1}
BINS = {"rgb": ("v_rgb", "grey levels", [(-np.inf, 3, "below 3"), (3, 10, "3 to 10"), (10, 20, "10 to 20"),
                                         (20, 45, "20 to 45"), (45, np.inf, "above 45")]),
        "depth": ("v_missing", "share of pixels missing", [(0.5, np.inf, "hole (0.5 or more)"),
                                                          (0.05, 0.5, "partial (0.05 to 0.5)"),
                                                          (-np.inf, 0.05, "see-through (below 0.05)")]),
        "thermal": ("v_thermal", "noise units", [(-np.inf, 3, "below 3"), (3, 10, "3 to 10"), (10, 30, "10 to 30"),
                                                 (30, 103, "30 to 103"), (103, np.inf, "above 103")])}
BLIND = {"rgb": "below 3", "depth": "see-through (below 0.05)", "thermal": "below 3"}   # the simulator's blind states


# ------------------------------------------------------------------------------------------------ windows
def windows(mask_all, reg, rng):
    """(y0, x0, ny, nx, side) of the glass window and its background window, or None and why not."""
    h, w = reg.shape
    ys, xs = np.nonzero(reg)
    s = int(round(K * max(ys.max() - ys.min() + 1, xs.max() - xs.min() + 1)))
    y0, x0 = int(round((ys.max() + ys.min()) / 2 - s / 2)), int(round((xs.max() + xs.min()) / 2 - s / 2))
    if s > min(h, w) or y0 < 0 or x0 < 0 or y0 + s > h or x0 + s > w:
        return None, "nofit"
    for _ in range(TRIES):
        r, a = s * rng.uniform(LO, HI), rng.uniform(0, 2 * np.pi)
        ny, nx = int(round(y0 + r * np.sin(a))), int(round(x0 + r * np.cos(a)))
        if ny < 0 or nx < 0 or ny + s > h or nx + s > w:
            continue
        if mask_all[ny:ny + s, nx:nx + s].mean() <= COVER:
            return (y0, x0, ny, nx, s), "pair"
    return None, "noneg"


def fit32(a):
    """(h, w) or (h, w, c) -> (c, 32, 32): area mean when shrinking, bilinear when enlarging."""
    import torch
    import torch.nn.functional as F
    t = torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32))
    t = t[None, None] if t.ndim == 2 else t.permute(2, 0, 1)[None]
    if a.shape[0] >= SIZE:
        t = F.interpolate(t, (SIZE, SIZE), mode="area")
    else:
        t = F.interpolate(t, (SIZE, SIZE), mode="bilinear", align_corners=False)
    return t[0].numpy()


def rgb_win(rgb, y, x, s):
    c = rgb[y:y + s, x:x + s].astype(np.float32) / 255.0
    return fit32(c if c.ndim == 3 else np.repeat(c[..., None], 3, -1))


def _relative(v, valid, scale):
    """Value minus the window's median over valid pixels, / scale, clipped; mean over the valid pixels of each cell."""
    rel = np.zeros(v.shape, np.float32)
    if valid.any():
        rel[valid] = np.clip((v[valid] - np.median(v[valid])) / scale, -1, 1)
    share = fit32(valid.astype(np.float32))
    out = np.where(share > 1e-3, fit32(rel) / np.maximum(share, 1e-3), 0.0)
    return np.clip(out, -1, 1).astype(np.float32), share


def depth_win(depth_m, y, x, s):
    d = depth_m[y:y + s, x:x + s]
    rel, share = _relative(d, d > 0, 0.10)
    return np.concatenate([rel, 1.0 - share])


def thermal_win(counts, y, x, s):
    t = counts[y:y + s, x:x + s]
    return _relative(t, t > 0, TRANSPOSE_NOISE * 100.0)[0]


def _save(out, recs, counts, extra):
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    n = len(recs)
    arr = {f"x_{k}": np.zeros((n, 2, c, SIZE, SIZE), np.float16) for k, c in SENSORS.items()}
    for k in SENSORS:
        arr[f"has_{k}"] = np.zeros(n, bool)
    for k in ("v_rgb", "v_missing", "v_behind", "v_thermal"):
        arr[k] = np.full(n, np.nan, np.float32)
    for i, r in enumerate(recs):
        for k in SENSORS:
            if k in r:
                arr[f"x_{k}"][i], arr[f"has_{k}"][i] = r[k], True
        for k in ("v_rgb", "v_missing", "v_behind", "v_thermal"):
            if r.get(k) is not None:
                arr[k][i] = r[k]
    arr["group"] = np.array([r["group"] for r in recs])
    tmp = out.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, **arr)
    os.replace(tmp, out)
    json.dump(dict(pairs=n, regions=counts, **extra), open(out.with_suffix(".json"), "w"), indent=1)
    print(f"{n} pairs -> {out}; regions: {counts}")


# ------------------------------------------------------------------------------------------------ real
@click.group()
def cli():
    pass


@cli.command()
@click.option("--root", default=os.path.expanduser("~/datasets/transparent/unified"), show_default=True)
@click.option("--out", default="data/real_check/real.npz", show_default=True)
def real(root, out):
    import pandas as pd
    from PIL import Image
    root = Path(root)
    man = pd.read_parquet(root / "manifest.parquet")
    recs, counts, frames = [], {}, {}
    for di, (ds, per) in enumerate((("transpose", 126), ("trobi", None))):
        rows = man[(man.dataset == ds) & man.mask_path.notna() & (man.glass_frac > 0)]
        if per:
            rows = pd.concat([g.sample(min(len(g), per), random_state=0) for _, g in rows.groupby("group")])
        frames[ds], c = len(rows), dict(pair=0, nofit=0, noneg=0)
        for fi, r in enumerate(rows.itertuples()):
            mk = np.asarray(Image.open(root / r.mask_path))
            mk = (mk[..., 0] if mk.ndim == 3 else mk) == 1
            regs = glass_regions(mk)
            if not regs:
                continue
            rgb = np.asarray(Image.open(root / r.rgb_path))
            dep = np.asarray(Image.open(root / r.depth_path)).astype(np.float32)
            gt = np.asarray(Image.open(root / r.depth_gt_path)).astype(np.float32) if isinstance(r.depth_gt_path, str) else None
            th = np.asarray(Image.open(root / r.thermal_path)).astype(np.float32) if isinstance(r.thermal_path, str) else None
            lum = luminance(rgb)
            for ri, (reg, ring) in enumerate(regs):
                w, why = windows(mk, reg, np.random.default_rng([0, di, fi, ri]))
                c[why] += 1
                if w is None:
                    continue
                y0, x0, ny, nx, s = w
                st = depth_stats(dep, reg, gt)
                rec = dict(group=f"{ds}/{r.group}", v_missing=st["missing"], v_behind=st.get("behind"),
                           depth=np.stack([depth_win(dep / 1000.0, y0, x0, s), depth_win(dep / 1000.0, ny, nx, s)]))
                if ds == "transpose":      # colour and thermal: the tabletop objects only (T-ROBI is a grey camera)
                    rec.update(v_rgb=rgb_contrast(lum, reg, ring),
                               rgb=np.stack([rgb_win(rgb, y0, x0, s), rgb_win(rgb, ny, nx, s)]))
                    tc = thermal_contrast(th, reg, ring) if th is not None else None
                    if tc is not None:
                        rec.update(v_thermal=tc / TRANSPOSE_NOISE,
                                   thermal=np.stack([thermal_win(th, y0, x0, s), thermal_win(th, ny, nx, s)]))
                recs.append(rec)
        counts[ds] = c
    _save(out, recs, counts, dict(frames=frames))


# ------------------------------------------------------------------------------------------------ simulator
@cli.command()
@click.option("--level", default="V2_place", show_default=True)
@click.option("-n", default=3000, show_default=True, help="episodes (consecutive seeds from 0)")
@click.option("--out", default="data/real_check/sim.npz", show_default=True)
def sim(level, n, out):
    sys.path.insert(0, "sim")
    from env import GlassPickEnv
    from testsets import manifest
    env = GlassPickEnv(config=manifest()["levels"][level])
    s_ = env.sensors
    recs, c, cells = [], dict(pair=0, nofit=0, noneg=0, noregion=0), []
    for seed in range(n):
        env.reset(seed)
        rgb, depth, _, raw = s_.render(env.data, env.temps, env.emis, env.t_env, np.random.default_rng(seed), env.sensor_scene)
        r = s_.r
        r.enable_depth_rendering()
        r.update_scene(env.data, camera=s_.camera, scene_option=s_.show_all)
        true = r.render().copy()
        r.disable_depth_rendering()
        r.enable_segmentation_rendering()
        r.update_scene(env.data, camera=s_.camera, scene_option=s_.show_all)
        seg = r.render()[..., 0].copy()
        r.disable_segmentation_rendering()
        glass = np.isin(seg, [g for g in s_.transparent_geoms if env.model.geom_bodyid[g] == env.model.body(env.obj).id])
        every = np.isin(seg, list(s_.transparent_geoms))
        regs = glass_regions(glass)
        if not regs:
            c["noregion"] += 1
            continue
        reg, ring = max(regs, key=lambda x: x[0].sum())
        w, why = windows(every, reg, np.random.default_rng([1, seed]))
        c[why] += 1
        if w is None:
            continue
        y0, x0, ny, nx, s = w
        d, t = depth[reg], true[reg]
        rec = dict(group=str(seed), v_rgb=rgb_contrast(luminance(rgb), reg, ring), v_missing=float((d == 0).mean()),
                   v_behind=float(((d - t > BEHIND_MM / 1000) & (d > 0)).mean()),
                   rgb=np.stack([rgb_win(rgb, y0, x0, s), rgb_win(rgb, ny, nx, s)]),
                   depth=np.stack([depth_win(depth, y0, x0, s), depth_win(depth, ny, nx, s)]))
        f = glass.shape[0] // raw.shape[0]
        reg_t, ring_t = reg[::f, ::f], ring[::f, ::f]
        if reg_t.sum() >= 20 and ring_t.sum() >= 20:
            a = [int(round(v / f)) for v in (y0, x0, ny, nx)]
            st = max(2, int(round(s / f)))
            a = [min(v, lim - st) for v, lim in zip(a, (raw.shape[0], raw.shape[1]) * 2)]
            rec.update(v_thermal=abs(float(np.median(raw[reg_t]) - np.median(raw[ring_t]))) / TRANSPOSE_NOISE,
                       thermal=np.stack([thermal_win(raw, a[0], a[1], st), thermal_win(raw, a[2], a[3], st)]))
        recs.append(rec)
        cells.append(int(env.episode_info["cell"]))
    env.close()
    _save(out, recs, {level: c}, dict(level=level, episodes=n, cells=cells))


@cli.command()
@click.option("--level", default="V2_place", show_default=True)
@click.option("-n", default=3000, show_default=True, help="episodes (consecutive seeds from 0)")
@click.option("--out", default="data/real_check/simpair.npz", show_default=True)
@click.option("--config", default=None, help="a level file in place of --level (a draft that is not in the manifest)")
@click.option("--optical", default=None, help='JSON overrides of glass_optics.PARAMS, e.g. \'{"F0": 0, "k_s": 1.2}\' (a variant)')
def simpair(level, n, out, config, optical):
    """The window around the glass, and the same window of the same scene rendered with the glass 10 m away."""
    import mujoco
    sys.path.insert(0, "sim")
    from env import GlassPickEnv
    from testsets import manifest
    env = GlassPickEnv(config=config or manifest()["levels"][level])
    level = env.cfg.name if config else level
    if optical:
        env.cfg.optical = env.sensors.cfg.optical = dict(env.cfg.optical or {}, **json.loads(optical))
        level += " " + optical
    s_ = env.sensors

    def frame(rng):
        return s_.render(env.data, env.temps, env.emis, env.t_env, rng, env.sensor_scene)

    recs, c, cells = [], dict(pair=0, nofit=0, noregion=0), []
    for seed in range(n):
        env.reset(seed)
        rgb, depth, _, raw = frame(np.random.default_rng([2, seed, 0]))
        r = s_.r
        r.enable_depth_rendering()
        r.update_scene(env.data, camera=s_.camera, scene_option=s_.show_all)
        true = r.render().copy()
        r.disable_depth_rendering()
        r.enable_segmentation_rendering()
        r.update_scene(env.data, camera=s_.camera, scene_option=s_.show_all)
        seg = r.render()[..., 0].copy()
        r.disable_segmentation_rendering()
        glass = np.isin(seg, [g for g in s_.transparent_geoms if env.model.geom_bodyid[g] == env.model.body(env.obj).id])
        regs = glass_regions(glass)
        if not regs:
            c["noregion"] += 1
            continue
        reg, ring = max(regs, key=lambda x: x[0].sum())
        h, w = reg.shape
        ys, xs = np.nonzero(reg)
        s = int(round(K * max(ys.max() - ys.min() + 1, xs.max() - xs.min() + 1)))
        y0, x0 = int(round((ys.max() + ys.min()) / 2 - s / 2)), int(round((xs.max() + xs.min()) / 2 - s / 2))
        if s > min(h, w) or y0 < 0 or x0 < 0 or y0 + s > h or x0 + s > w:
            c["nofit"] += 1
            continue
        c["pair"] += 1
        q = env.obj_q[env.obj]
        env.data.qpos[q] += 10.0                      # the glass out of the scene; nothing else changes
        mujoco.mj_forward(env.model, env.data)
        rgb0, depth0, _, raw0 = frame(np.random.default_rng([2, seed, 1]))
        d, t = depth[reg], true[reg]
        rec = dict(group=str(seed), v_rgb=rgb_contrast(luminance(rgb), reg, ring), v_missing=float((d == 0).mean()),
                   v_behind=float(((d - t > BEHIND_MM / 1000) & (d > 0)).mean()),
                   rgb=np.stack([rgb_win(rgb, y0, x0, s), rgb_win(rgb0, y0, x0, s)]),
                   depth=np.stack([depth_win(depth, y0, x0, s), depth_win(depth0, y0, x0, s)]))
        f = glass.shape[0] // raw.shape[0]
        reg_t, ring_t = reg[::f, ::f], ring[::f, ::f]
        if reg_t.sum() >= 20 and ring_t.sum() >= 20:
            st = max(2, int(round(s / f)))
            a = [min(int(round(v / f)), lim - st) for v, lim in zip((y0, x0), raw.shape)]
            rec.update(v_thermal=abs(float(np.median(raw[reg_t]) - np.median(raw[ring_t]))) / TRANSPOSE_NOISE,
                       thermal=np.stack([thermal_win(raw, a[0], a[1], st), thermal_win(raw0, a[0], a[1], st)]))
        recs.append(rec)
        cells.append(int(env.episode_info["cell"]))
    states = [int(env.cfg.cells[c][1]) for c in cells]          # colour state of the pair: 0 shows, 1 blind, 2 faint
    env.close()
    _save(out, recs, {level: c}, dict(level=level, episodes=n, cells=cells, rgb_state=states))


# ------------------------------------------------------------------------------------------------ section 12
def _lum(x):
    """(3, h, w) in 0..1 -> luminance in grey levels of 255."""
    return 255.0 * (0.299 * x[0] + 0.587 * x[1] + 0.114 * x[2])


def look_stats(win, mask):
    """Glass pixels against the other pixels of a 32 x 32 window: differences of the median, the 95th and the 5th
    percentile of the luminance, and the standard deviation of the other pixels (how plain the surroundings are)."""
    lum = _lum(np.asarray(win, np.float32))
    a, b = lum[mask], lum[~mask]
    if a.size < 20 or b.size < 20:
        return None
    qa, qb = np.percentile(a, [50, 95, 5]), np.percentile(b, [50, 95, 5])
    return dict(dmed=float(qa[0] - qb[0]), dp95=float(qa[1] - qb[1]), dp5=float(qa[2] - qb[2]), ring_std=float(b.std()))


LOOK = ("dmed", "dp95", "dp5")


@cli.command()
@click.option("--root", default=os.path.expanduser("~/datasets/transparent/unified"), show_default=True)
@click.option("--out", default="data/real_check/real_rgb.npz", show_default=True)
def realrgb(root, out):
    """Colour pairs (glass window, background window) of every TRansPose frame, the region's mask at 32 x 32 and the
    look statistics of the glass window."""
    import pandas as pd
    from PIL import Image
    root = Path(root)
    man = pd.read_parquet(root / "manifest.parquet")
    rows = man[(man.dataset == "transpose") & man.mask_path.notna() & (man.glass_frac > 0)].sort_values("sample_id")
    recs, c = [], dict(pair=0, nofit=0, noneg=0)
    for fi, r in enumerate(rows.itertuples()):
        mk = np.asarray(Image.open(root / r.mask_path))
        mk = (mk[..., 0] if mk.ndim == 3 else mk) == 1
        regs = glass_regions(mk)
        if not regs:
            continue
        rgb = lum = None
        for ri, (reg, ring) in enumerate(regs):
            w, why = windows(mk, reg, np.random.default_rng([3, fi, ri]))
            c[why] += 1
            if w is None:
                continue
            if rgb is None:
                rgb = np.asarray(Image.open(root / r.rgb_path))
                lum = luminance(rgb)
            y0, x0, ny, nx, s = w
            x = np.stack([rgb_win(rgb, y0, x0, s), rgb_win(rgb, ny, nx, s)])
            mask = fit32(reg[y0:y0 + s, x0:x0 + s].astype(np.float32))[0] > 0.5
            st = look_stats(x[0], mask) or dict(dmed=np.nan, dp95=np.nan, dp5=np.nan, ring_std=np.nan)
            recs.append(dict(x=x, mask=mask, group=str(r.group), v_rgb=rgb_contrast(lum, reg, ring), **st))
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    arr = dict(x_rgb=np.stack([r["x"] for r in recs]).astype(np.float16), mask=np.stack([r["mask"] for r in recs]),
               group=np.array([r["group"] for r in recs]),
               **{k: np.array([r[k] for r in recs], np.float32) for k in ("v_rgb", "ring_std") + LOOK})
    tmp = out.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, **arr)
    os.replace(tmp, out)
    json.dump(dict(pairs=len(recs), frames=len(rows), regions=c), open(out.with_suffix(".json"), "w"), indent=1)
    print(f"{len(recs)} pairs of {len(rows)} frames -> {out}; regions: {c}")


GRID = dict(k_s=(0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0, 1.2, 1.6), w_rim=(0.0, 0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0))


def _medians(x, mask):
    """Median luminance of the glass pixels and of the other pixels of every window."""
    lum = np.stack([_lum(w) for w in np.asarray(x, np.float32)])
    return np.array([np.median(a[k]) for a, k in zip(lum, mask)]), np.array([np.median(a[~k]) for a, k in zip(lum, mask)])


@cli.command()
@click.option("--dir", "d", default="data/real_check", show_default=True)
@click.option("--config", default="sim/configs/V2_place_opt.yaml", show_default=True)
@click.option("-n", default=400, show_default=True, help="episodes of the simulator in which colour shows the glass")
@click.option("--veil", type=click.Choice(["regression", "physical"]), default="physical", show_default=True,
              help="regression: T0 1 and F0 from the real regression (revision 0); physical: T0 0.92, F0 0.08 (as first listed; round 3)")
def look(d, config, n, veil):
    """The optical model's numbers from real images (revision 0 of section 12, before any judge was looked at).
    Veil: the regression of the glass's median luminance on its surroundings' over real windows gives how far the
    glass pulls its area towards the surroundings' average (F0). Highlights and rim: k_s and w_rim at which the
    spread inside the glass in excess of the spread around it (95th and 5th percentile, each minus the median)
    matches real glass on plain surroundings; the same plainness limit on the simulator's windows."""
    import itertools
    sys.path.insert(0, "sim")
    from env import GlassPickEnv
    from glass_optics import PARAMS, compose, terms
    d = Path(d)
    z = np.load(d / "real_rgb.npz")
    ok = np.isfinite(z["dmed"])
    gl, bg = _medians(z["x_rgb"][:, 0], z["mask"])
    wide = ok & (z["ring_std"] < 40)
    slope, icpt = np.linalg.lstsq(np.stack([bg[wide], np.ones(wide.sum())], 1), gl[wide], rcond=None)[0]
    f0 = float(np.clip(round((1 - slope) / 0.05) * 0.05, 0.05, 0.9))
    print(f"veil: over {int(wide.sum())} real windows of {len(np.unique(z['group'][wide]))} sequences (std of the other pixels below 40) the glass's median "
          f"luminance = {slope:.2f} * surroundings + {icpt:.0f} (r {np.corrcoef(bg[wide], gl[wide])[0, 1]:.2f}): the glass pulls its area "
          f"{1 - slope:.2f} of the way to a level of {icpt / (1 - slope):.0f} (the surroundings' median is {np.median(bg[wide]):.0f}). F0 = {f0:.2f}, T0 = 1.")
    limit = 15 if (ok & (z["ring_std"] < 15)).sum() >= 100 else 25
    plain = ok & (z["ring_std"] < limit)
    ex = dict(hi=z["dp95"] - z["dmed"], lo=z["dp5"] - z["dmed"])
    target, iqr = {}, {}
    print(f"spread: {int(plain.sum())} real windows of {len(np.unique(z['group'][plain]))} sequences with plain surroundings (std below {limit}); "
          f"their surroundings' median luminance {np.median(bg[plain]):.0f}")
    for k, name in (("hi", "p95 - median, glass minus surroundings"), ("lo", "p5 - median, glass minus surroundings")):
        q = np.percentile(ex[k][plain], [10, 25, 50, 75, 90])
        target[k], iqr[k] = float(q[2]), float(q[3] - q[1])
        print(f"  {name}: p10 {q[0]:.1f}, p25 {q[1]:.1f}, median {q[2]:.1f}, p75 {q[3]:.1f}, p90 {q[4]:.1f}")
    print(f"  glass median minus surroundings' median there: {np.median((gl - bg)[plain]):+.1f}")
    env = GlassPickEnv(config=config)
    s_ = env.sensors
    params = dict(PARAMS, **(env.cfg.optical or {}), **(dict(T0=1.0, F0=f0) if veil == "regression" else dict(T0=0.92, F0=0.08)))
    f0 = params["F0"]
    cache, seed, dropped = [], 0, 0
    while len(cache) < n:
        env.reset(seed)
        seed += 1
        if env.cfg.cells[int(env.episode_info["cell"])][1] != 0:
            continue
        r = s_.r
        r.enable_depth_rendering()
        r.update_scene(env.data, camera=s_.camera, scene_option=s_.show_all)
        depth_true = r.render().copy()
        r.disable_depth_rendering()
        r.enable_segmentation_rendering()
        r.update_scene(env.data, camera=s_.camera, scene_option=s_.show_all)
        seg = r.render()[..., 0].copy()
        r.disable_segmentation_rendering()
        behind, glasses = s_._optical_inputs(env.data, depth_true, seg)
        body = int(env.model.body(env.obj).id)
        g = [x for x in glasses if x[0] == body]
        regs = glass_regions(g[0][1]) if g else []
        if not regs:
            continue
        reg = max(regs, key=lambda x: x[0].sum())[0]
        ys, xs = np.nonzero(reg)
        s = int(round(K * max(ys.max() - ys.min() + 1, xs.max() - xs.min() + 1)))
        y0, x0 = int(round((ys.max() + ys.min()) / 2 - s / 2)), int(round((xs.max() + xs.min()) / 2 - s / 2))
        if s > min(reg.shape) or y0 < 0 or x0 < 0 or y0 + s > reg.shape[0] or x0 + s > reg.shape[1]:
            continue
        mask = fit32(reg[y0:y0 + s, x0:x0 + s].astype(np.float32))[0] > 0.5
        st = look_stats(fit32(behind[y0:y0 + s, x0:x0 + s].astype(np.float32) / 255.0), mask)
        if st is None or st["ring_std"] >= limit:          # the same plainness limit as for the real windows
            dropped += 1
            continue
        cache.append((behind, terms(behind, depth_true, g[0][1], s_.focal, g[0][2], params), (y0, x0, s), mask, st["ring_std"]))
    env.close()
    sim_bg = float(np.median([np.median(_lum(fit32(b[y0:y0 + s, x0:x0 + s].astype(np.float32) / 255.0))[~mk]) for b, _, (y0, x0, s), mk, _ in cache]))
    print(f"\nveil \"{veil}\": T0 {params['T0']}, F0 {params['F0']}. The simulator's plain surroundings have a median luminance of {sim_bg:.0f}; "
          f"the real regression gives a glass median of {slope * sim_bg + icpt:.0f} there, {slope * sim_bg + icpt - sim_bg:+.0f} against the surroundings "
          f"(real windows reach a surroundings' luminance of {np.percentile(bg[wide], 99):.0f} at their 99th percentile, so this is beyond their range).")

    def spread(k_s, w_rim, strength=1.0):
        rows = []
        for behind, t, (y0, x0, s), mask, _ in cache:
            img = behind.astype(np.float32)
            img[t["box"]] = compose(behind, t, k_s, w_rim, params["rho"], strength)
            st = look_stats(fit32(img[y0:y0 + s, x0:x0 + s] / 255.0), mask)
            rows.append([st["dp95"] - st["dmed"], st["dp5"] - st["dmed"], st["dmed"]])
        return np.median(np.array(rows), 0)

    res = []
    for k_s, w_rim in itertools.product(*GRID.values()):
        med = spread(k_s, w_rim)
        res.append((float(((med[0] - target["hi"]) / iqr["hi"]) ** 2 + ((med[1] - target["lo"]) / iqr["lo"]) ** 2), k_s, w_rim, med))
    res.sort(key=lambda x: x[0])
    print(f"\nsimulator: {len(cache)} episodes of {config} in which colour shows the glass and the window's surroundings are plain "
          f"({dropped} others left out); rho {params['rho']}; grid of {len(res)} points; the eight closest:")
    print("| distance | k_s | w_rim | p95 - median, glass minus surroundings | p5 - median, glass minus surroundings | glass median minus surroundings |\n|---|---|---|---|---|---|")
    for dist, k_s, w_rim, med in res[:8]:
        print(f"| {dist:.3f} | {k_s} | {w_rim} | {med[0]:.1f} | {med[1]:.1f} | {med[2]:+.1f} |")
    print(f"| real, plain surroundings | | | {target['hi']:.1f} | {target['lo']:.1f} | |")
    dist, k_s, w_rim, med = res[0]
    med_f = spread(k_s, w_rim, 0.3)
    print(f"\nchosen: F0 {f0:.2f}, T0 {params['T0']}, k_s {k_s}, w_rim {w_rim} (rho {params['rho']}). \"Faint\" (strength 0.3) with them: {med_f[0]:.1f}, {med_f[1]:.1f}, {med_f[2]:+.1f}.")
    json.dump(dict(veil=veil, T0=params["T0"], F0=f0, k_s=k_s, w_rim=w_rim, rho=params["rho"], distance=dist, limit=limit, plain=int(plain.sum()),
                   regression=dict(slope=float(slope), intercept=float(icpt), windows=int(wide.sum())), target=target, iqr=iqr,
                   sim=[float(v) for v in med], faint=[float(v) for v in med_f]), open(d / "optical_fit.json", "w"), indent=1)


def _halves(groups):
    """The sequences in their order, split by even and odd position: boolean masks of the pairs of half A and half B."""
    ug = np.array(sorted(np.unique(groups)))
    even = set(ug[0::2])
    a = np.array([g in even for g in groups])
    return a, ~a


class _Mean:
    """The mean score of several networks (one network's verdict moved by 0.07 between two trainings on the same pairs)."""
    def __init__(self, nets):
        self.nets = nets


N_JUDGE = 5


def _judge(d, which, device):
    """Judge A is trained on the real colour pairs of the even sequences, judge B on the odd ones: the mean of
    N_JUDGE networks (seeds 0 to 4). Returns it, its scores on the other half, that half's groups, the number of
    pairs it was trained on and which pairs of the other half have plain surroundings."""
    z = np.load(Path(d) / "real_rgb.npz")
    a, b = _halves(z["group"])
    own, other = (a, b) if which == "A" else (b, a)
    x = z["x_rgb"]
    net = _Mean([_train(x[own], device, seed=k) for k in range(N_JUDGE)])
    return net, _score(net, x[other], device), z["group"][other], int(own.sum()), z["ring_std"][other] < 25


def _judge_table(net, own_scores, own_groups, d, sims, device, which):
    a_own = auc(own_scores[:, 0], own_scores[:, 1])
    need = 0.5 + 0.5 * (a_own - 0.5)
    print(f"Judge {which}: its own AUC on the real pairs of the other half {a_own:.2f} ({ci(boot(own_scores, own_groups))}); "
          f"a state passes at {need:.2f} (0.5 + half of the margin).\n")
    print("| simulator's pairs | state | pairs | judge's AUC, with against without the glass (95%) | share of pairs with the glass scored higher | responds? |")
    print("|---|---|---|---|---|---|")
    out = {}
    for name in sims:
        z, meta = np.load(Path(d) / f"{name}.npz"), json.load(open(Path(d) / f"{name}.json"))
        sc = _score(net, z["x_rgb"], device)
        state = np.array(meta["rgb_state"]) if "rgb_state" in meta else np.array([_STATE_OF_CELL[c] for c in meta["cells"]])
        out[name] = (sc, state, z["v_rgb"], z["group"])
        for st, label in ((0, "shows"), (2, "faint"), (1, "blind")):
            m = state == st
            if not m.any():
                continue
            a_s = auc(sc[m, 0], sc[m, 1])
            verdict = "-" if st == 1 else ("yes" if a_s >= need else "no")
            print(f"| {name} | {label} | {int(m.sum())} | {a_s:.2f} ({ci(boot(sc[m], z['group'][m], seed=1))}) | {np.mean(sc[m, 0] > sc[m, 1]):.2f} | {verdict} |")
    return a_own, need, out


_STATE_OF_CELL = {0: 0, 1: 1, 2: 0, 3: 0, 4: 1, 5: 1, 6: 0, 7: 2, 8: 1, 9: 1, 10: 2}   # colour state of the eleven cells


@cli.command()
@click.option("--dir", "d", default="data/real_check", show_default=True)
@click.option("--which", type=click.Choice(["A", "B"]), default="A", show_default=True)
@click.option("--sims", multiple=True, default=("simpair_opt",), show_default=True)
@click.option("--device", default="cuda", show_default=True)
def judge(d, which, sims, device):
    """A judge that learned real glass on one half of the sequences, on the simulator's paired windows, per state."""
    net, own, groups, n_own, plain = _judge(d, which, device)
    print(f"(the mean of {N_JUDGE} networks, each trained on {n_own} real pairs; on the {int(plain.sum())} pairs of the other half whose "
          f"surroundings are plain (std below 25) its AUC is {auc(own[plain, 0], own[plain, 1]):.2f}, on the other {int((~plain).sum())} {auc(own[~plain, 0], own[~plain, 1]):.2f})")
    _judge_table(net, own, groups, d, sims, device, which)


@cli.command()
@click.option("--dir", "d", default="data/real_check", show_default=True)
@click.option("--optical", default="simpair_opt", show_default=True)
@click.option("--alpha", default="simpair", show_default=True)
@click.option("--device", default="cuda", show_default=True)
def readout3(d, optical, alpha, device):
    """The final read-out of section 12: judge B once, the floor, the simulator's network on real pairs."""
    d = Path(d)
    real = np.load(d / "real_rgb.npz")
    zo, mo = np.load(d / f"{optical}.npz"), json.load(open(d / f"{optical}.json"))
    print("# The optical colour model of the glass against real images (section 12)\n")
    print(f"Real: {len(real['group'])} colour pairs of {len(np.unique(real['group']))} TRansPose sequences. Simulator: {mo['pairs']} paired windows "
          f"of `{mo['level']}` (the same window with and without the glass); for comparison the alpha model, `{alpha}`.\n")
    # 1. the floor and the simulator's own difficulty
    xo, go, state = zo["x_rgb"], zo["group"], np.array(mo["rgb_state"])
    oof = np.full((len(xo), 2), np.nan, np.float32)
    for tr, te in _folds(go):
        oof[te] = _score(_train(xo[tr], device), xo[te], device)
    print("## 1. The simulator's own network, out of fold\n\n| state | pairs | AUC (95%) |\n|---|---|---|")
    for st, label in ((0, "shows"), (2, "faint"), (1, "blind")):
        m = state == st
        print(f"| {label} | {int(m.sum())} | {auc(oof[m, 0], oof[m, 1]):.2f} ({ci(boot(oof[m], go[m], seed=1))}) |")
    fl = auc(oof[state == 1, 0], oof[state == 1, 1])
    print(f"\nFloor (rule 1): {fl:.2f} in the colour-blind pairs: " + ("within 0.45 to 0.55." if 0.45 <= fl <= 0.55 else "OUTSIDE 0.45 to 0.55: nothing else is read.") + "\n")
    # 2. judge B, once
    print("## 2. Does a network that learned real glass respond? (judge B)\n")
    net, own, groups, n_own, _ = _judge(d, "B", device)
    a_own, need, out = _judge_table(net, own, groups, d, (optical, alpha), device, "B")
    print("\nPer contrast bin (grey levels between the glass and a ring around it), the optical model:\n")
    print("| bin | pairs | judge B's AUC | share with the glass scored higher |\n|---|---|---|---|")
    sc, _, v, _ = out[optical]
    for lo, hi, label in BINS["rgb"][2]:
        m = (v >= lo) & (v < hi)
        if m.sum() >= 2:
            print(f"| {label} | {int(m.sum())} | {auc(sc[m, 0], sc[m, 1]):.2f} | {np.mean(sc[m, 0] > sc[m, 1]):.2f} |")
    # 3. the simulator's network on real pairs
    xr, gr = real["x_rgb"], real["group"]
    own_r = np.full((len(xr), 2), np.nan, np.float32)
    for tr, te in _folds(gr):
        own_r[te] = _score(_train(xr[tr], device), xr[te], device)
    a_r = auc(own_r[:, 0], own_r[:, 1])
    need_r = 0.5 + 0.5 * (a_r - 0.5)
    print(f"\n## 3. Does a network that learned the simulator's glass find real glass?\n")
    print(f"The real network's own AUC out of fold: {a_r:.2f} ({ci(boot(own_r, gr))}); the simulator's network passes at {need_r:.2f}.\n")
    print("| simulator's network trained on | AUC on all real pairs (95%) | transfers? |\n|---|---|---|")
    for name in (optical, alpha):
        zs = np.load(d / f"{name}.npz")
        s2r = _score(_train(zs["x_rgb"], device), xr, device)
        a = auc(s2r[:, 0], s2r[:, 1])
        print(f"| {name} | {a:.2f} ({ci(boot(s2r, gr))}) | {'transfers' if a >= need_r else 'does not transfer'} |")


# ------------------------------------------------------------------------------------------------ classifiers
def _net(c):
    import torch.nn as nn

    def block(i, o):
        return [nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(), nn.MaxPool2d(2)]
    return nn.Sequential(*block(c, 32), *block(32, 64), *block(64, 64), nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(64, 1))


def _train(x, device, seed=0, epochs=30, batch=256):
    """x: (n, 2, c, 32, 32), glass window and background window. Returns the trained network."""
    import torch
    import torch.nn.functional as F
    torch.manual_seed(seed)
    xs = torch.from_numpy(np.concatenate([x[:, 0], x[:, 1]]).astype(np.float32)).to(device)
    ys = torch.cat([torch.ones(len(x)), torch.zeros(len(x))]).to(device)
    net = _net(x.shape[2]).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    g = torch.Generator(device="cpu").manual_seed(seed)
    for _ in range(epochs):
        net.train()
        perm = torch.randperm(len(xs), generator=g).to(device)
        for i in range(0, len(xs), batch):
            idx = perm[i:i + batch]
            if len(idx) < 2:
                continue
            xb = xs[idx]
            flip = (torch.rand(len(idx), generator=g) < 0.5).to(device)
            xb = torch.where(flip[:, None, None, None], xb.flip(-1), xb)
            dy, dx = (int(v) for v in torch.randint(0, 5, (2,), generator=g))
            xb = F.pad(xb, (2, 2, 2, 2), mode="replicate")[..., dy:dy + SIZE, dx:dx + SIZE]
            loss = F.binary_cross_entropy_with_logits(net(xb)[:, 0], ys[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
    return net.eval()


def _score(net, x, device):
    import torch
    if isinstance(net, _Mean):
        return np.mean([_score(k, x, device) for k in net.nets], 0)
    with torch.no_grad():
        flat = torch.from_numpy(x.reshape(-1, *x.shape[2:]).astype(np.float32)).to(device)
        out = torch.cat([net(flat[i:i + 2048])[:, 0] for i in range(0, len(flat), 2048)])
    return out.cpu().numpy().reshape(len(x), 2)


def _folds(groups, k=5):
    """Five folds that keep a group (a sequence, an episode) together: the groups, largest first, each to the fold
    with the fewest pairs so far."""
    ug, inv, cnt = np.unique(groups, return_inverse=True, return_counts=True)
    fold, size = np.zeros(len(ug), int), np.zeros(k, int)
    for g in np.argsort(-cnt, kind="stable"):
        fold[g] = int(np.argmin(size))
        size[fold[g]] += cnt[g]
    f = fold[inv]
    return [(np.flatnonzero(f != i), np.flatnonzero(f == i)) for i in range(k)]


@cli.command()
@click.option("--dir", "d", default="data/real_check", show_default=True)
@click.option("--device", default="cuda", show_default=True)
@click.option("--sim", "simname", default="sim", show_default=True, help="sim (first design) or simpair (second)")
def fit(d, device, simname):
    dom = {"real": np.load(Path(d) / "real.npz"), "sim": np.load(Path(d) / f"{simname}.npz")}
    out = {}
    for sensor in SENSORS:
        x, grp = {}, {}
        for k, z in dom.items():
            has = z[f"has_{sensor}"]
            x[k], grp[k] = z[f"x_{sensor}"][has], z["group"][has]
            oof = np.full((len(x[k]), 2), np.nan, np.float32)
            for tr, te in _folds(grp[k]):
                oof[te] = _score(_train(x[k][tr], device), x[k][te], device)
            full = np.full((len(has), 2), np.nan, np.float32)
            full[has] = oof
            out[f"{k}_{sensor}"] = full
            print(f"{sensor} {k}: {len(x[k])} pairs, out of fold: glass window scores higher in {np.mean(oof[:, 0] > oof[:, 1]):.3f}", flush=True)
        for a, b in (("sim", "real"), ("real", "sim")):
            has = dom[b][f"has_{sensor}"]
            full = np.full((len(has), 2), np.nan, np.float32)
            full[has] = _score(_train(x[a], device), x[b], device)
            out[f"{a}2{b}_{sensor}"] = full
    name = "scores" if simname == "sim" else f"scores_{simname}"
    tmp = Path(d) / f"{name}.tmp.npz"
    np.savez_compressed(tmp, **out)
    os.replace(tmp, Path(d) / f"{name}.npz")


# ------------------------------------------------------------------------------------------------ read-out
def auc(p, n):
    from scipy.stats import rankdata
    if len(p) == 0 or len(n) == 0:
        return np.nan
    r = rankdata(np.concatenate([p, n]))
    return float((r[:len(p)].sum() - len(p) * (len(p) + 1) / 2) / (len(p) * len(n)))


def boot(scores, groups, n_boot=1000, seed=0):
    """AUC of every resample of the groups (sequences or episodes)."""
    rng = np.random.default_rng(seed)
    ug, inv = np.unique(groups, return_inverse=True)
    members = [np.flatnonzero(inv == i) for i in range(len(ug))]
    out = np.empty(n_boot)
    for b in range(n_boot):
        idx = np.concatenate([members[i] for i in rng.integers(0, len(ug), len(ug))])
        out[b] = auc(scores[idx, 0], scores[idx, 1])
    return out


def ci(v):
    return "{:.2f} to {:.2f}".format(*np.nanquantile(v, [0.025, 0.975]))


@cli.command()
@click.option("--dir", "d", default="data/real_check", show_default=True)
def readout(d):
    d = Path(d)
    dom = {k: np.load(d / f"{k}.npz") for k in ("real", "sim")}
    meta = {k: json.load(open(d / f"{k}.json")) for k in ("real", "sim")}
    sc = np.load(d / "scores.npz")
    print("# Check against real sensor images: the glass window against its background window\n")
    print(f"Real: {meta['real']['pairs']} pairs, frames {meta['real']['frames']}, regions {meta['real']['regions']}.")
    print(f"Simulator: {meta['sim']['pairs']} pairs of {meta['sim']['episodes']} episodes of `{meta['sim']['level']}`, "
          f"regions {meta['sim']['regions']}.\n")
    for sensor, (key, unit, bins) in BINS.items():
        print(f"## {sensor} (bins: {unit})\n")
        print("| bin | real: pairs | real: AUC (95%) | simulator: pairs | simulator: AUC (95%) | simulator - real (95%) | same difficulty? "
              "| simulator's network on real | real network on the simulator | does the look transfer? |")
        print("|---|---|---|---|---|---|---|---|---|---|")
        floor = None
        for lo, hi, label in bins + [(-np.inf, np.inf, "all")]:
            col, bs = {}, {}
            for k in ("real", "sim"):
                v, s = dom[k][key], sc[f"{k}_{sensor}"]
                sel = np.isfinite(s[:, 0]) & np.isfinite(v) & (v >= lo) & (v < hi)
                col[k] = (int(sel.sum()), auc(s[sel, 0], s[sel, 1]), sel)
                bs[k] = boot(s[sel], dom[k]["group"][sel], seed=1 if k == "sim" else 0) if sel.sum() >= 2 else np.full(1000, np.nan)
            cross = {}
            for a, b in (("sim", "real"), ("real", "sim")):
                s = sc[f"{a}2{b}_{sensor}"]
                sel = col[b][2]
                cross[b] = auc(s[sel, 0], s[sel, 1])
            (nr, ar, _), (ns, as_, _) = col["real"], col["sim"]
            diff = bs["sim"] - bs["real"]
            if label == "all" or min(nr, ns) < 50:
                same = "-" if label == "all" else "not judged (fewer than 50 pairs)"
            else:
                lo_d, hi_d = np.nanquantile(diff, [0.025, 0.975])
                dd = as_ - ar
                same = "the same" if abs(dd) <= 0.05 or lo_d <= 0 <= hi_d else ("easier in the simulator" if dd > 0 else "harder in the simulator")
            if label == "all" or nr < 50 or not np.isfinite(ar) or ar < 0.60:
                tr = "-" if label == "all" else "not judged"
            else:
                tr = "transfers" if cross["real"] - 0.5 >= 0.5 * (ar - 0.5) else "does not transfer"
            if label == BLIND[sensor]:
                floor = (ns, as_, ci(bs["sim"]))
            f = lambda n_, a_, b_: f"{a_:.2f} ({ci(b_)})" if n_ >= 2 else "-"
            print(f"| {label} | {nr} | {f(nr, ar, bs['real'])} | {ns} | {f(ns, as_, bs['sim'])} | "
                  + (f"{as_ - ar:+.2f} ({ci(diff)})" if min(nr, ns) >= 2 else "-")
                  + f" | {same} | " + (f"{cross['real']:.2f}" if nr >= 2 else "-") + " | " + (f"{cross['sim']:.2f}" if ns >= 2 else "-") + f" | {tr} |")
        if floor:
            print(f"\nThe floor of the test (the simulator's blind state, bin \"{BLIND[sensor]}\"): AUC {floor[1]:.2f} ({floor[2]}) on {floor[0]} pairs; 0.5 if the windows give nothing away without the glass.\n")


@cli.command()
@click.option("--dir", "d", default="data/real_check", show_default=True)
def readout2(d):
    """Second design: the simulator's pairs are the same window with and without the glass."""
    d = Path(d)
    dom = {"real": np.load(d / "real.npz"), "sim": np.load(d / "simpair.npz")}
    meta = {"real": json.load(open(d / "real.json")), "sim": json.load(open(d / "simpair.json"))}
    sc = np.load(d / "scores_simpair.npz")
    print("# Check against real sensor images, second design: in the simulator the same window with and without the glass\n")
    print(f"Real: {meta['real']['pairs']} pairs (glass window, background window of the same frame). Simulator: {meta['sim']['pairs']} pairs "
          f"of {meta['sim']['episodes']} episodes of `{meta['sim']['level']}`, regions {meta['sim']['regions']}.\n")
    for sensor, (key, unit, bins) in BINS.items():
        print(f"## {sensor} (bins: {unit})\n")
        print("| bin | simulator: pairs | simulator's own AUC, with against without the glass (95%) | real: pairs | real network's own AUC (95%) "
              "| real network on the simulator's pairs: AUC | share of pairs in which the window with the glass scores higher "
              "| does a real-glass network respond to the simulator's glass? | simulator's network on real pairs: AUC | does it find real glass? |")
        print("|---|---|---|---|---|---|---|---|---|---|")
        notes = []
        for lo, hi, label in bins + [(-np.inf, np.inf, "all")]:
            sel, own, bs = {}, {}, {}
            for k in ("real", "sim"):
                v, s = dom[k][key], sc[f"{k}_{sensor}"]
                sel[k] = np.isfinite(s[:, 0]) & np.isfinite(v) & (v >= lo) & (v < hi)
                own[k] = auc(s[sel[k], 0], s[sel[k], 1])
                bs[k] = boot(s[sel[k]], dom[k]["group"][sel[k]], seed=1 if k == "sim" else 0) if sel[k].sum() >= 2 else np.full(1000, np.nan)
            nr, ns = int(sel["real"].sum()), int(sel["sim"].sum())
            r2s, s2r = sc[f"real2sim_{sensor}"][sel["sim"]], sc[f"sim2real_{sensor}"][sel["real"]]
            a_r2s, share, a_s2r = auc(r2s[:, 0], r2s[:, 1]), (float(np.mean(r2s[:, 0] > r2s[:, 1])) if ns else np.nan), auc(s2r[:, 0], s2r[:, 1])
            judged = label != "all" and min(nr, ns) >= 50 and np.isfinite(own["real"]) and own["real"] >= 0.60
            need = 0.5 + 0.5 * (own["real"] - 0.5) if judged else np.nan
            v2 = ("yes" if a_r2s >= need else "no") if judged else ("-" if label == "all" else "not judged")
            v3 = ("transfers" if a_s2r >= need else "does not transfer") if judged else ("-" if label == "all" else "not judged")
            f = lambda n_, a_, b_: f"{a_:.2f} ({ci(b_)})" if n_ >= 2 else "-"
            print(f"| {label} | {ns} | {f(ns, own['sim'], bs['sim'])} | {nr} | {f(nr, own['real'], bs['real'])} | "
                  + (f"{a_r2s:.2f}" if ns >= 2 else "-") + " | " + (f"{share:.2f}" if ns >= 2 else "-") + f" | {v2} | "
                  + (f"{a_s2r:.2f}" if nr >= 2 else "-") + f" | {v3} |")
            if label == BLIND[sensor]:
                lo_b, hi_b = np.nanquantile(bs["sim"], [0.025, 0.975])
                if sensor == "thermal":
                    notes.append(f"Thermal blind (below 3 noise units): the simulator's network reaches {own['sim']:.2f} ({lo_b:.2f} to {hi_b:.2f}) "
                                 f"on {ns} pairs; 0.5 would be blind to a network.")
                else:
                    ok = 0.45 <= own["sim"] <= 0.55
                    notes.append(f"Floor (rule 1): the simulator's own AUC in its blind state is {own['sim']:.2f} ({lo_b:.2f} to {hi_b:.2f}) on {ns} pairs: "
                                 + ("within 0.45 to 0.55, the paired test gives nothing away." if ok else "OUTSIDE 0.45 to 0.55: nothing else of this sensor is read."))
        print("\n" + "\n".join(notes) + "\n")


if __name__ == "__main__":
    cli()
