"""
Is glass of a given visibility as hard to find in the simulator as in real sensor images?
(results/hard_scenario/bench_v2/README.md, section 11; everything below is fixed there before it was run.)

  python real_data/findability.py real    [--root ~/datasets/transparent/unified]   windows of real glass regions
  python real_data/findability.py sim     [--level V2_place] [-n 3000]              the same from first frames of the simulator
  python real_data/findability.py fit     [--device cuda]                           classifiers: out of fold and across the domains
  python real_data/findability.py readout                                           the tables

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
def fit(d, device):
    dom = {k: np.load(Path(d) / f"{k}.npz") for k in ("real", "sim")}
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
    tmp = Path(d) / "scores.tmp.npz"
    np.savez_compressed(tmp, **out)
    os.replace(tmp, Path(d) / "scores.npz")


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


if __name__ == "__main__":
    cli()
