"""
Calibrates simulated thermal against real TRansPose 14-bit raw thermal by image statistics (not pixels or absolute
temperature). Same metric code (thermal_metrics.py) for both, at native resolution and after the same resize to 96x96.

Compare only:  python sim/thermal_calib.py --real-root $PROJECT/transparent/transpose --thermal v0
Fit + report:  python sim/thermal_calib.py --real-root $PROJECT/transparent/transpose --fit --blur 1.0 --save calibrated_v1

--fit matches only the camera model (raw gain/offset, noise, and blur unless --blur fixes it). The task's temperature design (glass gap,
crossover) is kept, so the fit uses "realistic" sim frames (glass slightly colder than t_env, no liquid, like
TRansPose) and fits that realistic gap together with the camera; the gap is saved as the (off) realistic option.
Output: sim/out/thermal_calib/ (per-frame CSVs, summary table, histograms, image grid, unmatched effects).
"""
import argparse
import copy
import csv
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import thermal_metrics as tm
from config import THERMAL_DIR, load_scene_config, load_thermal_config
from env import GlassPickEnv
from sensors import Sensors, band_radiance

TRANSPARENT = ("C", "H", "T") # TRansPose object categories that are transparent (N = not)
# Background ring (px) and gap from the glass edge (for cnr_gap) per resolution: 8 px at 640 px width as in
# encoder/dataset.py, 2 px at 160 (sim native) and 96; the gap skips ~2 px of blurred edge at 96
RING = {"real": (8, 13), "sim": (2, 3), "96": (2, 2)}
METRICS = ["cnr", "cnr_gap", "delta", "ring_std", "noise", "delta_noise", "delta_range", "noise_range",
           "sharpness", "glass_frac", "median", "p1", "p99"]
# Fit targets (96x96): count levels and counts-per-degree independent ratios; CNR is reported only
TARGETS = ["median", "p1", "p99", "delta_range", "delta_noise", "noise_range", "noise", "sharpness"]
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]   # real, then sim sets in fixed order

def real_frames(root, seq, every):
    """TRansPose thermal: raw (H, W) uint16 counts, glass mask = union of transparent object masks."""
    cam = Path(root) / seq / "cam_T"
    for name in sorted(os.listdir(cam / "14bit"))[::every]:
        idx = name[:-4]
        raw = np.array(Image.open(cam / "14bit" / name))
        mask = np.zeros(raw.shape, bool)
        for o in json.loads((cam / "pose" / f"{idx}.json").read_text()).values():
            path = cam / "mask" / o["mask_img"]
            if o["obj_name"].split("_")[1] in TRANSPARENT and path.exists():
                mask |= np.array(Image.open(path)) > 0
        yield {"frame": idx}, raw.astype(np.float32), mask

def scene_config(scene, thermal, liquid=True):
    cfg = load_scene_config(scene)
    cfg["thermal"] = load_thermal_config(thermal)
    if not liquid:
        cfg["liquid"]["fill_levels"] = [0.0]
    return cfg

def sim_frames(cfg, split, n, seed0=0):
    """Reset frames: raw counts and glass mask at thermal resolution (same path as env.get_obs)."""
    env = GlassPickEnv(config=cfg, split=split, render=False)
    sens = Sensors(env.model, thermal=env.th)
    for seed in range(seed0, seed0 + n):
        env.reset(seed)
        rgb, depth_true, depth_behind, seg = sens.geometry(env.data)
        on_glass = np.isin(seg, sens.glass_geoms)
        below = sens.below_liquid(env.data, depth_true, on_glass, env.liquid[0]) if env.liquid else None
        raw = sens.thermal_raw(seg, env.temps, env.emis, env.t_env, env.rng, below, env.liquid[1] if env.liquid else None)
        gap = env.temps["glass"] - env.t_env
        yield {"seed": seed, "gap": gap, "crossover": abs(gap) < 2}, raw.astype(np.float32), sens.to_thermal_res(on_glass)
    sens.r.close()

def stats(frames, native):
    rows = []
    for info, raw, mask in frames:
        row = dict(info)
        row.update({f"{k}_native": v for k, v in tm.frame_stats(raw, mask, *RING[native]).items()})
        r96, m96 = tm.resize(raw, mask)
        row.update({f"{k}_96": v for k, v in tm.frame_stats(r96, m96, *RING["96"]).items()})
        row["_raw96"] = r96
        rows.append(row)
    return rows

def column(rows, key):
    x = np.array([r[key] for r in rows], float)
    return x[np.isfinite(x)]

def w1(a, b):
    """1-D Wasserstein distance between two samples (mean gap between their quantile functions)."""
    q = np.linspace(0, 1, 101)
    return np.mean(np.abs(np.quantile(a, q) - np.quantile(b, q)))

def distance(real, sim, keys=tuple(TARGETS)):
    """Sum over metrics of W1(sim, real) / IQR(real) on the 96x96 metrics."""
    out = {}
    for k in keys:
        r = column(real, f"{k}_96")
        out[k] = w1(r, column(sim, f"{k}_96")) / (np.subtract(*np.percentile(r, [75, 25])) + 1e-9)
    return sum(out.values()), out

# ---------- fit ----------
def fit_cache(scene, split, n, seed0=0):
    """Realistic fit scenes (no liquid): segmentation at thermal resolution + temperatures, rendered once."""
    cfg = scene_config(scene, None, liquid=False)
    env = GlassPickEnv(config=cfg, split=split, render=False)
    sens = Sensors(env.model, thermal=env.th)
    cache = []
    for seed in range(seed0, seed0 + n):
        env.reset(seed)
        seg = sens.to_thermal_res(sens.geometry(env.data)[3])
        cache.append(dict(seed=seed, seg=seg, mask=np.isin(seg, sens.glass_geoms), temps=dict(env.temps), emis=dict(env.emis),
                          t_env=env.t_env, u=np.random.default_rng([seed, 98]).random()))
    return sens, cache

def fit_eval(sens, cache, real, p, m_real):
    """Rows for camera/realistic parameters p; offset is set so the median count level matches the real one."""
    th = copy.deepcopy(sens.th)
    th.update(blur_sigma_px=p["blur"], noise={"rel": 0.0, "counts": p["noise"]})
    th["raw"] = {"gain": p["gain"], "offset": m_real - p["gain"] * band_radiance(22.0)}
    sens.th = th
    frames = []
    for c in cache:
        temps = dict(c["temps"], glass=c["t_env"] - (p["c"] + p["w"] * (2 * c["u"] - 1)))
        raw = sens.thermal_raw(c["seg"], temps, c["emis"], c["t_env"], np.random.default_rng([c["seed"], 99]))
        frames.append(({}, raw.astype(np.float32), c["mask"]))
    rows = stats(frames, "sim")
    shift = m_real - np.median(column(rows, "median_96"))    # exact for levels; ratios do not depend on offset
    for r in rows:
        for k in ["median", "p1", "p99"]:
            r[f"{k}_96"] += shift
            r[f"{k}_native"] += shift
    th["raw"]["offset"] += shift
    return rows, th

def fit(sens, cache, real, n_trials, blur=None, seed=0):
    """Seeded random search, then local refinement around the best 5. blur: fixed sigma (px), else searched;
    with a fixed blur the edge sharpness cannot be fitted and is only reported."""
    rng = np.random.default_rng(seed)
    m_real = np.median(column(real, "median_96"))
    keys = [k for k in TARGETS if blur is None or k != "sharpness"]
    def sample():
        return {"gain": np.exp(rng.uniform(np.log(10), np.log(150))), "noise": np.exp(rng.uniform(np.log(0.2), np.log(8))),
                "blur": rng.uniform(0.3, 2.0) if blur is None else blur, "c": rng.uniform(0, 8), "w": rng.uniform(0, 3)}
    def perturb(p):
        q = {k: v * np.exp(rng.normal(0, 0.15)) for k, v in p.items()}
        q["blur"] = float(np.clip(q["blur"], 0.3, 2.5)) if blur is None else blur
        return q
    trials = []
    for i in range(n_trials):
        p = sample() if i < int(0.6 * n_trials) else perturb(sorted(trials, key=lambda t: t[0])[i % 5][1])
        rows, th = fit_eval(sens, cache, real, p, m_real)
        d, parts = distance(real, rows, keys)
        trials.append((d, p, th, parts))
        if i % 50 == 0:
            print(f"trial {i}: best distance {min(t[0] for t in trials):.3f}", flush=True)
    return sorted(trials, key=lambda t: t[0]), keys

def save_config(name, th, p, best, seqs, keys, blur):
    lines = [f"# Camera model fitted to TRansPose {', '.join(seqs)} 14-bit raw by sim/thermal_calib.py --fit (seed 0).",
             f"# Temperatures keep the v0 task design. Fit distance {best:.3f} (sum of W1 / real IQR over {', '.join(keys)}).",
             "# Blur fixed (not fitted): the sim glass is much smaller than TRansPose objects, so a fitted blur absorbs that;"
             " refit it once the camera is decided." if blur is not None else "# Blur fitted.",
             f"name: {name}",
             f"raw: {{gain: {th['raw']['gain']:.4g}, offset: {th['raw']['offset']:.5g}}}",
             f"noise: {{rel: 0.0, counts: {p['noise']:.3g}}}",
             f"blur_sigma_px: {p['blur']:.3g}",
             "# Realistic glass (fitted with the camera, glass colder than t_env); off until the ratio is decided",
             f"realistic: {{p: 0.0, gap: [{-(p['c'] + p['w']):.3g}, {-(p['c'] - p['w']):.3g}]}}"]
    path = THERMAL_DIR / f"{name}.yaml"
    path.write_text("\n".join(lines) + "\n")
    return path

# ---------- report ----------
def write_csv(path, rows):
    keys = [k for k in rows[0] if not k.startswith("_")]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

def report(out, real, sims, seqs, keys=tuple(TARGETS)):
    """sims: {name: rows}; keys: metrics the fit used. Table, low-contrast fractions, histograms, image grid."""
    names = ["real"] + list(sims)
    sets = [real] + list(sims.values())
    lines = [f"# Real ({', '.join(seqs)}, {len(real)} frames) vs sim", "",
             "median [p10, p90]; *_noise / *_range ratios do not depend on counts per degree; fit targets: " + ", ".join(keys), "",
             "| metric | " + " | ".join(f"{n} (n={len(s)})" for n, s in zip(names, sets)) + " |", "|---" * (len(names) + 1) + "|"]
    for res in ["96", "native"]:
        for m in METRICS:
            cells = []
            for s in sets:
                x = column(s, f"{m}_{res}")
                cells.append(f"{np.median(x):.3g} [{np.percentile(x, 10):.3g}, {np.percentile(x, 90):.3g}]" if len(x) else "-")
            lines.append(f"| {m}_{res} | " + " | ".join(cells) + " |")
    lines += ["", f"Fit distance (sum of W1 / real IQR over the targets, 96x96):"]
    for n, s in sims.items():
        d, parts = distance(real, s)
        lines.append(f"- {n}: {d:.2f} (" + ", ".join(f"{k} {v:.2f}" for k, v in parts.items()) + ")")
    lines += ["", f"Low-contrast frames (96x96): CNR < {tm.LOW_CNR} / cnr_gap < {tm.LOW_CNR}:"]
    for n, s in zip(names, sets):
        lines.append(f"- {n}: {np.mean(column(s, 'cnr_96') < tm.LOW_CNR):.1%} / {np.mean(column(s, 'cnr_gap_96') < tm.LOW_CNR):.1%}")
    lines += ["", "## Effects the sim cannot match (not forced)",
              "- Glass size: the sim glass covers ~0.26% of the image vs ~5.8% for TRansPose objects (camera pose / field of view, not a thermal parameter). With the small glass the 2 px ring is mostly blurred edge, so the encoder CNR saturates near 5-7 in sim; use cnr_gap and the ratios instead.",
              "- Background thermal structure: real background rings vary ~20x the noise (other warm objects, temperature patterns); sim backgrounds are flat except the warm robot and distractors.",
              "- The top of a filled glass shows the glass temperature (closed cylinder), not the liquid surface.",
              "- Reflections on glass (LWIR glass is specular), fixed-pattern noise, vignetting, and drift of the real camera are not rendered.",
              "- Temperatures: only the camera model is fitted; the task's glass gap / crossover design is kept (the fitted realistic gap is an option, off)."]
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    panels = [("cnr_gap_96", True), ("delta_range_96", True), ("delta_noise_96", True), ("noise_range_96", True), ("median_96", False)]
    fig, axes = plt.subplots(1, len(panels), figsize=(21, 3.8))
    for ax, (key, logx) in zip(axes, panels):
        vals = [column(s, key) for s in sets]
        vals = [v[v > 0] if logx else v for v in vals]
        lo, hi = min(v.min() for v in vals), max(v.max() for v in vals)
        bins = np.geomspace(lo, hi, 40) if logx else np.linspace(lo, hi, 40)
        for n, v, color in zip(names, vals, COLORS):
            ax.hist(v, bins=bins, histtype="step", linewidth=2, color=color, weights=np.full(len(v), 1 / len(v)), label=n)
        if logx:
            ax.set_xscale("log")
        ax.set_title(key.replace("_96", " (96x96)"), fontsize=10, color="#0b0b0b")
        ax.set_ylabel("fraction of frames", fontsize=8, color="#52514e")
        ax.tick_params(colors="#52514e", labelsize=8)
        for s in ["top", "right"]:
            ax.spines[s].set_visible(False)
    axes[0].legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(out / "hist.png", dpi=120)

    # Image grid: rows = data sets, 6 frames each, all with one fixed count range from the real frames (display only)
    lo, hi = np.percentile(np.stack([r["_raw96"] for r in real]), [0.5, 99.5])
    tiles = []
    for s in sets:
        picks = [s[i]["_raw96"] for i in np.linspace(0, len(s) - 1, 6).astype(int)]
        tiles.append(np.concatenate([np.pad(np.clip((x - lo) / (hi - lo), 0, 1), 2, constant_values=1) for x in picks], 1))
    img = (255 * np.concatenate(tiles, 0)).astype(np.uint8)
    Image.fromarray(img).resize((img.shape[1] * 2, img.shape[0] * 2), Image.NEAREST).save(out / "grid_fixed_range.png")
    (out / "grid_rows.txt").write_text(f"rows: {', '.join(names)}; fixed display range {lo:.0f}-{hi:.0f} counts (real 0.5-99.5 percentile)\n")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--real-root", required=True, help="folder of TRansPose sequences (<seq>/cam_T/...)")
    p.add_argument("--seq", nargs="+", default=["seq_all_01"])
    p.add_argument("--every", type=int, default=3, help="use every k-th real frame")
    p.add_argument("--scene", default="randomized_v1")
    p.add_argument("--split", default="train")
    p.add_argument("--thermal", nargs="+", default=["v0"], help="thermal configs to compare (without --fit)")
    p.add_argument("--n-sim", type=int, default=150)
    p.add_argument("--fit", action="store_true")
    p.add_argument("--n-fit", type=int, default=100, help="realistic sim scenes used by the fit")
    p.add_argument("--n-trials", type=int, default=500)
    p.add_argument("--blur", type=float, default=None, help="fix the Gaussian blur sigma (thermal px) instead of fitting it")
    p.add_argument("--save", default="calibrated_v1")
    p.add_argument("--out", default="sim/out/thermal_calib")
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    real = [r for s in args.seq for r in stats(real_frames(args.real_root, s, args.every), "real")]
    write_csv(out / "frames_real.csv", real)
    names, keys = list(args.thermal), TARGETS
    if args.fit:
        assert args.save != "v0", "v0 is kept unchanged"
        sens, cache = fit_cache(args.scene, args.split, args.n_fit)
        trials, keys = fit(sens, cache, real, args.n_trials, args.blur)
        best, bp, bth, parts = trials[0]
        print(f"best distance {best:.3f}: " + ", ".join(f"{k} {v:.3g}" for k, v in bp.items()))
        print(f"saved {save_config(args.save, bth, bp, best, args.seq, keys, args.blur)}")
        sens.r.close()
        names = ["v0", args.save, f"{args.save}+realistic"]

    sims = {}
    for name in names:
        th = load_thermal_config(name.replace("+realistic", ""))
        if name.endswith("+realistic"):
            th["realistic"]["p"] = 1.0
        rows = list(stats(sim_frames(scene_config(args.scene, th, liquid=not name.endswith("+realistic")), args.split, args.n_sim), "sim"))
        write_csv(out / f"frames_sim_{name}.csv", rows)
        sims[f"sim {name}"] = rows
    report(out, real, sims, args.seq, keys)
    print("saved", out)
