"""
Compares simulated thermal with real TRansPose 14-bit raw thermal by image statistics (not pixels or absolute
temperature): glass-vs-background CNR and contrast, low-contrast frame fraction, noise, edge sharpness, raw levels.
Same metric code (thermal_metrics.py) for both, at native resolution and after the same resize to 96x96.

run via python sim/thermal_calib.py --real-root $PROJECT/transparent/transpose --seq seq_all_01 --thermal v0
Output: sim/out/thermal_calib/ (per-frame CSV, summary table, histograms)
"""
import argparse
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
from config import load_scene_config
from env import GlassPickEnv
from sensors import Sensors

TRANSPARENT = ("C", "H", "T") # TRansPose object categories that are transparent (N = not)
REAL_RING = 8                  # background ring (px) at 640 px width, as in encoder/dataset.py
METRICS = ["cnr", "delta", "ring_std", "noise", "delta_noise", "delta_range", "noise_range", "sharpness", "glass_frac", "median", "p1", "p99"]
COLORS = {"real": "#2a78d6", "sim": "#eb6834"}

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

def sim_frames(scene, split, thermal, n, seed0=0):
    """Reset frames of the scene config with the given thermal config: raw counts and glass mask at thermal resolution."""
    cfg = load_scene_config(scene)
    cfg["thermal"] = thermal
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

def stats(frames, native_ring):
    rows = []
    for info, raw, mask in frames:
        row = dict(info)
        row.update({f"{k}_native": v for k, v in tm.frame_stats(raw, mask, native_ring).items()})
        r96, m96 = tm.resize(raw, mask)
        row.update({f"{k}_96": v for k, v in tm.frame_stats(r96, m96, 2).items()})
        rows.append(row)
    return rows

def summary(rows, key):
    x = np.array([r[key] for r in rows], float)
    x = x[np.isfinite(x)]
    return {"median": np.median(x), "p10": np.percentile(x, 10), "p90": np.percentile(x, 90), "min": x.min(), "max": x.max()}

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--real-root", required=True, help="folder of TRansPose sequences (<seq>/cam_T/...)")
    p.add_argument("--seq", nargs="+", default=["seq_all_01"])
    p.add_argument("--every", type=int, default=3, help="use every k-th real frame")
    p.add_argument("--scene", default="randomized_v1")
    p.add_argument("--split", default="train")
    p.add_argument("--thermal", default="v0")
    p.add_argument("--n-sim", type=int, default=150)
    p.add_argument("--out", default="sim/out/thermal_calib")
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    real = [r for s in args.seq for r in stats(real_frames(args.real_root, s, args.every), REAL_RING)]
    sim = stats(sim_frames(args.scene, args.split, args.thermal, args.n_sim), 2)   # 2 px at 160 px width = 8 px at 640
    for name, rows in [("real", real), (f"sim_{args.thermal}", sim)]:
        with open(out / f"frames_{name}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)

    # Summary table: median [p10, p90] per metric, native and 96x96
    lines = [f"# Real ({', '.join(args.seq)}, {len(real)} frames) vs sim {args.thermal} ({args.scene}/{args.split}, {len(sim)} frames)", "",
             "| metric | real median [p10, p90] | sim median [p10, p90] | sim / real (median) |", "|---|---|---|---|"]
    for res in ["native", "96"]:
        for m in METRICS:
            key = f"{m}_{res}"
            a, b = summary(real, key), summary(sim, key)
            ratio = b["median"] / a["median"] if a["median"] else np.nan
            lines.append(f"| {key} | {a['median']:.3g} [{a['p10']:.3g}, {a['p90']:.3g}] | {b['median']:.3g} [{b['p10']:.3g}, {b['p90']:.3g}] | {ratio:.3g} |")
    for res in ["native", "96"]:
        key = f"cnr_{res}"
        low_real = np.mean([r[key] < tm.LOW_CNR for r in real])
        low_sim = np.mean([r[key] < tm.LOW_CNR for r in sim])
        cross = np.mean([r["crossover"] for r in sim])
        lines.append(f"\nCNR < {tm.LOW_CNR} ({res}): real {low_real:.1%}, sim {low_sim:.1%} (sim episodes with |gap| < 2 C: {cross:.1%})")
    sim_c = [r for r in sim if r["crossover"]]
    sim_n = [r for r in sim if not r["crossover"]]
    for name, rows in [("sim crossover", sim_c), ("sim non-crossover", sim_n)]:
        if rows:
            s = summary(rows, "cnr_96")
            lines.append(f"{name}: cnr_96 median {s['median']:.3g} [{s['p10']:.3g}, {s['p90']:.3g}] (n={len(rows)})")
    (out / f"summary_{args.thermal}.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    # Histograms of the 96x96 metrics, real vs sim (fraction of frames per bin; log x for ratios)
    panels = [("cnr_96", True), ("delta_noise_96", True), ("noise_range_96", True), ("median_96", False)]
    fig, axes = plt.subplots(1, len(panels), figsize=(17, 3.6))
    for ax, (key, logx) in zip(axes, panels):
        vals = {k: np.array([r[key] for r in rows], float) for k, rows in [("real", real), ("sim", sim)]}
        vals = {k: v[np.isfinite(v) & ((v > 0) if logx else True)] for k, v in vals.items()}
        lo, hi = min(v.min() for v in vals.values()), max(v.max() for v in vals.values())
        bins = np.geomspace(lo, hi, 40) if logx else np.linspace(lo, hi, 40)
        for k, v in vals.items():
            ax.hist(v, bins=bins, histtype="step", linewidth=2, color=COLORS[k], weights=np.full(len(v), 1 / len(v)),
                    label=f"{'real' if k == 'real' else 'sim ' + args.thermal} (n={len(v)})")
        if logx:
            ax.set_xscale("log")
        ax.set_title(key.replace("_96", " (96x96)"), fontsize=10, color="#0b0b0b")
        ax.set_ylabel("fraction of frames", fontsize=8, color="#52514e")
        ax.tick_params(colors="#52514e", labelsize=8)
        for s in ["top", "right"]:
            ax.spines[s].set_visible(False)
        ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(out / f"hist_{args.thermal}.png", dpi=120)
    print("saved", out)
