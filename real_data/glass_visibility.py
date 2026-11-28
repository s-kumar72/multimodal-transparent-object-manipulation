"""
How do real sensors show real glass? Per glass region of the unified real datasets (real_data/README.md), the three
quantities the visibility cells of the benchmark set by hand, measured with one definition each so that the same
script can be pointed at simulated frames (sim/measure_visibility.py uses these functions):

  python real_data/glass_visibility.py [--root ~/datasets/transparent/unified] [--n 600] [--out FILE.json]

A region is a connected part of the glass mask of at least 0.13% of the image (400 px at 640x480).
  depth    missing: share of the region's pixels without a reading. With a ground-truth depth (trobi, transpose) also
           behind: share with a reading more than 2 cm farther than the true surface (the sensor looked through the
           glass), right: within 2 cm.  A region is hole-like when missing > 0.5, see-through when behind > 0.5.
  rgb      contrast: |median luminance of the region - median luminance of a ring around it|, in grey levels of 255
           (ring: 12 px at 640x480, other glass left out).
  thermal  contrast: |median of the region - median of the ring| (real_data/thermal_contrast.py), in degC for rgbt_glass
           and in units of the camera noise (0.89 counts) for transpose.
Frames are drawn evenly from every group (scene, sequence, location) of a dataset, seed 0. Printed: quantiles per
dataset and sensor; for transpose, the only dataset with all three sensors on tabletop objects, the three per region
together.
"""
import json
import os
from pathlib import Path

import click
import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage

BEHIND_MM = 20.0
TRANSPOSE_NOISE = 0.89   # counts per pixel (sim/configs/thermal/calibrated_v1.yaml)
SENSOR = {"mju_glass": "Orbbec Gemini Pro (structured light)", "trobi": "Ensenso (active stereo)",
          "transpose": "RealSense L515 (lidar)", "rgbt_glass": "-", "rgbd_gsd": "-"}
Q = (0.1, 0.25, 0.5, 0.75, 0.9)


def luminance(rgb):
    rgb = np.asarray(rgb, np.float32)
    return rgb if rgb.ndim == 2 else rgb[..., :3] @ np.array([0.299, 0.587, 0.114], np.float32)


def glass_regions(mask, min_frac=400 / (640 * 480)):
    """Boolean masks of the connected glass regions that are large enough, and the ring around each."""
    scale = np.sqrt(mask.size / (640 * 480))
    lab, n = ndimage.label(mask)
    out = []
    for i in range(1, n + 1):
        reg = lab == i
        if reg.sum() < min_frac * mask.size:
            continue
        ring = ndimage.binary_dilation(reg, iterations=max(1, int(round(12 * scale)))) & ~mask
        if ring.sum() >= 50:
            out.append((reg, ring))
    return out


def depth_stats(depth_mm, reg, gt_mm=None):
    d = depth_mm[reg]
    s = dict(missing=float((d == 0).mean()))
    if gt_mm is not None:
        g = gt_mm[reg]
        ok = (d > 0) & (g > 0)
        known = g > 0
        if known.mean() > 0.5:
            s["behind"] = float(((d - g > BEHIND_MM) & ok).sum() / known.sum())
            s["right"] = float(((np.abs(d - g) <= BEHIND_MM) & ok).sum() / known.sum())
    return s


def rgb_contrast(lum, reg, ring):
    return float(abs(np.median(lum[reg]) - np.median(lum[ring])))


def thermal_contrast(th, reg, ring, min_valid=0.8):
    valid = th > 0
    if valid[reg].mean() < min_valid or (ring & valid).sum() < 50:
        return None
    return float(abs(np.median(th[reg & valid]) - np.median(th[ring & valid])))


def quant(v):
    return {f"p{int(100 * q)}": float(x) for q, x in zip(Q, np.quantile(v, Q))}


def line(v, fmt="{:.2f}"):
    return ", ".join(f"p{int(100 * q)} " + fmt.format(x) for q, x in zip(Q, np.quantile(v, Q)))


@click.command()
@click.option("--root", default=os.path.expanduser("~/datasets/transparent/unified"), show_default=True)
@click.option("--n", default=600, show_default=True, help="frames per dataset")
@click.option("--out", default=None)
def main(root, n, out):
    root = Path(root)
    man = pd.read_parquet(root / "manifest.parquet")
    res = {}
    for ds in ("mju_glass", "trobi", "transpose", "rgbt_glass", "rgbd_gsd"):
        rows = man[(man.dataset == ds) & man.mask_path.notna() & (man.glass_frac > 0)]
        per = max(1, n // rows.group.nunique())
        rows = pd.concat([g.sample(min(len(g), per), random_state=0) for _, g in rows.groupby("group")])
        if len(rows) > n:
            rows = rows.sample(n, random_state=0)
        recs = []
        for r in rows.itertuples():
            mk = np.asarray(Image.open(root / r.mask_path))
            mk = (mk[..., 0] if mk.ndim == 3 else mk) == 1
            lum = luminance(Image.open(root / r.rgb_path))
            metric = isinstance(r.depth_path, str) and r.depth_kind != "relative"
            dep = np.asarray(Image.open(root / r.depth_path)).astype(np.float32) if metric else None
            gt = np.asarray(Image.open(root / r.depth_gt_path)).astype(np.float32) if isinstance(r.depth_gt_path, str) else None
            th = np.asarray(Image.open(root / r.thermal_path)).astype(np.float32) if isinstance(r.thermal_path, str) else None
            for reg, ring in glass_regions(mk):
                rec = dict(rgb=rgb_contrast(lum, reg, ring))
                if dep is not None:
                    rec.update(depth_stats(dep, reg, gt))
                if th is not None:
                    c = thermal_contrast(th, reg, ring)
                    if c is not None:
                        rec["thermal"] = c * (0.01 if r.thermal_kind == "kelvin_x100" else 1.0 / TRANSPOSE_NOISE)
                recs.append(rec)
        df = pd.DataFrame(recs)
        res[ds] = dict(frames=len(rows), groups=int(rows.group.nunique()), regions=len(df))
        print(f"## {ds}: {len(df)} glass regions in {len(rows)} frames of {rows.group.nunique()} groups"
              + (f"; depth sensor: {SENSOR[ds]}" if "missing" in df else ""))
        print(f"  rgb contrast (grey levels of 255): {line(df.rgb, '{:.1f}')}; below 3: {(df.rgb < 3).mean():.3f}, "
              f"below 10: {(df.rgb < 10).mean():.3f}")
        res[ds]["rgb"] = dict(quant(df.rgb), below_3=float((df.rgb < 3).mean()), below_10=float((df.rgb < 10).mean()))
        if "missing" in df:
            m = df.missing.dropna()
            print(f"  depth missing: {line(m)}; hole-like regions (missing > 0.5): {(m > 0.5).mean():.3f}")
            res[ds]["depth_missing"] = dict(quant(m), hole_like=float((m > 0.5).mean()))
        if "behind" in df:
            b = df.dropna(subset=["behind"])
            print(f"  depth behind the glass by > 2 cm: {line(b.behind)}; see-through regions (behind > 0.5): "
                  f"{(b.behind > 0.5).mean():.3f}; right (within 2 cm): {line(b.right)}; regions mostly right "
                  f"(right > 0.5): {(b.right > 0.5).mean():.3f}")
            res[ds]["depth_behind"] = dict(quant(b.behind), see_through=float((b.behind > 0.5).mean()))
            res[ds]["depth_right"] = dict(quant(b.right), mostly_right=float((b.right > 0.5).mean()))
        if "thermal" in df:
            t, unit = df.thermal.dropna(), ("degC" if ds == "rgbt_glass" else "noise std")
            lim = (0.1, 0.5, 1.0, 3.0) if ds == "rgbt_glass" else (1.0, 3.0, 10.0, 30.0)
            print(f"  thermal contrast ({unit}): {line(t)}; below " + ", ".join(f"{x:g}: {(t < x).mean():.3f}" for x in lim))
            res[ds]["thermal"] = dict(quant(t), unit=unit, below={str(x): float((t < x).mean()) for x in lim})
        if ds == "transpose":      # the three sensors on the same tabletop regions
            j = df.dropna(subset=["behind", "thermal"])
            dep_blind, th_weak, rgb_weak = j.behind > 0.5, j.thermal < 3.0, j.rgb < 3.0
            print(f"  the three together ({len(j)} regions): depth see-through {dep_blind.mean():.3f}, thermal below 3 "
                  f"noise std {th_weak.mean():.3f}, rgb contrast below 3 grey levels {rgb_weak.mean():.3f}; depth "
                  f"see-through and thermal strong (>= 10 std) {(dep_blind & (j.thermal >= 10)).mean():.3f}; all three "
                  f"weak {(dep_blind & th_weak & rgb_weak).mean():.3f}")
            res[ds]["joint"] = dict(regions=len(j), depth_see_through=float(dep_blind.mean()),
                                    thermal_below_3=float(th_weak.mean()), rgb_below_3=float(rgb_weak.mean()))
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        json.dump(res, open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
