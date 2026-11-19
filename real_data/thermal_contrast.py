"""
How much thermal contrast does real glass have against what is around it? Measured on the unified real datasets
(real_data/README.md) that have a thermal image and a glass mask, to see how often a thermal camera cannot tell
real glass from its surroundings.

  python real_data/thermal_contrast.py [--root ~/datasets/transparent/unified] [--n 800] [--out FILE.json]

Per glass region (connected part of the mask, at least 400 px of a 640x480 image, with a valid thermal reading on at
least 80% of it): contrast = median thermal value on the region - median on a 12 px ring around it (pixels of other
glass regions left out).
  rgbt_glass  thermal in degC (Kelvin x 100), glass panes: windows, doors, walls;
  transpose   thermal in raw 14-bit counts of the FLIR camera (not calibrated to temperature), transparent tabletop
              objects. Its contrast is given in counts and in units of the camera noise (0.89 counts per pixel, the
              value sim/configs/thermal/calibrated_v1.yaml was fitted with).
Printed: quantiles of |contrast| and the share of regions below a few limits. Frames are drawn evenly from every
sequence / location of the dataset (--n frames per dataset, seed 0).
"""
import json
import os
from pathlib import Path

import click
import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage

TRANSPOSE_NOISE = 0.89   # counts per pixel at 640x512 (sim/configs/thermal/calibrated_v1.yaml)


def regions(thermal, mask, min_px=400, ring=12, min_valid=0.8):
    lab, n = ndimage.label(mask)
    out = []
    for i in range(1, n + 1):
        reg = lab == i
        if reg.sum() < min_px:
            continue
        around = ndimage.binary_dilation(reg, iterations=ring) & ~mask
        valid = thermal > 0
        if valid[reg].mean() < min_valid or (around & valid).sum() < 50:
            continue
        out.append((float(np.median(thermal[reg & valid]) - np.median(thermal[around & valid])), int(reg.sum())))
    return out


@click.command()
@click.option("--root", default=os.path.expanduser("~/datasets/transparent/unified"), show_default=True)
@click.option("--n", default=800, show_default=True, help="frames per dataset")
@click.option("--out", default=None)
def main(root, n, out):
    root = Path(root)
    man = pd.read_parquet(root / "manifest.parquet")
    res = {}
    for ds, unit, scale, limits in (("rgbt_glass", "degC", 0.01, (0.1, 0.2, 0.3, 0.5, 1.0)),
                                    ("transpose", "counts", 1.0, (1.0, 2.0, 3.0, 5.0, 10.0))):
        rows = man[(man.dataset == ds) & man.thermal_path.notna() & man.mask_path.notna() & (man.glass_frac > 0)]
        per = max(1, n // rows.group.nunique())
        rows = pd.concat([g.sample(min(len(g), per), random_state=0) for _, g in rows.groupby("group")])
        vals = []
        for r in rows.itertuples():
            th = np.asarray(Image.open(root / r.thermal_path)).astype(np.float32) * scale
            mk = np.asarray(Image.open(root / r.mask_path)) > 0
            if mk.ndim == 3:
                mk = mk[..., 0]
            vals += regions(th, mk)
        c = np.abs(np.array([v[0] for v in vals]))
        q = np.quantile(c, [0.1, 0.25, 0.5, 0.75, 0.9])
        res[ds] = dict(unit=unit, frames=len(rows), groups=int(rows.group.nunique()), regions=len(c),
                       quantiles=dict(zip(("p10", "p25", "p50", "p75", "p90"), map(float, q))),
                       below={str(l): float((c < l).mean()) for l in limits},
                       signed_median=float(np.median([v[0] for v in vals])))
        print(f"{ds}: {len(c)} glass regions in {len(rows)} frames of {rows.group.nunique()} groups; |contrast| in "
              f"{unit}: p10 {q[0]:.2f}, p25 {q[1]:.2f}, median {q[2]:.2f}, p75 {q[3]:.2f}, p90 {q[4]:.2f}; "
              f"signed median {res[ds]['signed_median']:+.2f}")
        print("   share of regions below: " + ", ".join(f"{l:g} {unit} {(c < l).mean():.3f}" for l in limits))
        if ds == "transpose":
            print("   in units of the camera noise (0.89 counts): median "
                  f"{q[2] / TRANSPOSE_NOISE:.1f}, share below 1 noise std {(c < TRANSPOSE_NOISE).mean():.3f}, "
                  f"below 3 {(c < 3 * TRANSPOSE_NOISE).mean():.3f}")
            res[ds]["below_1_noise"] = float((c < TRANSPOSE_NOISE).mean())
            res[ds]["below_3_noise"] = float((c < 3 * TRANSPOSE_NOISE).mean())
    if out:
        json.dump(res, open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
