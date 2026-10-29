"""
Validate the unified datasets and save one preview grid per dataset.

run via python real_data/inspect_unified.py [--root ~/datasets/transparent/unified] [--full]
Without --full every 10th sample is checked; with --full every sample is.
Previews go to real_data/out/preview_<dataset>.png (rows = samples; cols = rgb, depth, depth_gt, thermal, mask).
"""
import argparse
import sys
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DEFAULT_ROOT  # noqa: E402


def check(row, root):
    errs, st = [], {}
    h, w = row["height"], row["width"]
    for m in ("rgb", "depth", "depth_gt", "thermal", "mask"):
        p = row[f"{m}_path"]
        if not isinstance(p, str):
            continue
        f = Path(root) / p
        if not f.exists():
            errs.append(f"missing {m}"); continue
        a = np.asarray(Image.open(f))
        if a.shape[:2] != (h, w):
            errs.append(f"{m} shape {a.shape[:2]} != {(h, w)}")
        if m in ("depth", "depth_gt", "thermal") and a.dtype != np.uint16:
            errs.append(f"{m} dtype {a.dtype}")
        if m == "mask":
            bad = set(np.unique(a).tolist()) - {0, 1, 255}
            if bad: errs.append(f"mask values {sorted(bad)[:5]}")
            gf = float((a == 1).mean())
            if abs(gf - row["glass_frac"]) > 1e-4: errs.append(f"glass_frac {row['glass_frac']} vs {gf}")
        if m == "depth":
            v = a[a > 0]
            st["depth_valid"] = float((a > 0).mean())
            if row["depth_kind"] == "metric_mm" and v.size:
                st["depth_med_m"] = float(np.median(v)) / 1000
            if isinstance(row["mask_path"], str):
                mk = np.asarray(Image.open(Path(root) / row["mask_path"])) == 1
                if mk.any(): st["depth_valid_on_glass"] = float((a[mk] > 0).mean())
        if m == "depth_gt" and isinstance(row["depth_path"], str) and isinstance(row["mask_path"], str):
            raw = np.asarray(Image.open(Path(root) / row["depth_path"])).astype(np.float64)
            mk = np.asarray(Image.open(Path(root) / row["mask_path"])) == 1
            g = a.astype(np.float64)
            on = mk & (g > 0)
            if on.any():
                err = np.abs(raw[on] - g[on])
                # "bad" = no reading, or off by more than 2 cm from the ground truth
                st["depth_bad_on_glass"] = float(((raw[on] == 0) | (err > 20)).mean())
                ok = raw[on] > 0
                if ok.any(): st["depth_err_on_glass_mm"] = float(np.median(err[ok]))
        if m == "thermal" and row["thermal_kind"] == "kelvin_x100":
            v = a[a > 0] / 100 - 273.15
            if v.size: st["t_min_c"], st["t_max_c"] = float(v.min()), float(v.max())
    return row["sample_id"], errs, st


def preview(df, root, out, n=6):
    rows = []
    pick = df.sample(n=min(n, len(df)), random_state=0) if len(df) > n else df
    for _, r in pick.iterrows():
        tiles = []
        for m in ("rgb", "depth", "depth_gt", "thermal", "mask"):
            p = r[f"{m}_path"]
            if not isinstance(p, str):
                tiles.append(np.full((180, 240, 3), 40, np.uint8)); continue
            a = np.asarray(Image.open(Path(root) / p))
            if m == "rgb":
                img = Image.fromarray(a).convert("RGB")
            elif m == "mask":
                img = Image.fromarray(np.where(a == 255, 128, a * 255).astype(np.uint8)).convert("RGB")
            else:
                v = a > 0; x = np.zeros(a.shape, np.float32)
                if v.any():
                    lo, hi = np.percentile(a[v], [1, 99]); x[v] = np.clip((a[v] - lo) / max(hi - lo, 1), 0, 1) * 235 + 20
                img = Image.fromarray(x.astype(np.uint8)).convert("RGB")
            tiles.append(np.asarray(img.resize((240, 180), Image.NEAREST if m != "rgb" else Image.BILINEAR)))
        rows.append(np.concatenate([np.pad(t, ((2, 2), (2, 2), (0, 0)), constant_values=255) for t in tiles], 1))
    Image.fromarray(np.concatenate(rows, 0)).save(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    df = pd.read_parquet(Path(a.root) / "manifest.parquet")
    assert df["sample_id"].is_unique
    # no capture group may straddle splits, except the known train/test overlap of RGBD-GSD's official split
    span = df.groupby(["dataset", "group"])["split"].agg(lambda s: "+".join(sorted(set(s))))
    span = span[span.str.contains(r"\+")]
    known = span[(span.index.get_level_values(0) == "rgbd_gsd") & (span == "test+train")]
    leaks = span.drop(known.index)
    sub = df if a.full else df.iloc[::10]
    with Pool(a.workers) as pool:
        res = pool.map(partial(check, root=a.root), sub.to_dict("records"), chunksize=16)
    errs = [(s, e) for s, e, _ in res if e]
    st = pd.DataFrame([dict(sample_id=s, **x) for s, _, x in res]).set_index("sample_id")
    st["dataset"] = sub.set_index("sample_id")["dataset"]
    print(f"checked {len(sub)} / {len(df)} samples, {len(errs)} with errors")
    for s, e in errs[:20]:
        print("  ", s, e)
    print("\nsamples per split:")
    print(df.groupby(["dataset", "split"]).size().unstack(fill_value=0).to_string())
    print("\nmodalities present (fraction of samples):")
    print(df.groupby("dataset")[[f"{m}_path" for m in ("rgb", "depth", "depth_gt", "thermal", "mask")]].agg(lambda c: c.notna().mean()).round(2).to_string())
    print("\nper-dataset stats (checked subset):")
    agg = {"depth_valid": "mean", "depth_valid_on_glass": "mean", "depth_bad_on_glass": "mean", "depth_err_on_glass_mm": "median",
           "depth_med_m": "median", "t_min_c": "min", "t_max_c": "max"}
    print(st.groupby("dataset").agg({k: v for k, v in agg.items() if k in st}).round(3).to_string())
    print("\nglass_frac: mean / share of images with no glass")
    print(df.groupby("dataset")["glass_frac"].agg(mean="mean", no_glass=lambda g: (g == 0).mean()).round(3).to_string())
    print(f"\nsplit-straddling groups: {len(leaks)} unexpected" + ("" if leaks.empty else "\n" + leaks.to_string()))
    print(f"  known (RGBD-GSD official test shares groups with train): {len(known)} groups, "
          f"{int(df[df['dataset'].eq('rgbd_gsd') & df['split'].eq('test') & df['group'].isin(known.index.get_level_values(1))].shape[0])} test images")
    out = Path(__file__).resolve().parent / "out"
    out.mkdir(exist_ok=True)
    for name, g in df.groupby("dataset"):
        preview(g, a.root, out / f"preview_{name}.png")
    print(f"\npreviews saved to {out}/preview_<dataset>.png")
    sys.exit(1 if errs or not leaks.empty else 0)
