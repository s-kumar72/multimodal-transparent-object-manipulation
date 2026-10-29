"""
Render one or more samples of the unified data as a labelled, human-viewable PNG.
The stored depth / thermal / mask PNGs look black in an image viewer (e.g. 700 mm is 1% of the
16-bit range, mask values are 0/1), so this colour-maps them and prints their units.

run via
    python real_data/show_sample.py transpose/seq_H_05_000686 rgbt_glass/law_center_KWOE1357
    python real_data/show_sample.py --random 3 --dataset mju_glass
Output: real_data/out/samples/<dataset>__<local_id>.png
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DEFAULT_ROOT  # noqa: E402

TILE_W, TILE_H, BAR = 400, 300, 34
_VIRIDIS = np.array([[68, 1, 84], [59, 82, 139], [33, 145, 140], [94, 201, 98], [253, 231, 37]], np.float64)


def _cmap(x):
    """x in [0,1] -> viridis-like RGB."""
    x = np.clip(x, 0, 1) * (len(_VIRIDIS) - 1)
    i = np.minimum(x.astype(int), len(_VIRIDIS) - 2)
    t = (x - i)[..., None]
    return (_VIRIDIS[i] * (1 - t) + _VIRIDIS[i + 1] * t).astype(np.uint8)


def _scalar(a, scale, unit, lo=None, hi=None):
    """Colour-map valid (>0) pixels; invalid = black. Returns (image, caption)."""
    v = a > 0
    img = np.zeros(a.shape + (3,), np.uint8)
    if not v.any():
        return img, "no valid pixels"
    x = a.astype(np.float64) * scale[0] + scale[1]
    lo = np.percentile(x[v], 1) if lo is None else lo
    hi = np.percentile(x[v], 99) if hi is None else hi
    img[v] = _cmap((x[v] - lo) / max(hi - lo, 1e-9))
    fmt = ".2f" if hi - lo < 10 else ".0f"
    return img, f"{lo:{fmt}}-{hi:{fmt}} {unit}, {(~v).mean() * 100:.0f}% invalid (black)"


def render(row, root):
    root = Path(root)
    tiles = []
    rgb = Image.open(root / row["rgb_path"]).convert("RGB")
    tiles.append((np.asarray(rgb), f"rgb  {row['width']}x{row['height']}"))
    lo = hi = None
    if isinstance(row["depth_gt_path"], str):  # share the colour range so raw vs GT are comparable
        g = np.asarray(Image.open(root / row["depth_gt_path"])).astype(np.float64)
        lo, hi = np.percentile(g[g > 0], [1, 99])
    for key, label in (("depth_path", "depth"), ("depth_gt_path", "depth_gt")):
        p = row[key]
        if not isinstance(p, str):
            tiles.append((None, f"{label}: not in this dataset")); continue
        a = np.asarray(Image.open(root / p))
        if row["depth_kind"] == "metric_mm":
            img, cap = _scalar(a, (1, 0), "mm", lo, hi)
        else:
            img, cap = _scalar(a, (1 / 65535, 0), "relative", None, None)
            cap = cap.replace(" relative", " (relative, no unit)")
        tiles.append((img, f"{label}: {cap}"))
    p = row["thermal_path"]
    if isinstance(p, str):
        a = np.asarray(Image.open(root / p))
        if row["thermal_kind"] == "kelvin_x100":
            img, cap = _scalar(a, (0.01, -273.15), "C")
        else:
            img, cap = _scalar(a, (1, 0), "raw counts")
        tiles.append((img, f"thermal: {cap}"))
    else:
        tiles.append((None, "thermal: not in this dataset"))
    m = np.asarray(Image.open(root / row["mask_path"]))
    ov = np.asarray(rgb).astype(np.float64).copy()
    ov[m == 1] = ov[m == 1] * 0.45 + np.array([0, 255, 120]) * 0.55
    ov[m == 255] = [255, 0, 0]
    tiles.append((ov.astype(np.uint8), f"mask: green=transparent {(m == 1).mean() * 100:.0f}%, red=ignore"))

    canvas = Image.new("RGB", (TILE_W * 3, (TILE_H + BAR) * 2 + BAR), (255, 255, 255))
    d = ImageDraw.Draw(canvas)
    d.text((8, 8), f"{row['sample_id']}   split={row['split']}   group={row['group']}", fill=(0, 0, 0))
    for k, (img, cap) in enumerate(tiles):
        x, y = (k % 3) * TILE_W, BAR + (k // 3) * (TILE_H + BAR)
        if img is None:
            canvas.paste((235, 235, 235), (x + 2, y + BAR, x + TILE_W - 2, y + BAR + TILE_H - 2))
        else:
            im = Image.fromarray(img).resize((TILE_W - 4, TILE_H - 4), Image.NEAREST if k else Image.BILINEAR)
            canvas.paste(im, (x + 2, y + BAR))
        d.text((x + 6, y + 10), cap, fill=(0, 0, 0))
    return canvas


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("sample_ids", nargs="*")
    ap.add_argument("--random", type=int, default=0, help="also render N random samples")
    ap.add_argument("--dataset", default=None, help="restrict --random to one dataset")
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "out" / "samples"))
    a = ap.parse_args()
    df = pd.read_parquet(Path(a.root) / "manifest.parquet").set_index("sample_id", drop=False)
    ids = list(a.sample_ids)
    if a.random:
        pool = df if a.dataset is None else df[df["dataset"] == a.dataset]
        ids += pool.sample(n=a.random).index.tolist()
    assert ids, "give sample ids or --random N"
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    for sid in ids:
        f = out / (sid.replace("/", "__") + ".png")
        render(df.loc[sid], a.root).save(f)
        print(f)
