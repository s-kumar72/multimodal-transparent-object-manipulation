"""
Shared schema and writers for the unified transparent-object perception format.

Every converter turns one public dataset into
    <root>/<dataset>/{rgb,depth,thermal,mask}/<local_id>.<ext>
    <root>/<dataset>/manifest.parquet   (one row per sample, schema = COLUMNS)
    <root>/<dataset>/meta.json          (license, citation, notes, counts)
and build_manifest() concatenates the per-dataset manifests into <root>/manifest.parquet.

On-disk encodings (identical across datasets):
    rgb      8-bit RGB, original resolution, original codec kept (.jpg or .png)
    depth    16-bit PNG, 0 = invalid / no reading
               depth_kind == "metric_mm": value = millimetres
               depth_kind == "relative" : per-image normalised, larger = farther, no metric scale
    thermal  16-bit PNG, 0 = invalid
               thermal_kind == "kelvin_x100": value = temperature in Kelvin * 100
               thermal_kind == "raw_counts" : uncalibrated sensor counts (e.g. 14-bit radiometric)
    depth_gt same encoding as depth (metric_mm); completed / ground-truth depth where the
             dataset provides it (e.g. T-ROBI scans with anti-reflective spray)
    mask     8-bit PNG, 0 = background, 1 = transparent object / glass, 255 = ignore
All modalities of a sample share the RGB resolution and are pixel-aligned to it unless
the row's `aligned` flag is False.

"""
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

DEFAULT_ROOT = os.environ.get("TRANSPARENT_DATA_ROOT", str(Path.home() / "datasets/transparent/unified"))
DEFAULT_RAW = os.environ.get("TRANSPARENT_RAW_ROOT", str(Path.home() / "datasets/transparent/raw"))

MASK_BG, MASK_FG, MASK_IGNORE = 0, 1, 255

# column -> pandas dtype; nullable where a modality / calibration may be missing
COLUMNS = {
    "sample_id": "string",      # "<dataset>/<local_id>", globally unique
    "dataset": "string",
    "split": "string",          # train | val | test
    "group": "string",          # scene / session id; never split across train/val/test
    "rgb_path": "string",       # paths are relative to the unified root
    "depth_path": "string",
    "depth_gt_path": "string",  # completed / ground-truth depth, same encoding as depth
    "thermal_path": "string",
    "mask_path": "string",
    "height": "int32",
    "width": "int32",
    "depth_kind": "string",     # metric_mm | relative | <NA>
    "thermal_kind": "string",   # kelvin_x100 | raw_counts | <NA>
    "aligned": "boolean",       # depth/thermal/mask pixel-aligned with rgb
    "glass_frac": "float32",    # fraction of mask pixels == 1 (NaN if no mask)
    "fx": "float32", "fy": "float32", "cx": "float32", "cy": "float32",
    "extra": "string",          # JSON blob with dataset-specific fields
}


def write_png(path, arr):
    """Write uint8 / uint16 HxW or HxWx3 array as PNG (16-bit preserved)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.ascontiguousarray(arr)
    if arr.dtype == np.uint16:
        assert arr.ndim == 2, "16-bit PNGs must be single channel"
        arr = arr.astype("<u2", copy=False)
    im = Image.fromarray(arr)  # uint16 -> "I;16" in Pillow 10..13 (passing mode= errors in 13)
    assert arr.dtype != np.uint16 or im.mode == "I;16", im.mode
    im.save(path, optimize=False, compress_level=3)


def write_bytes(path, data: bytes):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def binarize_mask(m, thresh=127):
    """Any 8-bit/boolean glass mask -> {0,1} uint8."""
    m = np.asarray(m)
    if m.ndim == 3:
        m = m[..., 0]
    return (m > thresh).astype(np.uint8) if m.dtype != bool else m.astype(np.uint8)


def celsius_to_kelvin_x100(t_c, valid=None):
    t = np.asarray(t_c, dtype=np.float64)
    out = np.clip(np.round((t + 273.15) * 100.0), 1, 65535).astype(np.uint16)
    bad = ~np.isfinite(t) if valid is None else ~(valid & np.isfinite(t))
    out[bad] = 0
    return out


def make_row(**kw):
    row = {c: kw.get(c) for c in COLUMNS}
    if isinstance(row["extra"], dict):
        row["extra"] = json.dumps(row["extra"], sort_keys=True)
    return row


def group_split(groups, test_frac, val_frac, seed, fixed_test=None):
    """Deterministic group-level split. Returns {group: split}.

    fixed_test: optional set of groups forced into test (official test sets).
    Groups are shuffled with a seeded RNG and a group joins test (then val) only if that
    brings the split's sample count closer to its target, so large groups cannot overshoot
    by more than half their size. Everything left over is train.
    """
    counts = pd.Series(groups).value_counts().sort_index()
    rng = np.random.default_rng(seed)
    order = list(counts.index)
    rng.shuffle(order)
    total = counts.sum()
    assign, n_test, n_val = {}, 0, 0
    fixed_test = set(fixed_test or [])
    for g in order:
        if g in fixed_test:
            assign[g] = "test"; n_test += counts[g]
    for g in order:
        if g in assign:
            continue
        c = counts[g]
        if n_test + c / 2 <= test_frac * total:
            assign[g] = "test"; n_test += c
        elif n_val + c / 2 <= val_frac * total:
            assign[g] = "val"; n_val += c
        else:
            assign[g] = "train"
    return assign


def save_dataset(root, name, rows, meta):
    df = pd.DataFrame(rows).reindex(columns=list(COLUMNS)).astype(COLUMNS)
    df = df.sort_values("sample_id").reset_index(drop=True)
    assert df["sample_id"].is_unique, f"{name}: duplicate sample ids"
    out = Path(root) / name
    out.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out / "manifest.parquet", index=False)
    meta = dict(meta)
    meta["counts"] = {s: int(n) for s, n in df["split"].value_counts().items()}
    meta["n_samples"] = int(len(df))
    meta["modalities"] = [m for m in ("rgb", "depth", "depth_gt", "thermal", "mask") if df[f"{m}_path"].notna().any()]
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[{name}] {len(df)} samples  splits={meta['counts']}  modalities={meta['modalities']}")
    return df


def build_manifest(root=DEFAULT_ROOT):
    """Concatenate every <root>/<dataset>/manifest.parquet into <root>/manifest.parquet."""
    root = Path(root)
    parts = [pd.read_parquet(p) for p in sorted(root.glob("*/manifest.parquet"))]
    df = pd.concat([p.reindex(columns=list(COLUMNS)) for p in parts], ignore_index=True).astype(COLUMNS)
    df.to_parquet(root / "manifest.parquet", index=False)
    print(f"[manifest] {len(df)} samples from {df['dataset'].nunique()} datasets -> {root / 'manifest.parquet'}")
    print(df.groupby(["dataset", "split"]).size().unstack(fill_value=0).to_string())
    return df
