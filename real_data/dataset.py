"""
PyTorch loader over the unified manifest (see common.py for the on-disk format).

    from real_data.dataset import TransparentPerceptionDataset, balanced_sampler
    ds = TransparentPerceptionDataset(datasets=["rgbt_glass", "mju_glass"], splits=["train"], size=224)
    dl = DataLoader(ds, batch_size=32, sampler=balanced_sampler(ds), num_workers=8)

Every item has the same keys whatever the source, so mixed batches collate with the default collate_fn:
    rgb         (3,H,W) float32 in [0,1]   (T-ROBI grey is repeated to 3 channels)
    depth       (1,H,W) float32            encoding="metric": metres, 0 = invalid
                                           encoding="sim":    clip(m, 0, depth_max_m) / depth_max_m, 0 = invalid
                                           (sim stores this as uint8 * 255, see sim/env.py)
                                           relative-depth sources (RGBD-GSD) give value / 65535 in both encodings
    depth_gt    (1,H,W) float32            same encoding as depth (zeros when absent)
    thermal     (1,H,W) float32 in [0,1]   per-frame min-max normalised over valid pixels (same as the sim)
    mask        (H,W)   int64              0 bg, 1 transparent, 255 ignore (255 everywhere when absent)
    has_depth, has_depth_gt, has_thermal, has_mask, depth_is_metric : bool
    sample_id, dataset : str
Missing modalities are zero tensors with the matching has_* flag False.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, WeightedRandomSampler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DEFAULT_ROOT, MASK_IGNORE  # noqa: E402

MODALITIES = ("rgb", "depth", "depth_gt", "thermal", "mask")


def load_manifest(root=DEFAULT_ROOT, datasets=None, splits=None, require=()):
    df = pd.read_parquet(Path(root) / "manifest.parquet")
    if datasets:
        unknown = set(datasets) - set(df["dataset"].unique())
        assert not unknown, f"unknown datasets {unknown}; have {sorted(df['dataset'].unique())}"
        df = df[df["dataset"].isin(datasets)]
    if splits:
        df = df[df["split"].isin(splits)]
    for m in require:
        df = df[df[f"{m}_path"].notna()]
    return df.reset_index(drop=True)


class TransparentPerceptionDataset(Dataset):
    def __init__(self, root=DEFAULT_ROOT, datasets=None, splits=("train",), require=(), size=224,
                 resize="crop", encoding="metric", depth_max_m=2.0, manifest=None):
        """
        size     int (square) or (H, W); None keeps native resolution (batch_size=1 or a custom collate then).
        resize   "crop": scale the short side to size, then centre-crop (keeps aspect ratio)
                 "stretch": resize straight to size (what sim/env.py does)
        encoding "metric" or "sim" (see module docstring)
        require  modalities a sample must have, e.g. ("depth", "mask")
        """
        assert resize in ("crop", "stretch") and encoding in ("metric", "sim")
        self.root = Path(root)
        self.df = manifest if manifest is not None else load_manifest(root, datasets, splits, require)
        assert len(self.df), "no samples match the filters"
        self.size = None if size is None else ((size, size) if isinstance(size, int) else tuple(size))
        self.resize, self.encoding, self.depth_max_m = resize, encoding, depth_max_m
        self._rows = self.df.to_dict("records")

    def __len__(self):
        return len(self._rows)

    # -------------------------------------------------------------- geometry
    def _box(self, h, w):
        """Crop box (left, top, right, bottom) in source pixels."""
        if self.size is None or self.resize == "stretch":
            return (0, 0, w, h)
        th, tw = self.size
        s = max(th / h, tw / w)
        ch, cw = round(th / s), round(tw / s)
        top, left = (h - ch) // 2, (w - cw) // 2
        return (left, top, left + cw, top + ch)

    def _fit(self, im, box, nearest):
        im = im.crop(box)
        if self.size is None:
            return im
        th, tw = self.size
        return im.resize((tw, th), Image.NEAREST if nearest else Image.BILINEAR, reducing_gap=None if nearest else 2.0)

    def _read(self, rel):
        return Image.open(self.root / rel)

    # -------------------------------------------------------------- item
    def __getitem__(self, i):
        r = self._rows[i]
        h, w = int(r["height"]), int(r["width"])
        box = self._box(h, w)
        th, tw = self.size if self.size else (h, w)
        out = {"sample_id": r["sample_id"], "dataset": r["dataset"]}

        rgb = self._fit(self._read(r["rgb_path"]).convert("RGB"), box, nearest=False)
        out["rgb"] = torch.from_numpy(np.asarray(rgb, np.float32) / 255.0).permute(2, 0, 1).contiguous()

        metric = r["depth_kind"] == "metric_mm"
        out["depth_is_metric"] = bool(metric) if pd.notna(r["depth_kind"]) else False
        for key in ("depth", "depth_gt"):
            path = r[f"{key}_path"]
            has = isinstance(path, str)
            out[f"has_{key}"] = has
            if not has:
                out[key] = torch.zeros(1, th, tw)
                continue
            d = np.asarray(self._fit(self._read(path), box, nearest=True), np.float32)
            if metric:
                d = d / 1000.0
                if self.encoding == "sim":
                    d = np.where(d > 0, np.clip(d, 0, self.depth_max_m) / self.depth_max_m, 0.0)
            else:
                d = d / 65535.0
            out[key] = torch.from_numpy(d.astype(np.float32))[None]

        path = r["thermal_path"]
        out["has_thermal"] = isinstance(path, str)
        if out["has_thermal"]:
            t = np.asarray(self._fit(self._read(path), box, nearest=True), np.float32)
            v = t > 0
            if v.any():
                lo, hi = t[v].min(), t[v].max()
                t = np.where(v, (t - lo) / max(hi - lo, 1e-6), 0.0)
            out["thermal"] = torch.from_numpy(t.astype(np.float32))[None]
        else:
            out["thermal"] = torch.zeros(1, th, tw)

        path = r["mask_path"]
        out["has_mask"] = isinstance(path, str)
        if out["has_mask"]:
            out["mask"] = torch.from_numpy(np.asarray(self._fit(self._read(path), box, nearest=True), np.int64).copy())
        else:
            out["mask"] = torch.full((th, tw), MASK_IGNORE, dtype=torch.int64)
        return out


def balanced_sampler(ds, num_samples=None, by="dataset", generator=None):
    """WeightedRandomSampler giving every value of `by` (default: source dataset) equal total weight."""
    key = ds.df[by]
    w = 1.0 / key.map(key.value_counts()).to_numpy(np.float64)
    return WeightedRandomSampler(torch.as_tensor(w), num_samples or len(ds), replacement=True, generator=generator)
