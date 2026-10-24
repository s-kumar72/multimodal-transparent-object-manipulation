"""
Thermal glass-segmentation data: TRansPose frame loading, resize + CNR caching, thermal-safe augmentation.

TRansPose thermal layout per sequence: <seq>/cam_T/14bit/<frame>.png (uint16 raw, 640x512),
cam_T/pose/<frame>.json (objects: obj_name, mask_img), cam_T/mask/<frame>_<obj>.png (visible mask, > 0).
"""
import json
import os
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset

# obj_<category>_...: C = chemical lab, H = household, T = recyclable trash (all transparent); N = non-transparent
TRANSPARENT_CATEGORIES = ("C", "H", "T")


def list_sequences(root):
    """Sequence ids (folder names) under a TRansPose sequences root."""
    return sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d, "cam_T", "14bit")))


def list_frames(root, seq):
    """(cam_T dir, frame id) per thermal frame of one sequence."""
    cam = os.path.join(root, seq, "cam_T")
    return [(cam, f[:-4]) for f in sorted(os.listdir(os.path.join(cam, "14bit"))) if f.endswith(".png")]


def load_frame(frame):
    """-> raw (H, W) uint16 14-bit counts, mask (H, W) bool in the thermal view (all transparent objects = 1)."""
    cam, idx = frame
    raw = np.array(Image.open(os.path.join(cam, "14bit", f"{idx}.png")))
    with open(os.path.join(cam, "pose", f"{idx}.json")) as f:
        objs = json.load(f).values()
    mask = np.zeros(raw.shape, bool)
    for o in objs:
        path = os.path.join(cam, "mask", o["mask_img"])
        if o["obj_name"].split("_")[1] in TRANSPARENT_CATEGORIES and os.path.exists(path):
            mask |= np.array(Image.open(path)) > 0
    return raw, mask


def compute_cnr(raw, mask, ring=8):
    """Glass-vs-background contrast-to-noise ratio on raw counts: |mean_glass - mean_bg| / std_bg.
    Background = `ring`-px band around the glass (local contrast). NaN when there is no glass."""
    raw = torch.as_tensor(raw, dtype=torch.float32)
    m = torch.as_tensor(mask).bool()
    grown = F.max_pool2d(m[None, None].float(), 2 * ring + 1, stride=1, padding=ring)[0, 0].bool()
    bg = raw[grown & ~m]
    if not m.any() or bg.numel() < 2:
        return float("nan")
    return ((raw[m].mean() - bg.mean()).abs() / (bg.std() + 1e-6)).item()


def resize_frame(raw, mask, size=96):
    """Native raw counts + mask -> (size, size) float32 counts (area), uint8 mask (nearest), CNR at native res."""
    cnr = compute_cnr(raw, mask)
    raw = torch.as_tensor(np.asarray(raw, np.float32))[None, None]
    mask = torch.as_tensor(np.asarray(mask, np.float32))[None, None]
    raw = F.interpolate(raw, size=(size, size), mode="area")[0, 0].numpy()
    mask = F.interpolate(mask, size=(size, size), mode="nearest")[0, 0].numpy().astype(np.uint8)
    return raw, mask, cnr


def load_transpose(root, sequences, size=96, cache_dir=None):
    """Resized raw counts (N, S, S), masks (N, S, S), CNR (N,) for `sequences`, cached per sequence as npz."""
    parts = []
    for seq in sequences:
        path = cache_dir and os.path.join(cache_dir, f"{seq}_{size}.npz")
        if path and os.path.exists(path):
            z = np.load(path)
            parts.append((z["raw"], z["mask"], z["cnr"]))
            continue
        frames = [resize_frame(*load_frame(f), size) for f in list_frames(root, seq)]
        part = tuple(np.stack(a) for a in zip(*frames))
        if path:
            os.makedirs(cache_dir, exist_ok=True)
            np.savez_compressed(path, raw=part[0], mask=part[1], cnr=part[2])
        parts.append(part)
    return tuple(np.concatenate(a) for a in zip(*parts))


def split_sequences(sequences, val_frac=0.2, seed=0):
    """Seeded sequence-level split -> (train, val), so near-duplicate frames never straddle the split."""
    seqs = sorted(sequences)
    idx = np.random.default_rng(seed).permutation(len(seqs))
    n_val = max(1, round(len(seqs) * val_frac))
    return sorted(seqs[i] for i in idx[n_val:]), sorted(seqs[i] for i in idx[:n_val])


def random_arrays(n, size=96, seed=0):
    """Synthetic stand-in (smoke tests): noisy background + elliptical 'glass' of random contrast, incl. ~0."""
    g = np.random.default_rng(seed)
    yy, xx = np.mgrid[:size, :size]
    raw, mask, cnr = np.empty((n, size, size), np.float32), np.empty((n, size, size), np.uint8), np.empty(n, np.float32)
    for k in range(n):
        (cy, cx), (ry, rx) = g.uniform(0.25, 0.75, 2) * size, g.uniform(0.08, 0.25, 2) * size
        mask[k] = ((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2 < 1
        raw[k] = g.uniform(7000, 9000) + g.normal(0, 30, (size, size)) + mask[k] * g.uniform(-400, 400)
        cnr[k] = compute_cnr(raw[k], mask[k])
    return raw, mask, cnr


def _uniform(lo, hi):
    return float(torch.empty(()).uniform_(lo, hi))


def augment(x, m, crop, scale=(0.8, 1.0), hflip=True, gain=(0.9, 1.1), offset=0.05, noise=0.01):
    """(1, S, S) normalized thermal + mask -> random scaled square crop resized to crop x crop, h-flip,
    gain/offset, Gaussian noise. No color ops; no vertical flip (tabletop scenes)."""
    S = x.shape[-1]
    side = int(round(S * _uniform(*scale)))
    i, j = (int(torch.randint(0, S - side + 1, ())) for _ in range(2))
    x = F.interpolate(x[None, :, i:i + side, j:j + side], size=(crop, crop), mode="bilinear",
                      align_corners=False, antialias=True)[0]
    m = F.interpolate(m[None, :, i:i + side, j:j + side], size=(crop, crop), mode="nearest")[0]
    if hflip and torch.rand(()) < 0.5:
        x, m = x.flip(-1), m.flip(-1)
    x = x * _uniform(*gain) + _uniform(-offset, offset) + noise * torch.randn_like(x)
    return x.clamp(0, 1), m


class ThermalSegDataset(Dataset):
    """Normalized (N, S, S) thermal + masks. Train items: augmented crop x crop; val items: full S x S
    (evaluation center-crops like DP's CropRandomizer in eval mode)."""

    def __init__(self, thermal, mask, cnr, train, crop_size=84, aug=None):
        self.thermal = torch.as_tensor(thermal, dtype=torch.float32)[:, None]
        self.mask = torch.as_tensor(mask, dtype=torch.float32)[:, None]
        self.cnr = np.asarray(cnr, np.float32)
        self.train, self.crop_size, self.aug = train, crop_size, dict(aug or {})

    def __len__(self):
        return len(self.thermal)

    def __getitem__(self, i):
        x, m = self.thermal[i], self.mask[i]
        if self.train:
            x, m = augment(x, m, self.crop_size, **self.aug)
        return {"thermal": x, "mask": m, "cnr": self.cnr[i]}
