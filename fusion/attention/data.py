"""
Sim and real glass-segmentation data in one in-memory format including training-time corruptions.

Every source becomes a SegArrays with uint8 arrays at size S x S (96 = the policy's observation size):
    rgb (N,S,S,3) · depth (N,S,S,1) (sim encoding: clip(m, 0, 2) / 2 * 255, 0 = no reading) · thermal (N,S,S,1)
    mask (N,S,S) (0..255, soft at sim object borders) · valid (N,S,S) (0 = ignore) · avail (N,3) bool (rgb, depth,
    thermal present) · meta: is_crossover, thermal_gap (sim; NaN for real)

Items are torch tensors in [0, 1], cropped to `crop` and (when train=True) corrupted.

Sources (strings, as used by train_seg.py --train / --eval):
    sim:<npz>                     a file written by collect_sim.py, e.g. sim:data/fusion/sim/L4b_train.npz
    sim:<dir>/<glob>              several, e.g. "sim:data/fusion/sim/*_train.npz"
    real:<dataset>:<split>        one dataset of the unified real data (real_data/README.md), e.g. real:rgbt_glass:test

Real data is resized once with the real_data loader (resize="stretch", like the sim) and cached as npz.
"""
import glob
import math
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, Dataset

MODS = ("rgb", "depth", "thermal")
REAL_DATASETS = ("rgbt_glass", "transpose", "mju_glass", "rgbd_gsd", "trobi")


def minmax_u8(raw):
    """Per-frame min-max of (N, H, W, 1) raw thermal counts -> uint8."""
    raw = raw.astype(np.float32)
    lo = raw.min(axis=(1, 2, 3), keepdims=True)
    hi = raw.max(axis=(1, 2, 3), keepdims=True)
    return np.round(255 * (raw - lo) / np.maximum(hi - lo, 1e-6)).astype(np.uint8)


class SegArrays(Dataset):
    def __init__(self, arrays, name, crop=84, train=False, aug=None, modalities=MODS):
        self.a, self.name, self.crop, self.train = arrays, name, crop, train
        self.aug = Augment(**(aug or {})) if train else None
        self.mod_idx = [MODS.index(m) for m in modalities]
        self.modalities = list(modalities)
        n = len(arrays["rgb"])
        assert all(len(v) == n for v in arrays.values() if isinstance(v, np.ndarray) and v.ndim > 0), name

    def __len__(self):
        return len(self.a["rgb"])

    def __getitem__(self, i):
        a = self.a
        rgb = torch.from_numpy(a["rgb"][i]).permute(2, 0, 1).float() / 255
        depth = torch.from_numpy(a["depth"][i]).permute(2, 0, 1).float() / 255
        thermal = torch.from_numpy(a["thermal"][i]).permute(2, 0, 1).float() / 255
        mask = torch.from_numpy(a["mask"][i])[None].float() / 255
        valid = torch.from_numpy(a["valid"][i])[None].float() if "valid" in a else torch.ones_like(mask)
        avail = torch.from_numpy(a["avail"][i].copy())
        x = torch.cat([rgb, depth, thermal, mask, valid], 0)          # (7, S, S): crop / flip all together
        S, c = x.shape[-1], self.crop
        if self.train:
            top, left = np.random.randint(0, S - c + 1, 2)
            x = x[:, top:top + c, left:left + c]
            if np.random.rand() < 0.5:
                x = x.flip(-1)
        else:
            o = (S - c) // 2
            x = x[:, o:o + c, o:o + c]
        item = dict(rgb=x[0:3], depth=x[3:4], thermal=x[4:5], mask=x[5:6], valid=x[6:7], avail=avail,
                    corrupt=torch.zeros(3, dtype=torch.bool), unreliable=torch.zeros(3, dtype=torch.bool),
                    is_crossover=torch.tensor(float(a["is_crossover"][i]) if "is_crossover" in a else float("nan")),
                    thermal_gap=torch.tensor(float(a["thermal_gap"][i]) if "thermal_gap" in a else float("nan")))
        if item["is_crossover"] == 1:   # sim frame in thermal crossover: the glass is not in the thermal image
            item["unreliable"][2] = True
        if self.aug is not None:
            self.aug(item)
        # gate labels: unreliable sensors, and untouched ones (reliable); partly corrupted ones are left out
        item["labelled"] = item["unreliable"] | ~item["corrupt"]
        for k in ("avail", "corrupt", "unreliable", "labelled"):
            item[k] = item[k][self.mod_idx]
        return item


# ---------------------------------------------------------------------- corruptions
def _blobs(shape, frac, sigma):
    """Bool mask of connected blobs covering about `frac` of the image (blurred noise thresholded at a quantile),
    the same defect model as the hard-scenario sim's depth holes."""
    n = torch.randn(1, 1, *shape)
    r = max(1, int(3 * sigma))
    k = torch.exp(-0.5 * (torch.arange(-r, r + 1) / sigma) ** 2)
    k = k / k.sum()
    n = F.conv2d(F.pad(n, (r, r, 0, 0), mode="reflect"), k.view(1, 1, 1, -1))
    n = F.conv2d(F.pad(n, (0, 0, r, r), mode="reflect"), k.view(1, 1, -1, 1))[0, 0]
    return n < torch.quantile(n.flatten(), frac)


class Augment:
    """Sensor corruptions applied only in training. The modality stays
    'available' (avail = True), leaving the network has to notice from the image itself that it is unreliable (what gates are used for). 
    A modality the dataset does not have at all is avail = False.

    thermal_crossover  contrast scaled by U(0, 0.3) + noise, then per-frame min-max again (amplified noise)
    depth_holes        connected missing blobs over 5-50% of the image; depth_wrong: blobs with a smooth offset
    rgb_lowlight       darker + sensor noise; blank: the modality is all zeros (dead sensor / lens covered)
    drop               modality dropout: the modality is marked missing (avail = False), so every modality has to be
                       useful on its own and the fusion cannot lean on one of them only
    """

    def __init__(self, thermal_crossover=0.25, depth_holes=0.3, depth_wrong=0.15, rgb_lowlight=0.25, blank=0.05,
                 blank_rgb=0.02, drop=0.15):
        self.p = dict(thermal_crossover=thermal_crossover, depth_holes=depth_holes, depth_wrong=depth_wrong,
                      rgb_lowlight=rgb_lowlight, blank=blank, blank_rgb=blank_rgb, drop=drop)

    def __call__(self, it):
        p, r = self.p, np.random.rand
        av = it["avail"]
        if av[2] and r() < p["thermal_crossover"]:
            t = it["thermal"]
            t = t.mean() + np.random.uniform(0, 0.3) * (t - t.mean()) + np.random.uniform(0.005, 0.03) * torch.randn_like(t)
            it["thermal"] = (t - t.min()) / (t.max() - t.min()).clamp(min=1e-6)
            it["corrupt"][2] = it["unreliable"][2] = True
        if av[1] and r() < p["depth_holes"]:
            d = it["depth"]
            d[0][_blobs(d.shape[-2:], np.random.uniform(0.05, 0.5), np.random.uniform(1.5, 5))] = 0
            it["corrupt"][1] = True
        if av[1] and r() < p["depth_wrong"]:
            d = it["depth"]
            m = _blobs(d.shape[-2:], np.random.uniform(0.05, 0.3), np.random.uniform(2, 6))
            d[0][m & (d[0] > 0)] = (d[0] + np.random.uniform(-0.15, 0.15)).clamp(0.01, 1)[m & (d[0] > 0)]
            it["corrupt"][1] = True
        if r() < p["rgb_lowlight"]:
            it["rgb"] = (it["rgb"] * np.random.uniform(0.25, 0.8)
                         + np.random.uniform(0.01, 0.06) * torch.randn_like(it["rgb"])).clamp(0, 1)
            it["corrupt"][0] = True
        # blank at most all-but-one of the available modalities
        for k, name in ((1, "depth"), (2, "thermal"), (0, "rgb")):
            others = [j for j in range(3) if j != k and av[j]]
            pk = p["blank_rgb"] if k == 0 else p["blank"]
            if av[k] and others and r() < pk and not all(it[MODS[j]].abs().sum() == 0 for j in others):
                it[name] = torch.zeros_like(it[name])
                it["corrupt"][k] = it["unreliable"][k] = True
        for k in np.random.permutation(3):
            keep = [j for j in range(3) if j != k and av[j] and it[MODS[j]].abs().sum() > 0]
            if av[k] and keep and r() < p["drop"]:
                av[k] = False
                it[MODS[k]] = torch.zeros_like(it[MODS[k]])
        return it


# ---------------------------------------------------------------------- loading
def load_sim(path, size=96, sim_thermal="minmax"):
    d = dict(np.load(path))
    n = len(d["rgb"])
    assert d["rgb"].shape[1] == size, f"{path}: frames are {d['rgb'].shape[1]}px, asked for {size}"
    thermal = d["thermal"]
    if sim_thermal == "minmax" and "thermal_raw" in d:   
        thermal = minmax_u8(d["thermal_raw"])
    return dict(rgb=d["rgb"], depth=d["depth"], thermal=thermal, mask=d["mask"],
                avail=np.ones((n, 3), bool), is_crossover=d["is_crossover"].astype(np.float32),
                thermal_gap=d["thermal_gap"].astype(np.float32), object=d["object"], seed=d["seed"])


def load_real(dataset, split, size=96, cache_dir="data/fusion/real", root=None, workers=8):
    """One dataset at size x size (stretched like sim), cached as npz after the first call."""
    cache = Path(cache_dir) / f"{dataset}_{split}_{size}.npz"
    if cache.exists():
        return dict(np.load(cache))
    from real_data.dataset import TransparentPerceptionDataset
    from real_data.common import DEFAULT_ROOT
    ds = TransparentPerceptionDataset(root=root or DEFAULT_ROOT, datasets=[dataset], splits=[split],
                                      require=["mask"], size=size, resize="stretch", encoding="sim")
    dl = torch.utils.data.DataLoader(ds, batch_size=64, num_workers=workers)
    out = {k: [] for k in ["rgb", "depth", "thermal", "mask", "valid", "avail"]}
    for b in dl:
        u8 = lambda t: (t.clamp(0, 1) * 255).round().to(torch.uint8).permute(0, 2, 3, 1).numpy()
        out["rgb"].append(u8(b["rgb"]))
        has_d = (b["has_depth"] & b["depth_is_metric"]).numpy()          
        out["depth"].append(u8(b["depth"]) * has_d[:, None, None, None])
        out["thermal"].append(u8(b["thermal"]))
        m = b["mask"].numpy()
        out["mask"].append(((m == 1) * 255).astype(np.uint8))
        out["valid"].append((m != 255).astype(np.uint8))
        out["avail"].append(np.stack([np.ones_like(has_d), has_d, b["has_thermal"].numpy()], 1).astype(bool))
    data = {k: np.concatenate(v) for k, v in out.items()}
    n = len(data["rgb"])
    data["is_crossover"] = np.full(n, np.nan, np.float32)
    data["thermal_gap"] = np.full(n, np.nan, np.float32)
    cache.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache.with_name(f"{cache.stem}.{os.getpid()}.tmp.npz")   
    np.savez(tmp, **data)
    os.replace(tmp, cache)
    print(f"[fusion] cached {n} {dataset}/{split} samples -> {cache}", flush=True)
    return data


def expand_sources(specs):
    """'sim:dir/*_train.npz' -> one 'sim:<file>' per match."""
    out = []
    for s in specs:
        kind, rest = s.split(":", 1)
        if kind == "sim":
            files = sorted(glob.glob(rest))
            assert files, f"no sim files match {rest}"
            out += [f"sim:{f}" for f in files]
        elif kind == "real":
            out.append(s)
        else:
            raise ValueError(f"source must start with sim: or real:, got {s}")
    return out


def source_name(spec):
    kind, rest = spec.split(":", 1)
    return Path(rest).stem if kind == "sim" else "real_" + rest.replace(":", "_")


def load_source(spec, size=96, sim_thermal="minmax", real_root=None, cache_dir="data/fusion/real"):
    kind, rest = spec.split(":", 1)
    if kind == "sim":
        return load_sim(rest, size, sim_thermal)
    dataset, split = rest.split(":")
    return load_real(dataset, split, size, cache_dir, real_root)


def real_available(root=None):
    from real_data.common import DEFAULT_ROOT
    return (Path(root or DEFAULT_ROOT) / "manifest.parquet").exists()


def build_train_set(specs, modalities, size=96, crop=84, aug=None, sim_weight=0.5, **kw):
    """ConcatDataset over every source plus per-sample sampling weights."""
    parts, domain = [], []
    for spec in expand_sources(specs):
        parts.append(SegArrays(load_source(spec, size, **kw), source_name(spec), crop, True, aug, modalities))
        domain.append(spec.split(":", 1)[0])
    n_sim, n_real = domain.count("sim"), domain.count("real")
    share = {"sim": sim_weight if n_real else 1.0, "real": (1 - sim_weight) if n_sim else 1.0}
    w = np.concatenate([np.full(len(p), share[d] / (domain.count(d) * len(p))) for p, d in zip(parts, domain)])
    return ConcatDataset(parts), torch.as_tensor(w, dtype=torch.double)


def collate(batch):
    return {k: torch.stack([b[k] for b in batch]) for k in batch[0]}