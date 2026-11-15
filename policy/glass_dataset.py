"""
Diffusion Policy dataset for the sim glass-pick demos (zarr from sim/collect_demos.py).
Only the obs keys listed in shape_meta are loaded, so one class covers RGB / RGB-D / RGB-T / RGB-D-T.

Additions for the fusion experiments (all off unless the task config asks for them):
* `cue` obs key: reliability cues computed from the raw frames (policy/obs_util.py). Needs data/thermal_raw in the
  zarr for the thermal cue; older demos fall back to the 8-bit thermal image with a warning.
* `meta`: per-episode scene conditions (meta/ep_* written by collect_demos.py: thermal gap, crossover flag, ...),
  returned next to obs / action. Training-time supervision and analysis only, never a policy input.
* `augment`: online sensor corruption on training samples (Corruption below): thermal contrast scaled toward flat,
  structured depth holes. Applied to the raw frames before the cues are computed, so a cue always describes the
  frame the encoder actually sees.
"""
import copy

import numpy as np
import torch
import zarr

from diffusion_policy.common.normalize_util import get_image_range_normalizer
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import SequenceSampler, downsample_mask, get_val_mask
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer

from policy.obs_util import compute_cues, cue_roi, to_policy_obs

META_KEYS = ("ep_thermal_gap", "ep_is_crossover", "ep_t_env", "ep_obj_temp", "ep_level", "ep_liquid_fill")


def zarr_data_keys(zarr_path):
    """Names of the arrays under data/ (directory store or the single-file .zip store of the releases)."""
    if str(zarr_path).endswith(".zip"):
        ZipStore = getattr(zarr, "ZipStore", None) or zarr.storage.ZipStore
        store = ZipStore(zarr_path, mode="r")
    else:
        store = zarr_path
    return set(zarr.open(store, mode="r")["data"].keys())


class Corruption:
    """Online training corruption of the raw frames of one sample (all its time steps get the same draw).
    thermal: x <- median + s (x - median), s ~ U(thermal_contrast), with probability thermal_contrast_p. Applied to the
             8-bit image and to thermal_raw alike, so the contrast cue drops with the image.
    depth:   connected blobs of missing readings covering a fraction ~ U(depth_hole_frac) of the frame (blurred noise
             thresholded), with probability depth_hole_p. The validity channel and cue follow since both read depth > 0.
    """

    def __init__(self, thermal_contrast_p=0.5, thermal_contrast=(0.0, 0.6), depth_hole_p=0.5,
                 depth_hole_frac=(0.1, 0.6), depth_blob_cells=(6, 16)):
        self.thermal_contrast_p, self.thermal_contrast = thermal_contrast_p, tuple(thermal_contrast)
        self.depth_hole_p, self.depth_hole_frac = depth_hole_p, tuple(depth_hole_frac)
        self.depth_blob_cells = tuple(depth_blob_cells)

    def __call__(self, sample, rng):
        out = dict(sample)
        if "thermal" in out and rng.random() < self.thermal_contrast_p:
            s = rng.uniform(*self.thermal_contrast)
            for key in ("thermal", "thermal_raw"):
                if key in out:
                    x = out[key].astype(np.float32)
                    med = np.median(x, axis=(-3, -2), keepdims=True)
                    info = np.iinfo(out[key].dtype)
                    out[key] = np.clip(np.round(med + s * (x - med)), info.min, info.max).astype(out[key].dtype)
        if "depth" in out and rng.random() < self.depth_hole_p:
            d = out["depth"]
            H, W = d.shape[-3], d.shape[-2]
            cells = int(rng.integers(self.depth_blob_cells[0], self.depth_blob_cells[1] + 1))
            noise = rng.random((cells, cells))
            field = np.kron(noise, np.ones((H // cells + 1, W // cells + 1)))[:H, :W]   # blocky low-frequency field
            field = _box_blur(field, 3)
            frac = rng.uniform(*self.depth_hole_frac)
            hole = field < np.quantile(field, frac)
            d = d.copy()
            d[:, hole] = 0
            out["depth"] = d
        return out


def _box_blur(x, r):
    p = np.pad(x, r, mode="edge")
    c = np.cumsum(np.cumsum(p, 0), 1)
    c = np.pad(c, ((1, 0), (1, 0)))
    k = 2 * r + 1
    return (c[k:, k:] - c[:-k, k:] - c[k:, :-k] + c[:-k, :-k]) / k ** 2


class GlassImageDataset(BaseImageDataset):
    def __init__(self, zarr_path, shape_meta, horizon=1, pad_before=0, pad_after=0,
                 n_obs_steps=None, seed=42, val_ratio=0.0, max_train_episodes=None,
                 augment=None, with_meta=True, n_cue_fit_frames=2000):
        super().__init__()
        self.obs_meta = dict(shape_meta["obs"])
        self.has_cue = "cue" in self.obs_meta
        self.roi = cue_roi(self.obs_meta)
        data_keys = [k for k in self.obs_meta if k != "cue"]
        available = zarr_data_keys(zarr_path)
        if self.has_cue and "thermal" in self.obs_meta and "thermal_raw" in available:
            data_keys.append("thermal_raw")
        self.replay_buffer = ReplayBuffer.copy_from_path(zarr_path, keys=data_keys + ["action"])

        val_mask = get_val_mask(n_episodes=self.replay_buffer.n_episodes, val_ratio=val_ratio, seed=seed)
        train_mask = downsample_mask(mask=~val_mask, max_n=max_train_episodes, seed=seed)

        # Policy only looks at the first n_obs_steps observations, so skip loading the rest
        self.key_first_k = {k: n_obs_steps for k in data_keys} if n_obs_steps is not None else dict()
        self.sampler = SequenceSampler(replay_buffer=self.replay_buffer, sequence_length=horizon,
                                       pad_before=pad_before, pad_after=pad_after,
                                       key_first_k=self.key_first_k, episode_mask=train_mask)
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.episode_ends = np.asarray(self.replay_buffer.episode_ends[:])

        meta = self.replay_buffer.meta
        self.ep_meta = {k: np.asarray(meta[k][:], dtype=np.float32) for k in META_KEYS if with_meta and k in meta}
        self.augment = Corruption(**augment) if augment else None
        self.seed = seed
        self.n_cue_fit_frames = n_cue_fit_frames
        self._rng = None

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(replay_buffer=self.replay_buffer, sequence_length=self.horizon,
                                          pad_before=self.pad_before, pad_after=self.pad_after,
                                          key_first_k=self.key_first_k, episode_mask=~self.train_mask)
        val_set.train_mask = ~self.train_mask
        val_set.augment = None          # validation sees clean frames
        val_set._rng = None
        return val_set

    def get_normalizer(self, mode="limits", **kwargs):
        data = {"action": self.replay_buffer["action"]}
        for key, attr in self.obs_meta.items():
            if key != "cue" and attr.get("type", "low_dim") == "low_dim":
                data[key] = self.replay_buffer[key]
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        for key, attr in self.obs_meta.items():
            if attr.get("type", "low_dim") == "rgb":
                normalizer[key] = get_image_range_normalizer()
        if self.has_cue:
            # z-scores rather than limits: cues of corrupted / crossover frames must stay in range, and the fit
            # sample includes corrupted copies of the frames so the training distribution is covered
            normalizer["cue"] = SingleFieldLinearNormalizer.create_fit(self._cue_sample(), last_n_dims=1,
                                                                        mode="gaussian")
        return normalizer

    def _cue_sample(self):
        """Cues of a random subset of frames, clean and (if augment is on) corrupted, to fit the cue normalizer."""
        n = self.replay_buffer.n_steps
        rng = np.random.default_rng(self.seed)
        idx = np.sort(rng.choice(n, size=min(n, self.n_cue_fit_frames), replace=False))
        raw = {k: self.replay_buffer[k][idx] for k in ("rgb", "depth", "thermal", "thermal_raw")
               if k in self.replay_buffer.keys()}
        cues = [compute_cues(raw, self.roi)]
        if self.augment is not None:
            step = 50   # corrupt in small groups so the draws vary
            parts = []
            for i in range(0, len(idx), step):
                parts.append(compute_cues(self.augment({k: v[i:i + step] for k, v in raw.items()}, rng), self.roi))
            cues.append(np.concatenate(parts))
        return np.concatenate(cues)

    def get_all_actions(self):
        return torch.from_numpy(self.replay_buffer["action"][:])

    def __len__(self):
        return len(self.sampler)

    def _episode_of(self, idx):
        buffer_start = self.sampler.indices[idx][0]
        return int(np.searchsorted(self.episode_ends, buffer_start, side="right"))

    def __getitem__(self, idx):
        sample = self.sampler.sample_sequence(idx)
        if self.augment is not None:
            if self._rng is None:
                info = torch.utils.data.get_worker_info()
                self._rng = np.random.default_rng([self.seed, info.id if info else 0, int(torch.initial_seed()) % 2**31])
            sample = self.augment(sample, self._rng)
        data = {"obs": to_policy_obs(sample, self.obs_meta),
                "action": sample["action"].astype(np.float32)}
        if self.ep_meta:
            ep = self._episode_of(idx)
            data["meta"] = {k: np.asarray(v[ep], dtype=np.float32) for k, v in self.ep_meta.items()}
        return dict_apply(data, torch.from_numpy)
