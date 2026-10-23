"""
Diffusion Policy dataset for the sim glass-pick demos (zarr from sim/collect_demos.py).
Only the obs keys listed in shape_meta are loaded, so one class covers RGB / RGB-D / RGB-T / RGB-D-T.
"""
import copy

import numpy as np
import torch

from diffusion_policy.common.normalize_util import get_image_range_normalizer
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import SequenceSampler, downsample_mask, get_val_mask
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.model.common.normalizer import LinearNormalizer

from policy.obs_util import to_policy_obs

class GlassImageDataset(BaseImageDataset):
    def __init__(self, zarr_path, shape_meta, horizon=1, pad_before=0, pad_after=0,
                 n_obs_steps=None, seed=42, val_ratio=0.0, max_train_episodes=None):
        super().__init__()
        self.obs_meta = dict(shape_meta["obs"])
        self.replay_buffer = ReplayBuffer.copy_from_path(zarr_path, keys=list(self.obs_meta) + ["action"])

        val_mask = get_val_mask(n_episodes=self.replay_buffer.n_episodes, val_ratio=val_ratio, seed=seed)
        train_mask = downsample_mask(mask=~val_mask, max_n=max_train_episodes, seed=seed)

        # Policy only looks at the first n_obs_steps observations, so skip loading the rest
        self.key_first_k = {k: n_obs_steps for k in self.obs_meta} if n_obs_steps is not None else dict()
        self.sampler = SequenceSampler(replay_buffer=self.replay_buffer, sequence_length=horizon,
                                       pad_before=pad_before, pad_after=pad_after,
                                       key_first_k=self.key_first_k, episode_mask=train_mask)
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(replay_buffer=self.replay_buffer, sequence_length=self.horizon,
                                          pad_before=self.pad_before, pad_after=self.pad_after,
                                          key_first_k=self.key_first_k, episode_mask=~self.train_mask)
        val_set.train_mask = ~self.train_mask
        return val_set

    def get_normalizer(self, mode="limits", **kwargs):
        data = {"action": self.replay_buffer["action"]}
        for key, attr in self.obs_meta.items():
            if attr.get("type", "low_dim") == "low_dim":
                data[key] = self.replay_buffer[key]
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        for key, attr in self.obs_meta.items():
            if attr.get("type", "low_dim") == "rgb":
                normalizer[key] = get_image_range_normalizer()
        return normalizer

    def get_all_actions(self):
        return torch.from_numpy(self.replay_buffer["action"][:])

    def __len__(self):
        return len(self.sampler)

    def __getitem__(self, idx):
        sample = self.sampler.sample_sequence(idx)
        data = {"obs": to_policy_obs(sample, self.obs_meta),
                "action": sample["action"].astype(np.float32)}
        return dict_apply(data, torch.from_numpy)
