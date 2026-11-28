"""
Self-attention fusion of the sensors, as a baseline from the literature: the fusion of "See, Hear, and Feel: Smart
Sensory Fusion for Robotic Manipulation" (Li et al., CoRL 2022; MulSA), put in front of Diffusion Policy.

    rgb ───► ResNet ──► z_rgb(t) ──┐
  depth ───► ResNet ──► z_d(t) ────┼──► tokens of all sensors and observation steps ──► multi-head self-attention
thermal ───► ResNet ──► z_t(t) ────┘        ──► the attended tokens, concatenated ──► cond = [tokens ; joints] ──► FiLM

As in the paper, the tokens are the feature vectors of every sensor at every observation step (here 3 sensors x 2
steps), one standard multi-head self-attention attends across sensors and across time, and the output tokens are
concatenated. The paper feeds them to an MLP that classifies a discrete action; here they condition the denoiser of
Diffusion Policy, like the concatenated features of the plain baselines, so the fusion is the only thing that differs
from those. What the paper leaves open is set here and kept simple: one attention layer, 8 heads, LayerNorm on the
tokens before it, a residual connection around it, no positional or modality embedding (the concatenation keeps the
order of the tokens). The image encoders, crops and ImageNet start are those of the plain baselines.

Diffusion Policy calls the encoder on the observation steps of a batch flattened to (B x To, ...), a sample's steps
next to each other; n_obs_steps tells the encoder how to group them back. After each forward: last_attn
(B, tokens, tokens), averaged over the heads, for logging.
"""
import torch
import torch.nn as nn

from diffusion_policy.model.vision.multi_image_obs_encoder import MultiImageObsEncoder


class SelfAttnFusionObsEncoder(MultiImageObsEncoder):
    def __init__(self, shape_meta, rgb_model, resize_shape=None, crop_shape=None, random_crop=True,
                 use_group_norm=False, share_rgb_model=False, imagenet_norm=False, n_obs_steps=2, num_heads=8):
        assert not share_rgb_model, "one encoder per sensor, as the baselines"
        super().__init__(shape_meta, rgb_model, resize_shape=resize_shape, crop_shape=crop_shape,
                         random_crop=random_crop, use_group_norm=use_group_norm, share_rgb_model=False,
                         imagenet_norm=imagenet_norm)
        dims = {self._feature_dim(k) for k in self.rgb_keys}
        assert len(dims) == 1, f"the sensors' features must have one size to be tokens: {dims}"
        (d,) = dims
        self.n_obs_steps = int(n_obs_steps)
        self.norm = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, num_heads, batch_first=True)
        self.last_attn = None

    @torch.no_grad()
    def _feature_dim(self, key):
        c, h, w = self.key_shape_map[key]
        model = self.key_model_map[key]
        was = model.training
        model.eval()
        d = model(self.key_transform_map[key](torch.zeros(1, c, h, w))).shape[-1]
        model.train(was)
        return d

    def forward(self, obs_dict):
        feats = []
        for key in self.rgb_keys:
            img = obs_dict[key]
            assert img.shape[1:] == self.key_shape_map[key], (key, img.shape)
            feats.append(self.key_model_map[key](self.key_transform_map[key](img)))
        tok = torch.stack(feats, 1)                                  # (N, sensors, D), N = B x To
        n, m, d = tok.shape
        t = self.n_obs_steps if n % self.n_obs_steps == 0 else 1     # the shape probe of the parent has N = 1
        x = tok.reshape(n // t, t * m, d)                            # a sample's tokens: step 0's sensors, step 1's, ...
        out, w = self.attn(self.norm(x), self.norm(x), self.norm(x), need_weights=True)
        self.last_attn = w.detach()
        features = [(x + out).reshape(n, m * d)]
        for key in self.low_dim_keys:
            data = obs_dict[key]
            assert data.shape[1:] == self.key_shape_map[key], (key, data.shape)
            features.append(data)
        return torch.cat(features, dim=-1)
