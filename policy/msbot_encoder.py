"""
Stage-guided dynamic fusion of the sensors, as a baseline from the literature: the fusion of "Play to the Score:
Stage-Guided Dynamic Multi-Sensory Fusion for Robotic Manipulation" (Feng et al., CoRL 2024; MS-Bot), put in front of
Diffusion Policy.

    rgb / depth / thermal ──► ResNet each ──► feature tokens f (sensors x observation steps) ──────────────┐ key, value
    [f ; joints] ──► MLP ──► state token z_state ──► gate MLP ──► scores g over the stages                   │
    z_stage = (1 / S) sum_j g_j [stage_j]      z* = beta z_state + (1 - beta) z_stage ──► query ──► cross-attention
                                                                         fused token ──► cond = [fused ; joints] ──► FiLM

The parts are the paper's: a state token from the current features, learnable stage tokens (set to the mean state
token of their stage after a warm-up), a gate that scores the stages with the soft constraint L_gate = sum_j w_j g_j
(w_j = 0 for the frame's own stage and for every stage near a stage change, 1 otherwise), the stage-injected state
token as the query of a cross-attention over the sensors' feature tokens, random attention blur (the scores replaced
by uniform ones with probability p in training), loss = policy loss + lambda L_gate; beta 0.5, p 0.25, lambda 5.

What differs from the paper, because the benchmark fixes it for every policy or the setup lacks it:
  * the action head is the denoiser of Diffusion Policy conditioned on [fused token ; joints], not an MLP on discrete
    actions (the fusion is what is compared);
  * the state tokenizer reads the joint values of the observation steps where the paper runs an LSTM over the action
    history (the policies of the benchmark observe two steps and no past actions);
  * stages come from a rule on the demonstrated actions (stage_labels: reach, grasp, lift), as the paper's rule-based
    labels do; frames within gamma = 1 step of a stage change are not constrained (the paper's episodes are longer);
  * the paper does not give the number of heads or the sizes of the MLPs: 8 heads, state MLP 1024, gate MLP 256.

The stage of a training frame reaches the loss as the observation key `stage` (label, near a stage change), written
by the dataset from the actions; the encoder never reads it, and at evaluation it is zero.
After each forward: last_gate (B, S), last_scores (B, tokens; mean over the heads) for logging.
"""
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusion_policy.model.vision.multi_image_obs_encoder import MultiImageObsEncoder
from diffusion_policy.policy.diffusion_unet_image_policy import DiffusionUnetImagePolicy

STAGE_KEY = "stage"
STAGES = ("reach", "grasp", "lift")
JOINTS_KEY = "joints"


def stage_labels(action, ends, gamma=1):
    """action (frames, 7) of the demos, ends (episodes,) -> (frames, 2) float32: the stage of every frame and whether
    a stage change is within gamma steps. reach: the gripper is commanded open; grasp: commanded closed and the arm's
    target has not moved since; lift: from the first step after the close whose arm target differs from the one before."""
    action, out = np.asarray(action), np.zeros((len(action), 2), np.float32)
    for s, t in zip(np.r_[0, ends[:-1]], ends):
        a, st = action[s:t], np.zeros(t - s, int)
        closed = a[:, 6] > 0.4
        if closed.any():
            i = int(np.argmax(closed))
            st[i:] = 1
            moved = np.flatnonzero(np.linalg.norm(np.diff(a[i:, :6], axis=0), axis=1) > 1e-4)
            if len(moved):
                st[i + moved[0] + 1:] = 2
        near = np.zeros(t - s, bool)
        for k in range(1, gamma + 1):
            change = st[k:] != st[:-k]
            near[k:] |= change
            near[:-k] |= change
        out[s:t, 0], out[s:t, 1] = st, near
    return out


class StageFusionObsEncoder(MultiImageObsEncoder):
    def __init__(self, shape_meta, rgb_model, resize_shape=None, crop_shape=None, random_crop=True,
                 use_group_norm=False, share_rgb_model=False, imagenet_norm=False, n_obs_steps=2, num_heads=8,
                 beta=0.5, blur_p=0.25, warmup_steps=574, state_hidden=1024, gate_hidden=256):
        assert not share_rgb_model, "one encoder per sensor, as the baselines"
        super().__init__(shape_meta, rgb_model, resize_shape=resize_shape, crop_shape=crop_shape,
                         random_crop=random_crop, use_group_norm=use_group_norm, share_rgb_model=False,
                         imagenet_norm=imagenet_norm)
        assert STAGE_KEY in self.low_dim_keys and JOINTS_KEY in self.low_dim_keys, self.low_dim_keys
        dims = {self._feature_dim(k) for k in self.rgb_keys}
        assert len(dims) == 1, f"the sensors' features must have one size to be tokens: {dims}"
        (d,) = dims
        assert d % num_heads == 0
        self.d, self.heads, self.t = d, int(num_heads), int(n_obs_steps)
        self.beta, self.blur_p, self.warmup_steps = float(beta), float(blur_p), int(warmup_steps)
        self.jd = self.key_shape_map[JOINTS_KEY][0]
        n_tok, s = self.t * len(self.rgb_keys), len(STAGES)
        self.state = nn.Sequential(nn.Linear(n_tok * d + self.t * self.jd, state_hidden), nn.Mish(),
                                   nn.Linear(state_hidden, d))
        self.gate = nn.Sequential(nn.Linear(d, gate_hidden), nn.Mish(), nn.Linear(gate_hidden, s))
        self.stage_tokens = nn.Parameter(torch.zeros(s, d))
        self.q, self.k, self.v, self.o = (nn.Linear(d, d) for _ in range(4))
        self.register_buffer("steps", torch.zeros((), dtype=torch.long))      # training forwards so far (warm-up)
        self.register_buffer("warm_sum", torch.zeros(s, d))
        self.register_buffer("warm_n", torch.zeros(s))
        self.labels = None            # (B,) stage of the batch's frames, set by the policy for a training forward
        self.last_gate = self.last_scores = None

    @torch.no_grad()
    def _feature_dim(self, key):
        c, h, w = self.key_shape_map[key]
        model = self.key_model_map[key]
        was = model.training
        model.eval()
        d = model(self.key_transform_map[key](torch.zeros(1, c, h, w))).shape[-1]
        model.train(was)
        return d

    def output_shape(self):
        return (self.d + self.jd,)

    def forward(self, obs_dict):
        feats = []
        for key in self.rgb_keys:
            img = obs_dict[key]
            assert img.shape[1:] == self.key_shape_map[key], (key, img.shape)
            feats.append(self.key_model_map[key](self.key_transform_map[key](img)))
        tok = torch.stack(feats, 1)                                  # (N, sensors, D), N = B x To
        n, m, d = tok.shape
        assert n % self.t == 0, (n, self.t)
        b = n // self.t
        x = tok.reshape(b, self.t * m, d)                            # a sample's tokens: step 0's sensors, step 1's, ...
        joints = obs_dict[JOINTS_KEY]
        z_state = self.state(torch.cat([x.flatten(1), joints.reshape(b, -1)], 1))
        g = F.softmax(self.gate(z_state), -1)                        # (B, S)
        warm = self.training and int(self.steps) < self.warmup_steps
        z = z_state if warm else self.beta * z_state + (1 - self.beta) * (g @ self.stage_tokens) / g.shape[1]

        h, dh = self.heads, d // self.heads
        q = self.q(z).view(b, h, 1, dh)
        k = self.k(x).view(b, -1, h, dh).transpose(1, 2)             # (B, H, tokens, dh)
        v = self.v(x).view(b, -1, h, dh).transpose(1, 2)
        scores = F.softmax(q @ k.transpose(-1, -2) / math.sqrt(dh), -1)            # (B, H, 1, tokens)
        if self.training and self.blur_p > 0:                        # random attention blur
            blur = torch.rand(b, device=x.device) < self.blur_p
            scores = torch.where(blur[:, None, None, None], torch.full_like(scores, 1.0 / scores.shape[-1]), scores)
        fused = self.o((scores @ v).transpose(1, 2).reshape(b, d))   # (B, D)

        if self.training and torch.is_grad_enabled() and self.labels is not None:
            with torch.no_grad():
                if warm:                                             # the mean state token of every stage
                    one = F.one_hot(self.labels, g.shape[1]).to(z_state.dtype)
                    self.warm_sum += one.t() @ z_state.detach()
                    self.warm_n += one.sum(0)
                self.steps += 1
                if int(self.steps) == self.warmup_steps:
                    seen = self.warm_n > 0
                    self.stage_tokens.data[seen] = (self.warm_sum[seen] / self.warm_n[seen, None]).to(self.stage_tokens.dtype)
        self.last_gate, self.last_scores = g, scores.mean(1)[:, 0].detach()
        return torch.cat([fused.repeat_interleave(self.t, 0), joints], dim=-1)


class StageFusionDiffusionPolicy(DiffusionUnetImagePolicy):
    def __init__(self, *args, gate_weight=5.0, **kwargs):
        super().__init__(*args, **kwargs)
        assert isinstance(self.obs_encoder, StageFusionObsEncoder)
        self.gate_weight = float(gate_weight)
        self.last_gate_loss = None    # of the last batch, for logging

    def compute_loss(self, batch):
        stage = batch["obs"][STAGE_KEY][:, self.n_obs_steps - 1]      # (B, 2): the current frame's stage, near a change
        label, near = stage[:, 0].long(), stage[:, 1] > 0.5
        enc = self.obs_encoder
        enc.labels = label
        try:
            loss = super().compute_loss(batch)        # the encoder runs once in there, on the observation steps
        finally:
            enc.labels = None
        g = enc.last_gate
        w = (1 - F.one_hot(label, g.shape[1]).to(g.dtype)) * (~near)[:, None]
        gate = (w * g).sum(1).mean()
        self.last_gate_loss = gate.detach()
        return loss + self.gate_weight * gate
