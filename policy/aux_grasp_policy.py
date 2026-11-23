"""
Control for hindsight cue gating: the plain Diffusion Policy with the hindsight grasp point as an auxiliary target.

The same supervision as the cue (policy/hindsight_cue.py: the tool position at the first step that closes the gripper,
from the demo itself), but no cue: a small head on the encoder's features of every observation step predicts the grasp
point of the episode, and its error is added to the diffusion loss. Nothing is added to the conditioning, so at
evaluation the policy sees what the plain baseline sees.

The target reaches the loss as the observation key `hcue` of a cue folder made with
`python policy/hindsight_cue.py <demos> --out <dir> --mode label` (its train_cue.npy holds the grasp points; at
evaluation the vector is zero). The encoder must not read that key: policy.obs_encoder.gate=none and cue=false
(policy/cue_gate_encoder.py), which is the plain encoder.
"""
import torch.nn as nn
import torch.nn.functional as F

from diffusion_policy.policy.diffusion_unet_image_policy import DiffusionUnetImagePolicy

from policy.cue_gate_encoder import CUE_KEY


class AuxGraspDiffusionPolicy(DiffusionUnetImagePolicy):
    def __init__(self, *args, aux_weight=1.0, aux_hidden=256, **kwargs):
        super().__init__(*args, **kwargs)
        enc = self.obs_encoder
        assert getattr(enc, "gate", None) == "none" and not getattr(enc, "cue", True), \
            "the encoder must not read the target: policy.obs_encoder.gate=none, cue=false"
        self.aux_weight = float(aux_weight)
        self.aux_head = nn.Sequential(nn.Linear(self.obs_feature_dim, aux_hidden), nn.Mish(), nn.Linear(aux_hidden, 3))
        self.last_aux = None      # the auxiliary loss of the last batch, for logging

    def compute_loss(self, batch):
        feats = []
        hook = self.obs_encoder.register_forward_hook(lambda module, inputs, output: feats.append(output))
        try:
            loss = super().compute_loss(batch)       # the encoder runs once in there, on the observation steps
        finally:
            hook.remove()
        target = self.normalizer[CUE_KEY].normalize(batch["obs"][CUE_KEY])[:, :self.n_obs_steps, :3]
        aux = F.mse_loss(self.aux_head(feats[0]), target.reshape(-1, 3))
        self.last_aux = aux.detach()
        return loss + self.aux_weight * aux
