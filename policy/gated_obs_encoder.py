"""
Cue-driven reliability gate for Diffusion Policy's observation encoder (the "A" fusion of the midterm report).

    rgb ───► ResNet ──► z_rgb ──┐
  depth ───► ResNet ──► z_d ────┼──► w_m ⊙ z_m ──► cond = [w_rgb z_rgb ; w_d z_d ; w_t z_t ; w ; joints] ──► FiLM
thermal ───► ResNet ──► z_t ────┘        ▲
    raw frames ──► cues c (policy/obs_util.py) ──► w_m = σ(MLP_m(c[, z_m]))     one scalar per modality and frame

Compared with the static baseline (SharedCropObsEncoder: plain concat), the only changes are the gate weights w_m,
the modality-specific channels kept (gated concat, not a weighted sum), w itself in the conditioning so the policy knows
which sensor it is trusting, and a learned "missing" embedding for a sensor that is dropped (w = 0).

Options (all in the config under policy.obs_encoder):
  gate_input       cue | feature | both      what the gate MLP sees (cue = physical cues only, our v1)
  cross_modal_cue  the gate of one modality also sees the other modalities' cues (cheap; "depth is holey and thermal
                   is flat -> lean on rgb")
  gate_act         sigmoid (independent weights) | softmax (weights compete, sum to 1 over modalities)
  w_in_cond        append w to the conditioning
  cue_in_cond      also append the cue vector itself (the static-with-cue baseline has this and no gate)
  missing          learned (e_miss) | zero      feature of a dropped modality
  dropout_mode     none | random | targeted      training-time modality dropout; targeted drops the modality with the
                   highest gate weight in each sample with p, the others with p_other (Targeted Modality Dropout)
  rgb_gate_bias    initial logit bias of the rgb gate, so training starts from "trust rgb"
After each forward the encoder keeps last_w (B, M), last_z {m: (B, D)} and last_drop (B, M) for the policy's
auxiliary losses and for logging; force_drop = {"thermal"} switches a sensor off at inference (eval.py --drop).
"""
import torch
import torch.nn as nn

from policy.obs_util import CUES_PER_MODALITY, CUE_DIM, MODALITIES
from policy.shared_crop_encoder import SharedCropObsEncoder


class GatedObsEncoder(SharedCropObsEncoder):
    def __init__(self, shape_meta, rgb_model, resize_shape=None, crop_shape=None, random_crop=True,
                 use_group_norm=False, share_rgb_model=False, imagenet_norm=False, init_weights=None,
                 cue_key="cue", gate_input="cue", cross_modal_cue=True, gate_hidden=32, gate_act="sigmoid",
                 w_in_cond=True, cue_in_cond=False, missing="learned", dropout_mode="targeted", dropout_p=0.3,
                 dropout_p_other=0.1, rgb_gate_bias=2.0):
        assert not share_rgb_model, "the gate needs one encoder per modality"
        assert gate_input in ("cue", "feature", "both") and gate_act in ("sigmoid", "softmax")
        assert missing in ("learned", "zero") and dropout_mode in ("none", "random", "targeted")
        super().__init__(shape_meta, rgb_model, resize_shape=resize_shape, crop_shape=crop_shape,
                         random_crop=random_crop, use_group_norm=use_group_norm, share_rgb_model=False,
                         imagenet_norm=imagenet_norm, init_weights=init_weights)
        assert cue_key in self.low_dim_keys, f"shape_meta needs a '{cue_key}' low_dim observation (task glass_pick_cue)"
        assert tuple(self.key_shape_map[cue_key]) == (CUE_DIM,), "cue shape must match obs_util.CUE_DIM"
        self.cue_key = cue_key
        self.mods = [m for m in MODALITIES if m in self.rgb_keys]          # gated, in the cue's fixed order
        self.other_rgb = [k for k in self.rgb_keys if k not in self.mods]  # any other camera: concatenated ungated
        self.state_keys = [k for k in self.low_dim_keys if k != cue_key]
        self.gate_input, self.cross_modal_cue, self.gate_act = gate_input, cross_modal_cue, gate_act
        self.w_in_cond, self.cue_in_cond, self.missing = w_in_cond, cue_in_cond, missing
        self.dropout_mode, self.dropout_p, self.dropout_p_other = dropout_mode, dropout_p, dropout_p_other

        self.feat_dim = {m: self._feature_dim(m) for m in self.mods}
        n_cue = CUE_DIM if cross_modal_cue else CUES_PER_MODALITY
        self.gate = nn.ModuleDict()
        for m in self.mods:
            in_dim = {"cue": n_cue, "feature": self.feat_dim[m], "both": n_cue + self.feat_dim[m]}[gate_input]
            mlp = nn.Sequential(nn.Linear(in_dim, gate_hidden), nn.Mish(), nn.Linear(gate_hidden, 1))
            nn.init.zeros_(mlp[-1].weight)
            nn.init.constant_(mlp[-1].bias, rgb_gate_bias if m == "rgb" else 0.0)
            self.gate[m] = mlp
        self.e_miss = nn.ParameterDict({m: nn.Parameter(torch.zeros(self.feat_dim[m])) for m in self.mods}) \
            if missing == "learned" else None
        self.last_w = self.last_z = self.last_drop = self.last_cue = None

    # ------------------------------------------------------------------ helpers
    @torch.no_grad()
    def _feature_dim(self, key):
        c, h, w = self.key_shape_map[key]
        model = self.key_model_map[key]
        was = model.training
        model.eval()
        d = model(torch.zeros(1, c, h, w, device=next(model.parameters()).device)).shape[-1]
        model.train(was)
        return d

    def _cue_slice(self, cue, m):
        if self.cross_modal_cue:
            return cue
        i = MODALITIES.index(m) * CUES_PER_MODALITY
        return cue[:, i:i + CUES_PER_MODALITY]

    def _logits(self, z, cue):
        out = []
        for m in self.mods:
            c = self._cue_slice(cue, m)
            x = {"cue": c, "feature": z[m], "both": torch.cat([c, z[m]], -1)}[self.gate_input]
            out.append(self.gate[m](x))
        return torch.cat(out, -1)                                           # (B, M)

    def _weights(self, logits, drop):
        if self.gate_act == "softmax":
            w = logits.masked_fill(drop, float("-inf")).softmax(-1)
            w = torch.nan_to_num(w, nan=0.0)                                # all dropped -> zeros
        else:
            w = torch.sigmoid(logits)
        return w * (~drop).to(w.dtype)

    def _dropout_mask(self, logits):
        """(B, M) bool: which gated modalities to drop in this training step."""
        B, M = logits.shape
        dev = logits.device
        drop = torch.zeros(B, M, dtype=torch.bool, device=dev)
        if not self.training or self.dropout_mode == "none" or M < 2:
            return drop
        if self.dropout_mode == "random":
            drop = torch.rand(B, M, device=dev) < self.dropout_p
        else:   # targeted: the modality the policy currently leans on most
            top = logits.detach().argmax(-1)
            drop = torch.rand(B, M, device=dev) < self.dropout_p_other
            drop[torch.arange(B, device=dev), top] = torch.rand(B, device=dev) < self.dropout_p
        keep = torch.randint(0, M, (B,), device=dev)                        # never drop everything
        drop[drop.all(-1), keep[drop.all(-1)]] = False
        return drop

    # ------------------------------------------------------------------ forward
    def forward(self, obs_dict):
        obs = self._prepare(obs_dict)                                        # shared crop, force_drop -> zeros
        B = obs[self.cue_key].shape[0]
        cue = obs[self.cue_key]
        z = {}
        for m in self.mods:
            img = obs[m]
            assert img.shape[1:] == self.key_shape_map[m], (m, img.shape)
            z[m] = self.key_model_map[m](self.key_transform_map[m](img))

        logits = self._logits(z, cue)
        drop = self._dropout_mask(logits)
        if self.force_drop:
            for i, m in enumerate(self.mods):
                if m in self.force_drop:
                    drop[:, i] = True
        if drop.any():
            for i, m in enumerate(self.mods):
                fill = self.e_miss[m].expand(B, -1) if self.e_miss is not None else torch.zeros_like(z[m])
                z[m] = torch.where(drop[:, i:i + 1], fill, z[m])
            if self.gate_input != "cue":                                     # gate sees the replaced features
                logits = self._logits(z, cue)
        w = self._weights(logits, drop)

        feats = [w[:, i:i + 1] * z[m] for i, m in enumerate(self.mods)]
        for k in self.other_rgb:
            feats.append(self.key_model_map[k](self.key_transform_map[k](obs[k])))
        if self.w_in_cond:
            feats.append(w)
        if self.cue_in_cond:
            feats.append(cue)
        for k in self.state_keys:
            feats.append(obs[k])

        self.last_w, self.last_drop, self.last_cue = w.detach(), drop, cue.detach()
        self.last_z = z                                                      # with graph: the policy's aux heads use it
        return torch.cat(feats, -1)
