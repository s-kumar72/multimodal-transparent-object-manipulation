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
  route            none | visible. visible: no learned gate; a sensor is on (w = 1) when the target is visible in it and
                   off (w = 0, feature zeroed) otherwise, read from the two visibility flags at the end of the cue
                   (policy/obs_util.py, vis_thr). The same rule is applied in training and at inference, so each sensor
                   subset is a mode the policy has been trained in. rgb is never switched off.
                   soft: the same two measurements give a continuous weight instead of on / off,
                   w_m = sigmoid(a_m (s_m - thr_m) / tau_m + b_m) for depth and thermal (s = depth step / thermal step
                   at the glass mask, raw units: the cue key needs vis_raw; thr = the cue's vis_thr; tau = route_tau).
                   a (init 1) and b (init 0) are learned with the policy, so training can sharpen, soften or shift the
                   gate; with a = 1, b = 0 the weight is 0.5 at the threshold of route=visible. rgb stays at 1.
                   learned: on / off as with visible, but decided by a gate that is trained (GatedDiffusionPolicy
                   gate_train): p_m = sigmoid(MLP_m(cue)) for depth and thermal, the cue given WITHOUT its two
                   thresholded flags. Training: the decision is sampled, on with probability p clipped to
                   [route_explore, 1 - route_explore], or forced by the policy (route_force); inference: on when
                   p > 0.5. The decision is hard, so the imitation loss gives the gate no gradient: what trains it is
                   the policy's gate loss on last_p.
  route_tau        [depth, thermal] width of the soft gate in the units of the measurements (route=soft).
  route_rgb_only_p training only, with route=visible / soft / learned: extra probability of switching off depth and
                   thermal together.
  route_explore    route=learned, training: the sampled decision keeps at least this probability of each state.
  gate_norm        none | max. max: the learned sigmoid weights are divided by the largest one of the frame, so the most
                   trusted sensor has w = 1 and the others are relative to it (the weights cannot all shrink together).
At inference `route_eval` picks the rule: visible (as trained) | depth_first (thermal only when depth does not show
the target; route=visible only) | all_on (no routing) | rgb_only | flags (route=learned only: the thresholded
flags of route=visible decide instead of the trained gate).
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
                 dropout_p_other=0.1, rgb_gate_bias=2.0, route="none", route_rgb_only_p=0.0, route_tau=None,
                 gate_norm="none", route_explore=0.0):
        assert not share_rgb_model, "the gate needs one encoder per modality"
        assert route in ("none", "visible", "soft", "learned") and gate_norm in ("none", "max")
        assert route != "learned" or (gate_input == "cue" and cross_modal_cue), "route=learned gates on the full cue"
        assert gate_input in ("cue", "feature", "both") and gate_act in ("sigmoid", "softmax")
        assert missing in ("learned", "zero") and dropout_mode in ("none", "random", "targeted")
        super().__init__(shape_meta, rgb_model, resize_shape=resize_shape, crop_shape=crop_shape,
                         random_crop=random_crop, use_group_norm=use_group_norm, share_rgb_model=False,
                         imagenet_norm=imagenet_norm, init_weights=init_weights)
        assert cue_key in self.low_dim_keys, f"shape_meta needs a '{cue_key}' low_dim observation (task glass_pick_cue)"
        n_cue_all = int(self.key_shape_map[cue_key][0])
        assert len(self.key_shape_map[cue_key]) == 1 and n_cue_all >= CUE_DIM, \
            "cue shape must be obs_util.CUE_DIM plus the optional entries (depth_object_ref: 1, vis_thr: 4)"
        assert cross_modal_cue or n_cue_all == CUE_DIM, "per-modality cue slices assume the 6-vector"
        assert route == "none" or n_cue_all >= CUE_DIM + 4, "route=visible needs the visibility cues (vis_thr)"
        self.route, self.route_rgb_only_p, self.route_eval = route, route_rgb_only_p, "visible"
        self.route_explore, self.route_force, self.last_p = float(route_explore), None, None
        self.gate_norm = gate_norm
        if route == "soft":
            cue_attr = shape_meta["obs"][cue_key]
            assert cue_attr.get("vis_raw"), "route=soft reads the two measurements in raw units: set cue.vis_raw"
            assert route_tau is not None and len(route_tau) == 2, "route=soft needs route_tau = [depth, thermal]"
            self.register_buffer("route_thr", torch.tensor([float(x) for x in cue_attr["vis_thr"]]))
            self.register_buffer("route_tau", torch.tensor([float(x) for x in route_tau]))
            self.route_a = nn.Parameter(torch.ones(2))
            self.route_b = nn.Parameter(torch.zeros(2))
        self.cue_key = cue_key
        self.mods = [m for m in MODALITIES if m in self.rgb_keys]          # gated, in the cue's fixed order
        self.other_rgb = [k for k in self.rgb_keys if k not in self.mods]  # any other camera: concatenated ungated
        self.state_keys = [k for k in self.low_dim_keys if k != cue_key]
        self.gate_input, self.cross_modal_cue, self.gate_act = gate_input, cross_modal_cue, gate_act
        self.w_in_cond, self.cue_in_cond, self.missing = w_in_cond, cue_in_cond, missing
        self.dropout_mode, self.dropout_p, self.dropout_p_other = dropout_mode, dropout_p, dropout_p_other

        self.feat_dim = {m: self._feature_dim(m) for m in self.mods}
        n_cue = n_cue_all if cross_modal_cue else CUES_PER_MODALITY
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
        w = w * (~drop).to(w.dtype)
        if self.gate_norm == "max":
            w = w / w.max(-1, keepdim=True).values.clamp(min=1e-6)
        return w

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

    def _route_mask(self, cue):
        """(B, M) bool: sensors to switch off, from the visibility flags (last two cue entries: depth, thermal)."""
        B, dev = cue.shape[0], cue.device
        vis = {"depth": cue[:, -2] > 0.5, "thermal": cue[:, -1] > 0.5}
        rule = "visible" if self.training else self.route_eval
        assert self.route == "visible" or rule != "depth_first", "depth_first is a rule of route=visible"
        drop = torch.zeros(B, len(self.mods), dtype=torch.bool, device=dev)
        for i, m in enumerate(self.mods):
            if m == "rgb" or rule == "all_on":
                continue
            if rule == "rgb_only":
                drop[:, i] = True
            elif self.route == "soft":      # no hard switch: the weight itself goes to 0 (_soft_weights)
                continue
            elif rule == "depth_first" and m == "thermal":
                drop[:, i] = vis["depth"] | ~vis["thermal"]
            else:
                drop[:, i] = ~vis[m]
        if self.training and self.route_rgb_only_p > 0:
            off = torch.rand(B, device=dev) < self.route_rgb_only_p
            for i, m in enumerate(self.mods):
                if m != "rgb":
                    drop[:, i] |= off
        return drop

    def _learned_route(self, cue):
        """route=learned -> (B, M) bool sensors to switch off, (B, 2) gate probabilities [depth, thermal] (with graph).
        route_force ((B, 2), 1 = on) replaces the gate's own decision; route_eval all_on / rgb_only as for visible."""
        B, dev = cue.shape[0], cue.device
        c = cue.clone()
        c[:, -2:] = 0                                                       # no thresholded flags in the gate's input
        p = torch.cat([torch.sigmoid(self.gate[m](c)) if m in self.gate else torch.ones(B, 1, dtype=cue.dtype, device=dev)
                       for m in ("depth", "thermal")], -1)
        if self.route_force is not None:
            on = self.route_force.to(p.dtype) > 0.5
        elif self.training:
            e = self.route_explore
            on = torch.rand_like(p) < p.detach().clamp(e, 1 - e)
        elif self.route_eval == "flags":                                    # the rule of route=visible, not the gate
            on = cue[:, -2:] > 0.5
        else:
            on = p > 0.5
        rule = "visible" if self.training else self.route_eval
        drop = torch.zeros(B, len(self.mods), dtype=torch.bool, device=dev)
        for i, m in enumerate(self.mods):
            if m == "rgb" or rule == "all_on":
                continue
            drop[:, i] = True if rule == "rgb_only" else ~on[:, ("depth", "thermal").index(m)]
        if self.training and self.route_rgb_only_p > 0:
            off = torch.rand(B, device=dev) < self.route_rgb_only_p
            for i, m in enumerate(self.mods):
                if m != "rgb":
                    drop[:, i] |= off
        return drop, p

    def _soft_weights(self, cue):
        """(B, M) continuous weights from the two visibility measurements (cue entries -4, -3, raw units)."""
        B = cue.shape[0]
        w = torch.ones(B, len(self.mods), dtype=cue.dtype, device=cue.device)
        if not self.training and self.route_eval == "all_on":
            return w
        s = cue[:, -4:-2]
        g = torch.sigmoid(self.route_a * (s - self.route_thr) / self.route_tau + self.route_b)
        # no glass mask in the frame: compute_cues sets both flags to 1 with zero steps -> both sensors stay on
        found = ~((cue[:, -2] > 0.5) & (s[:, 0] <= self.route_thr[0]))
        g = torch.where(found[:, None], g, torch.ones_like(g))
        cols = [g[:, ("depth", "thermal").index(m)] if m in ("depth", "thermal") else w[:, i]
                for i, m in enumerate(self.mods)]
        return torch.stack(cols, -1)

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
        if self.route == "learned":
            drop, self.last_p = self._learned_route(cue)
        else:
            drop = self._route_mask(cue) if self.route != "none" else self._dropout_mask(logits)
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
        if self.route in ("visible", "learned"):
            w = (~drop).to(logits.dtype)
        elif self.route == "soft":
            w = self._soft_weights(cue) * (~drop).to(logits.dtype)
        else:
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
