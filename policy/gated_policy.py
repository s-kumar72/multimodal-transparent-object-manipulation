"""
DiffusionUnetImagePolicy with the training-time extras of the cue-gated encoder. The denoiser, the diffusion loss and
inference are DP's; with a plain encoder (SharedCropObsEncoder / MultiImageObsEncoder) this class is exactly DP.

Extra loss terms (each off at weight 0):
  gate loss   L_gate = mean (w_thermal - tanh(|thermal gap| / gate_tau))^2 on frames where thermal was not dropped.
              Uses the per-episode thermal gap from the zarr meta (sim only, never available on a robot), so it is
              annealed linearly to 0 over gate_anneal_steps: a warm start for the gate, not something it depends on.
  aux loss    unimodal action heads: from each modality's feature at the last observation step (+ robot state),
              predict the next n_action_steps normalized actions. Keeps every encoder learning even when rgb alone
              already solves the task (modality collapse); its per-modality error is also the "unimodal error" the
              gate weights should anti-correlate with (QMF).
Logging: loss terms are averaged and printed every log_every training steps (the DP workspace logs only the total).
For evaluation, set policy.w_trace = [] before a rollout; predict_action then appends the gate weights and raw cues of
the last observation step at every decision, one list per policy.reset() (= per env chunk of the runner).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusion_policy.policy.diffusion_unet_image_policy import DiffusionUnetImagePolicy


class GatedDiffusionPolicy(DiffusionUnetImagePolicy):
    def __init__(self, *args, gate_loss_weight=0.1, gate_anneal_steps=20000, gate_tau=5.0,
                 aux_weight=0.1, aux_hidden=256, log_every=500, **kwargs):
        super().__init__(*args, **kwargs)
        self.gate_loss_weight, self.gate_anneal_steps, self.gate_tau = gate_loss_weight, gate_anneal_steps, gate_tau
        self.aux_weight, self.log_every = aux_weight, log_every
        self.register_buffer("train_steps", torch.zeros((), dtype=torch.long), persistent=True)
        enc = self.obs_encoder
        self.gated = hasattr(enc, "mods") and hasattr(enc, "feat_dim")
        self.aux_heads = None
        if self.gated and aux_weight > 0:
            state_dim = sum(int(torch.tensor(enc.key_shape_map[k]).prod()) for k in enc.state_keys)
            out_dim = self.action_dim * self.n_action_steps
            self.aux_heads = nn.ModuleDict({m: nn.Sequential(nn.Linear(enc.feat_dim[m] + state_dim, aux_hidden),
                                                             nn.Mish(), nn.Linear(aux_hidden, out_dim))
                                            for m in enc.mods})
        self._terms, self._n_terms = {}, 0
        self.w_trace = None

    # ------------------------------------------------------------------ training
    def gate_lambda(self):
        if self.gate_loss_weight <= 0:
            return 0.0
        if self.gate_anneal_steps <= 0:
            return self.gate_loss_weight
        return self.gate_loss_weight * max(0.0, 1.0 - float(self.train_steps) / self.gate_anneal_steps)

    def compute_loss(self, batch):
        loss = super().compute_loss(batch)          # runs the encoder: last_w / last_z / last_drop are for this batch
        terms = {"ddpm": loss.detach()}
        enc = self.obs_encoder
        if self.gated and getattr(enc, "last_w", None) is not None:
            B, To = batch["action"].shape[0], self.n_obs_steps
            M = len(enc.mods)
            w = enc.last_w.view(B, To, M)
            drop = enc.last_drop.view(B, To, M)

            lam = self.gate_lambda()
            meta = batch.get("meta", {})
            if lam > 0 and "thermal" in enc.mods and "ep_thermal_gap" in meta:
                ti = enc.mods.index("thermal")
                target = torch.tanh(meta["ep_thermal_gap"].float().abs() / self.gate_tau)[:, None]   # (B, 1)
                keep = (~drop[:, :, ti]).float()
                l_gate = (((w[:, :, ti] - target) ** 2) * keep).sum() / keep.sum().clamp(min=1)
                loss = loss + lam * l_gate
                terms["gate"] = l_gate.detach()

            if self.aux_heads is not None:
                nobs = self.normalizer.normalize(batch["obs"])
                state = torch.cat([nobs[k][:, To - 1].reshape(B, -1) for k in enc.state_keys], -1)
                nact = self.normalizer["action"].normalize(batch["action"])
                target = nact[:, To - 1:To - 1 + self.n_action_steps].reshape(B, -1)
                for i, m in enumerate(enc.mods):
                    z = enc.last_z[m].view(B, To, -1)[:, -1]
                    keep = (~drop[:, -1, i]).float()
                    err = ((self.aux_heads[m](torch.cat([z, state], -1)) - target) ** 2).mean(-1)
                    l_aux = (err * keep).sum() / keep.sum().clamp(min=1)
                    loss = loss + self.aux_weight * l_aux
                    terms[f"aux_{m}"] = l_aux.detach()
            for i, m in enumerate(enc.mods):
                terms[f"w_{m}"] = w[:, :, i].mean().detach()

        if self.training and torch.is_grad_enabled():
            self.train_steps += 1
            self._log(terms)
        return loss

    def _log(self, terms):
        for k, v in terms.items():
            self._terms[k] = self._terms.get(k, 0.0) + float(v)
        self._n_terms += 1
        if self.log_every and self._n_terms >= self.log_every:
            msg = "  ".join(f"{k} {v / self._n_terms:.4f}" for k, v in self._terms.items())
            print(f"[gated step {int(self.train_steps)}] lambda_gate {self.gate_lambda():.3f}  {msg}", flush=True)
            self._terms, self._n_terms = {}, 0

    # ------------------------------------------------------------------ inference
    def reset(self):
        super().reset()
        if self.w_trace is not None:
            self.w_trace.append([])

    def predict_action(self, obs_dict):
        result = super().predict_action(obs_dict)
        enc = self.obs_encoder
        if self.w_trace is not None and self.gated and getattr(enc, "last_w", None) is not None:
            B = next(iter(obs_dict.values())).shape[0]
            w = enc.last_w.view(B, self.n_obs_steps, -1)[:, -1]
            cue = obs_dict[enc.cue_key][:, -1] if enc.cue_key in obs_dict else None
            if not self.w_trace:
                self.w_trace.append([])
            self.w_trace[-1].append({"w": w.cpu().numpy(), "cue": None if cue is None else cue.cpu().numpy()})
        return result
