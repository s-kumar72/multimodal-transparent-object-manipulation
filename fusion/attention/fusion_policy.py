"""
Diffusion Policy with the fusion encoder (fusion/policy_encoder.py) with optional training terms for the ablations.
With both weights at 0 (the default) the loss and inference are exactly DP's DiffusionUnetImagePolicy.

  anchor_weight     distillation toward the frozen pretrained segmentation network (glass mask + sensor weights),
                    so policy training can't wash out what the encoder learned in pre-training. Needs
                    policy.obs_encoder.anchor=true.
  gate_sup_weight   tells the sensor weights which sensor is unreliable, from the episode flags in the demo zarr
                    (meta/ep_glass_seethrough -> depth, meta/ep_is_crossover -> thermal): BCE of the mean gate score
                    toward 0 when the flag is set and 1 otherwise. Sim-only information, so it is annealed to 0 over
                    gate_sup_anneal_steps (0 = constant): a warm start (not something the policy relies on).

Logging: every log_every training steps the loss terms and the sensor weights are printed, with w_depth on
see-through / other samples and w_thermal on crossover / other samples (when the batch has both kinds).
Evaluation: like policy/gated_policy.py, set policy.w_trace = [] before a rollout and predict_action appends the
mean sensor weights at every decision; policy/eval.py --save_w does this and stores them in episodes.json.
Checkpoints trained before this class existed (plain DiffusionUnetImagePolicy) get the same trace through
fusion/eval_fusion.py.
"""
import torch
import torch.nn.functional as F
from diffusion_policy.policy.diffusion_unet_image_policy import DiffusionUnetImagePolicy


def record_w(policy, obs_dict):
    """Append the encoder's mean sensor weights of the last observation step to policy.w_trace."""
    enc = getattr(policy, "obs_encoder", None)
    if policy.w_trace is None or getattr(enc, "last_w", None) is None:
        return
    B = next(iter(obs_dict.values())).shape[0]
    w = enc.last_w.view(B, -1, enc.last_w.shape[-1])[:, -1]
    if not policy.w_trace:
        policy.w_trace.append([])
    policy.w_trace[-1].append({"w": w.float().cpu().numpy(), "cue": None})


class FusionDiffusionPolicy(DiffusionUnetImagePolicy):
    def __init__(self, *args, anchor_weight=0.0, gate_sup_weight=0.0, gate_sup_anneal_steps=20000, log_every=500,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.anchor_weight, self.gate_sup_weight = float(anchor_weight), float(gate_sup_weight)
        self.gate_sup_anneal_steps, self.log_every = int(gate_sup_anneal_steps), int(log_every)
        assert self.anchor_weight == 0 or getattr(self.obs_encoder, "_teacher", None) is not None, \
            "anchor_weight > 0 needs policy.obs_encoder.anchor=true"
        self.register_buffer("train_steps", torch.zeros((), dtype=torch.long), persistent=True)
        self._terms, self._counts, self._n = {}, {}, 0
        self.w_trace = None

    # ------------------------------------------------------------------ training
    def gate_sup_lambda(self):
        if self.gate_sup_weight <= 0:
            return 0.0
        if self.gate_sup_anneal_steps <= 0:
            return self.gate_sup_weight
        return self.gate_sup_weight * max(0.0, 1.0 - float(self.train_steps) / self.gate_sup_anneal_steps)

    def compute_loss(self, batch):
        loss = super().compute_loss(batch)              # runs the encoder on (B * n_obs_steps) frames
        terms = {"ddpm": loss.detach()}
        enc = self.obs_encoder
        if self.anchor_weight > 0 and getattr(enc, "last_anchor", None) is not None:
            loss = loss + self.anchor_weight * enc.last_anchor
            terms["anchor"] = enc.last_anchor.detach()

        meta = batch.get("meta", {})
        mods = list(getattr(getattr(enc, "net", None), "mods", []))
        B, To = batch["action"].shape[0], self.n_obs_steps
        w = enc.last_w.view(B, To, -1) if getattr(enc, "last_w", None) is not None else None
        lam = self.gate_sup_lambda()
        if lam > 0 and getattr(enc, "last_gate_logit", None) is not None:
            logit = enc.last_gate_logit.view(B, To, -1)
            parts = []
            for m, flag in (("depth", "ep_glass_seethrough"), ("thermal", "ep_is_crossover")):
                if m in mods and flag in meta and bool(enc.avail_mask[mods.index(m)]):
                    target = (1.0 - (meta[flag].float() > 0.5).float())[:, None].expand(B, To)
                    parts.append(F.binary_cross_entropy_with_logits(logit[:, :, mods.index(m)], target))
            if parts:
                l_sup = torch.stack(parts).mean()
                loss = loss + lam * l_sup
                terms["gate_sup"] = l_sup.detach()
        if w is not None:
            for i, m in enumerate(mods):
                terms[f"w_{m}"] = w[:, :, i].mean().detach()
            for m, flag, name in (("depth", "ep_glass_seethrough", "st"), ("thermal", "ep_is_crossover", "xo")):
                if m in mods and flag in meta:
                    wm, f = w[:, :, mods.index(m)].mean(-1), meta[flag].float() > 0.5
                    if f.any() and (~f).any():
                        terms[f"w_{m}_{name}"] = wm[f].mean().detach()
                        terms[f"w_{m}_not{name}"] = wm[~f].mean().detach()
        if self.training and torch.is_grad_enabled():
            self.train_steps += 1
            self._log(terms, lam)
        return loss

    def _log(self, terms, lam):
        for k, v in terms.items():
            self._terms[k] = self._terms.get(k, 0.0) + float(v)
            self._counts[k] = self._counts.get(k, 0) + 1
        self._n += 1
        if self.log_every and self._n >= self.log_every:
            msg = "  ".join(f"{k} {v / self._counts[k]:.4f}" for k, v in self._terms.items())
            print(f"[fusion step {int(self.train_steps)}] lambda_gate_sup {lam:.3f}  {msg}", flush=True)
            self._terms, self._counts, self._n = {}, {}, 0

    # ------------------------------------------------------------------ inference
    def reset(self):
        super().reset()
        if self.w_trace is not None:
            self.w_trace.append([])

    def predict_action(self, obs_dict):
        result = super().predict_action(obs_dict)
        record_w(self, obs_dict)
        return result