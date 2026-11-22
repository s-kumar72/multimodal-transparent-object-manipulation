"""
DiffusionUnetImagePolicy with the training-time extras of the cue-gated encoder. The denoiser, the diffusion loss and
inference are DP's; with a plain encoder (SharedCropObsEncoder / MultiImageObsEncoder) this class is exactly DP.

Extra loss terms (each off at weight 0):
  gate loss   L_gate = mean (w_thermal - tanh(|thermal gap| / gate_tau))^2 on frames where thermal was not dropped.
              Uses the per-episode thermal gap from the zarr meta (sim only, never available on a robot), so it is
              annealed linearly to 0 over gate_anneal_steps: a warm start for the gate, not something it depends on.
              gate_depth_target="seethrough" adds the same warm start for depth, (w_depth - (1 - see-through))^2 with
              the episode's ep_glass_seethrough flag, with the same weight and schedule (off by default).
  aux loss    unimodal action heads: from each modality's feature at the last observation step (+ robot state),
              predict the next n_action_steps normalized actions. Keeps every encoder learning even when rgb alone
              already solves the task (modality collapse); its per-modality error is also the "unimodal error" the
              gate weights should anti-correlate with (QMF).
Logging: loss terms are averaged and printed every log_every training steps (the DP workspace logs only the total),
with w_depth on see-through / other samples and w_thermal on crossover / other samples when the meta has the flags.
For evaluation, set policy.w_trace = [] before a rollout; predict_action then appends the gate weights and raw cues of
the last observation step at every decision, one list per policy.reset() (= per env chunk of the runner).

Inference options for an encoder with route=visible (policy/gated_obs_encoder.py), both off by default:
  route_latch  the visibility flags of the first decision after reset() are kept for the whole episode, so the arm
               passing in front of the glass later cannot switch a sensor off.
  guidance     s != 1: the denoiser output is eps_rgb + s (eps_routed - eps_rgb), with eps_rgb from the same network with
               depth and thermal switched off (classifier-free guidance toward the sensors the routing trusts); needs
               training with route_rgb_only_p > 0 so that the rgb-only mode has been trained. Two denoiser calls per step.
  gate_error   robustness tests (eval.py --gate_error): wrong visibility flags at inference. A dict with any of
               flip = probability that each flag (depth, thermal; per env) is inverted at a decision, drawn anew at
               every decision from a generator seeded with `seed` (0); depth / thermal = 0 or 1, the flag is forced
               to that value in every frame (0: the sensor is always off, 1: always on). None (default): no change.

Training the gate of an encoder with route=learned (gate_train; the decision is hard, so the imitation loss of the
samples the policy is trained on cannot move the gate):
  labels    (gate_balance=true: the two label classes of a sensor weigh the same in every batch, so the rare
            "not visible" class is not outvoted)
            the gate is a classifier of "the glass is visible in this sensor", trained with a cross-entropy on the
            episode labels of the simulator (depth: not see-through, thermal: no crossover). During training the
            sensors are switched by the labels, at inference by the gate.
  heldout   no labels. The episodes the dataset marks (gate_split, batch meta ep_gate_split) are left out of the
            policy's loss. On them, without gradient, the denoising loss is evaluated with each sensor on and off
            (same noise and timestep), and the gate is moved toward the state with the lower loss:
            loss_gate = sum_m p_m (L_on,m - L_off,m). A sensor is kept where it lowers the loss on demos the policy
            was not fitted to, which is what the training loss cannot tell (it is lower with every sensor on).
            The gate loss starts after gate_start_step training steps.
  frozen    two-stage training: the gate was fitted before (policy/fit_gate.py) and is loaded from gate_state
            ({"gate": state dict}); it is not trained here. The sensors are switched by its decision (p > 0.5)
            during training and at inference, and every demo trains the policy, as with route=visible.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import reduce

from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.policy.diffusion_unet_image_policy import DiffusionUnetImagePolicy


class GatedDiffusionPolicy(DiffusionUnetImagePolicy):
    def __init__(self, *args, gate_loss_weight=0.1, gate_anneal_steps=20000, gate_tau=5.0,
                 aux_weight=0.1, aux_hidden=256, log_every=500, gate_depth_target="none", gate_train="none",
                 gate_start_step=2000, gate_state=None, gate_balance=False, **kwargs):
        super().__init__(*args, **kwargs)
        assert gate_depth_target in ("none", "seethrough") and gate_train in ("none", "labels", "heldout", "frozen")
        assert (gate_train == "frozen") == (gate_state is not None), "gate_state is the gate of gate_train=frozen"
        assert (gate_train != "none") == (getattr(self.obs_encoder, "route", "none") == "learned"), \
            "gate_train labels / heldout goes with an encoder with route=learned"
        self.gate_train, self.gate_start_step = gate_train, gate_start_step
        self.gate_balance = bool(gate_balance)       # gate_train=labels: both label classes weigh the same per sensor
        if gate_train == "frozen":
            self.obs_encoder.gate.load_state_dict(torch.load(gate_state, map_location="cpu")["gate"])
            for q in self.obs_encoder.gate.parameters():
                q.requires_grad_(False)
            self.obs_encoder.route_sample = False          # training switches by p > 0.5, as inference does
        self.gate_loss_weight, self.gate_anneal_steps, self.gate_tau = gate_loss_weight, gate_anneal_steps, gate_tau
        self.gate_depth_target = gate_depth_target
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
        self._terms, self._counts, self._n_terms = {}, {}, 0
        self.w_trace = None
        self.route_latch, self.guidance = False, 1.0
        self._latched, self._guide_cond = None, None
        self.gate_error, self._ge_gen = None, None
        self.route_hold, self._held = False, None

    # ------------------------------------------------------------------ training
    def gate_lambda(self):
        if self.gate_loss_weight <= 0:
            return 0.0
        if self.gate_anneal_steps <= 0:
            return self.gate_loss_weight
        return self.gate_loss_weight * max(0.0, 1.0 - float(self.train_steps) / self.gate_anneal_steps)

    def _ddpm_per_sample(self, batch, noise=None, timesteps=None):
        """The loss of DiffusionUnetImagePolicy.compute_loss per sample: (B,), and the noise and timesteps used, so
        that the same sample can be scored again with other gate decisions."""
        assert self.obs_as_global_cond, "implemented for global conditioning"
        nobs = self.normalizer.normalize(batch["obs"])
        trajectory = self.normalizer["action"].normalize(batch["action"])
        B = trajectory.shape[0]
        this_nobs = dict_apply(nobs, lambda x: x[:, :self.n_obs_steps, ...].reshape(-1, *x.shape[2:]))
        global_cond = self.obs_encoder(this_nobs).reshape(B, -1)
        condition_mask = self.mask_generator(trajectory.shape)
        if noise is None:
            noise = torch.randn(trajectory.shape, device=trajectory.device)
        if timesteps is None:
            timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (B,),
                                      device=trajectory.device).long()
        noisy = self.noise_scheduler.add_noise(trajectory, noise, timesteps)
        noisy[condition_mask] = trajectory[condition_mask]
        pred = self.model(noisy, timesteps, local_cond=None, global_cond=global_cond)
        pred_type = self.noise_scheduler.config.prediction_type
        assert pred_type in ("epsilon", "sample"), pred_type
        loss = F.mse_loss(pred, noise if pred_type == "epsilon" else trajectory, reduction="none")
        loss = loss * (~condition_mask).type(loss.dtype)
        return reduce(loss, "b ... -> b", "mean"), noise, timesteps

    def _compute_loss_learned(self, batch):
        """Training loss with a trained routing gate (encoder route=learned), see the module docstring."""
        enc, meta = self.obs_encoder, batch.get("meta", {})
        B, To, M = batch["action"].shape[0], self.n_obs_steps, len(self.obs_encoder.mods)
        dev = batch["action"].device
        visible = None
        if "ep_glass_seethrough" in meta and "ep_is_crossover" in meta:      # (B, 2): depth, thermal show the glass
            visible = torch.stack([1 - meta["ep_glass_seethrough"].float(), 1 - meta["ep_is_crossover"].float()], -1)
        held = torch.zeros(B, dtype=torch.bool, device=dev)
        if self.gate_train == "labels":
            assert visible is not None, "gate_train=labels needs ep_glass_seethrough / ep_is_crossover in the batch meta"
            enc.route_force = visible[:, None, :].expand(B, To, 2).reshape(B * To, 2)
        elif self.gate_train == "heldout":
            assert "ep_gate_split" in meta, "gate_train=heldout needs the dataset's gate_split"
            held = meta["ep_gate_split"].float() > 0.5
        try:
            per, _, _ = self._ddpm_per_sample(batch)      # sensors switched by the labels / the sampled gate decision
        finally:
            enc.route_force = None
        fit = (~held).to(per.dtype)                        # the samples the policy is trained on
        loss = (per * fit).sum() / fit.sum().clamp(min=1)
        terms = {"ddpm": loss.detach()}
        p = enc.last_p.view(B, To, 2)                      # gate probabilities of this batch, with graph
        w, drop, z = enc.last_w.view(B, To, M), enc.last_drop.view(B, To, M), enc.last_z

        if self.aux_heads is not None:
            nobs = self.normalizer.normalize(batch["obs"])
            state = torch.cat([nobs[k][:, To - 1].reshape(B, -1) for k in enc.state_keys], -1)
            nact = self.normalizer["action"].normalize(batch["action"])
            target = nact[:, To - 1:To - 1 + self.n_action_steps].reshape(B, -1)
            for i, m in enumerate(enc.mods):
                keep = (~drop[:, -1, i]).float() * fit
                err = ((self.aux_heads[m](torch.cat([z[m].view(B, To, -1)[:, -1], state], -1)) - target) ** 2).mean(-1)
                l_aux = (err * keep).sum() / keep.sum().clamp(min=1)
                loss = loss + self.aux_weight * l_aux
                terms[f"aux_{m}"] = l_aux.detach()

        if self.gate_train == "labels":
            tgt = visible[:, None, :].expand(B, To, 2)
            if self.gate_balance:       # per sensor, "visible" and "not visible" each carry half of the weight of the batch
                share = tgt.mean((0, 1), keepdim=True)
                wgt = torch.where(tgt > 0.5, 0.5 / share.clamp(min=1e-6), 0.5 / (1 - share).clamp(min=1e-6))
                l_gate = (F.binary_cross_entropy(p, tgt, reduction="none") * wgt).mean()
            else:
                l_gate = F.binary_cross_entropy(p, tgt)
            loss = loss + l_gate
            terms["gate_bce"] = l_gate.detach()
            terms["gate_acc"] = ((p > 0.5) == (tgt > 0.5)).float().mean().detach()
        elif held.any() and int(self.train_steps) >= self.gate_start_step:
            sub = {"obs": {k: v[held] for k, v in batch["obs"].items()}, "action": batch["action"][held]}
            n_held = int(held.sum())
            L, noise, steps = {}, None, None
            was = enc.training
            enc.eval()                                     # center crop, no rgb-only draw: the four states differ by the gate only
            try:
                with torch.no_grad():
                    for d in (0, 1):
                        for t in (0, 1):
                            enc.route_force = torch.tensor([[d, t]], dtype=per.dtype, device=dev).expand(n_held * To, 2)
                            L[(d, t)], noise, steps = self._ddpm_per_sample(sub, noise, steps)
            finally:
                enc.route_force = None
                enc.train(was)
            ph = p[held].mean(1)                           # (n_held, 2)
            pd, pt = ph[:, 0].detach(), ph[:, 1].detach()
            delta_d = pt * (L[(1, 1)] - L[(0, 1)]) + (1 - pt) * (L[(1, 0)] - L[(0, 0)])   # loss with depth on - off
            delta_t = pd * (L[(1, 1)] - L[(1, 0)]) + (1 - pd) * (L[(0, 1)] - L[(0, 0)])   # loss with thermal on - off
            loss = loss + (ph[:, 0] * delta_d + ph[:, 1] * delta_t).mean()
            terms["gate_delta_depth"], terms["gate_delta_thermal"] = delta_d.mean(), delta_t.mean()
            if visible is not None:
                for j, (delta, name) in enumerate(((delta_d, "depth"), (delta_t, "thermal"))):
                    v = visible[held][:, j] > 0.5
                    if v.any() and (~v).any():
                        terms[f"gate_delta_{name}_vis"], terms[f"gate_delta_{name}_notvis"] = delta[v].mean(), delta[~v].mean()

        for i, m in enumerate(enc.mods):
            terms[f"w_{m}"] = w[:, :, i].mean().detach()
        if visible is not None:
            pm = p.mean(1).detach()
            for j, name in enumerate(("depth", "thermal")):
                v = visible[:, j] > 0.5
                if v.any() and (~v).any():
                    terms[f"p_{name}_vis"], terms[f"p_{name}_notvis"] = pm[v, j].mean(), pm[~v, j].mean()
        self.train_steps += 1
        self._log(terms)
        return loss

    def compute_loss(self, batch):
        if self.gate_train != "none" and self.training and torch.is_grad_enabled():
            return self._compute_loss_learned(batch)
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
            if lam > 0 and self.gate_depth_target == "seethrough" and "depth" in enc.mods \
                    and "ep_glass_seethrough" in meta:
                di = enc.mods.index("depth")
                target = (1.0 - meta["ep_glass_seethrough"].float())[:, None]                      # (B, 1)
                keep = (~drop[:, :, di]).float()
                l_gate_d = (((w[:, :, di] - target) ** 2) * keep).sum() / keep.sum().clamp(min=1)
                loss = loss + lam * l_gate_d
                terms["gate_depth"] = l_gate_d.detach()

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
            if getattr(enc, "route", "none") == "soft":      # the learned slope / shift of the soft gate
                for j, m in enumerate(("depth", "thermal")):
                    terms[f"route_a_{m}"], terms[f"route_b_{m}"] = enc.route_a[j].detach(), enc.route_b[j].detach()
            for m, flag, name in (("depth", "ep_glass_seethrough", "st"), ("thermal", "ep_is_crossover", "xo")):
                if m in enc.mods and flag in meta:
                    wm, f = w[:, :, enc.mods.index(m)].mean(-1), meta[flag].float() > 0.5
                    if f.any() and (~f).any():   # only batches with both kinds, so the averages are comparable
                        terms[f"w_{m}_{name}"] = wm[f].mean().detach()
                        terms[f"w_{m}_not{name}"] = wm[~f].mean().detach()

        if self.training and torch.is_grad_enabled():
            self.train_steps += 1
            self._log(terms)
        return loss

    def _log(self, terms):
        for k, v in terms.items():
            self._terms[k] = self._terms.get(k, 0.0) + float(v)
            self._counts[k] = self._counts.get(k, 0) + 1     # a term can be missing from some steps
        self._n_terms += 1
        if self.log_every and self._n_terms >= self.log_every:
            msg = "  ".join(f"{k} {v / self._counts[k]:.4f}" for k, v in self._terms.items())
            print(f"[gated step {int(self.train_steps)}] lambda_gate {self.gate_lambda():.3f}  {msg}", flush=True)
            self._terms, self._counts, self._n_terms = {}, {}, 0

    # ------------------------------------------------------------------ inference
    def reset(self):
        super().reset()
        self._latched = None
        self._held = None
        if self.w_trace is not None:
            self.w_trace.append([])

    def conditional_sample(self, condition_data, condition_mask, local_cond=None, global_cond=None, generator=None,
                           **kwargs):
        if self._guide_cond is None:
            return super().conditional_sample(condition_data, condition_mask, local_cond=local_cond,
                                              global_cond=global_cond, generator=generator, **kwargs)
        scheduler = self.noise_scheduler
        trajectory = torch.randn(size=condition_data.shape, dtype=condition_data.dtype, device=condition_data.device,
                                 generator=generator)
        scheduler.set_timesteps(self.num_inference_steps)
        for t in scheduler.timesteps:
            trajectory[condition_mask] = condition_data[condition_mask]
            eps = self.model(trajectory, t, local_cond=local_cond, global_cond=global_cond)
            eps_rgb = self.model(trajectory, t, local_cond=local_cond, global_cond=self._guide_cond)
            trajectory = scheduler.step(eps_rgb + self.guidance * (eps - eps_rgb), t, trajectory, generator=generator,
                                        **kwargs).prev_sample
        trajectory[condition_mask] = condition_data[condition_mask]
        return trajectory

    def predict_action(self, obs_dict):
        enc = self.obs_encoder
        routed = getattr(enc, "route", "none") != "none"
        if routed and self.route_hold:
            # A frame without a glass mask has both flags on and both steps 0 (obs_util.compute_cues). Instead of
            # switching both sensors on there, keep the flags of the last frame of the episode that had a mask; before
            # the first such frame both sensors are off (rgb only).
            assert enc.route == "visible", "route_hold keeps the on / off flags: route=visible only"
            cue = obs_dict[enc.cue_key].clone()                     # (B, To, C), raw
            if self._held is None or self._held.shape[0] != cue.shape[0]:
                self._held = torch.zeros(cue.shape[0], 2, dtype=cue.dtype, device=cue.device)
            for j in range(cue.shape[1]):                           # frames in time order
                c = cue[:, j]
                found = ~((c[:, -2] > 0.5) & (c[:, -1] > 0.5) & (c[:, -4] == 0) & (c[:, -3] == 0))
                self._held = torch.where(found[:, None], c[:, -2:], self._held)
                cue[:, j, -2:] = self._held
            obs_dict = dict(obs_dict)
            obs_dict[enc.cue_key] = cue
        if routed and self.gate_error:
            assert enc.route == "visible", "gate_error changes the on / off flags: route=visible only"
            cue = obs_dict[enc.cue_key].clone()
            flags = cue[..., -2:]                                   # (B, To, 2) view: depth, thermal
            p = float(self.gate_error.get("flip", 0.0))
            if p > 0:
                if self._ge_gen is None:
                    self._ge_gen = torch.Generator().manual_seed(int(self.gate_error.get("seed", 0)))
                flip = (torch.rand(flags.shape[0], 2, generator=self._ge_gen) < p).to(flags.device)
                flags[:] = torch.where(flip[:, None, :], 1 - flags, flags)
            for i, m in enumerate(("depth", "thermal")):
                if m in self.gate_error:
                    flags[..., i] = float(self.gate_error[m])
            obs_dict = dict(obs_dict)
            obs_dict[enc.cue_key] = cue
        if routed and self.route_latch:
            assert enc.route == "visible", "route_latch keeps the on / off flags: route=visible only"
            flags = obs_dict[enc.cue_key][:, -1, -2:]
            if self._latched is None or self._latched.shape != flags.shape:
                self._latched = flags.clone()
            cue = obs_dict[enc.cue_key].clone()
            cue[..., -2:] = self._latched[:, None, :]
            obs_dict = dict(obs_dict)
            obs_dict[enc.cue_key] = cue
        self._guide_cond = None
        if routed and self.guidance != 1.0:
            assert self.obs_as_global_cond, "guidance is implemented for global conditioning"
            nobs = self.normalizer.normalize(obs_dict)
            B = next(iter(nobs.values())).shape[0]
            this_nobs = dict_apply(nobs, lambda x: x[:, :self.n_obs_steps, ...].reshape(-1, *x.shape[2:]))
            rule, enc.route_eval = enc.route_eval, "rgb_only"
            self._guide_cond = enc(this_nobs).reshape(B, -1)
            enc.route_eval = rule
        result = super().predict_action(obs_dict)          # runs the encoder with the routing rule: last_w is the routed one
        self._guide_cond = None
        if self.w_trace is not None and self.gated and getattr(enc, "last_w", None) is not None:
            B = next(iter(obs_dict.values())).shape[0]
            w = enc.last_w.view(B, self.n_obs_steps, -1)[:, -1]
            cue = obs_dict[enc.cue_key][:, -1] if enc.cue_key in obs_dict else None
            if not self.w_trace:
                self.w_trace.append([])
            self.w_trace[-1].append({"w": w.cpu().numpy(), "cue": None if cue is None else cue.cpu().numpy()})
        return result
