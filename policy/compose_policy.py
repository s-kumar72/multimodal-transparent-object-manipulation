"""
Composition of separately trained Diffusion Policies at the noise-prediction level (a baseline, inference only).

Each member is a complete policy with its own observation encoder, e.g. one trained on rgb + depth and one on
rgb + thermal. At every denoising step the members predict the noise of the same noisy action sequence and the
predictions are combined with fixed weights,

    eps = sum_i w_i eps_i(x_t, t, obs_i),

and the scheduler takes its step with the combined prediction. This is the distribution-level composition of
Modality-Composable Diffusion Policy (arXiv 2503.12466) and of the policies of different sensors in PoCo (RSS 2024).
  weights summing to 1    a weighted mean of the members' scores, e.g. (0.5, 0.5) over (rgbd, rgbt)
  (1, 1, -1)              over (rgbd, rgbt, rgb): the product of the two sensor policies divided by their common rgb
                          prior, p(a | rgb, d) p(a | rgb, t) / p(a | rgb)
The members must share the action space, horizon, observation steps, scheduler and action normalizer (same demos);
this is checked. Nothing is trained and no member is changed.
"""
import torch
import torch.nn as nn

from diffusion_policy.common.pytorch_util import dict_apply


class ComposedPolicy(nn.Module):
    def __init__(self, members, weights, obs_keys):
        """members: DiffusionUnetImagePolicy objects; weights: one float each; obs_keys: for each member, the
        observation keys it was trained on (the keys of its shape_meta["obs"])."""
        super().__init__()
        assert len(members) == len(weights) == len(obs_keys) and len(members) >= 2
        self.members = nn.ModuleList(members)
        self.weights = [float(w) for w in weights]
        self.obs_keys = [list(k) for k in obs_keys]
        first = members[0]
        for m in members:
            assert m.obs_as_global_cond, "composition is implemented for global conditioning"
            for attr in ("horizon", "action_dim", "n_action_steps", "n_obs_steps", "num_inference_steps"):
                assert getattr(m, attr) == getattr(first, attr), attr
            assert m.noise_scheduler.config == first.noise_scheduler.config, "members use different schedulers"
            for name in ("scale", "offset"):
                a = m.normalizer["action"].params_dict[name]
                b = first.normalizer["action"].params_dict[name]
                assert torch.allclose(a, b), "members have different action normalizers"

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    def reset(self):
        for m in self.members:
            m.reset()

    @torch.no_grad()
    def predict_action(self, obs_dict):
        first = self.members[0]
        To, T, Da = first.n_obs_steps, first.horizon, first.action_dim
        conds = []
        for m, keys in zip(self.members, self.obs_keys):
            nobs = m.normalizer.normalize({k: obs_dict[k] for k in keys})
            B = next(iter(nobs.values())).shape[0]
            this_nobs = dict_apply(nobs, lambda x: x[:, :To, ...].reshape(-1, *x.shape[2:]))
            conds.append(m.obs_encoder(this_nobs).reshape(B, -1))
        scheduler = first.noise_scheduler
        trajectory = torch.randn(size=(B, T, Da), dtype=self.dtype, device=self.device)
        scheduler.set_timesteps(first.num_inference_steps)
        for t in scheduler.timesteps:
            eps = sum(w * m.model(trajectory, t, local_cond=None, global_cond=c)
                      for w, m, c in zip(self.weights, self.members, conds))
            trajectory = scheduler.step(eps, t, trajectory, **first.kwargs).prev_sample
        action_pred = first.normalizer["action"].unnormalize(trajectory[..., :Da])
        start = To - 1
        return {"action": action_pred[:, start:start + first.n_action_steps], "action_pred": action_pred}
