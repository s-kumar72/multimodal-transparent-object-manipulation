"""
A second policy on the benchmark's observation encoders: the action head of ACT (Zhao et al., "Learning Fine-Grained
Bimanual Manipulation with Low-Cost Hardware", RSS 2023) in place of Diffusion Policy's denoiser. No diffusion: the
policy regresses a chunk of actions.

    observation steps ──► obs_encoder (the benchmark's: a ResNet-18 per sensor, [the cue], the joints) ──► a token per step ─┐
    training:  [CLS ; joints ; action chunk] ──► transformer encoder ──► q(z | joints, actions) ──► z ──► latent token ──────┤
    inference: z = 0                                                                                                        ▼
                         transformer encoder ──► memory ──► transformer decoder (one learned query per step of the chunk)
                                                            ──► linear ──► action chunk (horizon steps)
    loss = L1(chunk, demonstrated chunk) + kl_weight * KL(q || N(0, I))

ACT's are: the conditional VAE over the action chunk (posterior from the joints and the chunk, prior mean at
inference), the transformer encoder-decoder with one learned query per action, the L1 loss, and the sizes (512 wide, 8
heads, 4 encoder layers, feed-forward 3200, dropout 0.1, latent 32, KL weight 10; one decoder layer, as the original
code effectively uses).

What differs from ACT, because the benchmark fixes it for every policy:
  * the observation tokens are the benchmark's encoder outputs, one per observation step (pooled ResNet-18 features of
    every sensor, the joints, and the cue when the encoder has it), not ACT's feature-map tokens of each camera. The
    cue therefore reaches this head the way it reaches the denoiser of Diffusion Policy: in the conditioning vector;
  * the chunk is the benchmark's horizon (16 steps, 8 executed, then a new chunk), not 100 steps with temporal
    ensembling: the env runner and the dataset windows are those of every other policy;
  * optimizer, schedule, batch size, EMA and number of epochs are the benchmark's.

Works with diffusion_policy's TrainDiffusionUnetImageWorkspace (compute_loss, predict_action, set_normalizer); the
diffusion settings of the parent config are accepted and not used.
After each compute_loss: last_l1, last_kl (of the batch) for logging.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.base_image_policy import BaseImagePolicy

JOINTS_KEY = "joints"


class ActChunkImagePolicy(BaseImagePolicy):
    def __init__(self, shape_meta, obs_encoder, horizon, n_action_steps, n_obs_steps, d_model=512, n_heads=8,
                 n_encoder_layers=4, n_decoder_layers=1, n_vae_encoder_layers=4, dim_feedforward=3200, dropout=0.1,
                 latent_dim=32, kl_weight=10.0, **diffusion_settings_not_used):
        super().__init__()
        action_shape = shape_meta["action"]["shape"]
        assert len(action_shape) == 1
        self.action_dim = action_shape[0]
        self.joint_dim = shape_meta["obs"][JOINTS_KEY]["shape"][0]
        self.obs_feature_dim = obs_encoder.output_shape()[0]
        self.obs_encoder = obs_encoder
        self.normalizer = LinearNormalizer()
        self.horizon, self.n_action_steps, self.n_obs_steps = int(horizon), int(n_action_steps), int(n_obs_steps)
        self.latent_dim, self.kl_weight = int(latent_dim), float(kl_weight)

        def encoder(n_layers):
            layer = nn.TransformerEncoderLayer(d_model, n_heads, dim_feedforward, dropout, batch_first=True)
            return nn.TransformerEncoder(layer, n_layers)

        # posterior q(z | joints, action chunk)
        self.vae_cls = nn.Parameter(torch.zeros(1, 1, d_model))
        self.vae_joints = nn.Linear(self.joint_dim, d_model)
        self.vae_action = nn.Linear(self.action_dim, d_model)
        self.vae_pos = nn.Parameter(torch.zeros(1, 2 + self.horizon, d_model))
        self.vae_encoder = encoder(n_vae_encoder_layers)
        self.vae_out = nn.Linear(d_model, 2 * self.latent_dim)

        # the policy: [latent ; observation steps] -> memory -> action chunk
        self.latent_in = nn.Linear(self.latent_dim, d_model)
        self.obs_in = nn.Linear(self.obs_feature_dim, d_model)
        self.enc_pos = nn.Parameter(torch.zeros(1, 1 + self.n_obs_steps, d_model))
        self.encoder = encoder(n_encoder_layers)
        layer = nn.TransformerDecoderLayer(d_model, n_heads, dim_feedforward, dropout, batch_first=True)
        self.decoder = nn.TransformerDecoder(layer, n_decoder_layers, norm=nn.LayerNorm(d_model))
        self.query = nn.Parameter(torch.zeros(1, self.horizon, d_model))
        self.head = nn.Linear(d_model, self.action_dim)
        for p in (self.vae_cls, self.vae_pos, self.enc_pos, self.query):
            nn.init.normal_(p, std=0.02)
        self.last_l1 = self.last_kl = None

    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def _observe(self, nobs):
        """Normalized observations (B, steps, ...) -> tokens (B, n_obs_steps, obs_feature_dim), joints of the last step."""
        to = self.n_obs_steps
        b = next(iter(nobs.values())).shape[0]
        this = dict_apply(nobs, lambda x: x[:, :to, ...].reshape(-1, *x.shape[2:]))
        return self.obs_encoder(this).reshape(b, to, -1), nobs[JOINTS_KEY][:, to - 1]

    def _posterior(self, joints, naction):
        b = naction.shape[0]
        x = torch.cat([self.vae_cls.expand(b, -1, -1), self.vae_joints(joints)[:, None], self.vae_action(naction)], 1)
        mu, logvar = self.vae_out(self.vae_encoder(x + self.vae_pos)[:, 0]).chunk(2, -1)
        return mu, logvar

    def _decode(self, feat, z):
        b = feat.shape[0]
        x = torch.cat([self.latent_in(z)[:, None], self.obs_in(feat)], 1) + self.enc_pos
        memory = self.encoder(x)
        return self.head(self.decoder(self.query.expand(b, -1, -1), memory))          # (B, horizon, action_dim)

    def predict_action(self, obs_dict):
        assert "past_action" not in obs_dict
        feat, _ = self._observe(self.normalizer.normalize(obs_dict))
        z = torch.zeros(feat.shape[0], self.latent_dim, device=feat.device, dtype=feat.dtype)     # the prior's mean
        action_pred = self.normalizer["action"].unnormalize(self._decode(feat, z))
        start = self.n_obs_steps - 1
        return {"action": action_pred[:, start:start + self.n_action_steps], "action_pred": action_pred}

    def compute_loss(self, batch):
        assert "valid_mask" not in batch
        nobs = self.normalizer.normalize(batch["obs"])
        naction = self.normalizer["action"].normalize(batch["action"])
        assert naction.shape[1] == self.horizon, (naction.shape, self.horizon)
        feat, joints = self._observe(nobs)
        mu, logvar = self._posterior(joints, naction)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        l1 = F.l1_loss(self._decode(feat, z), naction)
        kl = (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp())).sum(-1).mean()
        self.last_l1, self.last_kl = l1.detach(), kl.detach()
        return l1 + self.kl_weight * kl
