"""
Hindsight cue gating, policy part: Diffusion Policy's image encoder (the one of the plain baselines) with the cue.

    rgb ───► ResNet ──► z_rgb ──┐
  depth ───► ResNet ──► z_d ────┼──► gate: z_m or e_miss_m ──► cond = [z_d ; z_rgb ; z_t ; cue ; joints] ──► FiLM
thermal ───► ResNet ──► z_t ────┘          ▲
 first frame ──► per-sensor grasp maps ──► cue = [grasp point, confidence, r_rgb, r_depth, r_thermal]   (policy/hindsight_cue.py)

The cue is the observation key `hcue` (low_dim, hindsight_cue.CUE_DIM): for a training sample it is read from the file
of out-of-fold cues (policy/glass_dataset.py), in an evaluation the env computes it from the first frame of the episode
(policy/glass_gym_env.py); it stays the same for the whole episode. Its last entries are the gate r of rgb / depth /
thermal, left as they are by the normalizer. In a pick-and-place task the key may hold two cues one after the other,
the grasp cue and the place cue (2 x CUE_DIM, task glass_pick_hcue2): both are part of the conditioning, and a sensor's
gate is the larger of its two r, so a sensor stays on when it is trusted for either target.

Options (policy.obs_encoder):
  gate      none | hard | soft | top1. How r acts on the image features of the sensors.
            hard: a sensor with r < gate_thr is switched off, its feature replaced by a learned embedding; the most
            trusted sensor always stays on. top1: only the most trusted sensor is on. soft: the feature is multiplied
            by r. The rule is the same in training and at inference, so each set of sensors is a mode the policy has
            been trained in.
  gate_thr  threshold of gate=hard (0.5: more likely to show the glass than not).
  cue       true: the whole cue vector is part of the conditioning; false: only the on / off weights w are (the policy
            is told which sensors it has, not where the grasp point is).
  images    false: no image features at all, the conditioning is the cue and the joints.
With gate=none, cue=false and no `hcue` key this is the plain encoder.
After each forward: last_w (B, n image keys) for logging.
"""
import torch
import torch.nn as nn

from diffusion_policy.model.vision.multi_image_obs_encoder import MultiImageObsEncoder

from policy.hindsight_cue import CUE_DIM, SENSORS

CUE_KEY = "hcue"


class CueGateObsEncoder(MultiImageObsEncoder):
    def __init__(self, shape_meta, rgb_model, resize_shape=None, crop_shape=None, random_crop=True,
                 use_group_norm=False, share_rgb_model=False, imagenet_norm=False,
                 gate="hard", gate_thr=0.5, cue=True, images=True):
        assert not share_rgb_model, "the gate needs one encoder per sensor"
        assert gate in ("none", "hard", "soft", "top1")
        super().__init__(shape_meta, rgb_model, resize_shape=resize_shape, crop_shape=crop_shape,
                         random_crop=random_crop, use_group_norm=use_group_norm, share_rgb_model=False,
                         imagenet_norm=imagenet_norm)
        self.gate, self.gate_thr, self.cue, self.images = gate, float(gate_thr), bool(cue), bool(images)
        self.has_cue = CUE_KEY in self.low_dim_keys
        assert self.has_cue or (gate == "none" and not cue and images), f"gate / cue need the '{CUE_KEY}' observation"
        if self.has_cue:
            (n,) = self.key_shape_map[CUE_KEY]
            assert n % CUE_DIM == 0 and n > 0, self.key_shape_map[CUE_KEY]
        names = list(SENSORS)
        assert all(k in names for k in self.rgb_keys), "image keys must be rgb / depth / thermal"
        self.r_index = [CUE_DIM - len(names) + names.index(k) for k in self.rgb_keys]    # entry of r for every image key
        if not images:                       # no image encoders in the model (and in the optimizer)
            self.key_model_map = nn.ModuleDict()
        self.e_miss = nn.ParameterDict({k: nn.Parameter(torch.zeros(self._feature_dim(k))) for k in self.rgb_keys}) \
            if images and gate in ("hard", "top1") else None
        self.last_w = None

    @torch.no_grad()
    def _feature_dim(self, key):
        c, h, w = self.key_shape_map[key]
        model = self.key_model_map[key]
        was = model.training
        model.eval()
        d = model(self.key_transform_map[key](torch.zeros(1, c, h, w))).shape[-1]
        model.train(was)
        return d

    def weights(self, cue):
        """(B, n image keys): the weight of every sensor's feature under the gate rule."""
        r = cue.view(len(cue), -1, CUE_DIM)[:, :, self.r_index].amax(1)     # over the cues of the vector
        if self.gate == "none":
            return torch.ones_like(r)
        if self.gate == "soft":
            return r.clamp(0, 1)
        top = torch.zeros_like(r).scatter_(1, r.argmax(1, keepdim=True), 1.0)
        if self.gate == "top1":
            return top
        return torch.maximum((r >= self.gate_thr).to(r.dtype), top)

    def forward(self, obs_dict):
        cue = obs_dict[CUE_KEY] if self.has_cue else None
        B = next(iter(obs_dict.values())).shape[0]
        features, w = [], None
        if self.images:
            if cue is not None:
                w = self.weights(cue)
            for i, key in enumerate(self.rgb_keys):
                img = obs_dict[key]
                assert img.shape[1:] == self.key_shape_map[key], (key, img.shape)
                z = self.key_model_map[key](self.key_transform_map[key](img))
                if w is not None and self.gate != "none":
                    if self.e_miss is not None:
                        z = torch.where(w[:, i:i + 1] > 0.5, z, self.e_miss[key].expand(B, -1).to(z.dtype))
                    else:
                        z = z * w[:, i:i + 1]
                features.append(z)
        for key in self.low_dim_keys:
            data = obs_dict[key]
            assert data.shape[1:] == self.key_shape_map[key], (key, data.shape)
            if key == CUE_KEY:
                if self.cue:
                    features.append(data)
                elif w is not None and self.gate != "none":
                    features.append(w)
                continue
            features.append(data)
        self.last_w = None if w is None else w.detach()
        return torch.cat(features, dim=-1)
