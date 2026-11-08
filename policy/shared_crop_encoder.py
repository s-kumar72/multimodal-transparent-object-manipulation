"""
Diffusion Policy's MultiImageObsEncoder (one ResNet per image key, features concatenated) with two fixes for
multi-sensor input:

* one random crop shared by all image keys. DP gives every key its own CropRandomizer, so RGB, depth and thermal of the
  same frame are cropped at different offsets (up to 12 px of 96) and no longer line up;
* per-key initial weights (`init_weights`): "imagenet" or the path of a ResNet state dict, e.g. the TRansPose-pretrained
  thermal encoder from the encoder/thermal_pretrain branch. DP's `rgb_model.weights` applies one choice to all keys.

It also accepts `force_drop`, a set of image keys replaced by zeros at inference (a sensor that is switched off), used
by policy/eval.py --drop. Everything else (GroupNorm, ImageNet normalization, low-dim keys appended) is DP's.
Config: `_target_: policy.shared_crop_encoder.SharedCropObsEncoder` in place of MultiImageObsEncoder, same arguments.
"""
import torch
import torch.nn as nn
import torchvision

from diffusion_policy.model.vision.multi_image_obs_encoder import MultiImageObsEncoder


class SharedCropObsEncoder(MultiImageObsEncoder):
    def __init__(self, shape_meta, rgb_model, resize_shape=None, crop_shape=None, random_crop=True,
                 use_group_norm=False, share_rgb_model=False, imagenet_norm=False, init_weights=None):
        assert resize_shape is None, "shared crop is implemented for the raw 96x96 frames (resize_shape=None)"
        # No per-key CropRandomizer: the crop happens once in forward, before DP's per-key transforms
        super().__init__(shape_meta, rgb_model, resize_shape=None, crop_shape=None, random_crop=False,
                         use_group_norm=use_group_norm, share_rgb_model=share_rgb_model, imagenet_norm=imagenet_norm)
        self.crop_shape = tuple(crop_shape) if crop_shape is not None else None
        self.random_crop = random_crop
        if self.crop_shape is not None:
            for k in self.rgb_keys:   # DP's forward checks the image shape against this map after our crop
                self.key_shape_map[k] = (self.key_shape_map[k][0],) + self.crop_shape
        self.force_drop = set()
        if init_weights:
            self.load_init_weights(dict(init_weights))

    # ------------------------------------------------------------------ crop
    def shared_crop(self, imgs):
        """imgs: {key: (B, C, H, W)} -> same crop window for every key (random per sample in training, else center)."""
        if self.crop_shape is None:
            return imgs
        first = next(iter(imgs.values()))
        B, H, W = first.shape[0], first.shape[-2], first.shape[-1]
        ch, cw = self.crop_shape
        if self.training and self.random_crop:
            top = torch.randint(0, H - ch + 1, (B,), device=first.device)
            left = torch.randint(0, W - cw + 1, (B,), device=first.device)
            rows = top[:, None] + torch.arange(ch, device=first.device)[None]      # (B, ch)
            cols = left[:, None] + torch.arange(cw, device=first.device)[None]     # (B, cw)
            out = {}
            for k, x in imgs.items():
                x = x.gather(2, rows[:, None, :, None].expand(B, x.shape[1], ch, W))
                out[k] = x.gather(3, cols[:, None, None, :].expand(B, x.shape[1], ch, cw))
            return out
        t, l = (H - ch) // 2, (W - cw) // 2
        return {k: x[:, :, t:t + ch, l:l + cw] for k, x in imgs.items()}

    def _prepare(self, obs_dict):
        obs = dict(obs_dict)
        imgs = {k: obs[k] for k in self.rgb_keys}
        for k in self.force_drop:
            if k in imgs:
                imgs[k] = torch.zeros_like(imgs[k])
        obs.update(self.shared_crop(imgs))
        return obs

    def forward(self, obs_dict):
        return super().forward(self._prepare(obs_dict))

    # ------------------------------------------------------------------ weights
    def load_init_weights(self, spec):
        """spec: {image key: "imagenet" | path of a state dict}. BatchNorm statistics are dropped (the encoders use
        GroupNorm); a 1-channel conv1 is spread over the 3 input channels and a 3-channel one summed to 1."""
        for key, src in spec.items():
            if src is None or key not in self.key_model_map:
                continue
            model = self.key_model_map[key]
            if src == "imagenet":
                sd = _imagenet_state_dict(model)
            else:
                sd = torch.load(src, map_location="cpu", weights_only=False)
                if isinstance(sd, dict):
                    for k in ("state_dict", "model", "encoder"):
                        if k in sd and isinstance(sd[k], dict):
                            sd = sd[k]
                            break
                sd = {_strip(k): v for k, v in sd.items()}
            sd = {k: v for k, v in sd.items() if not k.startswith("fc.")
                  and not k.endswith(("running_mean", "running_var", "num_batches_tracked"))}
            if "conv1.weight" in sd:
                w, want = sd["conv1.weight"], model.conv1.weight.shape[1]
                if w.shape[1] == 1 and want == 3:
                    sd["conv1.weight"] = w.repeat(1, 3, 1, 1) / 3
                elif w.shape[1] == 3 and want == 1:
                    sd["conv1.weight"] = w.sum(1, keepdim=True)
            missing, unexpected = model.load_state_dict(sd, strict=False)
            print(f"[encoder] {key} <- {src}: loaded {len(sd) - len(unexpected)} tensors, "
                  f"missing {len(missing)}, unexpected {len(unexpected)}", flush=True)


def _strip(k):
    for p in ("module.", "encoder.", "backbone.", "model."):
        if k.startswith(p):
            k = k[len(p):]
    return k


def _imagenet_state_dict(model):
    """ImageNet weights of the torchvision ResNet the given (fc-less) model was built from."""
    blocks = tuple(len(getattr(model, f"layer{i}")) for i in range(1, 5))
    bottleneck = hasattr(model.layer1[0], "conv3")
    name = {(2, 2, 2, 2): "resnet18", (3, 4, 6, 3): "resnet50" if bottleneck else "resnet34"}[blocks]
    return getattr(torchvision.models, name)(weights="IMAGENET1K_V1").state_dict()
