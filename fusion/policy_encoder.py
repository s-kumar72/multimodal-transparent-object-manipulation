"""
Diffusion Policy observation encoder built on the glass-fusion network (drop-in replacement for DP's
MultiImageObsEncoder (one ResNet per modality, features concatenated = "static fusion")).

Per observation frame:
    rgb / depth / thermal ─► GlassFusionNet.encode ─► fused map ẑ = Σ_m w_m ⊙ f_m  (reliability-weighted)
                                                 └─► decoder ─► glass probability map p
    feature = Linear([avgpool(ẑ), Σ p·ẑ / Σ p (glass-attention pooling), centroid of p (x, y), area of p,
                      mean gate of each modality]) ++ joints

So the policy gets a global scene feature, a feature of "the glass the segmenter picked" and where it is, and how
much each sensor is trusted. Optionally initialized from a train_seg.py checkpoint (pretrained=...).

Differences from MultiImageObsEncoder:
  * one random crop shared by all modalities (DP crops each image independently, which misaligns them);
  * DP's normalizer maps images to [-1, 1]; this encoder maps them back to [0, 1] (what the segmenter was trained on);
  * depth / thermal arrive as 3 identical channels (policy/obs_util.py); channel 0 is used;
  * thermal_renorm="minmax" re-applies per-frame min-max to thermal, so sim levels with the calibrated "centered"
    thermal see the same normalization as the segmentation training data (train_seg.py --sim_thermal minmax).

Used from fusion/config/train_glass_fusion_image.yaml; see fusion/README.md.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from fusion.model import MODALITIES, GlassFusionNet, load_checkpoint


class FusionObsEncoder(nn.Module):
    def __init__(self, shape_meta, model_cfg=None, pretrained=None, crop_shape=(84, 84), random_crop=True,
                 out_dim=512, mask_pool=True, thermal_renorm="minmax", freeze_segmenter=False,
                 # MultiImageObsEncoder options left over from the base config; ignored
                 rgb_model=None, resize_shape=None, use_group_norm=None, share_rgb_model=None, imagenet_norm=None):
        super().__init__()
        if hasattr(shape_meta, "items") and not isinstance(shape_meta, dict):   # OmegaConf -> plain containers
            from omegaconf import OmegaConf
            shape_meta = OmegaConf.to_container(shape_meta, resolve=True)
        if model_cfg is not None and not isinstance(model_cfg, dict):
            from omegaconf import OmegaConf
            model_cfg = OmegaConf.to_container(model_cfg, resolve=True)
        obs = shape_meta["obs"]
        self.shape_meta = shape_meta
        self.image_keys = [k for k in MODALITIES if k in obs and obs[k].get("type") == "rgb"]
        assert "rgb" in self.image_keys, "the fusion encoder needs the rgb observation"
        others = [k for k, v in obs.items() if v.get("type") == "rgb" and k not in MODALITIES]
        assert not others, f"image keys {others} are not one of {MODALITIES}"
        self.low_dim_keys = sorted(k for k, v in obs.items() if v.get("type", "low_dim") == "low_dim")
        self.key_shape = {k: tuple(v["shape"]) for k, v in obs.items()}

        if pretrained:
            self.net, _ = load_checkpoint(pretrained)
            print(f"[fusion] policy encoder initialized from {pretrained} (modalities {self.net.mods})")
        else:
            cfg = dict(model_cfg or {})
            cfg.setdefault("modalities", self.image_keys)
            self.net = GlassFusionNet(**cfg)
        self.avail_mask = torch.tensor([m in self.image_keys for m in self.net.mods])
        assert self.avail_mask.any()
        if freeze_segmenter:
            self.net.requires_grad_(False)
        self.crop, self.random_crop = tuple(crop_shape) if crop_shape else None, random_crop
        self.mask_pool, self.thermal_renorm = mask_pool, thermal_renorm
        d, M = self.net.cfg["d_model"], len(self.net.mods)
        in_dim = d + (2 * d + 3 if mask_pool else 0) + M
        self.head = nn.Sequential(nn.Linear(in_dim, out_dim), nn.LayerNorm(out_dim), nn.Mish())
        self.out_dim = out_dim

    # DP's ModuleAttrMixin interface
    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    def _crop(self, imgs):
        """Same crop window for every modality (random per sample in training, center in eval)."""
        if self.crop is None:
            return imgs
        H, W = next(iter(imgs.values())).shape[-2:]
        ch, cw = self.crop
        B = next(iter(imgs.values())).shape[0]
        if self.training and self.random_crop:
            top = torch.randint(0, H - ch + 1, (B,))
            left = torch.randint(0, W - cw + 1, (B,))
            out = {}
            for k, x in imgs.items():
                out[k] = torch.stack([x[i, :, t:t + ch, l:l + cw] for i, (t, l) in enumerate(zip(top, left))])
            return out
        t, l = (H - ch) // 2, (W - cw) // 2
        return {k: x[:, :, t:t + ch, l:l + cw] for k, x in imgs.items()}

    def forward(self, obs_dict):
        imgs = {}
        for k in self.image_keys:
            x = (obs_dict[k] + 1) / 2                     # DP image normalizer: [0, 1] -> [-1, 1]
            if k != "rgb":
                x = x[:, :1]
            if k == "thermal" and self.thermal_renorm == "minmax":
                lo = x.amin((1, 2, 3), keepdim=True)
                hi = x.amax((1, 2, 3), keepdim=True)
                x = (x - lo) / (hi - lo).clamp(min=1e-6)
            imgs[k] = x.clamp(0, 1)
        imgs = self._crop(imgs)
        B, dev = imgs["rgb"].shape[0], imgs["rgb"].device
        for m in self.net.mods:                           # placeholders for modalities the policy does not get
            if m not in imgs:
                imgs[m] = torch.zeros(B, 1, *imgs["rgb"].shape[-2:], device=dev)
        avail = self.avail_mask.to(dev)[None].expand(B, -1)
        enc = self.net.encode(imgs, avail)
        z = enc["fused"]                                  # (B, d, h, w)
        feats = [z.mean((2, 3))]
        if self.mask_pool:
            logits, _, _ = self.net.decode(enc, imgs["rgb"].shape[-2:])
            p = torch.sigmoid(F.adaptive_avg_pool2d(logits, z.shape[-2:]))          # (B, 1, h, w)
            mass = p.sum((2, 3)).clamp(min=1e-4)
            h, w = z.shape[-2:]
            ys = torch.linspace(-1, 1, h, device=dev)[None, None, :, None]
            xs = torch.linspace(-1, 1, w, device=dev)[None, None, None, :]
            feats += [(z * p).sum((2, 3)) / mass, ((p * xs).sum((2, 3)) / mass), ((p * ys).sum((2, 3)) / mass),
                      p.mean((2, 3)), (z * p).amax((2, 3))]
        feats.append(enc["gates"].mean((2, 3)))
        out = [self.head(torch.cat(feats, 1))]
        for k in self.low_dim_keys:
            out.append(obs_dict[k])
        return torch.cat(out, -1)

    @torch.no_grad()
    def output_shape(self):
        ex = {k: torch.zeros((1,) + s, device=self.device, dtype=self.dtype) for k, s in self.key_shape.items()}
        was = self.training
        self.eval()
        shape = self.forward(ex).shape[1:]
        self.train(was)
        return shape