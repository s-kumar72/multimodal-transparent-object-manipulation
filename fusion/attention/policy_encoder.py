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

Readout parts (`readout`, in this order; the default is the original encoder's):
    avg       ẑ averaged over the image                                  (d)
    mask      ẑ averaged under the predicted glass, Σ p·ẑ / Σ p           (d)
    centroid  centre of p on the fusion grid (6x6 for an 84 px crop)     (2)
    area      mean of p                                                  (1)
    max       max over the image of p·ẑ                                  (d)
    gates     mean weight of each sensor                                 (M)
    keypoint  centre, spread and peak of p at full resolution            (5)   ablation: grasp precision
    ssm       spatial softmax of ẑ: K learned keypoints (x, y)           (2K)  ablation: grasp precision
Default: avg mask centroid area max gates (mask_pool=False: avg gates).

Ablation switches (all off by default; fusion/ABLATIONS.md):
    freeze_segmenter      the fusion network is not trained with the policy (frozen, eval mode)
    freeze_steps          frozen for the first N training steps, then trained with the policy
    anchor                keeps a frozen copy of the pretrained network and stores a distillation loss (mask + gates
                          of the copy) in last_anchor; FusionDiffusionPolicy adds it with anchor_weight
    model_overrides       options of fusion/model.py applied when loading `pretrained` (e.g. {gate_mode: static})
After each forward: last_w (B, M) mean gate per sensor, last_gate_logit (B, M) mean gate score, for logging and for
policy/eval.py --save_w.

Differences from MultiImageObsEncoder:
  * one random crop shared by all modalities (DP crops each image independently, which misaligns them);
  * DP's normalizer maps images to [-1, 1]; this encoder maps them back to [0, 1] (what the segmenter was trained on);
  * depth / thermal arrive as 3 identical channels (policy/obs_util.py); channel 0 is used;
  * thermal_renorm="minmax" re-applies per-frame min-max to thermal, so sim levels with the calibrated "centered"
    thermal see the same normalization as the segmentation training data (train_seg.py --sim_thermal minmax).

Used from fusion/config/train_glass_fusion_image.yaml; see fusion/README.md.
"""
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from fusion.attention.model import MODALITIES, GlassFusionNet, load_checkpoint

READOUT_ORDER = ("avg", "mask", "centroid", "area", "max", "gates", "keypoint", "ssm")
DEFAULT_READOUT = ("avg", "mask", "centroid", "area", "max", "gates")
NEEDS_MASK = {"mask", "centroid", "area", "max", "keypoint"}


def _plain(x):
    if x is not None and not isinstance(x, (dict, list, tuple, str, int, float, bool)):
        from omegaconf import OmegaConf
        return OmegaConf.to_container(x, resolve=True)
    return x

class SpatialSoftmax(nn.Module):
    """
    K keypoints of a feature map: 1x1 conv to K channels, softmax over positions, expected (x, y) in [-1, 1].
    """

    def __init__(self, in_ch, k):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, k, 1)
        self.log_temp = nn.Parameter(torch.zeros(()))

    def forward(self, z):
        B, _, h, w = z.shape
        a = (self.conv(z).flatten(2) / self.log_temp.exp()).softmax(-1)          # (B, K, h*w)
        ys = torch.linspace(-1, 1, h, device=z.device).repeat_interleave(w)
        xs = torch.linspace(-1, 1, w, device=z.device).repeat(h)
        return torch.cat([(a * xs).sum(-1), (a * ys).sum(-1)], 1)


class FusionObsEncoder(nn.Module):
    def __init__(self, shape_meta, model_cfg=None, pretrained=None, crop_shape=(84, 84), random_crop=True,
                 out_dim=512, mask_pool=True, thermal_renorm="minmax", freeze_segmenter=False,
                 readout=None, ssm_keypoints=32, freeze_steps=0, anchor=False, model_overrides=None,
                 # MultiImageObsEncoder options left over from the base config; ignored
                 rgb_model=None, resize_shape=None, use_group_norm=None, share_rgb_model=None, imagenet_norm=None):
        super().__init__()
        shape_meta, model_cfg, model_overrides = _plain(shape_meta), _plain(model_cfg), _plain(model_overrides)
        readout = _plain(readout)
        obs = shape_meta["obs"]
        self.shape_meta = shape_meta
        self.image_keys = [k for k in MODALITIES if k in obs and obs[k].get("type") == "rgb"]
        assert "rgb" in self.image_keys, "the fusion encoder needs the rgb observation"
        others = [k for k, v in obs.items() if v.get("type") == "rgb" and k not in MODALITIES]
        assert not others, f"image keys {others} are not one of {MODALITIES}"
        self.low_dim_keys = sorted(k for k, v in obs.items() if v.get("type", "low_dim") == "low_dim")
        self.key_shape = {k: tuple(v["shape"]) for k, v in obs.items()}

        if pretrained:
            self.net, _ = load_checkpoint(pretrained, **(model_overrides or {}))
            print(f"[fusion] policy encoder initialized from {pretrained} (modalities {self.net.mods}"
                 + (f", overrides {model_overrides}" if model_overrides else "") + ")")
        else:
            cfg = {**(model_cfg or {}), **(model_overrides or {})}
            cfg.setdefault("modalities", self.image_keys)
            self.net = GlassFusionNet(**cfg)
        self.avail_mask = torch.tensor([m in self.image_keys for m in self.net.mods])
        assert self.avail_mask.any()

        if readout is None:
            readout = DEFAULT_READOUT if mask_pool else ("avg", "gates")
        unknown = set(readout) - set(READOUT_ORDER)
        assert not unknown, f"unknown readout parts {unknown}; choose from {READOUT_ORDER}"
        self.readout = [r for r in READOUT_ORDER if r in readout]

        d, M = self.net.cfg["d_model"], len(self.net.mods)
        dims = dict(avg=d, mask=d, centroid=2, area=1, max=d, gates=M, keypoint=5, ssm=2 * ssm_keypoints)
        self.ssm = SpatialSoftmax(d, ssm_keypoints) if "ssm" in self.readout else None
        self.head = nn.Sequential(nn.Linear(sum(dims[r] for r in self.readout), out_dim), nn.LayerNorm(out_dim),
                                  nn.Mish())
        self.out_dim = out_dim

        self.freeze_segmenter, self.freeze_steps = bool(freeze_segmenter), int(freeze_steps)
        if self.freeze_segmenter:
            self.net.requires_grad_(False)
        self.register_buffer("train_forwards", torch.zeros((), dtype=torch.long), persistent=True)
        # the frozen copy for the anchor loss lives in a list, so it is not in the state dict and is moved to the right device on first use
        self._teacher = [copy.deepcopy(self.net).eval().requires_grad_(False)] if anchor else None
        assert not (anchor and not pretrained), "anchor needs a pretrained segmentation checkpoint"
        self.crop, self.random_crop = tuple(crop_shape) if crop_shape else None, random_crop
        self.thermal_renorm = thermal_renorm
        self.last_w = self.last_gate_logit = self.last_anchor = None

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        # checkpoints from before the step counter existed (the first L0_bench fusion runs) load unchanged
        state_dict.setdefault(prefix + "train_forwards", self.train_forwards.clone())
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    # DP's ModuleAttrMixin interface
    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    def segmenter_frozen(self):
        return self.freeze_segmenter or (self.training and int(self.train_forwards) < self.freeze_steps)

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

    def _images(self, obs_dict):
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
        return imgs, self.avail_mask.to(dev)[None].expand(B, -1)

    def forward(self, obs_dict):
        imgs, avail = self._images(obs_dict)
        training_step = self.training and torch.is_grad_enabled()
        frozen = self.segmenter_frozen()
        need_mask = bool(NEEDS_MASK & set(self.readout)) or (self._teacher is not None and training_step)
        self.net.train(self.training and not frozen)
        with torch.set_grad_enabled(torch.is_grad_enabled() and not frozen):
            enc = self.net.encode(imgs, avail)
            logits = self.net.decode(enc, imgs["rgb"].shape[-2:])[0] if need_mask else None
        z, gates = enc["fused"], enc["gates"]
        B, dev = z.shape[0], z.device
        self.last_w = gates.mean((2, 3)).detach()
        gl = enc.get("gate_logits")
        self.last_gate_logit = gl.mean((2, 3)) if gl is not None else None

        feats = {"avg": z.mean((2, 3)), "gates": gates.mean((2, 3))}
        if logits is not None:    
            p = torch.sigmoid(F.adaptive_avg_pool2d(logits, z.shape[-2:]))          # (B, 1, h, w)
            mass = p.sum((2, 3)).clamp(min=1e-4)
            h, w = z.shape[-2:]
            ys = torch.linspace(-1, 1, h, device=dev)[None, None, :, None]
            xs = torch.linspace(-1, 1, w, device=dev)[None, None, None, :]
            feats["mask"] = (z * p).sum((2, 3)) / mass
            feats["centroid"] = torch.cat([(p * xs).sum((2, 3)) / mass, (p * ys).sum((2, 3)) / mass], 1)
            feats["area"] = p.mean((2, 3))
            feats["max"] = (z * p).amax((2, 3))
            if "keypoint" in self.readout:                # full resolution: sub-pixel centre, spread and peak
                pf = torch.sigmoid(logits)
                H, W = pf.shape[-2:]
                Y = torch.linspace(-1, 1, H, device=dev)[None, None, :, None]
                X = torch.linspace(-1, 1, W, device=dev)[None, None, None, :]
                mf = pf.sum((2, 3)).clamp(min=1e-4)
                cx, cy = (pf * X).sum((2, 3)) / mf, (pf * Y).sum((2, 3)) / mf
                sx = ((pf * (X - cx[..., None, None]) ** 2).sum((2, 3)) / mf).clamp(min=1e-8).sqrt()
                sy = ((pf * (Y - cy[..., None, None]) ** 2).sum((2, 3)) / mf).clamp(min=1e-8).sqrt()
                feats["keypoint"] = torch.cat([cx, cy, sx, sy, pf.amax((2, 3))], 1)
        if self.ssm is not None:
            feats["ssm"] = self.ssm(z)

        self.last_anchor = None
        if self._teacher is not None and training_step:
            teacher = self._teacher[0]
            if next(teacher.parameters()).device != dev:
                teacher.to(dev)
            with torch.no_grad():
                t_enc = teacher.encode(imgs, avail)
                t_prob = torch.sigmoid(teacher.decode(t_enc, imgs["rgb"].shape[-2:])[0])
            self.last_anchor = F.binary_cross_entropy_with_logits(logits, t_prob) \
                + ((gates - t_enc["gates"]) ** 2).mean() * 10.0
        if training_step:
            self.train_forwards += 1

        out = [self.head(torch.cat([feats[r] for r in self.readout], 1))]
            
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