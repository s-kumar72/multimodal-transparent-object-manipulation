"""
Reliability-aware RGB-D-T glass segmentation using learned attention weighting, based on Huo et al., "Glass segmentation with RGB-Thermal Image Pairs".
(IEEE TIP 2023, https://github.com/Dong-Huo/RGB-T-Glass-Segmentation), generalized from two modalities to any subset of {rgb, depth, thermal}.

rgb ─► ResNet ─┐                                   ┌─► gates w_m (B, M, h, w)   "how much to trust modality m"
  depth ─► ResNet ─┼─► tokens ─► FusionEncoder (xL) ───┼─► fused = Σ_m w_m ⊙ f_m   (ẑ = Σ_m w_m z_m from project proposal)
thermal ─► ResNet ─┘                                   └─► Decoder: 4 stages; each adds every modality's skip feature
                                                           weighted by its own spatial attention map ─► glass mask

Per fusion layer i (Huo et al. Section III-B, written for M modalities):
    f_m   = SelfAttn_m(f_m)                         intra-modality, non-local context      (paper: top/bottom branch)
    f_all = JointAttn([f_1; ...; f_M])              inter-modality, all M*h*w tokens        (paper: middle branch)
    w     = sigmoid(Linear(GateAttn(f_all)))        one weight per token = per modality and position
    f_m   = f_m + (w ⊙ f_all)[m]                    gated residual update

fused = Σ_m (w ⊙ f_all)[m] after the last layer.

Changes from Huo et al. (can be switched off):
1. M modalities instead of 2; missing modalities are masked out of attention and get w = 0, so one model trains on RGB-T, RGB-D, and RGB-D-T data together.
2. reliability cues - per-token cues (local mean / std of the input, fraction of valid pixels) are added to each modality's tokens so gate can see them directly.
3. gate_norm = "softmax": weights sum to 1 over modalities at each position (convex fusion) - Huo et al. uses sigmoid.
4. gate_logit_max: gate logits are bounded (tanh) so a gate can't saturate at 0 or 1 and stop learning. Without this, short runs could collapse to one modality (winner takes all) in every gate mode.
5. fusion_mode="weighted_sum": joint attention only computes the weights, features are not mixed across modalities.
6. gate_reliability_loss (train_seg.py --gate_sup): gates are told which sensor is unreliable when that is known (blanked sensor, removed thermal contrast, sim crossover frames).
7. gate_mode="static": no gates, fused = mean_m f_m. (static-fusion baseline of project proposal, same network otherwise).
8. GroupNorm instead of BatchNorm (Diffusion Policy uses EMA, which breaks with BatchNorm), ResNet-18 by default (Diffusion Policy's encoder; Huo et al. uses ResNet50), sine position encoding & learned modality embeddings, no edge branch (the code from Huo et al. computes the edge loss but never adds it to the total).
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

MODALITIES = ("rgb", "depth", "thermal")
IN_CH = {"rgb": 3, "depth": 1, "thermal": 1}
NORM = {"rgb": ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)), "depth": ((0.5,), (0.25,)), "thermal": ((0.5,), (0.25,))}
STAGE_CH = {"resnet18": (64, 128, 256, 512), "resnet34": (64, 128, 256, 512), "resnet50": (256, 512, 1024, 2048)}

DEFAULT_CFG = dict(
    modalities=list(MODALITIES),
    backbone="resnet18",        # Select from resnet18, resnet34, or resnet50
    pretrained_rgb=True,        # ImageNet weights for the RGB encoder
    pretrained_other=False,     # ImageNet weights (conv1 averaged to 1 channel) for depth / thermal
    fusion_stride=16,           # 16: last ResNet stage dilated (Huo et al.'s DC5 option), 32: plain ResNet
    d_model=256, nhead=8, num_layers=4, dim_ff=1024, dropout=0.1,  # paper: 256 / 8 / 4 / 2048 / 0.1
    fusion_mode="paper",        # paper | weighted_sum (see FusionEncoder)
    gate_mode="learned",        # learned | static
    gate_norm="sigmoid",        # sigmoid (paper) | softmax (over modalities)
    gate_logit_max=3.0,         # gate logits squashed to +-this (tanh) so no gate saturates and dies; None = paper
    reliability_cues=True,
    dec_dim=256,
)

def make_backbone(name, in_ch, pretrained, dilate_last):
    """
    torchvision ResNet without avgpool / fc, BatchNorm -> GroupNorm(C // 16) like Diffusion Policy's encoder; DP-format ResNet-18 state dict (e.g. TRansPose-pretrained thermal encoder loads into it.)
    """

    net = getattr(torchvision.models, name)(weights="IMAGENET1K_V1" if pretrained else None)
    if in_ch != 3:
        w = net.conv1.weight.data
        net.conv1 = nn.Conv2d(in_ch, 64, 7, 2, 3, bias=False)
        if pretrained:
            net.conv1.weight.data = w.mean(1, keepdim=True).repeat(1, in_ch, 1, 1)

    if dilate_last:
        first = True
        for block in net.layer4:
            for m in block.modules():
                if isinstance(m, nn.Conv2d) and m.stride == (2, 2):
                    m.stride = (1, 1)
                if isinstance(m, nn.Conv2d) and m.kernel_size == (3, 3):
                    if first:
                        first = False
                    else:
                        m.dilation, m.padding = (2, 2), (2, 2)

    net.fc, net.avgpool = nn.Identity(), nn.Identity()
    return _bn_to_gn(net)

def _bn_to_gn(module):
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            setattr(module, name, nn.GroupNorm(max(child.num_features // 16, 1), child.num_features))
        else:
            _bn_to_gn(child)
    return module

def backbone_stages(net, x):
    x = net.maxpool(net.relu(net.bn1(net.conv1(x))))
    feats = []
    for layer in (net.layer1, net.layer2, net.layer3, net.layer4):
        x = layer(x)
        feats.append(x)
    return feats

def sine_pos(h, w, dim, device):
    """DETR 2D sine position encoding."""
    d = dim // 4
    y, x = torch.meshgrid(torch.arange(h, device=device, dtype=torch.float32),
                          torch.arange(w, device=device, dtype=torch.float32), indexing="ij")
    y, x = (y + 0.5) / h * 2 * math.pi, (x + 0.5) / w * 2 * math.pi
    t = 10000 ** (torch.arange(d, device=device, dtype=torch.float32) / d)
    py, px = y.flatten()[:, None] / t, x.flatten()[:, None] / t
    return torch.cat([py.sin(), py.cos(), px.sin(), px.cos()], 1)

class AttnLayer(nn.Module):
    """DETR-style post-norm transformer encoder layer, position added to queries and keys only."""
 
    def __init__(self, d, nhead, dim_ff, dropout):
        super().__init__()
        self.attn = nn.MultiheadAttention(d, nhead, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(d, dim_ff), nn.ReLU(inplace=True), nn.Dropout(dropout), nn.Linear(dim_ff, d))
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.drop = nn.Dropout(dropout)
 
    def forward(self, x, pos, pad=None):
        q = x + pos
        x = self.n1(x + self.drop(self.attn(q, q, x, key_padding_mask=pad, need_weights=False)[0]))
        return self.n2(x + self.drop(self.ff(x)))

class FusionEncoder(nn.Module):
    """Huo et al.'s multi-modal fusion module (Eq. 1-4) for M modalities.
 
    fusion_mode="paper": tokens of each modality are updated with gated joint-attention outputs, which already mix all
        modalities, so a low gate on thermal does not mean thermal is unused (its tokens are still attention keys).
    fusion_mode="weighted_sum": ẑ = Σ_m w_m z_m. Each modality's features only go through its
        own self-attention; the joint attention over all modalities is used only to compute the weights. So w_m is
        how much modality m contributes at each position (best with gate_norm="softmax")."""
 
    def __init__(self, n_mod, d, nhead, num_layers, dim_ff, dropout, gate_mode, gate_norm, fusion_mode="paper",
                 gate_logit_max=None):
        super().__init__()
        self.gate_logit_max = gate_logit_max
        self.fusion_mode = fusion_mode
        mk = lambda: AttnLayer(d, nhead, dim_ff, dropout)
        self.intra = nn.ModuleList(nn.ModuleList(mk() for _ in range(n_mod)) for _ in range(num_layers))
        self.joint = nn.ModuleList(mk() for _ in range(num_layers))
        self.gate_attn = nn.ModuleList(mk() for _ in range(num_layers)) if gate_mode == "learned" else None
        self.gate_lin = nn.ModuleList(nn.Linear(d, 1) for _ in range(num_layers)) if gate_mode == "learned" else None
        self.n_mod, self.gate_mode, self.gate_norm = n_mod, gate_mode, gate_norm
 
    def _gates(self, i, f_all, pos_all, pad, avail, B, n):
        if self.gate_mode == "static":   # equal weights over the available modalities
            w = avail.float()[:, :, None].expand(B, self.n_mod, n)
            return (w / w.sum(1, keepdim=True).clamp(min=1)).reshape(B, -1, 1)
        logit = self.gate_lin[i](self.gate_attn[i](f_all, pos_all, pad)).reshape(B, self.n_mod, n)
        if self.gate_logit_max:   
            logit = self.gate_logit_max * torch.tanh(logit / self.gate_logit_max)
        self.last_logit = logit                   
        if self.gate_norm == "softmax":
            w = logit.masked_fill(~avail[:, :, None], float("-inf")).softmax(1)
        else:
            w = torch.sigmoid(logit) * avail[:, :, None]
        return w.reshape(B, -1, 1)
 
    def forward(self, feats, pos, avail):
        """feats: list of M tensors (B, n, d); pos: list of M (n, d) or (B, n, d); avail: (B, M) bool.
        Returns fused (B, n, d) and the last layer's gates (B, M, n)."""
        B, n, _ = feats[0].shape
        pad = (~avail)[:, :, None].expand(B, self.n_mod, n).reshape(B, -1)   
        pos_all = torch.cat([p.expand(B, -1, -1) if p.dim() == 3 else p[None].expand(B, -1, -1) for p in pos], 1)
        for i in range(len(self.joint)):
            feats = [layer(f, p if p.dim() == 3 else p[None], None) for layer, f, p in zip(self.intra[i], feats, pos)]
            if self.fusion_mode == "weighted_sum" and i < len(self.joint) - 1:
                continue   
            f_all = self.joint[i](torch.cat(feats, 1), pos_all, pad)
            w = self._gates(i, f_all, pos_all, pad, avail, B, n)
            if self.fusion_mode == "paper":
                upd = (w * f_all).split(n, 1)
                feats = [f + u for f, u in zip(feats, upd)]
            else:   
                upd = (w * torch.cat(feats, 1)).split(n, 1)
        fused = torch.stack(upd, 0).sum(0)
        return fused, w.reshape(B, self.n_mod, n)
 
 
class SpatialAttention(nn.Module):
    """Decoder fusion (paper's PixelAttention, M modalities): x + Σ_m σ(conv([x, s_m])) ⊙ s_m."""
 
    def __init__(self, ch, n_mod):
        super().__init__()
        self.att = nn.ModuleList(nn.Sequential(
            nn.Conv2d(2 * ch, ch, 3, padding=1), nn.GroupNorm(8, ch), nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 3, padding=1), nn.GroupNorm(8, ch), nn.ReLU(inplace=True),
            nn.Conv2d(ch, 1, 1)) for _ in range(n_mod))
 
    def forward(self, x, skips, avail):
        maps = []
        for att, s, a in zip(self.att, skips, avail.unbind(1)):
            m = torch.sigmoid(att(torch.cat([x, s], 1))) * a[:, None, None, None].float()
            x = x + s * m
            maps.append(m)
        return x, torch.cat(maps, 1)
 
 
def conv_gn(i, o, k=3):
    return nn.Sequential(nn.Conv2d(i, o, k, padding=k // 2), nn.GroupNorm(min(8, o // 4), o), nn.ReLU(inplace=True))

class GlassFusionNet(nn.Module):
    def __init__(self, **cfg):
        super().__init__()
        unknown = set(cfg) - set(DEFAULT_CFG)
        assert not unknown, f"unknown model options {unknown}"
        cfg = {**DEFAULT_CFG, **cfg}
        cfg["modalities"] = list(cfg["modalities"])
        assert cfg["modalities"] and set(cfg["modalities"]) <= set(MODALITIES), cfg["modalities"]
        assert cfg["fusion_stride"] in (16, 32) and cfg["gate_mode"] in ("learned", "static")
        assert cfg["gate_norm"] in ("sigmoid", "softmax") and cfg["fusion_mode"] in ("paper", "weighted_sum")
        self.cfg = cfg
        self.mods = cfg["modalities"]
        M, d, dd = len(self.mods), cfg["d_model"], cfg["dec_dim"]
        ch = STAGE_CH[cfg["backbone"]]
        self.encoders = nn.ModuleDict({m: make_backbone(
            cfg["backbone"], IN_CH[m], cfg["pretrained_rgb"] if m == "rgb" else cfg["pretrained_other"],
            cfg["fusion_stride"] == 16) for m in self.mods})
        for m in self.mods:
            mean, std = NORM[m]
            self.register_buffer(f"mean_{m}", torch.tensor(mean).view(1, -1, 1, 1), persistent=False)
            self.register_buffer(f"std_{m}", torch.tensor(std).view(1, -1, 1, 1), persistent=False)
        self.proj = nn.ModuleDict({m: nn.Conv2d(ch[3], d, 1) for m in self.mods})
        self.mod_embed = nn.Parameter(torch.randn(M, d) * 0.02)
        self.cue = nn.ModuleDict({m: nn.Sequential(nn.Linear(3, d), nn.ReLU(inplace=True), nn.Linear(d, d))
                                  for m in self.mods}) if cfg["reliability_cues"] else None
        self.fusion = FusionEncoder(M, d, cfg["nhead"], cfg["num_layers"], cfg["dim_ff"], cfg["dropout"],
                                    cfg["gate_mode"], cfg["gate_norm"], cfg["fusion_mode"], cfg["gate_logit_max"])
        dims = [dd, dd, dd // 2, dd // 4, dd // 8, dd // 16]
        self.dec_in = conv_gn(d, dims[0], 1) if d != dims[0] else nn.Identity()
        self.sa = nn.ModuleList(SpatialAttention(c, M) for c in [dims[0], dims[1], dims[2], dims[3]])
        self.skip = nn.ModuleList(nn.ModuleDict({m: nn.Sequential(nn.Conv2d(c, o, 1), nn.GroupNorm(8, o),
                                                                    nn.ReLU(inplace=True)) for m in self.mods})
                                  for c, o in zip(ch[::-1], [dims[0], dims[1], dims[2], dims[3]]))
        self.stage = nn.ModuleList(nn.Sequential(conv_gn(i, o), conv_gn(o, o)) for i, o in zip(dims[:5], dims[1:6]))
        self.side = nn.ModuleList(nn.Conv2d(o, 1, 3, padding=1) for o in dims[1:6])
        self.final = nn.Conv2d(5, 1, 1)
 
    # ------------------------------------------------------------------ helpers
    def _avail(self, inputs, avail):
        B = inputs[self.mods[0]].shape[0]
        dev = inputs[self.mods[0]].device
        if avail is None:
            return torch.ones(B, len(self.mods), dtype=torch.bool, device=dev)
        return avail.to(dev).bool()
 
    @staticmethod
    def _cues(x, size):
        """(B, h*w, 3): local mean, local std and valid fraction (x > 0) of the [0, 1] input per token."""
        g = x.mean(1, keepdim=True)
        mean = F.adaptive_avg_pool2d(g, size)
        std = (F.adaptive_avg_pool2d(g * g, size) - mean ** 2).clamp(min=0).sqrt()
        valid = F.adaptive_avg_pool2d((x > 0).float().amax(1, keepdim=True), size)
        return torch.cat([mean, 4 * std, valid], 1).flatten(2).transpose(1, 2)
 
    # ------------------------------------------------------------------ forward
    def encode(self, inputs, avail=None):
        """inputs: {modality: (B, C, H, W) in [0, 1]} (extra keys ignored); avail: (B, M) bool or None (= all).
        Returns dict(fused (B, d, h, w), gates (B, M, h, w), skips [per stage: list over modalities], avail)."""
        avail = self._avail(inputs, avail)
        feats, tokens, pos, skips = {}, [], [], None
        for k, m in enumerate(self.mods):
            x = inputs[m]
            xin = (x - getattr(self, f"mean_{m}")) / getattr(self, f"std_{m}")
            st = backbone_stages(self.encoders[m], xin)
            feats[m] = st
            t = self.proj[m](st[3])
            h, w = t.shape[-2:]
            p = sine_pos(h, w, t.shape[1], t.device) + self.mod_embed[k]
            if self.cue is not None:   # cues are data, so they ride on the position term (queries / keys)
                p = p[None] + self.cue[m](self._cues(x, (h, w)))
            tokens.append(t.flatten(2).transpose(1, 2))
            pos.append(p)
        self.fusion.last_logit = None
        fused, gates = self.fusion(tokens, pos, avail)
        gate_logits = self.fusion.last_logit
        B = fused.shape[0]
        fused = fused.transpose(1, 2).reshape(B, -1, h, w)
        skips = [[feats[m][3 - s] for m in self.mods] for s in range(4)]   # deepest first
        if gate_logits is not None:
            gate_logits = gate_logits.reshape(B, -1, h, w)
        return dict(fused=fused, gates=gates.reshape(B, -1, h, w), gate_logits=gate_logits, skips=skips, avail=avail)
 
    def decode(self, enc, out_size):
        x, avail = self.dec_in(enc["fused"]), enc["avail"]
        sides, att = [], []
        for s in range(4):
            sk = [proj(f) for proj, f in zip(self.skip[s].values(), enc["skips"][s])]
            x = F.interpolate(x, size=sk[0].shape[-2:], mode="bilinear", align_corners=False)
            x, a = self.sa[s](x, sk, avail)
            att.append(a)
            x = self.stage[s](x)
            sides.append(self.side[s](x))
        x = F.interpolate(x, size=out_size, mode="bilinear", align_corners=False)
        x = self.stage[4](x)
        sides.append(self.side[4](x))
        sides = [F.interpolate(s, size=out_size, mode="bilinear", align_corners=False) for s in sides]
        logits = self.final(torch.cat(sides, 1))
        return logits, sides, att
 
    def forward(self, inputs, avail=None):
        """Returns dict(logits (B, 1, H, W), sides [5 x (B, 1, H, W)], gates (B, M, h, w),
        dec_att [4 x (B, M, h_s, w_s)], fused (B, d, h, w), avail (B, M))."""
        enc = self.encode(inputs, avail)
        size = inputs[self.mods[0]].shape[-2:]
        logits, sides, att = self.decode(enc, size)
        return dict(logits=logits, sides=sides, gates=enc["gates"], gate_logits=enc["gate_logits"], dec_att=att,
                    fused=enc["fused"],
                    avail=enc["avail"])
 
 
# ---------------------------------------------------------------------- losses
def soft_iou_loss(logits, target, valid):
    p = torch.sigmoid(logits) * valid
    t = target * valid
    inter = (p * t).flatten(1).sum(1)
    union = (p + t - p * t).flatten(1).sum(1)
    return 1 - (inter + 1) / (union + 1)
 
 
def seg_loss(out, target, valid, side_weight=0.5, iou_weight=1.0):
    """BCE (Huo et al.) + soft IoU on the final mask and on every side output (deep supervision, as in Huo et al.).
    target: (B, 1, H, W) in [0, 1] (soft at object borders for sim); valid: (B, 1, H, W) 0 = ignore.
    """
    def one(lg):
        bce = (F.binary_cross_entropy_with_logits(lg, target, reduction="none") * valid).flatten(1).sum(1)
        bce = bce / valid.flatten(1).sum(1).clamp(min=1)
        return bce + iou_weight * soft_iou_loss(lg, target, valid)
    loss = one(out["logits"])
    for s in out["sides"]:
        loss = loss + side_weight * one(s)
    return loss.mean()
 
 
def gate_reliability_loss(out, unreliable, labelled):
    """Optional auxiliary loss that tells the gates which sensor is unreliable when it is known: a blanked sensor, a
    thermal image with its contrast removed (training corruption) or a sim frame in thermal crossover -> target 0;
    an untouched sensor -> 1; partly corrupted ones (depth holes, low light) are not labelled. BCE on the mean gate
    logit over the image, so the logit reads as a reliability score and the softmax / sigmoid of it as the weight.
    unreliable, labelled: (B, M) bool."""
    lg = out.get("gate_logits")
    if lg is None:
        return lg
    lab = labelled.bool() & out["avail"].bool()
    if not lab.any():
        return lg.sum() * 0
    score = lg.float().mean((2, 3))
    return F.binary_cross_entropy_with_logits(score[lab], (~unreliable[lab]).float())
 
 
def build_model(cfg=None, **overrides):
    return GlassFusionNet(**{**(cfg or {}), **overrides})
 
 
def load_checkpoint(path, map_location="cpu", **overrides):
    """Model from a train_seg.py checkpoint (the architecture is stored in it). overrides: e.g. pretrained_rgb=False."""
    ck = torch.load(path, map_location=map_location, weights_only=False)
    cfg = {**ck["model_cfg"], "pretrained_rgb": False, "pretrained_other": False, **overrides}
    model = GlassFusionNet(**cfg)
    model.load_state_dict(ck["model"])
    return model, ck
 
 
def load_encoder_weights(model, modality, path):
    """Load a DP-format ResNet state dict (keys conv1.*, layer1.*, ...; e.g. the TRansPose-pretrained thermal encoder
    from encoder/export.py on the encoder/thermal_pretrain branch) into one modality's encoder."""
    sd = torch.load(path, map_location="cpu", weights_only=False)
    sd = sd.get("model", sd) if isinstance(sd, dict) else sd
    sd = {k.removeprefix("encoder."): v for k, v in sd.items() if not k.startswith(("fc.", "seg", "head", "lateral", "fuse"))}
    enc = model.encoders[modality]
    if sd["conv1.weight"].shape[1] != enc.conv1.weight.shape[1]:   # 3-channel export of a 1-channel input
        sd["conv1.weight"] = sd["conv1.weight"].sum(1, keepdim=True)
    missing, unexpected = enc.load_state_dict(sd, strict=False)
    print(f"[fusion] {modality} encoder <- {path}: {len(sd) - len(unexpected)} tensors, "
          f"missing {len(missing)}, unexpected {len(unexpected)}")
    return model
