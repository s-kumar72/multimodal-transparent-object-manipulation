"""
Thermal ResNet-18 encoder (Diffusion Policy layout) + light FPN segmentation head.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

STAGE_CHANNELS = (64, 128, 256, 512)


def bn_to_gn(module):
    """BatchNorm2d -> GroupNorm(C // 16, C), the rule in DP's MultiImageObsEncoder(use_group_norm=True)."""
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            setattr(module, name, nn.GroupNorm(child.num_features // 16, child.num_features))
        else:
            bn_to_gn(child)
    return module


def make_thermal_resnet18(init="scratch", in_chans=1):
    """ResNet-18 as DP builds it (get_resnet: fc = Identity, then BN -> GN), with an in_chans conv1.

    init: "scratch" or "imagenet" (conv1 = RGB weights averaged over channels; GN freshly
    initialized, same as DP when it swaps BN for GN on ImageNet weights).
    """
    if init not in ("scratch", "imagenet"):
        raise ValueError(f"unknown init: {init}")
    net = torchvision.models.resnet18(weights="IMAGENET1K_V1" if init == "imagenet" else None)
    rgb_w = net.conv1.weight.data
    net.conv1 = nn.Conv2d(in_chans, 64, kernel_size=7, stride=2, padding=3, bias=False)
    if init == "imagenet":
        net.conv1.weight.data = rgb_w.mean(1, keepdim=True).repeat(1, in_chans, 1, 1) / in_chans
    net.fc = nn.Identity()
    return bn_to_gn(net)


class ThermalSegNet(nn.Module):
    """Encoder stages -> FPN top-down sum -> 1-channel mask logits at input resolution."""

    def __init__(self, encoder, dim=64):
        super().__init__()
        self.encoder = encoder
        self.lateral = nn.ModuleList(nn.Conv2d(c, dim, 1) for c in STAGE_CHANNELS)
        self.fuse = nn.Sequential(nn.Conv2d(dim, dim, 3, padding=1), nn.GroupNorm(8, dim), nn.ReLU(inplace=True))
        self.head = nn.Conv2d(dim, 1, 1)

    def stages(self, x):
        e = self.encoder
        x = e.maxpool(e.relu(e.bn1(e.conv1(x))))
        feats = []
        for layer in (e.layer1, e.layer2, e.layer3, e.layer4):
            x = layer(x)
            feats.append(x)
        return feats

    def forward(self, x):
        feats = self.stages(x)
        y = self.lateral[-1](feats[-1])
        for lat, f in zip(self.lateral[-2::-1], feats[-2::-1]):
            y = F.interpolate(y, size=f.shape[-2:], mode="bilinear", align_corners=False) + lat(f)
        y = self.head(self.fuse(y))
        return F.interpolate(y, size=x.shape[-2:], mode="bilinear", align_corners=False)


def seg_loss(logits, target, eps=1.0):
    """BCE + soft Dice (per image)."""
    bce = F.binary_cross_entropy_with_logits(logits, target)
    p, t = logits.sigmoid().flatten(1), target.flatten(1)
    dice = 1 - (2 * (p * t).sum(1) + eps) / (p.sum(1) + t.sum(1) + eps)
    return bce + dice.mean()
