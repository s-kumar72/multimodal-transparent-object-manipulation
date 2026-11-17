"""
CPU smoke tests for fusion/ (no data, no GPU, no Diffusion Policy needed):  python -m fusion.test_fusion
"""
import numpy as np
import torch

from fusion.data import Augment, SegArrays, collate
from fusion.model import build_model, seg_loss
from fusion.policy_encoder import FusionObsEncoder

SMALL = dict(pretrained_rgb=False, num_layers=1, d_model=64, nhead=4, dim_ff=128, dec_dim=64)


def batch(B=3, S=84):
    return {"rgb": torch.rand(B, 3, S, S), "depth": torch.rand(B, 1, S, S), "thermal": torch.rand(B, 1, S, S)}


def test_shapes_and_missing_modalities():
    m = build_model(SMALL)
    avail = torch.tensor([[1, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=torch.bool)
    out = m(batch(), avail)
    assert out["logits"].shape == (3, 1, 84, 84) and len(out["sides"]) == 5
    assert out["gates"].shape[:2] == (3, 3)
    assert out["gates"][1, 1].abs().max() == 0 and out["gates"][2, 2].abs().max() == 0, "missing modality got weight"
    for a in out["dec_att"]:
        assert a[1, 1].abs().max() == 0 and a[2, 2].abs().max() == 0
    loss = seg_loss(out, torch.zeros(3, 1, 84, 84), torch.ones(3, 1, 84, 84))
    loss.backward()
    assert all(p.grad is not None for n, p in m.named_parameters() if n.startswith("fusion.gate_lin"))


def test_missing_modality_input_is_ignored():
    """Whatever is in an unavailable modality's image should not change the output."""
    m = build_model(SMALL).eval()
    x = batch(1)
    avail = torch.tensor([[1, 1, 0]], dtype=torch.bool)
    y = dict(x, thermal=torch.rand(1, 1, 84, 84))
    with torch.no_grad():
        a, b = m(x, avail)["logits"], m(y, avail)["logits"]
    assert torch.allclose(a, b, atol=1e-5)


def test_gate_modes():
    avail = torch.tensor([[1, 1, 1], [1, 0, 1]], dtype=torch.bool)
    g = build_model(SMALL, gate_norm="softmax")(batch(2), avail)["gates"]
    assert torch.allclose(g.sum(1), torch.ones_like(g.sum(1)), atol=1e-5)
    g = build_model(SMALL, gate_norm="softmax", fusion_mode="weighted_sum", num_layers=2)(batch(2), avail)["gates"]
    assert torch.allclose(g.sum(1), torch.ones_like(g.sum(1)), atol=1e-5) and g[1, 1].abs().max() == 0
    g = build_model(SMALL, gate_mode="static")(batch(2), avail)["gates"]
    assert torch.allclose(g[0], torch.full_like(g[0], 1 / 3)) and torch.allclose(g[1, 0], torch.full_like(g[1, 0], 0.5))


def test_subsets_and_strides():
    for mods in (["rgb"], ["rgb", "thermal"], ["rgb", "depth"]):
        for stride in (16, 32):
            out = build_model(SMALL, modalities=mods, fusion_stride=stride)(batch(2))
            assert out["logits"].shape == (2, 1, 84, 84) and out["gates"].shape[1] == len(mods)


def test_dataset_and_augment():
    n, S = 6, 96
    arr = dict(rgb=np.random.randint(0, 255, (n, S, S, 3), np.uint8), depth=np.random.randint(0, 255, (n, S, S, 1), np.uint8),
               thermal=np.random.randint(0, 255, (n, S, S, 1), np.uint8), mask=np.zeros((n, S, S), np.uint8),
               avail=np.ones((n, 3), bool), is_crossover=np.zeros(n, np.float32), thermal_gap=np.zeros(n, np.float32))
    arr["mask"][:, 40:50, 40:50] = 255
    aug = dict(thermal_crossover=1, depth_holes=1, depth_wrong=1, rgb_lowlight=1, blank=0, blank_rgb=0, drop=0)
    ds = SegArrays(arr, "fake", 84, train=True, aug=aug)
    b = collate([ds[i] for i in range(n)])
    assert b["rgb"].shape == (n, 3, 84, 84) and b["mask"].shape == (n, 1, 84, 84) and b["corrupt"].all()
    assert (b["depth"] == 0).float().mean() > 0.04, "depth holes not applied"
    assert b["mask"].sum() == n * 100, "crop must keep the mask (it is in the centre)"
    ds = SegArrays(arr, "fake", 84, train=True, aug=dict(aug, blank=1, blank_rgb=1, drop=1))
    for i in range(n):   # at least one modality always survives blanking and dropout
        it = ds[i]
        assert any(it[k].abs().sum() > 0 for k in ("rgb", "depth", "thermal")) and it["avail"].sum() >= 1


def test_policy_encoder():
    meta = {"obs": {"rgb": {"shape": [3, 96, 96], "type": "rgb"}, "depth": {"shape": [3, 96, 96], "type": "rgb"},
                    "joints": {"shape": [7], "type": "low_dim"}}}
    enc = FusionObsEncoder(meta, model_cfg=dict(SMALL, modalities=["rgb", "depth", "thermal"]), out_dim=32)
    assert tuple(enc.output_shape()) == (32 + 7,)
    obs = {"rgb": torch.rand(4, 3, 96, 96) * 2 - 1, "depth": torch.rand(4, 3, 96, 96) * 2 - 1, "joints": torch.rand(4, 7)}
    enc.train()
    assert enc(obs).shape == (4, 39)
    assert enc.avail_mask.tolist() == [True, True, False]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)