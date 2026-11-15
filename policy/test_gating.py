"""
CPU checks of the cue-gating pieces (no sim, no data, about a minute):
  PYTHONPATH=.:sim:../diffusion_policy python -m policy.test_gating
"""
import numpy as np
import torch

from policy.glass_dataset import Corruption
from policy.obs_util import CUE_DIM, DEFAULT_ROI, compute_cues, to_policy_obs

SHAPE_META = {"obs": {"rgb": {"shape": [3, 96, 96], "type": "rgb"},
                      "depth": {"shape": [3, 96, 96], "type": "rgb", "mask_channel": True},
                      "thermal": {"shape": [3, 96, 96], "type": "rgb"},
                      "cue": {"shape": [CUE_DIM], "type": "low_dim", "roi": list(DEFAULT_ROI)},
                      "joints": {"shape": [7], "type": "low_dim"}},
              "action": {"shape": [7]}}


def fake_raw(T=2, hot=True, holes=0.4, rng=None):
    """A frame: gray table, a 'glass' blob; thermal_raw in counts around 3000 with noise 2.7."""
    rng = rng or np.random.default_rng(0)
    rgb = np.full((T, 96, 96, 3), 120, np.uint8)
    rgb[:, 50:70, 40:56] = 200
    depth = np.full((T, 96, 96, 1), 150, np.uint8)
    depth[:, 50:70, 40:56][rng.random((T, 20, 16, 1)) < holes] = 0
    raw = rng.normal(3000, 2.7, (T, 96, 96, 1))
    if hot:
        raw[:, 50:70, 40:56] += 300
    thermal_raw = np.clip(np.round(raw), 0, 16383).astype(np.uint16)
    thermal = ((thermal_raw.astype(np.float32) - 2800) / 800 * 255).clip(0, 255).astype(np.uint8)
    return {"rgb": rgb, "depth": depth, "thermal": thermal, "thermal_raw": thermal_raw,
            "joints": np.zeros((T, 7), np.float32), "action": np.zeros((T, 7), np.float32)}


def test_cues():
    hot, flat = compute_cues(fake_raw(hot=True)), compute_cues(fake_raw(hot=False))
    assert hot.shape == (2, CUE_DIM) and np.isfinite(hot).all()
    assert hot[0, 4] > flat[0, 4] + 1.0, (hot[0, 4], flat[0, 4])       # thermal contrast: object vs crossover
    assert flat[0, 4] < 1.0, flat[0, 4]                                 # noise only -> log(1 + ~5 sigma / sigma) ~ 1
    assert hot[0, 5] > flat[0, 5]                                       # extent
    d = compute_cues(fake_raw(holes=0.9))
    assert d[0, 2] < hot[0, 2]                                          # more holes -> lower valid fraction
    print("ok cues: thermal contrast hot %.2f flat %.2f, depth valid %.3f -> %.3f" % (hot[0, 4], flat[0, 4], hot[0, 2], d[0, 2]))


def test_obs():
    obs = to_policy_obs(fake_raw(), SHAPE_META["obs"])
    assert obs["depth"].shape == (2, 3, 96, 96) and obs["cue"].shape == (2, CUE_DIM)
    mask = obs["depth"][:, 1]
    assert set(np.unique(mask)) <= {0.0, 1.0} and (mask == 0).any() and (mask == 1).any()
    assert np.allclose(obs["depth"][:, 0], obs["depth"][:, 2])
    print("ok obs: depth mask channel, cue attached")


def test_corruption():
    rng = np.random.default_rng(1)
    raw = fake_raw(hot=True)
    c = Corruption(thermal_contrast_p=1.0, thermal_contrast=(0.0, 0.0), depth_hole_p=1.0, depth_hole_frac=(0.5, 0.5))
    out = c(raw, rng)
    assert compute_cues(out)[0, 4] < compute_cues(raw)[0, 4] - 1.0      # contrast removed -> cue drops with it
    assert (out["depth"] == 0).mean() > (raw["depth"] == 0).mean() + 0.3
    assert out["depth"].dtype == np.uint8 and out["thermal_raw"].dtype == np.uint16
    print("ok corruption: thermal cue %.2f -> %.2f, depth holes %.2f -> %.2f" % (
        compute_cues(raw)[0, 4], compute_cues(out)[0, 4], (raw["depth"] == 0).mean(), (out["depth"] == 0).mean()))


def _resnet():
    from diffusion_policy.model.vision.model_getter import get_resnet
    return get_resnet("resnet18", weights=None)


def _obs_tensors(B=4):
    raw = fake_raw(T=B)
    obs = to_policy_obs(raw, SHAPE_META["obs"])
    return {k: torch.from_numpy(v) for k, v in obs.items()}


def test_shared_crop():
    from policy.shared_crop_encoder import SharedCropObsEncoder
    enc = SharedCropObsEncoder(SHAPE_META, _resnet(), crop_shape=(84, 84), random_crop=True, use_group_norm=True,
                               imagenet_norm=True)
    enc.train()
    x = torch.arange(96 * 96, dtype=torch.float32).view(1, 1, 96, 96).repeat(3, 3, 1, 1)   # pixel index pattern
    crops = enc.shared_crop({"rgb": x, "depth": x.clone(), "thermal": x.clone()})
    assert all(c.shape == (3, 3, 84, 84) for c in crops.values())
    assert torch.equal(crops["rgb"], crops["depth"]) and torch.equal(crops["rgb"], crops["thermal"])
    assert not torch.equal(crops["rgb"][0], crops["rgb"][1]) or not torch.equal(crops["rgb"][1], crops["rgb"][2])
    out = enc(_obs_tensors())
    assert out.shape == (4, 3 * 512 + CUE_DIM + 7), out.shape        # cue is a plain low_dim here
    enc.eval()
    assert torch.equal(enc.shared_crop({"a": x})["a"], x[:, :, 6:90, 6:90])
    print("ok shared crop: identical windows across modalities, random per sample, center crop in eval")


def test_init_weights():
    from policy.shared_crop_encoder import SharedCropObsEncoder
    import torchvision
    try:
        ref = torchvision.models.resnet18(weights="IMAGENET1K_V1").state_dict()["layer1.0.conv1.weight"]
    except Exception as e:   # no internet for the torchvision download
        print(f"skip init weights ({type(e).__name__}: cannot download ImageNet weights here)")
        return
    enc = SharedCropObsEncoder(SHAPE_META, _resnet(), crop_shape=(84, 84), use_group_norm=True,
                               init_weights={"rgb": "imagenet"})
    assert torch.allclose(enc.key_model_map["rgb"].layer1[0].conv1.weight, ref)
    assert not torch.allclose(enc.key_model_map["depth"].layer1[0].conv1.weight, ref)
    print("ok init weights: imagenet loaded into the rgb encoder only")


def test_gate():
    from policy.gated_obs_encoder import GatedObsEncoder
    B = 4
    enc = GatedObsEncoder(SHAPE_META, _resnet(), crop_shape=(84, 84), use_group_norm=True, imagenet_norm=True,
                          gate_input="cue", dropout_mode="targeted", dropout_p=1.0, dropout_p_other=0.0)
    obs = _obs_tensors(B)
    enc.train()
    out = enc(obs)
    assert out.shape == (B, 3 * 512 + 3 + 7), out.shape               # gated feats + w + joints
    w, drop = enc.last_w, enc.last_drop
    assert w.shape == (B, 3) and (w >= 0).all() and (w <= 1).all()
    assert drop.sum(-1).eq(1).all(), drop                              # p=1: exactly the top modality dropped
    assert (w[drop] == 0).all()
    for i, m in enumerate(enc.mods):                                   # dropped feature = e_miss, block = 0 in cond
        blk = out[:, i * 512:(i + 1) * 512]
        assert torch.allclose(blk[drop[:, i]], torch.zeros_like(blk[drop[:, i]]))
    enc.eval()
    out = enc(obs)
    assert not enc.last_drop.any() and (enc.last_w > 0).all()
    enc.force_drop = {"thermal"}
    enc(obs)
    assert enc.last_drop[:, enc.mods.index("thermal")].all() and (enc.last_w[:, enc.mods.index("thermal")] == 0).all()
    enc.force_drop = set()
    enc2 = GatedObsEncoder(SHAPE_META, _resnet(), crop_shape=(84, 84), use_group_norm=True, gate_act="softmax",
                           gate_input="both", cross_modal_cue=False, w_in_cond=False, cue_in_cond=True,
                           missing="zero", dropout_mode="none").eval()
    out2 = enc2(obs)
    assert out2.shape == (B, 3 * 512 + CUE_DIM + 7) and torch.allclose(enc2.last_w.sum(-1), torch.ones(B))
    print("ok gate: shapes, w in [0,1], targeted dropout, e_miss, force_drop, softmax variant")


def test_policy_loss():
    try:
        from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
    except ImportError:
        print("skip policy loss (diffusers not installed)")
        return
    from diffusion_policy.model.common.normalizer import LinearNormalizer
    from policy.gated_obs_encoder import GatedObsEncoder
    from policy.gated_policy import GatedDiffusionPolicy
    enc = GatedObsEncoder(SHAPE_META, _resnet(), crop_shape=(84, 84), use_group_norm=True, imagenet_norm=True)
    sched = DDPMScheduler(num_train_timesteps=10, beta_schedule="squaredcos_cap_v2", prediction_type="epsilon")
    policy = GatedDiffusionPolicy(SHAPE_META, sched, enc, horizon=16, n_action_steps=8, n_obs_steps=2,
                                  num_inference_steps=2, diffusion_step_embed_dim=32, down_dims=(64, 128),
                                  gate_loss_weight=0.1, gate_anneal_steps=100, aux_weight=0.1, aux_hidden=32,
                                  log_every=1)
    B, T = 3, 16
    rng = np.random.default_rng(0)
    obs = {k: torch.from_numpy(np.stack([to_policy_obs(fake_raw(T=T, rng=rng), SHAPE_META["obs"])[k] for _ in range(B)]))
           for k in SHAPE_META["obs"]}
    batch = {"obs": obs, "action": torch.randn(B, T, 7),
             "meta": {"ep_thermal_gap": torch.tensor([0.0, 10.0, -20.0]), "ep_is_crossover": torch.tensor([1., 0., 0.])}}
    norm = LinearNormalizer()
    norm.fit({"action": batch["action"].reshape(-1, 7), "joints": obs["joints"].reshape(-1, 7),
              "cue": obs["cue"].reshape(-1, CUE_DIM)}, last_n_dims=1, mode="limits")
    from diffusion_policy.common.normalize_util import get_image_range_normalizer
    for k in ("rgb", "depth", "thermal"):
        norm[k] = get_image_range_normalizer()
    policy.set_normalizer(norm)
    policy.train()
    loss = policy.compute_loss(batch)
    assert torch.isfinite(loss) and loss.requires_grad
    loss.backward()
    assert policy.aux_heads["thermal"][0].weight.grad is not None
    assert enc.gate["thermal"][0].weight.grad is not None
    assert int(policy.train_steps) == 1 and 0 < policy.gate_lambda() < 0.1
    policy.eval()
    policy.w_trace = []
    policy.reset()
    with torch.no_grad():
        res = policy.predict_action({k: v[:, :2] for k, v in obs.items()})
    assert res["action"].shape == (B, 8, 7) and len(policy.w_trace) == 1 and policy.w_trace[0][0]["w"].shape == (B, 3)
    print("ok policy: ddpm + gate + aux losses backprop, lambda anneals, w_trace recorded at inference")


if __name__ == "__main__":
    test_cues()
    test_obs()
    test_corruption()
    test_shared_crop()
    test_init_weights()
    test_gate()
    test_policy_loss()
    print("all ok")
