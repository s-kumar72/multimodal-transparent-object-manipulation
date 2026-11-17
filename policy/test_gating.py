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


def test_depth_object_cue():
    """Optional 7th cue: with depth_object_ref on the cue key the vector grows by one, the first six are unchanged; a
    gated encoder and the policy's depth warm start accept it."""
    import copy
    import os
    import tempfile
    ref = np.full((96, 96), 150, np.uint8)                              # fake_raw's table is at 150
    raw = fake_raw(holes=0.0)
    raw["depth"][:, 50:70, 40:56] = 120                                  # an object 30 units in front of the table
    seen = compute_cues(raw, DEFAULT_ROI, depth_ref=ref)
    raw_st = fake_raw(holes=0.0)                                         # see-through: only the table
    through = compute_cues(raw_st, DEFAULT_ROI, depth_ref=ref)
    assert seen.shape == (2, CUE_DIM + 1) and np.allclose(seen[:, :CUE_DIM], compute_cues(raw))
    assert seen[0, CUE_DIM] > 0.04 and through[0, CUE_DIM] == 0, (seen[0, CUE_DIM], through[0, CUE_DIM])
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "ref.npy")
        np.save(path, ref)
        meta = copy.deepcopy(SHAPE_META)
        meta["obs"]["cue"]["shape"] = [CUE_DIM + 1]
        meta["obs"]["cue"]["depth_object_ref"] = path
        obs = to_policy_obs(raw, meta["obs"])
        assert obs["cue"].shape == (2, CUE_DIM + 1) and np.allclose(obs["cue"], seen)
        try:
            from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
        except ImportError:
            print("ok depth_object cue (policy part skipped: diffusers not installed)")
            return
        from diffusion_policy.common.normalize_util import get_image_range_normalizer
        from diffusion_policy.model.common.normalizer import LinearNormalizer
        from policy.gated_obs_encoder import GatedObsEncoder
        from policy.gated_policy import GatedDiffusionPolicy
        enc = GatedObsEncoder(meta, _resnet(), crop_shape=(84, 84), use_group_norm=True, imagenet_norm=True)
        assert enc.gate["depth"][0].in_features == CUE_DIM + 1
        sched = DDPMScheduler(num_train_timesteps=10, beta_schedule="squaredcos_cap_v2", prediction_type="epsilon")
        policy = GatedDiffusionPolicy(meta, sched, enc, horizon=16, n_action_steps=8, n_obs_steps=2,
                                      num_inference_steps=2, diffusion_step_embed_dim=32, down_dims=(64, 128),
                                      gate_loss_weight=0.1, gate_anneal_steps=100, aux_weight=0.1, aux_hidden=32,
                                      log_every=1, gate_depth_target="seethrough")
        B, T = 3, 16
        rng = np.random.default_rng(0)
        obs = {k: torch.from_numpy(np.stack([to_policy_obs(fake_raw(T=T, rng=rng), meta["obs"])[k] for _ in range(B)]))
               for k in meta["obs"]}
        batch = {"obs": obs, "action": torch.randn(B, T, 7),
                 "meta": {"ep_thermal_gap": torch.tensor([0.0, 10.0, -20.0]),
                          "ep_is_crossover": torch.tensor([1., 0., 0.]),
                          "ep_glass_seethrough": torch.tensor([1., 0., 1.])}}
        norm = LinearNormalizer()
        norm.fit({"action": batch["action"].reshape(-1, 7), "joints": obs["joints"].reshape(-1, 7),
                  "cue": obs["cue"].reshape(-1, CUE_DIM + 1)}, last_n_dims=1, mode="limits")
        for k in ("rgb", "depth", "thermal"):
            norm[k] = get_image_range_normalizer()
        policy.set_normalizer(norm)
        policy.train()
        import contextlib
        import io
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):                            # log_every=1: one line per step
            policy.compute_loss(batch).backward()
        line = buf.getvalue()
        assert "gate_depth" in line and "w_depth_st" in line and "w_depth_notst" in line, line
        g = enc.gate["depth"][-1].bias.grad                              # last layer (zero-initialized weight)
        assert g is not None and g.abs().sum() > 0
    print("ok depth_object cue: 7th entry, first six unchanged, encoder and depth warm start accept it")


def test_visibility_route():
    """Target-visibility cues from a glass mask, and the encoder that switches sensors on / off with them."""
    import copy
    from policy.obs_util import visibility_cues
    from policy.gated_obs_encoder import GatedObsEncoder
    raw = fake_raw(T=4, hot=True, holes=0.0)
    raw["depth"][:, 50:70, 40:56] = 120                                  # glass 30 units in front of the table (150)
    mask = np.zeros((4, 96, 96, 1), np.uint8)
    mask[:, 50:70, 40:56] = 255
    raw["glass_mask"] = mask
    v = visibility_cues(raw["depth"], raw["thermal_raw"], mask)
    assert v.shape == (4, 3) and np.allclose(v[:, 0], 30) and (v[:, 1] > 50).all() and (v[:, 2] == 1).all(), v
    vis = {"thr": (15.0, 30.0), "seg_ckpt": None}
    seen = compute_cues(raw, DEFAULT_ROI, vis=vis)
    assert seen.shape == (4, CUE_DIM + 4) and np.allclose(seen[:, :CUE_DIM], compute_cues(raw))
    assert (seen[:, -2:] == 1).all()
    through = dict(raw, depth=np.full_like(raw["depth"], 150))           # depth sees only the table
    cold = dict(fake_raw(T=4, hot=False, holes=0.0), glass_mask=mask)    # thermal crossover, depth as fake_raw (flat)
    cold["depth"][:, 50:70, 40:56] = 120
    assert (compute_cues(through, DEFAULT_ROI, vis=vis)[:, -2:] == [0, 1]).all()
    assert (compute_cues(cold, DEFAULT_ROI, vis=vis)[:, -2:] == [1, 0]).all()
    empty = dict(raw, glass_mask=np.zeros_like(mask))                    # no mask: nothing is switched off
    assert (compute_cues(empty, DEFAULT_ROI, vis=vis)[:, -2:] == 1).all()

    meta = copy.deepcopy(SHAPE_META)
    meta["obs"]["cue"]["shape"] = [CUE_DIM + 4]
    meta["obs"]["cue"]["vis_thr"] = [15, 30]
    obs = to_policy_obs(raw, meta["obs"])
    assert obs["cue"].shape == (4, CUE_DIM + 4) and np.allclose(obs["cue"], seen)
    # mask errors for robustness tests (eval.py --mask_noise): opt-in, predicted masks only
    from policy.obs_util import cue_opts, degrade_mask
    noisy = copy.deepcopy(meta)
    noisy["obs"]["cue"]["mask_noise"] = {"drop": 1.0}
    assert cue_opts(noisy["obs"])["vis"]["mask_noise"] == {"drop": 1.0} and "mask_noise" not in cue_opts(meta["obs"])["vis"]
    assert np.allclose(to_policy_obs(raw, noisy["obs"])["cue"], seen)    # a mask stored with the frames is not degraded
    mb = mask[..., 0] > 0
    assert (degrade_mask(mb, raw["depth"]) == mb).all() and not degrade_mask(mb, raw["depth"], drop=1.0).any()
    moved = degrade_mask(mb, raw["depth"], shift=2)
    assert (moved.sum((1, 2)) == mb.sum((1, 2))).all() and (moved == degrade_mask(mb, raw["depth"], shift=2)).all()
    enc = GatedObsEncoder(meta, _resnet(), crop_shape=(84, 84), use_group_norm=True, imagenet_norm=True,
                          dropout_mode="none", route="visible", route_rgb_only_p=0.0)
    B = 4
    t = {k: torch.from_numpy(np.stack([to_policy_obs(fake_raw(T=1, holes=0.0), SHAPE_META["obs"])[k][0]
                                       for _ in range(B)])) for k in ("rgb", "depth", "thermal", "joints")}
    cue = torch.zeros(B, CUE_DIM + 4)
    cue[:, -2:] = torch.tensor([[1., 1.], [0., 1.], [1., 0.], [0., 0.]])   # normal, see-through, crossover, both
    t["cue"] = cue
    for mode in (True, False):
        enc.train(mode)
        enc(t)
        assert enc.last_w.tolist() == [[1, 1, 1], [1, 0, 1], [1, 1, 0], [1, 0, 0]], enc.last_w
    out = enc(t)
    assert torch.allclose(out[1, 512:1024], torch.zeros(512)) and not torch.allclose(out[0, 512:1024], torch.zeros(512))
    enc.route_eval = "depth_first"
    enc(t)
    assert enc.last_w.tolist() == [[1, 1, 0], [1, 0, 1], [1, 1, 0], [1, 0, 0]], enc.last_w
    enc.route_eval = "all_on"
    enc(t)
    assert (enc.last_w == 1).all()
    enc.route_eval = "rgb_only"
    enc(t)
    assert enc.last_w.tolist() == [[1, 0, 0]] * 4
    enc.route_eval, enc.route_rgb_only_p = "visible", 1.0
    enc.train()
    enc(t)
    assert enc.last_w.tolist() == [[1, 0, 0]] * 4                          # training-time rgb-only draw
    enc.route_rgb_only_p = 0.0
    try:
        from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
    except ImportError:
        print("ok visibility route (policy part skipped: diffusers not installed)")
        return
    from diffusion_policy.common.normalize_util import get_image_range_normalizer
    from diffusion_policy.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
    from policy.gated_policy import GatedDiffusionPolicy
    sched = DDPMScheduler(num_train_timesteps=10, beta_schedule="squaredcos_cap_v2", prediction_type="epsilon")
    policy = GatedDiffusionPolicy(meta, sched, enc, horizon=16, n_action_steps=8, n_obs_steps=2,
                                  num_inference_steps=2, diffusion_step_embed_dim=32, down_dims=(64, 128),
                                  gate_loss_weight=0, aux_weight=0.1, aux_hidden=32, log_every=0)
    norm = LinearNormalizer()
    norm.fit({"action": torch.randn(64, 7), "joints": torch.randn(64, 7)}, last_n_dims=1, mode="limits")
    norm["cue"] = SingleFieldLinearNormalizer.create_identity()
    for k in ("rgb", "depth", "thermal"):
        norm[k] = get_image_range_normalizer()
    policy.set_normalizer(norm)
    policy.eval()
    obs2 = {k: v[:, None].repeat(1, 2, *[1] * (v.dim() - 1)) for k, v in t.items()}       # (B, To=2, ...)
    policy.route_latch, policy.guidance = True, 2.0
    policy.reset()
    with torch.no_grad():
        res = policy.predict_action(obs2)
        assert res["action"].shape == (B, 8, 7) and torch.isfinite(res["action"]).all()
        first = enc.last_w.view(B, 2, -1)[:, -1].clone()
        flipped = dict(obs2, cue=obs2["cue"].clone())
        flipped["cue"][..., -2:] = 0                                      # later frames: nothing visible
        policy.predict_action(flipped)
        assert torch.equal(enc.last_w.view(B, 2, -1)[:, -1], first)       # latched: the first flags still decide
        policy.reset()
        policy.predict_action(flipped)
        assert (enc.last_w.view(B, 2, -1)[:, -1, 1:] == 0).all()          # after reset the new flags are taken
        # wrong flags for robustness tests (eval.py --gate_error): forced values, flips; the input is not changed
        last_w = lambda: enc.last_w.view(B, 2, -1)[:, -1].tolist()
        policy.route_latch, policy.guidance = False, 1.0
        policy.reset()
        policy.predict_action(obs2)
        assert last_w() == [[1, 1, 1], [1, 0, 1], [1, 1, 0], [1, 0, 0]]
        policy.gate_error = {"depth": 0}
        policy.predict_action(obs2)
        assert last_w() == [[1, 0, 1], [1, 0, 1], [1, 0, 0], [1, 0, 0]]
        policy.gate_error = {"depth": 1, "thermal": 1}
        policy.predict_action(obs2)
        assert (enc.last_w == 1).all()
        policy.gate_error = {"flip": 1.0}
        policy.predict_action(obs2)
        assert last_w() == [[1, 0, 0], [1, 1, 0], [1, 0, 1], [1, 1, 1]]
        policy.gate_error, policy._ge_gen = {"flip": 0.0}, None
        policy.predict_action(obs2)
        assert last_w() == [[1, 1, 1], [1, 0, 1], [1, 1, 0], [1, 0, 0]]
        policy.gate_error, policy._ge_gen = {"flip": 0.5}, None
        draws = []
        for _ in range(50):
            policy.predict_action(obs2)
            draws.append(enc.last_w.view(B, 2, -1)[:, -1, 1:].clone())
        wrong = (torch.stack(draws) != cue[:, -2:]).float().mean()
        assert 0.4 < wrong < 0.6, wrong
        assert torch.equal(obs2["cue"][:, :, -2:], cue[:, None, -2:].expand(B, 2, 2))
        policy.gate_error, policy._ge_gen = None, None
    assert enc.route_eval == "visible" and policy._guide_cond is None
    print("ok visibility route: cues at the mask, flags, sensors switched by the flags, eval rules, latch, guidance, "
          "gate_error, mask_noise")


def test_soft_route_and_norm():
    """route=soft (continuous weights from the visibility measurements, learned slope / shift) and gate_norm=max."""
    import copy
    import torch.nn as nn
    from policy.obs_util import cue_opts
    from policy.gated_obs_encoder import GatedObsEncoder
    meta = copy.deepcopy(SHAPE_META)
    meta["obs"]["cue"]["shape"] = [CUE_DIM + 4]
    meta["obs"]["cue"]["vis_thr"] = [15, 30]
    B = 5
    t = {k: torch.from_numpy(np.stack([to_policy_obs(fake_raw(T=1, holes=0.0), SHAPE_META["obs"])[k][0]
                                       for _ in range(B)])) for k in ("rgb", "depth", "thermal", "joints")}
    kw = dict(crop_shape=(84, 84), use_group_norm=True, imagenet_norm=True)
    try:                                                                  # soft needs the raw measurements
        GatedObsEncoder(meta, _resnet(), dropout_mode="none", route="soft", route_tau=[5.0, 2.0], **kw)
        raise RuntimeError("route=soft without cue.vis_raw must fail")
    except AssertionError:
        pass
    meta["obs"]["cue"]["vis_raw"] = True
    assert cue_opts(meta["obs"])["vis"]["raw"] is True and "raw" not in cue_opts({"cue": {"vis_thr": [15, 30]}})["vis"]
    enc = GatedObsEncoder(meta, _resnet(), dropout_mode="none", route="soft", route_tau=[5.0, 2.0], **kw)
    cue = torch.zeros(B, CUE_DIM + 4)
    #                      depth step, thermal step, depth flag, thermal flag
    cue[:, -4:] = torch.tensor([[50., 150., 1., 1.],      # both clearly visible
                                [-2., 150., 0., 1.],      # see-through
                                [50., 5., 1., 0.],        # crossover
                                [15., 30., 0., 0.],       # exactly at both thresholds
                                [0., 0., 1., 1.]])        # no glass mask found (flags 1 with zero steps)
    t["cue"] = cue
    sig = lambda x: 1 / (1 + np.exp(-x))
    want = np.array([[1, sig(35 / 5), sig(120 / 2)], [1, sig(-17 / 5), sig(120 / 2)], [1, sig(35 / 5), sig(-25 / 2)],
                     [1, 0.5, 0.5], [1, 1, 1]])
    for mode in (True, False):
        enc.train(mode)
        out = enc(t)
        assert np.allclose(enc.last_w.numpy(), want, atol=1e-6), enc.last_w
    assert enc.last_w[1, 1] < 0.04 and enc.last_w[2, 2] < 1e-4 and enc.last_w[0, 1:].min() > 0.99
    g = torch.autograd.grad(out.sum(), [enc.route_a, enc.route_b], allow_unused=True)
    assert all(x is not None and torch.isfinite(x).all() and x.abs().sum() > 0 for x in g), g    # the gate is learned
    with torch.no_grad():                                                 # a sharper gate moves the weights apart
        enc.route_a.fill_(3.0)
        enc(t)
        assert enc.last_w[1, 1] < want[1, 1] and enc.last_w[0, 1] > want[0, 1] and enc.last_w[3, 1] == 0.5
        enc.route_a.fill_(1.0)
        enc.route_b.fill_(2.0)                                            # a shift raises every weight
        enc(t)
        assert (enc.last_w[:4, 1:].numpy() > want[:4, 1:] - 1e-9).all() and abs(float(enc.last_w[3, 1]) - sig(2)) < 1e-6
        enc.route_b.fill_(0.0)
    enc.route_eval = "all_on"
    enc(t)
    assert (enc.last_w == 1).all()
    enc.route_eval = "rgb_only"
    enc(t)
    assert enc.last_w.tolist() == [[1, 0, 0]] * B
    enc.route_eval = "depth_first"
    try:
        enc(t)
        raise RuntimeError("depth_first must be refused for route=soft")
    except AssertionError:
        pass
    enc.route_eval, enc.route_rgb_only_p = "visible", 1.0
    enc.train()
    enc(t)
    assert enc.last_w.tolist() == [[1, 0, 0]] * B                         # training-time rgb-only draw
    assert {"route_a", "route_b", "route_thr", "route_tau"} <= set(enc.state_dict())

    # gate_norm=max on the learned gate: the largest weight of each frame is 1, ratios are kept, drops stay 0
    torch.manual_seed(0)
    plain = GatedObsEncoder(meta, _resnet(), dropout_mode="none", **kw)
    for m in plain.gate.values():
        nn.init.normal_(m[-1].weight, std=0.5)
        nn.init.constant_(m[-1].bias, -3.0)                               # small weights, as the collapsed gate
    normed = GatedObsEncoder(meta, _resnet(), dropout_mode="none", gate_norm="max", **kw)
    normed.load_state_dict(plain.state_dict())
    t["cue"] = torch.randn(B, CUE_DIM + 4)
    plain.eval(), normed.eval()
    plain(t), normed(t)
    wp, wn = plain.last_w, normed.last_w
    assert wp.max() < 0.5 and torch.allclose(wn.max(-1).values, torch.ones(B))
    assert torch.allclose(wn, wp / wp.max(-1, keepdim=True).values, atol=1e-6)
    normed.force_drop = {"thermal"}
    normed(t)
    assert (normed.last_w[:, 2] == 0).all() and torch.allclose(normed.last_w.max(-1).values, torch.ones(B))
    normed.force_drop = set()
    normed.train()
    normed.dropout_mode, normed.dropout_p = "targeted", 1.0               # the top sensor is dropped: the next one gets 1
    normed(t)
    assert (normed.last_w.min(-1).values == 0).all() and torch.allclose(normed.last_w.max(-1).values, torch.ones(B))

    try:
        from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
    except ImportError:
        print("ok soft route + gate_norm (policy part skipped: diffusers not installed)")
        return
    from diffusion_policy.common.normalize_util import get_image_range_normalizer
    from diffusion_policy.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
    from policy.gated_policy import GatedDiffusionPolicy
    enc.route_rgb_only_p = 0.1
    sched = DDPMScheduler(num_train_timesteps=10, beta_schedule="squaredcos_cap_v2", prediction_type="epsilon")
    policy = GatedDiffusionPolicy(meta, sched, enc, horizon=16, n_action_steps=8, n_obs_steps=2,
                                  num_inference_steps=2, diffusion_step_embed_dim=32, down_dims=(64, 128),
                                  gate_loss_weight=0, aux_weight=0.1, aux_hidden=32, log_every=0)
    norm = LinearNormalizer()
    norm.fit({"action": torch.randn(64, 7), "joints": torch.randn(64, 7)}, last_n_dims=1, mode="limits")
    norm["cue"] = SingleFieldLinearNormalizer.create_identity()
    for k in ("rgb", "depth", "thermal"):
        norm[k] = get_image_range_normalizer()
    policy.set_normalizer(norm)
    t["cue"] = cue
    obs2 = {k: v[:, None].repeat(1, 2, *[1] * (v.dim() - 1)) for k, v in t.items()}       # (B, To=2, ...)
    policy.train()
    batch = {"obs": obs2, "action": torch.randn(B, 16, 7),
             "meta": {"ep_glass_seethrough": torch.tensor([0, 1, 0, 0, 0]), "ep_is_crossover": torch.tensor([0, 0, 1, 0, 0])}}
    loss = policy.compute_loss(batch)
    loss.backward()
    assert enc.route_a.grad is not None and torch.isfinite(enc.route_a.grad).all()
    policy.eval()
    policy.guidance = 2.0
    policy.reset()
    with torch.no_grad():
        res = policy.predict_action(obs2)
    assert res["action"].shape == (B, 8, 7) and torch.isfinite(res["action"]).all()
    assert np.allclose(enc.last_w.view(B, 2, -1)[:, -1].numpy(), want, atol=1e-6)        # the soft weights, not rgb-only
    print("ok soft route + gate_norm")


if __name__ == "__main__":
    test_cues()
    test_obs()
    test_corruption()
    test_shared_crop()
    test_init_weights()
    test_gate()
    test_policy_loss()
    test_depth_object_cue()
    test_visibility_route()
    test_soft_route_and_norm()
    print("all ok")
