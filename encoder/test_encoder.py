"""
python -m unittest encoder.test_encoder   (DP checks use $DP_ROOT, default ../diffusion_policy; skipped if missing)
"""
import json
import os
import sys
import tempfile
import unittest
import numpy as np
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from PIL import Image
from encoder.dataset import (ThermalSegDataset, compute_cnr, list_frames, list_sequences, load_frame, load_transpose,
                             random_arrays, split_sequences)
from encoder.export import encoder_state_dict, load_into_dp_obs_encoder, load_thermal_encoder
from encoder.model import ThermalSegNet, make_thermal_resnet18, seg_loss

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.environ.get("DP_ROOT", os.path.join(REPO, "..", "diffusion_policy")))
try:
    from diffusion_policy.common.pytorch_util import replace_submodules
    from diffusion_policy.model.vision.model_getter import get_resnet
    from diffusion_policy.model.vision.multi_image_obs_encoder import MultiImageObsEncoder
    HAS_DP = True
except ImportError:
    HAS_DP = False


def save_fake_ckpt(path):
    torch.manual_seed(0)
    model = ThermalSegNet(make_thermal_resnet18())
    torch.save({"model": model.state_dict()}, path)
    return model.encoder.eval()


class TestModelAndData(unittest.TestCase):
    def test_seg_forward_and_loss(self):
        model = ThermalSegNet(make_thermal_resnet18())
        x, m = torch.rand(2, 1, 84, 84), (torch.rand(2, 1, 84, 84) > 0.7).float()
        logits = model(x)
        self.assertEqual(logits.shape, (2, 1, 84, 84))
        loss = seg_loss(logits, m)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertFalse(any(isinstance(mod, nn.BatchNorm2d) for mod in model.modules()))

    def test_imagenet_conv1_is_rgb_mean(self):
        path = os.path.join(torch.hub.get_dir(), "checkpoints", "resnet18-f37072fd.pth")
        if not os.path.exists(path):
            self.skipTest("ImageNet ResNet-18 weights not cached")
        rgb = torch.load(path, map_location="cpu")["conv1.weight"]
        torch.testing.assert_close(make_thermal_resnet18("imagenet").conv1.weight, rgb.mean(1, keepdim=True))

    def test_dataset_items_and_cnr(self):
        raw, mask, cnr = random_arrays(8, 96)
        train = ThermalSegDataset(raw / 16383, mask, cnr, train=True, crop_size=84)
        val = ThermalSegDataset(raw / 16383, mask, cnr, train=False, crop_size=84)
        item = train[0]
        self.assertEqual(item["thermal"].shape, (1, 84, 84))
        self.assertTrue(set(item["mask"].unique().tolist()) <= {0.0, 1.0})
        self.assertEqual(val[0]["thermal"].shape, (1, 96, 96))
        img, m = np.random.default_rng(0).normal(100, 2, (64, 64)), np.zeros((64, 64), bool)
        m[20:40, 20:40] = True
        self.assertAlmostEqual(compute_cnr(img + 10 * m, m), 5.0, delta=1.0)
        self.assertTrue(np.isnan(compute_cnr(img, np.zeros_like(m))))

    def test_transpose_reader_and_cache(self):
        with tempfile.TemporaryDirectory() as root:
            cam = os.path.join(root, "seq_all_01", "cam_T")
            for d in ("14bit", "pose", "mask"):
                os.makedirs(os.path.join(cam, d))
            Image.fromarray(np.full((32, 40), 3000, np.uint16)).save(os.path.join(cam, "14bit", "000001.png"))
            objs = {}
            for k, (name, rows) in enumerate([("obj_H_09_01_G", slice(0, 8)), ("obj_N_03_01_P", slice(20, 28))], 1):
                m = np.zeros((32, 40), np.uint8)
                m[rows, :10] = 60
                Image.fromarray(m).save(os.path.join(cam, "mask", f"000001_00000{k}.png"))
                objs[str(k)] = {"obj_name": name, "mask_img": f"000001_00000{k}.png"}
            with open(os.path.join(cam, "pose", "000001.json"), "w") as f:
                json.dump(objs, f)
            self.assertEqual(list_sequences(root), ["seq_all_01"])
            raw, mask = load_frame(list_frames(root, "seq_all_01")[0])
            self.assertEqual((raw.dtype, raw.max()), (np.uint16, 3000))
            self.assertEqual(mask.sum(), 80)  # glass only; the non-transparent (N) object is excluded
            cache = os.path.join(root, "cache")
            raw96, mask96, cnr = load_transpose(root, ["seq_all_01"], 16, cache)
            self.assertEqual((raw96.shape, mask96.shape, cnr.shape), ((1, 16, 16), (1, 16, 16), (1,)))
            self.assertTrue(os.path.exists(os.path.join(cache, "seq_all_01_16.npz")))

    def test_fit_centered_range(self):
        from encoder.thermal_stats import CENTERS, clipped, fit_centered_range
        hist = np.zeros(len(CENTERS))
        hist[(CENTERS >= -10) & (CENTERS <= 10)] = 1  # 21 counts of mass
        hist[CENTERS == 500] = 0.2  # a far-off hot spot (< 1% of mass)
        lo, hi = fit_centered_range(hist, clip=0.01)
        self.assertEqual((lo, hi), (-10.0, 10.0))
        self.assertLessEqual(clipped(hist, lo, hi), 0.01)

    def test_sequence_split_is_disjoint(self):
        tr, va = split_sequences([f"seq_{i:02d}" for i in range(7)], 0.2, seed=1)
        self.assertEqual(len(va), 1)
        self.assertFalse(set(tr) & set(va))
        self.assertEqual(len(tr) + len(va), 7)


class TestExport(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.ckpt = os.path.join(self.tmp.name, "best.pt")
        self.encoder = save_fake_ckpt(self.ckpt)
        self.x = torch.rand(2, 1, 96, 96)

    def tearDown(self):
        self.tmp.cleanup()

    def export(self, in_chans=1):
        path = os.path.join(self.tmp.name, f"enc{in_chans}.pth")
        torch.save(encoder_state_dict(self.ckpt, in_chans), path)
        return path

    def test_roundtrip_1ch_96(self):
        sd = torch.load(self.export())
        self.assertFalse(any(k.startswith(("fc.", "encoder.")) for k in sd))
        net = load_thermal_encoder(self.export()).eval()
        with torch.no_grad():
            out = net(self.x)
            self.assertEqual(out.shape, (2, 512))
            torch.testing.assert_close(out, self.encoder(self.x))

    def test_in_chans_3_matches_repeated_input(self):
        net = load_thermal_encoder(self.export(3)).eval()
        with torch.no_grad():
            torch.testing.assert_close(net(self.x.repeat(1, 3, 1, 1)), self.encoder(self.x), rtol=1e-4, atol=1e-5)

    @unittest.skipUnless(HAS_DP, "diffusion_policy not importable")
    def test_loads_into_dp_built_resnet(self):
        """Build the encoder exactly as DP does (get_resnet + BN->GN) with a 1-ch conv1; strict load + forward."""
        dp = get_resnet("resnet18", weights=None)
        dp.conv1 = nn.Conv2d(1, 64, 7, 2, 3, bias=False)
        dp = replace_submodules(dp, lambda m: isinstance(m, nn.BatchNorm2d),
                                lambda m: nn.GroupNorm(m.num_features // 16, m.num_features))
        sd = torch.load(self.export())
        self.assertEqual({k: v.shape for k, v in sd.items()}, {k: v.shape for k, v in dp.state_dict().items()})
        dp.load_state_dict(sd, strict=True)
        with torch.no_grad():
            torch.testing.assert_close(dp.eval()(self.x), self.encoder(self.x))

    @unittest.skipUnless(HAS_DP, "diffusion_policy not importable")
    def test_dp_obs_encoder_with_1ch_thermal(self):
        shape_meta = OmegaConf.create({"obs": {"thermal": {"shape": [1, 96, 96], "type": "rgb"},
                                               "joints": {"shape": [7], "type": "low_dim"}}})
        enc = MultiImageObsEncoder(shape_meta, get_resnet("resnet18"), crop_shape=(84, 84),
                                   use_group_norm=True, imagenet_norm=False)
        load_into_dp_obs_encoder(enc, self.export()).eval()
        out = enc({"thermal": self.x, "joints": torch.zeros(2, 7)})
        self.assertEqual(out.shape, (2, 512 + 7))


class TestTrainSmoke(unittest.TestCase):
    def test_train_eval_export_on_random_data(self):
        from encoder.train import run_eval, train
        with tempfile.TemporaryDirectory() as tmp:
            cfg = OmegaConf.load(os.path.join(REPO, "encoder/configs/transpose_seg.yaml"))
            cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist([
                f"out_dir={tmp}", "device=cpu", "data.name=random", "data.n_random=40",
                "train.epochs=1", "train.batch_size=8", "train.num_workers=0", "eval.n_vis=4"]))
            train(cfg)
            for f in ("best.pt", "last.pt", "log.csv", "val_grid.png", "splits.json"):
                self.assertTrue(os.path.exists(os.path.join(tmp, f)), f)
            run_eval(os.path.join(tmp, "best.pt"), OmegaConf.create())
            self.assertTrue(os.path.exists(os.path.join(tmp, "best_eval.json")))
            self.assertIn("conv1.weight", encoder_state_dict(os.path.join(tmp, "best.pt")))


if __name__ == "__main__":
    unittest.main()
