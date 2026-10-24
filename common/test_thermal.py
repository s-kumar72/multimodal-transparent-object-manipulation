import unittest
import numpy as np
from common.thermal import THERMAL_RAW_RANGE, normalize_thermal_raw


class TestNormalizeThermalRaw(unittest.TestCase):
    def test_fixed_keeps_absolute_contrast(self):
        lo, hi = THERMAL_RAW_RANGE
        flat = np.full((2, 4, 4), lo + 0.5 * (hi - lo))
        flat[:, 0, 0] += 10  # tiny contrast must stay tiny
        out = normalize_thermal_raw(flat)
        self.assertEqual(out.dtype, np.float32)
        self.assertLess(out.max() - out.min(), 0.01)
        np.testing.assert_allclose(normalize_thermal_raw(np.array([[lo, hi + 100]])), [[0.0, 1.0]])

    def test_per_image_modes_stretch(self):
        raw = np.stack([np.arange(16.0).reshape(4, 4), 100 + 2 * np.arange(16.0).reshape(4, 4)])
        for mode in ("minmax", "percentile"):
            out = normalize_thermal_raw(raw, mode)
            self.assertTrue(np.all(out.min(axis=(1, 2)) < 0.05) and np.all(out.max(axis=(1, 2)) > 0.95), mode)
        np.testing.assert_allclose(normalize_thermal_raw(raw, "minmax")[0], normalize_thermal_raw(raw, "minmax")[1])

    def test_centered_removes_frame_offset_keeps_scale(self):
        base = np.random.default_rng(0).normal(3000, 20, (8, 8))
        frames = np.stack([base, base + 600])  # same scene, frame-level drift
        out = normalize_thermal_raw(frames, "centered", {"range": (-100, 100)})
        np.testing.assert_allclose(out[0], out[1], atol=1e-6)
        bumped = base.copy()
        bumped[0, 0] += 50
        delta = normalize_thermal_raw(bumped, "centered", {"range": (-100, 100)}) - out[0]
        self.assertAlmostEqual(float(delta[0, 0]), 0.25, places=5)  # 50 counts / 200-count scale

    def test_custom_range_and_bad_mode(self):
        np.testing.assert_allclose(normalize_thermal_raw([[10, 20]], params={"range": (10, 30)}), [[0.0, 0.5]])
        with self.assertRaises(ValueError):
            normalize_thermal_raw([[0]], "bogus")


if __name__ == "__main__":
    unittest.main()
