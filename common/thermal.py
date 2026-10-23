"""
Shared thermal normalization for real (TRansPose 14-bit raw) and simulated thermal images.
"""
import numpy as np

# Raw-count range mapped to [0, 1] by the default "fixed" mode. The sim should map its
# radiance to the same counts so real and sim thermal share one scale.
# TODO: placeholder = full 14-bit range; to be set from TRansPose train-split statistics.
THERMAL_RAW_RANGE = (0, 16383)
THERMAL_PERCENTILES = (1.0, 99.0)


def normalize_thermal_raw(raw, mode="fixed", params=None):
    """Map raw thermal counts of shape (..., H, W) to float32 in [0, 1].

    fixed:      (raw - lo) / (hi - lo) with params["range"] (default THERMAL_RAW_RANGE).
                Keeps absolute contrast, so low-contrast scenes (thermal crossover) stay flat.
    percentile: per-image params["percentiles"] (default THERMAL_PERCENTILES) -> [0, 1].
    minmax:     per-image min/max -> [0, 1] (same as the 8-bit images; hides low contrast).
    """
    params = params or {}
    raw = np.asarray(raw, dtype=np.float32)
    if mode == "fixed":
        lo, hi = params.get("range", THERMAL_RAW_RANGE)
    elif mode == "percentile":
        lo, hi = np.percentile(raw, params.get("percentiles", THERMAL_PERCENTILES), axis=(-2, -1), keepdims=True)
    elif mode == "minmax":
        lo, hi = raw.min(axis=(-2, -1), keepdims=True), raw.max(axis=(-2, -1), keepdims=True)
    else:
        raise ValueError(f"unknown thermal normalization mode: {mode}")
    return np.clip((raw - lo) / np.maximum(hi - lo, 1e-6), 0.0, 1.0).astype(np.float32)
