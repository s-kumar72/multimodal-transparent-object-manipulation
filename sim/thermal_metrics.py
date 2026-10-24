"""
Thermal image statistics on raw counts, the same code for real (TRansPose 14-bit) and simulated frames.
Each function takes raw counts (H, W) and, where needed, the glass mask (H, W) bool.
cnr() matches compute_cnr in encoder/dataset.py (square ring around the glass, sample std).
"""
import numpy as np
from PIL import Image

LOW_CNR = 1.0 # frames with CNR below this count as low contrast (crossover-like)

def resize(raw, mask=None, size=96):
    """Same resize for real and sim: bilinear for counts (as env.py makes data/thermal_raw), nearest for masks."""
    out = np.asarray(Image.fromarray(np.asarray(raw, np.float32)).resize((size, size), Image.BILINEAR))
    if mask is None:
        return out
    return out, np.asarray(Image.fromarray(mask.astype(np.uint8)).resize((size, size), Image.NEAREST)) > 0

def dilate(mask, r):
    """Square (2r+1) dilation, like max_pool2d(2r+1, stride 1, padding r)."""
    out = mask.copy()
    for axis in (0, 1):
        pad = np.pad(out, [(r, r) if a == axis else (0, 0) for a in (0, 1)])
        n = out.shape[axis]
        out = np.any([np.take(pad, range(i, i + n), axis=axis) for i in range(2 * r + 1)], axis=0)
    return out

def ring(mask, r, gap=0):
    """Background band of width r around the glass, starting gap px away from it."""
    return dilate(mask, gap + r) & ~(dilate(mask, gap) if gap else mask)

def cnr(raw, mask, r=8, gap=0):
    """|mean glass - mean background ring| / std background ring. NaN without glass.
    gap=0 is the encoder definition; gap > 0 skips the blurred glass edge so the ring std is background only."""
    bg = raw[ring(mask, r, gap)]
    if not mask.any() or bg.size < 2:
        return np.nan
    return abs(raw[mask].mean() - bg.mean()) / (bg.std(ddof=1) + 1e-6)

def delta(raw, mask, r=8):
    """Signed mean glass - mean background ring (counts)."""
    bg = raw[ring(mask, r)]
    return raw[mask].mean() - bg.mean() if mask.any() and bg.size else np.nan

def noise(raw):
    """Noise std (counts) from the Laplacian residual; the median makes it robust to edges (flat areas dominate)."""
    x = np.asarray(raw, float)
    lap = x[:-2, :-2] - 2 * x[:-2, 1:-1] + x[:-2, 2:] - 2 * x[1:-1, :-2] + 4 * x[1:-1, 1:-1] - 2 * x[1:-1, 2:] + x[2:, :-2] - 2 * x[2:, 1:-1] + x[2:, 2:]
    return np.median(np.abs(lap)) / 0.6745 / 6.0     # white noise: std(lap) = 6 * sigma

def edge_sharpness(raw, mask):
    """Mean gradient magnitude on the glass boundary / |delta| (1/px): higher = sharper edge, independent of contrast."""
    d = delta(raw, mask, r=2)
    gy, gx = np.gradient(np.asarray(raw, float))
    edge = dilate(mask, 1) & dilate(~mask, 1)     # pixels on both sides of the boundary
    return np.hypot(gx, gy)[edge].mean() / (abs(d) + 1e-6) if mask.any() else np.nan

def frame_stats(raw, mask, r, gap):
    """All metrics of one frame (ring width r, cnr_gap skips gap px of blurred edge).
    The *_noise / *_range ratios do not depend on the camera's counts per degree."""
    c, d, n = cnr(raw, mask, r), delta(raw, mask, r), noise(raw)
    p1, p99 = float(np.percentile(raw, 1)), float(np.percentile(raw, 99))
    return {"cnr": c, "cnr_gap": cnr(raw, mask, r, gap), "delta": d, "ring_std": abs(d) / c if c > 0 else np.nan, "noise": n,
            "delta_noise": abs(d) / n, "delta_range": abs(d) / (p99 - p1), "noise_range": n / (p99 - p1),
            "sharpness": edge_sharpness(raw, mask), "glass_frac": float(mask.mean()),
            "median": float(np.median(raw)), "p1": p1, "p99": p99}
