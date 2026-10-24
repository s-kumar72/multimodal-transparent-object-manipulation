"""
Thermal statistics on TRansPose sequences (native resolution, raw counts):
- per sequence: sign / size of the glass contrast vs its surrounding ring and the share of near-zero
  |CNR| frames (how common thermal crossover is);
- the narrowest "centered" range (relative to the frame median) that clips at most --clip of the
  glass + ring pixels, and the normalized glass contrast it gives vs other ranges.

  python -m encoder.thermal_stats --root <TRansPose root> [--sequences ...] [--clip 0.01] [--compare -170 350]
"""
import argparse
import os
import numpy as np
from common.thermal import THERMAL_CENTERED_RANGE
from encoder.dataset import compute_cnr, list_frames, list_sequences, load_frame, ring_mask

EDGES = np.arange(-2000.5, 2001.5)  # 1-count bins of (raw - frame median)
CENTERS = (EDGES[:-1] + EDGES[1:]) / 2


def sequence_stats(root, seq, cache_dir=None):
    """Per frame: histograms of (raw - median) on glass / ring / whole frame, signed ring CNR, ring diff."""
    path = cache_dir and os.path.join(cache_dir, f"{seq}_stats.npz")
    if path and os.path.exists(path):
        return dict(np.load(path))
    out = {k: [] for k in ("hist_glass", "hist_ring", "hist_all", "cnr", "diff")}
    for frame in list_frames(root, seq):
        raw, mask = load_frame(frame)
        c = np.clip(raw.astype(np.float32) - np.median(raw), EDGES[0], EDGES[-1])
        ring = ring_mask(mask).numpy()
        for k, sel in (("hist_glass", mask), ("hist_ring", ring), ("hist_all", slice(None))):
            out[k].append(np.histogram(c[sel], EDGES)[0])
        out["cnr"].append(compute_cnr(raw, mask, signed=True))
        out["diff"].append(c[mask].mean() - c[ring].mean() if mask.any() else np.nan)
    out = {k: np.array(v) for k, v in out.items()}
    if path:
        os.makedirs(cache_dir, exist_ok=True)
        np.savez_compressed(path, **out)
    return out


def fit_centered_range(hist, clip=0.01):
    """Narrowest (lo, hi) bin centers with at most `clip` of the histogram mass strictly outside."""
    cdf = np.cumsum(hist) / hist.sum()
    below = np.concatenate([[0.0], cdf[:-1]])  # mass below each lo
    ok = below <= clip
    hi_idx = np.minimum(np.searchsorted(cdf, 1 - clip + below - 1e-12), len(cdf) - 1)
    width = np.where(ok, CENTERS[hi_idx] - CENTERS, np.inf)
    i = int(np.argmin(width))
    return float(CENTERS[i]), float(CENTERS[hi_idx[i]])


def clipped(hist, lo, hi):
    """Fraction of mass strictly outside [lo, hi], per row."""
    out = (CENTERS < lo) | (CENTERS > hi)
    return (hist * out).sum(-1) / np.maximum(hist.sum(-1), 1)


def normalized_diff(hg, hr, lo, hi):
    """Per-frame mean normalized glass - ring value under centered range (lo, hi), clipping included."""
    v = np.clip((CENTERS - lo) / (hi - lo), 0, 1)
    return (hg @ v) / np.maximum(hg.sum(1), 1) - (hr @ v) / np.maximum(hr.sum(1), 1)


def pcts(x, p=(5, 25, 50, 75, 95)):
    return np.round(np.nanpercentile(x, p), 3).tolist()


def main(args):
    seqs = args.sequences or list_sequences(args.root)
    stats = {s: sequence_stats(args.root, s, args.cache_dir) for s in seqs}
    print("== glass contrast vs ring, per sequence (raw counts)")
    for s, st in stats.items():
        cnr, diff = st["cnr"], st["diff"]
        print(f"{s}: frames {len(cnr)}, glass darker {np.mean(diff < 0):.3f} / brighter {np.mean(diff > 0):.3f}, "
              f"diff pcts(5..95) {pcts(diff)}, signed CNR pcts {pcts(cnr)}, |CNR| < {args.near_zero} "
              f"{np.mean(np.abs(cnr) < args.near_zero):.3f}")

    hg = np.concatenate([st["hist_glass"] for st in stats.values()])
    hr = np.concatenate([st["hist_ring"] for st in stats.values()])
    ha = np.concatenate([st["hist_all"] for st in stats.values()])
    fit = fit_centered_range((hg + hr).sum(0), args.clip)
    ranges = {f"fit: glass+ring clip <= {args.clip}": fit}
    for lo, hi in zip(args.compare[::2], args.compare[1::2]):
        ranges[f"compare ({lo:.0f}, {hi:.0f})"] = (lo, hi)
    print(f"\n== centered ranges over {len(seqs)} sequence(s)")
    for name, (lo, hi) in ranges.items():
        d, gr = normalized_diff(hg, hr, lo, hi), clipped(hg + hr, lo, hi)
        print(f"{name}: ({lo:.0f}, {hi:.0f}) width {hi - lo:.0f}")
        print(f"  normalized glass - ring pcts(5..95) {pcts(d)}, median |d| {np.nanmedian(np.abs(d)):.4f}")
        print(f"  glass+ring clipped: pooled {clipped((hg + hr).sum(0), lo, hi):.4f}, worst frame {gr.max():.4f}, "
              f"frames > 5% {np.mean(gr > 0.05):.3f}; whole frame clipped: pooled {clipped(ha.sum(0), lo, hi):.4f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--sequences", nargs="*")
    ap.add_argument("--cache_dir", default="data/transpose_cache")
    ap.add_argument("--clip", type=float, default=0.01)
    ap.add_argument("--near_zero", type=float, default=0.5)
    ap.add_argument("--compare", type=float, nargs="*", default=list(THERMAL_CENTERED_RANGE),
                    help="extra centered ranges to compare: lo hi [lo hi ...]")
    main(ap.parse_args())
