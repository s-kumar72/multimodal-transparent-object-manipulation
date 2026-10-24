"""
Thermal statistics on TRansPose sequences (native resolution, raw counts):
- per sequence: sign / size of the glass contrast vs its surrounding ring and the share of near-zero
  |CNR| frames (how common thermal crossover is); sequences with any such frame are flagged;
- two "centered" ranges (relative to the frame median) fitted on the given sequences (use the train split):
  wide   = whole-frame 0.5 / 99.5 percentiles,
  narrow = narrowest range with glass + ring clipping < --clip in at least --frame_frac of the frames,
  and the normalized glass contrast / clipping each gives.

  python -m encoder.thermal_stats --root <TRansPose root> [--sequences ...] [--compare -170 350]
"""
import argparse
import os
import numpy as np
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


def percentile_range(hist, lo_p=0.5, hi_p=99.5):
    """(lo, hi) bin centers at the given percentiles of a pooled histogram."""
    cdf = np.cumsum(hist) / hist.sum()
    lo = np.searchsorted(cdf, lo_p / 100, side="right")  # first bin with more than lo_p below or in it
    return float(CENTERS[lo]), float(CENTERS[np.searchsorted(cdf, hi_p / 100)])


def frame_clipped(hist, lo, hi):
    """Per-frame fraction of mass strictly outside [lo, hi] (hist: frames x bins)."""
    out = (CENTERS < lo) | (CENTERS > hi)
    return (hist * out).sum(-1) / np.maximum(hist.sum(-1), 1)


def fit_narrow_range(hist, clip=0.01, frame_frac=0.95):
    """Narrowest (lo, hi) bin centers with clipping < clip in at least frame_frac of the frames (frames x bins)."""
    cdf = np.cumsum(hist, 1) / np.maximum(hist.sum(1, keepdims=True), 1)
    below = np.concatenate([np.zeros((len(cdf), 1)), cdf[:, :-1]], 1)  # mass strictly below each bin
    n = cdf.shape[1]

    def ok(i, j):
        return np.mean(below[:, i] + 1 - cdf[:, j] < clip) >= frame_frac

    best = (np.inf, 0, n - 1)
    for i in range(n):
        if not ok(i, n - 1):  # lo too high; clipping below only grows from here
            break
        a, b = i, n - 1  # smallest hi index that still satisfies the criterion
        while a < b:
            mid = (a + b) // 2
            a, b = (a, mid) if ok(i, mid) else (mid + 1, b)
        if CENTERS[a] - CENTERS[i] < best[0]:
            best = (CENTERS[a] - CENTERS[i], i, a)
    return float(CENTERS[best[1]]), float(CENTERS[best[2]])


def normalized_diff(hg, hr, lo, hi):
    """Per-frame mean normalized glass - ring value under centered range (lo, hi), clipping included."""
    v = np.clip((CENTERS - lo) / (hi - lo), 0, 1)
    return (hg @ v) / np.maximum(hg.sum(1), 1) - (hr @ v) / np.maximum(hr.sum(1), 1)


def pcts(x, p=(5, 25, 50, 75, 95)):
    return np.round(np.nanpercentile(x, p), 3).tolist()


def main(args):
    seqs = args.sequences or list_sequences(args.root)
    stats = {s: sequence_stats(args.root, s, args.cache_dir) for s in seqs}
    print(f"== glass contrast vs ring, per sequence (raw counts); [CROSSOVER] = any frame with |CNR| < {args.near_zero}")
    flagged = []
    for s, st in stats.items():
        cnr, diff = st["cnr"], st["diff"]
        near = np.abs(cnr) < args.near_zero
        if near.any():
            flagged.append(s)
        print(f"{'[CROSSOVER] ' if near.any() else ''}{s}: frames {len(cnr)}, glass darker {np.mean(diff < 0):.3f} / "
              f"brighter {np.mean(diff > 0):.3f}, diff pcts(5..95) {pcts(diff)}, signed CNR pcts {pcts(cnr)}, "
              f"|CNR| < {args.near_zero}: {near.sum()} frames ({near.mean():.3f})")
    print(f"sequences with near-zero |CNR| frames: {flagged or 'none'}")

    hg = np.concatenate([st["hist_glass"] for st in stats.values()])
    hr = np.concatenate([st["hist_ring"] for st in stats.values()])
    ha = np.concatenate([st["hist_all"] for st in stats.values()])
    ranges = {"wide: whole frame p0.5/99.5": percentile_range(ha.sum(0)),
              f"narrow: glass+ring clip < {args.clip} in {args.frame_frac:.0%} of frames":
                  fit_narrow_range(hg + hr, args.clip, args.frame_frac)}
    for lo, hi in zip(args.compare[::2], args.compare[1::2]):
        ranges[f"compare ({lo:.0f}, {hi:.0f})"] = (lo, hi)
    print(f"\n== centered ranges over {len(seqs)} sequence(s): {seqs}")
    for name, (lo, hi) in ranges.items():
        d, gr = normalized_diff(hg, hr, lo, hi), frame_clipped(hg + hr, lo, hi)
        print(f"{name}: ({lo:.0f}, {hi:.0f}) width {hi - lo:.0f}")
        print(f"  normalized glass - ring pcts(5..95) {pcts(d)}, median |d| {np.nanmedian(np.abs(d)):.4f}")
        print(f"  glass+ring clipped: pooled {frame_clipped((hg + hr).sum(0), lo, hi):.4f}, frames < {args.clip} "
              f"{np.mean(gr < args.clip):.3f}, worst frame {gr.max():.4f}; "
              f"whole frame clipped: pooled {frame_clipped(ha.sum(0), lo, hi):.4f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--sequences", nargs="*")
    ap.add_argument("--cache_dir", default="data/transpose_cache")
    ap.add_argument("--clip", type=float, default=0.01)
    ap.add_argument("--frame_frac", type=float, default=0.95)
    ap.add_argument("--near_zero", type=float, default=0.5)
    ap.add_argument("--compare", type=float, nargs="*", default=[], help="extra centered ranges: lo hi [lo hi ...]")
    main(ap.parse_args())
