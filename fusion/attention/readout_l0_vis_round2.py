"""
Read-out of round 2 of the fusion method on L0_vis (results/hard_scenario/fusion_vis/README.md, "Round 2"), from the
episodes.json files of the final evaluations (no new runs):

  python fusion/attention/readout_l0_vis_round2.py > results/hard_scenario/fusion_vis/readout_round2.txt

Printed, in the order of the fixed read-out: (1) P9 with three seeds against the rgbdt baseline and against the best
baseline of each cell, seeds pooled; (2) P9 without the segmentation pre-training; (3) P9 with the pre-trained network
frozen, with its sensor weights; (4) P9 without the thermal image. A run without a finished evaluation is left out.
"""
import json
from pathlib import Path

import click
import numpy as np

from readout_l0_vis import BEST, REF, SHOWS, TEST_SETS, mark, ok, sign_p, versus

SEEDS = ("", "_seed43", "_seed44")   # seeds 42, 43, 44
P9 = "L0_vis_rgbdt_700demos_fusion_kp"
SCRATCH, FROZEN, RGBD = P9 + "_scratch", P9 + "_frozen", "L0_vis_rgbd_700demos_fusion_kp"
# Stage 6 of results/hard_scenario/grid_bench/README.md: which baseline is the best of each cell
WHICH = ["rgbdt", "rgbdt", "rgbt", "rgbd", "rgbt", "rgbd", "rgb", "rgb", "rgbdt", "rgbt", "rgbdt"]
# The cells in which a sensor shows the glass, fully or faintly (order of the saved weights: rgb, depth, thermal)
SENSOR_SHOWS = {"rgb": (0, 2, 3, 6, 7, 10), "depth": (0, 1, 3, 5, 8, 10), "thermal": (0, 1, 2, 4, 9, 10)}
GROUPS = {"all cells": range(11), "depth shows (0, 1, 3, 5, 8, 10)": (0, 1, 3, 5, 8, 10),
          "depth blind (2, 4, 6, 7, 9)": (2, 4, 6, 7, 9)}


def load(folder):
    """{(test set, seed): episode} of a run folder, or None while its final evaluation is not complete."""
    eps = {}
    for t in TEST_SETS:
        f = Path(folder) / t / "episodes.json"
        if not f.exists() or not (Path(folder) / t / "eval_log.json").exists():
            return None
        for e in json.loads(f.read_text()):
            eps[t, e["seed"]] = e
    return eps


def pooled(As, Bs, cells):
    """Several pairs of runs (one per seed) on the same episodes: the two rates, the episodes only one solves, p."""
    ra, rb, a, b = [], [], 0, 0
    for A, B in zip(As, Bs):
        assert set(A) == set(B), "not the same episodes"
        keys = [k for k in A if A[k]["cell"] in cells]
        ra += [ok(A[k]) for k in keys]
        rb += [ok(B[k]) for k in keys]
        a += sum(ok(A[k]) and not ok(B[k]) for k in keys)
        b += sum(ok(B[k]) and not ok(A[k]) for k in keys)
    return np.mean(ra), np.mean(rb), a, b, sign_p(a, b)


def macro(A):
    return np.mean([versus(A, A, (c,))[0] for c in range(11)])


def auc(pos, neg):
    pos, neg = np.asarray(pos), np.asarray(neg)
    return float((pos[:, None] > neg[None]).mean() + 0.5 * (pos[:, None] == neg[None]).mean())


def pair_lines(na, A, nb, B, groups):
    for g, cells in groups.items():
        sa, sb, a, b, p = versus(A, B, cells)
        print(f"{na} against {nb}, {g}: {sa:.3f} against {sb:.3f}, only {na} succeeds {a}, only {nb} {b}, "
              f"p = {p:.2g}{mark(p, sa - sb)}")
    cells = []
    for c in range(11):
        sa, sb, a, b, p = versus(A, B, (c,))
        if p < 0.05:
            cells.append(f"cell {c} {sa:.3f} against {sb:.3f} (p = {p:.2g})")
    print(f"  cells that differ with p < 0.05: {'; '.join(cells) or 'none'}")


def placed(name, A, groups=GROUPS):
    for g, cells in groups.items():
        eps = [e for e in A.values() if e["cell"] in cells]
        d = np.array([100 * e["closes"][-1]["d_target"] if e["closes"] else np.inf for e in eps])
        s = np.array([ok(e) for e in eps])
        print(f"| {g} | {name} | {s.mean():.3f} | {(d < 4).mean():.3f} | {(d < 1.5).mean():.3f} | "
              f"{s[d < 1.5].mean() if (d < 1.5).any() else float('nan'):.3f} | {np.median(d):.2f} |")


def weights(name, A):
    eps = list(A.values())
    W = np.array([np.mean(np.asarray(e["w"], float), axis=0) for e in eps])
    cell = np.array([e["cell"] for e in eps])
    print(f"\n{name}: mean weight per cell (rgb / depth / thermal; mean over the decisions of an episode, then over "
          "the 120 episodes)\n")
    print("| cell | sensors that show the glass | rgb / depth / thermal |")
    print("|---|---|---|")
    for c in range(11):
        print(f"| {c} | {SHOWS[c]} | " + " / ".join(f"{v:.3f}" for v in W[cell == c].mean(0)) + " |")
    for i, (s, cells) in enumerate(SENSOR_SHOWS.items()):
        m = np.isin(cell, cells)
        print(f"{name}, {s}: weight {W[m, i].mean():.3f} in the cells in which it shows the glass, "
              f"{W[~m, i].mean():.3f} in the others, AUC {auc(W[m, i], W[~m, i]):.2f}")


@click.command()
@click.option("--root", default="results/hard_scenario/fusion_vis")
@click.option("--base_root", default="results/hard_scenario/grid_bench")
def main(root, base_root):
    root, base_root = Path(root), Path(base_root)
    p9 = [load(root / (P9 + s)) for s in SEEDS]
    base = [load(base_root / ("L0_vis_rgbdt_700demos_imagenet" + s)) for s in SEEDS]
    best = {w: [load(base_root / (f"L0_vis_{w}_700demos_imagenet" + s)) for s in SEEDS] for w in sorted(set(WHICH))}
    have = [i for i, r in enumerate(p9) if r is not None]

    print(f"## 1. P9 with three seeds (seeds found: {', '.join(str(42 + i) for i in have)}; 120 episodes per cell "
          "and seed). Marks: the seeds pooled against the same seeds of the other policy on the same episodes, sign "
          "test, p < 0.05\n")
    A = [p9[i] for i in have]
    print("| cell | sensors that show the glass | " + " | ".join(f"P9 seed {42 + i}" for i in have)
          + " | P9 mean (against rgbdt ImageNet) | rgbdt ImageNet, same seeds | P9 against the best baseline of the "
          "cell | best baseline of the cell, same seeds | the same, frozen | reference |")
    print("|---|---|" + "---|" * (len(have) + 6))
    below_base, below_best, above_best = [], [], []
    for c in range(11):
        sa, sb, a, b, p = pooled(A, [base[i] for i in have], (c,))
        ta, tb, a2, b2, p2 = pooled(A, [best[WHICH[c]][i] for i in have], (c,))
        if p < 0.05 and sa < sb:
            below_base.append(c)
        if p2 < 0.05:
            (below_best if ta < tb else above_best).append(c)
        print(f"| {c} | {SHOWS[c]} | " + " | ".join(f"{versus(r, r, (c,))[0]:.3f}" for r in A)
              + f" | {sa:.3f}{mark(p, sa - sb)} | {sb:.3f} | {a2} / {b2}, p = {p2:.2g}{mark(p2, ta - tb)} | "
              f"{tb:.3f} ({WHICH[c]}) | {BEST[c]:.3f} | {REF[c]:.3f} |")
    sa, sb, a, b, p = pooled(A, [base[i] for i in have], range(11))
    best_mean = np.mean([pooled([best[WHICH[c]][i] for i in have], A, (c,))[0] for c in range(11)])
    print("| macro average | | " + " | ".join(f"{macro(r):.3f}" for r in A)
          + f" | {sa:.3f}{mark(p, sa - sb)} | {sb:.3f} | | {best_mean:.3f} | {np.mean(BEST):.3f} | {np.mean(REF):.3f} |")
    print(f"\nP9 against the rgbdt ImageNet baseline, seeds pooled, all episodes: {sa:.3f} against {sb:.3f}, only P9 "
          f"succeeds {a}, only the baseline {b}, p = {p:.2g}")
    print(f"Cells in which P9 is below the rgbdt ImageNet baseline with p < 0.05: {below_base or 'none'}")
    print(f"Cells in which P9 is below the best baseline of the cell with p < 0.05: {below_best or 'none'}; above: "
          f"{above_best or 'none'}")

    p9_42, base_42 = p9[0], base[0]
    runs = {n: load(root / f) for n, f in (("scratch", SCRATCH), ("frozen", FROZEN), ("rgbd", RGBD))}

    print("\n## 2. Pre-training: P9 without the segmentation pre-training (fusion_kp_scratch), seed 42\n")
    if runs["scratch"] is None:
        print("not finished")
    else:
        pair_lines("P9", p9_42, "scratch", runs["scratch"], {"all cells": range(11)})
        pair_lines("scratch", runs["scratch"], "rgbdt ImageNet", base_42, {"all cells": range(11)})
        print("\n| cells | policy | success | closes within 4 cm of the glass | within 1.5 cm | success when within "
              "1.5 cm | median distance, cm |")
        print("|---|---|---|---|---|---|---|")
        for n, r in (("scratch", runs["scratch"]), ("P9", p9_42), ("rgbdt ImageNet", base_42)):
            placed(n, r)

    print("\n## 3. Frozen encoder: P9 with the pre-trained network frozen (fusion_kp_frozen), seed 42\n")
    if runs["frozen"] is None:
        print("not finished")
    else:
        pair_lines("frozen", runs["frozen"], "P9", p9_42, {"all cells": range(11)})
        pair_lines("frozen", runs["frozen"], "rgbdt ImageNet", base_42, {"all cells": range(11)})
        weights("frozen", runs["frozen"])
        weights("P9", p9_42)

    print("\n## 4. Thermal: P9 against P9 without the thermal image (L0_vis_rgbd_700demos_fusion_kp), seed 42\n")
    if runs["rgbd"] is None:
        print("not finished")
    else:
        pair_lines("P9", p9_42, "P9 without thermal", runs["rgbd"],
                   {"all cells": range(11), "cells 2, 4, 9 (thermal shows the glass, depth does not)": (2, 4, 9),
                    "cells 3, 5, 6, 7, 8 (thermal does not show it)": (3, 5, 6, 7, 8)})

    done = {n: r for n, r in runs.items() if r is not None}
    if done:
        print("\n## Success per cell of the runs of seed 42\n")
        print("| cell | sensors that show the glass | P9 | " + " | ".join(done) + " | rgbdt ImageNet |")
        print("|---|---|---|" + "---|" * (len(done) + 1))
        for c in [*range(11), None]:
            cells = range(11) if c is None else (c,)
            row = [f"{versus(r, r, cells)[0]:.3f}" for r in [p9_42, *done.values(), base_42]]
            print(f"| {'macro average' if c is None else c} | {'' if c is None else SHOWS[c]} | " + " | ".join(row)
                  + " |")


if __name__ == "__main__":
    main()
