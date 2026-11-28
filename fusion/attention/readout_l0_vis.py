"""
Read-out of the L0_vis runs of the fusion method (results/hard_scenario/fusion_vis/README.md), from the episodes.json
files of the final evaluations (no new runs):

  python fusion/attention/readout_l0_vis.py > results/hard_scenario/fusion_vis/readout.txt

Printed: (1) success per cell of the three runs next to the rgbdt baseline of the same seed, with the frozen best
baseline and reference of each cell; (2) where the gripper closes (the last close, as policy/analyze_failures.py);
(3) P0 against P5 over all cells and over cells 1 to 9; (4) the mean weight of each sensor by cell.
"""
import json
from math import comb
from pathlib import Path

import click
import numpy as np

TEST_SETS = ("j_cells11", "k_cells11_heldout")
RUNS = {"P0 fusion": "L0_vis_rgbdt_700demos_fusion", "P9 fusion_kp": "L0_vis_rgbdt_700demos_fusion_kp",
        "P5 fusion_static": "L0_vis_rgbdt_700demos_fusion_static"}
BASE = "L0_vis_rgbdt_700demos_imagenet"
# Frozen in results/hard_scenario/grid_bench/README.md, stage 6: best baseline of the cell (three seeds), reference
BEST = [0.781, 0.758, 0.397, 0.700, 0.189, 0.697, 0.256, 0.181, 0.486, 0.094, 0.503]
REF = [0.900, 0.875, 0.642, 0.867, 0.608, 0.858, 0.492, 0.333, 0.742, 0.383, 0.717]
SHOWS = ["rgb+depth+thermal", "depth+thermal", "rgb+thermal", "rgb+depth", "thermal", "depth", "rgb", "rgb (faint)",
         "depth (faint)", "thermal (faint)", "all faint"]
GROUPS = {"all cells": range(11), "depth shows (0, 1, 3, 5, 8, 10)": (0, 1, 3, 5, 8, 10),
          "depth blind (2, 4, 6, 7, 9)": (2, 4, 6, 7, 9)}


def load(folder):
    """{(test set, seed): episode} of a run folder."""
    eps = {}
    for t in TEST_SETS:
        for e in json.loads((Path(folder) / t / "episodes.json").read_text()):
            eps[t, e["seed"]] = e
    return eps


def ok(e):
    return float(e["reward"]) > 0.5


def sign_p(a, b):
    """Two-sided sign test on the episodes in which exactly one of the two succeeds."""
    n, k = a + b, min(a, b)
    return 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def versus(A, B, cells):
    keys = [k for k in A if A[k]["cell"] in cells]
    a = sum(ok(A[k]) and not ok(B[k]) for k in keys)
    b = sum(ok(B[k]) and not ok(A[k]) for k in keys)
    return np.mean([ok(A[k]) for k in keys]), np.mean([ok(B[k]) for k in keys]), a, b, sign_p(a, b)


def mark(p, d):
    return (" (above)" if d > 0 else " (below)") if p < 0.05 else ""


@click.command()
@click.option("--root", default="results/hard_scenario/fusion_vis")
@click.option("--base_root", default="results/hard_scenario/grid_bench")
def main(root, base_root):
    runs = {n: load(Path(root) / f) for n, f in RUNS.items()}
    base = load(Path(base_root) / BASE)
    assert all(set(r) == set(base) for r in runs.values()), "the runs were not evaluated on the same episodes"

    print(f"## 1. Success per cell (120 episodes per cell and run; seed 42; against {BASE} on the same episodes, "
          "sign test, marked where p < 0.05)\n")
    print("| cell | sensors that show the glass | " + " | ".join(runs) + " | rgbdt ImageNet, seed 42 | "
          "best baseline (3 seeds, frozen) | reference |")
    print("|---|---|" + "---|" * (len(runs) + 3))
    for c in [*range(11), None]:
        cells = range(11) if c is None else (c,)
        row = []
        for r in runs.values():
            sa, sb, a, b, p = versus(r, base, cells)
            row.append(f"{sa:.3f}{mark(p, sa - sb)}")
        sb = versus(base, base, cells)[0]
        tail = (f"{np.mean(BEST):.3f} | {np.mean(REF):.3f}" if c is None else f"{BEST[c]:.3f} | {REF[c]:.3f}")
        print(f"| {'macro average' if c is None else c} | {'' if c is None else SHOWS[c]} | " + " | ".join(row)
              + f" | {sb:.3f} | {tail} |")
    print()
    for n, r in runs.items():
        sa, sb, a, b, p = versus(r, base, range(11))
        print(f"{n} against the baseline, all episodes: {sa:.3f} against {sb:.3f}, only the run succeeds {a}, "
              f"only the baseline {b}, p = {p:.2g}")

    print("\n## 2. Where the gripper closes (last close of the episode; no close counts as not there)\n")
    print("| cells | policy | success | closes within 4 cm of the glass | within 1.5 cm | success when within 1.5 cm | "
          "median distance, cm |")
    print("|---|---|---|---|---|---|---|")
    for g, cells in GROUPS.items():
        for n, r in {**runs, "rgbdt ImageNet, seed 42": base}.items():
            eps = [e for e in r.values() if e["cell"] in cells]
            d = np.array([100 * e["closes"][-1]["d_target"] if e["closes"] else np.inf for e in eps])
            s = np.array([ok(e) for e in eps])
            print(f"| {g} | {n} | {s.mean():.3f} | {(d < 4).mean():.3f} | {(d < 1.5).mean():.3f} | "
                  f"{s[d < 1.5].mean():.3f} | {np.median(d):.2f} |")
    a, b = runs["P9 fusion_kp"], runs["P0 fusion"]
    sa, sb, na, nb, p = versus(a, b, range(11))
    print(f"\nP9 against P0, all episodes: {sa:.3f} against {sb:.3f}, only P9 succeeds {na}, only P0 {nb}, p = {p:.2g}")

    print("\n## 3. P0 (learned weights) against P5 (weights fixed at 1/3), same episodes\n")
    a, b = runs["P0 fusion"], runs["P5 fusion_static"]
    for g, cells in {"all cells": range(11), "cells 1 to 9": range(1, 10)}.items():
        sa, sb, na, nb, p = versus(a, b, cells)
        print(f"{g}: P0 {sa:.3f}, P5 {sb:.3f}, only P0 succeeds {na}, only P5 {nb}, p = {p:.2g}")

    print("\n## 4. Mean weight of each sensor during the evaluation, by cell (rgb / depth / thermal; mean over the "
          "decisions of an episode, then over the 120 episodes)\n")
    print("| cell | sensors that show the glass | " + " | ".join(runs) + " |")
    print("|---|---|" + "---|" * len(runs))
    for c in range(11):
        row = []
        for r in runs.values():
            w = np.array([np.mean(np.asarray(e["w"], float), axis=0) for e in r.values() if e["cell"] == c])
            row.append(" / ".join(f"{v:.2f}" for v in w.mean(0)))
        print(f"| {c} | {SHOWS[c]} | " + " | ".join(row) + " |")


if __name__ == "__main__":
    main()
