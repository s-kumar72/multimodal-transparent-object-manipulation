"""
Read-out of the round-1 runs of hindsight cue gating (results/hard_scenario/cue_gate/README.md, items 1, 2 and 4),
from the episodes.json files of the final evaluations (no new runs):

  python policy/readout_cue_gate.py [--fusion_root DIR] > results/hard_scenario/cue_gate/readout.txt

Printed: (1) success per cell of each arm, against the rgbdt baseline of the same seed over all episodes and against
the seed-42 run of the best baseline of each cell (Stage 6 table), sign test on the same episodes; (2) the arms
against each other (full-cue, full-gate, cue-blind); (4) the references and the runs of the learned attention
weighting (extra supervision) on the same episodes, next to them; and, outside the fixed read-out, where the gripper
closes. Item 3 is policy/analyze_cue.py. Arms without a finished evaluation are left out.
"""
import json
from math import comb
from pathlib import Path

import click
import numpy as np

TEST_SETS = ("j_cells11", "k_cells11_heldout")
ARMS = {"full": "L0_vis_rgbdt_700demos_hcg_full", "cue": "L0_vis_rgbdt_700demos_hcg_cue",
        "gate": "L0_vis_rgbdt_700demos_hcg_gate", "blind": "L0_vis_rgbdt_700demos_hcg_blind"}
PAIRS = (("full", "cue"), ("full", "gate"), ("cue", "blind"))
BASE = "L0_vis_rgbdt_700demos_imagenet"
# Frozen in results/hard_scenario/grid_bench/README.md, stage 6: the best baseline of each cell (three seeds), which
# baseline that is, and the reference of the cell
BEST = [0.781, 0.758, 0.397, 0.700, 0.189, 0.697, 0.256, 0.181, 0.486, 0.094, 0.503]
WHICH = ["rgbdt", "rgbdt", "rgbt", "rgbd", "rgbt", "rgbd", "rgb", "rgb", "rgbdt", "rgbt", "rgbdt"]
REF = [0.900, 0.875, 0.642, 0.867, 0.608, 0.858, 0.492, 0.333, 0.742, 0.383, 0.717]
SHOWS = ["rgb+depth+thermal", "depth+thermal", "rgb+thermal", "rgb+depth", "thermal", "depth", "rgb", "rgb (faint)",
         "depth (faint)", "thermal (faint)", "all faint"]
FUSION = {"P0 fusion": "L0_vis_rgbdt_700demos_fusion", "P9 fusion_kp": "L0_vis_rgbdt_700demos_fusion_kp",
          "P5 fusion_static": "L0_vis_rgbdt_700demos_fusion_static"}
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
@click.option("--root", default="results/hard_scenario/cue_gate")
@click.option("--base_root", default="results/hard_scenario/grid_bench")
@click.option("--fusion_root", default="results/hard_scenario/fusion_vis")
def main(root, base_root, fusion_root):
    runs = {n: load(Path(root) / f) for n, f in ARMS.items()}
    missing = [n for n, r in runs.items() if r is None]
    runs = {n: r for n, r in runs.items() if r is not None}
    base = load(Path(base_root) / BASE)
    best = {w: load(Path(base_root) / f"L0_vis_{w}_700demos_imagenet") for w in sorted(set(WHICH))}
    assert all(set(r) == set(base) for r in [*runs.values(), *best.values()]), "not the same episodes"
    if missing:
        print(f"Not finished yet, left out: {', '.join(missing)}\n")

    print(f"## 1. Success per cell (120 episodes per cell and run; seed 42). Marks: against the seed-42 run of the "
          "best baseline of the cell on the same episodes, sign test, p < 0.05; in the last row against "
          f"{BASE} over all episodes\n")
    print("| cell | sensors that show the glass | " + " | ".join(runs) + " | best baseline of the cell, its seed-42 run "
          "| the same, 3 seeds (frozen) | rgbdt ImageNet, seed 42 | reference |")
    print("|---|---|" + "---|" * (len(runs) + 4))
    below = {n: [] for n in runs}
    for c in range(11):
        row = []
        for n, r in runs.items():
            sa, sb, a, b, p = versus(r, best[WHICH[c]], (c,))
            row.append(f"{sa:.3f}{mark(p, sa - sb)}")
            if p < 0.05 and sa < sb:
                below[n].append(c)
        sb = versus(best[WHICH[c]], base, (c,))[0]
        print(f"| {c} | {SHOWS[c]} | " + " | ".join(row) + f" | {sb:.3f} ({WHICH[c]}) | {BEST[c]:.3f} | "
              f"{versus(base, base, (c,))[0]:.3f} | {REF[c]:.3f} |")
    row = []
    for r in runs.values():
        sa, sb, a, b, p = versus(r, base, range(11))
        row.append(f"{sa:.3f}{mark(p, sa - sb)}")
    seed42 = np.mean([versus(best[WHICH[c]], base, (c,))[0] for c in range(11)])
    print("| macro average | | " + " | ".join(row) + f" | {seed42:.3f} | {np.mean(BEST):.3f} | "
          f"{versus(base, base, range(11))[0]:.3f} | {np.mean(REF):.3f} |")
    print()
    for n, r in runs.items():
        sa, sb, a, b, p = versus(r, base, range(11))
        print(f"{n} against {BASE}, all episodes: {sa:.3f} against {sb:.3f}, only the arm succeeds {a}, only the "
              f"baseline {b}, p = {p:.2g}; cells below their best baseline with p < 0.05: {below[n] or 'none'}")
    print("\nPer cell against the seed-42 run of the best baseline of the cell (only the arm succeeds / only the "
          "baseline, p):\n")
    print("| cell | best baseline | " + " | ".join(runs) + " |")
    print("|---|---|" + "---|" * len(runs))
    for c in range(11):
        row = []
        for r in runs.values():
            sa, sb, a, b, p = versus(r, best[WHICH[c]], (c,))
            row.append(f"{a} / {b}, p = {p:.2g}")
        print(f"| {c} | {WHICH[c]} | " + " | ".join(row) + " |")

    print("\n## 2. The arms against each other, same episodes\n")
    for x, y in PAIRS:
        if x not in runs or y not in runs:
            print(f"{x} against {y}: not finished yet")
            continue
        sa, sb, a, b, p = versus(runs[x], runs[y], range(11))
        print(f"{x} against {y}, all episodes: {sa:.3f} against {sb:.3f}, only {x} succeeds {a}, only {y} {b}, "
              f"p = {p:.2g}")
        cells = []
        for c in range(11):
            sa, sb, a, b, p = versus(runs[x], runs[y], (c,))
            if p < 0.05:
                cells.append(f"cell {c} {sa:.3f} against {sb:.3f} (p = {p:.2g})")
        print(f"  cells that differ with p < 0.05: {'; '.join(cells) or 'none'}")

    print("\n## 4. Next to them, not criteria: the references and the learned attention weighting (extra "
          "supervision: segmentation pre-training on simulator masks), seed 42, same episodes\n")
    fus = {n: load(Path(fusion_root) / f) for n, f in FUSION.items()}
    fus = {n: r for n, r in fus.items() if r is not None and set(r) == set(base)}
    if not fus:
        print(f"(no runs of the learned attention weighting found under {fusion_root})\n")
    print("| cell | sensors that show the glass | " + " | ".join([*runs, *fus]) + " | reference |")
    print("|---|---|" + "---|" * (len(runs) + len(fus) + 1))
    for c in [*range(11), None]:
        cells = range(11) if c is None else (c,)
        row = [f"{versus(r, base, cells)[0]:.3f}" for r in [*runs.values(), *fus.values()]]
        print(f"| {'macro average' if c is None else c} | {'' if c is None else SHOWS[c]} | " + " | ".join(row)
              + f" | {np.mean(REF) if c is None else REF[c]:.3f} |")
    print()
    for n, r in runs.items():
        for m, f in fus.items():
            sa, sb, a, b, p = versus(r, f, range(11))
            print(f"{n} against {m}, all episodes: {sa:.3f} against {sb:.3f}, only {n} succeeds {a}, only {m} {b}, "
                  f"p = {p:.2g}")

    print("\n## Outside the fixed read-out: where the gripper closes (last close of the episode; no close counts as "
          "not there)\n")
    print("| cells | policy | success | closes within 4 cm of the glass | within 1.5 cm | success when within 1.5 cm | "
          "median distance, cm |")
    print("|---|---|---|---|---|---|---|")
    for g, cells in GROUPS.items():
        for n, r in {**runs, "rgbdt ImageNet, seed 42": base}.items():
            eps = [e for e in r.values() if e["cell"] in cells]
            d = np.array([100 * e["closes"][-1]["d_target"] if e["closes"] else np.inf for e in eps])
            s = np.array([ok(e) for e in eps])
            print(f"| {g} | {n} | {s.mean():.3f} | {(d < 4).mean():.3f} | {(d < 1.5).mean():.3f} | "
                  f"{s[d < 1.5].mean() if (d < 1.5).any() else float('nan'):.3f} | {np.median(d):.2f} |")


if __name__ == "__main__":
    main()
