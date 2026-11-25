"""
Read-out of round 3 of the fusion method on L0_vis (results/hard_scenario/fusion_vis/README.md, "Round 3": the
ablations of fusion/attention/ablations.MD not run before), from the episodes.json files of the final evaluations
(no new runs):

  python fusion/attention/readout_l0_vis_round3.py > results/hard_scenario/fusion_vis/readout_round3.txt

Printed, in the order of the fixed read-out: (1) success per cell of every run; (2) every run against P9 of seed 42
on the same episodes, with the verdict of the README; (3) against the rgbdt ImageNet baseline of seed 42; (4) where
the gripper closes; (5) the sensor weights by cell and whether they follow what a sensor shows; (6) next to it, not a
criterion: against hindsight cue gating (hcg_full, seed 42). A run without a finished evaluation is left out and named.
"""
from pathlib import Path

import click
import numpy as np

from readout_l0_vis import SHOWS, mark, ok, versus
from readout_l0_vis_round2 import GROUPS, P9, SEEDS, SENSOR_SHOWS, auc, load, macro, pair_lines, placed

RUNS = {   # name in the tables: (id in ablations.MD, run folder)
    "warm": ("P3 on P9", P9 + "_warm"),
    "anchor": ("P4 on P9", P9 + "_anchor"),
    "gatesup": ("P6 on P9", P9 + "_gatesup"),
    "aug": ("P7 on P9", P9 + "_aug"),
    "global": ("P8", "L0_vis_rgbdt_700demos_fusion_global"),
    "stx": ("P11 on P9", P9 + "_stx"),
}
BASE = "L0_vis_rgbdt_700demos_imagenet"
METHOD = "L0_vis_rgbdt_700demos_hcg_full"
FOLLOWS = 0.8      # AUC from which a sensor's weight counts as following what the sensor shows (ablations.MD)
ALL = {"all cells": range(11)}


def weight_auc(A):
    """{sensor: (mean weight where it shows the glass, where it does not, AUC)} of a run evaluated with --save_w."""
    eps = list(A.values())
    if not all("w" in e for e in eps):
        return None
    W = np.array([np.mean(np.asarray(e["w"], float), axis=0) for e in eps])
    cell = np.array([e["cell"] for e in eps])
    out = {}
    for i, (s, cells) in enumerate(SENSOR_SHOWS.items()):
        m = np.isin(cell, cells)
        out[s] = (float(W[m, i].mean()), float(W[~m, i].mean()), auc(W[m, i], W[~m, i]))
    return out


@click.command()
@click.option("--root", default="results/hard_scenario/fusion_vis")
@click.option("--base_root", default="results/hard_scenario/grid_bench")
@click.option("--method_root", default="results/hard_scenario/cue_gate")
def main(root, base_root, method_root):
    root = Path(root)
    p9 = [load(root / (P9 + s)) for s in SEEDS]
    assert all(r is not None for r in p9), "the three seeds of P9 are needed"
    p9_42, base, method = p9[0], load(Path(base_root) / BASE), load(Path(method_root) / METHOD)
    lo, hi = min(macro(r) for r in p9), max(macro(r) for r in p9)
    runs = {n: load(root / f) for n, (_, f) in RUNS.items()}
    missing = [n for n, r in runs.items() if r is None]
    if missing:
        print(f"Not finished, left out: {', '.join(f'{n} ({RUNS[n][1]})' for n in missing)}\n")
    runs = {n: r for n, r in runs.items() if r is not None}
    if not runs:
        return
    assert all(set(r) == set(p9_42) for r in [*runs.values(), base]), "not the same episodes"

    print("## 1. Success per cell (seed 42, 120 episodes per cell)\n")
    print("| cell | sensors that show the glass | " + " | ".join(f"{n} ({RUNS[n][0]})" for n in runs)
          + " | P9, seed 42 | P9, lowest to highest of three seeds | rgbdt ImageNet, seed 42 |")
    print("|---|---|" + "---|" * (len(runs) + 3))
    for c in [*range(11), None]:
        cells = range(11) if c is None else (c,)
        rate = lambda r: versus(r, r, cells)[0]
        span = sorted(rate(r) for r in p9)
        print(f"| {'macro average' if c is None else c} | {'' if c is None else SHOWS[c]} | "
              + " | ".join(f"{rate(r):.3f}" for r in runs.values())
              + f" | {rate(p9_42):.3f} | {span[0]:.3f} to {span[-1]:.3f} | {rate(base):.3f} |")

    print(f"\n## 2. Every run against P9 of seed 42, same episodes (the seeds of P9: macro average {lo:.3f} to "
          f"{hi:.3f})\n")
    print("| run | run | P9 | only the run succeeds / only P9 | p | verdict |\n|---|---|---|---|---|---|")
    for n, r in runs.items():
        sa, sb, a, b, p = versus(r, p9_42, range(11))
        m = macro(r)
        verdict = ("above P9" if p < 0.05 and sa > sb and m > hi else
                   "below P9" if p < 0.05 and sa < sb and m < lo else "not separated from P9")
        print(f"| {n} ({RUNS[n][0]}) | {sa:.3f} | {sb:.3f} | {a} / {b} | {p:.2g} | {verdict} |")
    print()
    for n, r in runs.items():
        pair_lines(n, r, "P9", p9_42, ALL)

    print("\n## 3. Every run against the rgbdt ImageNet baseline of seed 42, same episodes\n")
    for n, r in runs.items():
        pair_lines(n, r, "rgbdt ImageNet", base, GROUPS)

    print("\n## 4. Where the gripper closes (last close of the episode)\n")
    print("| cells | policy | success | closes within 4 cm of the glass | within 1.5 cm | success when within 1.5 cm "
          "| median distance, cm |")
    print("|---|---|---|---|---|---|---|")
    for n, r in [*runs.items(), ("P9", p9_42), ("rgbdt ImageNet", base)]:
        placed(n, r, ALL)

    print("\n## 5. The sensor weights: mean weight in the cells in which the sensor shows the glass / in the others, "
          f"AUC; a weight follows what its sensor shows from AUC {FOLLOWS}\n")
    print("| run | rgb | depth | thermal | sensors whose weight follows |\n|---|---|---|---|---|")
    for n, r in [*runs.items(), ("P9", p9_42)]:
        w = weight_auc(r)
        if w is None:
            print(f"| {n} | no weights saved | | | |")
            continue
        follows = [s for s, v in w.items() if v[2] >= FOLLOWS]
        print(f"| {n} | " + " | ".join(f"{v[0]:.3f} / {v[1]:.3f}, {v[2]:.2f}" for v in w.values())
              + f" | {', '.join(follows) or 'none'} |")

    print("\n## 6. Next to it, not a criterion: hindsight cue gating (hcg_full, seed 42) on the same episodes\n")
    if method is None or set(method) != set(p9_42):
        print("the method's run was not found")
    else:
        print("| run | run | hcg_full | only the run succeeds / only hcg_full | p | cells with the run above / below "
              "hcg_full (p < 0.05) |\n|---|---|---|---|---|---|")
        for n, r in [*runs.items(), ("P9", p9_42)]:
            sa, sb, a, b, p = versus(r, method, range(11))
            per = [(c, *versus(r, method, (c,))) for c in range(11)]
            up = [c for c, xa, xb, _, _, pp in per if pp < 0.05 and xa > xb]
            down = [c for c, xa, xb, _, _, pp in per if pp < 0.05 and xa < xb]
            print(f"| {n} | {sa:.3f} | {sb:.3f} | {a} / {b} | {p:.2g}{mark(p, sa - sb)} | {up or 'none'} / "
                  f"{down or 'none'} |")


if __name__ == "__main__":
    main()
