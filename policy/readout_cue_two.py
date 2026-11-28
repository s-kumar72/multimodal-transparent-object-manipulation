"""
The method's cue for two targets (T3) on the training demos, out of fold, no policy involved
(results/hard_scenario/bench_v2/README.md, section 14):

  python policy/readout_cue_two.py > results/hard_scenario/bench_v2/fits/V2b_two_700/readout_cue_two.txt

Reads the cue of every demo as its fit wrote it for the demos it was not fitted on (train_cue.npy of
data/hcue/V2b_two_700_g1, _g2: a map and a gate per sensor; _g1_early, _g2_early: one network on all sensors; _r1,
_r2: the two release cues) and the hindsight points of the demos (policy/hindsight_cue_nth.py). A cue is right when
it is within 1.5 cm of its own point on the table. Reported per group of pair-cells and per pair-cell:

  first, second    the cue of the first / second grasp is right
  both             both are
  same glass       both cues are within 1.5 cm of one and the same grasp point (one glass twice, the other nowhere)
  swapped          the first cue is at the second point and the second at the first

and the two designs against each other on "both" (same demos, two-sided sign test). Groups: the same sensors show
both glasses (pair-cells 0 to 3 and 10), different sensors show them (4 to 9), the first fully and the second faintly
(11).
"""
from math import comb
from pathlib import Path

import click
import numpy as np
import zarr

from policy.hindsight_cue_nth import nth_goal

GROUPS = {"the same sensors show both (0-3, 10)": (0, 1, 2, 3, 10), "different sensors show them (4-9)": (4, 5, 6, 7, 8, 9),
          "the first fully, the second faintly (11)": (11,)}
NEAR = 1.5      # cm


def sign_p(a, b):
    n, k = a + b, min(a, b)
    return 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def near(cue, point):
    return 100 * np.linalg.norm(cue[:, :2] - point[:, :2], axis=1) < NEAR


def flags(g1, g2, p1, p2):
    a, b = near(g1, p1), near(g2, p2)
    same = (near(g1, p1) & near(g2, p1)) | (near(g1, p2) & near(g2, p2))
    return dict(first=a, second=b, both=a & b, same=same, swapped=near(g1, p2) & near(g2, p1))


@click.command()
@click.option("--demos", default="results/data/sim_demos_V2b_two_700.zarr")
@click.option("--stem", default="data/hcue/V2b_two_700")
def main(demos, stem):
    cells = zarr.open(demos, "r")["meta/ep_cell"][:]
    p1, p2 = nth_goal(demos, "grasp", None, 0), nth_goal(demos, "grasp", None, 1)
    cue = lambda name: np.load(f"{stem}_{name}/train_cue.npy")
    arms = {"a map and a gate per sensor": ("g1", "g2"), "one network": ("g1_early", "g2_early")}
    arms = {k: v for k, v in arms.items() if all(Path(f"{stem}_{n}/train_cue.npy").exists() for n in v)}
    f = {k: flags(cue(a), cue(b), p1, p2) for k, (a, b) in arms.items()}
    d = 100 * np.linalg.norm(p1[:, :2] - p2[:, :2], axis=1)
    print(f"# The cue for two targets on the training demos, out of fold: {len(cells)} demos of {demos}\n")
    print(f"The two grasp points of a demo are {d.min():.1f} to {d.max():.1f} cm apart (median {np.median(d):.1f}). "
          f"A cue is right within {NEAR} cm of its own point.\n")
    rows = [(k, np.isin(cells, v)) for k, v in GROUPS.items()] + [("all", np.ones(len(cells), bool))] \
        + [(f"pair-cell {c}", cells == c) for c in sorted(set(cells.tolist()))]
    for k, x in f.items():
        print(f"## {k}\n\n| demos | n | first | second | both | same glass | swapped |\n|---|---|---|---|---|---|---|")
        for name, m in rows:
            print(f"| {name} | {int(m.sum())} | " + " | ".join(f"{x[c][m].mean():.2f}" for c in ("first", "second", "both", "same", "swapped")) + " |")
        print()
    if len(f) == 2:
        (ka, a), (kb, b) = f.items()
        print(f"## Both cues right: {ka} against {kb}\n\n| demos | {ka} | {kb} | only the first design right / only the second | p |\n|---|---|---|---|---|")
        for name, m in rows[:4]:
            x, y = a["both"][m], b["both"][m]
            u, v = int((x & ~y).sum()), int((y & ~x).sum())
            print(f"| {name} | {x.mean():.2f} | {y.mean():.2f} | {u} / {v} | {sign_p(u, v):.2g} |")
        print()
    rel = [n for n in ("r1", "r2") if Path(f"{stem}_{n}/train_cue.npy").exists()]
    if rel:
        import policy.hindsight_cue as H
        table_z = H.camera_params("V2b_two")["table_z"]
        print("## The release cues (a map and a gate per sensor; shared by both designs)\n\n| demos | " + " | ".join(
            f"{'first' if n == 'r1' else 'second'} release right" for n in rel) + " |\n|---|" + "---|" * len(rel))
        ok = [near(cue(n), nth_goal(demos, "release", table_z, int(n[1]) - 1)) for n in rel]
        for name, m in rows[:4]:
            print(f"| {name} | " + " | ".join(f"{o[m].mean():.2f}" for o in ok) + " |")


if __name__ == "__main__":
    main()
