"""
Read-out of the pilot of the second version of the benchmark (results/hard_scenario/bench_v2/README.md, step 3), from
the episodes.json files of the final evaluations (no new runs):

  python policy/readout_v2_pilot.py > results/hard_scenario/bench_v2/readout_pilot.txt

For each of the two draft levels of task T1 (V2_place: the thermal states measured on real glass; V2_place_warm: those
of the first version) and the four runs of seed 42 (plain rgbdt, plain rgbd, the arm `cue` with both cues, the same
with the grasp cue of one network): success per cell, the share of episodes with the glass picked, the two stops of
the README (a cell below 0.30 for every run; every cell of every run above 0.95), `cue` against plain rgbdt and
against the one network on the same episodes. Then the two levels side by side. A run without a finished evaluation
is left out and named.
"""
from pathlib import Path

import click
import numpy as np

from readout_place_method import SHOWS, load, ok, paired, picked, rate

LEVELS = {"V2_place": "thermal as measured on real glass", "V2_place_warm": "thermal as in the first version"}
RUNS = {"plain rgbdt": "rgbdt_700demos_imagenet", "plain rgbd": "rgbd_700demos_imagenet",
        "`cue`": "rgbdt_700demos_hcg2_cue", "one network": "rgbdt_700demos_hcg2_cue_early"}
SOLVABLE, SATURATED = 0.30, 0.95


@click.command()
@click.option("--root", default="results/hard_scenario/bench_v2")
def main(root):
    root, macro = Path(root), {}
    for level, what in LEVELS.items():
        runs = {n: load(root / f"{level}_{r}") for n, r in RUNS.items()}
        missing = [n for n, r in runs.items() if r is None]
        runs = {n: r for n, r in runs.items() if r is not None}
        print(f"## {level} ({what}), seed 42, 1320 episodes per run\n")
        if missing:
            print(f"Not finished yet, left out: {missing}\n")
        if not runs:
            continue
        names = list(runs)
        print("| cell | sensors that show the glass | " + " | ".join(names) + " |\n|---|---|" + "---|" * len(names))
        cells = np.array([[rate(runs[n], cells=(c,)) for n in names] for c in range(11)])
        for c in range(11):
            print(f"| {c} | {SHOWS[c]} | " + " | ".join(f"{x:.3f}" for x in cells[c]) + " |")
        print("| macro average | | " + " | ".join(f"{rate(runs[n]):.3f}" for n in names) + " |")
        print("| glass picked | | " + " | ".join(f"{rate(runs[n], picked):.3f}" for n in names) + " |")
        macro[level] = {n: rate(runs[n]) for n in names}
        low = [c for c in range(11) if cells[c].max() < SOLVABLE]
        print(f"\nStops: cells in which every run is below {SOLVABLE}: {low or 'none'}; every cell of every run above "
              f"{SATURATED}: {'yes' if cells.min() > SATURATED else 'no'} (lowest cell of the best run: "
              f"{cells[:, int(np.argmax(cells.mean(0)))].min():.3f}).\n")
        keys = sorted(next(iter(runs.values())))
        for x, y in (("`cue`", "plain rgbdt"), ("`cue`", "one network"), ("plain rgbdt", "plain rgbd")):
            if x in runs and y in runs:
                a, b, p = paired(runs[x], runs[y], keys)
                print(f"{x} against {y}: {rate(runs[x]):.3f} against {rate(runs[y]):.3f}, only the first succeeds in {a}, "
                      f"only the second in {b}, p = {p:.2g}")
        print()
    if len(macro) == 2:
        print("## The two levels side by side (macro average)\n")
        a, b = LEVELS
        print(f"| run | {a} | {b} |\n|---|---|---|")
        for n in RUNS:
            print(f"| {n} | " + " | ".join(f"{macro[l][n]:.3f}" if n in macro[l] else "-" for l in (a, b)) + " |")


if __name__ == "__main__":
    main()
