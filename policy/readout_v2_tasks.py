"""
Read-out of step 4 of the second version of the benchmark (results/hard_scenario/bench_v2/README.md): can a policy
start on the tasks T2 (align) and T3 (two)? From the episodes.json files of the final evaluations (no new runs):

  python policy/readout_v2_tasks.py > results/hard_scenario/bench_v2/readout_tasks.txt

T2 `V2_align`: the plain rgbdt baseline and the arm `cue` of the method with the grasp cue, the place cue and the
hindsight angle of each; success per cell, the share of episodes with the glass picked, where the failures are
(outcome labels of the episodes), the angle error at the end. T3 `V2_two`: the plain rgbdt baseline; success per
pair of cells, the share of episodes with one and with both glasses on a coaster. The stop of step 4: a task on
which the plain baseline is below 0.05 and no run of the task reaches 0.30 in any cell. Seed 42; a run without a
finished evaluation is left out and named.
"""
import json
from collections import Counter
from pathlib import Path

import click
import numpy as np

from readout_place_method import SHOWS, ok, paired, picked

START, SOLVABLE = 0.05, 0.30
TASKS = {"V2_align": dict(sets=("j_cells11", "k_cells11_heldout"),
                          runs={"plain rgbdt": "rgbdt_700demos_imagenet", "`cue` with the angles": "rgbdt_700demos_hcg2a_cue"}),
         "V2_two": dict(sets=("j_pairs12", "k_pairs12_heldout"), runs={"plain rgbdt": "rgbdt_700demos_imagenet"})}
STATE = {0: "shows", 1: "blind", 2: "faint"}


def load(run, sets):
    eps = {}
    for ts in sets:
        f = Path(run) / ts / "episodes.json"
        if not f.exists():
            return None
        eps.update({(ts, e["seed"]): e for e in json.loads(f.read_text())})
    return eps


def rate(eps, f=ok, cell=None):
    r = [f(e) for e in eps.values() if cell is None or e["cell"] == cell]
    return float(np.mean(r)) if r else float("nan")


@click.command()
@click.option("--root", default="results/hard_scenario/bench_v2")
def main(root):
    root = Path(root)
    for level, t in TASKS.items():
        runs = {n: load(root / f"{level}_{r}", t["sets"]) for n, r in t["runs"].items()}
        missing = [n for n, r in runs.items() if r is None]
        runs = {n: r for n, r in runs.items() if r is not None}
        print(f"## {level}, seed 42\n")
        if missing:
            print(f"Not finished yet, left out: {missing}\n")
        if not runs:
            continue
        names = list(runs)
        first = runs[names[0]]
        cells = sorted({e["cell"] for e in first.values()})
        print(f"{len(first)} episodes per run on {', '.join(t['sets'])}.\n")
        print("| cell | " + " | ".join(names) + " |\n|---|" + "---|" * len(names))
        table = np.array([[rate(runs[n], cell=c) for n in names] for c in cells])
        for i, c in enumerate(cells):
            label = SHOWS[c] if level == "V2_align" else f"pair {c}"
            print(f"| {c}: {label} | " + " | ".join(f"{x:.3f}" for x in table[i]) + " |")
        print("| macro average | " + " | ".join(f"{np.nanmean(table[:, j]):.3f}" for j in range(len(names))) + " |")
        print("| glass picked | " + " | ".join(f"{rate(runs[n], picked):.3f}" for n in names) + " |")
        if level == "V2_two":
            for k in (1, 2):
                print(f"| {'one glass' if k == 1 else 'both glasses'} on a coaster | "
                      + " | ".join(f"{rate(runs[n], lambda e: e.get('n_placed', 0) >= k):.3f}" for n in names) + " |")
        print()
        for n in names:
            out = Counter(e["outcome"] for e in runs[n].values())
            print(f"{n}, outcomes: " + ", ".join(f"{k} {v}" for k, v in out.most_common()))
            if level == "V2_align":
                err = [e["yaw_error"] for e in runs[n].values() if e.get("picked") and e.get("yaw_error") is not None]
                if err:
                    print(f"  angle between the glass and the mat at the end, episodes with the glass picked: median "
                          f"{np.median(err):.1f} degrees, within 15 degrees in {np.mean(np.array(err) <= 15):.2f}")
        plain = np.nanmean(table[:, names.index("plain rgbdt")]) if "plain rgbdt" in names else float("nan")
        stop = plain < START and np.nanmax(table) < SOLVABLE
        print(f"\nStop of step 4 (the plain baseline below {START} and no run at {SOLVABLE} in any cell): "
              f"{'YES' if stop else 'no'} (plain rgbdt {plain:.3f}, best cell of any run {np.nanmax(table):.3f}).")
        if len(names) == 2:
            keys = sorted(first)
            a, b, p = paired(runs[names[1]], runs[names[0]], keys)
            print(f"{names[1]} against {names[0]} on the same episodes: only the first succeeds in {a}, only the second in {b}, p = {p:.2g}")
        print()


if __name__ == "__main__":
    main()
