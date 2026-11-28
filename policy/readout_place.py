"""
Read-out of the baselines on the pick-and-place level L0_vis_place (results/hard_scenario/place_bench/README.md,
step 2), from the episodes.json files of the final evaluations (no new runs):

  python policy/readout_place.py > results/hard_scenario/place_bench/readout_baselines.txt

Printed, per run (seed 42; a run without a finished evaluation is left out and named): 1. success per cell, per test
set and over all episodes; 2. the share of episodes in which the glass was picked (lifted 10 cm at some point) and the
success among those; 3. the failure labels; 4. next to it, not a test: the same inputs on the pick level L0_vis (the
frozen run of seed 42, same scenes and seeds) against "picked" here, on the same episodes; 5. the best baseline of
each cell.
"""
import json
from collections import Counter
from math import comb
from pathlib import Path

import click
import numpy as np

VARIANTS = ("rgbdt", "rgbd", "rgbt", "rgb")
TEST_SETS = ("j_cells11", "k_cells11_heldout")
SHOWS = ["rgb+depth+thermal", "depth+thermal", "rgb+thermal", "rgb+depth", "thermal", "depth", "rgb",
         "rgb (faint)", "depth (faint)", "thermal (faint)", "all three (faint)"]


def load(run):
    """{(test set, seed): episode record}, or None when an evaluation is missing."""
    eps = {}
    for ts in TEST_SETS:
        f = Path(run) / ts / "episodes.json"
        if not f.exists():
            return None
        eps.update({(ts, e["seed"]): e for e in json.loads(f.read_text())})
    return eps


def ok(e):
    return e["outcome"] == "success"


def sign_p(a, b):
    """Two-sided exact sign test on a against b discordant episodes."""
    n, k = a + b, min(a, b)
    return 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def rate(eps, f, cells=range(11), ts=None):
    r = [f(e) for (t, _), e in eps.items() if e["cell"] in cells and (ts is None or t == ts)]
    return float(np.mean(r)) if r else float("nan")


@click.command()
@click.option("--root", default="results/hard_scenario/place_bench")
@click.option("--pick_root", default="results/hard_scenario/grid_bench")
def main(root, pick_root):
    runs = {v: load(Path(root) / f"L0_vis_place_{v}_700demos_imagenet") for v in VARIANTS}
    missing = [v for v in VARIANTS if runs[v] is None]
    if missing:
        print(f"Not finished yet, left out: {missing}\n")
    runs = {v: r for v, r in runs.items() if r is not None}
    if not runs:
        return
    names = list(runs)
    n = len(next(iter(runs.values()))) // 11
    head = "| cell | sensors that show the glass | " + " | ".join(names) + " |"
    line = "|---|---|" + "---|" * len(names)

    print(f"## 1. Success (the glass placed), {n} episodes per cell, seed 42\n")
    print(head + "\n" + line)
    for c in range(11):
        print(f"| {c} | {SHOWS[c]} | " + " | ".join(f"{rate(runs[v], ok, (c,)):.3f}" for v in names) + " |")
    for label, kw in [("macro average over the cells", {}), ("j_cells11", dict(ts="j_cells11")),
                      ("k_cells11_heldout", dict(ts="k_cells11_heldout"))]:
        vals = [np.mean([rate(runs[v], ok, (c,), **kw) for c in range(11)]) for v in names]
        print(f"| {label} | | " + " | ".join(f"{x:.3f}" for x in vals) + " |")

    print("\n## 2. The pick and the place apart: share of episodes with the glass picked (lifted 10 cm at some "
          "point) / success among those\n")
    print(head + "\n" + line)
    for c in [*range(11), None]:
        cells = range(11) if c is None else (c,)
        row = []
        for v in names:
            picked = {k: e for k, e in runs[v].items() if e["picked"] and e["cell"] in cells}
            row.append(f"{rate(runs[v], lambda e: e['picked'], cells):.3f} / "
                       + (f"{np.mean([ok(e) for e in picked.values()]):.3f}" if picked else "-"))
        print(f"| {'all episodes' if c is None else c} | {'' if c is None else SHOWS[c]} | " + " | ".join(row) + " |")

    print("\n## 3. Failure labels (share of all episodes)\n")
    labels = sorted({e["outcome"] for r in runs.values() for e in r.values()} - {"success"})
    print("| label | " + " | ".join(names) + " |\n|---|" + "---|" * len(names))
    for lab in ["success", *labels]:
        print(f"| {lab} | " + " | ".join(
            f"{Counter(e['outcome'] for e in runs[v].values())[lab] / len(runs[v]):.3f}" for v in names) + " |")

    print("\n## 4. Next to it, not a test: the same inputs on the pick level L0_vis (frozen run of seed 42, same "
          "scenes and seeds) against \"picked\" here, same episodes\n")
    print("| inputs | cells | success on the pick level | picked here | only the pick level / only here | p |")
    print("|---|---|---|---|---|---|")
    for v in names:
        pick = load(Path(pick_root) / f"L0_vis_{v}_700demos_imagenet")
        if pick is None or set(pick) != set(runs[v]):
            print(f"| {v} | | pick-level run not found or not on the same episodes | | | |")
            continue
        for label, cells in [("all", range(11)), ("depth shows the glass at least faintly (0, 1, 3, 5, 8, 10)",
                                                  (0, 1, 3, 5, 8, 10)), ("it does not (2, 4, 6, 7, 9)", (2, 4, 6, 7, 9))]:
            keys = [k for k in pick if pick[k]["cell"] in cells]
            a = sum(ok(pick[k]) and not runs[v][k]["picked"] for k in keys)
            b = sum(runs[v][k]["picked"] and not ok(pick[k]) for k in keys)
            print(f"| {v} | {label} | {np.mean([ok(pick[k]) for k in keys]):.3f} | "
                  f"{np.mean([runs[v][k]['picked'] for k in keys]):.3f} | {a} / {b} | {sign_p(a, b):.2g} |")

    print("\n## 5. The best baseline of each cell (highest success; one seed, a first reading)\n")
    print("| cell | sensors that show the glass | best | success |\n|---|---|---|---|")
    best = []
    for c in range(11):
        r = {v: rate(runs[v], ok, (c,)) for v in names}
        top = max(r.values())
        best.append(top)
        print(f"| {c} | {SHOWS[c]} | {', '.join(v for v in names if r[v] == top)} | {top:.3f} |")
    print(f"| mean of the best per cell | | | {np.mean(best):.3f} |")


if __name__ == "__main__":
    main()
