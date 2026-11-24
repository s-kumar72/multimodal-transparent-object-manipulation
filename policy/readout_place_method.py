"""
Read-out of hindsight cue gating on the pick-and-place level L0_vis_place (results/hard_scenario/place_bench/README.md,
step 3), from the episodes.json files of the final evaluations (no new runs):

  python policy/readout_place_method.py > results/hard_scenario/place_bench/readout_method.txt

1. success against the rgbdt baseline and, per cell, against the cell's best baseline; 2. the pick and the place apart;
3. which part carries it (paired comparisons of the arms); 4. the two cues inside the episodes; 5. next to them, not a
test: the cells by whether the colour camera shows the glass. Seed 42; a run without a finished evaluation is left out
and named.
"""
import json
from math import comb
from pathlib import Path

import click
import numpy as np

METHOD = ("hcg2_full", "hcg2_cue", "hcg_cue", "hcg2_blind")
BASELINES = ("rgbdt", "rgbd", "rgbt", "rgb")
PAIRS = (("hcg2_full", "hcg2_cue", "gating the features"), ("hcg2_cue", "hcg_cue", "the place cue"),
         ("hcg2_cue", "hcg2_blind", "the images"))
TEST_SETS = ("j_cells11", "k_cells11_heldout")
SHOWS = ["rgb+depth+thermal", "depth+thermal", "rgb+thermal", "rgb+depth", "thermal", "depth", "rgb",
         "rgb (faint)", "depth (faint)", "thermal (faint)", "all three (faint)"]
RGB_SHOWS = (0, 2, 3, 6, 7, 10)      # cells in which the colour camera shows the glass (fully or faintly)
RIGHT = 1.5                          # cm
CUE_DIM = 9


def load(run):
    eps = {}
    for ts in TEST_SETS:
        f = Path(run) / ts / "episodes.json"
        if not f.exists():
            return None
        eps.update({(ts, e["seed"]): e for e in json.loads(f.read_text())})
    return eps


def ok(e):
    return e["outcome"] == "success"


def picked(e):
    return bool(e["picked"])


def sign_p(a, b):
    n, k = a + b, min(a, b)
    return 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def paired(x, y, keys, f=ok):
    a = sum(f(x[k]) and not f(y[k]) for k in keys)
    b = sum(f(y[k]) and not f(x[k]) for k in keys)
    return a, b, sign_p(a, b)


def rate(eps, f=ok, cells=range(11)):
    r = [f(e) for e in eps.values() if e["cell"] in cells]
    return float(np.mean(r)) if r else float("nan")


def cue_errors(e):
    """(grasp cue to the glass, place cue to the coaster's centre), cm in the table plane; nan without that cue."""
    c = e.get("hcue")
    grasp = 100 * np.linalg.norm(np.subtract(c[:2], e["target_xyz"][:2])) if c else np.nan
    place = 100 * np.linalg.norm(np.subtract(c[CUE_DIM:CUE_DIM + 2], e["coaster_xy"])) if c and len(c) > CUE_DIM else np.nan
    return grasp, place


@click.command()
@click.option("--root", default="results/hard_scenario/place_bench")
def main(root):
    root = Path(root)
    base = {v: load(root / f"L0_vis_place_{v}_700demos_imagenet") for v in BASELINES}
    assert all(b is not None for b in base.values()), "the baselines of step 2 are needed"
    runs = {t: load(root / f"L0_vis_place_rgbdt_700demos_{t}") for t in METHOD}
    missing = [t for t in METHOD if runs[t] is None]
    if missing:
        print(f"Not finished yet, left out: {missing}\n")
    runs = {t: r for t, r in runs.items() if r is not None}
    if not runs:
        return
    names, keys = list(runs), sorted(base["rgbdt"])
    for r in runs.values():
        assert set(r) == set(keys)

    print("## 1. Success (the glass placed), 120 episodes per cell, seed 42\n")
    print("Over all episodes, against the rgbdt baseline on the same episodes:\n")
    print("| run | success | rgbdt baseline | only the run / only the baseline | p |\n|---|---|---|---|---|")
    for t in names:
        a, b, p = paired(runs[t], base["rgbdt"], keys)
        print(f"| {t} | {rate(runs[t]):.3f} | {rate(base['rgbdt']):.3f} | {a} / {b} | {p:.2g} |")
    print("\nPer cell, against the cell's best baseline (+ / - : above / below it at p < 0.05, sign test):\n")
    print("| cell | sensors that show the glass | best baseline | " + " | ".join(names) + " |")
    print("|---|---|---|" + "---|" * len(names))
    for c in range(11):
        ck = [k for k in keys if base["rgbdt"][k]["cell"] == c]
        best = max(BASELINES, key=lambda v: rate(base[v], cells=(c,)))
        row = []
        for t in names:
            a, b, p = paired(runs[t], base[best], ck)
            row.append(f"{rate(runs[t], cells=(c,)):.3f}" + (" +" if p < 0.05 and a > b else " -" if p < 0.05 else ""))
        print(f"| {c} | {SHOWS[c]} | {best} {rate(base[best], cells=(c,)):.3f} | " + " | ".join(row) + " |")
    best_mean = np.mean([max(rate(base[v], cells=(c,)) for v in BASELINES) for c in range(11)])
    print(f"| all | | {best_mean:.3f} (mean of the best per cell) | " + " | ".join(f"{rate(runs[t]):.3f}" for t in names) + " |")

    print("\n## 2. The pick and the place apart\n")
    print("| run | glass picked, share of all episodes | placed, among the picked | never picked / picked and not placed |")
    print("|---|---|---|---|")
    for t, eps in [("rgbdt baseline", base["rgbdt"]), *runs.items()]:
        pk = [e for e in eps.values() if picked(e)]
        print(f"| {t} | {rate(eps, picked):.3f} | {np.mean([ok(e) for e in pk]):.3f} | "
              f"{1 - rate(eps, picked):.3f} / {np.mean([picked(e) and not ok(e) for e in eps.values()]):.3f} |")

    print("\n## 3. Which part carries it: the arms on the same episodes\n")
    print("| what differs | A | B | success A | success B | only A / only B | p | picked A | picked B | only A / only B | p |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for x, y, what in PAIRS:
        if x in runs and y in runs:
            a, b, p = paired(runs[x], runs[y], keys)
            a2, b2, p2 = paired(runs[x], runs[y], keys, picked)
            print(f"| {what} | {x} | {y} | {rate(runs[x]):.3f} | {rate(runs[y]):.3f} | {a} / {b} | {p:.2g} | "
                  f"{rate(runs[x], picked):.3f} | {rate(runs[y], picked):.3f} | {a2} / {b2} | {p2:.2g} |")

    print(f"\n## 4. The cues inside the episodes (right: within {RIGHT:g} cm in the table plane)\n")
    print("| run | grasp cue right | picked with it right / wrong | place cue right | placed among the picked with it "
          "right / wrong | success with both right |\n|---|---|---|---|---|---|")
    for t, eps in runs.items():
        err = np.array([cue_errors(e) for e in eps.values()])
        pk, s = np.array([picked(e) for e in eps.values()]), np.array([ok(e) for e in eps.values()])
        g = err[:, 0] < RIGHT
        cells = f"| {t} | {g.mean():.3f} | {pk[g].mean():.3f} / {pk[~g].mean():.3f} | "
        if np.isnan(err[:, 1]).all():
            cells += "no place cue | - | - |"
        else:
            pl = err[:, 1] < RIGHT
            cells += (f"{pl.mean():.3f} | {s[pk & pl].mean():.3f} / "
                      + (f"{s[pk & ~pl].mean():.3f}" if (pk & ~pl).any() else "-") + f" | {s[g & pl].mean():.3f} |")
        print(cells)

    print("\n## 5. Next to them, not a test: the cells by whether the colour camera shows the glass\n")
    other = tuple(c for c in range(11) if c not in RGB_SHOWS)
    print(f"| run | it shows the glass {RGB_SHOWS} | it shows the coaster and not the glass {other} |\n|---|---|---|")
    for t, eps in [("rgbdt baseline", base["rgbdt"]), *runs.items()]:
        print(f"| {t} | {rate(eps, cells=RGB_SHOWS):.3f} | {rate(eps, cells=other):.3f} |")


if __name__ == "__main__":
    main()
