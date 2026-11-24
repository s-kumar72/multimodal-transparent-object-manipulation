"""
Read-out of round 4 of hindsight cue gating (results/hard_scenario/cue_gate/README.md): the arm `cue` against the
one-network control over seeds 42, 43 and 44, from the episodes.json files of the final evaluations (no new runs):

  python policy/readout_round4.py > results/hard_scenario/cue_gate/readout_round4.txt

Each seed of `cue` is paired with the same seed of the control on the same episodes. Settings: 1. the training cells
(j_cells11, k_cells11_heldout); 2. the 15 new combinations (l_cells15_new, m_cells15_new_heldout); 3. the runs trained
without cells 4 to 9, in those cells and in the cells of their training set. Per setting: success per seed, mean and
SD, the sign test over the pooled pairs, and the verdict of the README (holds over the seeds / does not hold / mixed).
A seed whose two runs are not both finished is left out and named.
"""
import json
from math import comb
from pathlib import Path

import click
import numpy as np

SEEDS = (42, 43, 44)
LEFT_OUT = (4, 5, 6, 7, 8, 9)
SETTINGS = [   # title, stem of the two arms, test sets, groups of cells (None: all)
    ("1. Training cells", "L0_vis_rgbdt_700demos", ("j_cells11", "k_cells11_heldout"), [("all", None)]),
    ("2. New combinations", "L0_vis_rgbdt_700demos", ("l_cells15_new", "m_cells15_new_heldout"), [("all", None)]),
    ("3. Trained without cells 4 to 9", "L0_vis_rgbdt_400demos_nosingle", ("j_cells11", "k_cells11_heldout"),
     [("the cells left out (4 to 9)", LEFT_OUT), ("the cells of the training set (0, 1, 2, 3, 10)", (0, 1, 2, 3, 10))]),
]


def load(run, sets):
    eps = {}
    for ts in sets:
        f = Path(run) / ts / "episodes.json"
        if not f.exists():
            return None
        eps.update({(ts, e["seed"]): e for e in json.loads(f.read_text())})
    return eps


def ok(e):
    return e["outcome"] == "success"


def sign_p(a, b):
    n, k = a + b, min(a, b)
    return 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


@click.command()
@click.option("--root", default="results/hard_scenario/cue_gate")
def main(root):
    root = Path(root)
    for title, stem, sets, groups in SETTINGS:
        pairs, missing = {}, []
        for s in SEEDS:
            suffix = "" if s == 42 else f"_seed{s}"
            cue, early = load(root / f"{stem}_hcg_cue{suffix}", sets), load(root / f"{stem}_hcg_cue_early{suffix}", sets)
            if cue is None or early is None:
                missing.append(s)
            else:
                assert set(cue) == set(early)
                pairs[s] = (cue, early)
        print(f"## {title} ({', '.join(sets)})\n")
        if missing:
            print(f"Seeds not finished, left out: {missing}\n")
        if not pairs:
            continue
        for label, cells in groups:
            print(f"**{label}**\n" if label != "all" else "", end="\n" if label != "all" else "")
            print("| seed | episodes | `cue` | one network | only `cue` / only the one network | p |\n|---|---|---|---|---|---|")
            A = B = 0
            rc, re_ = [], []
            for s, (cue, early) in pairs.items():
                keys = [k for k in cue if cells is None or cue[k]["cell"] in cells]
                a = sum(ok(cue[k]) and not ok(early[k]) for k in keys)
                b = sum(ok(early[k]) and not ok(cue[k]) for k in keys)
                rc.append(np.mean([ok(cue[k]) for k in keys])); re_.append(np.mean([ok(early[k]) for k in keys]))
                A, B = A + a, B + b
                print(f"| {s} | {len(keys)} | {rc[-1]:.3f} | {re_[-1]:.3f} | {a} / {b} | {sign_p(a, b):.2g} |")
            sd = lambda v: float(np.std(v, ddof=1)) if len(v) > 1 else float("nan")
            p = sign_p(A, B)
            print(f"| mean (SD) and pooled | | {np.mean(rc):.3f} ({sd(rc):.3f}) | {np.mean(re_):.3f} ({sd(re_):.3f}) | "
                  f"{A} / {B} | {p:.2g} |")
            if len(pairs) == len(SEEDS):
                every = all(c > e for c, e in zip(rc, re_))
                verdict = "does not hold" if p >= 0.05 else "holds over the seeds" if every and A > B else "mixed"
                print(f"\nVerdict: the difference {verdict}.\n")
            else:
                print()
        if title.startswith("1.") or title.startswith("2."):
            cells = sorted({e["cell"] for cue, _ in pairs.values() for e in cue.values()})
            print("Per cell, pooled over the seeds (+ / - : `cue` above / below the one network at p < 0.05):\n")
            print("| cell | `cue` | one network | only `cue` / only the one network | p |\n|---|---|---|---|---|")
            for c in cells:
                a = b = 0
                sc, se = [], []
                for cue, early in pairs.values():
                    keys = [k for k in cue if cue[k]["cell"] == c]
                    a += sum(ok(cue[k]) and not ok(early[k]) for k in keys)
                    b += sum(ok(early[k]) and not ok(cue[k]) for k in keys)
                    sc += [ok(cue[k]) for k in keys]; se += [ok(early[k]) for k in keys]
                p = sign_p(a, b)
                print(f"| {c} | {np.mean(sc):.3f} | {np.mean(se):.3f} | {a} / {b} | {p:.2g}"
                      + (" +" if p < 0.05 and a > b else " -" if p < 0.05 else "") + " |")
            print()


if __name__ == "__main__":
    main()
