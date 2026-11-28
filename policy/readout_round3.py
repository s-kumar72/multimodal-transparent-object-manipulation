"""
Read-out of round 3 of hindsight cue gating (results/hard_scenario/cue_gate/README.md, "Round 3": visibility
combinations that are not in the training set), from files that exist (no new runs):

  python policy/readout_round3.py cue-new            # 3a step 1: the cues on the test sets with the 15 new cells
  python policy/readout_round3.py policies-new       # 3a step 2: the policies on them
  python policy/readout_round3.py cue-nosingle       # 3b step 1: the cues fitted without cells 4 to 9
  python policy/readout_round3.py policies-nosingle  # 3b step 2: the policies trained without cells 4 to 9

The cue files are those of `policy/check_cue_env.py --save data/hcue/<folder>/env_<test set>.npz`; the policies are
read from <run>/<test set>/episodes.json. What has not been run yet is left out. "Same episodes" = the same test set
and episode seed; sign test on the episodes in which exactly one of the two is right (cue within 1.5 cm) or succeeds.
"""
import json
import sys
from math import comb
from pathlib import Path

import click
import numpy as np

STATE = ("shows", "blind", "faint")
NEW_SETS = ("l_cells15_new", "m_cells15_new_heldout")
OLD_SETS = ("j_cells11", "k_cells11_heldout")
LEFT_OUT, KEPT = (4, 5, 6, 7, 8, 9), (0, 1, 2, 3, 10)
RIGHT = 1.5      # cm
P = "L0_vis_rgbdt_700demos_"
N = "L0_vis_rgbdt_400demos_nosingle_"


def cells_of(test_set):
    m = json.loads((Path(__file__).parent.parent / "sim/configs/manifest.json").read_text())
    return [" / ".join(STATE[f] for f in c[1:]) for c in m["test_sets"][test_set]["overrides"]["cells"]]


def sign_p(a, b):
    n, k = a + b, min(a, b)
    return 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def mark(p, d):
    return (" (above)" if d > 0 else " (below)") if p < 0.05 else ""


def load_cue(folder, sets):
    """{(test set, seed): (cell, cue right)} of a cue folder, or None when a file is missing."""
    out = {}
    for t in sets:
        f = Path(folder) / f"env_{t}.npz"
        if not f.exists():
            return None
        d = np.load(f)
        err = 100 * np.linalg.norm(d["cue"][:, :2] - d["target"][:, :2], axis=1)
        out.update({(t, int(s)): (int(c), bool(e < RIGHT)) for s, c, e in zip(d["seed"], d["cell"], err)})
    return out


def load_run(folder, sets):
    """{(test set, seed): (cell, success)} of a run folder, or None while an evaluation is missing."""
    out = {}
    for t in sets:
        f = Path(folder) / t / "episodes.json"
        if not f.exists() or not (Path(folder) / t / "eval_log.json").exists():
            return None
        out.update({(t, e["seed"]): (e["cell"], float(e["reward"]) > 0.5) for e in json.loads(f.read_text())})
    return out


def rate(A, cells):
    v = [ok for c, ok in A.values() if c in cells]
    return float(np.mean(v)) if v else float("nan")


def versus(A, B, cells):
    keys = [k for k in A if A[k][0] in cells]
    assert all(k in B and B[k][0] == A[k][0] for k in keys), "not the same episodes"
    a = sum(A[k][1] and not B[k][1] for k in keys)
    b = sum(B[k][1] and not A[k][1] for k in keys)
    return rate(A, cells), rate(B, cells), a, b, sign_p(a, b)


def table(what, runs, labels, groups, ref=None):
    """Per cell and per group of cells: the rate of every entry; marks against the entry `ref` on the same episodes."""
    names = list(runs)
    print(f"| cell | rgb / depth / thermal | " + " | ".join(names) + " |")
    print("|---|---|" + "---|" * len(names))
    rows = [(str(c), labels[c], (c,)) for c in range(len(labels))] + [(g, "", cs) for g, cs in groups.items()]
    for name, lab, cs in rows:
        cols = []
        for n in names:
            if ref is None or n == ref:
                cols.append(f"{rate(runs[n], cs):.3f}")
            else:
                sa, sb, a, b, p = versus(runs[n], runs[ref], cs)
                cols.append(f"{sa:.3f}{mark(p, sa - sb)}")
        print(f"| {name} | {lab} | " + " | ".join(cols) + " |")
    if ref:
        print(f"\nMarks: against {ref} on the same episodes, p < 0.05 ({what}).")


def pairs(what, runs, todo, groups):
    print()
    for x, y in todo:
        if x not in runs or y not in runs:
            print(f"{x} against {y}: not run yet")
            continue
        for g, cs in groups.items():
            sa, sb, a, b, p = versus(runs[x], runs[y], cs)
            print(f"{x} against {y}, {g}: {sa:.3f} against {sb:.3f}, only {x} {what} {a}, only {y} {b}, "
                  f"p = {p:.2g}{mark(p, sa - sb)}")
        diff = []
        n_cells = max(c for c, _ in runs[x].values()) + 1
        for c in range(n_cells):
            sa, sb, a, b, p = versus(runs[x], runs[y], (c,))
            if p < 0.05:
                diff.append(f"cell {c} {sa:.3f} against {sb:.3f} (p = {p:.2g})")
        print(f"  cells that differ with p < 0.05: {'; '.join(diff) or 'none'}")


def keep(d):
    missing = [n for n, r in d.items() if r is None]
    if missing:
        print(f"Not there yet, left out: {', '.join(missing)}\n")
    return {n: r for n, r in d.items() if r is not None}


@click.command()
@click.argument("part", type=click.Choice(["cue-new", "policies-new", "cue-nosingle", "policies-nosingle"]))
@click.option("--root", default="results/hard_scenario/cue_gate")
@click.option("--base_root", default="results/hard_scenario/grid_bench")
@click.option("--hcue", default="data/hcue")
def main(part, root, base_root, hcue):
    root, base_root, hcue = Path(root), Path(base_root), Path(hcue)
    if part == "cue-new":
        labels = cells_of(NEW_SETS[0])
        groups = {"all 15 cells": range(15), "a sensor shows the glass fully (0-11)": range(12),
                  "none does (12-14)": (12, 13, 14)}
        cues = keep({"method": load_cue(hcue / "L0_vis_700", NEW_SETS), "one network": load_cue(hcue / "L0_vis_700_early", NEW_SETS),
                     "no gate": load_cue(hcue / "L0_vis_700_nogate", NEW_SETS),
                     "gate on map numbers": load_cue(hcue / "L0_vis_700_v0", NEW_SETS)})
        print(f"## 3a, step 1: share of episodes with the cue within {RIGHT} cm of the glass, cues fitted on the 700 "
              f"demos, test sets {' and '.join(NEW_SETS)} (80 episodes per cell)\n")
        table("cue right", cues, labels, groups, ref="method")
        pairs("is right in", cues, [("method", n) for n in cues if n != "method"], groups)
        old = keep({"method": load_cue(hcue / "L0_vis_700", OLD_SETS), "one network": load_cue(hcue / "L0_vis_700_early", OLD_SETS),
                    "no gate": load_cue(hcue / "L0_vis_700_nogate", OLD_SETS),
                    "gate on map numbers": load_cue(hcue / "L0_vis_700_v0", OLD_SETS)})
        if old:
            print("\nFor orientation, the same cues on the eleven training cells (j_cells11 and k_cells11_heldout): "
                  + ", ".join(f"{n} {rate(r, range(11)):.3f}" for n, r in old.items()))
    elif part == "policies-new":
        labels = cells_of(NEW_SETS[0])
        groups = {"all 15 cells": range(15), "a sensor shows the glass fully (0-11)": range(12),
                  "none does (12-14)": (12, 13, 14)}
        folders = {"cue": root / (P + "hcg_cue"), "full": root / (P + "hcg_full"), "blind": root / (P + "hcg_blind"),
                   "early": root / (P + "hcg_cue_early"), "nogate": root / (P + "hcg_cue_nogate"),
                   "mapgate": root / (P + "hcg_cue_mapgate"), "dp_aux": root / (P + "dp_aux"),
                   "rgbdt ImageNet": base_root / (P + "imagenet")}
        runs = keep({n: load_run(f, NEW_SETS) for n, f in folders.items()})
        print(f"## 3a, step 2: success of the policies trained on the 700 demos, test sets {' and '.join(NEW_SETS)} "
              "(80 episodes per cell; seed 42)\n")
        table("success", runs, labels, groups, ref="cue")
        pairs("succeeds in", runs, [("cue", "early"), ("cue", "nogate"), ("cue", "mapgate"), ("full", "cue"),
                                    *[(n, "rgbdt ImageNet") for n in runs if n != "rgbdt ImageNet"]], groups)
        print("\nFor orientation, not a test: success on the 15 new cells next to the 11 training cells "
              "(j_cells11 and k_cells11_heldout)\n")
        print("| policy | 15 new cells | 11 training cells |")
        print("|---|---|---|")
        for n, r in runs.items():
            o = load_run(folders[n], OLD_SETS)
            print(f"| {n} | {rate(r, range(15)):.3f} | {rate(o, range(11)) if o else float('nan'):.3f} |")
    else:
        labels = cells_of(OLD_SETS[0])
        groups = {"cells left out (4-9)": LEFT_OUT, "cells of the training set (0-3, 10)": KEPT}
        if part == "cue-nosingle":
            cues = keep({"method": load_cue(hcue / "L0_vis_400_nosingle", OLD_SETS),
                         "one network": load_cue(hcue / "L0_vis_400_nosingle_early", OLD_SETS),
                         "no gate": load_cue(hcue / "L0_vis_400_nosingle_nogate", OLD_SETS)})
            print(f"## 3b, step 1: share of episodes with the cue within {RIGHT} cm of the glass, cues fitted on the 400 "
                  f"demos without cells 4 to 9, test sets {' and '.join(OLD_SETS)} (120 episodes per cell)\n")
            table("cue right", cues, labels, groups, ref="method")
            pairs("is right in", cues, [("method", n) for n in cues if n != "method"], groups)
            full = keep({"method": load_cue(hcue / "L0_vis_700", OLD_SETS), "one network": load_cue(hcue / "L0_vis_700_early", OLD_SETS),
                         "no gate": load_cue(hcue / "L0_vis_700_nogate", OLD_SETS)})
            print("\nThe same cues fitted on all 700 demos, same episodes (what leaving the cells out costs):\n")
            for n in cues:
                if n in full:
                    for g, cs in groups.items():
                        sa, sb, a, b, p = versus(cues[n], full[n], cs)
                        print(f"{n}, {g}: {sa:.3f} without the cells against {sb:.3f} with them, p = {p:.2g}")
        else:
            runs = keep({"cue": load_run(root / (N + "hcg_cue"), OLD_SETS), "full": load_run(root / (N + "hcg_full"), OLD_SETS),
                         "early": load_run(root / (N + "hcg_cue_early"), OLD_SETS),
                         "rgbdt ImageNet": load_run(root / (N + "imagenet"), OLD_SETS)})
            print(f"## 3b, step 2: success of the policies trained on the 400 demos without cells 4 to 9, test sets "
                  f"{' and '.join(OLD_SETS)} (120 episodes per cell; seed 42)\n")
            table("success", runs, labels, groups, ref="cue")
            pairs("succeeds in", runs, [("cue", "early"), ("full", "cue"),
                                        *[(n, "rgbdt ImageNet") for n in runs if n != "rgbdt ImageNet"]], groups)
            full = keep({"cue": load_run(root / (P + "hcg_cue"), OLD_SETS), "full": load_run(root / (P + "hcg_full"), OLD_SETS),
                         "early": load_run(root / (P + "hcg_cue_early"), OLD_SETS),
                         "rgbdt ImageNet": load_run(base_root / (P + "imagenet"), OLD_SETS)})
            print("\nAgainst the counterparts trained on all 700 demos, same episodes (what leaving the cells out costs):\n")
            for n in runs:
                if n in full:
                    for g, cs in groups.items():
                        sa, sb, a, b, p = versus(runs[n], full[n], cs)
                        print(f"{n}, {g}: {sa:.3f} without the cells against {sb:.3f} with them, only without {a}, "
                              f"only with {b}, p = {p:.2g}")


if __name__ == "__main__":
    sys.exit(main())
