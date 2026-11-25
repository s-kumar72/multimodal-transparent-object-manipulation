"""
The cue alone on the training sets with cells left out (results/hard_scenario/cue_gate/README.md, round 5, step 1):
the cue of the method (maps per sensor with the verified gate) against the cue of one network on all sensors, both
fitted on the same demos, from the files of `policy/check_cue_env.py --save data/hcue/<folder>/env_<test set>.npz`
(the cue the evaluation env computes at reset; no policy involved):

  python policy/readout_leaveout_cue.py > results/hard_scenario/cue_gate/perception/round5_cue_leaveout.txt

Per training set: the share of episodes with the cue within 1.5 cm of the glass per cell, in the cells left out and in
the cells of the training set; the method against the one network on the same episodes (two-sided sign test on the
episodes in which exactly one of the two cues is right); each against the same cue fitted on all 700 demos.
"""
from math import comb
from pathlib import Path

import click
import numpy as np

SETS = ("j_cells11", "k_cells11_heldout")
SPLITS = {"nosingle": ("L0_vis_400_nosingle", (4, 5, 6, 7, 8, 9)), "nopair": ("L0_vis_490_nopair", (1, 2, 3)),
          "nofaint": ("L0_vis_500_nofaint", (7, 8, 9, 10))}
ALL = "L0_vis_700"
STATES = ["shows / shows / shows", "blind / shows / shows", "shows / blind / shows", "shows / shows / blind",
          "blind / blind / shows", "blind / shows / blind", "shows / blind / blind", "faint / blind / blind",
          "blind / faint / blind", "blind / blind / faint", "faint / faint / faint"]
RIGHT = 1.5      # cm


def sign_p(a, b):
    n, k = a + b, min(a, b)
    return 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def load(folder):
    """(cell, cue right) over the two test sets, or None."""
    cell, right = [], []
    for t in SETS:
        f = Path(folder) / f"env_{t}.npz"
        if not f.exists():
            return None
        d = np.load(f)
        cell.append(d["cell"])
        right.append(100 * np.linalg.norm(d["cue"][:, :2] - d["target"][:, :2], axis=1) < RIGHT)
    return np.concatenate(cell), np.concatenate(right)


def versus(x, y, cells):
    m = np.isin(x[0], cells)
    a, b = int((x[1] & ~y[1])[m].sum()), int((y[1] & ~x[1])[m].sum())
    return x[1][m].mean(), y[1][m].mean(), a, b, sign_p(a, b)


@click.command()
@click.option("--hcue", default="data/hcue")
def main(hcue):
    hcue = Path(hcue)
    full = {"method": load(hcue / ALL), "one network": load(hcue / f"{ALL}_early")}
    for split, (folder, out) in SPLITS.items():
        kept = tuple(c for c in range(11) if c not in out)
        cue = {"method": load(hcue / folder), "one network": load(hcue / f"{folder}_early")}
        print(f"## {split}: cues fitted without cells {list(out)}; share of episodes with the cue within {RIGHT} cm of "
              "the glass (120 episodes per cell)\n")
        if any(v is None for v in cue.values()):
            print("not checked in the env yet\n")
            continue
        m, e = cue["method"], cue["one network"]
        assert (m[0] == e[0]).all()
        print("| cell | rgb / depth / thermal | in the training set | method | one network | only the method right / "
              "only the one network | p |\n|---|---|---|---|---|---|---|")
        groups = [(str(c), STATES[c], "no" if c in out else "yes", (c,)) for c in range(11)]
        groups += [(f"cells left out {list(out)}", "", "no", out), (f"cells of the training set {list(kept)}", "", "yes", kept)]
        for name, states, inside, cells in groups:
            sa, sb, a, b, p = versus(m, e, cells)
            print(f"| {name} | {states} | {inside} | {sa:.3f} | {sb:.3f} | {a} / {b} | {p:.2g}"
                  + (" +" if p < 0.05 and a > b else " -" if p < 0.05 else "") + " |")
        print("\nAgainst the same cue fitted on all 700 demos, in the cells left out (same episodes):\n")
        print("| cue | fitted without the cells | fitted on all cells | only without / only with all | p |\n|---|---|---|---|---|")
        for n in cue:
            if full[n] is None:
                continue
            sa, sb, a, b, p = versus(cue[n], full[n], out)
            print(f"| {n} | {sa:.3f} | {sb:.3f} | {a} / {b} | {p:.2g} |")
        print()


if __name__ == "__main__":
    main()
