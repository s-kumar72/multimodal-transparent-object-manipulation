"""
The cue alone over several seeds of its fit (results/hard_scenario/cue_gate/README.md, round 6a): is the difference
between the cue of the method (maps per sensor with the verified gate) and the cue of one network on all sensors a
property of the two, or of one fit of each? From the files of `policy/check_cue_env.py --save` (the cue the evaluation
env computes at reset; no policy involved):

  python policy/readout_cue_fitseeds.py > results/hard_scenario/cue_gate/perception/round6_cue_fitseeds.txt

Fit 0 is the fit the policies were trained with (data/hcue/<folder>[_early]); fits 1 to 4 are
data/hcue/<folder>[_early]_fit<k> (policy/hindsight_cue_fitseed.py). Per training set and group of cells: the share
of episodes with the cue within 1.5 cm of the glass per fit, the method against the one network of the same fit
number on the same episodes (two-sided sign test), mean and standard deviation over the fits, the pooled test and the
verdict of the README. A fit without its files is left out and named.
"""
from pathlib import Path

import click
import numpy as np

import readout_leaveout_cue as lc

FITS = (0, 1, 2, 3, 4)
SPLITS = {"all 700 demos": ("L0_vis_700", ()), **lc.SPLITS}
NEW = ("l_cells15_new", "m_cells15_new_heldout")


def folder(hcue, name, early, k):
    return hcue / (name + ("_early" if early else "") + ("" if k == 0 else f"_fit{k}"))


def verdict(per, a, b, p, complete):
    if not complete:
        return "not final"
    if p >= 0.05:
        return "does not hold"
    if a > b and all(x > y for x, y in per):
        return "holds over the fits (the method above)"
    if b > a and all(y > x for x, y in per):
        return "holds over the fits (the one network above)"
    return "mixed"


def block(title, fits, cells):
    print(f"**{title}**\n")
    print("| fit | method | one network | only the method right / only the one network | p |\n|---|---|---|---|---|")
    per, A, B = [], 0, 0
    for k, (m, e) in fits.items():
        sa, sb, a, b, p = lc.versus(m, e, cells)
        per.append((sa, sb)); A += a; B += b
        print(f"| {k}{' (the fit of the policies)' if k == 0 else ''} | {sa:.3f} | {sb:.3f} | {a} / {b} | {p:.2g} |")
    sd = lambda v: f"{np.std(v, ddof=1):.3f}" if len(v) > 1 else "-"
    p = lc.sign_p(A, B)
    print(f"| mean (SD) and pooled | {np.mean([x for x, _ in per]):.3f} ({sd([x for x, _ in per])}) | "
          f"{np.mean([y for _, y in per]):.3f} ({sd([y for _, y in per])}) | {A} / {B} | {p:.2g} |")
    v = verdict(per, A, B, p, len(fits) == len(FITS))
    print(f"\nVerdict: the difference {v}.\n" if v != "not final" else "\nNot final: fits are missing.\n")
    return v, f"{np.mean([x for x, _ in per]):.3f} / {np.mean([y for _, y in per]):.3f}"


@click.command()
@click.option("--hcue", default="data/hcue")
def main(hcue):
    hcue, summary = Path(hcue), []
    for split, (name, out) in SPLITS.items():
        fits = {k: (lc.load(folder(hcue, name, False, k)), lc.load(folder(hcue, name, True, k))) for k in FITS}
        missing = [k for k, v in fits.items() if v[0] is None or v[1] is None]
        fits = {k: v for k, v in fits.items() if k not in missing}
        print(f"## Cues fitted on {split}" + (f", without cells {list(out)}" if out else "") + "\n")
        if missing:
            print(f"Fits without their files, left out: {missing}\n")
        if not fits:
            continue
        kept = tuple(c for c in range(11) if c not in out)
        groups = [("all eleven cells", kept)] if not out else [(f"the cells left out {list(out)}", out),
                                                                (f"the cells of the training set {list(kept)}", kept)]
        for title, cells in groups:
            v, nums = block(title, fits, cells)
            summary.append((split, title, nums, v))
        if not out:      # the cues of all 700 demos on the 15 new combinations
            lc_sets, lc.SETS = lc.SETS, NEW
            new = {k: (lc.load(folder(hcue, name, False, k)), lc.load(folder(hcue, name, True, k))) for k in fits}
            lc.SETS = lc_sets
            new = {k: v for k, v in new.items() if v[0] is not None and v[1] is not None}
            if new:
                v, nums = block("the 15 new combinations", new, tuple(range(15)))
                summary.append((split, "the 15 new combinations", nums, v if len(new) == len(FITS) else "not final"))
    print("## The verdicts side by side\n")
    print("| cues fitted on | cells | method / one network, mean over the fits | the difference |\n|---|---|---|---|")
    for row in summary:
        print("| " + " | ".join(row) + " |")


if __name__ == "__main__":
    main()
