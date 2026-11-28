"""
Read-out of the runs trained without some visibility cells (results/hard_scenario/cue_gate/README.md, round 3b and
round 5): the arm `cue` (maps per sensor with the gate) against the one-network control, from the episodes.json files
of the final evaluations (no new runs):

  python policy/readout_leaveout.py > results/hard_scenario/cue_gate/readout_leaveout.txt

Per level (L0_vis; L0_vis_ell, the evaluations in contact_check) and per training set: 1. the cells left out and the
cells of the training set, `cue` against the one network per seed and pooled (same seed, same episodes; two-sided
sign test), with the verdict of round 4 for the cells left out; 2. the cells left out one by one; 3. what leaving the
cells out costs: each arm against the same arm trained on all 700 demos; 4. each arm of seed 42 against the plain
rgbdt baseline trained on the same demos. At the end the verdicts side by side.
A seed whose two runs are not both finished on a level is left out there and named.
"""
from pathlib import Path

import click
import numpy as np

from readout_round4 import load, ok, sign_p

SETS = ("j_cells11", "k_cells11_heldout")
SPLITS = {   # training set: (stem of its runs, cells left out)
    "nosingle": ("L0_vis_rgbdt_400demos_nosingle", (4, 5, 6, 7, 8, 9)),
    "nopair": ("L0_vis_rgbdt_490demos_nopair", (1, 2, 3)),
    "nofaint": ("L0_vis_rgbdt_500demos_nofaint", (7, 8, 9, 10)),
}
ALL = "L0_vis_rgbdt_700demos"                               # the same arms trained on all eleven cells
LEVELS = {"L0_vis": "cue_gate", "L0_vis_ell": "contact_check"}
ARMS = {"`cue`": "hcg_cue", "one network": "hcg_cue_early"}
SHOWS = ["rgb+depth+thermal", "depth+thermal", "rgb+thermal", "rgb+depth", "thermal", "depth", "rgb", "rgb (faint)",
         "depth (faint)", "thermal (faint)", "all faint"]


def run(stem, arm, seed):
    return f"{stem}_{arm}" + ("" if seed == 42 else f"_seed{seed}")


def rate(runs, cells):
    return float(np.mean([ok(e) for r in runs for e in r.values() if e["cell"] in cells]))


def pooled(A, B, cells):
    a = b = 0
    for x, y in zip(A, B):
        keys = [k for k in x if x[k]["cell"] in cells]
        a += sum(ok(x[k]) and not ok(y[k]) for k in keys)
        b += sum(ok(y[k]) and not ok(x[k]) for k in keys)
    return rate(A, cells), rate(B, cells), a, b, sign_p(a, b)


def sd(v):
    return f"{np.std(v, ddof=1):.3f}" if len(v) > 1 else "-"


def verdict(per_seed, A, B, p, complete):
    """Round 4's rule: holds when the pooled test is significant and `cue` is above in every seed."""
    if not complete:
        return "not final"
    if p >= 0.05:
        return "does not hold"
    return "holds over the seeds" if A > B and all(c > e for c, e in per_seed) else "mixed"


@click.command()
@click.option("--root", default="results/hard_scenario")
@click.option("--seeds", default="42,43,44", help="for testing the script on the seeds that exist")
def main(root, seeds):
    root, want = Path(root), [int(s) for s in seeds.split(",")]
    summary = []
    for level, group in LEVELS.items():
        get = lambda name: load(root / group / name, SETS)
        for split, (stem, out) in SPLITS.items():
            kept = tuple(c for c in range(11) if c not in out)
            pairs = {s: (get(run(stem, "hcg_cue", s)), get(run(stem, "hcg_cue_early", s))) for s in want}
            seeds_ok = [s for s in want if all(r is not None for r in pairs[s])]
            print(f"# {split} on {level}: trained without cells {list(out)}\n")
            if len(seeds_ok) < len(want):
                print(f"Seeds without both runs finished on this level, left out: {[s for s in want if s not in seeds_ok]}\n")
            if not seeds_ok:
                summary.append((split, level, "no runs", ""))
                continue
            C, E = [pairs[s][0] for s in seeds_ok], [pairs[s][1] for s in seeds_ok]
            assert all(set(x) == set(C[0]) for x in C + E), "not the same episodes"

            print("## 1. `cue` against the one network, same seed and same episodes\n")
            for label, cells in ((f"the cells left out {list(out)}", out), (f"the cells of the training set {list(kept)}", kept)):
                print(f"**{label}**\n")
                print("| seed | episodes | `cue` | one network | only `cue` succeeds / only the one network | p |")
                print("|---|---|---|---|---|---|")
                per = []
                for s, c, e in zip(seeds_ok, C, E):
                    sa, sb, a, b, p = pooled([c], [e], cells)
                    per.append((sa, sb))
                    print(f"| {s} | {sum(x['cell'] in cells for x in c.values())} | {sa:.3f} | {sb:.3f} | {a} / {b} | {p:.2g} |")
                sa, sb, a, b, p = pooled(C, E, cells)
                print(f"| mean (SD) and pooled | | {np.mean([x for x, _ in per]):.3f} ({sd([x for x, _ in per])}) | "
                      f"{np.mean([y for _, y in per]):.3f} ({sd([y for _, y in per])}) | {a} / {b} | {p:.2g} |")
                v = verdict(per, a, b, p, len(seeds_ok) == len(want))
                print(f"\nVerdict: the difference {v}.\n" if v != "not final" else "\nNot final: seeds are missing.\n")
                if cells is out:
                    summary.append((split, level, v, f"{sa:.3f} / {sb:.3f}, {a} / {b}, p = {p:.2g}"))

            print("## 2. The cells left out one by one (seeds pooled; + / - : `cue` above / below the one network, p < 0.05)\n")
            print("| cell | sensors that show the glass | `cue` | one network | only `cue` / only the one network | p |")
            print("|---|---|---|---|---|---|")
            for c in out:
                sa, sb, a, b, p = pooled(C, E, (c,))
                print(f"| {c} | {SHOWS[c]} | {sa:.3f} | {sb:.3f} | {a} / {b} | {p:.2g}"
                      + (" +" if p < 0.05 and a > b else " -" if p < 0.05 else "") + " |")

            print("\n## 3. What leaving the cells out costs: against the same arm trained on all 700 demos, in the "
                  "cells left out (same seed, pooled)\n")
            print("| arm | seeds with both runs | trained without the cells | trained on all cells | only without / "
                  "only with all | p |\n|---|---|---|---|---|---|")
            for name, arm in ARMS.items():
                both = [(s, get(run(stem, arm, s)), get(run(ALL, arm, s))) for s in seeds_ok]
                both = [(s, x, y) for s, x, y in both if x is not None and y is not None]
                if not both:
                    print(f"| {name} | none | | | | |")
                    continue
                sa, sb, a, b, p = pooled([x for _, x, _ in both], [y for _, _, y in both], out)
                print(f"| {name} | {[s for s, _, _ in both]} | {sa:.3f} | {sb:.3f} | {a} / {b} | {p:.2g} |")

            print("\n## 4. Against the plain rgbdt baseline trained on the same demos (seed 42)\n")
            plain = get(f"{stem}_imagenet")
            if plain is None or 42 not in seeds_ok:
                print("the plain baseline of this training set, or seed 42 of the arms, is not evaluated on this level\n")
            else:
                print("| arm | cells | the arm | plain rgbdt | only the arm / only plain | p |\n|---|---|---|---|---|---|")
                for name, arm in ARMS.items():
                    x = get(run(stem, arm, 42))
                    for label, cells in (("left out", out), ("of the training set", kept)):
                        sa, sb, a, b, p = pooled([x], [plain], cells)
                        print(f"| {name} | {label} | {sa:.3f} | {sb:.3f} | {a} / {b} | {p:.2g} |")
                print()

    print("# The verdicts side by side: `cue` against the one network in the cells left out\n")
    print("| training set | level | `cue` / one network, only `cue` / only the one network, pooled p | verdict |\n|---|---|---|---|")
    for split, level, v, nums in summary:
        print(f"| {split} | {level} | {nums} | {v} |")


if __name__ == "__main__":
    main()
