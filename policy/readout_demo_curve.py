"""
Success against the number of demos (results/hard_scenario/cue_gate/README.md, round 6b): the arm `cue` and the plain
rgbdt baseline trained on 140, 350 and 700 demos of L0_vis, from the episodes.json files of the final evaluations (no
new runs):

  python policy/readout_demo_curve.py > results/hard_scenario/cue_gate/readout_demo_curve.txt

On L0_vis and on L0_vis_ell (the evaluations in contact_check): 1. the macro average per seed, mean and standard
deviation, per number of demos; 2. `cue` against the plain baseline at each number of demos; 3. `cue` trained on fewer
demos against the plain baseline trained on all 700. Runs are paired by seed on the same episodes and pooled;
two-sided sign test. A run without a finished evaluation is left out and named.
"""
from pathlib import Path

import click
import numpy as np

from readout_round4 import load, ok, sign_p

SETS = ("j_cells11", "k_cells11_heldout")
SIZES = {140: "L0_vis_rgbdt_140demos_fifth", 350: "L0_vis_rgbdt_350demos_half", 700: "L0_vis_rgbdt_700demos"}
ARMS = {"`cue`": "hcg_cue", "plain rgbdt": "imagenet"}
SEEDS = (42, 43, 44)


def group(level, n, arm):
    if level == "L0_vis_ell":
        return "contact_check"
    return "grid_bench" if (n == 700 and arm == "imagenet") else "cue_gate"


def pooled(A, B):
    a = sum(ok(x[k]) and not ok(y[k]) for x, y in zip(A, B) for k in x)
    b = sum(ok(y[k]) and not ok(x[k]) for x, y in zip(A, B) for k in x)
    r = lambda R: float(np.mean([ok(e) for x in R for e in x.values()]))
    return r(A), r(B), a, b, sign_p(a, b)


@click.command()
@click.option("--root", default="results/hard_scenario")
def main(root):
    root = Path(root)
    for level in ("L0_vis", "L0_vis_ell"):
        R = {(n, a): {s: load(root / group(level, n, arm) / (f"{stem}_{arm}" + ("" if s == 42 else f"_seed{s}")), SETS)
                      for s in SEEDS} for n, stem in SIZES.items() for a, arm in ARMS.items()}
        print(f"# {level}\n")
        missing = {f"{a}, {n} demos": [s for s in SEEDS if R[n, a][s] is None] for (n, a) in R}
        missing = {k: v for k, v in missing.items() if v}
        if missing:
            print(f"Seeds without a finished evaluation on this level, left out: {missing}\n")
        print("## 1. Macro average per seed (1320 episodes per run)\n")
        print("| demos | run | " + " | ".join(f"seed {s}" for s in SEEDS) + " | mean | standard deviation |")
        print("|---|---|" + "---|" * (len(SEEDS) + 2))
        for (n, a), runs in R.items():
            m = [None if r is None else float(np.mean([ok(e) for e in r.values()])) for r in runs.values()]
            have = [x for x in m if x is not None]
            print(f"| {n} | {a} | " + " | ".join("-" if x is None else f"{x:.3f}" for x in m)
                  + (f" | {np.mean(have):.3f} | {np.std(have, ddof=1):.3f} |" if len(have) > 1 else
                     f" | {have[0]:.3f} | - |" if have else " | - | - |"))

        def pair(n1, a1, n2, a2):
            both = [s for s in SEEDS if R[n1, a1][s] is not None and R[n2, a2][s] is not None]
            if not both:
                return f"| {a1}, {n1} demos | {a2}, {n2} demos | none | | | | |"
            sa, sb, a, b, p = pooled([R[n1, a1][s] for s in both], [R[n2, a2][s] for s in both])
            rel = "not separated" if p >= 0.05 else "above" if a > b else "below"
            return f"| {a1}, {n1} demos | {a2}, {n2} demos | {both} | {sa:.3f} | {sb:.3f} | {a} / {b}, p = {p:.2g} | {rel} |"

        head = ("| first | second | seeds with both | first | second | only the first succeeds / only the second | the first is |\n"
                "|---|---|---|---|---|---|---|")
        print("\n## 2. `cue` against the plain baseline at each number of demos (same seed, pooled)\n\n" + head)
        for n in SIZES:
            print(pair(n, "`cue`", n, "plain rgbdt"))
        print("\n## 3. `cue` on fewer demos against the plain baseline on all 700 (same seed, pooled)\n\n" + head)
        for n in (140, 350):
            print(pair(n, "`cue`", 700, "plain rgbdt"))
        print()


if __name__ == "__main__":
    main()
