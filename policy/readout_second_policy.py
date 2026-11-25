"""
Read-out of the second policy (results/hard_scenario/second_policy/README.md): the action head of ACT on the
benchmark's encoders, without and with the hindsight cue, from the episodes.json files of the final evaluations (no
new runs):

  python policy/readout_second_policy.py > results/hard_scenario/second_policy/readout.txt

On each level (L0_vis; L0_vis_ell, the evaluations in contact_check): 1. the macro average per seed, next to Diffusion
Policy without and with the cue; 2. the cue in the second policy: with against without, per seed and pooled, per cell,
with the verdict of the README; 3. where the gripper first closes; 4. next to it, not a criterion: the two heads
against each other. Runs are paired by seed on the same episodes; two-sided sign test. A seed without both runs of the
second policy finished on a level is left out there and named.
"""
from pathlib import Path

import click
import numpy as np

from readout_lit_base import TRAIN_SETS, first_close, load, ok, sign_p

RUNS = {   # name: (group of the results on L0_vis, run of seed 42)
    "ACT head": ("second_policy", "L0_vis_rgbdt_700demos_act"),
    "ACT head + cue": ("second_policy", "L0_vis_rgbdt_700demos_act_cue"),
    "Diffusion Policy": ("grid_bench", "L0_vis_rgbdt_700demos_imagenet"),
    "Diffusion Policy + cue": ("cue_gate", "L0_vis_rgbdt_700demos_hcg_cue"),
}
LEVELS = {"L0_vis": None, "L0_vis_ell": "contact_check"}
CELLS = range(11)


def rate(runs, cells=CELLS):
    return float(np.mean([ok(e) for r in runs for e in r.values() if e["cell"] in cells]))


def pooled(A, B, cells=CELLS):
    a = b = 0
    for x, y in zip(A, B):
        keys = [k for k in x if x[k]["cell"] in cells]
        a += sum(ok(x[k]) and not ok(y[k]) for k in keys)
        b += sum(ok(y[k]) and not ok(x[k]) for k in keys)
    return rate(A, cells), rate(B, cells), a, b, sign_p(a, b)


def sd(v):
    return f"{np.std(v, ddof=1):.3f}" if len(v) > 1 else "-"


@click.command()
@click.option("--root", default="results/hard_scenario")
@click.option("--seeds", default="42,43,44", help="for testing the script on the seeds that exist")
@click.option("--act", default=None, help="for testing: <group>/<run> standing in for the ACT head")
@click.option("--act_cue", default=None, help="for testing: <group>/<run> standing in for the ACT head with the cue")
def main(root, seeds, act, act_cue):
    root, want = Path(root), [int(s) for s in seeds.split(",")]
    runs = dict(RUNS)
    for name, stand_in in (("ACT head", act), ("ACT head + cue", act_cue)):
        if stand_in:
            runs[name] = tuple(stand_in.split("/"))
    for level, group in LEVELS.items():
        get = lambda n, s: load(root / (group or runs[n][0]) / (runs[n][1] + ("" if s == 42 else f"_seed{s}")), TRAIN_SETS)
        R = {n: {s: get(n, s) for s in want} for n in runs}
        seeds_ok = [s for s in want if R["ACT head"][s] is not None and R["ACT head + cue"][s] is not None]
        print(f"# {level}\n")
        if len(seeds_ok) < len(want):
            print(f"Seeds without both runs of the second policy finished on this level, left out: "
                  f"{[s for s in want if s not in seeds_ok]}\n")
        if not seeds_ok:
            continue
        A, C = [R["ACT head"][s] for s in seeds_ok], [R["ACT head + cue"][s] for s in seeds_ok]

        print("## 1. Macro average per seed (1320 episodes per run)\n")
        print("| run | " + " | ".join(f"seed {s}" for s in want) + " | mean | standard deviation |")
        print("|---|" + "---|" * (len(want) + 2))
        for n in runs:
            m = [rate([R[n][s]]) if R[n][s] is not None else None for s in want]
            have = [x for x in m if x is not None]
            print(f"| {n} | " + " | ".join("-" if x is None else f"{x:.3f}" for x in m)
                  + (f" | {np.mean(have):.3f} | {sd(have)} |" if have else " | - | - |"))

        print("\n## 2. The cue in the second policy: with the cue against without, same seed and same episodes\n")
        print("| seed | with the cue | without | only with the cue succeeds / only without | p |\n|---|---|---|---|---|")
        per = []
        for s, x, y in zip(seeds_ok, C, A):
            sa, sb, a, b, p = pooled([x], [y])
            per.append((sa, sb))
            print(f"| {s} | {sa:.3f} | {sb:.3f} | {a} / {b} | {p:.2g} |")
        sa, sb, a, b, p = pooled(C, A)
        print(f"| pooled | {sa:.3f} | {sb:.3f} | {a} / {b} | {p:.2g} |")
        if len(seeds_ok) == len(want):
            v = ("does not hold" if p >= 0.05 else "holds over the seeds" if a > b and all(c > e for c, e in per) else "mixed")
            print(f"\nVerdict: \"the cue carries over to the second policy\" {v} on {level}.")
        else:
            print("\nNot final: seeds are missing.")
        cells = [(c, *pooled(C, A, (c,))) for c in CELLS]
        print(f"Cells with the cue above (p < 0.05): {[c for c, x, y, _, _, pp in cells if pp < 0.05 and x > y] or 'none'}; "
              f"below: {[c for c, x, y, _, _, pp in cells if pp < 0.05 and x < y] or 'none'}.\n")
        print("| cell | with the cue | without | only with / only without | p |\n|---|---|---|---|---|")
        for c, x, y, a, b, pp in cells:
            print(f"| {c} | {x:.3f} | {y:.3f} | {a} / {b} | {pp:.2g} |")

        print("\n## 3. The first close of the gripper (seed 42)\n")
        print("| run | first close within 1.5 cm of the glass | median distance (cm) | success after a first close within 1.5 cm |")
        print("|---|---|---|---|")
        for n in runs:
            if R[n][42] is not None:
                print(f"| {n} | {first_close(R[n][42])} |")

        print("\n## 4. Next to it, not a criterion: the two heads against each other (same seed, pooled)\n")
        print("| pair | seeds with both | ACT head | Diffusion Policy | only the ACT head succeeds / only Diffusion Policy | p |")
        print("|---|---|---|---|---|---|")
        for x, y in (("ACT head", "Diffusion Policy"), ("ACT head + cue", "Diffusion Policy + cue")):
            both = [s for s in seeds_ok if R[y][s] is not None]
            if both:
                sa, sb, a, b, p = pooled([R[x][s] for s in both], [R[y][s] for s in both])
                print(f"| {x} against {y} | {both} | {sa:.3f} | {sb:.3f} | {a} / {b} | {p:.2g} |")
        print()


if __name__ == "__main__":
    main()
