"""
The true grasp point in the cue (results/hard_scenario/cue_gate/README.md, round 6c; a reference that reads the
simulator), from the episodes.json files of the evaluations (no new runs):

  python policy/readout_oracle_cue.py > results/hard_scenario/oracle_cue/readout.txt

On L0_vis and on L0_vis_ell: 1. the three seeds of `cue` with their own cue against the same checkpoints with the
true grasp point (same episodes, two-sided sign test), over all episodes and per cell; 2. the policy trained on the
hindsight labels themselves and evaluated with the true grasp point, next to them. A run without a finished
evaluation is left out and named.
"""
from pathlib import Path

import click
import numpy as np

from readout_round4 import load, ok, sign_p

SETS = ("j_cells11", "k_cells11_heldout")
CUE, LABEL = "L0_vis_rgbdt_700demos_hcg_cue", "L0_vis_rgbdt_700demos_hcg_oracle"
OWN = {"L0_vis": "cue_gate", "L0_vis_ell": "contact_check"}
SEEDS = (42, 43, 44)
SHOWS = ["rgb+depth+thermal", "depth+thermal", "rgb+thermal", "rgb+depth", "thermal", "depth", "rgb", "rgb (faint)",
         "depth (faint)", "thermal (faint)", "all faint"]


def rate(runs, cells=range(11)):
    return float(np.mean([ok(e) for r in runs for e in r.values() if e["cell"] in cells]))


def pooled(A, B, cells=range(11)):
    a = sum(ok(x[k]) and not ok(y[k]) for x, y in zip(A, B) for k in x if x[k]["cell"] in cells)
    b = sum(ok(y[k]) and not ok(x[k]) for x, y in zip(A, B) for k in x if x[k]["cell"] in cells)
    return rate(A, cells), rate(B, cells), a, b, sign_p(a, b)


@click.command()
@click.option("--root", default="results/hard_scenario")
@click.option("--oracle_group", default="oracle_cue")
def main(root, oracle_group):
    root = Path(root)
    for level, own_group in OWN.items():
        name = lambda s: CUE + ("" if s == 42 else f"_seed{s}")
        own = {s: load(root / own_group / name(s), SETS) for s in SEEDS}
        orc = {s: load(root / oracle_group / level / name(s), SETS) for s in SEEDS}
        seeds = [s for s in SEEDS if own[s] is not None and orc[s] is not None]
        print(f"# {level}\n")
        if len(seeds) < len(SEEDS):
            print(f"Seeds without both evaluations on this level, left out: {[s for s in SEEDS if s not in seeds]}\n")
        if seeds:
            O, T = [own[s] for s in seeds], [orc[s] for s in seeds]
            assert all(set(x) == set(y) for x, y in zip(O, T)), "not the same episodes"
            print("## 1. `cue`: the true grasp point against the policy's own cue, same checkpoint and same episodes\n")
            print("| seed | own cue | true grasp point | only with the true point succeeds / only with the own cue | p |")
            print("|---|---|---|---|---|")
            for s, x, y in zip(seeds, O, T):
                sa, sb, a, b, p = pooled([y], [x])
                print(f"| {s} | {sb:.3f} | {sa:.3f} | {a} / {b} | {p:.2g} |")
            sa, sb, a, b, p = pooled(T, O)
            print(f"| pooled | {sb:.3f} | {sa:.3f} | {a} / {b} | {p:.2g} |")
            print(f"\nOf the failures with the own cue, the share that is gone with the true point: "
                  f"{(sa - sb) / (1 - sb):.2f} (failure rate {1 - sb:.3f} -> {1 - sa:.3f}).\n")
            print("| cell | sensors that show the glass | own cue | true grasp point | only true / only own | p |\n|---|---|---|---|---|---|")
            for c in range(11):
                sa, sb, a, b, p = pooled(T, O, (c,))
                print(f"| {c} | {SHOWS[c]} | {sb:.3f} | {sa:.3f} | {a} / {b} | {p:.2g}"
                      + (" +" if p < 0.05 and a > b else " -" if p < 0.05 else "") + " |")
        lab = load(root / oracle_group / level / LABEL, SETS)
        print("\n## 2. The policy trained on the hindsight labels, evaluated with the true grasp point (seed 42)\n")
        if lab is None:
            print("not evaluated on this level yet\n")
            continue
        print("| run | success |\n|---|---|")
        print(f"| trained on the labels, true grasp point at test | {rate([lab]):.3f} |")
        for s in seeds[:1]:
            sa, sb, a, b, p = pooled([lab], [orc[s]])
            print(f"| `cue` of seed {s}, true grasp point at test | {sb:.3f} |")
            print(f"| `cue` of seed {s}, own cue | {rate([own[s]]):.3f} |")
            print(f"\nTrained on the labels against `cue` of seed {s}, both with the true point: only the first succeeds "
                  f"in {a}, only the second in {b}, p = {p:.2g}.\n")
        print("| cell | trained on the labels, true point |\n|---|---|")
        for c in range(11):
            print(f"| {c} | {rate([lab], (c,)):.3f} |")
        print()


if __name__ == "__main__":
    main()
