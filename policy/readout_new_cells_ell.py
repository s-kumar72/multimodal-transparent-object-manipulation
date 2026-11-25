"""
The 15 visibility combinations that are not in the training set, on L0_vis and on L0_vis_ell
(results/hard_scenario/cue_gate/README.md, round 5), from the episodes.json files of the evaluations (no new runs):

  python policy/readout_new_cells_ell.py > results/hard_scenario/contact_check/readout_new_cells_ell.txt

Seed-42 runs, 1200 episodes (l_cells15_new, m_cells15_new_heldout), same episodes, two-sided sign test: every run
against `hcg_cue` and against the plain rgbdt baseline, on each level. `cue` against the one network over the three
seeds is setting 2 of policy/readout_round4.py (--root results/hard_scenario/contact_check for L0_vis_ell).
A run without a finished evaluation on a level is left out there and named.
"""
from pathlib import Path

import click

from readout_round4 import load, ok, sign_p

SETS = ("l_cells15_new", "m_cells15_new_heldout")
STEM = "L0_vis_rgbdt_700demos_"
RUNS = {   # name in the table: (group of the results on L0_vis, run)
    "`cue`": ("cue_gate", STEM + "hcg_cue"), "`full`": ("cue_gate", STEM + "hcg_full"),
    "one network": ("cue_gate", STEM + "hcg_cue_early"), "gate on map numbers": ("cue_gate", STEM + "hcg_cue_mapgate"),
    "no gate": ("cue_gate", STEM + "hcg_cue_nogate"), "auxiliary loss": ("cue_gate", STEM + "dp_aux"),
    "plain rgbdt": ("grid_bench", STEM + "imagenet"),
    "self-attention fusion (MulSA)": ("lit_base", STEM + "mulsa"), "sensor dropout": ("lit_base", STEM + "dropout_imagenet"),
    "stage-guided fusion (MS-Bot)": ("lit_base", STEM + "msbot"),
}


def versus(x, y):
    a = sum(ok(x[k]) and not ok(y[k]) for k in x)
    b = sum(ok(y[k]) and not ok(x[k]) for k in x)
    return sum(map(ok, x.values())) / len(x), a, b, sign_p(a, b)


@click.command()
@click.option("--root", default="results/hard_scenario")
@click.option("--ell_group", default="contact_check")
def main(root, ell_group):
    root = Path(root)
    for level in ("L0_vis", "L0_vis_ell"):
        runs = {n: load(root / (g if level == "L0_vis" else ell_group) / r, SETS) for n, (g, r) in RUNS.items()}
        missing = [n for n, r in runs.items() if r is None]
        runs = {n: r for n, r in runs.items() if r is not None}
        print(f"## The 15 new combinations on {level} (seed 42, same episodes)\n")
        if missing:
            print(f"Not evaluated on this level, left out: {missing}\n")
        if "`cue`" not in runs or "plain rgbdt" not in runs:
            print("`cue` and the plain rgbdt baseline are needed\n")
            continue
        assert all(set(r) == set(runs["`cue`"]) for r in runs.values()), "not the same episodes"
        print("| run | success | against `cue`: only the run succeeds / only `cue` | p | against plain rgbdt: only the "
              "run / only plain | p |\n|---|---|---|---|---|---|")
        for n, r in runs.items():
            s, a, b, p = versus(r, runs["`cue`"])
            _, a2, b2, p2 = versus(r, runs["plain rgbdt"])
            vs_cue = "" if n == "`cue`" else f"{a} / {b} | {p:.2g}"
            vs_plain = "" if n == "plain rgbdt" else f"{a2} / {b2} | {p2:.2g}"
            print(f"| {n} | {s:.3f} | {vs_cue or ' | '} | {vs_plain or ' | '} |")
        print()


if __name__ == "__main__":
    main()
