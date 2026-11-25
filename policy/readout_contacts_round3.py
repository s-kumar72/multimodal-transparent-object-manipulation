"""
Read-out of round 3 of the contact check (results/hard_scenario/contact_check/README.md): the controls of the method
(seed 42) and the baselines from the literature on L0_vis_ell, the pick level with the contact model of the place
level, from the episodes.json files of the evaluations (no new runs):

  PYTHONPATH=$PWD/policy python policy/readout_contacts_round3.py > results/hard_scenario/contact_check/readout_round3.txt

A. The pairs of rounds 1 and 2 of cue_gate (what the method is made of), each on L0_vis and on L0_vis_ell, same
episodes, sign test; the share of the method's gain over the rgbdt baseline that every run keeps; the cells in which a
run is below the best plain baseline of the cell. B. Items 1, 2 and 4 of the read-out of lit_base on L0_vis_ell.
Seed 42 throughout. The seeds of round 4 are read with policy/readout_round4.py --root
results/hard_scenario/contact_check. A run without a finished evaluation on L0_vis_ell is left out and named.
"""
from pathlib import Path

import click
import numpy as np

from readout_lit_base import LIT, first_close, load, mark, ok, versus

SETS = ("j_cells11", "k_cells11_heldout")
STEM = "L0_vis_rgbdt_700demos_"
PLAIN = {v: f"L0_vis_{v}_700demos_imagenet" for v in ("rgbdt", "rgbd", "rgbt", "rgb")}
RUNS = {   # name in the tables: (group of the results on L0_vis, run folder)
    "full": ("cue_gate", STEM + "hcg_full"), "cue": ("cue_gate", STEM + "hcg_cue"),
    "blind": ("cue_gate", STEM + "hcg_blind"), "one network": ("cue_gate", STEM + "hcg_cue_early"),
    "gate on map numbers": ("cue_gate", STEM + "hcg_cue_mapgate"), "no gate": ("cue_gate", STEM + "hcg_cue_nogate"),
    "auxiliary loss": ("cue_gate", STEM + "dp_aux"),
}
PAIRS = [  # what differs, first, second (rounds 1 and 2 of cue_gate)
    ("gating the image features", "full", "cue"),
    ("the images", "cue", "blind"),
    ("maps per sensor with the gate, against one network", "cue", "one network"),
    ("a gate that looks at the image, against a gate on map numbers", "cue", "gate on map numbers"),
    ("a gate among the sensors, against none", "cue", "no gate"),
    ("the cue in the conditioning, against the label as an auxiliary loss", "cue", "auxiliary loss"),
]
CELLS = range(11)


def rate(eps, cells=CELLS):
    return float(np.mean([ok(e) for e in eps.values() if e["cell"] in cells]))


def fmt(p):
    return "< 1e-300" if p < 1e-300 else f"{p:.2g}"


def relation(sa, sb, p):
    return "not separated" if p >= 0.05 else "above" if sa > sb else "below"


@click.command()
@click.option("--root", default="results/hard_scenario")
@click.option("--out_group", default="contact_check")
def main(root, out_group):
    root = Path(root)
    ell = lambda run: load(root / out_group / run, SETS)
    plain_old = {v: load(root / "grid_bench" / r, SETS) for v, r in PLAIN.items()}
    plain = {v: ell(r) for v, r in PLAIN.items()}
    assert all(r is not None for r in [*plain_old.values(), *plain.values()]), "the plain baselines are needed"
    old = {n: load(root / g / r, SETS) for n, (g, r) in RUNS.items()}
    new = {n: ell(r) for n, (g, r) in RUNS.items()}
    missing = [f"{n} ({RUNS[n][1]})" for n in RUNS if new[n] is None]
    if missing:
        print(f"Not evaluated on L0_vis_ell yet, left out: {', '.join(missing)}\n")
    have = [n for n in RUNS if new[n] is not None and old[n] is not None]
    keys = set(plain["rgbdt"])
    assert all(set(r) == keys for n in have for r in (old[n], new[n])), "not the same episodes"
    best = [max(PLAIN, key=lambda v: rate(plain[v], (c,))) for c in CELLS]     # per cell, seed 42, on L0_vis_ell

    print("## A. What the method is made of, on L0_vis and on L0_vis_ell (seed 42, 1320 episodes, same episodes)\n")
    print("| what differs | first | second | L0_vis: first / second | only the first succeeds / only the second | p | "
          "L0_vis_ell: first / second | only the first / only the second | p | on L0_vis_ell the first is |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for what, x, y in PAIRS:
        if x not in have or y not in have:
            continue
        sa, sb, a, b, p = versus(old[x], old[y])
        ta, tb, a2, b2, p2 = versus(new[x], new[y])
        print(f"| {what} | {x} | {y} | {sa:.3f} / {sb:.3f} | {a} / {b} | {fmt(p)} | {ta:.3f} / {tb:.3f} | {a2} / {b2} | "
              f"{fmt(p2)} | {relation(ta, tb, p2)} |")

    if "cue" in have:
        print("\nThe share of the gain of `cue` over the plain rgbdt baseline that a run keeps, and the cells in which "
              "it is below the best plain baseline of the cell (seed-42 runs, p < 0.05):\n")
        print("| run | L0_vis | kept | L0_vis_ell | kept | cells below their best plain baseline on L0_vis_ell | "
              "cells in which it differs from `cue` on L0_vis_ell: above / below |")
        print("|---|---|---|---|---|---|---|")
        bo, bn = rate(plain_old["rgbdt"]), rate(plain["rgbdt"])
        go, gn = rate(old["cue"]) - bo, rate(new["cue"]) - bn
        for n in have:
            under = [c for c in CELLS for sa, sb, _, _, p in [versus(new[n], plain[best[c]], (c,))] if p < 0.05 and sa < sb]
            up = [c for c in CELLS for sa, sb, _, _, p in [versus(new[n], new["cue"], (c,))] if p < 0.05 and sa > sb]
            down = [c for c in CELLS for sa, sb, _, _, p in [versus(new[n], new["cue"], (c,))] if p < 0.05 and sa < sb]
            diff = "" if n == "cue" else f"{up or 'none'} / {down or 'none'}"
            print(f"| {n} | {rate(old[n]):.3f} | {(rate(old[n]) - bo) / go:.2f} | {rate(new[n]):.3f} | "
                  f"{(rate(new[n]) - bn) / gn:.2f} | {under or 'none'} | {diff} |")
        print(f"| plain rgbdt | {bo:.3f} | 0 | {bn:.3f} | 0 | | |")

    print("\n## B. The baselines from the literature on L0_vis_ell (seed 42, 1320 episodes, same episodes)\n")
    lit_old = {n: load(root / "lit_base" / r, SETS) for n, r in LIT.items()}
    lit = {n: ell(r) for n, r in LIT.items()}
    missing = [f"{n} ({LIT[n]})" for n in LIT if lit[n] is None]
    if missing:
        print(f"Not evaluated on L0_vis_ell yet, left out: {', '.join(missing)}\n")
    lit = {n: r for n, r in lit.items() if r is not None}
    if not lit:
        return
    assert all(set(r) == keys for r in lit.values()), "not the same episodes"
    names = list(lit)

    print("**1. Against the plain baselines.**\n")
    print("| run | L0_vis | L0_vis_ell | plain rgbdt on L0_vis_ell | only the run succeeds / only plain rgbdt | p |")
    print("|---|---|---|---|---|---|")
    for n in names:
        sx, sy, a, b, p = versus(lit[n], plain["rgbdt"])
        print(f"| {n} | {rate(lit_old[n]):.3f} | {sx:.3f} | {sy:.3f} | {a} / {b} | {fmt(p)} |")
    print("\nPer cell (+ / - : above / below at p < 0.05), against plain rgbdt and against the cell's best plain "
          "baseline, all on L0_vis_ell:\n")
    print("| cell | plain rgbdt | best plain baseline | " + " | ".join(f"{n}: success, vs rgbdt, vs best" for n in names) + " |")
    print("|---|---|---|" + "---|" * len(names))
    for c in CELLS:
        row = []
        for n in names:
            sx, _, a, b, p = versus(lit[n], plain["rgbdt"], (c,))
            _, _, a2, b2, p2 = versus(lit[n], plain[best[c]], (c,))
            row.append(f"{sx:.3f}, {a} / {b}{mark(a, b, p)}, {a2} / {b2}{mark(a2, b2, p2)}")
        print(f"| {c} | {rate(plain['rgbdt'], (c,)):.3f} | {best[c]} {rate(plain[best[c]], (c,)):.3f} | "
              + " | ".join(row) + " |")

    print("\n**2. Against the method.**\n")
    print("| run | method run | run | method | only the run / only the method | p | cells with the run above / below the "
          "method (p < 0.05) |\n|---|---|---|---|---|---|---|")
    for n in names:
        for m in ("full", "cue"):
            if new[m] is None:
                continue
            sx, sy, a, b, p = versus(lit[n], new[m])
            per = [(c, *versus(lit[n], new[m], (c,))[2:]) for c in CELLS]
            up = [c for c, a_, b_, p_ in per if p_ < 0.05 and a_ > b_]
            down = [c for c, a_, b_, p_ in per if p_ < 0.05 and b_ > a_]
            print(f"| {n} | {m} | {sx:.3f} | {sy:.3f} | {a} / {b} | {fmt(p)} | {up or 'none'} / {down or 'none'} |")

    print("\n**4. The first close of the gripper.**\n")
    print("| run | first close within 1.5 cm of the glass | median distance (cm) | success after a first close within 1.5 cm |")
    print("|---|---|---|---|")
    for n, eps in [("plain rgbdt", plain["rgbdt"]), *lit.items(), *[(m, new[m]) for m in ("full", "cue") if new[m]]]:
        print(f"| {n} | {first_close(eps)} |")


if __name__ == "__main__":
    main()
