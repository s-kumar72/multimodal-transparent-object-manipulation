"""
Read-out of round 2 of hindsight cue gating (results/hard_scenario/cue_gate/README.md, "Policy runs, round 2"): the
controls of the cue against `hcg_cue` and against the baselines, from the episodes.json files of the final evaluations
(no new runs):

  python policy/readout_cue_controls.py > results/hard_scenario/cue_gate/readout_round2.txt

A control without a finished evaluation is left out. The cue inside the episodes is policy/analyze_cue.py.
"""
from pathlib import Path

import click
import numpy as np

from readout_cue_gate import BASE, BEST, GROUPS, REF, SHOWS, WHICH, load, mark, ok, versus

MAIN = ("cue", "L0_vis_rgbdt_700demos_hcg_cue")
CONTROLS = {"nogate": "L0_vis_rgbdt_700demos_hcg_cue_nogate", "mapgate": "L0_vis_rgbdt_700demos_hcg_cue_mapgate",
            "early": "L0_vis_rgbdt_700demos_hcg_cue_early", "dp_aux": "L0_vis_rgbdt_700demos_dp_aux"}


def pair(na, A, nb, B):
    sa, sb, a, b, p = versus(A, B, range(11))
    print(f"{na} against {nb}, all episodes: {sa:.3f} against {sb:.3f}, only {na} succeeds {a}, only {nb} {b}, "
          f"p = {p:.2g}{mark(p, sa - sb)}")
    cells = []
    for c in range(11):
        sa, sb, a, b, p = versus(A, B, (c,))
        if p < 0.05:
            cells.append(f"cell {c} {sa:.3f} against {sb:.3f} (p = {p:.2g})")
    print(f"  cells that differ with p < 0.05: {'; '.join(cells) or 'none'}")


@click.command()
@click.option("--root", default="results/hard_scenario/cue_gate")
@click.option("--base_root", default="results/hard_scenario/grid_bench")
def main(root, base_root):
    main_run = load(Path(root) / MAIN[1])
    runs = {n: load(Path(root) / f) for n, f in CONTROLS.items()}
    missing = [n for n, r in runs.items() if r is None]
    runs = {n: r for n, r in runs.items() if r is not None}
    base = load(Path(base_root) / BASE)
    best = {w: load(Path(base_root) / f"L0_vis_{w}_700demos_imagenet") for w in sorted(set(WHICH))}
    assert all(set(r) == set(base) for r in [main_run, *runs.values(), *best.values()]), "not the same episodes"
    if missing:
        print(f"Not finished yet, left out: {', '.join(missing)}\n")

    print(f"## 1. Success per cell (120 episodes per cell and run; seed 42). Marks: against {MAIN[0]} "
          f"({MAIN[1]}) on the same episodes, sign test, p < 0.05\n")
    print(f"| cell | sensors that show the glass | {MAIN[0]} | " + " | ".join(runs)
          + " | rgbdt ImageNet, seed 42 | best baseline of the cell (3 seeds, frozen) | reference |")
    print("|---|---|---|" + "---|" * (len(runs) + 3))
    for c in [*range(11), None]:
        cells = range(11) if c is None else (c,)
        row = []
        for r in runs.values():
            sa, sb, a, b, p = versus(r, main_run, cells)
            row.append(f"{sa:.3f}{mark(p, sa - sb)}")
        tail = f"{np.mean(BEST):.3f} | {np.mean(REF):.3f}" if c is None else f"{BEST[c]:.3f} | {REF[c]:.3f}"
        print(f"| {'macro average' if c is None else c} | {'' if c is None else SHOWS[c]} | "
              f"{versus(main_run, main_run, cells)[0]:.3f} | " + " | ".join(row)
              + f" | {versus(base, base, cells)[0]:.3f} | {tail} |")
    print()
    for n, r in runs.items():
        pair(MAIN[0], main_run, n, r)

    print("\n## 2. The controls against the baselines, same episodes\n")
    for n, r in runs.items():
        pair(n, r, "rgbdt ImageNet", base)
        below, above = [], []
        for c in range(11):
            sa, sb, a, b, p = versus(r, best[WHICH[c]], (c,))
            if p < 0.05:
                (below if sa < sb else above).append(c)
        print(f"  against the seed-42 run of the best baseline of each cell: below with p < 0.05 in {below or 'none'}, "
              f"above in {above or 'none'}")

    print("\n## Outside the fixed read-out: where the gripper closes (last close of the episode; no close counts as "
          "not there)\n")
    print("| cells | policy | success | closes within 4 cm of the glass | within 1.5 cm | success when within 1.5 cm | "
          "median distance, cm |")
    print("|---|---|---|---|---|---|---|")
    for g, cells in GROUPS.items():
        for n, r in {MAIN[0]: main_run, **runs, "rgbdt ImageNet, seed 42": base}.items():
            eps = [e for e in r.values() if e["cell"] in cells]
            d = np.array([100 * e["closes"][-1]["d_target"] if e["closes"] else np.inf for e in eps])
            s = np.array([ok(e) for e in eps])
            print(f"| {g} | {n} | {s.mean():.3f} | {(d < 4).mean():.3f} | {(d < 1.5).mean():.3f} | "
                  f"{s[d < 1.5].mean() if (d < 1.5).any() else float('nan'):.3f} | {np.median(d):.2f} |")


if __name__ == "__main__":
    main()
