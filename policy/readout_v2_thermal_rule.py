"""
The rule of the thermal states on the pilot of task T1 (results/hard_scenario/bench_v2/README.md, "The thermal states
are fixed by a rule", points 2 and 3), from the episodes.json files of the final evaluations (no new runs):

  python policy/readout_v2_thermal_rule.py > results/hard_scenario/bench_v2/readout_thermal_rule.txt

Success per cell of the four pilot runs of the level and of the reference run (plain rgbt on the 430 demos of the cells
in which thermal shows the glass fully or faintly), then the rule as written: a cell below 0.30 in all four pilot runs
counts as not solvable if the reference is below 0.30 in it as well; a cell that a run or the reference solves stays.
If the cells that are not solvable are cells in which thermal is faint, point 3 follows; if one of them is a cell in
which thermal shows the glass fully or is blind, the stop of section 8. A pilot run without a finished evaluation is
named and the verdict stays open: that run can still solve the cell.
"""
from pathlib import Path

import click

from readout_place_method import SHOWS, load, picked, rate
from readout_v2_pilot import RUNS, SOLVABLE

REFERENCE = "rgbt_430demos_thermal_imagenet"
REF_CELLS = (0, 1, 2, 4, 9, 10)       # the cells whose demos the reference is trained on
THERMAL_FAINT = (9, 10)


@click.command()
@click.option("--root", default="results/hard_scenario/bench_v2")
@click.option("--level", default="V2_place", show_default=True)
def main(root, level):
    root = Path(root)
    runs = {n: load(root / f"{level}_{r}") for n, r in RUNS.items()}
    missing = [n for n, r in runs.items() if r is None]
    runs = {n: r for n, r in runs.items() if r is not None}
    ref = load(root / f"{level}_{REFERENCE}")
    cols = dict(runs, **({"reference (plain rgbt, 430 demos)": ref} if ref is not None else {}))
    print(f"## {level}, seed 42, 1320 episodes per run: the rule of the thermal states\n")
    if missing:
        print(f"Pilot runs not finished yet: {missing}\n")
    if ref is None:
        print("The reference run has no finished evaluation.\n")
    print("| cell | sensors that show the glass | " + " | ".join(cols) + " |\n|---|---|" + "---|" * len(cols))
    for c in range(11):
        mark = "" if ref is None or c in REF_CELLS else " (no demos of this cell in the reference)"
        print(f"| {c} | {SHOWS[c]}{mark} | " + " | ".join(f"{rate(r, cells=(c,)):.3f}" for r in cols.values()) + " |")
    print("| macro average | | " + " | ".join(f"{rate(r):.3f}" for r in cols.values()) + " |")
    print("| glass picked | | " + " | ".join(f"{rate(r, picked):.3f}" for r in cols.values()) + " |\n")

    low = [c for c in range(11) if max(rate(r, cells=(c,)) for r in runs.values()) < SOLVABLE] if runs else []
    print(f"Cells below {SOLVABLE} in every finished pilot run ({len(runs)} of {len(RUNS)}): {low or 'none'}")
    if not low:
        print("No cell is below the line in every run: no thermal state is changed.")
        return
    if ref is not None:
        for c in low:
            print(f"  cell {c} ({SHOWS[c]}): best pilot run {max(rate(r, cells=(c,)) for r in runs.values()):.3f}, "
                  f"reference {rate(ref, cells=(c,)):.3f}")
    if missing or ref is None:
        print("Verdict: open. " + ("A pilot run that has not ended can still solve the cell. " if missing else "")
              + ("The reference has not ended." if ref is None else ""))
        return
    unsolved = [c for c in low if rate(ref, cells=(c,)) < SOLVABLE]
    stay = [c for c in low if c not in unsolved]
    if stay:
        print(f"Cells the reference solves, they stay as they are: {stay}")
    if not unsolved:
        print("Verdict: every such cell is solved by the reference: no thermal state is changed.")
    elif all(c in THERMAL_FAINT for c in unsolved):
        print(f"Verdict: not solvable: {unsolved}, cells in which thermal is faint. Point 3: thermal \"faint\" becomes "
              "the real p10 to p25 (10 to 26 noise units) and \"shows\" the real p25 to p90 (26 to 103), the level "
              "file is fitted to these with sim/measure_visibility.py, and the pilot runs of the level are repeated once.")
    else:
        other = [c for c in unsolved if c not in THERMAL_FAINT]
        print(f"Verdict: not solvable: {unsolved}; {other} is not a cell in which thermal is faint. The stop of section 8 "
              "applies: the states are changed with the user.")


if __name__ == "__main__":
    main()
