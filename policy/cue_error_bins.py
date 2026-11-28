"""
How much of a cue control's deficit against the method's cue is the cue itself, from the episodes.json files of the
final evaluations (no new runs; next to the read-outs of rounds 2 and 3a, not a test):

  python policy/cue_error_bins.py > results/hard_scenario/cue_gate/cue_error_bins.txt

Per run and pair of test sets: the share of episodes and the success per bin of the cue's error (distance in the table
plane between the cue's grasp point and the glass, as in policy/analyze_cue.py), and "expected": the success the run
would have if its policy succeeded as often as `hcg_cue` in every bin, so that only the cue's errors differ. The rest
of the deficit is the policy succeeding less often at the same cue error.
"""
import json
from pathlib import Path

import click
import numpy as np

RUNS = ("hcg_cue", "hcg_cue_early", "hcg_cue_mapgate", "hcg_cue_nogate")
SETS = {"the 11 training cells (j_cells11, k_cells11_heldout)": ("j_cells11", "k_cells11_heldout"),
        "the 15 new cells (l_cells15_new, m_cells15_new_heldout)": ("l_cells15_new", "m_cells15_new_heldout")}
BINS = ((0, 0.5), (0.5, 1.0), (1.0, 1.5), (1.5, np.inf))   # cm


@click.command()
@click.option("--root", default="results/hard_scenario/cue_gate")
def main(root):
    for label, sets in SETS.items():
        print(f"## Success by the cue's error, {label}; seed 42\n")
        print("| run | success | " + " | ".join(
            f"{a} to {b} cm: share / success" if np.isfinite(b) else f"over {a} cm: share / success" for a, b in BINS)
            + " | expected with the success of `hcg_cue` per bin | of the deficit, the cue's part |")
        print("|---|---|" + "---|" * (len(BINS) + 2))
        ref = top = None
        for run in RUNS:
            eps = [e for ts in sets for e in
                   json.loads((Path(root) / f"L0_vis_rgbdt_700demos_{run}" / ts / "episodes.json").read_text())]
            err = np.array([100 * np.linalg.norm(np.subtract(e["hcue"][:2], e["target_xyz"][:2])) for e in eps])
            ok = np.array([e["outcome"] == "success" for e in eps])
            share = [((err >= a) & (err < b)).mean() for a, b in BINS]
            succ = [ok[(err >= a) & (err < b)].mean() for a, b in BINS]
            if ref is None:
                ref, top = succ, ok.mean()
            exp = float(np.dot(share, ref))
            part = "" if run == RUNS[0] else f"{(top - exp) / (top - ok.mean()):.2f}"
            print(f"| `{run}` | {ok.mean():.3f} | " + " | ".join(f"{s:.2f} / {r:.2f}" for s, r in zip(share, succ))
                  + f" | {exp:.3f} | {part} |")
        print()


if __name__ == "__main__":
    main()
