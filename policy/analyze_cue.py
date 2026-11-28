"""
The hindsight cue in the evaluations of a cue-gated policy, from episodes.json (no new runs): how far the cue's grasp
point was from the glass, what the gate said per sensor, and the success with a right and with a wrong cue.

  python policy/analyze_cue.py results/hard_scenario/cue_gate/<run> [<run> ...]

The cue of an episode ("hcue": x, y, z, u, v, confidence, r_rgb, r_depth, r_thermal) and the position of the glass at
the start ("target_xyz") are written by the evaluation env; the cells are used for this report only.
"""
import json
import sys
from pathlib import Path

import numpy as np

TEST_SETS = ("j_cells11", "k_cells11_heldout")
RIGHT = 1.5   # cm: a cue within this distance of the glass counts as right


def load(runs):
    eps = []
    for run in runs:
        for ts in TEST_SETS:
            f = Path(run) / ts / "episodes.json"
            if f.exists():
                eps += [e for e in json.loads(f.read_text()) if "hcue" in e]
    return eps


def main(runs):
    eps = load(runs)
    assert eps, "no episodes with a cue in " + ", ".join(runs)
    cell = np.array([e["cell"] for e in eps])
    cue = np.array([e["hcue"] for e in eps])
    err = 100 * np.linalg.norm(cue[:, :2] - np.array([e["target_xyz"][:2] for e in eps]), axis=1)
    ok = np.array([e["outcome"] == "success" for e in eps])
    print(f"{', '.join(Path(r).name for r in runs)}: {len(eps)} episodes")
    print("| cell | n | success | cue error, cm (median) | cue within 1 cm | within 1.5 cm | success with a right cue | "
          "with a wrong cue | gate rgb / depth / thermal (median) | sensors on (r >= 0.5), mean |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    rate = lambda m: f"{ok[m].mean():.3f} ({int(m.sum())})" if m.sum() else "-"
    for c in [*sorted(set(cell.tolist())), None]:
        m = np.ones(len(eps), bool) if c is None else cell == c
        right = err < RIGHT
        print(f"| {'all' if c is None else c} | {int(m.sum())} | {ok[m].mean():.3f} | {np.median(err[m]):.2f} | "
              f"{(err[m] < 1).mean():.2f} | {right[m].mean():.2f} | {rate(m & right)} | {rate(m & ~right)} | "
              + " / ".join(f"{np.median(cue[m, 6 + i]):.2f}" for i in range(3))
              + f" | {(cue[m, 6:] >= 0.5).sum(1).mean():.2f} |")


if __name__ == "__main__":
    main(sys.argv[1:])
