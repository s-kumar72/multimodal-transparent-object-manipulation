"""
The choice of the training setting of the frozen second version of the benchmark (results/hard_scenario/bench_v2/
README.md, section 13, part A 1), from the episodes.json files of the validation evaluations (no new runs):

  python policy/readout_v2_tune.py > results/hard_scenario/v2_tune/readout.txt

T1 (V2b_place, 700 demos), seed 42: plain rgbdt, `cue` and the one network, each trained with {101, 201} epochs x
{1e-4, 3e-4} learning rate and evaluated on 660 validation episodes (the scenes of j_cells11 with the seeds 200000 + k,
which are in no test set). Per policy the setting with the highest macro average over the eleven cells is chosen;
settings equal to three decimals go to fewer epochs, then to 1e-4. Nothing is chosen for a policy before its four runs
have a finished evaluation.
"""
import json
from pathlib import Path

import click
import numpy as np

POLICIES = {"plain rgbdt": "imagenet", "`cue`": "hcg2_cue", "one network": "hcg2_cue_early"}
SETTINGS = [(101, "1e-4"), (101, "3e-4"), (201, "1e-4"), (201, "3e-4")]      # in the order ties are broken
VAL_START = 200000


def load(folder):
    f = Path(folder) / "j_cells11" / "episodes.json"
    if not f.exists():
        return None
    eps = json.loads(f.read_text())
    assert all(e["seed"] >= VAL_START for e in eps), f"{folder}: not the validation seeds"
    return eps


def macro(eps):
    cells = sorted({e["cell"] for e in eps})
    return float(np.mean([np.mean([e["outcome"] == "success" for e in eps if e["cell"] == c]) for c in cells]))


@click.command()
@click.option("--root", default="results/hard_scenario/v2_tune")
def main(root):
    root = Path(root)
    print("## Tuning on T1 (V2b_place, 700 demos, seed 42): macro average on the 660 validation episodes\n")
    print("| policy | " + " | ".join(f"{e} epochs, {lr}" for e, lr in SETTINGS) + " | chosen |\n|---|" + "---|" * (len(SETTINGS) + 1))
    for name, tag in POLICIES.items():
        runs = [load(root / f"V2b_place_rgbdt_700demos_{tag}_e{e}_lr{lr}_s42") for e, lr in SETTINGS]
        m = [None if r is None else round(macro(r), 3) for r in runs]
        if any(x is None for x in m):
            chosen = "open: " + ", ".join(f"{e} epochs {lr}" for (e, lr), x in zip(SETTINGS, m) if x is None) + " not finished"
        else:
            e, lr = SETTINGS[int(np.argmax(m))]           # argmax keeps the first of equal ones: fewer epochs, then 1e-4
            chosen = f"{e} epochs, {lr}"
        print(f"| {name} | " + " | ".join("-" if x is None else f"{x:.3f}" for x in m) + f" | {chosen} |")
    print("\nThe setting of plain rgbdt is that of the four plain baselines and of the layout reference on every task; "
          "those of `cue` and of the one network are theirs on every task.")


if __name__ == "__main__":
    main()
