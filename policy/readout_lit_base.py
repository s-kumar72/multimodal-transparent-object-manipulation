"""
Read-out of the baselines from the literature on L0_vis (results/hard_scenario/lit_base/README.md), from the
episodes.json files of the final evaluations (no new runs):

  python policy/readout_lit_base.py > results/hard_scenario/lit_base/readout.txt

Per run of seed 42, on the same episodes (sign test): 1. against the plain rgbdt baseline, over all episodes and per
cell, and per cell against the cell's best plain baseline; 2. against the method (hcg_full, hcg_cue); 3. on the 15
new combinations against the rgbdt baseline and hcg_cue; 4. where the gripper first closes. A run without a finished
evaluation is left out and named.
"""
import json
from math import comb
from pathlib import Path

import click
import numpy as np

LIT = {"self-attention fusion (MulSA)": "L0_vis_rgbdt_700demos_mulsa",
       "sensor dropout": "L0_vis_rgbdt_700demos_dropout_imagenet",
       "stage-guided fusion (MS-Bot)": "L0_vis_rgbdt_700demos_msbot"}
PLAIN = {v: f"grid_bench/L0_vis_{v}_700demos_imagenet" for v in ("rgbdt", "rgbd", "rgbt", "rgb")}
METHOD = {"hcg_full": "cue_gate/L0_vis_rgbdt_700demos_hcg_full", "hcg_cue": "cue_gate/L0_vis_rgbdt_700demos_hcg_cue"}
TRAIN_SETS = ("j_cells11", "k_cells11_heldout")
NEW_SETS = ("l_cells15_new", "m_cells15_new_heldout")
NEAR = 1.5      # cm


def load(run, sets):
    eps = {}
    for ts in sets:
        f = Path(run) / ts / "episodes.json"
        if not f.exists():
            return None
        eps.update({(ts, e["seed"]): e for e in json.loads(f.read_text())})
    return eps


def ok(e):
    return e["outcome"] == "success"


def sign_p(a, b):
    n, k = a + b, min(a, b)
    return 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def versus(x, y, cells=None):
    keys = [k for k in x if cells is None or x[k]["cell"] in cells]
    a = sum(ok(x[k]) and not ok(y[k]) for k in keys)
    b = sum(ok(y[k]) and not ok(x[k]) for k in keys)
    return np.mean([ok(x[k]) for k in keys]), np.mean([ok(y[k]) for k in keys]), a, b, sign_p(a, b)


def mark(a, b, p):
    return " +" if p < 0.05 and a > b else " -" if p < 0.05 else ""


def first_close(eps):
    d = np.array([100 * e["closes"][0]["d_target"] if e["closes"] else np.nan for e in eps.values()])
    s = np.array([ok(e) for e in eps.values()])
    return f"{(d < NEAR).mean():.3f} | {np.nanmedian(d):.2f} | {s[d < NEAR].mean():.3f}"


@click.command()
@click.option("--root", default="results/hard_scenario")
@click.option("--group", default="lit_base")
def main(root, group):
    root = Path(root)
    lit = {n: load(root / group / r, TRAIN_SETS) for n, r in LIT.items()}
    missing = [n for n, r in lit.items() if r is None]
    if missing:
        print(f"Not finished yet, left out: {missing}\n")
    lit = {n: r for n, r in lit.items() if r is not None}
    if not lit:
        return
    plain = {v: load(root / r, TRAIN_SETS) for v, r in PLAIN.items()}
    method = {n: load(root / r, TRAIN_SETS) for n, r in METHOD.items()}
    names = list(lit)

    print("## 1. Against the plain baselines (seed 42, 1320 episodes, 120 per cell)\n")
    print("| run | success | plain rgbdt | only the run / only plain rgbdt | p |\n|---|---|---|---|---|")
    for n in names:
        sx, sy, a, b, p = versus(lit[n], plain["rgbdt"])
        print(f"| {n} | {sx:.3f} | {sy:.3f} | {a} / {b} | {p:.2g} |")
    print("\nPer cell (+ / - : above / below at p < 0.05), against plain rgbdt and against the cell's best plain baseline:\n")
    print("| cell | plain rgbdt | best plain baseline | " + " | ".join(f"{n}: success, vs rgbdt, vs best" for n in names) + " |")
    print("|---|---|---|" + "---|" * len(names))
    for c in range(11):
        rates = {v: np.mean([ok(e) for e in plain[v].values() if e["cell"] == c]) for v in plain}
        best = max(rates, key=rates.get)
        row = []
        for n in names:
            sx, _, a, b, p = versus(lit[n], plain["rgbdt"], (c,))
            _, _, a2, b2, p2 = versus(lit[n], plain[best], (c,))
            row.append(f"{sx:.3f}, {a} / {b}{mark(a, b, p)}, {a2} / {b2}{mark(a2, b2, p2)}")
        print(f"| {c} | {rates['rgbdt']:.3f} | {best} {rates[best]:.3f} | " + " | ".join(row) + " |")

    print("\n## 2. Against the method (seed 42, same episodes)\n")
    print("| run | method run | run | method | only the run / only the method | p | cells with the run above / below the "
          "method (p < 0.05) |\n|---|---|---|---|---|---|---|")
    for n in names:
        for m, eps in method.items():
            sx, sy, a, b, p = versus(lit[n], eps)
            cells = [(c, *versus(lit[n], eps, (c,))[2:]) for c in range(11)]
            up = [c for c, a_, b_, p_ in cells if p_ < 0.05 and a_ > b_]
            down = [c for c, a_, b_, p_ in cells if p_ < 0.05 and b_ > a_]
            print(f"| {n} | {m} | {sx:.3f} | {sy:.3f} | {a} / {b} | {p:.2g} | {up or 'none'} / {down or 'none'} |")

    print("\n## 3. The 15 new combinations (1200 episodes)\n")
    print("| run | success | plain rgbdt | only the run / only plain rgbdt | p | hcg_cue | only the run / only hcg_cue | p |")
    print("|---|---|---|---|---|---|---|---|")
    base_new, cue_new = load(root / PLAIN["rgbdt"], NEW_SETS), load(root / METHOD["hcg_cue"], NEW_SETS)
    for n in names:
        new = load(root / group / LIT[n], NEW_SETS)
        if new is None:
            print(f"| {n} | not evaluated yet | | | | | | |")
            continue
        sx, sy, a, b, p = versus(new, base_new)
        _, sz, a2, b2, p2 = versus(new, cue_new)
        print(f"| {n} | {sx:.3f} | {sy:.3f} | {a} / {b} | {p:.2g} | {sz:.3f} | {a2} / {b2} | {p2:.2g} |")

    print(f"\n## 4. The first close of the gripper (training cells)\n")
    print("| run | first close within 1.5 cm of the glass | median distance (cm) | success after a first close within 1.5 cm |")
    print("|---|---|---|---|")
    for n, eps in [("plain rgbdt", plain["rgbdt"]), *lit.items(), *method.items()]:
        print(f"| {n} | {first_close(eps)} |")


if __name__ == "__main__":
    main()
