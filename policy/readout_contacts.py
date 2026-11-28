"""
Read-out of the contact check (results/hard_scenario/contact_check/README.md): checkpoints trained on L0_vis, evaluated
on L0_vis_ell (the same scenes and seeds with the contact model of the pick-and-place level), from the episodes.json
files (no new runs):

  python policy/readout_contacts.py > results/hard_scenario/contact_check/readout.txt

1. success on L0_vis against L0_vis_ell, same checkpoint and episodes; 2. the first close of the gripper under both
contact models, and on the place level for the baselines; 3. the verdict of the README per baseline with depth;
4. the method against the rgbdt baseline under both contact models; 5. success per cell on L0_vis_ell.
A run without a finished evaluation on L0_vis_ell is left out and named.
"""
import json
from math import comb
from pathlib import Path

import click
import numpy as np

BASELINES = {v: f"grid_bench/L0_vis_{v}_700demos_imagenet" for v in ("rgbdt", "rgbd", "rgbt", "rgb")}
METHOD = {"hcg_cue": "cue_gate/L0_vis_rgbdt_700demos_hcg_cue", "hcg_full": "cue_gate/L0_vis_rgbdt_700demos_hcg_full"}
TEST_SETS = ("j_cells11", "k_cells11_heldout")
DEPTH = (0, 1, 3, 5, 8, 10)                                 # cells in which depth shows the glass at least faintly
BINS = ((0, 1.0), (1.0, 1.5), (1.5, 3.0), (3.0, np.inf))   # cm
NEAR = 1.5                                                  # cm


def load(run):
    eps = {}
    for ts in TEST_SETS:
        f = Path(run) / ts / "episodes.json"
        if not f.exists():
            return None
        eps.update({(ts, e["seed"]): e for e in json.loads(f.read_text())})
    return eps


def ok(e):
    return e["outcome"] == "success"


def first(e):
    return 100 * e["closes"][0]["d_target"] if e["closes"] else np.nan


def sign_p(a, b):
    n, k = a + b, min(a, b)
    return 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def paired(x, y, keys, fx=ok, fy=ok):
    a = sum(fx(x[k]) and not fy(y[k]) for k in keys)
    b = sum(fy(y[k]) and not fx(x[k]) for k in keys)
    return a, b, sign_p(a, b)


def close_row(eps, got=ok):
    keys = sorted(eps)
    d = np.array([first(eps[k]) for k in keys])
    s = np.array([got(eps[k]) for k in keys])
    n = np.array([len(eps[k]["closes"]) for k in keys])
    return (f"{s.mean():.3f} | {n.mean():.2f} | {(d < NEAR).mean():.3f} | {s[d < NEAR].mean():.3f} | "
            + " | ".join(f"{s[(d >= a) & (d < b)].mean():.3f}" for a, b in BINS)), float(s[d < NEAR].mean())


@click.command()
@click.option("--root", default="results/hard_scenario")
@click.option("--out_group", default="contact_check")
@click.option("--place_group", default="place_bench")
def main(root, out_group, place_group):
    root = Path(root)
    runs, missing = {}, []
    for name, rel in {**BASELINES, **METHOD}.items():
        old, new = load(root / rel), load(root / out_group / Path(rel).name)
        if new is None:
            missing.append(name)
        else:
            assert set(old) == set(new), name
            runs[name] = (old, new)
    if missing:
        print(f"Not evaluated on L0_vis_ell yet, left out: {missing}\n")
    if not runs:
        return
    groups = [("all", range(11)), (f"depth shows the glass at least faintly {DEPTH}", DEPTH),
              ("it does not (2, 4, 6, 7, 9)", tuple(c for c in range(11) if c not in DEPTH))]

    print("## 1. Success on L0_vis and on L0_vis_ell: the same checkpoint on the same episodes, seed 42\n")
    print("| run | cells | L0_vis | L0_vis_ell | only L0_vis / only L0_vis_ell | p |\n|---|---|---|---|---|---|")
    for name, (old, new) in runs.items():
        for label, cells in groups:
            keys = [k for k in old if old[k]["cell"] in cells]
            a, b, p = paired(old, new, keys)
            print(f"| {name} | {label} | {np.mean([ok(old[k]) for k in keys]):.3f} | "
                  f"{np.mean([ok(new[k]) for k in keys]):.3f} | {a} / {b} | {p:.2g} |")

    print("\n## 2. The first close of the gripper (distance in the table plane to the glass's grasp point)\n")
    print("| run | level | success (picked, on the place level) | closes per episode | first close within 1.5 cm | "
          "success after a first close within 1.5 cm | " + " | ".join(
              f"after one {a:g} to {b:g} cm away" if np.isfinite(b) else f"over {a:g} cm away" for a, b in BINS) + " |")
    print("|---|---|---|---|---|---|" + "---|" * len(BINS))
    near = {}
    for name, (old, new) in runs.items():
        rows = [("L0_vis", old, ok), ("L0_vis_ell", new, ok)]
        if name in BASELINES:
            place = load(root / place_group / f"L0_vis_place_{name}_700demos_imagenet")
            if place is not None:
                rows.append(("L0_vis_place", place, lambda e: bool(e["picked"])))
        for level, eps, got in rows:
            text, near[name, level] = close_row(eps, got)
            print(f"| {name} | {level} | {text} |")

    print("\n## 3. Do the contacts explain the pick of the place level? (README, read-out 3; baselines with depth)\n")
    print("| run | a: success after a near first close on L0_vis | b: picked after one on L0_vis_place | "
          "x: success after one on L0_vis_ell | (x - a) / (b - a) | verdict |\n|---|---|---|---|---|---|")
    for name in ("rgbdt", "rgbd"):
        if (name, "L0_vis_ell") not in near or (name, "L0_vis_place") not in near:
            continue
        a, b, x = near[name, "L0_vis"], near[name, "L0_vis_place"], near[name, "L0_vis_ell"]
        f = (x - a) / (b - a)
        verdict = "explained by the contacts" if f >= 0.8 else "not by the contacts" if f <= 0.2 else "in part"
        print(f"| {name} | {a:.3f} | {b:.3f} | {x:.3f} | {f:.2f} | {verdict} |")

    print("\n## 4. The method against the rgbdt baseline under both contact models, same episodes\n")
    print("| method run | level | method | rgbdt baseline | difference | only the method / only the baseline | p |")
    print("|---|---|---|---|---|---|---|")
    if "rgbdt" in runs:
        for name in METHOD:
            if name not in runs:
                continue
            for level, i in (("L0_vis", 0), ("L0_vis_ell", 1)):
                m, b = runs[name][i], runs["rgbdt"][i]
                x, y, p = paired(m, b, sorted(m))
                sm, sb = np.mean([ok(e) for e in m.values()]), np.mean([ok(e) for e in b.values()])
                print(f"| {name} | {level} | {sm:.3f} | {sb:.3f} | {sm - sb:+.3f} | {x} / {y} | {p:.2g} |")

    print("\n## 5. Success per cell on L0_vis_ell (L0_vis in brackets), 120 episodes per cell\n")
    names = list(runs)
    print("| cell | " + " | ".join(names) + " |\n|---|" + "---|" * len(names))
    for c in [*range(11), None]:
        row = []
        for name in names:
            old, new = runs[name]
            sel = lambda eps: np.mean([ok(e) for e in eps.values() if c is None or e["cell"] == c])
            row.append(f"{sel(new):.3f} ({sel(old):.3f})")
        print(f"| {'all' if c is None else c} | " + " | ".join(row) + " |")


if __name__ == "__main__":
    main()
