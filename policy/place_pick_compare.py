"""
Why the glass is picked more often on L0_vis_place than on the pick level L0_vis with the same inputs: the first close
of the gripper on both levels, from the episodes.json files of the final evaluations (no new runs; next to item 4 of
the read-out of results/hard_scenario/place_bench/README.md, not a test):

  python policy/place_pick_compare.py > results/hard_scenario/place_bench/pick_compare.txt

"Picked" is the success of the pick level (the glass lifted 10 cm) and `picked` on the place level (lifted 10 cm at
some point). The distance of a close is in the table plane, between the tool and the glass's grasp point
(`closes[0].d_target`). The two runs of a row pair differ in more than the level: see the README.
"""
import json
from collections import Counter
from math import comb
from pathlib import Path

import click
import numpy as np

VARIANTS = ("rgbdt", "rgbd", "rgbt", "rgb")
TEST_SETS = ("j_cells11", "k_cells11_heldout")
BINS = ((0, 1.0), (1.0, 1.5), (1.5, 3.0), (3.0, np.inf))   # cm
NEAR = 1.5                                                  # cm


def load(run):
    return {(ts, e["seed"]): e for ts in TEST_SETS for e in json.loads((Path(run) / ts / "episodes.json").read_text())}


def first(e):
    return 100 * e["closes"][0]["d_target"] if e["closes"] else np.nan


def sign_p(a, b):
    n, k = a + b, min(a, b)
    return 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


@click.command()
@click.option("--root", default="results/hard_scenario/place_bench")
@click.option("--pick_root", default="results/hard_scenario/grid_bench")
@click.option("--method_run", default="results/hard_scenario/cue_gate/L0_vis_rgbdt_700demos_hcg_cue",
              help="a run of the method on the pick level, for the last table")
def main(root, pick_root, method_run):
    runs = {v: (load(Path(pick_root) / f"L0_vis_{v}_700demos_imagenet"),
                load(Path(root) / f"L0_vis_place_{v}_700demos_imagenet")) for v in VARIANTS}
    got = {"pick": lambda e: e["outcome"] == "success", "place": lambda e: bool(e["picked"])}

    print("## The first close on the pick level and on the place level; seed 42, 1320 episodes per run\n")
    print("| inputs | level | picked | closes per episode | picked with a single close | first close: median distance "
          "to the glass (cm) | first close within 1.5 cm | " + " | ".join(
              f"picked when the first close is {a:g} to {b:g} cm away" if np.isfinite(b) else
              f"over {a:g} cm away" for a, b in BINS) + " |")
    print("|---|---|---|---|---|---|---|" + "---|" * len(BINS))
    for v, pair in runs.items():
        for level, eps in zip(("pick", "place"), pair):
            keys = sorted(eps)
            d = np.array([first(eps[k]) for k in keys])
            ok = np.array([got[level](eps[k]) for k in keys])
            n = np.array([len(eps[k]["closes"]) for k in keys])
            print(f"| {v} | {level} | {ok.mean():.3f} | {n.mean():.2f} | {(ok & (n == 1)).mean():.3f} | "
                  f"{np.nanmedian(d):.2f} | {(d < NEAR).mean():.3f} | "
                  + " | ".join(f"{ok[(d >= a) & (d < b)].mean():.3f}" for a, b in BINS) + " |")

    print(f"\n## The same episodes with the first close within {NEAR:g} cm of the glass on both levels\n")
    print("| inputs | episodes | picked on the pick level | picked here | only the pick level / only here | p | "
          "labels on the pick level of those not picked there |")
    print("|---|---|---|---|---|---|---|")
    for v, (pick, place) in runs.items():
        keys = [k for k in sorted(pick) if first(pick[k]) < NEAR and first(place[k]) < NEAR]
        a = sum(got["pick"](pick[k]) and not got["place"](place[k]) for k in keys)
        b = sum(got["place"](place[k]) and not got["pick"](pick[k]) for k in keys)
        labels = Counter(pick[k]["outcome"] for k in keys if not got["pick"](pick[k]))
        print(f"| {v} | {len(keys)} | {np.mean([got['pick'](pick[k]) for k in keys]):.3f} | "
              f"{np.mean([got['place'](place[k]) for k in keys]):.3f} | {a} / {b} | {sign_p(a, b):.2g} | "
              + ", ".join(f"{lab} {c}" for lab, c in labels.most_common()) + " |")

    print(f"\n## Picked after a first close within {NEAR:g} cm, per object: pick level / here (episodes)\n")
    objects = sorted({e["obj_name"] for e in runs[VARIANTS[0]][0].values()})
    print("| inputs | " + " | ".join(objects) + " |\n|---|" + "---|" * len(objects))
    for v, pair in runs.items():
        row = []
        for o in objects:
            cell = []
            for level, eps in zip(("pick", "place"), pair):
                r = [got[level](e) for e in eps.values() if e["obj_name"] == o and first(e) < NEAR]
                cell.append(f"{np.mean(r):.2f} ({len(r)})")
            row.append(" / ".join(cell))
        print(f"| {v} | " + " | ".join(row) + " |")

    print(f"\n## On the pick level: episodes lost after a first close within {NEAR:g} cm, seed 42\n")
    print("| run | success | first close within 1.5 cm | success after it | failed after it, share of all episodes | "
          "share of the run's failed episodes | their labels |")
    print("|---|---|---|---|---|---|---|")
    for name, eps in [("rgbdt baseline", runs["rgbdt"][0]), (Path(method_run).name, load(method_run))]:
        near = np.array([first(e) < NEAR for e in eps.values()])
        ok = np.array([got["pick"](e) for e in eps.values()])
        labels = Counter(e["outcome"] for e, m, o in zip(eps.values(), near, ok) if m and not o)
        print(f"| {name} | {ok.mean():.3f} | {near.mean():.3f} | {ok[near].mean():.3f} | {(near & ~ok).mean():.3f} | "
              f"{(near & ~ok).sum() / (~ok).sum():.2f} | " + ", ".join(f"{k} {c}" for k, c in labels.most_common())
              + " |")


if __name__ == "__main__":
    main()
