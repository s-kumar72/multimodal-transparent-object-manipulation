"""
Where the policies of the L0_vis baseline programme fail, from the episodes.json files of their evaluations (no new
runs): how far from the glass the gripper was when it closed for the last time (the grasp the outcome is judged from,
policy/glass_gym_env.py), and what happened then.

  python policy/analyze_failures.py [--root results/hard_scenario/grid_bench] > failure_analysis.txt

Printed: (1) success against the horizontal distance of that close, all listed runs pooled; (2) per group of cells and
per policy: success, the share of episodes in which the gripper closed within 4 cm of the glass ("reached") and within
1.5 cm ("placed"), the median distance, success among the placed episodes, and the share of each failure type.
"""
import json
from pathlib import Path

import click
import numpy as np

TEST_SETS = ("j_cells11", "k_cells11_heldout")
SEEDS = ("", "_seed43", "_seed44")
POLICIES = {   # name -> run folders (three seeds pooled where they exist)
    "rgbdt (3 seeds)": [f"L0_vis_rgbdt_700demos{s}" for s in SEEDS],
    "rgbd (3 seeds)": [f"L0_vis_rgbd_700demos{s}" for s in SEEDS],
    "rgbt (3 seeds)": [f"L0_vis_rgbt_700demos{s}" for s in SEEDS],
    "rgb (3 seeds)": [f"L0_vis_rgb_700demos{s}" for s in SEEDS],
    "rgbdt, sensor dropout (3 seeds)": [f"L0_vis_rgbdt_700demos_drop{s}" for s in SEEDS],
    "rgbdt, ImageNet (3 seeds)": [f"L0_vis_rgbdt_700demos_imagenet{s}" for s in SEEDS],
    "depth only, 330 fully visible": ["L0_vis_depth_330demos_full"],
    "depth only, 430": ["L0_vis_depth_430demos_shown"],
    "depth only, all 700": ["L0_vis_depth_700demos"],
    "thermal only, 330 fully visible": ["L0_vis_thermal_330demos_full"],
    "thermal only, all 700": ["L0_vis_thermal_700demos"],
    "rgb only, 330 fully visible": ["L0_vis_rgb_330demos_full"],
    "rgb only, 430": ["L0_vis_rgb_430demos_shown"],
}
GROUPS = {
    "depth shows the glass fully (cells 0, 1, 3, 5)": [0, 1, 3, 5],
    "depth faint (cells 8, 10)": [8, 10],
    "no depth, thermal shows fully (cells 2, 4)": [2, 4],
    "no depth, rgb shows fully (cells 2, 6)": [2, 6],
    "rgb faint only (cell 7)": [7],
    "thermal faint only (cell 9)": [9],
    "all cells without depth (2, 4, 6, 7, 9)": [2, 4, 6, 7, 9],
}
OUTCOMES = ["missed_grasp", "dropped", "wrong_height", "other", "wrong_object", "timeout"]
REACHED, PLACED = 4.0, 1.5   # cm
BINS = [(0, 0.5), (0.5, 1), (1, 1.5), (1.5, 2), (2, 3), (3, 4), (4, 8), (8, 1e9)]


def load(root, runs):
    """Episodes of the run folders that exist (both test sets); None if none of them has results."""
    eps = []
    for run in runs:
        for ts in TEST_SETS:
            f = Path(root) / run / ts / "episodes.json"
            if f.exists():
                eps += json.loads(f.read_text())
    return eps or None


def dist(e):
    """Horizontal distance (cm) between the tool and the glass at the last gripper close; inf if it never closed."""
    return 100 * e["closes"][-1]["d_target"] if e["closes"] else float("inf")


@click.command()
@click.option("--root", default="results/hard_scenario/grid_bench")
def main(root):
    data = {name: load(root, runs) for name, runs in POLICIES.items()}
    data = {k: v for k, v in data.items() if v}
    pooled = [e for eps in data.values() for e in eps]
    d = np.array([dist(e) for e in pooled])
    ok = np.array([e["outcome"] == "success" for e in pooled])
    print(f"Success against the horizontal distance between gripper and glass at the last close "
          f"({len(pooled)} episodes of {len(data)} policies pooled)")
    print("| distance (cm) | episodes | success |")
    print("|---|---|---|")
    for lo, hi in BINS:
        m = (d >= lo) & (d < hi)
        print(f"| {lo}-{hi} | {m.sum()} | {ok[m].mean():.3f} |" if hi < 1e9 else f"| over {lo} or never closed | {m.sum()} | {ok[m].mean():.3f} |")
    for group, cells in GROUPS.items():
        print(f"\n{group}")
        print("| policy | episodes | success | reached (< 4 cm) | placed (< 1.5 cm) | median distance (cm) | "
              "success when placed | " + " | ".join(OUTCOMES) + " |")
        print("|---|---|---|---|---|---|---|" + "---|" * len(OUTCOMES))
        for name, eps in data.items():
            g = [e for e in eps if e["cell"] in cells]
            d = np.array([dist(e) for e in g])
            ok = np.array([e["outcome"] == "success" for e in g])
            placed = d < PLACED
            when = f"{ok[placed].mean():.3f}" if placed.sum() >= 20 else "-"
            outs = " | ".join(f"{np.mean([e['outcome'] == o for e in g]):.3f}" for o in OUTCOMES)
            print(f"| {name} | {len(g)} | {ok.mean():.3f} | {(d < REACHED).mean():.3f} | {placed.mean():.3f} | "
                  f"{np.median(d):.1f} | {when} | {outs} |")


if __name__ == "__main__":
    main()
