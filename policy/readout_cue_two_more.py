"""
Further description of the cue for two targets (T3) on the training demos, out of fold. Not fixed in advance: written
after policy/readout_cue_two.py was read (results/hard_scenario/bench_v2/README.md, section 14), to see where the cues
that are not within 1.5 cm lie. No policy, no test scene.

  python policy/readout_cue_two_more.py > results/hard_scenario/bench_v2/fits/V2b_two_700/readout_cue_two_more.txt

1. per group of pair-cells: the error of each grasp cue (median, within 1.5 / 3 / 5 cm), and of the cues more than
   5 cm off, the share within 5 cm of the other glass's grasp point and the share near neither;
2. with 5 cm in place of 1.5 cm: both cues at their own glass, both at the same glass, the two swapped;
3. by the state of the glass a cue points at (which sensors show it), whatever the other glass's state;
4. the release cues: at their own coaster, at the other one (5 cm).
"""
import sys

import click
import numpy as np
import zarr

from policy.hindsight_cue_nth import nth_goal

GROUPS = {"the same sensors show both (0-3, 10)": (0, 1, 2, 3, 10), "different sensors show them (4-9)": (4, 5, 6, 7, 8, 9),
          "the first fully, the second faintly (11)": (11,), "all": tuple(range(12))}
ARMS = {"a map and a gate per sensor": ("g1", "g2"), "one network": ("g1_early", "g2_early")}
SENSORS = ("colour", "depth", "thermal")


def err(cue, point):
    return 100 * np.linalg.norm(cue[:, :2] - point[:, :2], axis=1)


def state(s):
    s = tuple(s)
    if s == (0, 0, 0):
        return "all three show it"
    if s == (2, 2, 2):
        return "all three faintly"
    return " and ".join(n for n, v in zip(SENSORS, s) if v == 0) + " only"


@click.command()
@click.option("--demos", default="results/data/sim_demos_V2b_two_700.zarr")
@click.option("--stem", default="data/hcue/V2b_two_700")
@click.option("--level", default="V2b_two")
def main(demos, stem, level):
    sys.path.insert(0, "sim")
    from config import load_config
    from testsets import manifest
    import policy.hindsight_cue as H
    z = zarr.open(demos, "r")
    cells = z["meta/ep_cell"][:]
    p = [nth_goal(demos, "grasp", None, k) for k in (0, 1)]
    cue = lambda n: np.load(f"{stem}_{n}/train_cue.npy")
    print(f"# The cue for two targets, further description (not fixed in advance): {len(cells)} training demos, out of fold\n")

    print("## 1. Error of each grasp cue, and where the cues more than 5 cm off lie\n")
    print("| design | demos | cue | median, cm | within 1.5 | within 3 | within 5 | off and within 5 cm of the other glass | off and near neither |\n|---|---|---|---|---|---|---|---|---|")
    for arm, names in ARMS.items():
        for g, cs in GROUPS.items():
            m = np.isin(cells, cs)
            for k, n in enumerate(names):
                e, o = err(cue(n), p[k])[m], err(cue(n), p[1 - k])[m]
                print(f"| {arm} | {g} | {'first' if k == 0 else 'second'} | {np.median(e):.2f} | {np.mean(e < 1.5):.2f} | {np.mean(e < 3):.2f} | "
                      f"{np.mean(e < 5):.2f} | {np.mean((e >= 5) & (o < 5)):.2f} | {np.mean((e >= 5) & (o >= 5)):.2f} |")

    print("\n## 2. With 5 cm in place of 1.5 cm\n")
    print("| design | demos | both at their own glass | both at the same glass | swapped |\n|---|---|---|---|---|")
    for arm, (a, b) in ARMS.items():
        e = {(i, j): err(cue(n), p[j]) < 5 for i, n in enumerate((a, b)) for j in (0, 1)}      # cue i at point j
        for g, cs in GROUPS.items():
            m = np.isin(cells, cs)
            own, same = e[0, 0] & e[1, 1], (e[0, 0] & e[1, 0] & ~e[1, 1]) | (e[0, 1] & e[1, 1] & ~e[0, 0])
            swap = e[0, 1] & e[1, 0] & ~e[0, 0] & ~e[1, 1]
            print(f"| {arm} | {g} | {own[m].mean():.2f} | {same[m].mean():.2f} | {swap[m].mean():.2f} |")

    cc = load_config(manifest()["levels"][level]).cells
    first_is_obj = z["meta/ep_plan_first"][:] == z["meta/ep_object"][:]     # the teacher takes the cell's first glass first
    print(f"\n## 3. By the state of the glass the cue points at (the teacher takes the cell's first glass first in "
          f"{first_is_obj.mean():.2f} of the demos)\n")
    rows = {}
    for arm, names in ARMS.items():
        for k, n in enumerate(names):
            idx = np.where(first_is_obj, k, 1 - k)
            e = err(cue(n), p[k])
            for c, i, x in zip(cells, idx, e):
                rows.setdefault(state(cc[c][1 + 3 * i:4 + 3 * i]), {}).setdefault(arm, []).append(x)
    print("| the glass is shown by | cues | " + " | ".join(f"{a}: within 1.5 cm / within 3 cm / median" for a in ARMS) + " |\n|---|---|" + "---|" * len(ARMS))
    for lab, d in sorted(rows.items(), key=lambda kv: -len(next(iter(kv[1].values())))):
        v = {a: np.array(x) for a, x in d.items()}
        print(f"| {lab} | {len(next(iter(v.values())))} | " + " | ".join(
            f"{np.mean(v[a] < 1.5):.2f} / {np.mean(v[a] < 3):.2f} / {np.median(v[a]):.2f}" for a in ARMS) + " |")

    table_z = H.camera_params(level)["table_z"]
    q = [nth_goal(demos, "release", table_z, k) for k in (0, 1)]
    print("\n## 4. The release cues (5 cm)\n\n| demos | first: at its own coaster / at the other | second: at its own / at the other | both at their own | both at the same coaster |\n|---|---|---|---|---|")
    r = {(i, j): err(cue(f"r{i + 1}"), q[j]) < 5 for i in (0, 1) for j in (0, 1)}
    for g, cs in GROUPS.items():
        m = np.isin(cells, cs)
        same = (r[0, 0] & r[1, 0] & ~r[1, 1]) | (r[0, 1] & r[1, 1] & ~r[0, 0])
        print(f"| {g} | {r[0, 0][m].mean():.2f} / {(r[0, 1] & ~r[0, 0])[m].mean():.2f} | {r[1, 1][m].mean():.2f} / "
              f"{(r[1, 0] & ~r[1, 1])[m].mean():.2f} | {(r[0, 0] & r[1, 1])[m].mean():.2f} | {same[m].mean():.2f} |")


if __name__ == "__main__":
    main()
