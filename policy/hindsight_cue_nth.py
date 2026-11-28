"""
The hindsight cue of the n-th pick-and-place of a demo (task T3, two glasses: results/hard_scenario/bench_v2/README.md).

policy/hindsight_cue.py fits the cue of the point where the gripper first closes (--event grasp) or first opens again
(--event release). This runs the same program, with all its options, on the n-th such event of every demo:

  python policy/hindsight_cue_nth.py --nth 1 <demos> --level V2b_two --out data/hcue/V2b_two_700_g2 [--mode early]
  python policy/hindsight_cue_nth.py --nth 1 <demos> --level V2b_two --event release --out data/hcue/V2b_two_700_r2

--nth 0 is the first event: the labels are those of policy/hindsight_cue.py (checked with --check, which fits
nothing). Nothing of policy/hindsight_cue.py is changed: its `load` is replaced in this process only.

The per-sensor lines of the report ("own peak ... where it shows the glass / faintly / not") use the states of the
cell's first glass for --nth 0 and of its second for --nth 1. The teacher takes the glass nearer to the gripper first,
which is either of the two, so on T3 those lines are not the states of the glass the cue points at;
policy/readout_cue_two.py reports the cue by the states of its own glass.
"""
import sys

import numpy as np

import policy.hindsight_cue as H

_load = H.load


def nth_goal(path, event, table_z, nth):
    """The tool where the gripper closes for the (nth + 1)-th time (grasp), or the point of the table plane under the
    tool where it opens for the (nth + 1)-th time (release), of every demo."""
    import zarr
    z = zarr.open(str(path), "r")
    ends = z["meta/episode_ends"][:]
    act = z["data/action"][:]
    steps = []
    for s, t in zip(np.r_[0, ends[:-1]], ends):
        g = act[s:t, 6] > 0                                   # the gripper is commanded closed
        prev = np.r_[False, g[:-1]]
        on, off = np.flatnonzero(g & ~prev), np.flatnonzero(~g & prev)
        assert len(on) > nth and len(off) > nth, f"a demo with fewer than {nth + 1} grasps and releases"
        steps.append(s + int((on if event == "grasp" else off)[nth]))
    goal = H.fk(act[steps, :6])
    if event == "release":
        goal[:, 2] = table_z
    return goal


def main():
    argv = sys.argv[1:]
    assert "--nth" in argv, "usage: hindsight_cue_nth.py --nth N <the arguments of hindsight_cue.py>"
    i = argv.index("--nth")
    nth = int(argv[i + 1])
    check = "--check" in argv
    sys.argv = [sys.argv[0]] + [a for k, a in enumerate(argv) if k not in (i, i + 1) and a != "--check"]

    if check:        # the labels of --nth 0 against those of hindsight_cue.py; nothing is fitted
        import argparse
        ap = argparse.ArgumentParser()
        ap.add_argument("train"); ap.add_argument("--level", default="L0_vis"); ap.add_argument("--event", default="grasp")
        a, _ = ap.parse_known_args()
        cam = H.camera_params(a.level)
        ref = _load(a.train, 1, a.event, cam["table_z"])[1]
        mine = nth_goal(a.train, a.event, cam["table_z"], 0)
        print(f"{a.train}, event {a.event}: {len(ref)} demos, labels of --nth 0 equal to hindsight_cue.py: "
              f"{bool(np.array_equal(ref, mine))}")
        if nth:
            other = nth_goal(a.train, a.event, cam["table_z"], nth)
            d = 100 * np.linalg.norm(other[:, :2] - ref[:, :2], axis=1)
            print(f"--nth {nth} against --nth 0: distance on the table {d.min():.1f} to {d.max():.1f} cm (median {np.median(d):.1f})")
        return

    def load(path, n_frames, event="grasp", table_z=None):
        x, _, ep, cells = _load(path, n_frames, event, table_z)
        return x, nth_goal(path, event, table_z, nth), ep, cells

    H.load = load
    sys.path.insert(0, "sim")
    from config import load_config
    from testsets import manifest
    level = sys.argv[sys.argv.index("--level") + 1] if "--level" in sys.argv else "L0_vis"
    cells = load_config(manifest()["levels"][level]).cells or []
    if cells and len(cells[0]) == 7:          # T3: [weight, rgb, depth, thermal of the first glass, the same of the second]
        H.FLAGS = [tuple(c[1 + 3 * nth:4 + 3 * nth]) for c in cells]
    print(f"hindsight_cue_nth: event number {nth + 1} of every demo", flush=True)
    H.main()


if __name__ == "__main__":
    main()
