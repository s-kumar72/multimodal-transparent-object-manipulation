"""
The hindsight cue in the evaluation env, without a policy: the cue the env computes at reset against the position of
the glass, per cell. Checks the path the policy will be evaluated through (policy/glass_gym_env.py). A cue of the
place target (hindsight_cue.py --event release) is compared with the centre of the coaster instead.

  python policy/check_cue_env.py data/hcue/L0_vis_700/cue.pt [--level L0_vis] [--test_set j_cells11] [-n 660]
                                 [--save FILE.npz]

--save keeps the rows per episode (seed, cell, cue, position of the glass), to compare two cues on the same episodes.

The position of the glass and the cells are read from the simulator for this report only.
"""
import argparse

import numpy as np

from policy.glass_gym_env import GlassPickGymEnv
from policy.hindsight_cue import CUE_DIM


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--level", default="L0_vis")
    ap.add_argument("--test_set", default="j_cells11")
    ap.add_argument("-n", type=int, default=660)
    ap.add_argument("--start_seed", type=int, default=100000)
    ap.add_argument("--save", default=None)
    a = ap.parse_args()
    img = dict(shape=[3, 96, 96], type="rgb")
    meta = dict(rgb=img, depth=img, thermal=img, joints=dict(shape=[7], type="low_dim"),
                hcue=dict(shape=[CUE_DIM], type="low_dim", ckpt=a.ckpt))
    import torch
    release = torch.load(a.ckpt, map_location="cpu", weights_only=False)["cam"].get("event") == "release"
    env = GlassPickGymEnv(meta, level=a.level, test_set_name=a.test_set)
    env.set_split("test")
    rows = []
    for s in range(a.start_seed, a.start_seed + a.n):
        env.seed(s)
        o = env.reset()
        r = env.episode_record
        rows.append((r["cell"], *o["hcue"], *((*r["coaster_xy"], 0.0) if release else r["target_xyz"])))
    env.close()
    rows = np.array(rows, np.float64)
    cell, cue, tgt = rows[:, 0].astype(int), rows[:, 1:1 + CUE_DIM], rows[:, 1 + CUE_DIM:]
    if a.save:
        np.savez(a.save, seed=np.arange(a.start_seed, a.start_seed + a.n), cell=cell, cue=cue, target=tgt)
    err = 100 * np.linalg.norm(cue[:, :2] - tgt[:, :2], axis=1)
    print(f"{a.ckpt} on {a.test_set}, {a.n} episodes: the cue at reset against the position of "
          f"{'the coaster' if release else 'the glass'}")
    print("| cell | n | error, cm (median) | within 1 cm | within 1.5 cm | within 4 cm | gate rgb / depth / thermal (median) | "
          "sensors on (r >= 0.5), mean |")
    print("|---|---|---|---|---|---|---|---|")
    for c in [*sorted(set(cell.tolist())), None]:
        m = np.ones(len(cell), bool) if c is None else cell == c
        print(f"| {'all' if c is None else c} | {m.sum()} | {np.median(err[m]):.2f} | {(err[m] < 1).mean():.2f} | "
              f"{(err[m] < 1.5).mean():.2f} | {(err[m] < 4).mean():.2f} | "
              + " / ".join(f"{np.median(cue[m, 6 + i]):.2f}" for i in range(3))
              + f" | {(cue[m, 6:] >= 0.5).sum(1).mean():.2f} |")


if __name__ == "__main__":
    main()
