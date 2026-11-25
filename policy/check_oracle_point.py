"""
Is the grasp point that policy/oracle_cue_env.py reads from the simulator the hindsight label of the demos? For every
demo of a training set: the scene is made again from the demo's seed and object, and its true grasp point is compared
with the label of that demo (the tool position at the first close, data/hcue/<folder>_label/train_cue.npy).

  python policy/check_oracle_point.py results/data/sim_demos_L0_vis_700.zarr data/hcue/L0_vis_700_label/train_cue.npy
"""
import sys

import numpy as np
import zarr

sys.path.insert(0, "sim")
from env import OBJECT_NAMES, GlassPickEnv  # noqa: E402
from testsets import manifest  # noqa: E402

from policy.oracle_cue_env import true_grasp_point  # noqa: E402


def main(zarr_path, label_path, level="L0_vis"):
    z, label = zarr.open(zarr_path, "r"), np.load(label_path)
    seeds, objs = z["meta/ep_seed"][:], z["meta/ep_object"][:]
    assert len(seeds) == len(label)
    env = GlassPickEnv(config=manifest()["levels"][level])
    d = []
    for s, o, lab in zip(seeds, objs, label):
        env.reset(int(s))                                 # as sim/collect_demos.py: the env draws the object
        assert env.obj == OBJECT_NAMES[int(o)], (s, env.obj, OBJECT_NAMES[int(o)])
        d.append(true_grasp_point(env) - lab[:3])
    env.close()
    d = 100 * np.array(d)
    xy, dz = np.linalg.norm(d[:, :2], axis=1), d[:, 2]
    print(f"{len(d)} demos of {zarr_path}: true grasp point of the scene minus the hindsight label")
    print(f"  in the table plane: median {np.median(xy):.2f} cm, 95% {np.percentile(xy, 95):.2f} cm, max {xy.max():.2f} cm")
    print(f"  height: median {np.median(dz):+.2f} cm, 5% to 95% {np.percentile(dz, 5):+.2f} to {np.percentile(dz, 95):+.2f} cm")
    for o in np.unique(objs):
        m = objs == o
        print(f"  {OBJECT_NAMES[int(o)]}: {m.sum()} demos, plane median {np.median(xy[m]):.2f} cm, height median {np.median(dz[m]):+.2f} cm")


if __name__ == "__main__":
    main(*sys.argv[1:])
