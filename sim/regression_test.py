"""
Checks that the default config still reproduces main: same seed -> bit-identical depth, thermal, joints and teacher
actions. RGB may differ by 1 on a few pixels: EGL's alpha blending is not reproducible even between two runs of main.

  python sim/regression_test.py --save data/regress_main.npz    # on main (or before a change)
  python sim/regression_test.py --check data/regress_main.npz   # after the change
"""
import argparse

import numpy as np

from env import GlassPickEnv, OBJECT_NAMES
from teacher import ScriptedTeacher

KEYS = ["rgb", "depth", "thermal", "joints"]

def record(n_reset=20, n_episodes=5):
    env, teacher, out = GlassPickEnv(), ScriptedTeacher(), {}
    # First frames: random object (collect_demos) and object fixed by the seed (eval env)
    for seed in range(n_reset):
        for tag, obj in [("rand", None), ("fixed", OBJECT_NAMES[seed % len(OBJECT_NAMES)])]:
            obs = env.reset(seed, obj=obj)
            for k in KEYS:
                out[f"reset_{tag}_{seed}_{k}"] = obs[k]
    # Whole teacher episodes, as collect_demos runs them
    for seed in range(n_episodes):
        obs = env.reset(seed)
        teacher.reset(env, np.random.default_rng(seed))
        ep = {k: [] for k in KEYS + ["action"]}
        while not teacher.done():
            action = teacher.act()
            for k in KEYS:
                ep[k].append(obs[k])
            ep["action"].append(action)
            obs, _ = env.step(action)
        for k, v in ep.items():
            out[f"ep_{seed}_{k}"] = np.stack(v)
        out[f"ep_{seed}_success"] = np.array(env.success())
    env.close()
    return out

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--save")
    g.add_argument("--check")
    args = p.parse_args()

    out = record()
    if args.save:
        np.savez_compressed(args.save, **out)
        print(f"saved {len(out)} arrays to {args.save}")
    else:
        ref = np.load(args.check)

        def same(k):
            if k not in out:
                return False
            if k.endswith("_rgb"):  # EGL alpha blending is not bit-reproducible run to run: allow rare off-by-one
                d = np.abs(ref[k].astype(int) - out[k].astype(int))
                return d.max() <= 1 and (d > 0).mean() < 1e-3
            return np.array_equal(ref[k], out[k])
        bad = [k for k in ref.files if not same(k)]
        for k in bad[:20]:
            diff = np.abs(ref[k].astype(float) - out[k].astype(float)).max() if k in out else "missing"
            print(f"MISMATCH {k}: max abs diff {diff}")
        print(f"{len(ref.files) - len(bad)}/{len(ref.files)} arrays identical")
        raise SystemExit(1 if bad else 0)
