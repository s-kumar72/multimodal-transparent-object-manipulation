"""
Runs scripted teacher & saves successful demos in Diffusion Policy's zarr format.

run via python sim/collect_demos.py --n 200 --out data/sim_demos.zarr with venv activated
Author: Sadhana Kumar
"""
import argparse
import time

import numpy as np
import zarr

from env import GlassPickEnv
from teacher import ScriptedTeacher

KEYS = ["rgb", "depth", "thermal", "joints", "action"]

p = argparse.ArgumentParser()
p.add_argument("--n", type=int, default=200, helps="number of successful demos to save")
p.add_argument("--out", default="data/sim_demos.zarr")
p.add_argument("--seed", type=int, default=0)
args = p.parse_args()

env, teacher = GlassPickEnv(), ScriptedTeacher()
root = zarr.open(args.out, mode="w")
data, meta = root.create_group("data"), root.create_group("meta")
arrays, ends, total, saved, tried = {}, [], 0, 0, 0
t0 = time.time()

while saved < args.n:
    seed = args.seed + tried
    tried += 1
    obs = env.reset(seed)
    teacher.reset(env, np.random.default_rng(seed))
    ep = {k: [] for k in KEYS}
    while not teacher.done():
        action = teacher.act()
        for k in KEYS[:-1]:
            ep[k].append(obs[k])
        ep["action"].append(action)
        obs, _ = env.step(action)

    if not env.success():
        print(f"seed {seed}: teacher failed, skipping")
        continue

    for k in KEYS:
        x = np.stack(ep[k])
        if k not in arrays:
            chunks = (1,) + x.shape[1:] if x.ndim > 2 else (1024,) + x.shape[1:]
            arrays[k] = data.zeros(k, shape=(0,) + x.shape[1:], chunks=chunks, dtype=x.dtype)
        arrays[k].append(x)

    total += len(ep["action"])
    ends.append(total)
    saved += 1
    print(f"demo {saved}/{args.n}: {len(ep['action'])} steps ({time.time() - t0:.0f}s)")

meta.array("episode_ends", np.array(ends, dtype=np.int64))
env.close()