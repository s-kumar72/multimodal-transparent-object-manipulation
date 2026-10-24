"""
Runs scripted teacher & saves successful demos in Diffusion Policy's zarr format.

run via python sim/collect_demos.py --n 200 --out data/sim_demos.zarr with venv activated
add --workers 16 to run episodes in parallel processes (rendering is the bottleneck)
add --obj bottle to collect one object type only (default: a random object per episode)
Author: Sadhana Kumar
"""
import argparse
import itertools
import multiprocessing as mp
import sys
import time
from collections import Counter

import numpy as np
import zarr

from env import GlassPickEnv, OBJECT_NAMES
from teacher import ScriptedTeacher

KEYS = ["rgb", "depth", "thermal", "joints", "action"]

def init_worker(obj=None):
    global env, teacher, OBJ
    env, teacher, OBJ = GlassPickEnv(), ScriptedTeacher(), obj

def run_episode(seed):
    obs = env.reset(seed, obj=OBJ)
    teacher.reset(env, np.random.default_rng(seed))
    ep = {k: [] for k in KEYS}
    while not teacher.done():
        action = teacher.act()
        for k in KEYS[:-1]:
            ep[k].append(obs[k])
        ep["action"].append(action)
        obs, _ = env.step(action)

    if not env.success():
        return seed, env.obj, None
    return seed, env.obj, {k: np.stack(v) for k, v in ep.items()}

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=200, help="number of successful demos to save")
    p.add_argument("--out", default="data/sim_demos.zarr")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--obj", choices=OBJECT_NAMES, default=None, help="collect this object only")
    args = p.parse_args()

    # Results come back in seed order, so the saved dataset is the same for any number of workers.
    # macOS cannot fork a process that will open a GL context, so workers are spawned there.
    if args.workers > 1:
        pool = mp.get_context("fork" if sys.platform == "linux" else "spawn").Pool(args.workers, init_worker, (args.obj,))
        results = pool.imap(run_episode, range(args.seed, args.seed + 3 * args.n + 50))
    else:
        pool = None
        init_worker(args.obj)
        results = map(run_episode, itertools.count(args.seed))

    root = zarr.open(args.out, mode="w")
    data, meta = root.create_group("data"), root.create_group("meta")
    arrays, ends, objs, total, saved = {}, [], [], 0, 0
    tried, failed = Counter(), Counter()
    t0 = time.time()

    for seed, obj, ep in results:
        tried[obj] += 1
        if ep is None:
            failed[obj] += 1
            print(f"seed {seed} ({obj}): teacher failed, skipping")
            continue

        for k in KEYS:
            x = ep[k]
            if k not in arrays:
                chunks = (1,) + x.shape[1:] if x.ndim > 2 else (1024,) + x.shape[1:]
                arrays[k] = data.zeros(k, shape=(0,) + x.shape[1:], chunks=chunks, dtype=x.dtype)
            arrays[k].append(x)

        total += len(ep["action"])
        ends.append(total)
        objs.append(OBJECT_NAMES.index(obj))
        saved += 1
        print(f"demo {saved}/{args.n} ({obj}): {len(ep['action'])} steps ({time.time() - t0:.0f}s)", flush=True)
        if saved >= args.n:
            break

    if saved < args.n:
        print(f"only {saved}/{args.n} demos succeeded")
    meta.array("episode_ends", np.array(ends, dtype=np.int64))
    meta.array("object_id", np.array(objs, dtype=np.int64))
    meta.attrs["object_names"] = OBJECT_NAMES
    for obj in OBJECT_NAMES:
        if tried[obj]:
            print(f"teacher success {obj:10s} {tried[obj] - failed[obj]}/{tried[obj]}")
    if pool is not None:
        pool.terminate()
    else:
        env.close()
