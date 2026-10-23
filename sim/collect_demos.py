"""
Runs scripted teacher & saves successful demos in Diffusion Policy's zarr format.

run via python sim/collect_demos.py --n 200 --out data/sim_demos.zarr with venv activated
add --workers 16 to run episodes in parallel processes (rendering is the bottleneck)
Author: Sadhana Kumar
"""
import argparse
import itertools
import multiprocessing as mp
import time

import numpy as np
import zarr

from env import GlassPickEnv
from teacher import ScriptedTeacher

KEYS = ["rgb", "depth", "thermal", "joints", "action"]

def init_worker():
    global env, teacher
    env, teacher = GlassPickEnv(), ScriptedTeacher()

def run_episode(seed):
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
        return seed, None
    return seed, {k: np.stack(v) for k, v in ep.items()}

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=200, help="number of successful demos to save")
    p.add_argument("--out", default="data/sim_demos.zarr")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=1)
    args = p.parse_args()

    # Results come back in seed order, so the saved dataset is the same for any number of workers
    if args.workers > 1:
        pool = mp.get_context("fork").Pool(args.workers, init_worker)
        results = pool.imap(run_episode, range(args.seed, args.seed + 3 * args.n + 50))
    else:
        pool = None
        init_worker()
        results = map(run_episode, itertools.count(args.seed))

    root = zarr.open(args.out, mode="w")
    data, meta = root.create_group("data"), root.create_group("meta")
    arrays, ends, total, saved = {}, [], 0, 0
    t0 = time.time()

    for seed, ep in results:
        if ep is None:
            print(f"seed {seed}: teacher failed, skipping")
            continue

        for k in KEYS:
            x = ep[k]
            if k not in arrays:
                chunks = (1,) + x.shape[1:] if x.ndim > 2 else (1024,) + x.shape[1:]
                arrays[k] = data.zeros(k, shape=(0,) + x.shape[1:], chunks=chunks, dtype=x.dtype)
            arrays[k].append(x)

        total += len(ep["action"])
        ends.append(total)
        saved += 1
        print(f"demo {saved}/{args.n}: {len(ep['action'])} steps ({time.time() - t0:.0f}s)", flush=True)
        if saved >= args.n:
            break

    if saved < args.n:
        print(f"only {saved}/{args.n} demos succeeded")
    meta.array("episode_ends", np.array(ends, dtype=np.int64))
    if pool is not None:
        pool.terminate()
    else:
        env.close()
