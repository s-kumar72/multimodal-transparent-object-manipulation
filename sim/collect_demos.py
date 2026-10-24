"""
Runs scripted teacher & saves successful demos in Diffusion Policy's zarr format.

run via python sim/collect_demos.py --n 200 --out data/sim_demos.zarr with venv activated
add --workers 16 to run episodes in parallel processes (rendering is the bottleneck)
add --config randomized_v1 --split train|test for scene randomization; per-episode conditions go to meta/ep_*
Author: Sadhana Kumar
"""
import argparse
import itertools
import multiprocessing as mp
import time

import numpy as np
import zarr

from config import load_scene_config, load_thermal_config
from dataset_meta import write_episode_meta
from env import GlassPickEnv
from teacher import ScriptedTeacher

KEYS = ["rgb", "depth", "thermal", "thermal_raw", "joints", "action"]
THERMAL_NORM = "minmax" # data/thermal is per-frame min-max for now (becomes a config choice with common/thermal.py)

def init_worker(config=None, split="train"):
    global env, teacher
    env, teacher = GlassPickEnv(config=config, split=split), ScriptedTeacher()

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
        return seed, None, None
    return seed, {k: np.stack(v) for k, v in ep.items()}, dict(env.episode_meta(), thermal_norm=THERMAL_NORM)

def collect(out, cfg, split, seeds, n, workers=1):
    """Runs the teacher on seeds (in order) until n demos succeed and saves them to out.
    Returns [(seed, success)] for every tried seed."""
    # Results come back in seed order, so the saved dataset is the same for any number of workers
    if workers > 1:
        pool = mp.get_context("fork").Pool(workers, init_worker, (cfg, split))
        results = pool.imap(run_episode, seeds)
    else:
        pool = None
        init_worker(cfg, split)
        results = map(run_episode, seeds)

    root = zarr.open(out, mode="w")
    data, meta = root.create_group("data"), root.create_group("meta")
    arrays, ends, episodes, tried, total, saved = {}, [], [], [], 0, 0
    t0 = time.time()

    for seed, ep, ep_meta in results:
        tried.append((seed, ep is not None))
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
        episodes.append(ep_meta)
        saved += 1
        print(f"demo {saved}/{n}: {len(ep['action'])} steps ({time.time() - t0:.0f}s)", flush=True)
        if saved >= n:
            break

    if saved < n:
        print(f"only {saved}/{n} demos succeeded")
    meta.array("episode_ends", np.array(ends, dtype=np.int64))
    write_episode_meta(meta, episodes, cfg["distractors"]["max_count"],
                       {"scene_config": cfg, "thermal_config": load_thermal_config(cfg["thermal"]), "split": split})
    if pool is not None:
        pool.terminate()
    else:
        env.close()
    return tried

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=200, help="number of successful demos to save")
    p.add_argument("--out", default="data/sim_demos.zarr")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--config", default=None, help="scene config name in sim/configs/scene/ or YAML path (default: original scene)")
    p.add_argument("--split", default="train", choices=["train", "test"])
    args = p.parse_args()

    seeds = range(args.seed, args.seed + 3 * args.n + 50) if args.workers > 1 else itertools.count(args.seed)
    collect(args.out, load_scene_config(args.config), args.split, seeds, args.n, args.workers)
