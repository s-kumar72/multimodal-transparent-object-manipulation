"""
Runs the scripted teacher without rendering (physics only, no GPU needed) and reports success rate per scene condition.

run via python sim/teacher_check.py --config randomized_v1 --split train --n 100 --workers 4
"""
import argparse
import multiprocessing as mp
from collections import defaultdict

import numpy as np

from env import GlassPickEnv
from teacher import ScriptedTeacher

def init_worker(config, split):
    global env, teacher
    env, teacher = GlassPickEnv(config=config, split=split, render=False), ScriptedTeacher()

def run_episode(seed):
    env.reset(seed)
    teacher.reset(env, np.random.default_rng(seed))
    while not teacher.done():
        env.step(teacher.act())
    c = env.cond
    return {"seed": seed, "success": env.success(), "background": c["background"], "lighting": c["lighting"],
            "n_distractors": len(c["distractors"]), "fill_level": c["fill_level"],
            "is_crossover": abs(env.temps["glass"] - env.t_env) < 2}

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=None, help="scene config name in sim/configs/scene/ or YAML path")
    p.add_argument("--split", default="train", choices=["train", "test"])
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=1)
    args = p.parse_args()

    seeds = range(args.seed, args.seed + args.n)
    with mp.get_context("fork").Pool(args.workers, init_worker, (args.config, args.split)) as pool:
        rows = pool.map(run_episode, seeds)

    ok = np.mean([r["success"] for r in rows])
    print(f"config {args.config} split {args.split}: teacher success {ok:.1%} ({len(rows)} episodes)")
    for key in ["background", "lighting", "n_distractors", "fill_level", "is_crossover"]:
        groups = defaultdict(list)
        for r in rows:
            groups[r[key]].append(r["success"])
        print(f" {key}: " + ", ".join(f"{k}={np.mean(v):.0%} (n={len(v)})" for k, v in sorted(groups.items(), key=lambda kv: str(kv[0]))))
    failed = [r["seed"] for r in rows if not r["success"]]
    print(" failed seeds:", failed)
