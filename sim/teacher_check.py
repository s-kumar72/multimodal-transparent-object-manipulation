"""
Scripted-teacher success rate per difficulty level, on training seeds and on the fully held-out test set (g).
Prints the failures with a reason (target not lifted / distractor knocked over or moved).

run via python sim/teacher_check.py --n 100 --workers 4
"""
import argparse
import multiprocessing as mp
from collections import Counter

import numpy as np

from env import GlassPickEnv, OBJECT_NAMES
from teacher import ScriptedTeacher
from testsets import manifest, test_set

def run(job):
    level, split, seed, obj = job
    cfg = test_set(level, "g_heldout_all")[0] if split == "test" else None
    from config import load_config
    env = GlassPickEnv(config=cfg or load_config(manifest()["levels"][level]))
    teacher = ScriptedTeacher()
    env.reset(seed, obj=obj)
    start = {n: env.data.body(n).xpos.copy() for n in env.extra_z0}
    teacher.reset(env, np.random.default_rng(seed))
    while not teacher.done():
        env.step(teacher.act())
    moved = [n for n, p in start.items() if np.linalg.norm(env.data.body(n).xpos[:2] - p[:2]) > 0.02]
    ok = bool(env.success())
    env.close()
    return level, split, env.obj, ok, moved, seed

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=100, help="training seeds per level")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--levels", nargs="+", default=list(manifest()["levels"]))
    args = p.parse_args()

    jobs = []
    for level in args.levels:
        jobs += [(level, "train", s, None) for s in range(args.n)]
        _, seeds, objects = test_set(level, "g_heldout_all")
        jobs += [(level, "test", s, objects[s % len(objects)]) for s in seeds]
    with mp.get_context("fork").Pool(args.workers) as pool:
        results = pool.map(run, jobs, chunksize=4)

    for level in args.levels:
        for split in ("train", "test"):
            rs = [r for r in results if r[0] == level and r[1] == split]
            per_obj = Counter(r[2] for r in rs if r[3])
            tot = Counter(r[2] for r in rs)
            n_moved = sum(bool(r[4]) for r in rs)
            print(f"{level} {split:5s} teacher {sum(r[3] for r in rs)}/{len(rs)}  "
                  + " ".join(f"{o} {per_obj[o]}/{tot[o]}" for o in OBJECT_NAMES)
                  + f"  | episodes that moved a distractor > 2 cm: {n_moved}")
            for r in rs:
                if not r[3]:
                    print(f"   FAIL seed {r[5]} {r[2]}: target not lifted; moved {r[4]}")
