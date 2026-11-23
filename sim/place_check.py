"""
Scripted-teacher check of a pick-and-place level (cfg.task "place"): success per object and visibility cell on
training seeds and on a test set, why the failures failed, and the statistics of the place target. The steps are not
rendered (only the first frame of each episode is), so it is quick.

  python sim/place_check.py --level L0_vis_place --n 700 --test_set k_cells11_heldout --n_test 660 --workers 8
"""
import argparse
import multiprocessing as mp
from collections import Counter

import mujoco
import numpy as np

from config import load_config
from env import COASTER_H, OBJECT_NAMES, PLACE_UPRIGHT, PLACE_Z_TOL, GlassPickEnv
from teacher import ScriptedTeacher
from testsets import manifest, test_set

ENVS = {}

def run(job):
    level, split, test_name, seed, obj = job
    if (level, split) not in ENVS:
        cfg = test_set(level, test_name)[0] if split == "test" else load_config(manifest()["levels"][level])
        ENVS[level, split] = (GlassPickEnv(config=cfg), ScriptedTeacher())
    env, teacher = ENVS[level, split]
    env.reset(seed, obj=obj)
    start = {n: env.data.body(n).xpos.copy() for n in env.extra_z0}
    teacher.reset(env, np.random.default_rng(seed))
    lift, first, t = 0.0, None, 0
    while not teacher.done():
        env.data.ctrl[:7] = teacher.act()
        for _ in range(env.n_sub):
            mujoco.mj_step(env.model, env.data)
        t += 1
        lift = max(lift, env.data.body(env.obj).xpos[2] - env.obj_z0)
        if first is None and env.success():
            first = t
    b = env.data.body(env.obj)
    ok = bool(env.success())
    if ok:
        why = "success"
    elif lift < 0.10:
        why = "not picked"
    elif b.xmat[8] <= PLACE_UPRIGHT:
        why = "fell over"
    elif abs(b.xpos[2] - (env.obj_z0 + COASTER_H)) >= PLACE_Z_TOL:
        why = "not on the coaster's top"
    elif env.place_offset() > env.cfg.place_radius - env.obj_r[env.obj]:
        why = "footprint not inside the coaster"
    else:
        why = "not released or still moving"
    moved = [n for n, p in start.items() if np.linalg.norm(env.data.body(n).xpos[:2] - p[:2]) > 0.02]
    info = env.episode_info
    return dict(split=split, seed=seed, obj=env.obj, cell=info.get("cell", -1), ok=ok, why=why, steps=t, first=first,
                offset=env.place_offset(), dist=info["place_dist"], clear=info["place_clear"], moved=moved,
                n_extra=len(start))

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--level", default="L0_vis_place")
    p.add_argument("--n", type=int, default=700, help="training seeds, from --seed")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--test_set", default="k_cells11_heldout")
    p.add_argument("--n_test", type=int, default=660)
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args()

    jobs = [(args.level, "train", args.test_set, s, None) for s in range(args.seed, args.seed + args.n)]
    _, seeds, objects = test_set(args.level, args.test_set, n=args.n_test)
    jobs += [(args.level, "test", args.test_set, s, objects[s % len(objects)]) for s in seeds]
    with mp.get_context("fork").Pool(args.workers) as pool:
        results = pool.map(run, jobs, chunksize=4)

    for split in ("train", "test"):
        rs = [r for r in results if r["split"] == split]
        if not rs:
            continue
        name = f"{args.level} {'training seeds' if split == 'train' else args.test_set}"
        print(f"\n## {name}: teacher {sum(r['ok'] for r in rs)}/{len(rs)} = {np.mean([r['ok'] for r in rs]):.3f}")
        print("per object: " + ", ".join(
            f"{o} {sum(r['ok'] for r in rs if r['obj'] == o)}/{sum(r['obj'] == o for r in rs)}" for o in OBJECT_NAMES))
        cells = sorted({r["cell"] for r in rs})
        print("per cell: " + ", ".join(
            f"{c}: {sum(r['ok'] for r in rs if r['cell'] == c)}/{sum(r['cell'] == c for r in rs)}" for c in cells))
        print("outcomes: " + ", ".join(f"{k} {v}" for k, v in Counter(r["why"] for r in rs).most_common()))
        steps, first = [r["steps"] for r in rs], [r["first"] for r in rs if r["first"]]
        print(f"teacher steps: median {np.median(steps):.0f}, range {min(steps)}-{max(steps)}; first step with success: "
              f"median {np.median(first):.0f}, max {max(first)}")
        dist = [r["dist"] for r in rs]
        print(f"glass to coaster at the start (m): median {np.median(dist):.3f}, range {min(dist):.3f}-{max(dist):.3f}")
        off = [r["offset"] for r in rs if r["ok"]]
        print(f"final offset of the glass from the coaster's centre in the successes (cm): median "
              f"{100 * np.median(off):.2f}, max {100 * max(off):.2f}")
        print(f"straight path not clear of distractors: {sum(r['clear'] == 0 for r in rs)}; no free spot for the "
              f"coaster: {sum(r['clear'] == -1 for r in rs)}; episodes that moved a distractor > 2 cm: "
              f"{sum(bool(r['moved']) for r in rs)}")
        for r in rs:
            if not r["ok"]:
                print(f"   FAIL seed {r['seed']} {r['obj']} cell {r['cell']}: {r['why']}; offset "
                      f"{100 * r['offset']:.1f} cm; path clear {r['clear']}; moved {r['moved']}")
