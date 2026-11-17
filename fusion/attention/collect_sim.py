"""
Glass segmentation frames from the MuJoCo sim: the policy's 96x96 RGB / depth / thermal observations plus the exact mask
of the target glass, along scripted-teacher episodes (so the arm is in view and moves, as in the policy's data).
Does not change sim/: the mask comes from one extra segmentation render with the env's own renderer.

Works with main's scene (--levels main) and with the sim/hard-scenario levels (L0 L1 L2 L3 L3b L4 L4b ...).
fusion/train.sh runs this script automatically. To run it manually, the working directory must be the top folder of
the repo copy that contains the wanted sim/ (for the hard levels: data/worktrees/sim_hard, created by train.sh),
because sim/ loads its scene files by relative path. PYTHONPATH must include that folder, its sim/ and this repo.

  python -m fusion.collect_sim --levels L1 L2 L3b L4b --n_train 400 --out data/fusion/sim
Writes <out>/<level>_<split>.npz, split in train | val | test_<test set>:
    rgb (N,96,96,3) u8 · depth (N,96,96,1) u8 (0 = no reading) · thermal (N,96,96,1) u8 (as the policy sees it)
    thermal_raw (N,96,96,1) u16 (hard-scenario only) · mask (N,96,96) u8 (fraction of the pixel covered by the target
    glass x 255) · episode, seed, step, object, is_crossover, thermal_gap, lookalike (N,)
"""
import argparse
import multiprocessing as mp
import os
import time
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

VAL_SEED0 = 50_000   # validation episodes use training scenes from seeds no training episode uses


def target_mask(env, size=96):
    """Fraction of every policy pixel covered by the target object's geoms (glass + any liquid in it)."""
    s = env.sensors
    r = s.r
    r.enable_segmentation_rendering()
    r.update_scene(env.data, camera=s.camera, scene_option=s.show_all)
    seg = r.render().copy()
    r.disable_segmentation_rendering()
    geoms = np.flatnonzero(env.model.geom_bodyid == env.model.body(env.obj).id)
    hit = np.isin(seg[..., 0], geoms) & (seg[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM))
    small = Image.fromarray(hit.astype(np.float32), mode="F").resize((size, size), Image.BOX)
    return np.clip(np.round(255 * np.asarray(small)), 0, 255).astype(np.uint8)


def _make_env(level, split, test_set):
    """(env, objects or None). level 'main' = the env on this checkout without a scene config."""
    from env import GlassPickEnv
    if level == "main":
        return GlassPickEnv(), None
    from config import load_config
    from testsets import manifest
    if split == "test":
        from testsets import test_set as get_test_set
        cfg, _, objects = get_test_set(level, test_set)
        return GlassPickEnv(config=cfg), objects
    return GlassPickEnv(config=load_config(manifest()["levels"][level]), split="train"), None


def _run(job):
    level, split, test_set, seeds, stride = job
    from teacher import ScriptedTeacher
    from env import OBJECT_NAMES
    env, objects = _make_env(level, split, test_set)
    teacher = ScriptedTeacher()
    out = {k: [] for k in ["rgb", "depth", "thermal", "thermal_raw", "mask", "seed", "step", "object",
                           "is_crossover", "thermal_gap", "lookalike"]}
    for seed in seeds:
        obs = env.reset(seed, obj=objects[seed % len(objects)] if objects else None)
        teacher.reset(env, np.random.default_rng(seed))
        info = getattr(env, "episode_info", {})
        gap = info.get("thermal_gap", env.temps[env.obj] - env.t_env)
        step = 0
        while True:
            if step % stride == 0:
                for k in ["rgb", "depth", "thermal"]:
                    out[k].append(obs[k])
                if "thermal_raw" in obs:
                    out["thermal_raw"].append(obs["thermal_raw"])
                out["mask"].append(target_mask(env))
                out["seed"].append(seed)
                out["step"].append(step)
                out["object"].append(OBJECT_NAMES.index(env.obj))
                out["is_crossover"].append(int(info.get("is_crossover", abs(gap) <= 1.0)))
                out["thermal_gap"].append(float(gap))
                out["lookalike"].append(int(any(n.startswith("la_") for n in getattr(env, "extra_z0", {}))))
            if teacher.done():
                break
            obs, _ = env.step(teacher.act())
            step += 1
    env.close()
    return {k: np.stack(v) for k, v in out.items() if v}


def collect(level, split, seeds, stride, workers, test_set=None):
    chunks = [seeds[i::workers * 4] for i in range(workers * 4)]
    jobs = [(level, split, test_set, c, stride) for c in chunks if c]
    ctx = mp.get_context("fork" if os.name == "posix" and os.uname().sysname == "Linux" else "spawn")
    with ctx.Pool(workers) as pool:
        parts = pool.map(_run, jobs)
    data = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    order = np.lexsort((data["step"], data["seed"]))
    data = {k: v[order] for k, v in data.items()}
    data["episode"] = np.unique(data["seed"], return_inverse=True)[1]
    return data


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--levels", nargs="+", default=["L1", "L2", "L3b", "L4b"])
    p.add_argument("--n_train", type=int, default=400, help="training episodes per level")
    p.add_argument("--n_val", type=int, default=40, help="validation episodes per level (training scenes)")
    p.add_argument("--n_test", type=int, default=50, help="episodes per test set (seeds 100000+, as the policy eval)")
    p.add_argument("--test_sets", nargs="*", default=["a_in_distribution", "f_crossover", "g_heldout_all"])
    p.add_argument("--stride", type=int, default=6, help="keep every stride-th step (episodes are ~60-70 steps)")
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    p.add_argument("--out", default="data/fusion/sim")
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for level in a.levels:
        jobs = [("train", list(range(a.n_train)), None), ("val", list(range(VAL_SEED0, VAL_SEED0 + a.n_val)), None)]
        if level == "main":
            jobs.append(("test", list(range(100_000, 100_000 + a.n_test)), "main"))
        else:
            jobs += [("test", list(range(100_000, 100_000 + a.n_test)), t) for t in a.test_sets]
        for split, seeds, ts in jobs:
            name = f"{level}_{split}" if split != "test" else f"{level}_test_{ts}"
            path = out / f"{name}.npz"
            if path.exists():
                print(f"have {path}")
                continue
            if not seeds:
                continue
            t0 = time.time()
            data = collect(level, "train" if split == "val" else split, seeds, a.stride, a.workers, ts)
            np.savez_compressed(path, level=level, **data)
            fg = (data["mask"] > 127).mean(axis=(1, 2))
            print(f"{path}: {len(data['rgb'])} frames from {len(seeds)} episodes in {time.time() - t0:.0f}s, "
                  f"target covers {100 * fg.mean():.2f}% of the image (empty in {100 * (fg == 0).mean():.0f}% "
                  f"of frames), crossover {100 * data['is_crossover'].mean():.0f}%", flush=True)


if __name__ == "__main__":
    main()