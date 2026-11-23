"""
Figure of a pick-and-place level (cfg.task "place"): one row per episode of the scripted teacher, columns = the colour
image at the start, while the glass is carried and at the end, then depth and thermal at the start (the policy's
96x96 inputs). The glass is circled at the start; the coaster is marked with a square.

run via python sim/figure_place.py  ->  results/hard_scenario/place_bench/figure_place.png
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Circle, Rectangle

from config import load_config
from env import IMG, GlassPickEnv
from probe import target_pixel
from teacher import ScriptedTeacher
from testsets import manifest

STATE = ("shows", "blind", "faint")


def pixel(env, p):
    """Image position (96x96 pixels) of a world point."""
    s, d = env.sensors, env.data
    pc = d.cam_xmat[s.cam_id].reshape(3, 3).T @ (p - d.cam_xpos[s.cam_id])
    h, w = s.r.height, s.r.width
    return (w / 2 + s.focal * pc[0] / -pc[2]) * IMG / w, (h / 2 - s.focal * pc[1] / -pc[2]) * IMG / h


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--level", default="L0_vis_place")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 22, 33, 49])
    p.add_argument("--out", default="results/hard_scenario/place_bench/figure_place.png")
    args = p.parse_args()

    cfg = load_config(manifest()["levels"][args.level])
    env, teacher = GlassPickEnv(config=cfg), ScriptedTeacher()
    fig, axes = plt.subplots(len(args.seeds), 5, figsize=(11, 2.25 * len(args.seeds)),
                             gridspec_kw=dict(wspace=0.04, hspace=0.2))
    for ax, seed in zip(axes, args.seeds):
        obs = env.reset(seed)
        info = env.episode_info
        glass, coaster = target_pixel(env), pixel(env, env.place_position())
        teacher.reset(env, np.random.default_rng(seed))
        n, frames = len(teacher.plan), [obs]
        carry = n - 45                       # the glass is in the air between the two spots
        while not teacher.done():
            obs, _ = env.step(teacher.act())
            if teacher.t in (carry, n):
                frames.append(obs)
        ok = env.success()
        cell = cfg.cells[info["cell"]]
        for a, f, title in zip(ax[:3], frames, ("colour, start", f"colour, step {carry}", f"colour, step {n} (end)")):
            a.imshow(f["rgb"], interpolation="nearest")
            a.set_title(title, fontsize=8.5, pad=3)
        d = np.ma.masked_equal(frames[0]["depth"][..., 0], 0)     # no reading -> red
        ax[3].imshow(np.zeros(d.shape), cmap=ListedColormap(["#d62728"]), vmin=0, vmax=1, interpolation="nearest")
        ax[3].imshow(d, cmap="gray", vmin=70, vmax=200, interpolation="nearest")
        ax[3].set_title("depth, start", fontsize=8.5, pad=3)
        ax[4].imshow(frames[0]["thermal"][..., 0], cmap="inferno", vmin=0, vmax=255, interpolation="nearest")
        ax[4].set_title("thermal, start", fontsize=8.5, pad=3)
        for a in (ax[0], ax[3], ax[4]):
            a.add_patch(Circle(glass, 7, fill=False, ec="#00e5ff", lw=1.1))
            a.add_patch(Rectangle((coaster[0] - 8, coaster[1] - 5), 16, 10, fill=False, ec="#7CFC00", lw=1.0))
        for a in ax:
            a.set_xticks([])
            a.set_yticks([])
        ax[0].set_ylabel(f"seed {seed}, {env.obj}\ncell {info['cell']}: rgb {STATE[cell[1]]},\ndepth {STATE[cell[2]]}, "
                         f"thermal {STATE[cell[3]]}\nglass to coaster {100 * info['place_dist']:.0f} cm\n"
                         f"teacher: {'placed' if ok else 'FAILED'}", rotation=0, ha="right", va="center", fontsize=8,
                         labelpad=6)
    env.close()
    fig.suptitle(f"{args.level}: episodes of the scripted teacher (policy inputs, 96x96). Circle = the glass at the "
                 "start, square = the coaster. Red in depth = no reading.", fontsize=9, y=0.93)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print("Saved", args.out)


if __name__ == "__main__":
    main()
