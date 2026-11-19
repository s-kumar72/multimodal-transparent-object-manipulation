"""
Figure of the visibility cells of a level (sim/config.py cells): one row per cell, all with the same seed (same target
object, position and distractors), columns = the three images the policy gets (96x96 RGB, depth, thermal). The target
glass is circled; a column title in brackets means that sensor is blind in the cell.

run via python sim/figure_cells.py --seed 3  ->  results/hard_scenario/grid_bench/figure_cells.png
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Circle

from config import _load_yaml, make_config
from env import GlassPickEnv, OBJECT_NAMES
from probe import target_pixel
from testsets import manifest

NAMES = ("RGB", "depth", "thermal")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--level", default="L0_grid")
    p.add_argument("--seed", type=int, default=3)
    p.add_argument("--out", default="results/hard_scenario/grid_bench/figure_cells.png")
    args = p.parse_args()

    base = _load_yaml(manifest()["levels"][args.level])
    cells = base["cells"]
    fig, axes = plt.subplots(len(cells), 3, figsize=(6.6, 2.15 * len(cells)),
                             gridspec_kw=dict(wspace=0.04, hspace=0.16))
    for r, c in enumerate(cells):
        cfg = make_config(dict(base, cells=[[1, *c[1:4]]]), split="test", heldout=[])   # every seed gets this cell
        env = GlassPickEnv(config=cfg)
        obs = env.reset(args.seed, obj=OBJECT_NAMES[args.seed % len(OBJECT_NAMES)])
        info = env.episode_info
        u, v = target_pixel(env)
        env.close()
        ax = axes[r]
        ax[0].imshow(obs["rgb"], interpolation="nearest")
        d = np.ma.masked_equal(obs["depth"][..., 0], 0)     # no reading -> red
        ax[1].imshow(np.zeros(d.shape), cmap=ListedColormap(["#d62728"]), vmin=0, vmax=1, interpolation="nearest")
        ax[1].imshow(d, cmap="gray", vmin=70, vmax=200, interpolation="nearest")
        ax[2].imshow(obs["thermal"][..., 0], cmap="inferno", vmin=0, vmax=255, interpolation="nearest")
        shows = [n for n, blind in zip(NAMES, c[1:4]) if not blind]
        for a, n, blind in zip(ax, NAMES, c[1:4]):
            a.add_patch(Circle((u, v), 7, fill=False, ec="#00e5ff", lw=1.1))
            a.set_xticks([])
            a.set_yticks([])
            a.set_title(f"[{n}: blind]" if blind else n, fontsize=8.5, color="#b22222" if blind else "black", pad=3)
        ax[0].set_ylabel(f"cell {r}\nshown by\n" + ",\n".join(shows), rotation=0, ha="right", va="center",
                         fontsize=8.5, labelpad=6)
        ax[2].text(1.05, 0.5, f"glass {info['thermal_gap']:+.1f} °C vs room\n"
                   f"{info['thermal_contrast']:+.2f} °C vs table\n(apparent)", transform=ax[2].transAxes,
                   fontsize=7.5, va="center")
    fig.suptitle(f"{args.level}: the seven visibility cells (seed {args.seed}, {OBJECT_NAMES[args.seed % 5]}; policy "
                 "inputs, 96x96).\nCircle = target glass. Red in depth = no reading.", fontsize=9, y=0.925)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print("Saved", args.out)


if __name__ == "__main__":
    main()
