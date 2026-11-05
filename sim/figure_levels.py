"""
Report figure of the difficulty levels: one row per level, all with the same seed (same target object, position and
temperatures; the distractor layout is redrawn when a level adds a look-alike or absorbers). Columns: RGB, depth and thermal at the sensors' own resolution
(cropped to the table), and the 96x96 RGB the policy actually gets. The target glass is circled.

run via python sim/figure_levels.py --seed 1  ->  results/hard_scenario/figure_levels.png (+ .pdf)
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Circle

from common.thermal import normalize_thermal_raw
from config import load_config
from env import BACKGROUNDS, IMG, LIGHTING_NAMES, GlassPickEnv
from probe import target_pixel
from testsets import manifest

LEVELS = {
    "L0": "main: one transparent object, plain table",
    "L1": "+ several objects, varied tables / light,\n   fainter tinted glass (lever A)",
    "L2": "+ opaque look-alike of the target (lever B)",
    "L3b": "+ depth defects on glass, black / metal\n   objects and table patches (lever C)",
    "L4b": "+ low light and RGB noise",
}
CROP = (55, 45, 265, 215)   # x0, y0, x1, y1 in the 320x240 renders: the table and the arm's workspace

def sensor_views(env):
    """RGB, depth and thermal at sensor resolution, thermal normalized as the level's data/thermal is."""
    th = env.cfg.thermal
    rgb, depth, thermal, raw = env.sensors.render(env.data, env.temps, env.emis, env.t_env,
                                                  np.random.default_rng(0), env.sensor_scene)
    if th.pipeline == "raw" and th.normalize != "legacy":
        params = {"range": tuple(th.norm_range)} if th.norm_range is not None else None
        thermal = normalize_thermal_raw(raw, th.normalize, params)
    return rgb, depth, thermal

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--levels", nargs="+", default=list(LEVELS))
    p.add_argument("--out", default="results/hard_scenario/figure_levels.png")
    args = p.parse_args()

    x0, y0, x1, y1 = CROP
    depth_cmap = plt.get_cmap("gray").copy()
    fig, axes = plt.subplots(len(args.levels), 4, figsize=(13, 2.75 * len(args.levels)),
                             gridspec_kw=dict(width_ratios=[1, 1, 1, 0.78], wspace=0.04, hspace=0.12))
    for r, level in enumerate(args.levels):
        env = GlassPickEnv(config=load_config(manifest()["levels"][level]))
        obs = env.reset(args.seed)
        info = env.episode_info
        rgb, depth, thermal = sensor_views(env)
        u, v = target_pixel(env)                           # in the 96x96 image
        su, sv = u * 320 / IMG - x0, v * 240 / IMG - y0    # in the cropped 320x240 renders
        tu, tv = (u * 160 / IMG - x0 / 2), (v * 120 / IMG - y0 / 2)
        env.close()

        ax = axes[r]
        ax[0].imshow(rgb[y0:y1, x0:x1])
        d = np.ma.masked_equal(depth[y0:y1, x0:x1], 0.0)    # no reading -> drawn in red
        ax[1].imshow(np.zeros_like(d.data), cmap=ListedColormap(["#d62728"]), vmin=0, vmax=1)
        ax[1].imshow(d, cmap=depth_cmap, vmin=0.6, vmax=1.6)
        ax[2].imshow(thermal[y0 // 2:y1 // 2, x0 // 2:x1 // 2], cmap="inferno", vmin=0, vmax=1)
        ax[3].imshow(obs["rgb"], interpolation="nearest")
        for a, (cu, cv, rad) in zip(ax, [(su, sv, 16), (su, sv, 16), (tu, tv, 8), (u, v, 6)]):
            a.add_patch(Circle((cu, cv), rad, fill=False, ec="#00e5ff", lw=1.3))
            a.set_xticks([])
            a.set_yticks([])
        extras = f"{info.get('n_distractors', 0)} distractors" + (", look-alike" if info.get("lookalike") else "") \
            + (f", {info['n_absorbers']} black/metal" if info.get("n_absorbers") else "")
        ax[0].set_ylabel(f"{level}\n{LEVELS.get(level, '')}", rotation=0, ha="right", va="center", fontsize=9.5,
                         labelpad=8)
        table = "plain" if BACKGROUNDS[info["background"]] == "xml" else BACKGROUNDS[info["background"]]
        ax[3].text(1.04, 0.5, f"{env.obj}\n{table} table\n"
                   f"{LIGHTING_NAMES[info['lighting']]} light\n{extras if 'n_distractors' in info else 'no distractors'}\n"
                   f"glass {info['thermal_gap']:+.0f} °C vs room", transform=ax[3].transAxes, fontsize=8,
                   va="center")
    titles = ["RGB\n(320×240 render, cropped)", "depth\n(320×240, cropped; red = no reading)",
              "thermal\n(160×120, cropped, as normalized)", "policy input\n(RGB 96×96)"]
    for a, t in zip(axes[0], titles):
        a.set_title(t, fontsize=9.5)
    fig.suptitle(f"Difficulty levels (seed {args.seed}: same target, position and temperatures in every row; "
                 f"distractors redrawn per level). Circle = target glass.", fontsize=10.5, y=0.94)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    fig.savefig(Path(args.out).with_suffix(".pdf"), bbox_inches="tight")
    print("Saved", args.out)

if __name__ == "__main__":
    main()
