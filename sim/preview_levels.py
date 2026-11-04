"""
Preview of every difficulty level as the policy sees it (96x96 RGB / depth / thermal, first frame).
Rows: level x condition (two in-distribution seeds, held-out everything, liquid, thermal crossover).
Thermal is shown as stored in data/thermal (the level's normalization; "centered" for L1-L4).

run via python sim/preview_levels.py  ->  sim/out/levels_grid.png
"""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from env import GlassPickEnv, BACKGROUNDS, LIGHTING_NAMES
from testsets import manifest, test_set

ROWS = [("a_in_distribution", 0), ("a_in_distribution", 3), ("g_heldout_all", 1), ("e_liquid", 0), ("f_crossover", 2)]
SCALE = 3

p = argparse.ArgumentParser()
p.add_argument("--levels", nargs="+", default=list(manifest()["levels"]))
p.add_argument("--out", default="sim/out/levels_grid.png")
args = p.parse_args()

tiles = []
for level in args.levels:
    for name, k in ROWS:
        cfg, seeds, objects = test_set(level, name)
        env = GlassPickEnv(config=cfg)
        obs = env.reset(seeds[k], obj=objects[seeds[k] % len(objects)])
        info = env.episode_info
        env.close()
        imgs = [obs["rgb"], np.repeat(obs["depth"], 3, -1), np.repeat(obs["thermal"], 3, -1)]
        row = Image.new("RGB", (3 * 96 * SCALE + 4, 96 * SCALE + 18), (255, 255, 255))
        for i, im in enumerate(imgs):
            row.paste(Image.fromarray(im).resize((96 * SCALE, 96 * SCALE), Image.NEAREST), (i * (96 * SCALE + 2), 18))
        extras = info.get("n_distractors", 0), info.get("lookalike", 0), info.get("n_absorbers", 0)
        label = (f"{level} {name[:12]} | {env.obj} | {BACKGROUNDS[info['background']]}, {LIGHTING_NAMES[info['lighting']]}"
                 f" | distr/lookalike/absorb {extras} | fill {info['fill_level']:.2f} | gap {info['thermal_gap']:+.1f}C")
        ImageDraw.Draw(row).text((4, 3), label, fill=(0, 0, 0))
        tiles.append(row)
        print(label)

w, h = tiles[0].size
grid = Image.new("RGB", (w, h * len(tiles)), (255, 255, 255))
for i, t in enumerate(tiles):
    grid.paste(t, (0, i * h))
Path(args.out).parent.mkdir(parents=True, exist_ok=True)
grid.save(args.out)
print("Saved", args.out)
