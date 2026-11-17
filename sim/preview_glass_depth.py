"""
Depth on the glass, first frame, for levels with blob depth defects (e.g. L2g, L2gs): shows whether an episode is
see-through (no holes or wrong depth on the glass) or has the structured defects.
Per row: RGB 96x96, depth 96x96 (the policy input), and the 320x240 depth with the glass outlined in cyan, missing
pixels in red and wrong-depth pixels in yellow. Prints, per episode, the fraction of glass pixels missing / wrong
and of the other pixels missing.

run via python sim/preview_glass_depth.py --levels L2g L2gs --seeds 0 1 2 3 4 5 6 7 --out sim/out/glass_depth.png
"""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from config import load_config
from env import GlassPickEnv
from testsets import manifest

SCALE = 2

p = argparse.ArgumentParser()
p.add_argument("--levels", nargs="+", default=["L2g", "L2gs"])
p.add_argument("--seeds", nargs="+", type=int, default=list(range(8)))
p.add_argument("--out", default="sim/out/glass_depth.png")
args = p.parse_args()

def gray(depth):
    return np.repeat((255 * np.clip(depth, 0, 2.0) / 2.0).astype(np.uint8)[..., None], 3, -1)

tiles = []
for level in args.levels:
    env = GlassPickEnv(config=load_config(manifest()["levels"][level]))
    s, frames = env.sensors, []
    render = s.render
    s.render = lambda *a, **k: frames.append(render(*a, **k)) or frames[-1]   # keep the full-resolution frame
    for seed in args.seeds:
        frames.clear()
        obs = env.reset(seed)
        depth = frames[0][1]
        # Glass mask and the depth behind the glass, rendered again without random draws
        s.r.enable_segmentation_rendering()
        s.r.update_scene(env.data, camera=s.camera, scene_option=s.show_all)
        on_glass = np.isin(s.r.render()[..., 0], s.transparent_geoms)
        s.r.disable_segmentation_rendering()
        s.r.enable_depth_rendering()
        s.r.update_scene(env.data, camera=s.camera, scene_option=s.hide_transparent)
        behind = s.r.render().copy()
        s.r.disable_depth_rendering()
        missing = on_glass & (depth == 0)
        wrong = on_glass & (depth > 0) & (np.abs(depth - behind) > 1e-4)
        info = env.episode_info
        n = max(on_glass.sum(), 1)
        stats = (f"see-through {info['glass_seethrough']} | glass px {on_glass.sum()}: missing {missing.sum() / n:.2f}, "
                 f"wrong {wrong.sum() / n:.2f} | other px missing {((depth == 0) & ~on_glass).mean():.3f}")
        print(f"{level} seed {seed} {env.obj}: {stats}")

        full = gray(depth)
        full[wrong] = (255, 220, 0)
        full[missing] = (230, 0, 0)
        edge = on_glass & ~(np.roll(on_glass, 1, 0) & np.roll(on_glass, -1, 0) & np.roll(on_glass, 1, 1) & np.roll(on_glass, -1, 1))
        full[edge] = (0, 220, 255)
        small = [obs["rgb"], np.repeat(obs["depth"], 3, -1)]
        side = 96 * SCALE * 2
        row = Image.new("RGB", (2 * (side + 2) + 320 * 2, max(side, 480) + 18), (255, 255, 255))
        for i, im in enumerate(small):
            row.paste(Image.fromarray(im).resize((side, side), Image.NEAREST), (i * (side + 2), 18))
        row.paste(Image.fromarray(full).resize((640, 480), Image.NEAREST), (2 * (side + 2), 18))
        ImageDraw.Draw(row).text((4, 3), f"{level} seed {seed} | {env.obj} | {stats}", fill=(0, 0, 0))
        tiles.append(row)
    env.close()

w, h = tiles[0].size
grid = Image.new("RGB", (w, h * len(tiles)), (255, 255, 255))
for i, t in enumerate(tiles):
    grid.paste(t, (0, i * h))
Path(args.out).parent.mkdir(parents=True, exist_ok=True)
grid.save(args.out)
print("Saved", args.out)
