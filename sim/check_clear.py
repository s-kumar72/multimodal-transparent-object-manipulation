"""
Check of the "clear" rgb-blind case (SceneConfig.rgb_blind, level L0_grid_clear) against L0_grid, before any training.
  1. In an rgb-blind episode the colour image is the image of the scene without the glass: the camera frame rendered
     with and without the transparent objects, first frame and every frame of a teacher episode.
  2. Nothing else changes: first-frame depth and thermal are those of L0_grid for the same seed (in a dark episode
     L0_grid draws the colour noise after the first frame is rendered, so later frames use other random numbers), and
     in the cells where rgb shows the glass the whole teacher episode is the same.

  python sim/check_clear.py [--n 250] [--episodes 3]
"""
import argparse

import numpy as np

from config import cell_pattern, load_config
from env import GlassPickEnv, OBJECT_NAMES
from teacher import ScriptedTeacher


def glass_diff(env):
    """Camera frame with and without the transparent objects: (pixels that differ, largest difference), full size."""
    s = env.sensors
    r, d = s.r, env.data
    r.update_scene(d, camera=s.camera, scene_option=s.show_all)
    a = r.render().astype(int)
    r.update_scene(d, camera=s.camera, scene_option=s.hide_transparent)
    b = r.render().astype(int)
    diff = np.abs(a - b).max(-1)
    return int((diff > 0).sum()), int(diff.max())


def episode(env, seed):
    obs, teacher, frames = env.reset(seed), ScriptedTeacher(), []
    teacher.reset(env, np.random.default_rng(seed))
    while not teacher.done():
        frames.append(dict(obs, glass=glass_diff(env)))
        obs, _ = env.step(teacher.act())
    return frames, env.success()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=250, help="training seeds 0..n-1 (first frames)")
    p.add_argument("--episodes", type=int, default=3, help="teacher episodes per cell")
    args = p.parse_args()
    grid, clear = GlassPickEnv(config="sim/configs/L0_grid.yaml"), GlassPickEnv(config="sim/configs/L0_grid_clear.yaml")
    cells = load_config("sim/configs/L0_grid_clear.yaml").cells
    pat = cell_pattern(cells, len(OBJECT_NAMES))

    print(f"== first frames, training seeds 0..{args.n - 1}")
    print("| cell | rgb blind | episodes | glass pixels in the colour frame (median; max) | largest grey-level difference "
          "| mean brightness | depth = L0_grid | thermal = L0_grid | rgb = L0_grid (within 1 level) |")
    print("|---|---|---|---|---|---|---|---|---|")
    rows = {}
    for seed in range(args.n):
        c = int(pat[seed % len(pat)])
        a, b = grid.reset(seed), clear.reset(seed)
        assert clear.episode_info["cell"] == c and clear.episode_info["rgb_dark"] == 0
        n, m = glass_diff(clear)
        rows.setdefault(c, []).append((n, m, b["rgb"].mean(), np.array_equal(a["depth"], b["depth"]),
                                       np.array_equal(a["thermal"], b["thermal"]) and np.array_equal(a["thermal_raw"], b["thermal_raw"]),
                                       np.abs(a["rgb"].astype(int) - b["rgb"].astype(int)).max() <= 1))
    for c in sorted(rows):
        r = np.array(rows[c], float)
        print(f"| {c} | {'yes' if cells[c][1] else 'no'} | {len(r)} | {np.median(r[:, 0]):.0f}; {r[:, 0].max():.0f} | {r[:, 1].max():.0f} "
              f"| {r[:, 2].mean():.0f} | {int(r[:, 3].sum())}/{len(r)} | {int(r[:, 4].sum())}/{len(r)} | {int(r[:, 5].sum())}/{len(r)} |")

    print(f"\n== teacher episodes, {args.episodes} per cell")
    print("| cell | rgb blind | episodes | teacher success | frames | glass pixels in the colour frame (max over frames) "
          "| largest grey-level difference | whole episode = L0_grid (rgb within 1 level) |")
    print("|---|---|---|---|---|---|---|---|")
    for c in range(len(cells)):
        seeds = [s for s in range(len(pat)) if pat[s] == c][:args.episodes]
        ok = nf = px = mx = same = 0
        for seed in seeds:
            fb, sb = episode(clear, seed)
            ok += sb; nf += len(fb)
            px, mx = max(px, max(f["glass"][0] for f in fb)), max(mx, max(f["glass"][1] for f in fb))
            if not cells[c][1]:
                fa, _ = episode(grid, seed)
                same += len(fa) == len(fb) and all(
                    np.abs(x["rgb"].astype(int) - y["rgb"].astype(int)).max() <= 1 and np.array_equal(x["joints"], y["joints"])
                    and all(np.array_equal(x[k], y[k]) for k in ("depth", "thermal", "thermal_raw")) for x, y in zip(fa, fb))
        print(f"| {c} | {'yes' if cells[c][1] else 'no'} | {len(seeds)} | {ok}/{len(seeds)} | {nf} | {px} | {mx} "
              f"| {'%d/%d' % (same, len(seeds)) if not cells[c][1] else '(dark in L0_grid)'} |")
