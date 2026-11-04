"""
Compares the default camera with the optional narrow one (camera: front_cam_narrow): how much of the image the target
covers, how often it is out of view, and how often the policy's random 84x84 crop of the 96x96 image cuts its centre.

run via python sim/camera_check.py --level L3 --n 300
"""
import argparse

import numpy as np

from config import load_config
from env import GlassPickEnv, IMG
from probe import target_pixel
from testsets import manifest

CROP = 84

def crop_cut_prob(xy):
    """Probability over the uniform crop offsets (0..12 on each axis) that the point falls outside the crop."""
    offs = np.arange(IMG - CROP + 1)
    inside = [((offs <= c) & (c < offs + CROP)).mean() for c in xy]
    return 1 - inside[0] * inside[1]

p = argparse.ArgumentParser()
p.add_argument("--level", default="L3")
p.add_argument("--n", type=int, default=300)
args = p.parse_args()

for camera in ["front_cam", "front_cam_narrow"]:
    cfg = load_config(manifest()["levels"][args.level], camera=camera)
    env = GlassPickEnv(config=cfg)
    area, out, cut, extras_out = [], 0, [], 0
    for seed in range(args.n):
        env.reset(seed)
        r, s = env.sensors.r, env.sensors
        r.enable_segmentation_rendering()
        r.update_scene(env.data, camera=camera, scene_option=s.show_all)
        seg = r.render()[..., 0]
        r.disable_segmentation_rendering()
        body_of = env.model.geom_bodyid
        mask = np.isin(seg, np.flatnonzero(body_of == env.model.body(env.obj).id))
        area.append(mask.mean())
        xy = target_pixel(env)
        out += not mask.any()
        cut.append(crop_cut_prob(xy))
        visible = {env.model.body(body_of[g]).name for g in np.unique(seg[seg >= 0])}
        extras_out += sum(n not in visible for n in env.extra_z0)
    env.close()
    print(f"{camera:17s} target covers {100 * np.mean(area):.2f}% of the image (median {100 * np.median(area):.2f}%), "
          f"out of view {out}/{args.n}, P(84-crop cuts its centre) mean {np.mean(cut):.3f} max {np.max(cut):.3f}, "
          f"distractors out of view {extras_out}")
