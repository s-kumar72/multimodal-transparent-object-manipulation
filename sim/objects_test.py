"""
Checks every transparent object: renders RGB/depth/thermal with the object on the table (one row per object),
verifies the parked objects are out of the camera's view, and runs the scripted teacher on a few seeds.

run via python sim/objects_test.py  ->  sim/out/objects.png
"""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from env import GlassPickEnv, OBJECT_NAMES
from teacher import ScriptedTeacher

p = argparse.ArgumentParser()
p.add_argument("--demos", type=int, default=3, help="teacher episodes per object")
args = p.parse_args()

env, teacher = GlassPickEnv(), ScriptedTeacher()
m, sens = env.model, env.sensors
rows = []
for i, name in enumerate(OBJECT_NAMES):
    env.reset(seed=i, obj=name)
    env.temps[name] = env.t_env + 20.0 # warm object so it shows in thermal
    rgb, depth, th = sens.render(env.data, env.temps, env.emis, env.t_env, np.random.default_rng(0))

    # Only the object on the table may be visible among the transparent bodies
    r = sens.r
    r.enable_segmentation_rendering()
    r.update_scene(env.data, camera=sens.camera, scene_option=sens.show_all)
    seg = r.render()[..., 0].copy()
    r.disable_segmentation_rendering()
    visible = {m.body(m.geom_bodyid[g]).name for g in np.unique(seg[seg >= 0])}
    parked = [n for n in OBJECT_NAMES if n != name and n in visible]
    assert not parked, f"parked objects visible while {name} is on the table: {parked}"
    px = int(np.isin(seg, np.flatnonzero(m.geom_bodyid == m.body(name).id)).sum())
    top = env.data.body(name).xpos[2] + max(m.geom_pos[g][2] + m.geom_size[g].max()
                                             for g in np.flatnonzero(m.geom_bodyid == m.body(name).id))

    # Teacher on a few seeds
    ok, steps = 0, []
    for s in range(args.demos):
        env.reset(seed=1000 * i + s, obj=name)
        teacher.reset(env, np.random.default_rng(s))
        while not teacher.done():
            env.step(teacher.act())
        ok += env.success()
        steps.append(env.steps)
    print(f"{name:10s} {px:5d} px in view, top {top:.3f} m, teacher {ok}/{args.demos} ({min(steps)}-{max(steps)} steps)")

    d = np.clip(depth, 0, 2.0) / 2.0
    tiles = [Image.fromarray(rgb),
             Image.fromarray((255 * d).astype(np.uint8)).convert("RGB"),
             Image.fromarray((255 * th).astype(np.uint8)).resize((320, 240), Image.NEAREST).convert("RGB")]
    row = Image.new("RGB", (960, 240))
    for j, t in enumerate(tiles):
        row.paste(t, (320 * j, 0))
    ImageDraw.Draw(row).text((8, 8), f"{name}  teacher {ok}/{args.demos}", fill=(255, 255, 0))
    rows.append(row)

Path("sim/out").mkdir(exist_ok=True)
grid = Image.new("RGB", (960, 240 * len(rows)))
for i, row in enumerate(rows):
    grid.paste(row, (0, 240 * i))
grid.save("sim/out/objects.png")
env.close()
print("Saved sim/out/objects.png")
