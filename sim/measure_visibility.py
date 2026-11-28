"""
How does the simulator show the glass, in the quantities real_data/glass_visibility.py measures on real sensors? Per
episode of a level: the first frame at the sensors' own resolution (colour and depth 320x240, thermal 160x120), the
true glass mask, and per sensor the same numbers with the same functions:

  python sim/measure_visibility.py LEVEL [-n 220] [--out FILE.json]

  rgb      contrast between the glass and a ring around it, grey levels of 255
  depth    missing: share of the glass's pixels without a reading; behind: share with a reading more than 2 cm behind
           the glass's surface
  thermal  contrast between the glass and the ring in raw counts of the thermal model (fitted to the camera of
           TRansPose, sim/configs/thermal/calibrated_v1.yaml), divided by that camera's noise (0.89 counts) as the
           real measurement is, and the same in degC with the model's gain
Printed per visibility cell and sensor state (shows / blind / faint) of the level, with the cell's flags.
"""
import json
import sys
from pathlib import Path

import click
import mujoco
import numpy as np

sys.path.insert(0, "sim")
sys.path.insert(0, ".")
from env import GlassPickEnv  # noqa: E402
from testsets import manifest  # noqa: E402

from real_data.glass_visibility import BEHIND_MM, TRANSPOSE_NOISE, glass_regions, luminance, rgb_contrast  # noqa: E402

STATE = {0: "shows", 1: "blind", 2: "faint"}


def first_frame(env):
    """Colour, noisy depth (m), true depth (m), raw thermal counts and the glass mask of the current scene."""
    s = env.sensors
    rgb, depth, _, raw = s.render(env.data, env.temps, env.emis, env.t_env, np.random.default_rng(0), env.sensor_scene)
    r = s.r
    r.enable_depth_rendering()
    r.update_scene(env.data, camera=s.camera, scene_option=s.show_all)
    true = r.render().copy()
    r.disable_depth_rendering()
    r.enable_segmentation_rendering()
    r.update_scene(env.data, camera=s.camera, scene_option=s.show_all)
    seg = r.render()[..., 0].copy()
    r.disable_segmentation_rendering()
    glass = np.isin(seg, [g for g in s.transparent_geoms if env.model.geom_bodyid[g] == env.model.body(env.obj).id])
    return rgb, depth, true, raw, glass


def measure(env):
    rgb, depth, true, raw, glass = first_frame(env)
    regs = glass_regions(glass)
    if not regs:
        return None
    reg, ring = max(regs, key=lambda x: x[0].sum())
    d, t = depth[reg], true[reg]
    out = dict(rgb=rgb_contrast(luminance(rgb), reg, ring), missing=float((d == 0).mean()),
               behind=float(((d - t > BEHIND_MM / 1000) & (d > 0)).mean()), px=int(reg.sum()))
    f = glass.shape[0] // raw.shape[0]
    reg_t, ring_t = reg[::f, ::f], ring[::f, ::f]
    if reg_t.sum() >= 20 and ring_t.sum() >= 20:
        c = abs(float(np.median(raw[reg_t]) - np.median(raw[ring_t])))
        out.update(thermal_counts=c, thermal_std=c / TRANSPOSE_NOISE, thermal_degc=c / env.cfg.thermal.gain)
    return out


@click.command()
@click.argument("level")
@click.option("-n", default=220, show_default=True, help="episodes (consecutive seeds from 0)")
@click.option("--out", default=None)
def main(level, n, out):
    env = GlassPickEnv(config=manifest()["levels"][level])
    cells = env.cfg.cells
    rows = []
    for seed in range(n):
        env.reset(seed)
        m = measure(env)
        if m is not None:
            rows.append(dict(m, cell=int(env.episode_info["cell"]), obj=env.obj))
    env.close()
    q = lambda v: "p10 {:.2f}, median {:.2f}, p90 {:.2f}".format(*np.quantile(v, [0.1, 0.5, 0.9]))
    res = {}
    print(f"{level}: {len(rows)} episodes with the glass in view; glass region median {int(np.median([r['px'] for r in rows]))} px of 320x240")
    for name, key, col, keys in (("rgb", 1, "rgb", ("rgb",)), ("depth", 2, "missing", ("missing", "behind")),
                                 ("thermal", 3, "thermal_std", ("thermal_std", "thermal_degc"))):
        for st, label in STATE.items():
            sel = [r for r in rows if cells[r["cell"]][key] == st and col in r]
            if not sel:
                continue
            res[f"{name} {label}"] = {k: dict(zip(("p10", "p50", "p90"), map(float, np.quantile([r[k] for r in sel], [0.1, 0.5, 0.9]))))
                                      for k in keys}
            print(f"  {name} {label} ({len(sel)} episodes): " + "; ".join(f"{k}: {q([r[k] for r in sel])}" for k in keys))
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        json.dump(res, open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
