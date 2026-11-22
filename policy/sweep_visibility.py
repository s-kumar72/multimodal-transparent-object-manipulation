"""
How well does one sensor show the glass as its "strength" goes from blind to shows? Used to set the intermediate
("faint") level of each sensor of the visibility cells by measurement, as the thermal-blind limit was
(policy/sweep_thermal_contrast.py).

  python policy/sweep_visibility.py SEG_CKPT [SEG_CKPT ...] [--level L0_grid_clear] [--n 300] [-d cuda:0] [--out FILE.json]

For the first frame of N test episodes (seeds 100000+, before the arm moves) one sensor's view of the glass is set by
hand and the frame is rendered again:

  rgb      the alpha of the glass (0.001 = the clear glass of an rgb-blind cell, 0.3 = shows in L0_grid);
  depth    the share of the glass pixels that are holes (0 = see-through, the depth-blind case; 0.4 = shows);
  thermal  the apparent temperature of the glass against the table behind it, in degC, sign alternating with the seed
           (0.1 = the thermal-blind limit; 5-35 = shows).

Reported per value and segmentation network: the share of frames where the network, given that sensor's image only,
puts the glass within 4 px ("located"), and the same share against the glass of other episodes ("chance").
"half" is the value at which located is midway between chance and the located share at the strongest value
(linear interpolation between the two neighbouring values).
"""
import json
import sys

import click
import numpy as np

sys.path.insert(0, "sim")
from testsets import test_set  # noqa: E402
from env import GlassPickEnv  # noqa: E402

from fusion.collect_sim import target_mask  # noqa: E402
from policy.check_cells import located_and_chance  # noqa: E402
from policy.glass_seg import predict_glass_mask  # noqa: E402
from policy.sweep_thermal_contrast import glass_temp_for  # noqa: E402

VALUES = {"rgb": [0.001, 0.02, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5],
          "depth": [0.0, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4],
          "thermal": [0.1, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0, 20.0]}
KEYS = ("rgb", "depth", "thermal", "thermal_raw")


def set_strength(env, sensor, v, seed):
    if sensor == "rgb":
        for geoms in env.glass_geoms.values():
            env.model.geom_rgba[geoms, 3] = v
    elif sensor == "depth":
        env.sensors.cfg.glass_holes = v
        env.sensor_scene["seethrough"] = 0
    else:
        env.temps[env.obj] = glass_temp_for(env, v if seed % 2 == 0 else -v)


def half_point(values, located, chance):
    """Value at which located is midway between chance and the located share of the strongest value."""
    target = (np.mean(chance) + located[-1]) / 2
    for i in range(1, len(values)):
        if located[i - 1] < target <= located[i]:
            t = (target - located[i - 1]) / (located[i] - located[i - 1])
            return float(values[i - 1] + t * (values[i] - values[i - 1])), float(target)
    return None, float(target)


@click.command()
@click.argument("ckpts", nargs=-1, required=True)
@click.option("--level", default="L0_grid_clear", show_default=True)
@click.option("--test_set_name", default="h_cells", show_default=True)
@click.option("--n", default=300, show_default=True)
@click.option("-d", "--device", default="cuda:0", show_default=True)
@click.option("--out", default=None)
def main(ckpts, level, test_set_name, n, device, out):
    cfg, seeds, objects = test_set(level, test_set_name, n=n)
    env = GlassPickEnv(config=cfg)
    frames = {(s, v): {k: [] for k in KEYS} for s, vals in VALUES.items() for v in vals}
    masks = []
    holes0 = env.sensors.cfg.glass_holes
    for seed in seeds:
        env.reset(seed, obj=objects[seed % len(objects)])
        assert "glass" not in env.sensor_scene.get("defects", {}), "dot holes expected on the glass"
        masks.append(target_mask(env) > 0)
        temp0, see0 = env.temps[env.obj], env.sensor_scene["seethrough"]
        alpha0 = {k: env.model.geom_rgba[g, 3].copy() for k, g in env.glass_geoms.items()}
        for (sensor, v), f in frames.items():
            set_strength(env, sensor, v, seed)
            obs = env.get_obs()
            for key in KEYS:
                f[key].append(obs[key])
            env.temps[env.obj], env.sensor_scene["seethrough"], env.sensors.cfg.glass_holes = temp0, see0, holes0
            for k, g in env.glass_geoms.items():
                env.model.geom_rgba[g, 3] = alpha0[k]
    env.close()
    gt = np.stack(masks)
    print(f"{level} ({test_set_name}), first frame of {len(seeds)} episodes; glass mask {gt.sum((1, 2)).mean():.0f} px "
          f"of {gt.shape[-2]}x{gt.shape[-1]}")
    res = []
    for sensor, vals in VALUES.items():
        print(f"\n== {sensor}")
        print("value | " + " | ".join(f"located / chance ({c})" for c in ckpts))
        cols = []
        for ckpt in ckpts:
            rng, loc, ch = np.random.default_rng(0), [], []
            for v in vals:
                d = {k: np.stack(x) for k, x in frames[(sensor, v)].items()}
                a, b = located_and_chance(predict_glass_mask(d, ckpt, device, mods=[sensor]), gt, rng)
                loc.append(a); ch.append(b)
            cols.append((loc, ch))
        for i, v in enumerate(vals):
            print(f"{v:6.3f} | " + " | ".join(f"{loc[i]:.3f} / {ch[i]:.3f}" for loc, ch in cols))
        for ckpt, (loc, ch) in zip(ckpts, cols):
            h, target = half_point(vals, loc, ch)
            print(f"half ({ckpt}): located {target:.3f} at " + ("no crossing" if h is None else f"{h:.3f}"))
            res.append(dict(sensor=sensor, ckpt=ckpt, values=vals, located=loc, chance=ch, half=h, half_located=target))
    if out:
        json.dump(dict(level=level, test_set=test_set_name, n=len(seeds), rows=res), open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
