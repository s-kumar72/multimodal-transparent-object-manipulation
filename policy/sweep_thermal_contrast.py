"""
How much thermal contrast may the glass keep and still be invisible to the thermal camera? Used once, to fix the
definition of "thermal blind" of the visibility cells (sim/config.py cells) before any level with cells is built.

  python policy/sweep_thermal_contrast.py SEG_CKPT [--level L0_bench] [--n 150] [-d cuda:0] [--out FILE.json]

For the first frame of N test episodes (seeds 100000+, before the arm moves, so the arm gives no hint of where the
glass is) the glass temperature is set by hand and the frame is rendered again, for two definitions of the contrast:

  ambient  the glass is c degC from the room temperature (how thermal.crossover_gap defines a crossover episode);
           the table keeps its own temperature (room + table_dt), so the glass can still stand out against it;
  table    the apparent temperature of the glass is c degC from that of the table behind it. Apparent = what the
           camera measures: emissivity * emitted + (1 - emissivity) * reflected room radiance.

The sign of c alternates with the seed. Reported per (definition, c):
  located   share of frames where the segmentation network, given the thermal image only, puts the glass within 4 px;
  chance    the same share when each prediction is compared with the glass of another episode (500 shuffles): the
            glass spawns in a small part of the image, so a guess that ignores the frame is often within 4 px;
  cue       the thermal step of the visibility gate's hand rule at the true mask (policy/obs_util.py visibility_cues:
            |median on the glass - median of a ring around it| / camera noise; the rule calls > 30 visible).
It also prints the counts per degC, and how much of the glass has the table behind it in the image.
"""
import json
import sys

import click
import numpy as np

sys.path.insert(0, "sim")
from sensors import band_radiance, matching_temp  # noqa: E402
from testsets import test_set  # noqa: E402
from env import GlassPickEnv  # noqa: E402

from fusion.collect_sim import target_mask  # noqa: E402
from policy.check_cells import located_and_chance  # noqa: E402
from policy.glass_seg import predict_glass_mask  # noqa: E402
from policy.obs_util import THERMAL_NOISE_COUNTS, visibility_cues  # noqa: E402

CONTRASTS = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0]


def glass_temp_for(env, c):
    """Glass temperature whose apparent temperature is c degC from the table's (same room reflection for both)."""
    return matching_temp(env.temps["table"], env.emis["table"], env.emis[env.obj], env.t_env, c)


def behind_table(env):
    """Share of the glass pixels that have the table behind them (the rest: floor, a distractor, the arm)."""
    s, r = env.sensors, env.sensors.r
    r.enable_segmentation_rendering()
    r.update_scene(env.data, camera=s.camera, scene_option=s.show_all)
    seg = r.render()[..., 0].copy()
    r.update_scene(env.data, camera=s.camera, scene_option=s.hide_transparent)
    behind = r.render()[..., 0].copy()
    r.disable_segmentation_rendering()
    on = np.isin(seg, np.flatnonzero(env.model.geom_bodyid == env.model.body(env.obj).id))
    return float(np.isin(behind[on], s.table_geoms).mean()) if on.any() else np.nan


@click.command()
@click.argument("ckpt")
@click.option("--level", default="L0_bench", show_default=True)
@click.option("--n", default=150, show_default=True)
@click.option("-d", "--device", default="cuda:0", show_default=True)
@click.option("--out", default=None)
def main(ckpt, level, n, device, out):
    cfg, seeds, objects = test_set(level, "a_in_distribution", n=n)
    env = GlassPickEnv(config=cfg)
    frames = {(m, c): {k: [] for k in ("rgb", "depth", "thermal", "thermal_raw")} for m in ("ambient", "table")
              for c in CONTRASTS}
    masks, table_share, app = [], [], {c: [] for c in CONTRASTS}
    per_deg = []
    for seed in seeds:
        env.reset(seed, obj=objects[seed % len(objects)])
        masks.append(target_mask(env) > 0)
        table_share.append(behind_table(env))
        k = band_radiance(env.t_env + 0.5) - band_radiance(env.t_env - 0.5)
        per_deg.append(cfg.thermal.gain * k)
        sign = 1 if seed % 2 == 0 else -1
        zero = glass_temp_for(env, 0.0)
        for (mode, c), f in frames.items():
            env.temps[env.obj] = env.t_env + sign * c if mode == "ambient" else glass_temp_for(env, sign * c)
            if mode == "ambient":   # apparent contrast against the table that this room-temperature offset leaves
                app[c].append(abs(env.emis[env.obj] * (env.temps[env.obj] - zero)))
            obs = env.get_obs()
            for key in f:
                f[key].append(obs[key])
    env.close()
    gt = np.stack(masks)
    H, W = gt.shape[-2:]
    print(f"{level}, first frame of {len(seeds)} episodes; {np.mean(per_deg):.2f} raw counts per degC at room "
          f"temperature (noise {THERMAL_NOISE_COUNTS} counts per pixel); glass mask {gt.sum((1, 2)).mean():.0f} px of "
          f"{H}x{W}; table behind {np.nanmean(table_share):.3f} of the glass pixels "
          f"(less than 0.9 in {np.mean(np.array(table_share) < 0.9):.2f} of the episodes)")
    res, rng = [], np.random.default_rng(0)
    for mode in ("table", "ambient"):
        for c in CONTRASTS:
            d = {k: np.stack(v) for k, v in frames[(mode, c)].items()}
            pred = predict_glass_mask(d, ckpt, device, mods=["thermal"])
            loc, chance = located_and_chance(pred, gt, rng)
            cue = visibility_cues(d["depth"], d["thermal_raw"].astype(np.float32), gt, THERMAL_NOISE_COUNTS)[:, 1]
            res.append(dict(mode=mode, c=c, located=loc, chance=chance, cue_median=float(np.median(cue)),
                            cue_gt30=float((cue > 30).mean()),
                            apparent_vs_table=float(np.mean(app[c])) if mode == "ambient" else c))
    print(f"{'definition':10s} {'c (degC)':>8s} | located | chance | cue median | cue > 30 | mean apparent contrast "
          "against the table (degC)")
    for r in res:
        print(f"{r['mode']:10s} {r['c']:8.2f} |  {r['located']:.3f}  | {r['chance']:.3f}  |   {r['cue_median']:6.2f}   |"
              f"  {r['cue_gt30']:.3f}   | {r['apparent_vs_table']:.3f}")
    if out:
        json.dump(dict(level=level, n=len(seeds), counts_per_degC=float(np.mean(per_deg)),
                       table_behind=float(np.nanmean(table_share)), rows=res), open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
