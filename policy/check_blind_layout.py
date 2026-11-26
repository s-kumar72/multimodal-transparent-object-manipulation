"""
What a sensor that does not show the glass still tells about where it is (results/hard_scenario/bench_v2/README.md,
"What a blind sensor still shows"). Three checks, none of them trains anything:

  python policy/check_blind_layout.py peaks  <demos.zarr> <cue folder with maps.pt> --level V2_align
  python policy/check_blind_layout.py frames --level V2_align -n 330
  python policy/check_blind_layout.py layout --level V2_align -n 700

peaks   the out-of-fold maps of the cue networks (policy/hindsight_cue.py) on the first frame of every demo: how often
        the peak of a sensor that is blind in the cell is on the grasp pixel, against the same peaks paired with the
        grasp pixels of other demos (what the distribution of the grasp pixels alone gives).
frames  the 96 x 96 frame of a colour-blind or depth-see-through scene with the glass and with the glass taken out of
        the scene (10 m away), same noise stream: do they differ?
layout  no image at all: the spawn points at which the glass could be, given that the generator keeps the distractors
        GAP_TARGET away from it and the place target clear of it; the centroid of that set as a guess.
"""
import sys

import click
import numpy as np

sys.path.insert(0, "sim")
sys.path.insert(0, "policy")

FLAGS = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (1, 0, 1), (0, 1, 1), (2, 1, 1), (1, 2, 1), (1, 1, 2), (2, 2, 2)]


@click.group()
def cli():
    pass


@cli.command()
@click.argument("zarr_path")
@click.argument("cue_dir")
@click.option("--level", required=True)
def peaks(zarr_path, cue_dir, level):
    import torch
    import zarr
    argv, sys.argv = sys.argv, sys.argv[:1]
    import hindsight_cue as H
    sys.argv = argv
    cam = dict(H.camera_params(level), event="grasp")
    z = zarr.open(zarr_path, "r")
    ends = z["meta/episode_ends"][:]
    starts = np.r_[0, ends[:-1]]
    act = z["data/action"][:]
    close = [s + int(np.argmax(act[s:t, 6] > 0)) for s, t in zip(starts, ends)]
    uv = H.project(cam, H.fk(act[close, :6]))
    cells = z["meta/ep_cell"][:]
    m = torch.load(f"{cue_dir}/maps.pt", map_location="cpu", weights_only=False)
    first = np.arange(len(ends)) * m["args"]["frames"]
    pk = H.own_peaks(m["lps"][torch.from_numpy(first)]).numpy()
    flags = np.array(FLAGS)[cells]
    err = np.linalg.norm(pk - uv[:, None], axis=2)
    rng = np.random.default_rng(0)
    print(f"{level}: {len(ends)} demos, first frame; grasp pixel u {uv[:, 0].min():.1f} to {uv[:, 0].max():.1f}, v "
          f"{uv[:, 1].min():.1f} to {uv[:, 1].max():.1f}; the mean grasp pixel is within 1.5 px of the grasp pixel in "
          f"{(np.linalg.norm(uv - uv.mean(0), axis=1) < 1.5).mean():.2f} of the demos")
    for i, k in enumerate(H.SENSORS):
        b = flags[:, i] == 1
        other = np.mean([(np.linalg.norm(pk[b, i] - uv[b][rng.permutation(b.sum())], axis=1) < 1.5).mean() for _ in range(200)])
        print(f"  {k} blind ({b.sum()} demos): own peak within 1.5 px {(err[b, i] < 1.5).mean():.2f}, within 3 px "
              f"{(err[b, i] < 3).mean():.2f}; paired with the grasp pixels of other demos {other:.2f}; correlation of the "
              f"peak with the grasp pixel u {np.corrcoef(pk[b, i, 0], uv[b, 0])[0, 1]:.2f}, v {np.corrcoef(pk[b, i, 1], uv[b, 1])[0, 1]:.2f}")


def _env(level):
    from env import GlassPickEnv
    from testsets import manifest
    return GlassPickEnv(config=manifest()["levels"][level])


@cli.command()
@click.option("--level", required=True)
@click.option("-n", default=330, show_default=True, help="scenes (seeds from 0)")
def frames(level, n):
    import mujoco
    import env as E
    env = _env(level)
    s_ = env.sensors
    rows = {"colour": [], "depth": []}
    depth8 = lambda v: (255 * np.clip(v, 0, 2.0) / 2.0).astype(np.uint8)
    for seed in range(n):
        env.reset(seed)
        cell = int(env.episode_info["cell"])
        if FLAGS[cell][0] != 1 and FLAGS[cell][1] != 1:
            continue
        a = s_.render(env.data, env.temps, env.emis, env.t_env, np.random.default_rng([2, seed, 0]), env.sensor_scene)
        r = s_.r
        r.enable_segmentation_rendering()
        r.update_scene(env.data, camera=s_.camera, scene_option=s_.show_all)
        seg = r.render()[..., 0].copy()
        r.disable_segmentation_rendering()
        glass = np.isin(seg, [g for g in s_.transparent_geoms if env.model.geom_bodyid[g] == env.model.body(env.obj).id])
        env.data.qpos[env.obj_q[env.obj]] += 10.0                 # the glass out of the scene; nothing else changes
        mujoco.mj_forward(env.model, env.data)
        b = s_.render(env.data, env.temps, env.emis, env.t_env, np.random.default_rng([2, seed, 0]), env.sensor_scene)
        g96 = E._resize((255 * glass).astype(np.uint8)) > 0
        ys, xs = np.nonzero(g96)
        box = np.zeros_like(g96)
        if len(ys):
            box[max(ys.min() - 3, 0):ys.max() + 4, max(xs.min() - 3, 0):xs.max() + 4] = True
        if FLAGS[cell][0] == 1:
            d = np.abs(E._resize(a[0]).astype(int) - E._resize(b[0]).astype(int)).max(-1)
            rows["colour"].append((d.max(), (d > 0).sum(), ((d > 0) & box).sum()))
        if FLAGS[cell][1] == 1:
            d = np.abs(E._resize(depth8(a[1])).astype(int) - E._resize(depth8(b[1])).astype(int))
            rows["depth"].append((d.max(), (d > 0).sum(), ((d > 0) & box).sum()))
    env.close()
    for k, v in rows.items():
        v = np.array(v, float)
        dif = v[:, 0] > 0
        print(f"{level}, {k} blind, {len(v)} scenes of seeds 0 to {n - 1}, 96 x 96 frame with and without the glass: any "
              f"pixel differs in {dif.mean():.2f} of the scenes (in those: {np.median(v[dif, 1]) if dif.any() else 0:.0f} pixels, "
              f"by at most {v[:, 0].max():.0f} of 255); a pixel within 3 px of the glass differs in {np.mean(v[:, 2] > 0):.2f}")


@cli.command()
@click.option("--level", required=True)
@click.option("-n", default=700, show_default=True, help="scenes (seeds from 0)")
def layout(level, n):
    import env as E
    env = _env(level)
    gx, gy = np.meshgrid(np.linspace(*E.SPAWN_X, 49), np.linspace(*E.SPAWN_Y, 121), indexing="ij")
    grid = np.stack([gx.ravel(), gy.ravel()], 1)
    centre, rows = grid.mean(0), []
    for seed in range(n):
        env.reset(seed)
        d = env.data
        glass = d.qpos[env.obj_q[env.obj]:env.obj_q[env.obj] + 2].copy()
        r_obj = env.obj_r[env.obj]
        names = [E.EXTRA_NAMES[i] for i in env.episode_info["extra_ids"] if i >= 0]
        ok_d = np.ones(len(grid), bool)
        for nm in names:
            p = d.qpos[env.extra_q[nm]:env.extra_q[nm] + 2]
            ok_d &= np.linalg.norm(grid - p, axis=1) - env.extra_r[nm] - r_obj >= E.GAP_TARGET
        ok_p = np.ones(len(grid), bool)
        if getattr(env, "place_xy", None) is not None:
            R = float(np.hypot(*E.MAT_HALF)) if env.align else env.cfg.place_radius
            ok_p = np.linalg.norm(grid - env.place_xy, axis=1) - R - r_obj >= E.GAP_EXTRA
        guess = lambda m: grid[m].mean(0) if m.any() else centre
        rows.append([len(names), ok_d.mean(), (ok_d & ok_p).mean(),
                     *(100 * np.linalg.norm(g - glass) for g in (centre, guess(ok_d), guess(ok_d & ok_p)))])
    env.close()
    r = np.array(rows)
    print(f"{level}, seeds 0 to {n - 1}: spawn region {100 * (E.SPAWN_X[1] - E.SPAWN_X[0]):.0f} x "
          f"{100 * (E.SPAWN_Y[1] - E.SPAWN_Y[0]):.0f} cm; distractors on the table: median {np.median(r[:, 0]):.0f} "
          f"({r[:, 0].min():.0f} to {r[:, 0].max():.0f}); share of the spawn region the glass can be in given the "
          f"distractors: median {np.median(r[:, 1]):.2f}, given the place target too: {np.median(r[:, 2]):.2f}")
    for j, k in zip(range(3, 6), ("the centre of the spawn region (no layout)", "the distractors", "the distractors and the place target")):
        e = r[:, j]
        print(f"  guess from {k}: median error {np.median(e):.1f} cm, within 1.5 cm {np.mean(e < 1.5):.2f}, within 3 cm "
              f"{np.mean(e < 3):.2f}, within 5 cm {np.mean(e < 5):.2f}")


if __name__ == "__main__":
    cli()
