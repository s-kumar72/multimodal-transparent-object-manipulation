"""
Can the scripted teacher do a task of the second version of the benchmark in every scene? Runs it on consecutive seeds
of a level's training scenes or of one of its test sets (no policy, the policy images are rendered as in a demo):

  python sim/task_check.py LEVEL [-n 220] [--test_set j_cells11] [--max_steps 240]

Printed: success over the episodes, per glass shape and per visibility cell (or pair of cells), the number of steps,
and every failed episode with what the episode record says about it. For task "two" also how many scenes needed their
distractors taken off the table for a clear plan (place_clear 2).
"""
import sys
from collections import Counter

import click
import numpy as np

sys.path.insert(0, "sim")
from env import GlassPickEnv  # noqa: E402
from teacher import ScriptedTeacher  # noqa: E402
from testsets import manifest, test_set  # noqa: E402


@click.command()
@click.argument("level")
@click.option("-n", default=220, show_default=True)
@click.option("--test_set", "ts", default=None, help="a test set of the manifest (default: the training scenes)")
@click.option("--max_steps", default=None, type=int, help="default: 480 for task two, else 240")
def main(level, n, ts, max_steps):
    if ts is None:
        cfg, seeds, names = manifest()["levels"][level], range(n), None
    else:
        cfg, seeds, names = test_set(level, ts, n=n)
    probe = GlassPickEnv(config=cfg)
    two = probe.two
    probe.close()
    env = GlassPickEnv(max_steps=max_steps or (480 if two else 240), config=cfg)
    teacher, rows = ScriptedTeacher(), []
    for seed in seeds:
        env.reset(seed, obj=None if names is None else names[seed % len(names)])
        teacher.reset(env, np.random.default_rng(seed))
        steps = 0
        while not teacher.done():
            env.step(teacher.act())
            steps += 1
        info = env.episode_info
        row = dict(seed=seed, obj=env.obj, cell=int(info["cell"]), ok=bool(env.success()), steps=steps,
                   clear=info.get("place_clear"))
        if two:
            row.update(obj2=env.obj2, placed=env.placed_count())
        if env.align:
            row.update(yaw_error=round(env.yaw_error(), 1), offset_cm=round(100 * env.place_offset(), 1))
        rows.append(row)
    env.close()
    ok = np.array([r["ok"] for r in rows])
    print(f"{level}{' / ' + ts if ts else ' (training scenes)'}: teacher success {ok.mean():.3f} ({ok.sum()} of {len(ok)}); steps "
          f"median {int(np.median([r['steps'] for r in rows]))}, max {max(r['steps'] for r in rows)}")
    per = lambda key: ", ".join(f"{k} {sum(r['ok'] for r in rows if r[key] == k)}/{sum(r[key] == k for r in rows)}"
                                for k in sorted({r[key] for r in rows}))
    print("  per glass: " + per("obj"))
    print("  per cell:  " + per("cell"))
    if rows[0]["clear"] is not None:
        print("  place_clear: " + ", ".join(f"{k}: {v}" for k, v in sorted(Counter(r["clear"] for r in rows).items())))
    for r in rows:
        if not r["ok"]:
            print("  failed:", r)


if __name__ == "__main__":
    main()
