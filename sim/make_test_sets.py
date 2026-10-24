"""
Builds the evaluation test sets. A test set is (scene config, split, seeds): GlassPickEnv(config, split).reset(seed)
rebuilds exactly the same scene, so every modality (rgb / rgbd / rgbt / rgbdt) is evaluated on the same scenes.
All sets use the same seeds (default 100000..), so they differ only in the held-out factor; training uses seeds from 0.

Writes <out>/<set>/manifest.json (config, split, seeds, teacher success per seed) and,
with --demos, <out>/<set>/demos.zarr with the teacher's successful episodes (same format as collect_demos.py).

run via python sim/make_test_sets.py --sets a b c d g --n 50 --out data/pilot_v0/test --demos --workers 5
"""
import argparse
import json
from pathlib import Path

from collect_demos import collect
from config import load_scene_config, load_thermal_config

# key: (scene config in sim/configs/scene/, split)
TEST_SETS = {
    "a": ("test/a_indist", "train"),                 # in-distribution (train pools, unseen seeds)
    "b": ("test/b_heldout_background", "test"),      # held-out backgrounds
    "c": ("test/c_heldout_lighting", "test"),        # held-out lighting
    "d": ("test/d_heldout_distractors", "test"),     # held-out distractors
    "e": ("test/e_liquid", "train"),                 # liquid: every fill level x temperature
    "f": ("test/f_crossover", "train"),              # forced thermal crossover (|gap| < 1 C)
    "g": ("test/g_heldout_all", "test"),             # all held-out factors together
}
TEST_SEED = 100000

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--sets", nargs="+", default=list(TEST_SETS), choices=list(TEST_SETS))
    p.add_argument("--n", type=int, default=50, help="episodes (seeds) per test set")
    p.add_argument("--seed", type=int, default=TEST_SEED)
    p.add_argument("--out", default="data/test_sets")
    p.add_argument("--demos", action="store_true", help="also run the teacher and save its successful episodes")
    p.add_argument("--workers", type=int, default=1)
    args = p.parse_args()

    for key in args.sets:
        config, split = TEST_SETS[key]
        cfg = load_scene_config(config)
        out = Path(args.out) / cfg["name"]
        out.mkdir(parents=True, exist_ok=True)
        seeds = list(range(args.seed, args.seed + args.n))
        manifest = {"name": cfg["name"], "config": config, "split": split, "seeds": seeds,
                    "scene_config": cfg, "thermal_config": load_thermal_config(cfg["thermal"])}
        if args.demos:
            tried = collect(str(out / "demos.zarr"), cfg, split, seeds, args.n, args.workers)
            manifest["teacher_success"] = {str(s): ok for s, ok in tried}
            print(f"{cfg['name']}: teacher success {sum(ok for _, ok in tried)}/{len(tried)}", flush=True)
        (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
        print("saved", out / "manifest.json", flush=True)
