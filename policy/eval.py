"""
Evaluates a trained checkpoint in the sim: success rate overall and per object, plus a few rollout videos.

Run from the repo root (the scene path is relative) with the PYTHONPATH that policy/train.sh uses:
  python policy/eval.py -c CKPT -o data/eval/NAME [--n_test 50] [--test_start_seed 100000] [--n_envs 18]
  python policy/eval.py -c CKPT -o OUT --level L3 --test_set a_in_distribution --test_set g_heldout_all
With --level, each test set (sim/configs/manifest.json) goes to OUT/<test set>/ with eval_log.json and
episodes.json (scene conditions and failure type per episode); the level defaults to the one the policy was trained on.

Test episode k uses seed test_start_seed + k; the object is seed % 5, so n_test=50 gives 10 episodes per object.
The seeds 100000-100049 are the ones evaluated during training; any other range gives new scenes drawn the same way.
Works with full training checkpoints and with EMA-only exports (the EMA weights are evaluated when present).
"""
import json
import pathlib

import click
import dill
import hydra
import torch
from omegaconf import OmegaConf

OmegaConf.register_new_resolver("eval", eval, replace=True)

@click.command()
@click.option("-c", "--checkpoint", required=True)
@click.option("-o", "--output_dir", required=True)
@click.option("--n_test", type=int, default=50, show_default=True)
@click.option("--test_start_seed", type=int, default=100000, show_default=True)
@click.option("--n_envs", type=int, default=None, help="sim envs run in parallel (default: all; 18 fits a 16 GB GPU)")
@click.option("--n_vis", type=int, default=3, show_default=True, help="test episodes saved as videos")
@click.option("-d", "--device", default="cuda:0", show_default=True)
@click.option("--level", default=None, help="scene level (default: the training config's task.env_runner.level)")
@click.option("--test_set", "test_sets", multiple=True, help="test set name(s) from sim/configs/manifest.json")
def main(checkpoint, output_dir, n_test, test_start_seed, n_envs, n_vis, device, level, test_sets):

    payload = torch.load(open(checkpoint, "rb"), pickle_module=dill, weights_only=False, map_location="cpu")
    cfg = payload["cfg"]
    workspace = hydra.utils.get_class(cfg._target_)(cfg, output_dir=str(output_dir))
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    policy = workspace.ema_model if "ema_model" in payload["state_dicts"] else workspace.model
    policy.to(torch.device(device))
    policy.eval()

    runner_cfg = cfg.task.env_runner
    runner_cfg.n_train, runner_cfg.n_train_vis = 0, 0
    runner_cfg.n_test, runner_cfg.n_test_vis = n_test, min(n_vis, n_test)
    runner_cfg.test_start_seed = test_start_seed
    runner_cfg.n_envs = min(n_envs, n_test) if n_envs else None
    OmegaConf.set_struct(runner_cfg, False)
    if level is not None:
        runner_cfg.level = level
    level = runner_cfg.get("level", None)

    def plain(v):
        if hasattr(v, "_path"):          # wandb.Video -> path of the mp4
            return v._path
        return float(v)
    for name in (test_sets or [runner_cfg.get("test_set", "a_in_distribution")]):
        out = pathlib.Path(output_dir) / name if level is not None else pathlib.Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        runner_cfg.test_set = name
        runner = hydra.utils.instantiate(runner_cfg, output_dir=str(out))
        log = runner.run(policy)
        runner.env.close()
        result = {"checkpoint": str(checkpoint), "level": level, "test_set": name, "n_test": n_test,
                  "test_start_seed": test_start_seed, **{k: plain(v) for k, v in sorted(log.items())}}
        (out / "eval_log.json").write_text(json.dumps(result, indent=2))
        (out / "episodes.json").write_text(json.dumps(runner.last_records, indent=1))
        print(f"saved {out / 'eval_log.json'}", flush=True)

if __name__ == "__main__":
    main()
