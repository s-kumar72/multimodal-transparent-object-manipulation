"""
Evaluates a composition of separately trained Diffusion Policy checkpoints in the sim (policy/compose_policy.py):
their noise predictions are combined with fixed weights at every denoising step. Same outputs as policy/eval.py.

Run from the repo root with the PYTHONPATH that policy/train.sh uses:
  python policy/eval_compose.py -c RGBD_CKPT -c RGBT_CKPT --weights 0.5,0.5 -o OUT --level L0_bench \
      --test_set a_in_distribution --test_set g_heldout_all [--n_test 200] [--test_start_seed 100000] [--n_envs 10]
  python policy/eval_compose.py -c RGBD_CKPT -c RGBT_CKPT -c RGB_CKPT --weights 1,1,-1 -o OUT ...
  python policy/eval_compose.py -c RGBD_CKPT -c RGBT_CKPT --switch ROUTED_CKPT -o OUT ...
Each test set goes to OUT/<test set>/ with eval_log.json and episodes.json. The env renders the union of the
members' observation keys; every member gets the keys it was trained on.
--switch: instead of fixed weights, the first member acts where the glass is visible in depth and the second where
it is not, decided per environment at every decision with the visibility cue of the route=visible checkpoint
ROUTED_CKPT (only its cue settings are read: thresholds and glass-mask network); episodes.json then has the weights
"w" and the cue per decision.
"""
import json
import pathlib

import click
import dill
import hydra
import torch
from omegaconf import OmegaConf

from policy.compose_policy import ComposedPolicy
from policy.eval import attach_w_trace

OmegaConf.register_new_resolver("eval", eval, replace=True)


@click.command()
@click.option("-c", "--checkpoint", "checkpoints", multiple=True, required=True, help="one per member (repeat)")
@click.option("--weights", default=None, help="comma separated, one per checkpoint")
@click.option("--switch", "switch_ckpt", default=None,
              help="route=visible checkpoint whose visibility cue picks member 1 (depth shows the glass) or member 2")
@click.option("-o", "--output_dir", required=True)
@click.option("--n_test", type=int, default=50, show_default=True)
@click.option("--test_start_seed", type=int, default=100000, show_default=True)
@click.option("--n_envs", type=int, default=None, help="sim envs run in parallel (default: all)")
@click.option("-d", "--device", default="cuda:0", show_default=True)
@click.option("--level", required=True, help="scene level, e.g. L0_bench")
@click.option("--test_set", "test_sets", multiple=True, required=True, help="test set name(s) of the level")
def main(checkpoints, weights, switch_ckpt, output_dir, n_test, test_start_seed, n_envs, device, level, test_sets):
    assert (weights is None) != (switch_ckpt is None), "give --weights or --switch"
    weights = [float(w) for w in weights.split(",")] if weights else [0.0] * len(checkpoints)
    assert len(weights) == len(checkpoints), "one weight per checkpoint"
    members, obs_keys, cfgs = [], [], []
    for ckpt in checkpoints:
        payload = torch.load(open(ckpt, "rb"), pickle_module=dill, weights_only=False, map_location="cpu")
        cfg = payload["cfg"]
        workspace = hydra.utils.get_class(cfg._target_)(cfg, output_dir=str(output_dir))
        workspace.load_payload(payload, exclude_keys=None, include_keys=None)
        policy = workspace.ema_model if "ema_model" in payload["state_dicts"] else workspace.model
        members.append(policy.eval())
        obs_keys.append(list(cfg.task.shape_meta.obs.keys()))
        cfgs.append(cfg)
    cue_cfg = None
    if switch_ckpt:
        routed = torch.load(open(switch_ckpt, "rb"), pickle_module=dill, weights_only=False, map_location="cpu")["cfg"]
        assert routed.policy.obs_encoder.get("route", "none") == "visible", "--switch needs a route=visible checkpoint"
        cue_cfg = routed.task.env_runner.shape_meta.obs.cue
        assert not any("cue" in keys for keys in obs_keys), "the members must not have a cue observation themselves"
    policy = ComposedPolicy(members, weights, obs_keys, switch_key="cue" if switch_ckpt else None)
    policy = policy.to(torch.device(device)).eval()

    # the env of the first member, rendering the union of the members' observation keys
    runner_cfg = cfgs[0].task.env_runner
    OmegaConf.set_struct(runner_cfg, False)
    env_obs = runner_cfg.shape_meta.obs
    for other in cfgs[1:]:
        for key, attr in other.task.shape_meta.obs.items():
            if key in env_obs:
                assert env_obs[key] == attr, f"members disagree on observation '{key}'"
            else:
                env_obs[key] = attr
    if cue_cfg is not None:
        env_obs["cue"] = cue_cfg
    runner_cfg.n_train, runner_cfg.n_train_vis = 0, 0
    runner_cfg.n_test, runner_cfg.n_test_vis = n_test, 0
    runner_cfg.test_start_seed = test_start_seed
    runner_cfg.n_envs = min(n_envs, n_test) if n_envs else None
    runner_cfg.level = level

    for name in test_sets:
        out = pathlib.Path(output_dir) / name
        out.mkdir(parents=True, exist_ok=True)
        runner_cfg.test_set = name
        runner = hydra.utils.instantiate(runner_cfg, output_dir=str(out))
        if switch_ckpt:
            policy.w_trace = []
        log = runner.run(policy)
        runner.env.close()
        records = runner.last_records
        if switch_ckpt:
            records = attach_w_trace(records, policy.w_trace, runner.n_envs)
            policy.w_trace = None
        result = {"checkpoints": [str(c) for c in checkpoints], "weights": None if switch_ckpt else weights,
                  "switch": str(switch_ckpt) if switch_ckpt else None, "obs_keys": obs_keys,
                  "level": level, "test_set": name, "n_test": n_test, "test_start_seed": test_start_seed,
                  **{k: float(v) for k, v in sorted(log.items())}}
        (out / "eval_log.json").write_text(json.dumps(result, indent=2))
        (out / "episodes.json").write_text(json.dumps(records, indent=1))
        print(f"saved {out / 'eval_log.json'}", flush=True)


if __name__ == "__main__":
    main()
