"""
Evaluates a trained checkpoint in the sim: success rate overall and per object, plus a few rollout videos.

Run from the repo root (the scene path is relative) with the PYTHONPATH that policy/train.sh uses:
  python policy/eval.py -c CKPT -o data/eval/NAME [--n_test 50] [--test_start_seed 100000] [--n_envs 18]
  python policy/eval.py -c CKPT -o OUT --level L3 --test_set a_in_distribution --test_set g_heldout_all
With --level, each test set (sim/configs/manifest.json) goes to OUT/<test set>/ with eval_log.json and
episodes.json (scene conditions and failure type per episode); the level defaults to the one the policy was trained on.
--drop thermal (repeatable) switches a sensor off at inference: a gated encoder uses its missing embedding with weight 0,
a plain encoder gets a black image. The output folder gets a "_drop_<mods>" suffix.
--save_w stores, per episode in episodes.json, the gate weights and raw cues at every decision step ("w", "cue",
policies with a cue-gated encoder only), for the w vs. thermal-gap plots.

Robustness of a route=visible gate (all off by default; the test-set folder names do not change, so give each
condition its own output folder): --gate_error flip=0.1 inverts each visibility flag with that probability at every
decision, --gate_error depth=0 (or thermal=0, =1) forces a flag; --vis_thr 7.5,15 recomputes the flags with other
thresholds than the ones of training; --mask_noise drop=0.3 / shift=2 degrades the predicted glass mask the flags are
measured on (policy/obs_util.py degrade_mask). --route_hold: a frame without a glass mask keeps the flags of the last
frame that had one (both off before the first) instead of switching both sensors on.

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
@click.option("--drop", "drops", multiple=True, help="image modality switched off at inference (repeatable)")
@click.option("--save_w", is_flag=True, help="store gate weights and cues per decision step in episodes.json")
@click.option("--route", default=None, type=click.Choice(["visible", "depth_first", "all_on", "rgb_only"]),
              help="routing rule at inference for a route=visible encoder (default: as trained)")
@click.option("--route_latch", is_flag=True, help="keep the first step's visibility flags for the whole episode")
@click.option("--guidance", default=1.0, type=float,
              help="s: denoise with eps_rgb + s (eps_routed - eps_rgb) (route=visible encoders; 1 = off)")
@click.option("--gate_error", default=None, help="wrong visibility flags: flip=P and / or depth=0|1, thermal=0|1 (comma separated)")
@click.option("--vis_thr", default=None, help="DEPTH,THERMAL visibility thresholds at inference (default: as trained)")
@click.option("--mask_noise", default=None, help="errors of the predicted glass mask: drop=P and / or shift=K (comma separated)")
@click.option("--route_hold", is_flag=True, help="no glass mask in a frame: keep the last flags instead of both on")
def main(checkpoint, output_dir, n_test, test_start_seed, n_envs, n_vis, device, level, test_sets, drops, save_w,
         route, route_latch, guidance, gate_error, vis_thr, mask_noise, route_hold):

    payload = torch.load(open(checkpoint, "rb"), pickle_module=dill, weights_only=False, map_location="cpu")
    cfg = payload["cfg"]
    workspace = hydra.utils.get_class(cfg._target_)(cfg, output_dir=str(output_dir))
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    policy = workspace.ema_model if "ema_model" in payload["state_dicts"] else workspace.model
    policy.to(torch.device(device))
    policy.eval()
    encoder = getattr(policy, "obs_encoder", None)
    if drops:
        assert hasattr(encoder, "force_drop"), "--drop needs SharedCropObsEncoder / GatedObsEncoder"
        encoder.force_drop = set(drops)
    if route:
        assert getattr(encoder, "route", "none") != "none", "--route needs an encoder trained with route=visible / soft"
        assert encoder.route == "visible" or route != "depth_first", "depth_first is a rule of route=visible"
        encoder.route_eval = route
    if route_latch or guidance != 1.0:
        assert getattr(encoder, "route", "none") != "none", "--route_latch / --guidance need route=visible / soft"
        assert encoder.route == "visible" or not route_latch, "--route_latch needs route=visible"
        policy.route_latch, policy.guidance = route_latch, guidance
    spec = lambda s: {k.strip(): float(v) for k, v in (kv.split("=") for kv in s.split(","))} if s else None
    gate_error, mask_noise = spec(gate_error), spec(mask_noise)
    vis_thr = [float(v) for v in vis_thr.split(",")] if vis_thr else None
    if gate_error or vis_thr or mask_noise:
        assert getattr(encoder, "route", "none") == "visible", "--gate_error / --vis_thr / --mask_noise need route=visible"
        assert not gate_error or set(gate_error) <= {"flip", "depth", "thermal", "seed"}, gate_error
        assert not mask_noise or set(mask_noise) <= {"drop", "shift", "seed"}, mask_noise
        policy.gate_error = gate_error
        cue_cfg = cfg.task.env_runner.shape_meta.obs.cue     # the eval env computes the cue with these options
        OmegaConf.set_struct(cue_cfg, False)
        if vis_thr:
            assert len(vis_thr) == 2
            cue_cfg.vis_thr = vis_thr
        if mask_noise:
            cue_cfg.mask_noise = mask_noise
    if route_hold:
        assert getattr(encoder, "route", "none") == "visible", "--route_hold needs route=visible"
        thr = cfg.task.env_runner.shape_meta.obs.cue.vis_thr
        assert min(float(v) for v in thr) >= 0, "--route_hold tells a missing mask by flags on with zero steps"
        policy.route_hold = True
    suffix = ("_drop_" + "_".join(sorted(drops))) if drops else ""

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
        out = pathlib.Path(output_dir) / (name + suffix) if level is not None else pathlib.Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        runner_cfg.test_set = name
        runner = hydra.utils.instantiate(runner_cfg, output_dir=str(out))
        if save_w and hasattr(policy, "w_trace"):
            policy.w_trace = []
        log = runner.run(policy)
        runner.env.close()
        records = runner.last_records
        if save_w and getattr(policy, "w_trace", None):
            records = attach_w_trace(records, policy.w_trace, runner.n_envs)
            policy.w_trace = None
        result = {"checkpoint": str(checkpoint), "level": level, "test_set": name, "n_test": n_test,
                  "test_start_seed": test_start_seed, "drop": sorted(drops), "route": route,
                  "route_latch": route_latch, "guidance": guidance,
                  "gate_error": gate_error, "vis_thr": vis_thr, "mask_noise": mask_noise, "route_hold": route_hold,
                  **{k: plain(v) for k, v in sorted(log.items())}}
        (out / "eval_log.json").write_text(json.dumps(result, indent=2))
        (out / "episodes.json").write_text(json.dumps(records, indent=1))
        print(f"saved {out / 'eval_log.json'}", flush=True)

def attach_w_trace(records, trace, n_envs):
    """trace[chunk][step] = {"w": (n_envs, M), "cue": (n_envs, 6)} from GatedDiffusionPolicy; episode i of the runner
    ran in chunk i // n_envs, slot i % n_envs (the last chunk is padded with copies that are not in records)."""
    out = []
    for i, rec in enumerate(records):
        chunk, slot = i // n_envs, i % n_envs
        steps = trace[chunk] if chunk < len(trace) else []
        rec = dict(rec, w=[s["w"][slot].tolist() for s in steps],
                   cue=[s["cue"][slot].tolist() for s in steps if s["cue"] is not None])
        out.append(rec)
    return out


if __name__ == "__main__":
    main()
