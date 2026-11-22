"""
Offline probe of sensor guidance (policy/gated_policy.py): what the difference between the routed and the rgb-only
noise prediction looks like on demo frames, and what extrapolating along it does. No training, no simulator.

  python policy/analyze_guidance.py -c CKPT -o OUT.json [--split val|train] [--n_seq 1500] [--n_act 600]

For each demo sequence (n_obs_steps frames + the expert's action chunk) of the chosen split, clean frames:
  noise level   at the diffusion steps --steps, one noise draw per sequence and step: e_r = denoiser output with the
                encoder's routing rule (all sensors for a policy trained without routing), e_0 = with depth and thermal
                off, d = e_r - e_0. Stored: |d|, |e_r|, |eps - e_0|, <d, eps - e_0> and the mean squared error of
                e_0 + s d against eps for every scale s. The least-squares scale of a set of sequences is
                s* = sum <d, eps - e_0> / sum |d|^2: above 1, going further along d than the network does lowers the
                denoising error on those sequences.
  action level  the full sampler from the same initial noise and the same sampler noise for every scale; squared
                error of the executed action steps against the expert's, normalized action units. With --n_noise K
                this is repeated for K noise draws per sequence (errors are averaged over them); the executed action
                steps of every draw and scale are stored (act_pred, normalized), with the expert's (act_gt) and the
                time of the last observation in its episode (t_obs, ep_len), for shifts and spread across draws.
Per-sequence values go to OUT.npz (episode index and scene flags included, for bootstraps over episodes); OUT.json has
the summaries per group (all / neither / see-through / crossover, from the episode flags of the demo zarr); size,
direction and s* are over the sequences with a sensor on (with both off the two predictions are identical).
"""
import copy
import json
import pathlib

import click
import dill
import hydra
import numpy as np
import torch
from omegaconf import OmegaConf

from diffusion_policy.common.pytorch_util import dict_apply

OmegaConf.register_new_resolver("eval", eval, replace=True)


@click.command()
@click.option("-c", "--checkpoint", required=True)
@click.option("-o", "--output", required=True, help="summary .json; per-sequence values go to the same name .npz")
@click.option("--split", type=click.Choice(["val", "train"]), default="val", show_default=True)
@click.option("--n_seq", type=int, default=1500, show_default=True, help="sequences for the noise-level part")
@click.option("--n_act", type=int, default=600, show_default=True, help="of them, sequences for the action-level part")
@click.option("--steps", default="5,20,40,60,80,95", show_default=True, help="diffusion steps of the noise-level part")
@click.option("--scales", default="0,1,1.5,3,5,7", show_default=True)
@click.option("--n_noise", type=int, default=1, show_default=True, help="noise draws per sequence, action-level part")
@click.option("--batch", type=int, default=64, show_default=True)
@click.option("-d", "--device", default="cuda:0", show_default=True)
@click.option("--seed", type=int, default=0, show_default=True)
def main(checkpoint, output, split, n_seq, n_act, steps, scales, n_noise, batch, device, seed):
    steps = [int(x) for x in steps.split(",")]
    scales = [float(x) for x in scales.split(",")]
    out = pathlib.Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = torch.load(open(checkpoint, "rb"), pickle_module=dill, weights_only=False, map_location="cpu")
    cfg = payload["cfg"]
    workspace = hydra.utils.get_class(cfg._target_)(cfg, output_dir=str(out.parent))
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    policy = workspace.ema_model if "ema_model" in payload["state_dicts"] else workspace.model
    del payload
    policy.to(torch.device(device)).eval()
    enc, model, sched = policy.obs_encoder, policy.model, policy.noise_scheduler
    assert getattr(enc, "route", "none") != "none" and policy.obs_as_global_cond, "needs a route=visible / soft encoder"
    To, Ta = policy.n_obs_steps, policy.n_action_steps

    dataset = hydra.utils.instantiate(cfg.task.dataset)
    if split == "val":
        ds = dataset.get_validation_dataset()
    else:
        ds = copy.copy(dataset)
        ds.augment, ds._rng = None, None
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(ds))[:min(n_seq, len(ds))]          # random order: the first n_act get the sampler too
    episode = np.array([ds._episode_of(int(i)) for i in idx])
    ends = ds.episode_ends
    ep_start = np.where(episode > 0, ends[np.maximum(episode - 1, 0)], 0)
    rows = np.asarray(ds.sampler.indices)[idx]                    # buffer start, buffer end, sample start, sample end
    t_obs = rows[:, 0] - rows[:, 2] - ep_start + policy.n_obs_steps - 1   # time of the last observation in its episode
    ep_len = ends[episode] - ep_start
    loader = torch.utils.data.DataLoader(torch.utils.data.Subset(ds, idx.tolist()), batch_size=batch, shuffle=False,
                                         num_workers=4)
    print(f"{split} split: {len(ds)} sequences, using {len(idx)} from {len(set(episode.tolist()))} episodes; "
          f"action-level part on the first {min(n_act, len(idx))}", flush=True)

    rec = {k: [] for k in ("nd", "ner", "nr", "dot", "err", "w", "st", "xo", "gap")}
    act = {k: [] for k in ("err", "err_first", "err_pos", "err_grip", "pred", "gt")}
    done_act = 0
    flat = lambda a: a.reshape(a.shape[0], -1)
    with torch.no_grad():
        for bi, b in enumerate(loader):
            obs = dict_apply(b["obs"], lambda x: x.to(device))
            nobs = policy.normalizer.normalize(obs)
            nact = policy.normalizer["action"].normalize(b["action"].to(device))
            B = nact.shape[0]
            this = dict_apply(nobs, lambda x: x[:, :To, ...].reshape(-1, *x.shape[2:]))
            rule = enc.route_eval
            cond_r = enc(this).reshape(B, -1)
            w = enc.last_w.view(B, To, -1)[:, -1]
            enc.route_eval = "rgb_only"
            cond_0 = enc(this).reshape(B, -1)
            enc.route_eval = rule
            meta = b.get("meta", {})
            rec["w"].append(w.cpu().numpy())
            for k, name in (("st", "ep_glass_seethrough"), ("xo", "ep_is_crossover"), ("gap", "ep_thermal_gap")):
                rec[k].append(meta[name].numpy() if name in meta else np.full(B, np.nan, np.float32))

            g = torch.Generator(device).manual_seed(seed * 100003 + bi)
            cols = {k: [] for k in ("nd", "ner", "nr", "dot", "err")}
            for t in steps:
                eps = torch.randn(nact.shape, generator=g, device=device, dtype=nact.dtype)
                tt = torch.full((B,), t, device=device, dtype=torch.long)
                x = sched.add_noise(nact, eps, tt)
                e_r = model(x, tt, local_cond=None, global_cond=cond_r)
                e_0 = model(x, tt, local_cond=None, global_cond=cond_0)
                d, r = flat(e_r - e_0), flat(eps - e_0)
                cols["nd"].append(d.norm(dim=1))
                cols["ner"].append(flat(e_r).norm(dim=1))
                cols["nr"].append(r.norm(dim=1))
                cols["dot"].append((d * r).sum(1))
                cols["err"].append(torch.stack([((s * d - r) ** 2).mean(1) for s in scales], 1))
            for k, v in cols.items():
                rec[k].append(torch.stack(v, 1).cpu().numpy())                 # (B, steps[, scales])

            if done_act < n_act:
                gt = nact[:, To - 1:To - 1 + Ta]
                preds = []
                for k in range(n_noise):
                    x0 = torch.randn(nact.shape, generator=g, device=device, dtype=nact.dtype)
                    row = []
                    for s in scales:
                        sched.set_timesteps(policy.num_inference_steps)
                        # the same sampler noise for every scale of a draw
                        gs = torch.Generator(device).manual_seed(seed * 100003 + 7919 * (k + 1) + bi)
                        x = x0.clone()
                        for t in sched.timesteps:
                            if s == 0:
                                e = model(x, t, local_cond=None, global_cond=cond_0)
                            else:
                                e = model(x, t, local_cond=None, global_cond=cond_r)
                                if s != 1:
                                    e_0 = model(x, t, local_cond=None, global_cond=cond_0)
                                    e = e_0 + s * (e - e_0)
                            x = sched.step(e, t, x, generator=gs, **policy.kwargs).prev_sample
                        row.append(x[:, To - 1:To - 1 + Ta])
                    preds.append(torch.stack(row, 1))
                pred = torch.stack(preds, 1)                                      # (B, K, scales, Ta, Da)
                sq = ((pred - gt[:, None, None]) ** 2).mean(1)                    # (B, scales, Ta, Da), mean over draws
                act["err"].append(sq.mean((2, 3)).cpu().numpy())
                act["err_first"].append(sq[:, :, 0].mean(2).cpu().numpy())
                act["err_pos"].append(sq[..., :3].mean((2, 3)).cpu().numpy())
                act["err_grip"].append(sq[..., -1].mean(2).cpu().numpy())
                act["pred"].append(pred.cpu().numpy().astype(np.float32))
                act["gt"].append(gt.cpu().numpy().astype(np.float32))
                done_act += B
            print(f"batch {bi + 1}/{len(loader)}", flush=True)

    R = {k: np.concatenate(v) for k, v in rec.items()}
    A = {k: np.concatenate(v) for k, v in act.items()}
    n_a = len(A["err"])
    np.savez_compressed(out.with_suffix(".npz"), episode=episode, steps=np.array(steps), scales=np.array(scales),
                        n_act=n_a, t_obs=t_obs, ep_len=ep_len,
                        **{f"noise_{k}": v for k, v in R.items()}, **{f"act_{k}": v for k, v in A.items()})

    st, xo = R["st"] > 0.5, R["xo"] > 0.5
    on = R["nd"][:, 0] > 0                    # a sensor is on: otherwise routed = rgb-only and d = 0 exactly
    groups = {"all": np.ones(len(st), bool), "neither": ~st & ~xo, "see-through": st, "crossover": xo}
    summary = {"checkpoint": str(checkpoint), "split": split, "n_seq": int(len(st)), "n_act": int(n_a),
               "n_noise": n_noise,
               "n_episodes": int(len(set(episode.tolist()))), "steps": steps, "scales": scales, "groups": {}}
    for name, m in groups.items():
        if not m.any():
            continue
        ma = m[:n_a]
        mo = m & on                           # size, direction and s* only where d is not identically 0
        if not mo.any():
            continue
        gsum = {"n_seq": int(m.sum()), "n_episodes": int(len(set(episode[m].tolist()))),
                "n_seq_sensor_on": int(mo.sum()), "w_mean": [round(float(v), 3) for v in R["w"][m].mean(0)],
                "rel_size_median": np.median(R["nd"][mo] / R["ner"][mo], 0).round(4).tolist(),
                "cos_median": np.median(R["dot"][mo] / (R["nd"][mo] * R["nr"][mo]), 0).round(4).tolist(),
                "s_star": (R["dot"][mo].sum(0) / (R["nd"][mo] ** 2).sum(0)).round(3).tolist(),
                "s_star_all_steps": round(float(R["dot"][mo].sum() / (R["nd"][mo] ** 2).sum()), 3),
                "noise_err": R["err"][m].mean(0).round(5).tolist()}
        if ma.any():
            gsum.update(n_act=int(ma.sum()), **{f"act_{k}": v[ma].mean(0).round(5).tolist() for k, v in A.items()
                                                if k not in ("pred", "gt")})
        summary["groups"][name] = gsum
        print(f"\n[{name}] {gsum['n_seq']} sequences ({gsum['n_seq_sensor_on']} with a sensor on), "
              f"{gsum['n_episodes']} episodes, mean w {gsum['w_mean']}")
        print("  step            " + "  ".join(f"{t:>7d}" for t in steps))
        print("  |d|/|e_r| med   " + "  ".join(f"{v:7.3f}" for v in gsum["rel_size_median"]))
        print("  cos(d, eps-e_0) " + "  ".join(f"{v:7.3f}" for v in gsum["cos_median"]))
        print("  s*              " + "  ".join(f"{v:7.2f}" for v in gsum["s_star"]) + f"   all steps {gsum['s_star_all_steps']:.2f}")
        if ma.any():
            print("  scale                " + "  ".join(f"{s:>8g}" for s in scales))
            print("  action err (x1e3)    " + "  ".join(f"{1e3 * v:8.3f}" for v in gsum["act_err"]) + f"   ({gsum['n_act']} sequences)")
            print("  position err (x1e3)  " + "  ".join(f"{1e3 * v:8.3f}" for v in gsum["act_err_pos"]))
    out.write_text(json.dumps(summary, indent=1))
    print(f"\nsaved {out} and {out.with_suffix('.npz')}", flush=True)


if __name__ == "__main__":
    main()
