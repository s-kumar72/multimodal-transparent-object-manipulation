"""
Fits the routing gate of a route=learned policy after training, with the policy frozen.

The gate loss of gate_train=heldout (policy/gated_policy.py) is evaluated on the demos the policy was not fitted to
(the dataset's gate_split): for every sample of these demos the denoising loss in the four states of the two sensors
(depth on / off x thermal on / off), with the same noise and timesteps in the four states, averaged over --draws
draws. A fresh gate (the initialization of training: p = 0.5 for every input) is then trained on these cached losses
alone, loss = mean_n sum_m p_m (L_on,m - L_off,m), with its own optimizer and learning rate. Nothing else in the
checkpoint changes. No labels enter the fit: the simulator's episode labels and the thresholded flags of
route=visible appear in the diagnostics only.

  python policy/fit_gate.py -c data/runs/<run>/checkpoints/latest.ckpt -o results/.../gate_fit/<run>
writes <run>.pt ({"gate": state dict of the encoder's gate}; evaluate with eval.py --gate_state <run>.pt) and
<run>.json (diagnostics: the gate of the checkpoint, the fresh gate and the fitted gate on the same cached losses).
Run from the repo root with the PYTHONPATH that policy/train.sh uses. The EMA weights are used when present, as in
eval.py.
"""
import copy
import json
import pathlib

import click
import numpy as np
import torch

SENSORS = ("depth", "thermal")


def gate_probs(gate, cue):
    """cue (..., C) as the encoder sees it -> (..., 2) probabilities [depth, thermal]. As in
    GatedObsEncoder._learned_route, the two thresholded flags (last two entries) are not an input of the gate."""
    c = cue.clone()
    c[..., -2:] = 0
    return torch.cat([torch.sigmoid(gate[m](c)) for m in SENSORS], -1)


def gate_objective(p, L):
    """p (n, 2): gate probabilities [depth, thermal] per sample; L (n, 2, 2): denoising loss with depth d and thermal
    t, L[:, d, t], index 1 = on. Returns the gate loss of gate_train=heldout and the per-sample loss differences
    (sensor on - sensor off; the other sensor weighted by its current probability), depth and thermal."""
    pd, pt = p[:, 0].detach(), p[:, 1].detach()
    delta_d = pt * (L[:, 1, 1] - L[:, 0, 1]) + (1 - pt) * (L[:, 1, 0] - L[:, 0, 0])
    delta_t = pd * (L[:, 1, 1] - L[:, 1, 0]) + (1 - pd) * (L[:, 0, 1] - L[:, 0, 0])
    return (p[:, 0] * delta_d + p[:, 1] * delta_t).mean(), delta_d, delta_t


def fresh_gate(gate, seed=0):
    """Re-initializes the depth and thermal gates the way GatedObsEncoder builds them (last layer zero: p = 0.5)."""
    torch.manual_seed(seed)
    for m in SENSORS:
        for layer in gate[m]:
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()
        torch.nn.init.zeros_(gate[m][-1].weight)
        torch.nn.init.zeros_(gate[m][-1].bias)


def fit_gate(gate, cue, L, steps=2000, lr=1e-3):
    """Full-batch Adam on the cached losses; cue (n, To, C), L (n, 2, 2). The probability of a sample is the mean
    over its To observation steps, as in training. Returns the objective at every step."""
    params = [q for m in SENSORS for q in gate[m].parameters()]
    for q in params:
        q.requires_grad_(True)
    opt = torch.optim.Adam(params, lr=lr)
    hist = []
    for _ in range(steps):
        opt.zero_grad()
        loss, _, _ = gate_objective(gate_probs(gate, cue).mean(1), L)
        loss.backward()
        opt.step()
        hist.append(float(loss))
    return hist


@torch.no_grad()
def collect(policy, loader, draws, device):
    """Denoising loss of every sample of the loader in the four sensor states -> dict of CPU tensors:
    cue (n, To, C) as the encoder sees it, flags (n, To, 2) thresholded flags of route=visible, L (n, 2, 2), and the
    episode labels seethrough / crossover (n,) when the dataset has them."""
    from diffusion_policy.common.pytorch_util import dict_apply
    enc, To = policy.obs_encoder, policy.n_obs_steps
    out = {k: [] for k in ("cue", "flags", "L", "seethrough", "crossover")}
    try:
        for batch in loader:
            batch = dict_apply(batch, lambda x: x.to(device))
            n = batch["action"].shape[0]
            L = torch.zeros(n, 2, 2, device=device)
            for _ in range(draws):
                noise = steps = None                       # drawn in the first state, reused in the other three
                for d in (1, 0):
                    for t in (1, 0):
                        enc.route_force = torch.tensor([[d, t]], dtype=torch.float32, device=device).expand(n * To, 2)
                        per, noise, steps = policy._ddpm_per_sample(batch, noise, steps)
                        L[:, d, t] += per / draws
            out["cue"].append(enc.last_cue.view(n, To, -1).cpu())
            out["flags"].append(batch["obs"][enc.cue_key][:, :To, -2:].cpu())
            out["L"].append(L.cpu())
            meta = batch.get("meta", {})
            if "ep_glass_seethrough" in meta and "ep_is_crossover" in meta:
                out["seethrough"].append(meta["ep_glass_seethrough"].cpu())
                out["crossover"].append(meta["ep_is_crossover"].cpu())
    finally:
        enc.route_force = None
    return {k: torch.cat(v) for k, v in out.items() if v}


def describe(gate, data):
    """Diagnostics of a gate on the cached losses: objective, loss differences and decisions against the flags of
    the rule (per observation step) and against the episode labels (per sample)."""
    with torch.no_grad():
        p_steps = gate_probs(gate, data["cue"])                     # (n, To, 2)
        p = p_steps.mean(1)
        obj, dd, dt = gate_objective(p, data["L"])
    res = {"objective": float(obj), "n_samples": int(p.shape[0])}
    flag = data["flags"] > 0.5                                       # (n, To, 2)
    on = p_steps > 0.5
    labels = None
    if "seethrough" in data:
        labels = torch.stack([data["seethrough"] < 0.5, data["crossover"] < 0.5], -1)      # (n, 2): sensor shows the glass
    for j, (m, delta) in enumerate(zip(SENSORS, (dd, dt))):
        f, o = flag[..., j], on[..., j]
        r = {"on": float(o.float().mean()), "visible_by_rule": float(f.float().mean()),
             "agreement_with_rule": float((o == f).float().mean()),
             "on_where_rule_off": float(o[~f].float().mean()) if (~f).any() else None,
             "off_where_rule_on": float((~o)[f].float().mean()) if f.any() else None,
             "delta_mean": float(delta.mean())}
        fs = f[:, -1]                                                # the rule's flag at the last observation step
        for name, sel in (("rule_on", fs), ("rule_off", ~fs)):
            if sel.any():
                r[f"delta_{name}"] = float(delta[sel].mean())
                r[f"p_{name}"] = float(p[sel, j].mean())
        if labels is not None:
            v = labels[:, j]
            for name, sel in (("label_visible", v), ("label_not_visible", ~v)):
                if sel.any():
                    r[f"delta_{name}"] = float(delta[sel].mean())
                    r[f"p_{name}"] = float(p[sel, j].mean())
                    r[f"on_{name}"] = float(o[sel].float().mean())
        res[m] = r
    return res


@click.command()
@click.option("-c", "--checkpoint", required=True)
@click.option("-o", "--output", required=True, help="path without extension: writes .pt and .json")
@click.option("--draws", type=int, default=8, show_default=True, help="noise / timestep draws per sample and state")
@click.option("--steps", type=int, default=2000, show_default=True, help="optimizer steps of the gate fit")
@click.option("--lr", type=float, default=1e-3, show_default=True)
@click.option("--seed", type=int, default=0, show_default=True)
@click.option("--batch_size", type=int, default=128, show_default=True)
@click.option("--num_workers", type=int, default=4, show_default=True)
@click.option("--max_samples", type=int, default=0, help="use only the first N held-out samples (smoke tests)")
@click.option("-d", "--device", default="cuda:0", show_default=True)
def main(checkpoint, output, draws, steps, lr, seed, batch_size, num_workers, max_samples, device):
    import dill
    import hydra
    from omegaconf import OmegaConf
    OmegaConf.register_new_resolver("eval", eval, replace=True)
    payload = torch.load(open(checkpoint, "rb"), pickle_module=dill, weights_only=False, map_location="cpu")
    cfg = payload["cfg"]
    out = pathlib.Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    workspace = hydra.utils.get_class(cfg._target_)(cfg, output_dir=str(out.parent))
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    ema = "ema_model" in payload["state_dicts"]
    policy = workspace.ema_model if ema else workspace.model
    enc = policy.obs_encoder
    assert getattr(enc, "route", "none") == "learned", "the checkpoint needs an encoder with route=learned"
    assert cfg.task.dataset.get("gate_split", 0), "the checkpoint's dataset has no gate_split: no held-out demos"
    policy.to(torch.device(device))
    policy.eval()                                          # center crop, no rgb-only draw: the states differ by the gate only
    for q in policy.parameters():
        q.requires_grad_(False)

    torch.manual_seed(seed)
    np.random.seed(seed)
    dataset = hydra.utils.instantiate(cfg.task.dataset)
    split = dataset.ep_meta["ep_gate_split"]
    held = [i for i in range(len(dataset)) if split[dataset._episode_of(i)] > 0.5]
    episodes = sorted({dataset._episode_of(i) for i in held})
    if max_samples:
        held = held[:max_samples]
    loader = torch.utils.data.DataLoader(torch.utils.data.Subset(dataset, held), batch_size=batch_size, shuffle=False,
                                         num_workers=num_workers)
    print(f"{len(held)} samples of {len(episodes)} held-out demos, {draws} draws x 4 states", flush=True)
    data = collect(policy, loader, draws, device)
    data = {k: v.to(device) for k, v in data.items()}

    L = data["L"]
    report = {"checkpoint": str(checkpoint), "weights": "ema" if ema else "model", "n_samples": len(held),
              "n_demos": len(episodes), "draws": draws, "steps": steps, "lr": lr, "seed": seed,
              "loss_by_state": {f"depth{d}_thermal{t}": float(L[:, d, t].mean()) for d in (1, 0) for t in (1, 0)},
              "checkpoint_gate": describe(enc.gate, data)}
    fresh_gate(enc.gate, seed)
    report["fresh_gate"] = describe(enc.gate, data)
    hist = fit_gate(enc.gate, data["cue"], L, steps=steps, lr=lr)
    report["fitted_gate"] = describe(enc.gate, data)
    report["objective_curve"] = [hist[i] for i in sorted({0, *range(steps // 20 - 1, steps, max(steps // 20, 1)), steps - 1})]

    state = {k: v.detach().cpu() for k, v in copy.deepcopy(enc.gate.state_dict()).items()}
    torch.save({"gate": state, "checkpoint": str(checkpoint), "draws": draws, "steps": steps, "lr": lr, "seed": seed},
               str(out) + ".pt")
    pathlib.Path(str(out) + ".json").write_text(json.dumps(report, indent=2))
    for name in ("checkpoint_gate", "fresh_gate", "fitted_gate"):
        r = report[name]
        print(f"{name}: objective {r['objective']:+.6f}; " + "; ".join(
            f"{m} on {r[m]['on']:.3f} (rule {r[m]['visible_by_rule']:.3f}), agreement {r[m]['agreement_with_rule']:.3f}"
            for m in SENSORS), flush=True)
    print(f"saved {out}.pt and {out}.json", flush=True)


if __name__ == "__main__":
    main()
