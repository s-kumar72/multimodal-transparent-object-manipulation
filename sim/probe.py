"""
Information probe: does each modality carry what the policy needs, before training any policy?
A small CNN sees the first frame (96x96) and predicts the target's image position and its object type
(the type sets the grasp height). One CNN per level x input combination; tested on the level's test set (a).

  python sim/probe.py collect --levels L0 L1 L2 L3 L4 --n_train 3000 --n_test 600 --workers 4
  python sim/probe.py train --levels L0 L1 L2 L3 L4      # GPU if available, ~1 min per CNN on a T4
Frames go to data/probe/<level>.npz; the table to results/hard_scenario/probe.md.
"""
import argparse
import multiprocessing as mp
from pathlib import Path

import numpy as np

INPUTS = {"rgb": ["rgb"], "depth": ["depth"], "thermal": ["thermal"],
          "rgbd": ["rgb", "depth"], "rgbt": ["rgb", "thermal"], "rgbdt": ["rgb", "depth", "thermal"]}

def target_pixel(env):
    """Image position (96x96 pixels) of the target's centre at its grasp height."""
    from env import IMG, OBJECTS
    s, m, d = env.sensors, env.model, env.data
    p = d.body(env.obj).xpos + [0, 0, OBJECTS[env.obj]]
    R = d.cam_xmat[s.cam_id].reshape(3, 3)
    pc = R.T @ (p - d.cam_xpos[s.cam_id])
    h, w = s.r.height, s.r.width
    u = w / 2 + s.focal * pc[0] / -pc[2]
    v = h / 2 - s.focal * pc[1] / -pc[2]
    return np.array([u * IMG / w, v * IMG / h], np.float32)

def _collect(job):
    level, split, seeds = job
    from config import load_config
    from env import GlassPickEnv, OBJECT_NAMES
    from testsets import manifest, test_set
    if split == "test":
        cfg, _, objects = test_set(level, "a_in_distribution")
    else:
        cfg, objects = load_config(manifest()["levels"][level]), None
    env = GlassPickEnv(config=cfg)
    out = {k: [] for k in ["rgb", "depth", "thermal", "xy", "obj"]}
    for s in seeds:
        obs = env.reset(s, obj=objects[s % len(objects)] if objects else None)
        for k in ["rgb", "depth", "thermal"]:
            out[k].append(obs[k])
        out["xy"].append(target_pixel(env))
        out["obj"].append(OBJECT_NAMES.index(env.obj))
    env.close()
    return {k: np.stack(v) for k, v in out.items()}

def collect(args):
    from testsets import manifest
    start = manifest()["seed_start"]
    for level in args.levels:
        jobs = []
        for split, seeds in [("train", range(args.n_train)), ("test", range(start, start + args.n_test))]:
            seeds = list(seeds)
            jobs += [(level, split, seeds[i::args.workers * 4]) for i in range(args.workers * 4)]
        with mp.get_context("fork").Pool(args.workers) as pool:
            parts = pool.map(_collect, jobs)
        half = len(jobs) // 2
        data = {f"{split}_{k}": np.concatenate([p[k] for p in ps])
                for split, ps in [("train", parts[:half]), ("test", parts[half:])] for k in parts[0]}
        Path("data/probe").mkdir(parents=True, exist_ok=True)
        np.savez_compressed(f"data/probe/{level}.npz", **data)
        print(f"{level}: {len(data['train_obj'])} train / {len(data['test_obj'])} test frames", flush=True)

def make_net(c_in):
    import torch.nn as nn
    def block(i, o):
        return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.GroupNorm(8, o), nn.ReLU(), nn.MaxPool2d(2))
    return nn.Sequential(block(c_in, 32), block(32, 64), block(64, 128), block(128, 128),  # 96 -> 6
                         nn.Flatten(), nn.Linear(128 * 36, 256), nn.ReLU(), nn.Linear(256, 2 + 5))

def stack(d, split, keys):
    x = np.concatenate([d[f"{split}_{k}"] for k in keys], -1).astype(np.float32) / 255
    return np.moveaxis(x, -1, 1)

def train_one(d, keys, epochs, device, seed=0):
    import torch
    import torch.nn.functional as F
    torch.manual_seed(seed)
    xtr, xte = (torch.from_numpy(stack(d, s, keys)) for s in ("train", "test"))
    ytr, yte = (torch.from_numpy(d[f"{s}_xy"] / 96) for s in ("train", "test"))
    otr, ote = (torch.from_numpy(d[f"{s}_obj"]).long() for s in ("train", "test"))
    net = make_net(xtr.shape[1]).to(device)
    opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=epochs * ((len(xtr) + 63) // 64))
    for _ in range(epochs):
        net.train()
        for idx in torch.randperm(len(xtr)).split(64):
            x, y, o = xtr[idx].to(device), ytr[idx].to(device), otr[idx].to(device)
            # shift augmentation like the policy's random crop (+-6 px), labels shifted with the image
            dx, dy = np.random.randint(-6, 7, 2)
            x = torch.roll(x, (int(dy), int(dx)), (2, 3))
            y = y + torch.tensor([dx, dy], device=device) / 96
            out = net(x)
            loss = F.smooth_l1_loss(out[:, :2], y, beta=0.02) + 0.2 * F.cross_entropy(out[:, 2:], o)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
    net.eval()
    with torch.no_grad():
        out = torch.cat([net(x.to(device)).cpu() for x in xte.split(256)])
    err = ((out[:, :2] - yte) * 96).norm(dim=1).numpy()
    acc = (out[:, 2:].argmax(1) == ote).float().mean().item()
    return dict(err_median=float(np.median(err)), within3=float((err < 3).mean()), obj_acc=acc)

def train(args):
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rows = []
    for level in args.levels:
        d = dict(np.load(f"data/probe/{level}.npz"))
        for name in args.inputs:
            r = train_one(d, INPUTS[name], args.epochs, device)
            rows.append((level, name, r))
            print(f"{level} {name:7s} position error median {r['err_median']:.2f} px, within 3 px {r['within3']:.2f}, "
                  f"object accuracy {r['obj_acc']:.2f}", flush=True)
    lines = ["| level | input | position error median (px) | within 3 px | object type accuracy |", "|---|---|---|---|---|"]
    lines += [f"| {lv} | {n} | {r['err_median']:.2f} | {r['within3']:.2f} | {r['obj_acc']:.2f} |" for lv, n, r in rows]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(lines) + "\n")
    print("Saved", args.out)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["collect", "train"])
    p.add_argument("--levels", nargs="+", default=["L0", "L1", "L2", "L3", "L4"])
    p.add_argument("--inputs", nargs="+", default=["rgb", "rgbd", "rgbdt", "depth", "thermal"])
    p.add_argument("--n_train", type=int, default=3000)
    p.add_argument("--n_test", type=int, default=600)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--out", default="results/hard_scenario/probe.md")
    args = p.parse_args()
    collect(args) if args.stage == "collect" else train(args)
