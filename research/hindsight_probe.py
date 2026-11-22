"""
Feasibility probe for a label-free fusion method: can each sensor learn, from the demos alone, where the grasp will
happen and how far to trust itself?

Supervision comes from the demo itself (no cell, no visibility flag, no mask): the arm joints at the first step that
closes the gripper, turned into the tool position by the arm's own kinematics ("hindsight grasp point").
Per sensor (one image, no joints):
  1. K-fold cross-fitting: a network regresses the grasp point; every training episode gets a prediction from a
     network that never saw it (out-of-fold).
  2. A second network of the same form predicts the size of that out-of-fold error from the image (Gaussian NLL):
     the sensor's own estimate of how wrong it is on episodes it has not memorized. Its scale is set on training
     episodes held back from both networks.
  3. Further passes: a two-component mixture on the out-of-fold error of each training episode says how likely the
     episode is one the sensor can read; the next pass weights the regression by that (bounded weights), so episodes
     the sensor cannot read stop pulling on it.
Fusion: precision-weighted mean of the three sensors' grasp points.
Evaluated on demos of other seeds; the cells are used for the report only.

  python research/hindsight_probe.py TRAIN.zarr EVAL.zarr [--frames 4] [--folds 4] [--epochs 120] [-d cuda:0] [--out FILE.json]
"""
import argparse
import json

import mujoco
import numpy as np
import torch
import torch.nn as nn
import torchvision
import zarr

ARM = "sim/assets/gen3_lite/gen3_lite.xml"
SENSORS = {"rgb": 3, "depth": 1, "thermal": 1}
FLAGS = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (1, 0, 1), (0, 1, 1), (2, 1, 1), (1, 2, 1), (1, 1, 2), (2, 2, 2)]   # per cell: rgb, depth, thermal; 0 shows, 1 blind, 2 faint
NAMES = ["rgb+depth+thermal", "depth+thermal", "rgb+thermal", "rgb+depth", "thermal", "depth", "rgb", "rgb faint", "depth faint", "thermal faint", "all faint"]


def fk(qs):
    m = mujoco.MjModel.from_xml_path(ARM)
    d, bid, out = mujoco.MjData(m), m.body("tool_frame").id, []
    for q in qs:
        d.qpos[:6] = q
        mujoco.mj_kinematics(m, d)
        out.append(d.xpos[bid].copy())
    return np.array(out, np.float32)


def load(path, n_frames):
    z = zarr.open(path, "r")
    ends = z["meta/episode_ends"][:]
    starts = np.r_[0, ends[:-1]]
    act = z["data/action"][:]
    close = [s + int(np.argmax(act[s:t, 6] > 0)) for s, t in zip(starts, ends)]
    goal = fk(act[close, :6])                                   # (episodes, 3) m
    idx = np.concatenate([np.arange(s, s + n_frames) for s in starts])
    x = {k: torch.from_numpy(z["data"][k].get_orthogonal_selection((idx,))).permute(0, 3, 1, 2).contiguous() for k in SENSORS}
    ep = np.repeat(np.arange(len(ends)), n_frames)
    return x, goal, ep, z["meta/ep_cell"][:]


class Net(nn.Module):
    """ResNet-18 trunk to stride 8, spatial soft-argmax keypoints, MLP."""

    def __init__(self, channels, out=3, keypoints=32):
        super().__init__()
        r = torchvision.models.resnet18(weights=None)
        r.conv1 = nn.Conv2d(channels, 64, 7, 2, 3, bias=False)
        self.trunk = nn.Sequential(r.conv1, r.bn1, r.relu, r.layer1, r.layer2, nn.Conv2d(128, keypoints, 1))
        self.head = nn.Sequential(nn.Linear(2 * keypoints, 256), nn.ReLU(), nn.Linear(256, out))

    def forward(self, x):
        f = self.trunk(x.float() / 255.0)
        b, k, h, w = f.shape
        p = f.flatten(2).softmax(-1).view(b, k, h, w)
        ys = torch.linspace(-1, 1, h, device=x.device)
        xs = torch.linspace(-1, 1, w, device=x.device)
        kp = torch.stack([(p.sum(3) * ys).sum(2), (p.sum(2) * xs).sum(2)], -1)
        return self.head(kp.flatten(1))


def fit(x, y, loss_fn, channels, epochs, device, seed, w=None):
    torch.manual_seed(seed)
    net = Net(channels).to(device)
    opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 1e-3, total_steps=epochs * ((len(x) + 63) // 64))
    x, y = x.to(device), y.to(device)
    w = torch.ones(len(x), device=device) if w is None else w.to(device)
    for _ in range(epochs):
        for b in torch.randperm(len(x), device=device).split(64):
            loss = loss_fn(net(x[b]), y[b], w[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
    return net.eval()


@torch.no_grad()
def predict(net, x, device):
    return torch.cat([net(x[i:i + 512].to(device)) for i in range(0, len(x), 512)]).cpu()


def auroc(score, positive):
    """Probability that a positive has a higher score than a negative."""
    s, p = np.asarray(score), np.asarray(positive, bool)
    order = s.argsort()
    rank = np.empty(len(s))
    rank[order] = np.arange(1, len(s) + 1)
    return float((rank[p].sum() - p.sum() * (p.sum() + 1) / 2) / (p.sum() * (~p).sum()))


def mixture(x, steps=100):
    """Two Gaussians on x (1-D): responsibility of the component with the lower mean, and the two means."""
    m = np.quantile(x, [0.25, 0.75]); sd = np.full(2, x.std() / 2 + 1e-6); pi = np.array([0.5, 0.5])
    for _ in range(steps):
        d = pi * np.exp(-0.5 * ((x[:, None] - m) / sd) ** 2) / sd
        r = d / d.sum(1, keepdims=True)
        n = r.sum(0) + 1e-9
        m, pi = (r * x[:, None]).sum(0) / n, n / len(x)
        sd = np.sqrt((r * (x[:, None] - m) ** 2).sum(0) / n) + 1e-3
    lo = int(np.argmin(m))
    return r[:, lo], m[lo], m[1 - lo], pi[lo]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("train")
    ap.add_argument("eval")
    ap.add_argument("--frames", type=int, default=4)
    ap.add_argument("--folds", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--iters", type=int, default=3, help="passes; from the second on the regression is weighted by the mixture")
    ap.add_argument("-d", "--device", default="cuda:0")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    xt, gt, ept, cells_t = load(a.train, a.frames)
    xe, ge, epe, cells = load(a.eval, 1)                   # first frame only: the arm has not moved
    mean, std = gt.mean(0), gt.std(0)
    yt = torch.from_numpy((gt[ept] - mean) / std)
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(gt))
    cal_ep = np.zeros(len(gt), bool); cal_ep[perm[:int(0.15 * len(gt))]] = True    # held back: sets the scale of sigma
    fold = np.where(cal_ep, -1, rng.permutation(len(gt)) % a.folds)
    fit_fr, cal_fr = torch.from_numpy(~cal_ep[ept]), torch.from_numpy(cal_ep[ept])
    mse = lambda p, y, w: (w[:, None] * (p - y) ** 2).mean()
    nll = lambda p, r, w: (p + r * torch.exp(-p)).mean()   # p = log variance
    err = lambda m: 100 * np.linalg.norm((m - ge)[:, :2], axis=1)   # horizontal error, cm
    sig = lambda v: 100 * np.sqrt(v[:, :2].sum(1))
    f2 = lambda e: f"{np.median(e):.1f} ({(e < 2).mean():.2f})"
    cell_ep, flags = cells[epe], np.array(FLAGS)
    report = {}
    w_ep = {k: np.ones(len(gt), np.float32) for k in SENSORS}
    for it in range(a.iters):
        mu, var, info = {}, {}, {}
        for i, (k, c) in enumerate(SENSORS.items()):
            oof, nets, w = torch.zeros_like(yt), [], torch.from_numpy(w_ep[k][ept])
            for f in range(a.folds):
                tr = torch.from_numpy((fold[ept] != f) & ~cal_ep[ept])
                te = torch.from_numpy(fold[ept] == f)
                nets.append(fit(xt[k][tr], yt[tr], mse, c, a.epochs, a.device, f, w[tr] / w[tr].mean()))
                oof[te] = predict(nets[-1], xt[k][te], a.device)
            ens = lambda x: torch.stack([predict(n, x, a.device) for n in nets]).mean(0)
            oof[cal_fr] = ens(xt[k][cal_fr])
            res2 = (oof - yt) ** 2                              # squared error on episodes the network did not train on
            vnet = fit(xt[k][fit_fr], res2[fit_fr], nll, c, a.epochs, a.device, 100)
            ratio = res2[cal_fr] / torch.exp(predict(vnet, xt[k][cal_fr], a.device))
            scale = (ratio.median(0).values / 0.4549).numpy()   # the median of a chi-square with 1 dof is 0.4549
            mu[k] = ens(xe[k]).numpy() * std + mean
            var[k] = torch.exp(predict(vnet, xe[k], a.device)).numpy() * scale * std ** 2
            # per training episode: out-of-fold error -> how likely the sensor can read the episode
            e_ep = np.sqrt(np.bincount(ept, res2.mean(1).numpy()) / a.frames)
            r, lo, hi, share = mixture(np.log(e_ep[~cal_ep]))
            shows_t = np.array(FLAGS)[cells_t, i][~cal_ep]
            info[k] = dict(scale=[float(x) for x in scale], mix_low_cm=float(np.exp(lo)), mix_high=float(np.exp(hi)), share_low=float(share),
                           auroc_train=auroc(r[shows_t != 2], (shows_t == 0)[shows_t != 2]),
                           weight_shows=float(r[shows_t == 0].mean()), weight_faint=float(r[shows_t == 2].mean()), weight_blind=float(r[shows_t == 1].mean()))
            nw = np.ones(len(gt), np.float32); nw[~cal_ep] = np.clip(r, 0.02, 1.0)
            w_ep[k] = nw
            print(f"pass {it + 1} {k}: mixture share {share:.2f}; weight of the next pass where the sensor shows / faint / blind: "
                  f"{info[k]['weight_shows']:.2f} / {info[k]['weight_faint']:.2f} / {info[k]['weight_blind']:.2f}; sigma scale {np.sqrt(scale).round(2).tolist()}", flush=True)
        prec = {k: 1 / var[k] for k in SENSORS}
        fused = sum(prec[k] * mu[k] for k in SENSORS) / sum(prec.values())
        best = np.stack([sig(var[k]) for k in SENSORS]).argmin(0)        # the sensor that trusts itself most
        pick = np.stack([mu[k] for k in SENSORS])[best, np.arange(len(ge))]
        print(f"\n=== pass {it + 1}; held-out demos: {len(ge)}; horizontal error of the predicted grasp point, median cm (share within 2 cm)")
        print("| cell | shows the glass | n | rgb | depth | thermal | fused by predicted precision | most confident sensor | plain mean |")
        print("|---|---|---|---|---|---|---|---|---|")
        rows = []
        for c in [*sorted(set(cell_ep.tolist())), None]:
            m = np.ones(len(ge), bool) if c is None else cell_ep == c
            r_ = dict(**{k: err(mu[k])[m] for k in SENSORS}, fused=err(fused)[m], pick=err(pick)[m], plain=err(sum(mu.values()) / 3)[m])
            print(f"| {'all' if c is None else c} | {'' if c is None else NAMES[c]} | {int(m.sum())} | " + " | ".join(f2(v) for v in r_.values()) + " |")
            rows.append(dict(cell=c, n=int(m.sum()), **{k: (float(np.median(v)), float((v < 2).mean())) for k, v in r_.items()}))
        print("own error predicted by each sensor (sigma, cm) and actual error, median: shows / faint / blind; AUROC shows against blind")
        for i, k in enumerate(SENSORS):
            fl = flags[cell_ep, i]
            s_, e_ = sig(var[k]), err(mu[k])
            g = lambda v, x: float(np.median(v[fl == x])) if (fl == x).any() else float("nan")
            info[k].update(sigma=[g(s_, 0), g(s_, 2), g(s_, 1)], error=[g(e_, 0), g(e_, 2), g(e_, 1)], auroc=auroc(-s_[fl != 2], (fl == 0)[fl != 2]))
            print(f"  {k}: sigma {g(s_, 0):.1f} / {g(s_, 2):.1f} / {g(s_, 1):.1f}; actual {g(e_, 0):.1f} / {g(e_, 2):.1f} / {g(e_, 1):.1f}; AUROC {info[k]['auroc']:.3f}", flush=True)
        report[f"pass{it + 1}"] = dict(rows=rows, sensors=info)
    print("\n=== reference that uses the cells: trained only on demos where the sensor shows the glass; error where it shows it fully / faintly")
    ft = np.array(FLAGS)[cells_t[ept]]
    for i, (k, c) in enumerate(SENSORS.items()):
        tr = torch.from_numpy(ft[:, i] != 1)
        m = predict(fit(xt[k][tr], yt[tr], mse, c, a.epochs, a.device, 0), xe[k], a.device).numpy() * std + mean
        fl = flags[cell_ep, i]
        report.setdefault("oracle", {})[k] = [float(np.median(err(m)[fl == 0])), float((err(m)[fl == 0] < 2).mean()),
                                              float(np.median(err(m)[fl == 2])) if (fl == 2).any() else None]
        print(f"  {k}: {f2(err(m)[fl == 0])}" + (f" / {f2(err(m)[fl == 2])}" if (fl == 2).any() else ""), flush=True)
    if a.out:
        json.dump(dict(train=a.train, eval=a.eval, frames=a.frames, folds=a.folds, epochs=a.epochs, **report), open(a.out, "w"), indent=1)
