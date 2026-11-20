"""
Feasibility probe, second form: per-sensor grasp heatmaps from the demos alone, fused as a product.

Label from the demo itself: the tool position at the first step that closes the gripper (arm kinematics), projected
into the image with the camera calibration ("the demonstrator grasped at this pixel"). No cell, flag or mask.
Per sensor a small fully convolutional network (local receptive field, no global pooling, no coordinates) outputs a
distribution over the 96 x 96 pixels of where the grasp will be, trained with cross-entropy and image shifts.
A sensor that does not show the glass has nothing local to go on, so its map should stay flat instead of memorizing.
Fusion: sum of the log maps (product of the sensors' distributions), then the arg max.
Evaluated on demos of other seeds, first frame; the cells are used for the report only.

  python research/heatmap_probe.py TRAIN.zarr EVAL.zarr [--level L0_vis] [--frames 4] [--epochs 60] [-d cuda:0] [--out FILE.json]
"""
import argparse
import json
import sys

import mujoco
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import zarr

sys.path.insert(0, "sim")
ARM = "sim/assets/gen3_lite/gen3_lite.xml"
SENSORS = {"rgb": 3, "depth": 1, "thermal": 1}
FLAGS = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (1, 0, 1), (0, 1, 1), (2, 1, 1), (1, 2, 1), (1, 1, 2), (2, 2, 2)]
NAMES = ["rgb+depth+thermal", "depth+thermal", "rgb+thermal", "rgb+depth", "thermal", "depth", "rgb", "rgb faint", "depth faint", "thermal faint", "all faint"]
S = 96


def fk(qs):
    m = mujoco.MjModel.from_xml_path(ARM)
    d, bid, out = mujoco.MjData(m), m.body("tool_frame").id, []
    for q in qs:
        d.qpos[:6] = q
        mujoco.mj_kinematics(m, d)
        out.append(d.xpos[bid].copy())
    return np.array(out, np.float32)


def camera(level):
    """Projection of world points to the 96 x 96 image (fixed camera: its pose and focal length)."""
    from env import GlassPickEnv
    from testsets import manifest
    env = GlassPickEnv(config=manifest()["levels"][level])
    env.reset(0)
    s, d = env.sensors, env.data
    R, t, f, h, w = d.cam_xmat[s.cam_id].reshape(3, 3).copy(), d.cam_xpos[s.cam_id].copy(), s.focal, s.r.height, s.r.width
    env.close()

    def project(p):
        pc = (p - t) @ R
        return np.stack([(w / 2 + f * pc[:, 0] / -pc[:, 2]) * S / w, (h / 2 - f * pc[:, 1] / -pc[:, 2]) * S / h], -1).astype(np.float32)
    return project


def load(path, n_frames, project):
    z = zarr.open(path, "r")
    ends = z["meta/episode_ends"][:]
    starts = np.r_[0, ends[:-1]]
    act = z["data/action"][:]
    close = [s + int(np.argmax(act[s:t, 6] > 0)) for s, t in zip(starts, ends)]
    goal = fk(act[close, :6])
    idx = np.concatenate([np.arange(s, s + n_frames) for s in starts])
    x = {k: torch.from_numpy(z["data"][k].get_orthogonal_selection((idx,))).permute(0, 3, 1, 2).contiguous() for k in SENSORS}
    return x, goal, project(goal), np.repeat(np.arange(len(ends)), n_frames), z["meta/ep_cell"][:]


def block(i, o, stride=1, dilation=1):
    return nn.Sequential(nn.Conv2d(i, o, 3, stride, dilation, dilation), nn.GroupNorm(8, o), nn.ReLU())


class FCN(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.e1 = nn.Sequential(block(c, 32), block(32, 32))
        self.e2 = nn.Sequential(block(32, 64, 2), block(64, 64))
        self.e3 = nn.Sequential(block(64, 128, 2), block(128, 128, dilation=2), block(128, 128, dilation=2))
        self.d2 = block(128 + 64, 64)
        self.d1 = block(64 + 32, 32)
        self.out = nn.Conv2d(32, 1, 1)

    def forward(self, x):
        x = x.float() / 255.0
        a = self.e1(x)
        b = self.e2(a)
        c = self.e3(b)
        b = self.d2(torch.cat([F.interpolate(c, scale_factor=2, mode="bilinear", align_corners=False), b], 1))
        a = self.d1(torch.cat([F.interpolate(b, scale_factor=2, mode="bilinear", align_corners=False), a], 1))
        return self.out(a).flatten(1)                       # logits over the pixels


def target_map(uv, sigma=1.5):
    ys, xs = torch.meshgrid(torch.arange(S, device=uv.device), torch.arange(S, device=uv.device), indexing="ij")
    g = torch.exp(-((xs[None] - uv[:, 0, None, None]) ** 2 + (ys[None] - uv[:, 1, None, None]) ** 2) / (2 * sigma ** 2))
    return (g / g.sum((1, 2), keepdim=True).clamp_min(1e-9)).flatten(1)


def shift(x, uv, k=8):
    """The same random shift of the image (edge padded) and of the label."""
    dx, dy = np.random.randint(-k, k + 1, 2)
    xp = F.pad(x.float(), (k, k, k, k), mode="replicate")
    return xp[..., k - dy:k - dy + S, k - dx:k - dx + S], uv + torch.tensor([dx, dy], device=uv.device)


def fit(x, uv, c, epochs, device, seed=0):
    torch.manual_seed(seed); np.random.seed(seed)
    net = FCN(c).to(device)
    opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=epochs * ((len(x) + 63) // 64))
    x, uv = x.to(device), torch.from_numpy(uv).to(device)
    for _ in range(epochs):
        for b in torch.randperm(len(x), device=device).split(64):
            xb, ub = shift(x[b], uv[b])
            loss = -(target_map(ub) * F.log_softmax(net(xb), 1)).sum(1).mean()
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    return net.eval()


@torch.no_grad()
def logp(net, x, device):
    return torch.cat([F.log_softmax(net(x[i:i + 256].to(device)), 1) for i in range(0, len(x), 256)]).cpu()


def peak(lp):
    """Arg max pixel, refined by the centre of mass of a 5 x 5 window, and the probability mass of that window."""
    p = lp.exp().view(-1, S, S)
    idx = p.flatten(1).argmax(1)
    v, u = (idx // S).numpy(), (idx % S).numpy()
    out, mass = [], []
    for i in range(len(p)):
        v0, v1, u0, u1 = max(v[i] - 2, 0), min(v[i] + 3, S), max(u[i] - 2, 0), min(u[i] + 3, S)
        w = p[i, v0:v1, u0:u1].numpy()
        ys, xs = np.mgrid[v0:v1, u0:u1]
        out.append([(w * xs).sum() / w.sum(), (w * ys).sum() / w.sum()]); mass.append(w.sum())
    return np.array(out, np.float32), np.array(mass)


def auroc(score, positive):
    s, p = np.asarray(score), np.asarray(positive, bool)
    rank = np.empty(len(s)); rank[s.argsort()] = np.arange(1, len(s) + 1)
    return float((rank[p].sum() - p.sum() * (p.sum() + 1) / 2) / (p.sum() * (~p).sum()))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("train"); ap.add_argument("eval")
    ap.add_argument("--level", default="L0_vis")
    ap.add_argument("--frames", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--folds", type=int, default=4)
    ap.add_argument("-d", "--device", default="cuda:0")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    project = camera(a.level)
    xt, gt, uvt, ept, cells_t = load(a.train, a.frames, project)
    xe, ge, uve, epe, cells = load(a.eval, 1, project)
    # pixel -> table position (cm), a least-squares affine map on the training demos, to report errors in cm
    A = np.linalg.lstsq(np.c_[uvt, np.ones(len(uvt))], gt[:, :2], rcond=None)[0]
    to_xy = lambda uv: np.c_[uv, np.ones(len(uv))] @ A
    print(f"labels: grasp pixel u {uvt[:, 0].min():.0f}-{uvt[:, 0].max():.0f}, v {uvt[:, 1].min():.0f}-{uvt[:, 1].max():.0f}; "
          f"affine pixel -> table fit error {100 * np.median(np.linalg.norm(to_xy(uvt) - gt[:, :2], axis=1)):.2f} cm; "
          f"1 px = {100 * np.linalg.norm(A[0]):.2f} / {100 * np.linalg.norm(A[1]):.2f} cm", flush=True)
    err = lambda uv: 100 * np.linalg.norm(to_xy(uv) - ge[:, :2], axis=1)
    f2 = lambda e: f"{np.median(e):.1f} ({(e < 2).mean():.2f})"
    cell_ep, flags = cells[epe], np.array(FLAGS)
    def stats(lp):
        """Per frame: sub-pixel peak and four numbers about the map (mass at the peak, entropy, peak log prob, mass of the best pixel)."""
        u_, m_ = peak(lp)
        p_ = lp.exp()
        return u_, np.stack([m_, -(p_ * lp).sum(1).numpy(), lp.max(1).values.numpy(), p_.max(1).values.numpy()], 1)

    def logistic(X, y, steps=2000):
        mu_, sd_ = X.mean(0), X.std(0) + 1e-6
        Xt, yt_ = torch.tensor((X - mu_) / sd_, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
        w = torch.zeros(X.shape[1] + 1, requires_grad=True)
        opt = torch.optim.Adam([w], 0.05)
        for _ in range(steps):
            loss = F.binary_cross_entropy_with_logits(Xt @ w[:-1] + w[-1], yt_) + 1e-3 * (w[:-1] ** 2).sum()
            opt.zero_grad(); loss.backward(); opt.step()
        w = w.detach().numpy()
        return lambda Z: 1 / (1 + np.exp(-(((Z - mu_) / sd_) @ w[:-1] + w[-1])))

    px = lambda uv_: np.linalg.norm(uv_ - uve, axis=1)
    fold = np.random.default_rng(0).permutation(len(gt)) % a.folds
    lps, uv, rel_e, info = {}, {}, {}, {}
    for k, c in SENSORS.items():
        # reliability without labels: out-of-fold maps on the training demos -> did the peak hit the grasp pixel (2 px)?
        X, hit = [], []
        for f in range(a.folds):
            tr, te = fold[ept] != f, fold[ept] == f
            net = fit(xt[k][torch.from_numpy(tr)], uvt[ept][tr], c, a.epochs, a.device, f)
            u_, st_ = stats(logp(net, xt[k][torch.from_numpy(te)], a.device))
            X.append(st_); hit.append(np.linalg.norm(u_ - uvt[ept][te], axis=1) < 2)
        X, hit = np.concatenate(X), np.concatenate(hit)
        clf = logistic(X, hit.astype(np.float32))
        lps[k] = logp(fit(xt[k], uvt[ept], c, a.epochs, a.device), xe[k], a.device)
        uv[k], st_e = stats(lps[k])
        rel_e[k] = clf(st_e)
        info[k] = dict(oof_hit=float(hit.mean()))
        print(f"{k}: out-of-fold hit rate on the training demos {hit.mean():.2f}", flush=True)
    N = S * S
    uv["product"], _ = peak(torch.log_softmax(sum(lps.values()), 1))
    robust = sum(torch.log(torch.from_numpy(rel_e[k])[:, None].float() * lps[k].exp() + (1 - torch.from_numpy(rel_e[k])[:, None].float()) / N) for k in SENSORS)
    uv["robust"], _ = peak(torch.log_softmax(robust, 1))
    best = np.stack([rel_e[k] for k in SENSORS]).argmax(0)
    uv["pick"] = np.stack([uv[k] for k in SENSORS])[best, np.arange(len(ge))]
    f3 = lambda e: f"{np.median(e):.2f} ({(e < 1.5).mean():.2f})"
    print(f"\nheld-out demos: {len(ge)}; error of the predicted grasp pixel, median px (share within 1.5 px); 1 px is about 1.9 x 1.3 cm on the table")
    print("| cell | shows the glass | n | rgb | depth | thermal | plain product | product with each sensor's own reliability | most reliable sensor |")
    print("|---|---|---|---|---|---|---|---|---|")
    rows = []
    for c in [*sorted(set(cell_ep.tolist())), None]:
        m = np.ones(len(ge), bool) if c is None else cell_ep == c
        r = {k: px(uv[k])[m] for k in (*SENSORS, "product", "robust", "pick")}
        print(f"| {'all' if c is None else c} | {'' if c is None else NAMES[c]} | {int(m.sum())} | " + " | ".join(f3(v) for v in r.values()) + " |")
        rows.append(dict(cell=c, n=int(m.sum()), **{k: (float(np.median(v)), float((v < 1.5).mean())) for k, v in r.items()}))
    print("per sensor, where it shows the glass / faintly / not: pixel error (median), share within 1.5 px, own reliability (median); AUROC shows against blind")
    rel = {}
    for i, k in enumerate(SENSORS):
        fl = flags[cell_ep, i]
        e_ = px(uv[k])
        g = lambda v, x: float(np.median(v[fl == x]))
        rel[k] = dict(error=[g(e_, 0), g(e_, 2), g(e_, 1)], hit=[float((e_[fl == x] < 1.5).mean()) for x in (0, 2, 1)],
                      reliability=[g(rel_e[k], 0), g(rel_e[k], 2), g(rel_e[k], 1)], auroc=auroc(rel_e[k][fl != 2], (fl == 0)[fl != 2]), **info[k])
        print(f"  {k}: error {rel[k]['error'][0]:.2f} / {rel[k]['error'][1]:.2f} / {rel[k]['error'][2]:.2f} px; within 1.5 px {rel[k]['hit'][0]:.2f} / {rel[k]['hit'][1]:.2f} / {rel[k]['hit'][2]:.2f}; "
              f"reliability {rel[k]['reliability'][0]:.2f} / {rel[k]['reliability'][1]:.2f} / {rel[k]['reliability'][2]:.2f}; AUROC {rel[k]['auroc']:.3f}", flush=True)
    err = px
    f2 = f3
    print("\nreference that uses the cells: trained only on demos where the sensor shows the glass; pixel error where it shows it fully / faintly")
    ft = np.array(FLAGS)[cells_t[ept]]
    orc = {}
    for i, (k, c) in enumerate(SENSORS.items()):
        tr = ft[:, i] != 1
        u_, _ = peak(logp(fit(xt[k][torch.from_numpy(tr)], uvt[ept][tr], c, a.epochs, a.device), xe[k], a.device))
        fl = flags[cell_ep, i]
        orc[k] = [float(np.median(err(u_)[fl == 0])), float((err(u_)[fl == 0] < 2).mean()), float(np.median(err(u_)[fl == 2])), float((err(u_)[fl == 2] < 2).mean())]
        print(f"  {k}: {f2(err(u_)[fl == 0])} / {f2(err(u_)[fl == 2])}", flush=True)
    if a.out:
        json.dump(dict(train=a.train, eval=a.eval, frames=a.frames, epochs=a.epochs, rows=rows, sensors=rel, oracle=orc), open(a.out, "w"), indent=1)
