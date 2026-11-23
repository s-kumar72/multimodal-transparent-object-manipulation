"""
Hindsight cue gating, perception part: per-sensor grasp cues and their reliability, learned from the demos alone.

Label from the demo itself ("hindsight"): the tool position at the first step that closes the gripper (arm kinematics),
projected into the image with the camera calibration. No cell, visibility flag or mask is used anywhere.

  cue     per sensor a small fully convolutional network (local receptive field, no global pooling) gives a distribution
          over the 96 x 96 pixels of where the grasp will be and a map of the grasp height, from the first frames of an
          episode (the arm has not moved yet, so the map can only come from the scene). A sensor that does not show
          the glass still has to put its peak somewhere.
  gate    r_m in [0, 1] per sensor: is there really a glass at this sensor's own peak? A small network (the verifier)
          looks at the image patch around the peak and answers. Its labels need no annotation: the maps of every demo
          come from networks that were not trained on it (4 folds), so a sensor that does not show the glass is as
          wrong on them as it will be at test time, and "the peak was within 2 px of the hindsight grasp pixel" is the
          label. The verifier of a fold never saw its demos either.
  fusion  verify (default): the map of the most trusted sensor times the maps of the trusted sensors (r >= 0.5) whose
          peak agrees with it within 2 px.
          Forms kept for the comparison: mixture | product | pick, a gate fitted on four numbers of every map and the
          distances between the peaks instead of the image (mixture: prod_m [r_m p_m(u) + (1 - r_m) / N]; product:
          prod_m p_m(u)^beta_m; pick: the sensor with the highest own reliability); evidence: the three sensors
          trained together on p(u) = softmax_u sum_m log(exp l_m(u) + 1 / N), the gate being a sensor's total
          evidence (one sensor ends up carrying the others, see the README).
          Controls without a gate (--mode): nogate, the per-sensor maps averaged (or multiplied) with no reliability
          estimate; early, one network on the five channels of the three sensors; label, no cue at all: the file of
          training cues holds the hindsight grasp points themselves, as the target of an auxiliary loss
          (policy/aux_grasp_policy.py), and at evaluation the vector is zero.
  output  the cue vector of an episode: [x, y, z of the grasp point (m, robot frame; the fused pixel on the plane of the
          predicted height), u / 96, v / 96, confidence (mass of the fused peak), r_rgb, r_depth, r_thermal].

  python policy/hindsight_cue.py TRAIN.zarr --out data/hcue/L0_vis_700 [--eval EVAL.zarr] [--level L0_vis] [-d cuda:0]

writes OUT/cue.pt (networks, gate, camera), OUT/train_cue.npy (cue vector of every training episode, from networks and
a gate that did not see it), OUT/report.txt / report.json (per cell: on the training demos out of fold, and on EVAL).
The cells are used for the report only. At evaluation `HindsightCue(OUT/cue.pt)(raw frame)` gives the cue vector.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SENSORS = {"rgb": 3, "depth": 1, "thermal": 1}
M = len(SENSORS)
S = 96
P = S * S
Z0, ZS = 0.11, 0.03          # grasp height (m) -> network units
ARM = "sim/assets/gen3_lite/gen3_lite.xml"
CUE_DIM = 6 + M
MODES = ("mixture", "product", "pick")      # the earlier form; the default is "evidence"
LOG_P = float(np.log(P))
# report only: per cell of L0_vis, rgb / depth / thermal shows (0), is blind (1) or faint (2)
FLAGS = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (1, 0, 1), (0, 1, 1), (2, 1, 1), (1, 2, 1), (1, 1, 2), (2, 2, 2)]


# ---------------------------------------------------------------------------------------------------- geometry, labels
def fk(qs):
    import mujoco
    m = mujoco.MjModel.from_xml_path(ARM)
    d, bid, out = mujoco.MjData(m), m.body("tool_frame").id, []
    for q in qs:
        d.qpos[:6] = q
        mujoco.mj_kinematics(m, d)
        out.append(d.xpos[bid].copy())
    return np.array(out, np.float32)


def camera_params(level):
    """Pose and focal length of the fixed camera (the calibration)."""
    sys.path.insert(0, "sim")
    from env import GlassPickEnv
    from testsets import manifest
    env = GlassPickEnv(config=manifest()["levels"][level])
    env.reset(0)
    s, d = env.sensors, env.data
    cam = dict(R=d.cam_xmat[s.cam_id].reshape(3, 3).copy(), t=d.cam_xpos[s.cam_id].copy(), f=float(s.focal),
               h=int(s.r.height), w=int(s.r.width))
    env.close()
    return cam


def project(cam, p):
    pc = (np.asarray(p, np.float64) - cam["t"]) @ cam["R"]
    return np.stack([(cam["w"] / 2 + cam["f"] * pc[:, 0] / -pc[:, 2]) * S / cam["w"],
                     (cam["h"] / 2 - cam["f"] * pc[:, 1] / -pc[:, 2]) * S / cam["h"]], -1).astype(np.float32)


def backproject(cam, uv, z):
    """The point of the pixel's ray at height z (robot frame)."""
    uv, z = np.asarray(uv, np.float64), np.asarray(z, np.float64)
    a = (uv[:, 0] * cam["w"] / S - cam["w"] / 2) / cam["f"]
    b = -(uv[:, 1] * cam["h"] / S - cam["h"] / 2) / cam["f"]
    ray = np.stack([a, b, -np.ones_like(a)], -1) @ cam["R"].T
    d = (z - cam["t"][2]) / ray[:, 2]
    return (cam["t"] + d[:, None] * ray).astype(np.float32)


def load(path, n_frames):
    """First n_frames frames of every episode, and the hindsight grasp point of every episode."""
    import zarr
    z = zarr.open(str(path), "r")
    ends = z["meta/episode_ends"][:]
    starts = np.r_[0, ends[:-1]]
    act = z["data/action"][:]
    close = [s + int(np.argmax(act[s:t, 6] > 0)) for s, t in zip(starts, ends)]
    goal = fk(act[close, :6])
    idx = np.concatenate([np.arange(s, s + n_frames) for s in starts])
    x = {k: torch.from_numpy(z["data"][k].get_orthogonal_selection((idx,))).permute(0, 3, 1, 2).contiguous() for k in SENSORS}
    cells = z["meta/ep_cell"][:] if "ep_cell" in z["meta"] else None
    return x, goal, np.repeat(np.arange(len(ends)), n_frames), cells


# ---------------------------------------------------------------------------------------------------- per-sensor maps
def block(i, o, stride=1, dilation=1):
    return nn.Sequential(nn.Conv2d(i, o, 3, stride, dilation, dilation), nn.GroupNorm(8, o), nn.ReLU())


class FCN(nn.Module):
    """uint8 image -> logits over the pixels and a grasp-height map (network units)."""
    def __init__(self, c):
        super().__init__()
        self.e1 = nn.Sequential(block(c, 32), block(32, 32))
        self.e2 = nn.Sequential(block(32, 64, 2), block(64, 64))
        self.e3 = nn.Sequential(block(64, 128, 2), block(128, 128, dilation=2), block(128, 128, dilation=2))
        self.d2 = block(128 + 64, 64)
        self.d1 = block(64 + 32, 32)
        self.out = nn.Conv2d(32, 2, 1)

    def forward(self, x):
        x = x.float() / 255.0
        a = self.e1(x)
        b = self.e2(a)
        c = self.e3(b)
        b = self.d2(torch.cat([F.interpolate(c, scale_factor=2, mode="bilinear", align_corners=False), b], 1))
        a = self.d1(torch.cat([F.interpolate(b, scale_factor=2, mode="bilinear", align_corners=False), a], 1))
        o = self.out(a).flatten(2)
        return o[:, 0], o[:, 1]


def target_map(uv, sigma=1.5):
    ys, xs = torch.meshgrid(torch.arange(S, device=uv.device), torch.arange(S, device=uv.device), indexing="ij")
    g = torch.exp(-((xs[None] - uv[:, 0, None, None]) ** 2 + (ys[None] - uv[:, 1, None, None]) ** 2) / (2 * sigma ** 2))
    return (g / g.sum((1, 2), keepdim=True).clamp_min(1e-9)).flatten(1)


def shift(x, uv, k=8):
    """The same random shift of the image (edge padded) and of the label."""
    dx, dy = np.random.randint(-k, k + 1, 2)
    xp = F.pad(x.float(), (k, k, k, k), mode="replicate")
    return xp[..., k - dy:k - dy + S, k - dx:k - dx + S], uv + torch.tensor([dx, dy], device=uv.device)


def fit_net(x, uv, zn, c, epochs, device, seed=0):
    torch.manual_seed(seed); np.random.seed(seed)
    net = FCN(c).to(device)
    opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=epochs * ((len(x) + 63) // 64))
    x = x.to(device)
    uv, zn = torch.as_tensor(uv, device=device), torch.as_tensor(zn, dtype=torch.float32, device=device)
    for _ in range(epochs):
        for b in torch.randperm(len(x), device=device).split(64):
            xb, ub = shift(x[b], uv[b])
            lg, h = net(xb)
            t = target_map(ub)
            loss = -(t * F.log_softmax(lg, 1)).sum(1).mean() \
                + (t * F.smooth_l1_loss(h, zn[b, None].expand_as(h), reduction="none")).sum(1).mean()
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    return net.eval()


@torch.no_grad()
def maps(net, x, device):
    lp, hm = [], []
    for i in range(0, len(x), 256):
        lg, h = net(x[i:i + 256].to(device))
        lp.append(F.log_softmax(lg, 1).cpu()); hm.append(h.cpu())
    return torch.cat(lp), torch.cat(hm)


def peak(lp):
    """Arg max pixel refined by the centre of mass of its 5 x 5 window -> (N, 2) u, v; the mass of the window; and the
    window itself (indices and weights) to read other maps at the same place."""
    p = lp.exp().view(-1, S, S)
    idx = p.flatten(1).argmax(1)
    v, u = idx // S, idx % S
    d = torch.arange(-2, 3, device=lp.device)
    vv, uu = (v[:, None, None] + d[None, :, None]).expand(-1, 5, 5), (u[:, None, None] + d[None, None, :]).expand(-1, 5, 5)
    ok = (vv >= 0) & (vv < S) & (uu >= 0) & (uu < S)
    vv, uu = vv.clamp(0, S - 1), uu.clamp(0, S - 1)
    n = torch.arange(len(p), device=lp.device)[:, None, None]
    w = p[n, vv, uu] * ok
    mass = w.sum((1, 2)).clamp_min(1e-12)
    uv = torch.stack([(w * uu).sum((1, 2)) / mass, (w * vv).sum((1, 2)) / mass], 1)
    return uv, mass, (n, vv, uu, w / mass[:, None, None])


def read(hm, win):
    n, vv, uu, w = win
    return (hm.view(-1, S, S)[n, vv, uu] * w).sum((1, 2))


def map_stats(lp):
    """(N, 2) sub-pixel peak and (N, 4) numbers about the map: mass at the peak, entropy, peak log prob, peak prob."""
    uv, mass, _ = peak(lp)
    p = lp.exp()
    return uv, torch.stack([mass, -(p * lp).sum(1), lp.max(1).values, p.max(1).values], 1)


# ---------------------------------------------------------------------------------------------------- evidence form
def fit_joint(x, uv, zn, epochs, device, seed=0):
    """The three sensors' networks trained together on the fused evidence map (see the module docstring)."""
    torch.manual_seed(seed); np.random.seed(seed)
    nets = {k: FCN(c).to(device) for k, c in SENSORS.items()}
    params = [p for n in nets.values() for p in n.parameters()]
    n = len(uv)
    opt = torch.optim.AdamW(params, 1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=epochs * ((n + 63) // 64))
    x = {k: v.to(device) for k, v in x.items()}
    uv, zn = torch.as_tensor(uv, device=device), torch.as_tensor(zn, dtype=torch.float32, device=device)
    for _ in range(epochs):
        for b in torch.randperm(n, device=device).split(64):
            state = np.random.get_state()
            fused, h_loss = 0, 0
            for k in SENSORS:
                np.random.set_state(state)                  # the same shift for the three sensors
                xb, ub = shift(x[k][b], uv[b])
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=str(device).startswith("cuda")):
                    lg, h = nets[k](xb)
                lg, h, t = lg.float(), h.float(), target_map(ub)
                fused = fused + torch.logaddexp(lg, torch.full_like(lg, -LOG_P))
                # height: only as far as the sensor itself has evidence at the grasp pixel
                w = torch.sigmoid((t * lg).sum(1) + LOG_P).detach()
                h_loss = h_loss + (w * (t * F.smooth_l1_loss(h, zn[b, None].expand_as(h), reduction="none")).sum(1)).mean()
            loss = -(t * F.log_softmax(fused, 1)).sum(1).mean() + h_loss
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    return {k: n.eval() for k, n in nets.items()}


@torch.no_grad()
def raw_maps(nets, x, device):
    """-> evidence maps l (N, M, P) and height maps (N, M, P), on the cpu."""
    lg, hm = [], []
    for k in SENSORS:
        a, b = zip(*[[o.float().cpu() for o in nets[k](x[k][i:i + 256].to(device))] for i in range(0, len(x[k]), 256)])
        lg.append(torch.cat(a)); hm.append(torch.cat(b))
    return torch.stack(lg, 1), torch.stack(hm, 1)


def evidence_fuse(lg):
    """lg (N, M, P) -> fused log map (N, P) and the gate r (N, M)."""
    fused = torch.logaddexp(lg, torch.full_like(lg, -LOG_P)).sum(1)
    return F.log_softmax(fused, 1), torch.sigmoid(torch.logsumexp(lg, 2))


@torch.no_grad()
def evidence_cue(cam, lg, hms, device):
    """Evidence and height maps (N, M, P) -> cue vectors (N, CUE_DIM), the gate (N, M), every sensor's own peak (N, M, 2)."""
    lg, hms = lg.to(device).float(), hms.to(device).float()
    fused, r = evidence_fuse(lg)
    uv, mass, win = peak(fused)
    w = torch.stack([torch.sigmoid(read(lg[:, m], win) + LOG_P) for m in range(M)], 1)   # who has evidence at the peak
    w = w / w.sum(1, keepdim=True).clamp_min(1e-6)
    zn = (w * torch.stack([read(hms[:, m], win) for m in range(M)], 1)).sum(1)
    uv, z = uv.cpu().numpy(), Z0 + ZS * zn.cpu().numpy()
    own = torch.stack([peak(F.log_softmax(lg[:, m], 1))[0] for m in range(M)], 1).cpu().numpy()
    vec = np.concatenate([backproject(cam, uv, z), uv / S, mass.cpu().numpy()[:, None], r.cpu().numpy()], 1)
    return vec.astype(np.float32), r.cpu().numpy(), own


# ---------------------------------------------------------------------------------------------------- verified form
R_PATCH = 12      # the verifier sees 25 x 25 pixels around a sensor's own peak
AGREE = 2.0       # px: peaks closer than this agree; also the tolerance of the verifier's labels
TRUST = 0.5


def own_peaks(lps):
    """lps (N, M, P) -> (N, M, 2) the sensors' own sub-pixel peaks."""
    return torch.stack([peak(lps[:, m].float())[0] for m in range(M)], 1)


def patches(img, lp, pk):
    """img (N, C, 96, 96) uint8, lp (N, P) log map, pk (N, 2) -> (N, C + 1, 25, 25): the image and the map around the peak."""
    n = len(img)
    z = torch.cat([img.float() / 255, (lp.float().view(n, 1, S, S) + 12).clamp(-8, 12) / 12], 1)
    z = F.pad(z, (R_PATCH, R_PATCH, R_PATCH, R_PATCH))
    c = pk.round().long().clamp(0, S - 1)
    d = torch.arange(2 * R_PATCH + 1, device=img.device)
    vv = (c[:, 1, None, None] + d[None, :, None]).expand(-1, len(d), len(d))
    uu = (c[:, 0, None, None] + d[None, None, :]).expand(-1, len(d), len(d))
    return z[torch.arange(n, device=img.device)[:, None, None], :, vv, uu].permute(0, 3, 1, 2).contiguous()


class Verifier(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.f = nn.Sequential(block(c + 1, 32), block(32, 32), nn.MaxPool2d(2), block(32, 64), block(64, 64),
                               nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(64, 1))

    def forward(self, p):
        return self.f(p)[:, 0]


def fit_verifier(p, y, device, seed=0, epochs=40):
    torch.manual_seed(seed)
    net = Verifier(p.shape[1] - 1).to(device)
    opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=epochs * ((len(p) + 63) // 64))
    p, y = p.to(device), torch.as_tensor(y, dtype=torch.float32, device=device)
    for _ in range(epochs):
        for b in torch.randperm(len(p), device=device).split(64):
            pb = p[b]
            if torch.rand(1).item() < 0.5:      # the peak is rounded to a pixel: the patch may sit one pixel off
                pb = torch.roll(pb, shifts=(int(torch.randint(-1, 2, (1,))), int(torch.randint(-1, 2, (1,)))), dims=(2, 3))
            loss = F.binary_cross_entropy_with_logits(net(pb), y[b])
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    return net.eval()


@torch.no_grad()
def trust(net, p, device):
    return torch.sigmoid(torch.cat([net(p[i:i + 512].to(device)) for i in range(0, len(p), 512)])).cpu()


RULES = ("consensus", "pick", "trusted")


@torch.no_grad()
def verified_cue(cam, lps, hms, r, device, rule="consensus"):
    """lps, hms (N, M, P), gate r (N, M) -> cue vectors (N, CUE_DIM). Whose maps are multiplied:
    consensus  the most trusted sensor and the trusted sensors (r >= 0.5) whose peak agrees with it (the method)
    pick       the most trusted sensor alone
    trusted    every trusted sensor, agreeing or not"""
    lps, hms, r = lps.to(device).float(), hms.to(device).float(), torch.as_tensor(r, dtype=torch.float32, device=device)
    pk = own_peaks(lps)
    top = r.argmax(1)
    near = (pk - pk[torch.arange(len(pk), device=device), top][:, None]).norm(dim=2) < AGREE
    ok = {"consensus": (r >= TRUST) & near, "pick": torch.zeros_like(near), "trusted": r >= TRUST}[rule]
    w = torch.maximum(ok.float(), F.one_hot(top, M).float())
    uv, mass, win = peak(F.log_softmax((w[:, :, None] * lps).sum(1), 1))
    zn = (w * torch.stack([read(hms[:, m], win) for m in range(M)], 1)).sum(1) / w.sum(1)
    uv, z = uv.cpu().numpy(), Z0 + ZS * zn.cpu().numpy()
    vec = np.concatenate([backproject(cam, uv, z), uv / S, mass.cpu().numpy()[:, None], r.cpu().numpy()], 1)
    return vec.astype(np.float32)


# ---------------------------------------------------------------------------------------------------- controls without a gate
UNGATED = ("mean", "product")
C_ALL = sum(SENSORS.values())


@torch.no_grad()
def ungated_cue(cam, lps, hms, device, rule="mean"):
    """lps, hms (N, M, P) -> cue vectors (N, CUE_DIM) from the three maps with no reliability estimate: the mean of the
    three distributions or their product. Height: every sensor's map read at the fused peak, weighted by the sensor's
    share of the probability there. The gate entries of the cue are 1."""
    lps, hms = lps.to(device).float(), hms.to(device).float()
    uv, mass, win = peak(F.log_softmax(torch.logsumexp(lps, 1) if rule == "mean" else lps.sum(1), 1))
    w = torch.stack([read(lps[:, m].exp(), win) for m in range(M)], 1)
    w = w / w.sum(1, keepdim=True).clamp_min(1e-12)
    zn = (w * torch.stack([read(hms[:, m], win) for m in range(M)], 1)).sum(1)
    uv, z = uv.cpu().numpy(), Z0 + ZS * zn.cpu().numpy()
    vec = np.concatenate([backproject(cam, uv, z), uv / S, mass.cpu().numpy()[:, None], np.ones((len(uv), M))], 1)
    return vec.astype(np.float32)


@torch.no_grad()
def single_cue(cam, lp, hm, device):
    """One map lp, hm (N, P) (mode early) -> cue vectors (N, CUE_DIM) and the peak (N, 2). The gate entries are 1."""
    uv, mass, win = peak(lp.to(device).float())
    zn = read(hm.to(device).float(), win)
    uv, z = uv.cpu().numpy(), Z0 + ZS * zn.cpu().numpy()
    vec = np.concatenate([backproject(cam, uv, z), uv / S, mass.cpu().numpy()[:, None], np.ones((len(uv), M))], 1)
    return vec.astype(np.float32), uv


def stacked(x):
    """The three sensors' frames as one image of five channels."""
    return torch.cat([x[k] for k in SENSORS], 1)


# ---------------------------------------------------------------------------------------------------- gate and fusion
def gate_features(lps):
    """lps (N, M, P) -> (N, 4 M + M (M - 1) / 2): the four numbers of every map and the distances between the peaks."""
    uv, st = zip(*[map_stats(lps[:, m]) for m in range(M)])
    dist = [torch.log1p((uv[i] - uv[j]).norm(dim=1)) for i in range(M) for j in range(i + 1, M)]
    return torch.cat([*st, torch.stack(dist, 1)], 1), torch.stack(uv, 1)


class Gate(nn.Module):
    def __init__(self, n_in, mode, mu, sd):
        super().__init__()
        self.mode = mode
        self.register_buffer("mu", mu); self.register_buffer("sd", sd)
        if mode == "pick":           # every sensor from its own four numbers
            self.net = nn.ModuleList([nn.Linear(4, 1) for _ in range(M)])
        else:
            self.net = nn.Sequential(nn.Linear(n_in, 32), nn.Mish(), nn.Linear(32, M))

    def forward(self, f):
        f = (f - self.mu) / self.sd
        if self.mode == "pick":
            return torch.sigmoid(torch.cat([self.net[m](f[:, 4 * m:4 * m + 4]) for m in range(M)], 1))
        o = self.net(f)
        return F.softplus(o) if self.mode == "product" else torch.sigmoid(o)


def fuse(lps, r, mode):
    """lps (N, M, P) log maps, r (N, M) gate -> fused log map (N, P) and the weights (N, M) of the sensors in it."""
    if mode == "mixture":
        r_ = r.clamp(1e-6, 1 - 1e-6)[:, :, None]
        out = torch.logaddexp(torch.log(r_) + lps, torch.log1p(-r_) - np.log(P)).sum(1)
        w = r
    elif mode == "product":
        out, w = (r[:, :, None] * lps).sum(1), r
    else:
        w = F.one_hot(r.argmax(1), M).to(lps.dtype)
        out = (w[:, :, None] * lps).sum(1)
    return F.log_softmax(out, 1), w / w.sum(1, keepdim=True).clamp_min(1e-6)


def fit_gate(feat, lps, uv_m, uv_true, mode, device, steps=1500, batch=512, seed=0):
    torch.manual_seed(seed)
    feat, lps, uv_true = feat.to(device), lps.to(device).float(), torch.as_tensor(uv_true, device=device)
    gate = Gate(feat.shape[1], mode, feat.mean(0), feat.std(0) + 1e-6).to(device)
    opt = torch.optim.Adam(gate.parameters(), 1e-2, weight_decay=1e-4)
    hit = ((uv_m.to(device) - uv_true[:, None]).norm(dim=2) < 2).float()
    for _ in range(steps):
        b = torch.randint(0, len(feat), (batch,), device=device)
        r = gate(feat[b])
        if mode == "pick":           # did the sensor's own peak hit the grasp pixel?
            loss = F.binary_cross_entropy(r, hit[b])
        else:                        # does the fused map explain the grasp pixel?
            loss = -(target_map(uv_true[b]) * fuse(lps[b], r, mode)[0]).sum(1).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return gate.eval()


@torch.no_grad()
def cue_vectors(cam, lps, hms, gate, device):
    """lps, hms (N, M, P) -> cue vectors (N, CUE_DIM), and the gate (N, M)."""
    lps, hms = lps.to(device).float(), hms.to(device).float()
    feat, _ = gate_features(lps)
    r = gate(feat)
    fused, w = fuse(lps, r, gate.mode)
    uv, mass, win = peak(fused)
    zn = (w * torch.stack([read(hms[:, m], win) for m in range(M)], 1)).sum(1)
    uv, z = uv.cpu().numpy(), Z0 + ZS * zn.cpu().numpy()
    xyz = backproject(cam, uv, z)
    rn = r / r.max(1, keepdim=True).values.clamp_min(1e-6) if gate.mode == "product" else r
    return np.concatenate([xyz, uv / S, mass.cpu().numpy()[:, None], rn.cpu().numpy()], 1).astype(np.float32), r.cpu().numpy()


class HindsightCue:
    """The trained cue: raw frame(s) of the three sensors -> cue vector(s). Used by the evaluation env at reset."""
    def __init__(self, path, device="cpu"):
        ck = torch.load(path, map_location="cpu", weights_only=False)
        self.cam, self.device, self.mode, self.rule = ck["cam"], device, ck["mode"], ck.get("rule", "consensus")
        self.nets, self.ver, self.gate, self.net = {}, {}, None, None
        if self.mode == "label":             # no cue at evaluation
            return
        if self.mode == "early":
            self.net = FCN(C_ALL).to(device).eval()
            self.net.load_state_dict(ck["net"])
            return
        for k, c in SENSORS.items():
            self.nets[k] = FCN(c).to(device).eval()
            self.nets[k].load_state_dict(ck["nets"][k])
            if self.mode == "verify":
                self.ver[k] = Verifier(c).to(device).eval()
                self.ver[k].load_state_dict(ck["verifier"][k])
        if self.mode in MODES:
            g = ck["gate"]
            self.gate = Gate(g["mu"].numel(), self.mode, g["mu"], g["sd"]).to(device).eval()
            self.gate.load_state_dict(g)

    @torch.no_grad()
    def __call__(self, raw):
        """raw[k]: (H, W, C) or (N, H, W, C) uint8 -> (CUE_DIM,) or (N, CUE_DIM)."""
        single = raw["rgb"].ndim == 3
        if self.mode == "label":
            return np.zeros(CUE_DIM if single else (len(raw["rgb"]), CUE_DIM), np.float32)
        x = {k: torch.from_numpy(np.array(raw[k])) for k in SENSORS}
        x = {k: (v[None] if single else v).permute(0, 3, 1, 2) for k, v in x.items()}
        if self.mode == "early":
            v = single_cue(self.cam, *maps(self.net, stacked(x), self.device), self.device)[0]
        elif self.mode == "evidence":
            v = evidence_cue(self.cam, *raw_maps(self.nets, x, self.device), self.device)[0]
        else:
            out = [maps(self.nets[k], x[k], self.device) for k in SENSORS]
            lps, hms = torch.stack([o[0] for o in out], 1), torch.stack([o[1] for o in out], 1)
            if self.mode == "verify":
                pk = own_peaks(lps)
                r = torch.stack([trust(self.ver[k], patches(x[k], lps[:, m], pk[:, m]), self.device)
                                 for m, k in enumerate(SENSORS)], 1)
                v = verified_cue(self.cam, lps, hms, r, self.device, self.rule)
            elif self.mode == "nogate":
                v = ungated_cue(self.cam, lps, hms, self.device, self.rule)
            else:
                v, _ = cue_vectors(self.cam, lps, hms, self.gate, self.device)
        return v[0] if single else v


# ---------------------------------------------------------------------------------------------------- fit and report
def auroc(score, positive):
    s, p = np.asarray(score), np.asarray(positive, bool)
    if p.sum() == 0 or (~p).sum() == 0:
        return float("nan")
    rank = np.empty(len(s)); rank[s.argsort()] = np.arange(1, len(s) + 1)
    return float((rank[p].sum() - p.sum() * (p.sum() + 1) / 2) / (p.sum() * (~p).sum()))


def report(title, cam, vec, r, uv_m, goal, cells, out, per_sensor=True):
    """Per cell: the fused cue against the hindsight grasp point; per sensor: the gate against what the cell says."""
    uv_true = project(cam, goal)
    px = np.linalg.norm(vec[:, 3:5] * S - uv_true, axis=1)
    cm = 100 * np.linalg.norm(vec[:, :2] - goal[:, :2], axis=1)
    cm_z = 100 * np.linalg.norm(backproject(cam, vec[:, 3:5] * S, goal[:, 2])[:, :2] - goal[:, :2], axis=1)
    dz = 100 * np.abs(vec[:, 2] - goal[:, 2])
    best = np.min(np.linalg.norm(uv_m - uv_true[:, None], axis=2), axis=1)
    lines = [f"\n{title}: {len(goal)} episodes, first frame", "| cell | n | pixel error (median) | within 1.5 px | "
             "best single sensor within 1.5 px | error on the table, cm (median) | within 1 cm | within 1.5 cm | "
             "the same with the true height: within 1 cm | height within 2.5 cm | gate rgb / depth / thermal (median) |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    rows = []
    for c in [*sorted(set(cells.tolist())), None]:
        m = np.ones(len(goal), bool) if c is None else cells == c
        row = dict(cell=c, n=int(m.sum()), px=float(np.median(px[m])), hit=float((px[m] < 1.5).mean()),
                   best=float((best[m] < 1.5).mean()), cm=float(np.median(cm[m])), cm1=float((cm[m] < 1).mean()),
                   cm15=float((cm[m] < 1.5).mean()), cm1_z=float((cm_z[m] < 1).mean()), z_ok=float((dz[m] < 2.5).mean()),
                   gate=[float(np.median(r[m, i])) for i in range(M)])
        rows.append(row)
        lines.append(f"| {'all' if c is None else c} | {row['n']} | {row['px']:.2f} | {row['hit']:.2f} | {row['best']:.2f} | "
                     f"{row['cm']:.2f} | {row['cm1']:.2f} | {row['cm15']:.2f} | {row['cm1_z']:.2f} | {row['z_ok']:.2f} | "
                     + " / ".join(f"{g:.2f}" for g in row["gate"]) + " |")
    flags, sens = np.array(FLAGS)[cells], {}
    for i, k in enumerate(SENSORS if per_sensor else ()):
        fl, e = flags[:, i], np.linalg.norm(uv_m[:, i] - uv_true, axis=1)
        sens[k] = dict(hit=[float((e[fl == x] < 1.5).mean()) for x in (0, 2, 1)],
                       gate=[float(np.median(r[fl == x, i])) for x in (0, 2, 1)],
                       auroc=auroc(r[fl != 2, i], (fl == 0)[fl != 2]), auroc_hit=auroc(r[:, i], e < 1.5))
        lines.append(f"  {k}: own peak within 1.5 px where it shows the glass / faintly / not "
                     f"{sens[k]['hit'][0]:.2f} / {sens[k]['hit'][1]:.2f} / {sens[k]['hit'][2]:.2f}; gate (median) "
                     f"{sens[k]['gate'][0]:.2f} / {sens[k]['gate'][1]:.2f} / {sens[k]['gate'][2]:.2f}; AUROC of the gate: "
                     f"shows against blind {sens[k]['auroc']:.3f}, own peak right against wrong {sens[k]['auroc_hit']:.3f}")
    print("\n".join(lines), flush=True)
    out[title] = dict(rows=rows, sensors=sens)
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("train")
    ap.add_argument("--out", required=True)
    ap.add_argument("--eval", default=None)
    ap.add_argument("--level", default="L0_vis")
    ap.add_argument("--frames", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--folds", type=int, default=4)
    ap.add_argument("--mode", default="verify", choices=("verify", "evidence", "nogate", "early", "label") + MODES,
                    help="verify: the gate from the image patch at every sensor's own peak (default); mixture | product "
                         "| pick: a gate on numbers of the maps, all three reported and this one saved; evidence: the "
                         "sensors trained together; controls without a gate: nogate (the per-sensor maps averaged or "
                         "multiplied; both reported, --ungated saved), early (one network on the five channels), "
                         "label (the hindsight grasp points themselves, for an auxiliary loss)")
    ap.add_argument("--ungated", default="mean", choices=UNGATED, help="mode nogate: how the maps are combined")
    ap.add_argument("--rule", default="consensus", choices=RULES, help="mode verify: whose maps are multiplied; all "
                    "rules are reported and this one saved")
    ap.add_argument("-d", "--device", default="cuda:0")
    a = ap.parse_args()
    out_dir = Path(a.out); out_dir.mkdir(parents=True, exist_ok=True)
    cam = camera_params(a.level)
    x, goal, ep, cells = load(a.train, a.frames)
    n_ep = len(goal)
    uv_ep = project(cam, goal)
    uv, zn = uv_ep[ep], ((goal[:, 2] - Z0) / ZS)[ep]
    fold = np.random.default_rng(0).permutation(n_ep) % a.folds
    first = np.arange(n_ep) * a.frames                      # the first frame of every episode
    print(f"{n_ep} demos, {len(ep)} frames; grasp pixel u {uv_ep[:, 0].min():.0f}-{uv_ep[:, 0].max():.0f}, "
          f"v {uv_ep[:, 1].min():.0f}-{uv_ep[:, 1].max():.0f}; grasp height {100 * goal[:, 2].min():.1f}-"
          f"{100 * goal[:, 2].max():.1f} cm", flush=True)

    args = dict(frames=a.frames, epochs=a.epochs, folds=a.folds, level=a.level)
    ones = np.ones((n_ep, M), np.float32)
    if a.mode == "label":
        # no networks: the hindsight grasp point of every training demo, in the layout of a cue vector
        np.save(out_dir / "train_cue.npy", np.concatenate([goal, uv_ep / S, np.ones((n_ep, 1 + M))], 1).astype(np.float32))
        torch.save(dict(mode="label", cam=cam, args=args), out_dir / "cue.pt")
        print(f"hindsight grasp points of {n_ep} demos written to {out_dir}/train_cue.npy", flush=True)
        return

    if a.mode == "early":
        # one network on the five channels; the map of every training demo from a network that did not see it
        res, txt, xc = {}, [], stacked(x)
        vec, pk = np.zeros((n_ep, CUE_DIM), np.float32), np.zeros((n_ep, 2), np.float32)
        for f in range(a.folds):
            tr, te = torch.from_numpy(fold[ep] != f), torch.from_numpy(first[fold == f])
            net = fit_net(xc[tr], uv[tr.numpy()], zn[tr.numpy()], C_ALL, a.epochs, a.device, f)
            vec[fold == f], pk[fold == f] = single_cue(cam, *maps(net, xc[te], a.device), a.device)
            print(f"fold {f} trained", flush=True)
        np.save(out_dir / "train_cue.npy", vec)
        txt.append(report("training demos, out of fold, one network on the five channels", cam, vec, ones,
                          np.repeat(pk[:, None], M, 1), goal, cells, res, per_sensor=False))
        net = fit_net(xc, uv, zn, C_ALL, a.epochs, a.device, a.folds)
        torch.save(dict(net={n: v.cpu() for n, v in net.state_dict().items()}, mode="early", cam=cam, args=args),
                   out_dir / "cue.pt")
        if a.eval:
            xe, ge, _, ce = load(a.eval, 1)
            vec, pk = single_cue(cam, *maps(net, stacked(xe), a.device), a.device)
            txt.append(report("held-out demos, one network on the five channels", cam, vec, np.ones((len(ge), M), np.float32),
                              np.repeat(pk[:, None], M, 1), ge, ce, res, per_sensor=False))
        (out_dir / "report.txt").write_text("\n".join(txt) + "\n")
        json.dump(res, open(out_dir / "report.json", "w"), indent=1)
        return

    if a.mode == "evidence":
        # maps of every training demo from networks that did not see it, and the networks of all demos
        res, txt = {}, []
        vec, r, own = np.zeros((n_ep, CUE_DIM), np.float32), np.zeros((n_ep, M), np.float32), np.zeros((n_ep, M, 2), np.float32)
        for f in range(a.folds):
            tr, te = torch.from_numpy(fold[ep] != f), torch.from_numpy(first[fold == f])
            nets = fit_joint({k: x[k][tr] for k in SENSORS}, uv[tr.numpy()], zn[tr.numpy()], a.epochs, a.device, f)
            vec[fold == f], r[fold == f], own[fold == f] = evidence_cue(
                cam, *raw_maps(nets, {k: x[k][te] for k in SENSORS}, a.device), a.device)
            print(f"fold {f} trained", flush=True)
        np.save(out_dir / "train_cue.npy", vec)
        txt.append(report("training demos, out of fold, fusion evidence", cam, vec, r, own, goal, cells, res))
        nets = fit_joint(x, uv, zn, a.epochs, a.device, a.folds)
        torch.save(dict(nets={k: {n: v.cpu() for n, v in net.state_dict().items()} for k, net in nets.items()},
                        mode="evidence", cam=cam, args=dict(frames=a.frames, epochs=a.epochs, folds=a.folds, level=a.level)),
                   out_dir / "cue.pt")
        if a.eval:
            xe, ge, _, ce = load(a.eval, 1)
            vec, r, own = evidence_cue(cam, *raw_maps(nets, xe, a.device), a.device)
            txt.append(report("held-out demos, fusion evidence", cam, vec, r, own, ge, ce, res))
        (out_dir / "report.txt").write_text("\n".join(txt) + "\n")
        json.dump(res, open(out_dir / "report.json", "w"), indent=1)
        return

    # 1. maps of every training frame from a network that did not see its episode, and the networks of all demos
    #    (kept in OUT/maps.pt, so the gates can be refitted without training the networks again)
    cache = out_dir / "maps.pt"
    if cache.exists():
        ck = torch.load(cache, map_location="cpu", weights_only=False)
        assert ck["args"] == dict(train=a.train, eval=a.eval, frames=a.frames, epochs=a.epochs, folds=a.folds), ck["args"]
        lps, hms, nets_sd, le, he, ge, ce = (ck[k] for k in ("lps", "hms", "nets", "le", "he", "ge", "ce"))
        print(f"maps read from {cache}", flush=True)
    else:
        lps, hms, nets_sd = torch.zeros(len(ep), M, P), torch.zeros(len(ep), M, P), {}
        le = he = ge = ce = None
        if a.eval:
            xe, ge, _, ce = load(a.eval, 1)
            le, he = torch.zeros(len(ge), M, P), torch.zeros(len(ge), M, P)
        for i, (k, c) in enumerate(SENSORS.items()):
            for f in range(a.folds):
                tr, te = torch.from_numpy(fold[ep] != f), torch.from_numpy(fold[ep] == f)
                net = fit_net(x[k][tr], uv[tr.numpy()], zn[tr.numpy()], c, a.epochs, a.device, f)
                lps[te, i], hms[te, i] = maps(net, x[k][te], a.device)
            net = fit_net(x[k], uv, zn, c, a.epochs, a.device, a.folds)
            nets_sd[k] = {n: v.cpu() for n, v in net.state_dict().items()}
            if a.eval:                                   # held-out demos: the network of all training demos
                le[:, i], he[:, i] = maps(net, xe[k], a.device)
            print(f"{k}: networks trained", flush=True)
        lps, hms = lps.half(), hms.half()
        if a.eval:
            le, he = le.half(), he.half()
        torch.save(dict(lps=lps, hms=hms, nets=nets_sd, le=le, he=he, ge=ge, ce=ce,
                        args=dict(train=a.train, eval=a.eval, frames=a.frames, epochs=a.epochs, folds=a.folds)), cache)

    if a.mode == "nogate":
        # 2. no gate: the out-of-fold maps of the training demos combined as they are
        res, txt = {}, []
        i0 = torch.from_numpy(first)
        pk = own_peaks(lps[i0]).numpy()
        for rule in UNGATED:
            vec = ungated_cue(cam, lps[i0], hms[i0], a.device, rule)
            if rule == a.ungated:
                np.save(out_dir / "train_cue.npy", vec)
            txt.append(report(f"training demos, out of fold, no gate, {rule} of the maps", cam, vec, ones, pk, goal,
                              cells, res, per_sensor=False))
        torch.save(dict(nets=nets_sd, mode="nogate", rule=a.ungated, cam=cam, args=args), out_dir / "cue.pt")
        if a.eval:
            pke = own_peaks(le).numpy()
            for rule in UNGATED:
                txt.append(report(f"held-out demos, no gate, {rule} of the maps", cam, ungated_cue(cam, le, he, a.device, rule),
                                  np.ones((len(ge), M), np.float32), pke, ge, ce, res, per_sensor=False))
        (out_dir / "report.txt").write_text("\n".join(txt) + "\n")
        json.dump(res, open(out_dir / "report.json", "w"), indent=1)
        return

    if a.mode == "verify":
        # 2. verifiers: is a sensor's own peak right? Labels from the out-of-fold maps; for the training demos the
        #    verifier is out of fold as well, and one of all demos is kept for the evaluation.
        res, txt = {}, []
        pk = own_peaks(lps)
        hit = ((pk - torch.as_tensor(uv)[:, None]).norm(dim=2) < AGREE).numpy()
        r, vers = torch.zeros(len(ep), M), {}
        for m, k in enumerate(SENSORS):
            pt = patches(x[k], lps[:, m], pk[:, m])
            for f in range(a.folds):
                tr, te = torch.from_numpy(fold[ep] != f), torch.from_numpy(fold[ep] == f)
                r[te, m] = trust(fit_verifier(pt[tr], hit[tr.numpy(), m], a.device, f), pt[te], a.device)
            vers[k] = fit_verifier(pt, hit[:, m], a.device, a.folds)
            print(f"{k}: verifier trained; its label (own peak right) holds in {hit[:, m].mean():.2f} of the frames, "
                  f"out-of-fold accuracy {((r[:, m] > 0.5).numpy() == hit[:, m]).mean():.3f}", flush=True)
        i0 = torch.from_numpy(first)
        for rule in RULES:
            vec = verified_cue(cam, lps[i0], hms[i0], r[i0], a.device, rule)
            if rule == a.rule:
                np.save(out_dir / "train_cue.npy", vec)
            txt.append(report(f"training demos, out of fold, verified gate, rule {rule}", cam, vec, r[i0].numpy(),
                              pk[i0].numpy(), goal, cells, res))
        torch.save(dict(nets=nets_sd, verifier={k: {n: v.cpu() for n, v in net.state_dict().items()} for k, net in vers.items()},
                        mode="verify", rule=a.rule, cam=cam,
                        args=dict(frames=a.frames, epochs=a.epochs, folds=a.folds, level=a.level)), out_dir / "cue.pt")
        if a.eval:
            xe = load(a.eval, 1)[0]
            pke = own_peaks(le)
            re = torch.stack([trust(vers[k], patches(xe[k], le[:, m], pke[:, m]), a.device) for m, k in enumerate(SENSORS)], 1)
            for rule in RULES:
                vec = verified_cue(cam, le, he, re, a.device, rule)
                txt.append(report(f"held-out demos, verified gate, rule {rule}", cam, vec, re.numpy(), pke.numpy(), ge, ce, res))
        (out_dir / "report.txt").write_text("\n".join(txt) + "\n")
        json.dump(res, open(out_dir / "report.json", "w"), indent=1)
        return

    # 2. gates: for the training demos out of fold as well (the gate of a fold never saw its episodes), and one of all
    feat, uv_m = gate_features(lps.to(a.device).float())
    feat, uv_m = feat.cpu(), uv_m.cpu()
    res, txt, gates = {}, [], {}
    for mode in MODES:
        vec, r = np.zeros((n_ep, CUE_DIM), np.float32), np.zeros((n_ep, M), np.float32)
        for f in range(a.folds):
            tr, te = torch.from_numpy(fold[ep] != f), fold == f
            g = fit_gate(feat[tr], lps[tr], uv_m[tr], uv[tr.numpy()], mode, a.device, seed=f)
            i0 = torch.from_numpy(first[te])
            vec[te], r[te] = cue_vectors(cam, lps[i0], hms[i0], g, a.device)
        gates[mode] = fit_gate(feat, lps, uv_m, uv, mode, a.device, seed=a.folds)
        txt.append(report(f"training demos, out of fold, fusion {mode}", cam, vec, r,
                          uv_m[torch.from_numpy(first)].numpy(), goal, cells, res))
        if mode == a.mode:
            np.save(out_dir / "train_cue.npy", vec)
    torch.save(dict(nets=nets_sd, gate=gates[a.mode].state_dict(), mode=a.mode,
                    cam=cam, gates={m: g.state_dict() for m, g in gates.items()},
                    args=dict(frames=a.frames, epochs=a.epochs, folds=a.folds, level=a.level)), out_dir / "cue.pt")

    # 3. held-out demos: the networks of all training demos and the final gate
    if a.eval:
        _, uve = gate_features(le.to(a.device).float())
        for mode in MODES:
            vec, r = cue_vectors(cam, le, he, gates[mode], a.device)
            txt.append(report(f"held-out demos, fusion {mode}", cam, vec, r, uve.cpu().numpy(), ge, ce, res))
    (out_dir / "report.txt").write_text("\n".join(txt) + "\n")
    json.dump(res, open(out_dir / "report.json", "w"), indent=1)


if __name__ == "__main__":
    main()
