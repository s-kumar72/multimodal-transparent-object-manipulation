"""
Hindsight angle: for task "align" of the second version of the benchmark, the direction the gripper has to be turned
to, learned from the demos alone as the hindsight cue's point is (policy/hindsight_cue.py).

A demo says in hindsight how the tool was turned about the vertical when the gripper first closed (the direction of
the glass's long side) and when it opened again (the direction of the mat). The gripper is the same turned by 180
degrees, so the label is the angle modulo 180 degrees, as the unit vector (cos 2a, sin 2a). With the camera's
calibration the label becomes a direction in the image at the cue's point, and for every sensor a small network reads
that direction from the image patch around the point in the first frames. In the image the question is the same
whichever way the patch is turned, so a training patch is turned by a random angle and its label with it; without
that the networks of the colour and thermal images learn their few hundred patches by heart (error 1 degree on them,
35 on others). The patch is small (17 x 17 pixels: the glass and little else); in a larger one the mat, the
distractors and the arm are what a network sees first. A network gives a vector that is trained towards the unit
vector of the direction with a squared error, so it is short where the patch does not say much, and its length is
how sure it is. A sensor's network learns from a demo as far as the cue trusts that sensor in it (the gate entry is
the weight of the sample), the vectors of the sensors are added with the same gate entries as weights, so a sensor
that does not show the glass has no say and a sure sensor outweighs an unsure one, and the direction in the image is
turned back into the tool's angle. As for
the cue, the estimate a policy is trained with comes from networks that did not see that demo (four folds), and the
evaluation env computes it from the first frame of the episode.

  python policy/hindsight_angle.py TRAIN.zarr --cue data/hcue/V2_align_700 --out data/hang/V2_align_700 --level V2_align
                                   [--event release] [--mode early] [--eval EVAL.zarr --eval_cue DIR] [-d cuda:0]

--cue    the folder of the hindsight cue of the same demos and event (train_cue.npy: where the patch is cut and the gate
         entries; cue.pt: the camera). --event release: the angle where the gripper opens again (the mat), with the cue
         of --event release.
--mode early: a control, one network on the five channels of the three sensors and no gate.
Writes OUT/train_ang.npy (episodes x 2, out of fold), OUT/ang.pt (the networks of the fit on all demos) and
OUT/report.txt: the error of the angle in degrees per sensor and of the estimate, over the demos and per visibility cell.

Information used: sensor images, the demonstrated joint targets, arm kinematics, the cue. The glass's and the mat's
true angle appear nowhere.
"""
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from policy.hindsight_cue import ARM, S, SENSORS, backproject, load, project

R_PATCH = 8        # the networks see 17 x 17 pixels around the cue's point: the glass and little else
R_TURN = 12        # a training patch is cut this large, turned, and its middle kept
JITTER = 2         # px: the patch of a training sample is shifted by up to this, as the cue's point is off by a little
C_ALL = sum(SENSORS.values())


def vec(a):
    """Angle(s) -> (cos 2a, sin 2a): a direction, the same turned by 180 degrees."""
    a = np.asarray(a, np.float64)
    return np.stack([np.cos(2 * a), np.sin(2 * a)], -1).astype(np.float32)


def ang(v):
    """(cos 2a, sin 2a) -> a in (-90, 90] degrees, as radians."""
    v = np.asarray(v, np.float64)
    return 0.5 * np.arctan2(v[..., 1], v[..., 0])


def to_image(cam, p, a):
    """The direction in the image (pixels u right, v down) of the direction a in the table plane at the points p (N, 3)."""
    d = 0.03 * np.stack([np.cos(a), np.sin(a), np.zeros_like(a)], -1)
    q0, q1 = project(cam, p - d), project(cam, p + d)
    return np.arctan2(q1[:, 1] - q0[:, 1], q1[:, 0] - q0[:, 0])


def to_world(cam, uv, z, alpha):
    """The direction in the table plane (at height z) of the image direction alpha at the pixels uv (N, 2)."""
    d = 2.0 * np.stack([np.cos(alpha), np.sin(alpha)], -1)
    p0, p1 = backproject(cam, uv - d, z), backproject(cam, uv + d, z)
    return np.arctan2(p1[:, 1] - p0[:, 1], p1[:, 0] - p0[:, 0])


def tool_angle(qs):
    """Joint targets (N, 6) -> the angle (rad) the tool is turned about the vertical, from the pose pointing down."""
    import mujoco
    m = mujoco.MjModel.from_xml_path(ARM)
    d, bid, out = mujoco.MjData(m), m.body("tool_frame").id, []
    for q in qs:
        d.qpos[:6] = q
        mujoco.mj_kinematics(m, d)
        x = d.xmat[bid].reshape(3, 3)[:, 0]          # the closing direction of the gripper
        out.append(np.arctan2(-x[0], x[1]))
    return np.array(out)


def event_angle(path, event):
    """The tool angle (rad) of every demo at its first close (grasp) or where it opens again (release)."""
    import zarr
    z = zarr.open(str(path), "r")
    ends = z["meta/episode_ends"][:]
    starts, act = np.r_[0, ends[:-1]], z["data/action"][:]
    step = [s + int(np.argmax(act[s:t, 6] > 0)) for s, t in zip(starts, ends)]
    if event == "release":
        step = [c + int(np.argmax(act[c:t, 6] <= 0)) for c, t in zip(step, ends)]
    return tool_angle(act[step, :6])


def crop(img, uv, shift=None, r=R_PATCH):
    """img (N, C, 96, 96), uv (N, 2) pixels -> (N, C, 2r + 1, 2r + 1) around the rounded point (edge padded), optionally
    shifted by shift (N, 2) pixels of at most JITTER."""
    n, p = len(img), 2 * r + 1
    pad = nn.functional.pad(img.float() / 255, (r + JITTER,) * 4, mode="replicate")
    c = torch.as_tensor(np.round(uv), dtype=torch.long).clamp(0, S - 1) + JITTER
    if shift is not None:
        c = c + shift
    idx = torch.arange(p)
    rows = (c[:, 1, None] + idx)[:, None, :, None]
    cols = (c[:, 0, None] + idx)[:, None, None, :]
    return pad[torch.arange(n)[:, None, None, None], torch.arange(img.shape[1])[None, :, None, None], rows, cols]


def turn(patch, target, phi):
    """Turns patches (B, C, P, P) by phi (B,) rad about their middle and the directions target (B, 2) with them. With
    this sampling grid what is in the patch turns by -phi in pixel coordinates (u right, v down)."""
    c, s, o = torch.cos(phi), torch.sin(phi), torch.zeros_like(phi)
    grid = nn.functional.affine_grid(torch.stack([torch.stack([c, -s, o], 1), torch.stack([s, c, o], 1)], 1),
                                     list(patch.shape), align_corners=False)
    out = nn.functional.grid_sample(patch, grid, mode="bilinear", padding_mode="border", align_corners=False)
    c2, s2 = torch.cos(2 * phi), torch.sin(2 * phi)
    return out, torch.stack([target[:, 0] * c2 + target[:, 1] * s2, target[:, 1] * c2 - target[:, 0] * s2], 1)


class AngleNet(nn.Module):
    def __init__(self, c):
        super().__init__()
        def block(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.GroupNorm(4, o), nn.ReLU())
        self.net = nn.Sequential(block(c, 16), block(16, 32), nn.MaxPool2d(2), block(32, 48), nn.MaxPool2d(2), block(48, 64),
                                 nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(64, 2))

    def forward(self, x):
        return self.net(x)        # not normalized: the length says how sure the network is


def fit(img, uv, target, weight, device, seed=0, epochs=120):
    """A network from patches of img (N, C, 96, 96) around uv to the image directions target (N, 2); every patch is
    shifted a little and turned by a random angle with its label, and counts with weight (N,)."""
    torch.manual_seed(seed)
    net = AngleNet(img.shape[1]).to(device)
    opt = torch.optim.AdamW(net.parameters(), 2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=epochs * ((len(img) + 63) // 64))
    target, weight = torch.as_tensor(target), torch.as_tensor(weight, dtype=torch.float32)
    k = R_TURN - R_PATCH
    for _ in range(epochs):
        for b in torch.randperm(len(img)).split(64):
            shift = torch.randint(-JITTER, JITTER + 1, (len(b), 2))
            big = crop(img[b], uv[b.numpy()], shift, R_TURN).to(device)
            x, y = turn(big, target[b].to(device), 2 * np.pi * torch.rand(len(b), device=device))
            out = net(x[:, :, k:-k, k:-k])
            w = weight[b].to(device)
            loss = (((out - y) ** 2).sum(-1) * w).sum() / w.sum().clamp_min(1e-6)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
    return net.eval()


@torch.no_grad()
def predict(net, img, uv, device):
    return torch.cat([net(crop(img[i:i + 256], uv[i:i + 256]).to(device)).cpu() for i in range(0, len(img), 256)])


def stacked(x):
    return torch.cat([x[k] for k in SENSORS], 1)


def fuse(vs, r):
    """Per-sensor vectors (N, M, 2) and gate entries (N, M) -> one unit vector: the sensors weighted by the cue's trust
    in them and by how sure each is (the length of its vector)."""
    w = torch.as_tensor(r, dtype=torch.float32).clamp_min(1e-3)
    return nn.functional.normalize((vs * w[..., None]).sum(1), dim=-1)


def error_deg(v, t):
    """Angle (deg, 0 to 90) between two directions given as (cos 2a, sin 2a)."""
    v, t = (nn.functional.normalize(torch.as_tensor(x, dtype=torch.float32), dim=-1) for x in (v, t))
    c = (v * t).sum(-1).clamp(-1, 1)
    return (0.5 * torch.rad2deg(torch.acos(c))).numpy()


class HindsightAngle:
    """The trained angle: the raw first frame of the three sensors and the cue vector -> (cos 2a, sin 2a) of the tool."""
    def __init__(self, path, device="cpu"):
        ck = torch.load(path, map_location="cpu", weights_only=False)
        self.mode, self.device, self.cam = ck["mode"], device, ck["cam"]
        self.nets = {}
        for k, sd in ck["nets"].items():
            self.nets[k] = AngleNet(C_ALL if k == "all" else SENSORS[k]).to(device).eval()
            self.nets[k].load_state_dict(sd)

    @torch.no_grad()
    def __call__(self, raw, cue):
        x = {k: torch.from_numpy(np.array(raw[k]))[None].permute(0, 3, 1, 2) for k in SENSORS}
        uv = np.asarray(cue[3:5], np.float32)[None] * S
        if self.mode == "early":
            v = predict(self.nets["all"], stacked(x), uv, self.device)
        else:
            vs = torch.stack([predict(self.nets[k], x[k], uv, self.device) for k in SENSORS], 1)
            v = fuse(vs, np.asarray(cue[-len(SENSORS):], np.float32)[None])
        a = to_world(self.cam, uv, np.asarray(cue[2:3], np.float64), ang(v.numpy()))     # image direction -> tool angle
        return vec(a)[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("train")
    ap.add_argument("--cue", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--level", default="V2_align")
    ap.add_argument("--event", default="grasp", choices=("grasp", "release"))
    ap.add_argument("--mode", default="gated", choices=("gated", "early"))
    ap.add_argument("--frames", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--folds", type=int, default=4)
    ap.add_argument("-d", "--device", default="cuda:0")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    cue_ck = torch.load(Path(a.cue) / "cue.pt", map_location="cpu", weights_only=False)
    assert cue_ck["cam"].get("event", "grasp") == a.event, "the cue of the same event is needed"
    x, goal, ep, cells = load(a.train, a.frames, a.event, cue_ck["cam"]["table_z"])
    cue = np.load(Path(a.cue) / "train_cue.npy")             # per demo, out of fold: point, pixel, confidence, gates
    cam, n_ep = cue_ck["cam"], len(cue)
    world = event_angle(a.train, a.event)                    # the tool's angle of every demo, in hindsight
    assert len(world) == n_ep == ep.max() + 1
    target = vec(to_image(cam, goal, world))                 # as a direction in the image at the hindsight point
    uv, r = cue[:, 3:5] * S, cue[:, -len(SENSORS):]
    fold = np.random.default_rng(0).permutation(n_ep) % a.folds
    first = np.arange(n_ep) * a.frames
    keys = ["all"] if a.mode == "early" else list(SENSORS)
    img = {"all": stacked(x)} if a.mode == "early" else x
    weight = {"all": np.ones(n_ep, np.float32)} if a.mode == "early" else {k: r[:, m] for m, k in enumerate(SENSORS)}

    oof = {k: torch.zeros(n_ep, 2) for k in keys}
    for f in range(a.folds):
        tr = np.flatnonzero(fold[ep] != f)
        te = np.flatnonzero(fold == f)
        for k in keys:
            net = fit(img[k][tr], uv[ep[tr]], target[ep[tr]], weight[k][ep[tr]], a.device, seed=f, epochs=a.epochs)
            oof[k][te] = predict(net, img[k][first[te]], uv[te], a.device)
        print(f"fold {f} trained", flush=True)
    est = nn.functional.normalize(oof["all"], dim=-1) if a.mode == "early" else \
        fuse(torch.stack([oof[k] for k in SENSORS], 1), r)
    back = lambda v: vec(to_world(cam, uv, cue[:, 2], ang(np.asarray(v))))       # image direction -> the tool's angle
    np.save(out / "train_ang.npy", back(est))
    nets = {k: fit(img[k], uv[ep], target[ep], weight[k][ep], a.device, seed=a.folds, epochs=a.epochs) for k in keys}
    torch.save(dict(mode=a.mode, event=a.event, cam=cam, nets={k: n.state_dict() for k, n in nets.items()},
                    args=dict(frames=a.frames, epochs=a.epochs, folds=a.folds, level=a.level)), out / "ang.pt")

    truth = vec(world)
    lines = [f"hindsight angle, event {a.event}, mode {a.mode}: {n_ep} demos of {a.train}, out of fold; error of the "
             "tool's angle in degrees (0 to 90)", "",
             "| cells | demos | " + " | ".join(f"{k}: median / within 15 deg" for k in keys)
             + (" | the estimate: median / within 10 / within 15 deg |" if a.mode != "early" else " |"),
             "|---|---|" + "---|" * (len(keys) + (a.mode != "early"))]
    err = {k: error_deg(back(oof[k]), truth) for k in keys}
    e = error_deg(back(est), truth)
    groups = [("all", np.ones(n_ep, bool))] + ([(str(c), cells == c) for c in np.unique(cells)] if cells is not None else [])
    for name, m in groups:
        row = f"| {name} | {m.sum()} | " + " | ".join(f"{np.median(err[k][m]):.1f} / {(err[k][m] < 15).mean():.2f}" for k in keys)
        if a.mode != "early":
            row += f" | {np.median(e[m]):.1f} / {(e[m] < 10).mean():.2f} / {(e[m] < 15).mean():.2f}"
        lines.append(row + " |")
    (out / "report.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
