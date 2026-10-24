"""
Pretrain the thermal encoder by glass segmentation on TRansPose raw thermal.

  python -m encoder.train --config encoder/configs/transpose_seg.yaml [key=value ...]
  python -m encoder.train --eval <ckpt> [key=value ...]
"""
import argparse
import csv
import json
import os
import random
import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from common.thermal import normalize_thermal_raw
from encoder.dataset import ThermalSegDataset, list_sequences, load_transpose, random_arrays, split_sequences
from encoder.model import ThermalSegNet, make_thermal_resnet18, seg_loss

BINS = ("low", "mid", "high")


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_datasets(cfg, splits=("train", "val")):
    """-> ({split: ThermalSegDataset}, {split: sequence ids})."""
    d = cfg.data
    if d.name == "random":
        arrays = random_arrays(d.n_random, d.image_size, cfg.seed)
        n_val = d.n_random // 5
        parts = {"train": [a[n_val:] for a in arrays], "val": [a[:n_val] for a in arrays]}
        seqs = {"train": ["random"], "val": ["random"]}
    else:
        all_seqs = sorted(d.sequences or list_sequences(d.root))
        val = sorted(d.val_sequences) if d.val_sequences else split_sequences(all_seqs, d.val_frac, cfg.seed)[1]
        seqs = {"train": [s for s in all_seqs if s not in val], "val": val}
        parts = {k: load_transpose(d.root, seqs[k], d.image_size, d.cache_dir) for k in splits}
    norm_params = OmegaConf.to_container(d.norm_params) if d.norm_params else None
    aug = OmegaConf.to_container(cfg.aug)
    ds = {k: ThermalSegDataset(normalize_thermal_raw(parts[k][0], d.norm_mode, norm_params), *parts[k][1:],
                               train=k == "train", crop_size=d.crop_size, aug=aug) for k in splits}
    return ds, seqs


def center_crop(x, size):
    i, j = (x.shape[-2] - size) // 2, (x.shape[-1] - size) // 2
    return x[..., i:i + size, j:j + size]


def _scores(tp, fp, fn):
    return tp / max(tp + fp + fn, 1), 2 * tp / max(2 * tp + fp + fn, 1)


@torch.no_grad()
def evaluate(model, loader, crop, cnr_edges, device):
    """Pixel-accumulated IoU / F1 on the DP-style center crop (primary) and on the full frame (reference),
    plus center-crop IoU per CNR bin (frames without glass only count in the totals)."""
    model.eval()
    counts = {k: np.zeros(3) for k in ("crop", "full") + BINS}  # tp, fp, fn
    loss, n = 0.0, 0
    for b in loader:
        x, m = b["thermal"].to(device), b["mask"].to(device)
        for name, xi, mi in (("crop", center_crop(x, crop), center_crop(m, crop)), ("full", x, m)):
            logits = model(xi)
            pred, gt = logits > 0, mi > 0.5
            per = torch.stack([(pred & gt), (pred & ~gt), (~pred & gt)], 1).flatten(2).sum(2).cpu().numpy()
            counts[name] += per.sum(0)
            if name == "crop":
                loss += seg_loss(logits.float(), mi).item() * len(x)
                n += len(x)
                cnr = b["cnr"].numpy()
                bin_idx = np.digitize(cnr, cnr_edges)
                for k, bin_name in enumerate(BINS):
                    counts[bin_name] += per[(bin_idx == k) & ~np.isnan(cnr)].sum(0)
    iou, f1 = _scores(*counts["crop"])
    iou_full, f1_full = _scores(*counts["full"])
    out = {"val_loss": loss / max(n, 1), "iou": iou, "f1": f1, "iou_full96": iou_full, "f1_full96": f1_full}
    out.update({f"iou_cnr_{k}": _scores(*counts[k])[0] for k in BINS})
    return out


@torch.no_grad()
def save_grid(model, ds, crop, n, path, device):
    """Rows = val frames spread from lowest to highest CNR; columns = thermal input | GT mask | predicted mask."""
    model.eval()
    order = np.argsort(np.nan_to_num(ds.cnr, nan=-1.0))
    idx = order[np.linspace(0, len(order) - 1, min(n, len(order))).astype(int)]
    x, m = center_crop(ds.thermal[idx], crop), center_crop(ds.mask[idx], crop)
    pred = (model(x.to(device)) > 0).float().cpu()
    save_image(torch.stack([x, m, pred], 1).flatten(0, 1), path, nrow=3, padding=2, pad_value=0.5)


def make_model(cfg, device):
    return ThermalSegNet(make_thermal_resnet18(cfg.init)).to(device)


def train(cfg):
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    os.makedirs(cfg.out_dir, exist_ok=True)
    OmegaConf.save(cfg, os.path.join(cfg.out_dir, "config.yaml"), resolve=True)
    seed_everything(cfg.seed)
    ds, seqs = build_datasets(cfg)
    with open(os.path.join(cfg.out_dir, "splits.json"), "w") as f:
        json.dump(seqs, f, indent=1)
    edges = list(cfg.eval.cnr_edges or np.nanquantile(ds["train"].cnr, [1 / 3, 2 / 3]).tolist())
    t = cfg.train
    train_loader = DataLoader(ds["train"], t.batch_size, shuffle=True, drop_last=True, num_workers=t.num_workers,
                              pin_memory=True, generator=torch.Generator().manual_seed(cfg.seed))
    val_loader = DataLoader(ds["val"], t.batch_size, num_workers=t.num_workers)

    model = make_model(cfg, device)
    opt = torch.optim.AdamW(model.parameters(), lr=t.lr, weight_decay=t.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, t.epochs * len(train_loader))
    amp = t.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler(enabled=amp)
    best, log_path = -1.0, os.path.join(cfg.out_dir, "log.csv")
    print(f"train {len(ds['train'])} / val {len(ds['val'])} frames, CNR edges {edges}, device {device}")
    for epoch in range(1, t.epochs + 1):
        model.train()
        total = 0.0
        for b in train_loader:
            x, m = b["thermal"].to(device, non_blocking=True), b["mask"].to(device, non_blocking=True)
            with torch.autocast(device.type, enabled=amp):
                logits = model(x)
            loss = seg_loss(logits.float(), m)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            total += loss.item() * len(x)
        row = {"epoch": epoch, "train_loss": total / (len(train_loader) * t.batch_size), "lr": sched.get_last_lr()[0],
               **evaluate(model, val_loader, cfg.data.crop_size, edges, device)}
        with open(log_path, "w" if epoch == 1 else "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row))
            if epoch == 1:
                w.writeheader()
            w.writerow(row)
        ckpt = {"model": model.state_dict(), "cfg": OmegaConf.to_container(cfg, resolve=True),
                "epoch": epoch, "metrics": row, "cnr_edges": edges}
        torch.save(ckpt, os.path.join(cfg.out_dir, "last.pt"))
        if row["iou"] > best:
            best = row["iou"]
            torch.save(ckpt, os.path.join(cfg.out_dir, "best.pt"))
        print(" ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()))
    save_grid(model, ds["val"], cfg.data.crop_size, cfg.eval.n_vis, os.path.join(cfg.out_dir, "val_grid.png"), device)


def run_eval(ckpt_path, overrides):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = OmegaConf.merge(OmegaConf.create(ckpt["cfg"]), overrides)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    seed_everything(cfg.seed)
    ds = build_datasets(cfg, splits=("val",))[0]["val"]
    model = make_model(cfg, device)
    model.load_state_dict(ckpt["model"])
    loader = DataLoader(ds, cfg.train.batch_size, num_workers=cfg.train.num_workers)
    edges = list(cfg.eval.cnr_edges or ckpt["cnr_edges"])  # eval.cnr_edges=[a,b] overrides the train tertiles
    metrics = {"epoch": ckpt["epoch"], "cnr_edges": edges, **evaluate(model, loader, cfg.data.crop_size, edges, device)}
    out = os.path.splitext(ckpt_path)[0] + ("_cnr" + "-".join(f"{e:g}" for e in edges) if cfg.eval.cnr_edges else "")
    with open(out + "_eval.json", "w") as f:
        json.dump(metrics, f, indent=1)
    save_grid(model, ds, cfg.data.crop_size, cfg.eval.n_vis, out + "_val_grid.png", device)
    print(json.dumps(metrics, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="encoder/configs/transpose_seg.yaml")
    ap.add_argument("--eval", metavar="CKPT", help="evaluate a checkpoint on its val split instead of training")
    args, rest = ap.parse_known_args()
    overrides = OmegaConf.from_dotlist(rest)
    if args.eval:
        run_eval(args.eval, overrides)
    else:
        train(OmegaConf.merge(OmegaConf.load(args.config), overrides))
