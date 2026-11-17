"""
Train the reliability-aware RGB-D-T glass segmentation network (fusion/model.py) on sim and / or real data.

  python -m fusion.train_seg --name mix --modalities rgb depth thermal \
      --train "sim:data/fusion/sim/*_train.npz" real:rgbt_glass:train real:mju_glass:train \
      --val "sim:data/fusion/sim/*_val.npz" real:rgbt_glass:val \
      --eval "sim:data/fusion/sim/*_test_*.npz" real:rgbt_glass:test --iters 20000

Writes data/fusion/runs/<name>/: args.json, log.jsonl (training + validation), best.pt (best mean val IoU),
last.pt, final.json (best.pt on every --eval set, clean and corrupted), vis_<set>.png.
"""
import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, WeightedRandomSampler

from fusion.data import build_train_set, collate, real_available
from fusion.evaluate import CORRUPTIONS, eval_sets, evaluate
from fusion.model import DEFAULT_CFG, build_model, gate_reliability_loss, load_encoder_weights, seg_loss


def parse():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--name", required=True)
    p.add_argument("--out_dir", default="data/fusion/runs")
    p.add_argument("--train", nargs="+", required=True)
    p.add_argument("--val", nargs="*", default=[])
    p.add_argument("--eval", nargs="*", default=[])
    p.add_argument("--corruptions", nargs="*", default=list(CORRUPTIONS))
    p.add_argument("--sim_weight", type=float, default=0.5, help="share of sim samples when mixing sim and real")
    # model
    p.add_argument("--modalities", nargs="+", default=DEFAULT_CFG["modalities"])
    p.add_argument("--backbone", default=DEFAULT_CFG["backbone"])
    p.add_argument("--fusion_stride", type=int, default=DEFAULT_CFG["fusion_stride"])
    p.add_argument("--num_layers", type=int, default=DEFAULT_CFG["num_layers"])
    p.add_argument("--d_model", type=int, default=DEFAULT_CFG["d_model"])
    p.add_argument("--fusion_mode", default=DEFAULT_CFG["fusion_mode"], choices=["paper", "weighted_sum"])
    p.add_argument("--gate_mode", default=DEFAULT_CFG["gate_mode"], choices=["learned", "static"])
    p.add_argument("--gate_norm", default=DEFAULT_CFG["gate_norm"], choices=["sigmoid", "softmax"])
    p.add_argument("--no_cues", action="store_true", help="no explicit reliability cues (= the paper)")
    p.add_argument("--gate_sup", type=float, default=0.1,
                   help="weight of the gate reliability loss (known-unreliable sensors -> low gate); 0 = paper")
    p.add_argument("--gate_logit_max", type=float, default=DEFAULT_CFG["gate_logit_max"],
                   help="bound on gate logits (tanh); 0 = unbounded (paper)")
    p.add_argument("--no_pretrained", action="store_true", help="RGB encoder from scratch")
    p.add_argument("--init", default=None, help="start from this train_seg checkpoint (e.g. real -> sim fine-tuning)")
    p.add_argument("--init_thermal", default=None, help="DP-format ResNet state dict for the thermal encoder")
    # data
    p.add_argument("--size", type=int, default=96, help="image size (96 = Diffusion Policy observation)")
    p.add_argument("--crop", type=int, default=84, help="random crop in training, center crop in eval (DP: 84)")
    p.add_argument("--sim_thermal", default="minmax", choices=["minmax", "env"],
                   help="minmax: per-frame min-max of the raw counts, same as the real loader; env: as the policy sees it")
    p.add_argument("--real_root", default=None)
    p.add_argument("--no_aug", action="store_true", help="no sensor corruptions")
    # optimization
    p.add_argument("--iters", type=int, default=20000)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--lr_pretrained", type=float, default=0.1, help="lr factor of ImageNet-pretrained encoders")
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--warmup", type=int, default=500)
    p.add_argument("--eval_every", type=int, default=2000)
    p.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    p.add_argument("--amp", default="auto", choices=["auto", "bf16", "fp16", "off"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def drop_unavailable_real(specs, root=None):
    """Real sources need the converted real data; skip them (with a warning) if it is not there."""
    if any(s.startswith("real:") for s in specs) and not real_available(root):
        print("[fusion] WARNING: no unified real data (real_data/README.md), skipping", [s for s in specs if s.startswith("real:")])
        return [s for s in specs if not s.startswith("real:")]
    return specs


def save_vis(model, ds, path, device, n=8):
    """Rows: rgb | depth | thermal | ground truth | prediction | one gate map per modality (bright = trusted)."""
    model.eval()
    idx = np.linspace(0, len(ds) - 1, min(n, len(ds))).astype(int)
    b = collate([ds[i] for i in idx])
    with torch.no_grad():
        out = model({m: b[m].to(device) for m in model.mods}, b["avail"].to(device))
    model.train()
    H = b["rgb"].shape[-1]
    up = lambda g: torch.nn.functional.interpolate(g, size=(H, H), mode="nearest")
    cols = [b["rgb"], b["depth"].expand(-1, 3, -1, -1), b["thermal"].expand(-1, 3, -1, -1),
            b["mask"].expand(-1, 3, -1, -1), torch.sigmoid(out["logits"].float().cpu()).expand(-1, 3, -1, -1)]
    g = out["gates"].float().cpu()
    cols += [up(g[:, k:k + 1]).expand(-1, 3, -1, -1) for k in range(g.shape[1])]
    grid = torch.cat([torch.cat(list(c), 1) for c in cols], 2)              # (3, n*H, len(cols)*H)
    Image.fromarray((grid.permute(1, 2, 0).clamp(0, 1).numpy() * 255).astype(np.uint8)).save(path)


def main():
    a = parse()
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    out = Path(a.out_dir) / a.name
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(a.device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    amp = {"bf16": torch.bfloat16, "fp16": torch.float16, "off": None}.get(a.amp)
    if a.amp == "auto":
        amp = (torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16) if device.type == "cuda" else None

    a.train, a.val, a.eval = (drop_unavailable_real(s, a.real_root) for s in (a.train, a.val, a.eval))
    assert a.train, "no training data"
    json.dump(vars(a), open(out / "args.json", "w"), indent=1)
    load_kw = dict(sim_thermal=a.sim_thermal, real_root=a.real_root)

    # ---------------------------------------------------------------- model
    cfg = dict(modalities=a.modalities, backbone=a.backbone, fusion_stride=a.fusion_stride, num_layers=a.num_layers,
               d_model=a.d_model, fusion_mode=a.fusion_mode, gate_mode=a.gate_mode, gate_norm=a.gate_norm, reliability_cues=not a.no_cues,
               gate_logit_max=a.gate_logit_max or None,
               pretrained_rgb=not a.no_pretrained and a.init is None)
    model = build_model(cfg)
    if a.init:
        ck = torch.load(a.init, map_location="cpu", weights_only=False)
        missing, unexpected = model.load_state_dict(ck["model"], strict=False)
        print(f"[fusion] init from {a.init}: missing {len(missing)}, unexpected {len(unexpected)}")
    if a.init_thermal and "thermal" in a.modalities:
        load_encoder_weights(model, "thermal", a.init_thermal)
    model.to(device)
    pre = [p for n, p in model.named_parameters() if n.startswith("encoders.rgb.") and cfg["pretrained_rgb"]]
    pre_ids = {id(p) for p in pre}
    rest = [p for p in model.parameters() if id(p) not in pre_ids]
    opt = torch.optim.AdamW([{"params": rest, "lr": a.lr}, {"params": pre, "lr": a.lr * a.lr_pretrained}],
                            weight_decay=a.weight_decay)
    base_lrs = [g["lr"] for g in opt.param_groups]
    scaler = torch.amp.GradScaler(device.type, enabled=amp == torch.float16)
    print(f"[fusion] {a.name}: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M parameters, "
          f"modalities {a.modalities}, gates {a.gate_mode}/{a.gate_norm}, cues {not a.no_cues}, amp {amp}", flush=True)

    # ---------------------------------------------------------------- data
    aug = dict(thermal_crossover=0, depth_holes=0, depth_wrong=0, rgb_lowlight=0, blank=0, blank_rgb=0, drop=0) if a.no_aug else None
    train_ds, weights = build_train_set(a.train, a.modalities, a.size, a.crop, aug, a.sim_weight, **load_kw)
    sampler = WeightedRandomSampler(weights, a.iters * a.batch_size, replacement=True)
    loader = DataLoader(train_ds, a.batch_size, sampler=sampler, num_workers=a.workers, collate_fn=collate,
                        pin_memory=device.type == "cuda", drop_last=True, persistent_workers=a.workers > 0)
    print(f"[fusion] train: {len(train_ds)} samples from {[d.name for d in train_ds.datasets]}", flush=True)
    val_sets = eval_sets(a.val, a.modalities, a.size, a.crop, (), **load_kw)
    mk_loader = lambda ds: DataLoader(ds, 128, num_workers=min(4, a.workers), collate_fn=collate)

    # ---------------------------------------------------------------- loop
    log = open(out / "log.jsonl", "a")
    best, t0, run_loss = -1.0, time.time(), []
    model.train()
    for it, b in enumerate(loader, 1):
        lr_f = min(1.0, it / max(a.warmup, 1)) * 0.5 * (1 + math.cos(math.pi * min(it / a.iters, 1.0)))
        for g, lr in zip(opt.param_groups, base_lrs):
            g["lr"] = lr * lr_f
        inp = {m: b[m].to(device, non_blocking=True) for m in a.modalities}
        with torch.autocast(device.type, dtype=amp, enabled=amp is not None):
            o = model(inp, b["avail"].to(device))
        o = {k: (v.float() if torch.is_tensor(v) and v.is_floating_point() else
                 [x.float() for x in v] if isinstance(v, list) else v) for k, v in o.items()}
        loss = seg_loss(o, b["mask"].to(device), b["valid"].to(device))
        if a.gate_sup > 0 and a.gate_mode == "learned":
            loss = loss + a.gate_sup * gate_reliability_loss(o, b["unreliable"].to(device), b["labelled"].to(device))
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        run_loss.append(loss.item())
        if it % 50 == 0 or it == 1:
            gm = o["gates"].detach().mean((0, 2, 3)).tolist()
            rec = dict(it=it, loss=float(np.mean(run_loss)), lr=a.lr * lr_f, it_s=it / (time.time() - t0),
                       **{f"gate_{m}": g for m, g in zip(a.modalities, gm)})
            log.write(json.dumps(rec) + "\n")
            log.flush()
            print(f"it {it:6d}/{a.iters}  loss {rec['loss']:.4f}  {rec['it_s']:.1f} it/s  gates "
                  + " ".join(f"{m} {g:.2f}" for m, g in zip(a.modalities, gm)), flush=True)
            run_loss = []
        if it % a.eval_every == 0 or it == a.iters:
            ck = dict(model=model.state_dict(), model_cfg=model.cfg, args=vars(a), it=it)
            torch.save(ck, out / "last.pt")
            if val_sets:
                res = {n: evaluate(model, mk_loader(ds), device, amp) for n, ds in val_sets}
                score = float(np.mean([r["iou"] for r in res.values()]))
                log.write(json.dumps(dict(it=it, val=res, val_iou=score)) + "\n")
                log.flush()
                print(f"  val it {it}: mean IoU {score:.4f}  " +
                      "  ".join(f"{n} {r['iou']:.3f}/{r['loc']:.2f}" for n, r in res.items()), flush=True)
            else:
                score = it
            if score > best:
                best = score
                torch.save(ck, out / "best.pt")
        if it >= a.iters:
            break
    log.close()

    # ---------------------------------------------------------------- final evaluation of best.pt
    model.load_state_dict(torch.load(out / "best.pt", map_location=device, weights_only=False)["model"])
    final = {}
    for name, ds in eval_sets(a.eval, a.modalities, a.size, a.crop, a.corruptions, **load_kw):
        final[name] = evaluate(model, mk_loader(ds), device, amp)
        r = final[name]
        print(f"  test {name}: IoU {r['iou']:.3f}  loc {r['loc']:.2f}  gates "
              + " ".join(f"{m} {r['gate_' + m]:.2f}" for m in a.modalities), flush=True)
        if "+" not in name:
            save_vis(model, ds, out / f"vis_{name}.png", device)
    json.dump(dict(args=vars(a), best_val_iou=best, test=final), open(out / "final.json", "w"), indent=1)
    print(f"[fusion] {a.name} done in {(time.time() - t0) / 60:.1f} min -> {out}", flush=True)


if __name__ == "__main__":
    main()