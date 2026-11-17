"""
Evaluate a glass-segmentation checkpoint on sim / real sources, clean and under sensor corruptions.

  python -m fusion.evaluate --ckpt data/fusion/runs/mix/best.pt --eval "sim:data/fusion/sim/L4b_test_*.npz" \
      real:rgbt_glass:test --out data/fusion/runs/mix/eval.json

Metrics per source (masks binarized at 0.5; sim ground truth is the fraction of the pixel covered, so >= 0.5):
  iou        intersection / union summed over the whole set (Huo et al.'s IoU is per image; see miou)
  miou       mean per-image IoU over images that contain glass
  mae, ber   mean absolute error of the probability map; balanced error rate (%)
  loc        images with glass whose most-confident pixel is on the glass (within 2 px): "selects the right object"
  gate_<m>   mean fusion gate of modality m over all tokens, gate_fg_<m> over tokens on glass
             (the learned reliability weight; compare clean vs thermal_crossover / depth_holes / depth_seethrough)

Sim frames are also split by episode type (needs glass_seethrough param from collect_sim.py):
    miou_normal / miou_seethrough / miou_crossover          per-image IoU of each group
    gate_fg_<m>_normal / gate_fg_<m>_seethrough             weight of modality m on the glass in each group
    auc_depth_seethrough                                    how well the depth weight on the glass separates normal from see-through frames
    auc_thermal_crossover                                   the same for the thermal weight, contrast vs. crossover frames (0.5 = no separation, 1 = perfect)
"""
import argparse
import json

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from fusion.attention.data import MODS, Augment, SegArrays, collate, expand_sources, load_source, source_name

CORRUPTIONS = {"thermal_crossover": (2, dict(thermal_crossover=1.0)), "depth_holes": (1, dict(depth_holes=1.0)),
               "thermal_blank": (2, dict(blank=1.0)), "depth_seethrough": (1, dict(depth_seethrough=1.0))}


class CorruptedEval(SegArrays):
    """Center crop plus one corruption on every sample, with a fixed seed per sample (repeatable)."""

    def __init__(self, arrays, name, crop, modalities, corruption):
        super().__init__(arrays, name, crop, train=False, modalities=modalities)
        k, p = CORRUPTIONS[corruption]
        if corruption == "thermal_blank":
            def blank(item):
                if item["avail"][2]:
                    item["thermal"] = torch.zeros_like(item["thermal"])
                    item["corrupt"][2] = item["unreliable"][2] = True
            self.aug = blank
        else:
            base = dict(thermal_crossover=0, depth_holes=0, depth_wrong=0, rgb_lowlight=0, blank=0, blank_rgb=0, drop=0)
            self.aug = Augment(**{**base, **p})

    def __getitem__(self, i):
        st = np.random.get_state()
        np.random.seed(i)
        torch.manual_seed(i)
        item = super().__getitem__(i)
        np.random.set_state(st)
        return item

def _auc(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    pos, neg = pos[~np.isnan(pos)], neg=[~np.isnan(neg)]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    return float((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean())

def group_stats(per_img, mods):
    """
    Per-image IoU of the normal/ see-through/ crossover frames (sim; NaN for real data), the sensor weights on 
    the glass in normal and see through frames, and two AUCs: depth weight on the glass higher in normal than in 
    see-through frames, thermal weight on the glass higher without than with crossover (0.5 = no separation).
    """
    iou, st, xo = (np.asarray(per_img[k], float) for k in ("iou", "st", "xo"))
    G = np.asarray(per_img["g_fg"]. float).reshape(len(iou), -1)
    out = {}

    for name, sel in (("normal", st == 0), ("seethrough", st == 1), ("crossover", xo == 1)):
        x = iou[sel]
        out[f"miou_{name}"] = float(np.nanmean(x)) if np.isfinite(x).any() else float("nan")
    for k, m in enumerate(mods):
        for name, sel in (("normal", st == 0), ("seethrough", st == 1)):
            x = G[sel, k]
            out[f"gate_fg_{m}_{name}"] = float(np.nanmean(x)) if np.isfinite(x).any() else float("nan")
    if "depth" in mods:
        k = mods.index("depth")
        out["auc_depth_seethrough"] = _auc(G[st == 0, k], G[st == 1, k])
    if "thermal" in mods:
        k = mods.index("thermal")
        out["auc_thermal_crossover"] = _auc(G[xo == 0, k], G[xo == 1, k])
    
    return out

@torch.no_grad()
def evaluate(model, loader, device, amp_dtype=None, max_batches=None):
    model.eval()
    mods = model.mods
    tot = dict(inter=0.0, union=0.0, ae=0.0, n_pix=0.0, tp=0.0, tn=0.0, pos=0.0, neg=0.0)
    ious, locs = [], []
    g_sum, g_fg, g_n, g_fg_n = torch.zeros(len(mods)), torch.zeros(len(mods)), 0, torch.zeros(len(mods))
    g_x = {0: [torch.zeros(len(mods)), 0], 1: [torch.zeros(len(mods)), 0]}   # by is_crossover
    per_img = {"iou": [], "st": [], "xo": [], "g_fg": []}
    for bi, b in enumerate(loader):
        if max_batches and bi >= max_batches:
            break
        inp = {m: b[m].to(device, non_blocking=True) for m in mods}
        avail = b["avail"].to(device)
        with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            out = model(inp, avail)
        p = torch.sigmoid(out["logits"].float()).cpu()
        gt = (b["mask"] >= 0.5).float()
        v = b["valid"]
        pb = (p >= 0.5).float()
        inter, union = (pb * gt * v).flatten(1).sum(1), (((pb + gt) > 0).float() * v).flatten(1).sum(1)
        tot["inter"] += inter.sum().item()
        tot["union"] += union.sum().item()
        tot["ae"] += ((p - b["mask"]).abs() * v).sum().item()
        tot["n_pix"] += v.sum().item()
        tot["tp"] += (pb * gt * v).sum().item()
        tot["tn"] += ((1 - pb) * (1 - gt) * v).sum().item()
        tot["pos"] += (gt * v).sum().item()
        tot["neg"] += ((1 - gt) * v).sum().item()
        has = (gt * v).flatten(1).sum(1) > 0
        ious += (inter / union.clamp(min=1))[has].tolist()
        near = F.max_pool2d(gt, 5, 1, 2)                                   # glass dilated by 2 px
        flat = (p * v).flatten(1).argmax(1)
        locs += near.flatten(1).gather(1, flat[:, None])[:, 0][has].tolist()
        g = out["gates"].float().cpu()                                      # (B, M, h, w)
        g_sum += g.mean((2, 3)).sum(0)
        g_n += len(g)
        fg_tok = F.adaptive_max_pool2d(gt, g.shape[-2:])                    # tokens touching glass
        g_fg += (g * fg_tok).sum((0, 2, 3))
        g_fg_n += fg_tok.sum()
        img_iou = (inter / union.clamp(min=1)).masked_fill(~has, float("nan"))
        g_fg_img = (g * fg_tok).sum((2, 3)) / fg_tok.sum((2, 3)).clamp(min=1)          # (B, M) gate on the glass
        g_fg_img = g_fg_img.masked_fill(fg_tok.sum((2, 3)) == 0, float("nan"))
        per_img["iou"] += img_iou.tolist()
        per_img["st"] += b.get("is_seethrough", torch.full((len(g),), float("nan"))).tolist()
        per_img["xo"] += b["is_crossover"].tolist()
        per_img["g_fg"] += g_fg_img.tolist()
        for c in (0, 1):
            sel = b["is_crossover"] == c
            if sel.any():
                g_x[c][0] += g[sel].mean((2, 3)).sum(0)
                g_x[c][1] += int(sel.sum())
    r = dict(n=g_n, iou=tot["inter"] / max(tot["union"], 1), miou=float(np.mean(ious)) if ious else float("nan"),
             mae=tot["ae"] / max(tot["n_pix"], 1),
             ber=100 * (1 - 0.5 * (tot["tp"] / max(tot["pos"], 1) + tot["tn"] / max(tot["neg"], 1))),
             loc=float(np.mean(locs)) if locs else float("nan"))
    for k, m in enumerate(mods):
        r[f"gate_{m}"] = (g_sum[k] / max(g_n, 1)).item()
        r[f"gate_fg_{m}"] = (g_fg[k] / g_fg_n[k].clamp(min=1)).item()
        for c, name in ((0, "contrast"), (1, "crossover")):
            if g_x[c][1]:
                r[f"gate_{m}_{name}"] = (g_x[c][0][k] / g_x[c][1]).item()
    r.update(group_stats(per_img, mods))
    model.train()
    return r


def eval_sets(specs, modalities, size, crop, corruptions=(), **load_kw):
    """[(name, dataset)] for every source, clean and under each corruption the source can show."""
    sets = []
    for spec in expand_sources(specs):
        arr = load_source(spec, size, **load_kw)
        name = source_name(spec)
        sets.append((name, SegArrays(arr, name, crop, train=False, modalities=modalities)))
        for c in corruptions:
            k = CORRUPTIONS[c][0]
            if MODS[k] in modalities and arr["avail"][:, k].any():
                sets.append((f"{name}+{c}", CorruptedEval(arr, name, crop, modalities, c)))
    return sets


def main():
    from fusion.attention.model import load_checkpoint
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--eval", nargs="+", required=True)
    p.add_argument("--corruptions", nargs="*", default=list(CORRUPTIONS))
    p.add_argument("--out", default=None)
    p.add_argument("--batch_size", type=int, default=64)
    a = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ck = load_checkpoint(a.ckpt)
    model.to(device)
    args = ck["args"]
    res = {}
    for name, ds in eval_sets(a.eval, model.mods, args["size"], args["crop"], a.corruptions,
                              sim_thermal=args["sim_thermal"], real_root=args.get("real_root")):
        res[name] = evaluate(model, DataLoader(ds, a.batch_size, num_workers=4, collate_fn=collate), device)
        print(name, {k: round(v, 4) if isinstance(v, float) else v for k, v in res[name].items()}, flush=True)
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()