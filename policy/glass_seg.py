"""
Where the glass is, for the target-visibility cues (policy/obs_util.py visibility_cues).

A glass segmentation network (fusion/, train_seg.py checkpoint, e.g. data/fusion_bench/runs/mix_ws/best.pt) predicts the
glass mask of a frame from rgb + depth + thermal. The policy does not see the mask or the network's features: the mask
only says where to measure whether each sensor shows the glass.

Used in two places with the same preprocessing:
  * policy/make_glass_masks.py runs it once over a demo zarr (GPU) and saves the masks next to it, so training reads them;
  * the eval env (policy/obs_util.to_policy_obs, one frame per step, CPU, one thread) when a frame has no stored mask.
"""
import numpy as np
import torch

_models = {}


def load_seg(ckpt, device="cpu"):
    key = (str(ckpt), str(device))
    if key not in _models:
        from fusion.model import load_checkpoint
        model, _ = load_checkpoint(ckpt, map_location="cpu")
        _models[key] = model.eval().to(device)
    return _models[key]


@torch.no_grad()
def predict_glass_mask(raw, ckpt, device="cpu", batch=256, thr=0.5):
    """raw: dict with rgb (..., H, W, 3), depth (..., H, W, 1), thermal (..., H, W, 1), uint8. Returns (..., H, W) bool.
    Inputs as the segmentation network was trained: rgb and depth / 255, thermal min-max normalized per frame."""
    if device == "cpu":
        torch.set_num_threads(1)          # env workers are forked processes; one thread each, like DataLoader workers
    model = load_seg(ckpt, device)
    lead = raw["rgb"].shape[:-3]
    flat = {k: np.asarray(raw[k]).reshape((-1,) + raw[k].shape[-3:]) for k in ("rgb", "depth", "thermal")}
    out = []
    for i in range(0, len(flat["rgb"]), batch):
        x = {k: torch.from_numpy(v[i:i + batch]).to(device).permute(0, 3, 1, 2).float() / 255 for k, v in flat.items()}
        t = x["thermal"]
        lo, hi = t.amin((1, 2, 3), keepdim=True), t.amax((1, 2, 3), keepdim=True)
        x["thermal"] = (t - lo) / (hi - lo).clamp(min=1e-6)
        logits = model({m: x[m] for m in model.mods})["logits"]
        out.append((torch.sigmoid(logits[:, 0]) > thr).cpu().numpy())
    mask = np.concatenate(out)
    return mask.reshape(lead + mask.shape[-2:])
