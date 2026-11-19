"""
Glass masks of every frame of a demo zarr, from the glass segmentation network (policy/glass_seg.py), saved as one
uint8 array (N, H, W, 1), 255 = glass, aligned with the zarr's frames. The dataset reads it through the cue key's
`glass_mask` entry so that training computes the target-visibility cues without running the network.
  python policy/make_glass_masks.py results/data/sim_demos_L0_bench_500.zarr data/fusion_bench/runs/mix_ws/best.pt \
      results/data/sim_demos_L0_bench_500.glass_mask.npy [device] [sensors, e.g. rgb or rgb,depth]
The optional last argument restricts the sensors the network may use (the others are marked as not available).
"""
import sys

import numpy as np
import zarr

from policy.glass_seg import predict_glass_mask


def main(zarr_path, ckpt, out, device="cuda", mods=None):
    mods = mods.split(",") if mods else None
    z = zarr.open(zarr_path, "r")["data"]
    n = z["rgb"].shape[0]
    masks = np.zeros((n,) + z["rgb"].shape[1:3] + (1,), np.uint8)
    for i in range(0, n, 2048):
        raw = {k: z[k][i:i + 2048] for k in ("rgb", "depth", "thermal")}
        masks[i:i + 2048, ..., 0] = predict_glass_mask(raw, ckpt, device, mods=mods) * 255
    np.save(out, masks)
    px = (masks[..., 0] > 0).reshape(n, -1).sum(1)
    print(f"{out}: {masks.shape}, glass pixels per frame median {int(np.median(px))}, frames with < 5 pixels {(px < 5).sum()}")


if __name__ == "__main__":
    main(*sys.argv[1:6])
