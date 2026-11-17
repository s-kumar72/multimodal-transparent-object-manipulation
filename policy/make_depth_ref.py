"""
Empty-table reference depth of the fixed front camera, for the optional depth_object cue (policy/obs_util.py).

Per pixel, the 90th percentile of the 8-bit depth (255 = 2 m, larger = farther) over the first frame of every
episode of a demo set: objects and the arm stand on or above the table, so the far end of each pixel's values is
the table. The camera and the table do not change between levels, so one file serves every level.
  python policy/make_depth_ref.py results/data/sim_demos_L0_bench_500.zarr policy/config/depth_ref_front_cam.npy
"""
import sys

import numpy as np
import zarr


def main(zarr_path, out):
    z = zarr.open(zarr_path, "r")
    ends = z["meta/episode_ends"][:]
    starts = np.r_[0, ends[:-1]]
    first = np.stack([z["data/depth"][s][..., 0] for s in starts])
    ref = np.percentile(first, 90, axis=0).round().astype(np.uint8)
    np.save(out, ref)
    print(f"{out}: {ref.shape}, from {len(starts)} episodes; rows 32-96 range {ref[32:].min()}-{ref[32:].max()}")


if __name__ == "__main__":
    main(*sys.argv[1:3])
