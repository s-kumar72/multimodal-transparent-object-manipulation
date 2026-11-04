"""
Print dataset contents & saves a preview image.

run via python sim/inspect_dataset.py data/sim_demos.zarr
Author: Sadhana Kumar
"""

import sys
from pathlib import Path

import numpy as np
import zarr
from PIL import Image

path = sys.argv[1] if len(sys.argv) > 1 else "data/sim_demos.zarr"
root = zarr.open(path, mode="r")
ends = root["meta/episode_ends"][:]
lengths = np.diff(np.concatenate([[0], ends]))

print(f"{path}: {len(ends)} episodes, {ends[-1]} steps "
      f"(episode length {lengths.min()} - {lengths.max()}, mean {lengths.mean():.0f})")

for k, a in root["data"].items():
    print(f" data/{k:11s} shape {str(a.shape):18s} dtype {str(a.dtype):7s} chunks {a.chunks[0]}")
    assert a.shape[0] == ends[-1], f"data/{k} length does not match episode_ends"

if "object_id" in root["meta"]:
    names = root["meta"].attrs.get("object_names", [])
    ids, counts = np.unique(root["meta/object_id"][:], return_counts=True)
    print(" episodes per object", {names[i] if i < len(names) else int(i): int(c) for i, c in zip(ids, counts)})

# Scene conditions per episode (meta/ep_*, absent in zarrs from before the scene configs)
meta = root["meta"]
names = {"background": "background_names", "lighting": "lighting_names", "liquid_kind": "liquid_kinds",
         "thermal_config": "thermal_config_names"}
for k in ["level", "split", "thermal_config", "background", "lighting", "n_distractors", "lookalike", "n_absorbers",
          "n_table_blobs", "fill_level", "liquid_kind", "is_crossover", "realistic"]:
    if "ep_" + k in meta:
        ids, counts = np.unique(meta["ep_" + k][:], return_counts=True)
        label = meta.attrs.get(names.get(k, ""), [])
        print(f" ep_{k:15s}", {(label[int(i)] if 0 <= i < len(label) else round(float(i), 2)): int(c) for i, c in zip(ids, counts)})
for k in ["t_env", "thermal_gap", "alpha"]:
    if "ep_" + k in meta:
        v = meta["ep_" + k][:]
        print(f" ep_{k:15s} min {v.min():.2f} mean {v.mean():.2f} max {v.max():.2f}")

act = root["data/action"][:]
print(" action min", np.round(act.min(0), 2))
print(" action max", np.round(act.max(0), 2))

# Preview contains first episode at 5 moments, rows = rgb, d, thermal
idx = np.linspace(0, ends[0] - 1, 5).astype(int)
rows = []
for k in ["rgb", "depth", "thermal"]:  # thermal_raw is shown through data/thermal
    tiles = [root["data"][k][i] for i in idx]
    tiles = [np.repeat(t, 3, -1) if t.shape[-1] == 1 else t for t in tiles]
    rows.append(np.concatenate([np.pad(t, ((2, 2), (2, 2), (0, 0)), constant_values=255) for t in tiles], 1))

out = Path("sim/out/dataset_preview.png")
out.parent.mkdir(exist_ok=True)
Image.fromarray(np.concatenate(rows, 0)).resize((1000, 600), Image.NEAREST).save(out)
print("Dataset preview saved to ", out)