# Real transparent-object perception data

Five public datasets converted into one on-disk format with one manifest and one PyTorch loader.
Use them to pretrain / evaluate the RGB, depth and thermal encoders before plugging them into the policy.
The format and the loader's `encoding="sim"` option follow the conventions of `sim/` so real and simulated
inputs can go through the same encoder.

## What is in it

| dataset | train | val | test | modalities | resolution | license |
|---|---:|---:|---:|---|---|---|
| `rgbt_glass` ([Huo et al., TIP'23](https://github.com/Dong-Huo/RGB-T-Glass-Segmentation)) | 3974 | 453 | 1124 | rgb, thermal (°C), glass mask | 640x480 | code MIT, data not stated |
| `mju_glass` ([MJU-Glass, CVIDL'25](https://github.com/weduake/MJU_Glass)) | 2932 | 494 | 1474 | rgb, depth (mm), glass mask | 640x480 | not stated |
| `rgbd_gsd` ([RGBD-GSD, AAAI'25](https://huggingface.co/datasets/garrying/RGBD-GSD)) | 2099 | 301 | 609 | rgb, relative depth, glass mask | 561x427 to 1280x1024 | CC BY-NC 4.0 |
| `transpose` ([TRansPose, IJRR'24](https://sites.google.com/view/transpose-dataset)) | 2184 | 252 | 6552 | rgb, depth (mm), GT depth, thermal (raw counts), object mask, 6D poses, intrinsics | 640x480 | CC BY-NC-SA 4.0 |
| `trobi` ([T-ROBI](https://www.trailab.utias.utoronto.ca/t-robi-dataset)) | 0 | 0 | 660 | grey, depth (mm), GT depth, object mask, 6D poses, intrinsics | 1280x1024 | CC BY-NC-SA 4.0 |
| **total** | **11189** | **1500** | **10419** | | | |

How the depth sensor does on transparent pixels (why we need thermal / learned completion). "Bad" = no reading
or more than 2 cm away from the ground-truth depth; it needs `depth_gt`, so MJU-Glass has no value.

| dataset | sensor | reading present on glass | bad on glass | median error where present |
|---|---|---:|---:|---:|
| `mju_glass` | Orbbec Gemini Pro | 11% | | |
| `trobi` | Ensenso stereo | 8% | 95% | 2 mm |
| `transpose` | RealSense L515 | 86% | 95% | 132 mm |

The L515 rarely leaves a hole on glass: it reports the table *behind* the glass. A model cannot just look for holes.

All licenses are non-commercial or unstated. Keep the converted data off public hosting.

## Quick start

```bash
conda create -n transparent python=3.11 -y && conda activate transparent
pip install -r real_data/requirements.txt

bash real_data/download.sh                    # ~9.5 GB into ~/datasets/transparent/raw, checksummed
# TRansPose is behind a request form: put its seq_*.tar.gz + models_ply.tar.gz in ~/datasets/transparent/raw/transpose
# (or set TRANSPOSE_SRC). See "TRansPose" below.
python real_data/convert.py --datasets all    # ~18 GB into ~/datasets/transparent/unified (~20 min with 8 workers)
python real_data/inspect_unified.py --full    # validates every file, writes real_data/out/preview_<dataset>.png
```

Set `TRANSPARENT_RAW_ROOT` / `TRANSPARENT_DATA_ROOT` to put the raw downloads / converted data elsewhere.

```python
from torch.utils.data import DataLoader
from real_data.dataset import TransparentPerceptionDataset, balanced_sampler

# every RGB-thermal training pair, 224x224, sim-style encodings
ds = TransparentPerceptionDataset(splits=["train"], require=["thermal"], size=224, encoding="sim")

# all training data with a glass mask, each source dataset drawn equally often
ds = TransparentPerceptionDataset(splits=["train"], require=["mask"], size=224)
dl = DataLoader(ds, batch_size=32, sampler=balanced_sampler(ds), num_workers=8)
```

Each item has the same keys whatever the source, so mixed batches use the default `collate_fn`.
Missing modalities come back as zeros with `has_<modality> = False`; use those flags to mask losses.

| key | shape · dtype | meaning |
|---|---|---|
| `rgb` | (3,H,W) float32 | [0,1]. T-ROBI is greyscale repeated to 3 channels |
| `depth` | (1,H,W) float32 | `encoding="metric"`: metres. `encoding="sim"`: `clip(m,0,2)/2`, the float version of `sim/`'s uint8 depth. 0 = no reading in both |
| `depth_gt` | (1,H,W) float32 | completed / ground-truth depth, same encoding (T-ROBI, TRansPose) |
| `thermal` | (1,H,W) float32 | [0,1], min-max normalised per frame like `sim/` |
| `mask` | (H,W) int64 | 0 background, 1 transparent, 255 ignore |
| `has_depth`, `has_depth_gt`, `has_thermal`, `has_mask` | bool | modality present |
| `depth_is_metric` | bool | False for RGBD-GSD, whose depth is relative |
| `sample_id`, `dataset` | str | |

`size=` takes an int or `(H, W)`. `resize="crop"` (default) scales the short side and centre-crops.
`resize="stretch"` resizes straight to the target like `sim/env.py`. Depth, thermal and mask are resized
with nearest-neighbour so invalid zeros never blend into valid values.

## On-disk format

```
~/datasets/transparent/unified/
  manifest.parquet                 # all datasets, one row per sample
  <dataset>/manifest.parquet       # same rows for one dataset
  <dataset>/meta.json              # source, license, citation, split counts, caveats
  <dataset>/{rgb,depth,depth_gt,thermal,mask}/<local_id>.png|jpg
  transpose/pose/<local_id>.json   # per-frame object poses in the camera frame (mm)
```

| modality | encoding |
|---|---|
| rgb | 8-bit, original resolution, original codec (RGBD-GSD stays JPEG) |
| depth, depth_gt | 16-bit PNG, millimetres, 0 = invalid. RGBD-GSD: relative, larger = farther (`depth_kind = relative`) |
| thermal | 16-bit PNG, 0 = invalid. RGB-T: Kelvin × 100 (`thermal_kind = kelvin_x100`). TRansPose: 14-bit sensor counts, not temperature (`raw_counts`) |
| mask | 8-bit PNG, 0 / 1 / 255 = background / transparent / ignore |

Manifest columns: `sample_id, dataset, split, group, *_path, height, width, depth_kind, thermal_kind, aligned,
glass_frac, fx, fy, cx, cy, extra` (JSON with per-dataset fields). `group` is the capture unit (scene, session,
viewpoint) and never straddles train / val / test, with one exception noted below. See `common.py` for the schema.

## Splits and caveats

- **`rgbd_gsd`: the official test split leaks.** 211 of its 609 test images come from a Matterport viewpoint
  or a 2D-3D-S room that also appears in train. Test numbers on it are optimistic. Those rows have
  `extra.test_group_also_in_train = true`, so filter them out for a clean evaluation. `val` is carved only from train
  groups that never appear in test.
- **`rgbd_gsd`: depth is relative.** It is min-max normalised per image (max always 65535), larger = farther, with no
  metric scale and no intrinsics. `depth_is_metric` is False for it. 67 numeric image ids are reused for different
  images in train and test, so local ids carry a `train_` / `test_` prefix.
- **`mju_glass`: our own split.** The authors publish no split file. Ours cuts the capture timeline at gaps over 60 s
  (78 segments), keeps cam1 and cam2 of a timestamp together, and assigns whole segments 60 / 10 / 30 with seed 0.
  Scores are not comparable to the paper's. A random image-level split would put near-duplicate frames on both sides.
- **`rgbt_glass`**: the official split is by location, so it is clean. `val` is 10% of the training locations.
  Thermal is registered to RGB by the authors. `*_noglass` locations (7% of images) are negatives with empty masks.
- **`trobi` is evaluation-only** (12 bin-picking scenes x 55 views, all `test`). The image is the rectified left Ensenso
  camera in greyscale; there is no colour camera. `depth` is the raw Ensenso depth and `depth_gt` the scan with the
  objects sprayed matt. The mask is rendered from the CAD models at the GT poses (median 0.23 mm against `depth_gt`),
  and per-instance object-to-camera poses in mm are in `extra.obj_poses_cam_mm`. The 36 GB synthetic set is skipped.
- **`transpose`**: household sequences = train/val (val is `seq_H_05`), official test sequences = test. Test also
  contains chemistry glassware and recyclable trash, which train never shows, so test measures transfer to unseen
  object types too. Every 5th frame is kept. Details of what was repaired are in the next section.
- Only `trobi` and `transpose` ship intrinsics; `fx, fy, cx, cy` are null for the others.

## TRansPose

**Getting it.** Fill in the [request form](https://docs.google.com/forms/d/11NwmPGtbV9c6P61WrHAz_1dOfXXrADUnhTLXORThBX0/viewform)
on the [dataset site](https://sites.google.com/view/transpose-dataset), once per item (CAD models, household, test, ...).
We use: household sequences (9, ~25 GB), test sequences (26, ~69 GB), `models_ply.tar.gz`, sample code.
Put the archives in `~/datasets/transparent/raw/transpose/` or point `TRANSPOSE_SRC` at them; the converter
unpacks one sequence at a time (only the files and frames it needs) and deletes it afterwards.

**What the converter does.** One sample = one frame of `cam_R`, the L515 that has depth.

| output | source | notes |
|---|---|---|
| rgb | `cam_R/rgb` | copied |
| depth | `cam_R/depth/raw` | copied, mm |
| depth_gt | `cam_R/depth/rendered` on objects, raw elsewhere | the authors' `depth/fixed` is not used, see below |
| thermal | `cam_T/14bit`, re-projected into `cam_R` through depth_gt | z-buffered; 0 where the thermal camera cannot see the point |
| mask | `cam_R/depth/rendered > 0` | opaque `obj_N_*` objects set to 0 via a CAD ray-cast; unattributable edge pixels next to them 255 |
| pose/<id>.json | `cam_R/pose` | copied |

**Things in the raw release that are wrong or surprising** (all verified on the files):
- `mask/*.png` are image cut-outs (object pixels copied from the RGB / thermal image, rest black), not binary masks.
  Dark object pixels drop out (up to ~6.7k px in a frame). `depth/fixed` was built from these, so it keeps the wrong
  raw depth on those pixels. Hence masks and depth_gt are rebuilt from `depth/rendered`.
- `cam_T/camera_info.json` says 640x480, but the thermal images and intrinsics are 640x512.
- `extrinsic.json` `T_EE2cam_X` is the camera pose in the end-effector frame, in metres (poses are in mm).
  Checked per sequence against the GT poses: max 0.00 mm / 0.08 deg off.
- Thermal values are uncalibrated 14-bit counts, so their scale differs from RGB-T's temperatures. The loader's
  per-frame normalisation makes them comparable as network input.
- `obj_N_*` objects are opaque (drill, detergent bottle, ...; 5 test sequences). The `_A/_B/_C` suffix on some of
  them is a variant, not a material code; materials are only `G` (glass) and `P` (other).

## Files

| file | purpose |
|---|---|
| `download.sh` | fetch + checksum the raw data |
| `convert.py` | one converter per dataset, then `build_manifest` |
| `common.py` | schema, encodings, PNG writers, group-level split helper |
| `dataset.py` | `TransparentPerceptionDataset`, `balanced_sampler`, `load_manifest` |
| `inspect_unified.py` | validation, statistics, preview grids |
| `show_sample.py` | labelled, colour-mapped view of any sample (`--random N --dataset X` for random ones) |
