"""
Convert the public transparent-object datasets into the unified format (see common.py).

run via
    python real_data/convert.py --datasets all
    python real_data/convert.py --datasets rgbt_glass --workers 8
Raw downloads are expected under $TRANSPARENT_RAW_ROOT (default ~/datasets/transparent/raw),
laid out exactly as real_data/download.sh leaves them.
"""
import argparse
import glob
import io
import json
import os
import re
import tarfile
import zipfile
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from PIL import Image

from common import (DEFAULT_RAW, DEFAULT_ROOT, MASK_BG, MASK_IGNORE, binarize_mask, build_manifest, celsius_to_kelvin_x100,
                    group_split, make_row, save_dataset, write_bytes, write_png)

SEED = 0


def _rgb_ext(data: bytes):
    return ".jpg" if data[:2] == b"\xff\xd8" else ".png"


def _glass_frac(mask01):
    return float((mask01 == 1).mean())


# ---------------------------------------------------------------- RGBD-GSD (AAAI'25)
def _gsd_group(image_id):
    """Matterport-style '<32hex>_<i>_<j>' share a viewpoint; 2D-3D-S style
    'camera_<hash>_<room>_frame_<n>_domain' share a room. Numeric ids: unknown origin."""
    if re.fullmatch(r"[0-9a-f]{32}_\d+_\d+", image_id):
        return "mp:" + image_id.split("_")[0], "matterport_like"
    m = re.match(r"camera_[0-9a-f]+_(.+?)_frame_\d+_domain$", image_id)
    if m:
        return "s3d:" + m.group(1), "2d3ds_like"
    return "id:" + image_id, "unknown"


def _gsd_one(item, out):
    official, iid, rgb_b, mask_b, depth_b = item
    lid = f"{official}_{iid}"  # 67 numeric ids are reused for different images in train and test
    mask = binarize_mask(np.array(Image.open(io.BytesIO(mask_b))))
    depth = np.array(Image.open(io.BytesIO(depth_b)))
    if depth.ndim == 3:  # one sample (train 00002044) ships as grey replicated to RGB
        assert (depth[..., 0] == depth[..., 1]).all() and (depth[..., 1] == depth[..., 2]).all(), lid
        depth = depth[..., 0]
    if depth.dtype == np.uint8:  # same 0..max normalisation, 8-bit -> 16-bit range
        depth = depth.astype(np.uint16) * 257
    assert depth.dtype == np.uint16, (lid, depth.dtype)
    w, h = Image.open(io.BytesIO(rgb_b)).size
    assert mask.shape == depth.shape == (h, w), (lid, mask.shape, depth.shape, (h, w))
    ext = _rgb_ext(rgb_b)
    write_bytes(f"{out}/rgbd_gsd/rgb/{lid}{ext}", rgb_b)
    write_png(f"{out}/rgbd_gsd/depth/{lid}.png", depth)
    write_png(f"{out}/rgbd_gsd/mask/{lid}.png", mask)
    group, source = _gsd_group(iid)
    if source == "unknown":
        group = f"id:{lid}"
    return dict(lid=lid, iid=iid, ext=ext, official=official, group=group, source=source, h=h, w=w, gf=_glass_frac(mask))


def _gsd_items(files):
    for f in files:
        official = "test" if "/test-" in f else "train"
        t = pq.read_table(f)
        cols = [t.column(k).to_pylist() for k in ("image_id", "image", "mask", "depth")]
        for iid, im, mk, dp in zip(*cols):
            yield official, iid, im["bytes"], mk["bytes"], dp["bytes"]


def convert_rgbd_gsd(raw, out, workers):
    name = "rgbd_gsd"
    files = sorted(glob.glob(f"{raw}/rgbd_gsd/data/*.parquet"))
    assert files, f"no parquet files under {raw}/rgbd_gsd/data"
    with Pool(workers) as pool:
        recs = list(pool.imap(partial(_gsd_one, out=out), _gsd_items(files), chunksize=8))

    test_groups = {r["group"] for r in recs if r["official"] == "test"}
    train_groups = {r["group"] for r in recs if r["official"] == "train"}
    # val is carved only from train groups that never appear in the official test split
    clean_train = [r["group"] for r in recs if r["official"] == "train" and r["group"] not in test_groups]
    val_assign = group_split(clean_train, test_frac=0.0, val_frac=0.1 * len(recs) / max(1, len(clean_train)), seed=SEED)
    rows = []
    for r in recs:
        split = "test" if r["official"] == "test" else val_assign.get(r["group"], "train")
        rows.append(make_row(
            sample_id=f"{name}/{r['lid']}", dataset=name, split=split, group=r["group"],
            rgb_path=f"{name}/rgb/{r['lid']}{r['ext']}", depth_path=f"{name}/depth/{r['lid']}.png",
            mask_path=f"{name}/mask/{r['lid']}.png", height=r["h"], width=r["w"],
            depth_kind="relative", aligned=True, glass_frac=r["gf"],
            extra={"image_id": r["iid"], "source_style": r["source"], "official_split": r["official"],
                   "test_group_also_in_train": r["official"] == "test" and r["group"] in train_groups}))
    n_leak = sum(json.loads(x["extra"])["test_group_also_in_train"] for x in rows)
    save_dataset(out, name, rows, meta={
        "name": "RGBD-GSD", "paper": "Lin et al., Leveraging RGB-D Data with Cross-Modal Context Mining for Glass Surface Detection, AAAI 2025",
        "url": "https://huggingface.co/datasets/garrying/RGBD-GSD", "license": "CC BY-NC 4.0",
        "notes": [
            "Depth is per-image min-max normalised to uint16 (max is always 65535); larger = farther; NOT metric. 0 = missing (glass often reads 0), "
            "but the nearest valid pixel of an image may also have been normalised to 0.",
            "Mixed source resolutions (561x427 up to 1280x1024); no intrinsics published.",
            f"Official test split shares capture groups with train: {n_leak} of the test images come from a viewpoint/room that also appears in train. "
            "Test numbers on this split are optimistic; extra.test_group_also_in_train flags them.",
            "val = ~10% of samples, carved from train groups that never appear in test.",
            "Numeric image ids (00000001..) have unknown capture structure; each is its own group. 67 of them are reused "
            "for different images in train and test, so local ids are prefixed with the official split (train_/test_).",
        ]})


# ---------------------------------------------------------------- MJU-Glass (CVIDL'25)
def _mju_one(stem, src, out):
    ts, cam = re.match(r"(.+)_rgb_(cam\d)$", stem).groups()
    lid = f"{ts}_{cam}"
    rgb_b = Path(f"{src}/images/{stem}.png").read_bytes()
    depth_b = Path(f"{src}/depth/{ts}_grey_depth_{cam}.png").read_bytes()
    mask = binarize_mask(np.array(Image.open(f"{src}/mask/{stem}_mask.png")))
    depth = Image.open(io.BytesIO(depth_b))
    w, h = Image.open(io.BytesIO(rgb_b)).size
    assert depth.mode == "I;16" and depth.size == (w, h) and mask.shape == (h, w), stem
    write_bytes(f"{out}/mju_glass/rgb/{lid}.png", rgb_b)
    write_bytes(f"{out}/mju_glass/depth/{lid}.png", depth_b)  # already uint16 millimetres
    write_png(f"{out}/mju_glass/mask/{lid}.png", mask)
    return dict(lid=lid, ts=float(ts), cam=cam, h=h, w=w, gf=_glass_frac(mask))


def convert_mju_glass(raw, out, workers, gap_s=60.0):
    name, src = "mju_glass", f"{raw}/mju_glass/extracted"
    if not Path(src).exists():
        for z in ("images", "depth", "mask"):
            with zipfile.ZipFile(f"{raw}/mju_glass/{z}.zip") as zf:
                zf.extractall(src)
    stems = sorted(p.stem for p in Path(f"{src}/images").glob("*_rgb_cam*.png"))
    with Pool(workers) as pool:
        recs = pool.map(partial(_mju_one, src=src, out=out), stems, chunksize=32)
    # capture-session groups: cut the timeline wherever consecutive timestamps are > gap_s apart
    ts = np.array(sorted({r["ts"] for r in recs}))
    cut = np.concatenate([[0], np.cumsum(np.diff(ts) > gap_s)])
    seg_of = dict(zip(ts, cut))
    for r in recs:
        r["group"] = f"seg{seg_of[r['ts']]:03d}"
    assign = group_split([r["group"] for r in recs], test_frac=0.3, val_frac=0.1, seed=SEED)
    rows = [make_row(
        sample_id=f"{name}/{r['lid']}", dataset=name, split=assign[r["group"]], group=r["group"],
        rgb_path=f"{name}/rgb/{r['lid']}.png", depth_path=f"{name}/depth/{r['lid']}.png",
        mask_path=f"{name}/mask/{r['lid']}.png", height=r["h"], width=r["w"],
        depth_kind="metric_mm", aligned=True, glass_frac=r["gf"],
        extra={"camera": r["cam"], "timestamp": r["ts"]}) for r in recs]
    save_dataset(out, name, rows, meta={
        "name": "MJU-Glass", "paper": "Glass Surface Segmentation with an RGB-D Camera via Weighted Feature Fusion for Service Robots, CVIDL 2025 (arXiv 2508.01639)",
        "url": "https://github.com/weduake/MJU_Glass", "license": "not stated by the authors (ask before redistributing)",
        "notes": [
            "Two Orbbec Gemini Pro cameras (cam1, cam2) fire at the same timestamp; both views of a timestamp stay in the same split.",
            "Depth is uint16 millimetres (median ~1.4 m), already registered and resized to 640x480 by the authors; 0 = no reading.",
            "No official split file is published. Split here = 70/10/20-ish by sample count over capture segments "
            f"(timeline cut at gaps > {gap_s:.0f} s), seed {SEED}. Not comparable to the paper's 3430/1470 numbers.",
            "No intrinsics published.",
        ]})


# ---------------------------------------------------------------- RGB-T glass (TIP'23)
def _rgbt_one(d, zpath, out):
    split, scene, sid = d.split("/")
    lid = f"{scene}_{sid}"
    with zipfile.ZipFile(zpath) as z:
        rgb_b = z.read(f"{d}/rgb.png")
        temp = np.load(io.BytesIO(z.read(f"{d}/temperature.npy")))
        mask = binarize_mask(np.array(Image.open(io.BytesIO(z.read(f"{d}/mask.png")))))
    w, h = Image.open(io.BytesIO(rgb_b)).size
    assert temp.shape == mask.shape == (h, w), (d, temp.shape, mask.shape)
    write_bytes(f"{out}/rgbt_glass/rgb/{lid}.png", rgb_b)
    write_png(f"{out}/rgbt_glass/thermal/{lid}.png", celsius_to_kelvin_x100(temp))
    write_png(f"{out}/rgbt_glass/mask/{lid}.png", mask)
    return dict(lid=lid, official=split, scene=scene, h=h, w=w, gf=_glass_frac(mask),
                tmin=float(np.nanmin(temp)), tmax=float(np.nanmax(temp)))


def convert_rgbt_glass(raw, out, workers):
    name, zpath = "rgbt_glass", f"{raw}/rgbt_glass/RGB-T-glass.zip"
    with zipfile.ZipFile(zpath) as z:
        dirs = sorted({n.rsplit("/", 1)[0] for n in z.namelist() if n.endswith("/rgb.png")})
    with Pool(workers) as pool:
        recs = pool.map(partial(_rgbt_one, zpath=zpath, out=out), dirs, chunksize=16)
    train_scenes = [r["scene"] for r in recs if r["official"] == "train"]
    val_assign = group_split(train_scenes, test_frac=0.0, val_frac=0.1, seed=SEED)
    rows = [make_row(
        sample_id=f"{name}/{r['lid']}", dataset=name,
        split="test" if r["official"] == "test" else val_assign[r["scene"]], group=r["scene"],
        rgb_path=f"{name}/rgb/{r['lid']}.png", thermal_path=f"{name}/thermal/{r['lid']}.png",
        mask_path=f"{name}/mask/{r['lid']}.png", height=r["h"], width=r["w"],
        thermal_kind="kelvin_x100", aligned=True, glass_frac=r["gf"],
        extra={"scene": r["scene"], "official_split": r["official"], "no_glass_scene": r["scene"].endswith("_noglass"),
               "temp_c_min": round(r["tmin"], 2), "temp_c_max": round(r["tmax"], 2)}) for r in recs]
    save_dataset(out, name, rows, meta={
        "name": "RGB-T glass segmentation", "paper": "Huo et al., Glass Segmentation with RGB-Thermal Image Pairs, IEEE TIP 2023",
        "url": "https://github.com/Dong-Huo/RGB-T-Glass-Segmentation (data: Google Drive id 1g42XPaudslnIo1oR59FKQV9rVFOTqu_w)",
        "license": "code MIT; dataset license not stated (ask before redistributing)",
        "notes": [
            "Thermal = calibrated temperature (deg C in the source .npy), stored as Kelvin*100 uint16; registered to the RGB frame by the authors.",
            "Official split is by location (train 28 scenes / test 12), so it is leakage-free. val = ~10% of train scenes.",
            "*_noglass scenes are negatives (mask all zero).",
            "The source boundary.png edge maps are not copied; they are derivable from the mask.",
        ]})



# ---------------------------------------------------------------- T-ROBI (TRAIL, U. Toronto)
# Ensenso intrinsics from github.com/junyang224/ROBI_Toolkit cam/Ensenso_{DEPTH,LEFT}.yml. They are the
# ROBI calibration; verified for T-ROBI by rendering the CAD models at the GT poses (median |render - GT depth| 0.23 mm).
TROBI_K = dict(fx=1083.097046, fy=1083.097046, cx=379.326874, cy=509.437195, width=1280, height=1024)
TROBI_DEPTH_UNIT_MM = 0.03125
TROBI_MESH = {"Bottle": "Bottle.stl", "Pipe_Fitting": "Pipe_fitting.stl"}


def _trobi_depth_mm(path):
    d = np.array(Image.open(path)).astype(np.float64) * TROBI_DEPTH_UNIT_MM
    return np.clip(np.round(d), 0, 65535).astype(np.uint16)


def _trobi_scene(scene_dir, src, out):
    import trimesh  # only needed for T-ROBI mask rendering
    scene_dir = Path(scene_dir)
    obj, scene = scene_dir.parent.name, scene_dir.name
    mesh = trimesh.load(f"{src}/_T_ROBI_Object_models/{TROBI_MESH[obj]}")
    world_obj = {k: np.array(v).reshape(4, 4) for k, v in json.loads((scene_dir / "GT_world2obj.json").read_text()).items()}
    K, H, W = TROBI_K, TROBI_K["height"], TROBI_K["width"]
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    rays = np.stack([(u - K["cx"]) / K["fx"], (v - K["cy"]) / K["fy"], np.ones((H, W))], -1).reshape(-1, 3)
    recs = []
    for jf in sorted((scene_dir / "Depth").glob("DEPTH_view_*.json"), key=lambda p: int(p.stem.split("_")[-1])):
        view = int(jf.stem.split("_")[-1])
        cam_world = np.linalg.inv(np.array(json.loads(jf.read_text())["cam_pose"]).reshape(4, 4))  # cam_pose is cam->world (mm)
        cam_obj = {k: cam_world @ T for k, T in world_obj.items()}
        meshes = []
        for T in cam_obj.values():
            m = mesh.copy(); m.apply_transform(T); meshes.append(m)
        hit = trimesh.ray.ray_pyembree.RayMeshIntersector(trimesh.util.concatenate(meshes))
        _, ri, _ = hit.intersects_location(np.zeros_like(rays), rays, multiple_hits=False)
        mask = np.zeros(H * W, np.uint8); mask[ri] = 1; mask = mask.reshape(H, W)
        lid = f"{obj}_{scene}_view{view:02d}"
        left = np.array(Image.open(scene_dir / "Stereo" / f"LEFT_view_{view}.bmp").convert("L"))
        assert left.shape == (H, W), (lid, left.shape)
        write_png(f"{out}/trobi/rgb/{lid}.png", left)  # single-channel grey; loader repeats to 3 channels
        write_png(f"{out}/trobi/depth/{lid}.png", _trobi_depth_mm(scene_dir / "Depth" / f"DEPTH_view_{view}.png"))
        write_png(f"{out}/trobi/depth_gt/{lid}.png", _trobi_depth_mm(scene_dir / "GT_Depth" / f"DEPTH_view_{view}.png"))
        write_png(f"{out}/trobi/mask/{lid}.png", mask)
        recs.append(dict(lid=lid, obj=obj, scene=scene, view=view, gf=_glass_frac(mask),
                         cam_obj={k: np.round(T, 5).tolist() for k, T in cam_obj.items()}))
    return recs


def convert_trobi(raw, out, workers):
    name, src = "trobi", f"{raw}/trobi/extracted"
    if not Path(src).exists():
        for z in ("TROBI.zip", "T_ROBI_Object_models.zip"):
            with zipfile.ZipFile(f"{raw}/trobi/{z}") as zf:
                zf.extractall(src)
    scenes = sorted(glob.glob(f"{src}/TROBI/*/Scene*"))
    with Pool(min(workers, len(scenes))) as pool:
        recs = [r for part in pool.map(partial(_trobi_scene, src=src, out=out), scenes) for r in part]
    K = TROBI_K
    rows = [make_row(
        sample_id=f"{name}/{r['lid']}", dataset=name, split="test", group=f"{r['obj']}/{r['scene']}",
        rgb_path=f"{name}/rgb/{r['lid']}.png", depth_path=f"{name}/depth/{r['lid']}.png",
        depth_gt_path=f"{name}/depth_gt/{r['lid']}.png", mask_path=f"{name}/mask/{r['lid']}.png",
        height=K["height"], width=K["width"], depth_kind="metric_mm", aligned=True, glass_frac=r["gf"],
        fx=K["fx"], fy=K["fy"], cx=K["cx"], cy=K["cy"],
        extra={"object": r["obj"], "scene": r["scene"], "view": r["view"], "rgb_is_gray": True,
               "obj_poses_cam_mm": r["cam_obj"]}) for r in recs]
    save_dataset(out, name, rows, meta={
        "name": "T-ROBI", "url": "https://www.trailab.utias.utoronto.ca/t-robi-dataset",
        "citation": "Yang, Xue, Ghavidel, Waslander. Active 6D Pose Estimation for Textureless Objects using Multi-View RGB Frames. IJRR (submitted), 2025",
        "license": "CC BY-NC-SA 4.0",
        "notes": [
            "2 objects (Bottle: 6 scenes, Pipe_Fitting: 6 scenes) x 55 Ensenso viewpoints. Every sample is split=test (evaluation set).",
            "rgb is the rectified LEFT Ensenso image: single-channel grey, 1280x1024. There is no colour camera in this release.",
            "depth = raw Ensenso stereo depth (fails on the transparent objects); depth_gt = Ensenso depth with the objects "
            "sprayed with anti-reflective coating. Both converted from 1/32 mm to uint16 mm.",
            "mask is rendered by ray-casting the CAD models at the GT 6D poses (all instances -> 1). "
            "Per-instance object-to-camera poses (mm) are in extra.obj_poses_cam_mm.",
            "Intrinsics come from the ROBI toolkit Ensenso calibration; verified against GT depth (median error 0.23 mm).",
            "The 36 GB synthetic set (TROBI_synthetic.zip) and the RIGHT stereo images are not converted.",
        ]})


# ---------------------------------------------------------------- TRansPose (IJRR'24, SNU RPM lab)
# One sample = one frame seen by cam_R (the RealSense L515 that has depth). Verified on the real files:
#  - depth PNGs are uint16 mm; pose json = object pose in that camera ("rot" row-major 3x3, "tra" mm)
#  - cam_T intrinsics are for its 640x512 images (camera_info "size" says 640x480, which is wrong)
#  - extrinsic.json T_EE2cam_X = camera pose in the end-effector frame, metres; cam_R<-cam_T is rigid
#  - mask/*.png are image cut-outs, not binary masks: dark object pixels are lost, and depth/fixed was
#    built from them. So masks come from depth/rendered and depth_gt = rendered on objects, raw elsewhere.
#  - L515 raw depth on glass is mostly NOT missing but too far (it sees the table through the glass).
#  - obj_N_* objects are opaque; they are 0 in the transparent mask.
TP_STEP = 5            # keep frames 1, 6, 11, ...: consecutive frames are nearly identical
TP_FRAME_GLOB = "[16]"  # frame ids ending in 1 or 6  <=>  id % 5 == 1
_TP_MESH = {}


def _tp_pose(o):
    T = np.eye(4)
    T[:3, :3] = np.array(o["rot"]).reshape(3, 3)
    T[:3, 3] = o["tra"]
    return T


def _tp_opaque_owner(pose, K, shape, models):
    """Ray-cast every object; return (owned_by_opaque, owned_by_any) boolean maps."""
    import trimesh
    H, W = shape
    meshes, opaque = [], []
    for o in pose.values():
        if o["obj_id"] not in _TP_MESH:
            _TP_MESH[o["obj_id"]] = trimesh.load(f"{models}/obj_{o['obj_id']:06d}.ply", process=False)
        m = _TP_MESH[o["obj_id"]].copy(); m.apply_transform(_tp_pose(o)); meshes.append(m)
        opaque += [o["obj_name"].split("_")[1] == "N"] * len(m.faces)
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    rays = np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones((H, W))], -1).reshape(-1, 3)
    _, ri, ti = trimesh.ray.ray_pyembree.RayMeshIntersector(trimesh.util.concatenate(meshes)).intersects_location(
        np.zeros_like(rays), rays, multiple_hits=False)
    own, opq = np.zeros(H * W, bool), np.zeros(H * W, bool)
    own[ri] = True; opq[ri] = np.array(opaque)[ti]
    return opq.reshape(H, W), own.reshape(H, W)


def _tp_reproject(depth_mm, K_R, K_T, T_T_R, thermal):
    """Carry thermal into the cam_R image through depth, with a z-buffer so pixels the thermal camera
    cannot see (parallax shadows, outside its field of view) stay 0."""
    H, W = depth_mm.shape
    Ht, Wt = thermal.shape
    v, u = np.nonzero(depth_mm > 0)
    z = depth_mm[v, u].astype(np.float64)
    X = np.stack([(u - K_R[0, 2]) / K_R[0, 0] * z, (v - K_R[1, 2]) / K_R[1, 1] * z, z])
    Xt = T_T_R[:3, :3] @ X + T_T_R[:3, 3:4]
    zt = np.maximum(Xt[2], 1e-6)
    ut = np.round(K_T[0, 0] * Xt[0] / zt + K_T[0, 2]).astype(np.int64)
    vt = np.round(K_T[1, 1] * Xt[1] / zt + K_T[1, 2]).astype(np.int64)
    ok = np.flatnonzero((Xt[2] > 0) & (ut >= 0) & (ut < Wt) & (vt >= 0) & (vt < Ht))
    idx = vt[ok] * Wt + ut[ok]
    zbuf = np.full(Ht * Wt, np.inf)
    np.minimum.at(zbuf, idx, Xt[2][ok])
    vis = ok[Xt[2][ok] <= zbuf[idx] + 10.0]
    out = np.zeros((H, W), np.uint16)
    out[v[vis], u[vis]] = thermal[vt[vis], ut[vis]]
    return out


def _tp_sequence(arch, out, models, work):
    import shutil, subprocess
    import cv2
    name = "transpose"
    tmp = Path(work) / Path(arch).name.replace(".tar.gz", "")
    shutil.rmtree(tmp, ignore_errors=True); tmp.mkdir(parents=True)
    g = TP_FRAME_GLOB
    subprocess.run(["tar", "xzf", arch, "-C", str(tmp), "--wildcards", "*/extrinsic.json", "*/cam_R/camera_info.json",
                    "*/cam_T/camera_info.json", "*/cam_T/pose/000001.json", f"*/cam_R/rgb/*{g}.png",
                    f"*/cam_R/depth/raw/*{g}.png", f"*/cam_R/depth/rendered/*{g}.png", f"*/cam_R/pose/*{g}.json",
                    f"*/cam_T/14bit/*{g}.png"], check=True)
    (sd,) = list((tmp / "sequences").iterdir())
    seq = sd.name
    K_R = np.array(json.loads((sd / "cam_R/camera_info.json").read_text())["intrinsic"]).reshape(3, 3)
    K_T = np.array(json.loads((sd / "cam_T/camera_info.json").read_text())["intrinsic"]).reshape(3, 3)
    E = {k: np.array(v).reshape(4, 4) for k, v in json.loads((sd / "extrinsic.json").read_text()).items()}
    T_R_T = np.linalg.inv(E["T_EE2cam_R"]) @ E["T_EE2cam_T"]
    T_R_T[:3, 3] *= 1000.0  # metres -> mm
    # the rig transform must agree with the per-camera GT poses of frame 1
    pR = json.loads((sd / "cam_R/pose/000001.json").read_text()); pT = json.loads((sd / "cam_T/pose/000001.json").read_text())
    k = next(iter(set(pR) & set(pT)))
    D = _tp_pose(pR[k]) @ np.linalg.inv(_tp_pose(pT[k]))
    rig_mm = float(np.abs(D[:3, 3] - T_R_T[:3, 3]).max())
    rig_deg = float(np.degrees(np.arccos(np.clip((np.trace(D[:3, :3] @ T_R_T[:3, :3].T) - 1) / 2, -1, 1))))
    assert rig_mm < 2.0 and rig_deg < 0.5, (seq, rig_mm, rig_deg)
    T_T_R = np.linalg.inv(T_R_T)
    near = np.ones((7, 7), np.uint8)
    recs = []
    for pj in sorted((sd / "cam_R/pose").glob("*.json")):
        f = pj.stem
        if int(f) % TP_STEP != 1:
            continue
        pose = json.loads(pj.read_text())
        lid = f"{seq}_{f}"
        rgb_b = (sd / "cam_R/rgb" / f"{f}.png").read_bytes()
        raw_b = (sd / "cam_R/depth/raw" / f"{f}.png").read_bytes()
        raw = np.array(Image.open(io.BytesIO(raw_b)))
        ren = np.array(Image.open(sd / "cam_R/depth/rendered" / f"{f}.png"))
        th = np.array(Image.open(sd / "cam_T/14bit" / f"{f}.png"))
        assert raw.dtype == ren.dtype == th.dtype == np.uint16 and raw.shape == ren.shape == (480, 640) and th.shape == (512, 640), lid
        obj = ren > 0
        gt = np.where(obj, ren, raw).astype(np.uint16)
        mask = obj.astype(np.uint8)
        n_opaque = sum(o["obj_name"].split("_")[1] == "N" for o in pose.values())
        if n_opaque:
            opq, own = _tp_opaque_owner(pose, K_R, ren.shape, models)
            mask[obj & opq] = MASK_BG
            near_opq = cv2.dilate(opq.astype(np.uint8), near) > 0
            mask[obj & ~own & near_opq] = MASK_IGNORE  # silhouette pixels we cannot attribute safely
        thermal = _tp_reproject(gt, K_R, K_T, T_T_R, th)
        write_bytes(f"{out}/{name}/rgb/{lid}.png", rgb_b)
        write_bytes(f"{out}/{name}/depth/{lid}.png", raw_b)
        write_png(f"{out}/{name}/depth_gt/{lid}.png", gt)
        write_png(f"{out}/{name}/thermal/{lid}.png", thermal)
        write_png(f"{out}/{name}/mask/{lid}.png", mask)
        write_bytes(f"{out}/{name}/pose/{lid}.json", pj.read_bytes())
        on = mask == 1
        recs.append(dict(lid=lid, seq=seq, frame=f, gf=_glass_frac(mask), K=K_R,
                         objects=sorted({o["obj_name"] for o in pose.values()}), n_opaque=n_opaque,
                         thermal_cov=float((thermal > 0).mean()),
                         raw_err_on_glass=float(np.median(np.abs(raw[on & (raw > 0)].astype(float) - gt[on & (raw > 0)]))) if (on & (raw > 0)).any() else None,
                         rig=(rig_mm, rig_deg)))
    shutil.rmtree(tmp)
    print(f"  {seq}: {len(recs)} frames", flush=True)
    return recs


def convert_transpose(raw, out, workers):
    name = "transpose"
    src = os.environ.get("TRANSPOSE_SRC", f"{raw}/transpose")
    archives = sorted(glob.glob(f"{src}/seq_*.tar.gz"), key=lambda a: -Path(a).stat().st_size)
    assert archives, f"no seq_*.tar.gz under {src} (set TRANSPOSE_SRC)"
    models = f"{raw}/transpose_models/models"
    if not Path(models).exists():
        with tarfile.open(f"{src}/models_ply.tar.gz") as tf:
            tf.extractall(f"{raw}/transpose_models", filter="data")
    with Pool(workers) as pool:
        recs = [r for part in pool.imap_unordered(partial(_tp_sequence, out=out, models=models, work=f"{raw}/transpose_work"), archives) for r in part]
    house = [r["seq"] for r in recs if not r["seq"].startswith("seq_test")]
    val_assign = group_split(house, test_frac=0.0, val_frac=0.1, seed=SEED)
    rows = [make_row(
        sample_id=f"{name}/{r['lid']}", dataset=name,
        split="test" if r["seq"].startswith("seq_test") else val_assign[r["seq"]], group=r["seq"],
        rgb_path=f"{name}/rgb/{r['lid']}.png", depth_path=f"{name}/depth/{r['lid']}.png",
        depth_gt_path=f"{name}/depth_gt/{r['lid']}.png", thermal_path=f"{name}/thermal/{r['lid']}.png",
        mask_path=f"{name}/mask/{r['lid']}.png", height=480, width=640, depth_kind="metric_mm",
        thermal_kind="raw_counts", aligned=True, glass_frac=r["gf"],
        fx=r["K"][0, 0], fy=r["K"][1, 1], cx=r["K"][0, 2], cy=r["K"][1, 2],
        extra={"sequence": r["seq"], "frame": r["frame"], "objects": r["objects"], "n_opaque": r["n_opaque"],
               "pose_path": f"{name}/pose/{r['lid']}.json", "thermal_coverage": round(r["thermal_cov"], 3),
               "raw_depth_err_on_glass_mm": r["raw_err_on_glass"]}) for r in recs]
    rig = np.array([r["rig"] for r in recs])
    save_dataset(out, name, rows, meta={
        "name": "TRansPose", "url": "https://sites.google.com/view/transpose-dataset",
        "citation": "Kim, Jeon, Jung, Yang, Jung, Shin, Kim. TRansPose: Large-scale multispectral dataset for transparent object. IJRR 43(6), 2024",
        "license": "CC BY-NC-SA 4.0",
        "notes": [
            "Converted: household sequences (official train) and the official test sequences. Not downloaded: chemical, trash and mixed train sequences.",
            f"Every {TP_STEP}th frame of cam_R (frames 1, 6, 11, ...); consecutive frames are nearly identical.",
            "rgb = cam_R colour 640x480; depth = raw L515 depth in mm (on glass it is mostly present but too far, not missing).",
            "depth_gt = authors' depth/rendered (CAD at GT pose) on object pixels, raw depth elsewhere. The authors' depth/fixed is NOT used: "
            "it was built from cut-out masks that drop dark object pixels.",
            "thermal = 14-bit FLIR A65 counts (not temperature) re-projected from cam_T into cam_R through depth_gt with a z-buffer; "
            "0 where the thermal camera cannot see the point (parallax shadow) or outside its field of view.",
            "mask = depth/rendered > 0, i.e. all visible objects; opaque obj_N_* objects are set to 0 using a ray-cast of the CAD models, "
            "and unattributable silhouette pixels next to them are 255.",
            f"Rig check per sequence (extrinsic.json vs GT poses): max translation error {rig[:, 0].max():.2f} mm, rotation {rig[:, 1].max():.3f} deg.",
            "Per-frame object poses (cam_R frame, mm) are copied to pose/<id>.json (path in extra.pose_path).",
            "val = one household sequence, chosen by group_split with seed 0.",
        ]})

CONVERTERS = {"rgbd_gsd": convert_rgbd_gsd, "mju_glass": convert_mju_glass, "rgbt_glass": convert_rgbt_glass,
              "trobi": convert_trobi, "transpose": convert_transpose}

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+", default=["all"], help=f"any of {list(CONVERTERS)} or 'all'")
    p.add_argument("--raw", default=DEFAULT_RAW)
    p.add_argument("--out", default=DEFAULT_ROOT)
    p.add_argument("--workers", type=int, default=8)
    a = p.parse_args()
    names = list(CONVERTERS) if a.datasets == ["all"] else a.datasets
    for n in names:
        print(f"== {n}")
        CONVERTERS[n](a.raw, a.out, a.workers)
    build_manifest(a.out)
