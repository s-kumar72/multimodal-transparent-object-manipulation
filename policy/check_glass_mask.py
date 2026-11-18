"""
Is the glass mask of the segmentation network (policy/glass_seg.py) available where a sensor does not show the glass,
and which sensors does it need? No policy is involved.

  python policy/check_glass_mask.py SEG_CKPT --test data/fusion_bench/sim/L0_bench_test_a_in_distribution.npz \
      --test data/fusion_bench/sim/L0_bench_test_g_heldout_all.npz [--demos ZARR --masks NPY] [-d cuda:0]

--test: segmentation test frames with the true mask (fusion/collect_sim.py). The predicted mask with all sensors,
        without thermal, without depth and with rgb alone is compared with the true one (IoU, distance of the centroids),
        by whether depth / thermal show the glass (visibility_cues at the true mask, thresholds 15 / 30).
--demos / --masks: a demo zarr and its stored masks (policy/make_glass_masks.py, all sensors). 8000 random frames; the
        masks with fewer sensors are compared with the stored ones, by the scene flags of the demos.
A sensor is left out by marking it as not available (fusion/model.py, avail), as the network was trained.
"""
import click
import numpy as np

from policy.glass_seg import predict_glass_mask
from policy.obs_util import THERMAL_NOISE_COUNTS, glass_location, visibility_cues

SETS = (("all sensors", None), ("without thermal", ["rgb", "depth"]), ("without depth", ["rgb", "thermal"]), ("rgb only", ["rgb"]))


def compare(pred, ref, sel):
    """IoU and centroid distance (pixels; 99 when the predicted mask is not found) of pred against ref on frames sel."""
    H, W = ref.shape[-2:]
    lp, lr = glass_location(pred), glass_location(ref)
    inter, union = (pred & ref).reshape(len(ref), -1).sum(1), (pred | ref).reshape(len(ref), -1).sum(1)
    dist = np.where(lp[:, 0] > 0, np.hypot((lp[:, 1] - lr[:, 1]) * W, (lp[:, 2] - lr[:, 2]) * H), 99.0)
    return (f"found {lp[sel, 0].mean():.3f} | IoU {(inter / np.maximum(union, 1))[sel].mean():.3f} | centroid within 2 px "
            f"{np.mean(dist[sel] < 2):.3f}, within 4 px {np.mean(dist[sel] < 4):.3f}, median {np.median(dist[sel]):.2f} px")


@click.command()
@click.argument("ckpt")
@click.option("--test", "tests", multiple=True, help="npz with rgb, depth, thermal, thermal_raw, mask, is_crossover, episode")
@click.option("--demos", default=None, help="demo zarr")
@click.option("--masks", default=None, help="stored masks of the demo zarr (all sensors)")
@click.option("-d", "--device", default="cuda:0", show_default=True)
@click.option("--n_frames", default=8000, show_default=True)
def main(ckpt, tests, demos, masks, device, n_frames):
    for path in tests:
        d = dict(np.load(path, allow_pickle=True))
        gt = d["mask"] > 0
        ok = glass_location(gt)[:, 0] > 0
        v = visibility_cues(d["depth"], d["thermal_raw"].astype(np.float32), gt, THERMAL_NOISE_COUNTS)
        dvis, tvis = v[:, 0] > 15, v[:, 1] > 30
        print(f"== {path}: {len(gt)} frames, {len(set(d['episode']))} episodes; depth shows the glass {dvis.mean():.3f}, "
              f"thermal shows it {tvis.mean():.3f}, crossover {np.mean(d['is_crossover'] > 0):.3f}")
        preds = {name: predict_glass_mask(d, ckpt, device, mods=mods) for name, mods in SETS}
        for name, sel in (("depth and thermal show it", dvis & tvis), ("depth blind, thermal shows it", ~dvis & tvis),
                          ("thermal blind, depth shows it", dvis & ~tvis), ("both blind", ~dvis & ~tvis)):
            sel = sel & ok
            print(f"  {name}: {sel.sum()} frames, {len(set(d['episode'][sel]))} episodes")
            for lab, m in preds.items():
                if sel.any():
                    print(f"     {lab:16s} vs the true mask: {compare(m, gt, sel)}")
    if demos:
        import zarr
        z = zarr.open(demos, "r")
        ends = z["meta"]["episode_ends"][:]
        ep = np.repeat(np.arange(len(ends)), np.diff(np.r_[0, ends]))
        st, xo = z["meta"]["ep_glass_seethrough"][:][ep], z["meta"]["ep_is_crossover"][:][ep]
        idx = np.sort(np.random.default_rng(0).choice(len(ep), min(n_frames, len(ep)), replace=False))
        raw = {k: z["data"][k].get_orthogonal_selection((idx,)) for k in ("rgb", "depth", "thermal")}
        stored = np.load(masks)[idx][..., 0] > 0
        ok = glass_location(stored)[:, 0] > 0
        print(f"== {demos}: {len(idx)} frames; stored mask found in {ok.mean():.4f}")
        preds = {name: predict_glass_mask(raw, ckpt, device, mods=mods) for name, mods in SETS}
        print(f"  predicted again with all sensors: pixel agreement with the stored masks {(preds['all sensors'] == stored).mean():.4f}")
        for name, sel in (("normal", (st[idx] == 0) & (xo[idx] == 0)), ("see-through", (st[idx] == 1) & (xo[idx] == 0)),
                          ("crossover", (st[idx] == 0) & (xo[idx] == 1)), ("see-through and crossover", (st[idx] == 1) & (xo[idx] == 1))):
            sel = sel & ok
            print(f"  {name}: {sel.sum()} frames, {len(set(ep[idx][sel]))} demos")
            for lab, m in list(preds.items())[1:]:
                if sel.any():
                    print(f"     {lab:16s} vs the stored mask: {compare(m, stored, sel)}")


if __name__ == "__main__":
    main()
