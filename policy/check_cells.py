"""
Check of a level with visibility cells (sim/config.py cells), before any policy is trained: per cell, can the glass
be located from each sensor alone, from the sensors the cell leaves visible, and from all three? No policy is involved.

  python policy/check_cells.py SEG_CKPT LEVEL TEST_SET=NPZ [TEST_SET=NPZ ...] [-d cuda:0] [--out FILE.json]

NPZ: segmentation frames of that test set with the true mask (fusion/collect_sim.py --levels LEVEL --test_sets ...).
The cell of a frame is the one its episode seed gets in that test set (cell_pattern). A sensor is left out by marking
it as not available to the segmentation network (fusion/model.py avail), as the network was trained.

Only the first frame of each episode is used (step 0), pooled over the given test sets: later frames come from the
scripted teacher, whose arm is already on its way to the glass and shows where it is in every sensor.

Printed per cell and sensor set, "located / chance":
  located  share of frames with the predicted centroid within 4 px of the true one (a frame with no mask counts as
           not located);
  chance   the same share when each prediction is compared with the glass of another episode of the cell (mean of 500
           shuffles). The glass spawns in a small part of the 96x96 image, so a guess that ignores the frame is within
           4 px in about a quarter of the frames. A sensor is blind in a cell when located is not above chance.
The last columns are the hand rule of the visibility gate at the true mask (visibility_cues: depth step > 15,
thermal step > 30), which should agree with the cell.
"""
import json
import sys

import click
import numpy as np

sys.path.insert(0, "sim")
from config import cell_pattern  # noqa: E402
from testsets import test_set  # noqa: E402

from policy.glass_seg import predict_glass_mask  # noqa: E402
from policy.obs_util import THERMAL_NOISE_COUNTS, glass_location, visibility_cues  # noqa: E402

MODS = ("rgb", "depth", "thermal")
SETS = (("rgb",), ("depth",), ("thermal",), "visible", MODS)


def centre(mask):
    """(found, x, y) of a mask in pixels."""
    H, W = mask.shape[-2:]
    loc = glass_location(mask)
    return loc[:, 0] > 0, loc[:, 1] * W, loc[:, 2] * H


def located_and_chance(pred, ref, rng, shuffles=500):
    """Share of frames with the predicted centroid within 4 px of the true one, and the same for shuffled truths."""
    found, x, y = centre(pred)
    _, rx, ry = centre(ref)
    hit = lambda idx: (found & (np.hypot(x - rx[idx], y - ry[idx]) < 4)).mean()
    n = len(x)
    return float(hit(np.arange(n))), float(np.mean([hit(rng.permutation(n)) for _ in range(shuffles)]))


@click.command()
@click.argument("ckpt")
@click.argument("level")
@click.argument("sets", nargs=-1)
@click.option("-d", "--device", default="cuda:0", show_default=True)
@click.option("--out", default=None, help="also write the table as JSON")
def main(ckpt, level, sets, device, out):
    parts, cells = [], None
    for spec in sets:
        name, path = spec.split("=", 1)
        cfg, _, _ = test_set(level, name)
        pat = cell_pattern(cfg.cells)
        assert cells is None or cells == cfg.cells, "the test sets must share their cells"
        cells = cfg.cells
        d = dict(np.load(path, allow_pickle=True))
        first = d["step"] == 0
        part = {k: d[k][first] for k in ("rgb", "depth", "thermal", "thermal_raw", "mask", "seed")}
        part["cell"] = pat[part["seed"] % len(pat)]
        parts.append(part)
    d = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    gt = d["mask"] > 0
    ok = glass_location(gt)[:, 0] > 0
    v = visibility_cues(d["depth"], d["thermal_raw"].astype(np.float32), gt, THERMAL_NOISE_COUNTS)
    dvis, tvis = v[:, 0] > 15, v[:, 1] > 30
    pred = {}
    for i, c in enumerate(cells):
        visible = tuple(m for m, blind in zip(MODS, c[1:4]) if not blind)
        for key in (("rgb",), ("depth",), ("thermal",), visible, MODS):
            if key not in pred:
                pred[key] = predict_glass_mask(d, ckpt, device, mods=list(key))
    rng = np.random.default_rng(0)
    print(f"== {level}: first frame of {len(gt)} episodes from {', '.join(s.split('=')[0] for s in sets)}; {ckpt}")
    print(f"  {'cell (sensors that show the glass)':36s}    n | rgb only      | depth only    | thermal only  | "
          "visible ones  | all three     | rule: depth shows, thermal shows")
    rows = []
    for i, c in enumerate(cells):
        sel = ok & (d["cell"] == i)
        visible = tuple(m for m, blind in zip(MODS, c[1:4]) if not blind)
        row = dict(cell=i, visible=list(visible), n=int(sel.sum()))
        txt = []
        for name, key in (("rgb", ("rgb",)), ("depth", ("depth",)), ("thermal", ("thermal",)), ("visible", visible),
                          ("all", MODS)):
            loc, ch = located_and_chance(pred[key][sel], gt[sel], rng)
            row[name], row[name + "_chance"] = loc, ch
            txt.append(f"{loc:.3f} / {ch:.3f}")
        row.update(rule_depth=float(dvis[sel].mean()), rule_thermal=float(tvis[sel].mean()))
        rows.append(row)
        print(f"  {i} {'+'.join(visible):34s} {row['n']:4d} | " + " | ".join(txt)
              + f" | {row['rule_depth']:.3f}, {row['rule_thermal']:.3f}")
    if out:
        json.dump(dict(level=level, ckpt=str(ckpt), sets=[s.split("=")[0] for s in sets], rows=rows),
                  open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
