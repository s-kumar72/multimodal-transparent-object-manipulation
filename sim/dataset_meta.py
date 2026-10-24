"""
Per-episode scene / thermal conditions, saved next to Diffusion Policy's meta/episode_ends.
Flat numeric arrays meta/ep_<field> of length n_episodes: DP's ReplayBuffer loads every array in meta/,
so there are no subgroups or strings. Names are stored as ids; meta.attrs["id_maps"] has the id -> name lists.
Older zarr stores without these arrays still load (load_episode_meta returns None).
"""
import numpy as np

from config import BACKGROUNDS, DISTRACTORS, LIGHTING

ID_MAPS = {"split": ["train", "test"], "background": list(BACKGROUNDS), "lighting": list(LIGHTING),
           "distractors": DISTRACTORS, "thermal_norm": ["fixed", "percentile", "minmax"]}
NAMED = ["split", "background", "lighting", "thermal_config", "thermal_norm"]
FLOATS = ["fill_level", "liquid_temp", "t_env", "glass_temp", "table_temp", "thermal_gap", "emis_glass", "emis_table"]
PER_DISTRACTOR = ["distractor_temps", "emis_distractors"]   # padded with NaN, same order as ep_distractors

def write_episode_meta(meta, rows, max_distractors, attrs=None):
    """meta: zarr group; rows: env.episode_meta() dicts (+ "thermal_norm") of the saved episodes, in order."""
    maps = dict(ID_MAPS, thermal_config=sorted({r["thermal_config"] for r in rows}))
    k = max(max_distractors, 1)

    def padded(values, fill, dtype):
        return np.array([list(v) + [fill] * (k - len(v)) for v in values], dtype).reshape(-1, k)

    for f in NAMED:
        meta.array(f"ep_{f}", np.array([maps[f].index(r[f]) for r in rows], np.int16))
    meta.array("ep_seed", np.array([r["seed"] for r in rows], np.int64))
    for f in FLOATS:
        meta.array(f"ep_{f}", np.array([r[f] for r in rows], np.float32))
    meta.array("ep_is_crossover", np.array([r["is_crossover"] for r in rows], bool))
    meta.array("ep_n_distractors", np.array([len(r["distractors"]) for r in rows], np.int8))
    meta.array("ep_distractors", padded([[DISTRACTORS.index(n) for n in r["distractors"]] for r in rows], -1, np.int16))
    for f in PER_DISTRACTOR:
        meta.array(f"ep_{f}", padded([r[f] for r in rows], np.nan, np.float32))
    meta.attrs["id_maps"] = maps
    meta.attrs.update(attrs or {})

def load_episode_meta(root):
    """Dict of per-episode arrays with names decoded (distractors: list of name lists), or None for older datasets."""
    meta = root["meta"]
    if "ep_seed" not in meta:
        return None
    maps = meta.attrs["id_maps"]
    out = {k[3:]: meta[k][:] for k in meta.array_keys() if k.startswith("ep_")}
    for f in NAMED:
        out[f] = np.array(maps[f])[out[f]]
    out["distractors"] = [[maps["distractors"][i] for i in row if i >= 0] for row in out["distractors"]]
    return out
