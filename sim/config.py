"""
Scene randomization and thermal configs. DEFAULT_SCENE / DEFAULT_THERMAL reproduce the original scene and thermal
model ("v0"), so GlassPickEnv() behaves as before.
A YAML file only lists what it changes, e.g. sim/configs/scene/randomized_v1.yaml or sim/configs/thermal/v0.yaml.
Each randomized scene factor has a "train" and a "test" (held-out) pool; GlassPickEnv(split=...) picks one.
The scene config names its thermal config ("thermal": name in sim/configs/thermal/).
"""
import copy
from pathlib import Path

import yaml

CONFIG_DIR = Path("sim/configs/scene")
THERMAL_DIR = Path("sim/configs/thermal")

# Table materials (bg_<name> in transparent_scene.xml). emis = table emissivity range, None keeps the default draw
BACKGROUNDS = {
    "plain_brown": None, "plain_gray": None, "glasslike_blue": None,
    "glasslike_white": (0.85, 0.92),    # glossy laminate
    "checker_bw": (0.88, 0.95), "checker_color": (0.88, 0.95),
    "stripes_a": (0.88, 0.95), "stripes_b": (0.88, 0.95),
    "wood_light": (0.85, 0.93), "wood_dark": (0.85, 0.93),
    "noise_fine": (0.88, 0.95), "noise_coarse": (0.88, 0.95),
}

# key = top light, head = camera headlight, back/spot = extra lights (0 = off). Lighting never changes thermal.
LIGHTING = {
    "normal": dict(key=0.7, key_ambient=0.0, head=0.6, head_ambient=0.3, back=0.0, spot=0.0),
    "dim": dict(key=0.25, key_ambient=0.0, head=0.2, head_ambient=0.12, back=0.0, spot=0.0),
    "bright": dict(key=0.85, key_ambient=0.0, head=0.7, head_ambient=0.35, back=0.0, spot=0.0),
    "backlight": dict(key=0.2, key_ambient=0.0, head=0.25, head_ambient=0.15, back=0.9, spot=0.0),
    "spot": dict(key=0.1, key_ambient=0.0, head=0.25, head_ambient=0.1, back=0.0, spot=1.0),
}

DISTRACTORS = ["mug", "box", "can", "bowl", "bottle", "opaque_cup"]   # dist_<name> bodies in the scene XML

DEFAULT_SCENE = {
    "name": "default",
    "thermal": "v0",
    "background": {"train": ["plain_brown"], "test": ["plain_brown"], "color_jitter": 0.0, "randomize_floor": False},
    "lighting": {"train": ["normal"], "test": ["normal"],
                 "jitter": {"dir_deg": 0.0, "diffuse": 0.0, "ambient": 0.0, "headlight": 0.0}},
    # liquid temp (C): cold / warm are absolute, room is relative to t_env
    "liquid": {"fill_levels": [0.0], "alpha": [0.1, 0.2], "temp": {"cold": [5, 12], "room": [-1, 1], "warm": [40, 60]}},
    # min_dist_glass / min_dist_each = gap (m) between footprints, so the open gripper still fits around the glass
    "distractors": {"train": [], "test": [], "max_count": 0, "min_dist_glass": 0.09, "min_dist_each": 0.02},
    "glass_appearance": {"enabled": False, "alpha": [0.05, 0.2], "tint": 0.1},
}

# Thermal model (temperatures in C). "v0" = the original hand-picked values, not yet compared with real thermal images.
DEFAULT_THERMAL = {
    "name": "v0",
    "t_env": [18, 26],
    # glass - t_env: crossover (|gap| <= 1) with crossover_p, otherwise +-[5, 35]
    "crossover_p": 0.3, "crossover_gap": [-1, 1], "gap_abs": [5, 35],
    "table_dt": [-1, 1],    # table - t_env
    "link_dt": [5, 12],     # robot links - t_env (motors warm)
    # Distractors: mostly near t_env, sometimes hot or cold (absolute C), so the glass is not the only hot/cold object
    "distractor": {"p_off_ambient": 0.3, "hot": [35, 60], "cold": [5, 12], "ambient_dt": [-1, 1]},
    # Emissivity ranges; "other" = everything without a range (robot, floor). Table materials may override (BACKGROUNDS)
    "emis": {"glass": [0.85, 0.95], "table": [0.90, 0.97], "distractor": [0.90, 0.97], "other": 0.95},
    "liquid_mix": 0.8,      # glass wall below the liquid surface: liquid_mix * liquid + (1 - liquid_mix) * glass
    # Camera: raw counts = gain * band radiance + offset (14-bit, v0 placeholder: 0 C -> ~6830, 100 C -> ~15010)
    "raw": {"gain": 80.0, "offset": 4000.0},
    "resolution": [120, 160],
    "blur_sigma_px": None,  # None = original 3x3 binomial blur, else Gaussian sigma in thermal pixels
    # Noise std in raw counts = rel * (frame mean - offset) + counts (v0: 0.2% of the signal)
    "noise": {"rel": 0.002, "counts": 0.0},
}

def _merge(base, new, path=""):
    for k, v in new.items():
        if k not in base:
            raise KeyError(f"unknown config key {path}{k}")
        base[k] = _merge(base[k], v, f"{path}{k}.") if isinstance(base[k], dict) else v
    return base

def load_scene_config(path=None):
    """path: YAML file, a name in sim/configs/scene/, or None for DEFAULT_SCENE."""
    cfg = copy.deepcopy(DEFAULT_SCENE)
    if path is not None:
        p = Path(path) if str(path).endswith(".yaml") else CONFIG_DIR / f"{path}.yaml"
        _merge(cfg, yaml.safe_load(p.read_text()) or {})
    for factor, names in [("background", BACKGROUNDS), ("lighting", LIGHTING), ("distractors", DISTRACTORS)]:
        train, test = cfg[factor]["train"], cfg[factor]["test"]
        unknown = set(train + test) - set(names)
        assert not unknown, f"{factor}: unknown {unknown}"
        # Default (fixed) scene uses the same single value for both splits
        assert train == test or not set(train) & set(test), f"{factor}: train and test pools overlap"
    return cfg

def load_thermal_config(name=None):
    """name: YAML file, a name in sim/configs/thermal/, or None for DEFAULT_THERMAL (v0)."""
    cfg = copy.deepcopy(DEFAULT_THERMAL)
    if name is not None:
        p = Path(name) if str(name).endswith(".yaml") else THERMAL_DIR / f"{name}.yaml"
        _merge(cfg, yaml.safe_load(p.read_text()) or {})
    return cfg
