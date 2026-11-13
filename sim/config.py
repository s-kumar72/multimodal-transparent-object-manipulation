"""
Scene and thermal configs for GlassPickEnv, loaded from YAML (sim/configs/).

A YAML file may start with `base: other.yaml` (path relative to the file) and override any of its keys; nested dicts
are merged. `thermal:` is either a dict or the path of a thermal YAML. The default config reproduces main.

Factors with a value pool can give one pool per split, e.g. `background: {train: [checker, wood], test: [stripes]}`.
A factor draws from its test pool only when split == "test" and the factor is listed in `heldout`, so the
in-distribution test set is split "test" with heldout [] and the fully held-out one lists every factor.
"""
import dataclasses
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

CONFIG_DIR = Path(__file__).parent / "configs"
# Fixed ids of the per-factor RNG streams (np.random.default_rng([seed, id])); never renumber
FACTORS = {"background": 1, "lighting": 2, "transparent": 3, "liquid": 4, "distractors": 5,
           "thermal": 6, "sensor": 7}
HELDOUT_ALL = ["background", "lighting", "distractors"]
# Thermal config names stored as ids in meta/ep_thermal_config (append only)
THERMAL_NAMES = ["v0", "calibrated_v1", "calibrated_v1_narrow", "calibrated_v1_fixed"]

@dataclass
class ThermalConfig:
    name: str = "v0"
    pipeline: str = "v0"            # v0: main's float pipeline (data/thermal identical to main); raw: 14-bit counts
    # Scene temperatures (degC). The first block is drawn from the legacy RNG in main's order.
    t_env: tuple = (18.0, 26.0)
    crossover_p: float = 0.3        # object within crossover_gap of ambient
    crossover_gap: tuple = (-1.0, 1.0)
    gap: tuple = (5.0, 35.0)        # otherwise +-gap
    table_dt: tuple = (-1.0, 1.0)
    link_dt: tuple = (5.0, 12.0)
    obj_emis: tuple = (0.85, 0.95)
    table_emis: tuple = (0.90, 0.97)
    realistic_p: float = 0.0        # with this probability the glass is slightly below ambient (evaporation / AC)
    realistic_gap: tuple = (-1.25, -0.76)
    distractor_ambient_p: float = 0.7   # distractors within +-1 degC of ambient, else +-distractor_gap
    distractor_gap: tuple = (5.0, 35.0)
    liquid_temp: dict = field(default_factory=lambda: {"cold": (4.0, 10.0), "room": None, "warm": (40.0, 60.0)})
    # Camera: raw = gain * band radiance + offset (14-bit counts), then blur and noise at 160x120
    gain: float = 42.67
    offset: float = 811.2
    noise: float = 0.002            # v0: noise std as a fraction of the mean radiance (main)
    noise_counts: float = 2.73      # raw: noise std in counts at 160x120
    blur_sigma: float = 1.0         # raw: Gaussian blur in 160x120 pixels
    # data/thermal (uint8) from thermal_raw with common/thermal.py; "legacy" = main's per-frame min-max
    normalize: str = "legacy"
    norm_range: tuple = None

@dataclass
class SceneConfig:
    name: str = "default"
    level: int = 0
    split: str = "train"
    heldout: list = field(default_factory=lambda: list(HELDOUT_ALL))
    thermal: ThermalConfig = field(default_factory=ThermalConfig)
    camera: str = "front_cam"
    # Lever A: alpha / tint of the transparent objects (None keeps the XML rgba)
    alpha: tuple = None
    tint: float = 0.0
    # Table look and light (names defined in sim/env.py); "xml" = main's table
    background: dict = field(default_factory=lambda: {"train": ["xml"], "test": ["xml"]})
    background_jitter: float = 0.0
    lighting: dict = field(default_factory=lambda: {"train": ["normal"], "test": ["normal"]})
    light_dir_jitter: float = 0.0   # degrees
    light_jitter: float = 0.0       # relative intensity
    # Liquid inside the objects that can hold it (fill = fraction of the body height)
    liquid_fill: dict = field(default_factory=lambda: {"train": [0.0], "test": [0.0]})
    liquid_alpha: float = 0.15
    # Opaque distractors on the table, plus lever B (opaque copy of the target) and lever C (depth absorbers)
    n_distractors: tuple = (0, 0)
    distractors: dict = field(default_factory=lambda: {"train": ["mug", "box", "can", "bowl"],
                                                       "test": ["opaque_bottle", "opaque_cup"]})
    lookalike_p: float = 0.0
    n_absorbers: tuple = (0, 0)
    absorber_holes: tuple = (0.3, 0.8)   # fraction of an absorber's pixels with no depth reading
    # RGB camera: Gaussian noise (std, 0-255) on the 96x96 image, the shot / read noise of a low-light exposure
    rgb_noise: float = 0.0
    # Depth sensor
    glass_holes: float = 0.4
    n_table_blobs: tuple = (0, 0)        # lever C: hole patches on the table, glass-sized
    edge_dropout: float = 0.0            # lever C: fraction of depth-edge pixels with no reading
    # How depth defects look. dots: every pixel of a defect region independently missing (main, L3, L4).
    # blobs (L3b+): glass, absorbers and table patches all get the same structured defects, drawn per region and
    # episode from the same ranges: connected missing blobs (blurred noise thresholded at the missing fraction) and
    # blobs of wrong depth (multipath), so the shape of the holes does not tell glass apart.
    depth_defects: str = "dots"
    defect_missing: tuple = (0.1, 0.9)   # fraction of a region with no reading
    defect_size: tuple = (1.5, 5.0)      # blob size: blur sigma in pixels of the 320x240 depth image
    defect_wrong: tuple = (0.0, 0.3)     # fraction of the remaining pixels with wrong depth
    defect_offset: tuple = (0.05, 0.3)   # typical depth error (m) of those pixels
    # blobs only: per episode, probability that the depth sensor sees through the glass (no missing or wrong depth on
    # it, only the surface behind), like a time-of-flight sensor; 0 draws nothing, so existing levels are unchanged
    glass_seethrough_p: float = 0.0

    def pool(self, factor):
        """Value pool of a factor for this config's split."""
        p = getattr(self, factor)
        if isinstance(p, dict) and set(p) <= {"train", "test"}:
            return p["test" if self.split == "test" and factor in self.heldout else "train"]
        return p

def factor_rng(seed, factor):
    return np.random.default_rng(None if seed is None else [seed, FACTORS[factor]])

def _merge(a, b):
    out = dict(a)
    for k, v in b.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out

def _find(path, rel_to=None):
    """A config path: as given, next to the file that names it, or under sim/configs."""
    for p in [Path(path)] + ([Path(rel_to) / path] if rel_to else []) + [CONFIG_DIR / path]:
        if p.exists():
            return p
    raise FileNotFoundError(path)

def _load_yaml(path, rel_to=None):
    path = _find(path, rel_to)
    d = yaml.safe_load(path.read_text()) or {}
    if "base" in d:
        d = _merge(_load_yaml(d.pop("base"), path.parent), d)
    th = d.get("thermal")
    if isinstance(th, str):
        d["thermal"] = _load_yaml(th, path.parent)
    elif isinstance(th, dict) and "base" in th:   # thermal: {base: thermal/x.yaml, key: override}
        d["thermal"] = _merge(_load_yaml(th.pop("base"), path.parent), th)
    return d

def _tuples(cls, d):
    """YAML lists -> tuples for tuple-typed fields, so configs compare and hash like the defaults."""
    types = {f.name: f.type for f in dataclasses.fields(cls)}
    return {k: tuple(v) if types.get(k) in ("tuple", tuple) and isinstance(v, list) else v for k, v in d.items()}

def make_config(d=None, **overrides):
    d = _merge(d or {}, overrides)
    thermal = d.pop("thermal", {})
    if isinstance(thermal, str):
        thermal = _load_yaml(thermal)
    th = thermal if isinstance(thermal, ThermalConfig) else ThermalConfig(**_tuples(ThermalConfig, thermal))
    return SceneConfig(thermal=th, **_tuples(SceneConfig, d))

def load_config(path=None, **overrides):
    """path: YAML file (absolute, relative to cwd, or relative to sim/configs). None = default (= main)."""
    return make_config(_load_yaml(path) if path else {}, **overrides)

def config_dict(cfg):
    return dataclasses.asdict(cfg)
