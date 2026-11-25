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
           "thermal": 6, "sensor": 7, "cell": 8, "place": 9}
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
    # Visibility cells (SceneConfig.cells), thermal blind episode: the apparent temperature of the glass is within
    # this many degC of that of the table behind it (sensors.matching_temp). 0.1 is about the camera noise: 2.73
    # counts per pixel at 34 counts per degC = 0.08 degC. Not used without cells.
    blind_contrast: float = 0.1
    # Visibility cells, thermal faint episode (cell flag 2): the apparent temperature of the glass is this many degC
    # (range, either sign) from that of the table behind it.
    faint_contrast: tuple = None
    # Visibility cells, thermal shows (cell flag 0), second version of the benchmark: when set, the apparent temperature
    # of the glass is this many degC (range, either sign) from that of the table behind it, as for faint, in place
    # of `gap` degC from the room. None (every level before V2): the glass is `gap` from the room, as always.
    show_contrast: tuple = None
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
    # dots or blobs: per episode, probability that the depth sensor sees through the glass (no missing or wrong depth
    # on it, only the surface behind), like a time-of-flight sensor; 0 draws nothing, so existing levels are unchanged
    glass_seethrough_p: float = 0.0
    # Visibility cells (None = off: existing levels draw nothing and are unchanged). A list of
    # [weight, rgb blind, depth blind, thermal blind] with integer weights: each episode gets one cell and the cell sets
    # the three causes, overriding their own probabilities: rgb blind = the room light at dark_light of "normal" with
    # dark_rgb_noise on the image (a low-light exposure), depth blind = a see-through episode (glass_seethrough_p),
    # thermal blind = the glass looks as warm as the table behind it, to within thermal.blind_contrast (a crossover
    # episode; crossover_gap is measured from the room temperature and leaves the glass visible against a table that
    # is warmer or colder than the room). A sensor that is not blind in the cell is not blind in the episode: normal
    # light, dot holes on the glass, the glass thermal.gap away from the room temperature. The cells are dealt to the
    # seeds in a fixed shuffled order (cell_pattern), and with cells the object of seed k is object k % 5 unless the
    # caller names one (the test sets name the same one), so any 5 * sum(weights) consecutive seeds hold every
    # (cell, object) pair exactly weight times. Episode info then has cell (index), rgb_dark and thermal_contrast
    # (apparent, glass - table, degC).
    cells: list = None
    dark_light: float = 0.02
    dark_rgb_noise: float = 0.0
    # How an rgb-blind cell hides the glass from the colour camera. "dark": the dark room above. "clear": normal
    # light and the glass at clear_alpha, below one grey level, so the colour image shows the table, the arm and the
    # distractors but not the glass (rgb blind hides only the glass, as depth blind and thermal blind do).
    rgb_blind: str = "dark"
    clear_alpha: float = 0.001
    # A cell flag may also be 2, "faint": the sensor shows the glass weakly. rgb: the glass at faint_alpha; depth:
    # faint_holes of the glass pixels are holes instead of glass_holes; thermal: thermal.faint_contrast. Episode info
    # then also has rgb_faint, depth_faint and thermal_faint (only in levels that have such a cell).
    faint_alpha: float = None
    faint_holes: float = None
    # Second version of the benchmark (V2 levels; None or a single number leaves every earlier level as it is):
    # show_holes: in a cell where depth shows the glass, the share of the glass's pixels that are holes is drawn per
    # episode from this range (a hole-type depth sensor; None: glass_holes). faint_holes and clear_alpha may also be
    # a range [low, high], drawn per episode. These draws have RNG streams of their own (env._cell_draw).
    show_holes: tuple = None
    # Task. "pick": lift the glass 10 cm (every level before L0_vis_place; nothing below is drawn or loaded).
    # "place": pick the glass, carry it to the coaster and put it down upright on it, then let go. The coaster is an
    # opaque disc of this radius (m), 6 mm thick, fixed to the table at a random free spot within the arm's reach.
    # It always looks the same (dark with a light centre) and its temperature is within 1 degC of the room, so of the
    # three sensors it is the colour camera that shows it (sim/env.py, GlassPickEnv._place_target). Episode info
    # then has place_x, place_y, place_dist (glass to coaster at the start, m), place_clear (1: no distractor under
    # the straight path), place_temp and place_emis.
    task: str = "pick"
    place_radius: float = 0.06
    # Contact model. "default": the scene file's own (pyramidal friction cones, impratio 1, for every pick level;
    # place_scene.xml sets elliptic cones with impratio 10 itself). "elliptic": elliptic cones with impratio 10 on
    # any scene, i.e. the contacts of the pick-and-place level on a pick level (L0_vis_ell, a check of what the
    # contact model does to the pick; results/hard_scenario/contact_check/).
    contacts: str = "default"

    def pool(self, factor):
        """Value pool of a factor for this config's split."""
        p = getattr(self, factor)
        if isinstance(p, dict) and set(p) <= {"train", "test"}:
            return p["test" if self.split == "test" and factor in self.heldout else "train"]
        return p

def cell_pattern(cells, n_objects=5):
    """Cell index of seed k is cell_pattern(cells)[k % len(pattern)]. The pattern has n_objects * sum(weights) entries:
    the seeds with the same k % n_objects (the same object, see SceneConfig.cells) hold every cell exactly weight
    times, in a fixed shuffled order of their own."""
    w = [int(c[0]) for c in cells]
    assert all(x > 0 for x in w) and all(len(c) == 4 for c in cells), cells
    rng = np.random.default_rng(20261119)
    base = np.repeat(np.arange(len(w)), w)
    pat = np.empty(n_objects * len(base), int)
    for o in range(n_objects):
        pat[o::n_objects] = rng.permutation(base)
    return pat

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
