"""
Glass-picking environment with randomized scene.
The scene is set by a config (sim/config.py, sim/configs/*.yaml); the default config reproduces main exactly.
Author: Sadhana Kumar
"""
import mujoco
import numpy as np
from PIL import Image

from common.thermal import normalize_thermal_raw
from config import FACTORS, SceneConfig, cell_pattern, factor_rng, load_config
from sensors import Sensors, apparent_contrast, matching_temp

SCENE = "sim/assets/gen3_lite/transparent_scene.xml"
SCENE_DISTRACTORS = "sim/assets/gen3_lite/distractor_scene.xml" # SCENE + parked distractors (only when used)
SCENE_PLACE = "sim/assets/gen3_lite/place_scene.xml" # SCENE_DISTRACTORS + the coaster (only for cfg.task "place")
SCENE_ALIGN = "sim/assets/gen3_lite/align_scene.xml" # SCENE_DISTRACTORS + elongated glasses and a mat (cfg.task "align")
SCENE_PLACE_UNSEEN = "sim/assets/gen3_lite/place_unseen_scene.xml" # SCENE_PLACE + the glass shapes of no training set
SCENE_TWO = "sim/assets/gen3_lite/two_scene.xml"       # SCENE_PLACE + a second coaster (cfg.task "two")

# This position is arm above table, gripper down
READY = np.array([-0.118, -0.575, 0.968, 1.571, 1.599, -1.689])
IMG = 96 # image size for Diffusion Policy
CTRL_DT = 0.1 # 10 Hz, same as Diffusion Policy

# Transparent objects in transparent_scene.xml. Every body's origin is at its base, so the value is the
# height above the base where the teacher closes the gripper (the finger pads span +-2.5 cm around it).
# Necks and the wineglass bowl are grasped just below something wider (lip, bowl equator) so they cannot slip.
OBJECTS = {
    "tumbler": 0.07,     # cylinder 6 cm wide, 10 cm tall; grasped a bit above its middle
    "bottle": 0.135,     # 10 cm body + 7 cm tall, 2.6 cm wide neck with a lip on top; grasped at the neck
    "wineglass": 0.105,  # base, thin stem, 6 cm wide bowl; grasped just below the bowl's widest point
    "jar": 0.05,         # 5 x 5 x 8 cm box; grasped a bit above its middle
    "flask": 0.115,      # flat base, 7 cm bulb, 2.6 cm wide neck with a lip on top; grasped at the neck
}
OBJECT_NAMES = list(OBJECTS)
REST_Z = 0.022 # table top (0.02) + 2 mm, so the object starts upright on the table
# Where the object can spawn on the table. Top-down grasps cannot reach high enough beyond x = 0.42 to lift
# the taller objects, so the far 6 cm of the original 0.32-0.48 range is not used.
SPAWN_X, SPAWN_Y = (0.30, 0.42), (-0.15, 0.15)
PARK = [(1.6 + 0.25 * i, -1.0, 0.002) for i in range(len(OBJECT_NAMES))] # on the floor behind the camera
# Task "align" (second version of the benchmark, align_scene.xml): elongated glass boxes, long side along body x.
# The value is the grasp height above the base, as in OBJECTS. The policy is trained on ALIGN_NAMES; ALIGN_UNSEEN
# are shapes for a test set only. Ids in the episode records continue after those of OBJECT_NAMES.
ALIGN_OBJECTS = {"slab": 0.05, "bar": 0.045, "block": 0.065, "tile": 0.045}
ALIGN_NAMES, ALIGN_UNSEEN = ["slab", "bar", "block"], ["tile"]
ALIGN_PARK = {"slab": (3.2, -1.0, 0.002), "bar": (3.5, -1.0, 0.002), "block": (3.8, -1.0, 0.002), "tile": (4.1, -1.0, 0.002)}
# Glass shapes that are in no training set (place_unseen_scene.xml; cfg.objects "unseen", a test set): grasp heights.
UNSEEN_OBJECTS = {"tallglass": 0.085, "carafe": 0.125}
UNSEEN_PARK = {"tallglass": (4.4, -1.0, 0.002), "carafe": (4.7, -1.0, 0.002)}
GRASP_HEIGHT = {**OBJECTS, **ALIGN_OBJECTS, **UNSEEN_OBJECTS}
GLASS_IDS = list(GRASP_HEIGHT)   # the id of a glass in the episode records (the five first, as always)

def glass_names(cfg):
    """The glasses an episode of this config can have on the table."""
    if cfg.task == "align":
        return list(ALIGN_UNSEEN if cfg.objects == "unseen" else ALIGN_NAMES)
    return list(UNSEEN_OBJECTS) if cfg.objects == "unseen" else OBJECT_NAMES

# Table looks (materials in transparent_scene.xml); "xml" is main's plain brown table. The renderer uses the geom
# rgba (times the texture), so each look sets the table rgba; bright looks are toned down so they do not saturate.
BG_RGBA = {"solid": None, "checker": (0.9, 0.9, 0.9), "stripes": (0.9, 0.9, 0.9), "wood": (1.0, 1.0, 1.0),
           "noise": (0.95, 0.95, 0.95), "marble": (0.85, 0.85, 0.85), "skyblue": (0.40, 0.46, 0.51),
           "white": (0.5, 0.5, 0.5)}
BACKGROUNDS = ["xml", *BG_RGBA]
# Lighting presets: headlight (diffuse, ambient, colour), main light diffuse, extra lights switched on
LIGHTING = {
    "normal": dict(head=(0.6, 0.3), main=0.7, extra=[]),
    "dim": dict(head=(0.25, 0.12), main=0.3, extra=[]),
    "bright": dict(head=(0.62, 0.31), main=0.78, extra=[]),          # tuned so no table look saturates
    "backlight": dict(head=(0.25, 0.12), main=0.25, extra=["backlight"]),  # window behind the table
    "spot": dict(head=(0.25, 0.12), main=0.2, extra=["spot"]),     # one lamp over part of the table
    "dusk": dict(head=(0.18, 0.1), main=0.15, extra=[], colour=(1.0, 0.8, 0.6)),
    "dark": dict(head=(0.6, 0.3), main=0.7, extra=[]),   # "normal" times cfg.dark_light (visibility cells, rgb blind)
}
LIGHTING_NAMES = list(LIGHTING)
# Objects that can hold liquid, with the height of the part the liquid fills (m)
LIQUID_OBJECTS = {"tumbler": 0.10, "bottle": 0.10, "jar": 0.08}
LIQUID_KINDS = ["cold", "room", "warm"]
# Opaque objects in distractor_scene.xml: plain distractors, opaque look-alikes (lever B), depth absorbers (lever C)
DISTRACTORS = ["mug", "box", "can", "bowl", "opaque_bottle", "opaque_cup"]
LOOKALIKES = ["la_" + n for n in OBJECT_NAMES]
ABSORBERS = ["abs_black_box", "abs_black_cyl", "abs_metal_can", "abs_metal_box"]
EXTRA_NAMES = DISTRACTORS + LOOKALIKES + ABSORBERS
MAX_EXTRA = 6 # 3 distractors + 1 look-alike + 2 absorbers
# LWIR emissivity ranges: painted / ceramic / plastic / cardboard ~0.9, matte black paint ~0.97, bare metal 0.05-0.3
EMISSIVITY = {"abs_metal_can": (0.05, 0.3), "abs_metal_box": (0.05, 0.3),
              "abs_black_box": (0.95, 0.98), "abs_black_cyl": (0.95, 0.98)}
PALETTE = np.array([[0.85, 0.2, 0.15], [0.15, 0.45, 0.8], [0.2, 0.65, 0.3], [0.95, 0.8, 0.2], [0.9, 0.9, 0.9],
                    [0.35, 0.35, 0.4], [0.6, 0.3, 0.6], [0.95, 0.55, 0.2], [0.55, 0.75, 0.85], [0.45, 0.3, 0.2]])
EXTRA_X, EXTRA_Y = (0.24, 0.56), (-0.24, 0.24) # where distractors go (on the table, in view)
EXTRA_X_NARROW, EXTRA_Y_NARROW = (0.28, 0.50), (-0.13, 0.13) # inside the narrow camera's view
GAP_TARGET, GAP_EXTRA = 0.09, 0.02 # minimum edge-to-edge distance to the target / between distractors
# Pick-and-place (cfg.task "place"). The coaster goes on the near part of the table, around the region the glass spawns
# in: entirely on the table (which spans x 0.20-0.70, y +-0.35) and no further from the arm's base than the far
# corners of the spawn region (PLACE_REACH), where the top-down IK still reaches the carry height.
TABLE_TOP = 0.02
COASTER_H = 0.006           # thickness of the coaster (place_scene.xml)
PLACE_X, PLACE_Y = (0.27, 0.42), (-0.28, 0.28)
PLACE_REACH = float(np.hypot(SPAWN_X[1], SPAWN_Y[1]))
PLACE_CLEAR = 0.015         # minimum edge-to-edge distance between the carried glass and a distractor under its path
PLACE_UPRIGHT = 0.985       # cos 10 deg: the placed glass leans less than this
PLACE_Z_TOL = 0.005         # the placed glass rests this close (m) to the height of the coaster's top
PLACE_RELEASE = 0.05        # the gripper has let go when the tool is this far (m) above the glass's grasp point
PLACE_STILL = 0.05          # ... and the glass moves slower than this (m/s)
# Task "align": the mat (align_scene.xml), and how close the glass has to lie to it
MAT_HALF = (0.085, 0.035)   # half sides of the mat (m): long, short
ALIGN_POS_TOL = 0.02        # the centre of the glass is within this (m) of the centre of the mat
ALIGN_YAW_TOL = 15.0        # its long side within this many degrees of the mat's (either way round)
# Task "two": the two glasses start at least this far apart (m, edge to edge: the open gripper is 14 cm wide, and at
# 5 cm it knocked the neighbour over in 2 of 120 teacher episodes); a glass counts as let go when the tool
# is PLACE_RELEASE above its grasp point or this far from it in the table plane (the arm is at the other glass)
TWO_GAP = 0.07
TWO_AWAY = 0.08

def _resize(img):
    return np.asarray(Image.fromarray(img).resize((IMG, IMG), Image.BILINEAR))

def _resize_raw(raw):
    """Raw counts (float) -> 96x96 uint16, bilinear like the 8-bit images."""
    small = np.asarray(Image.fromarray(raw.astype(np.float32), mode="F").resize((IMG, IMG), Image.BILINEAR))
    return np.clip(np.round(small), 0, 16383).astype(np.uint16)

def _seg_dist(p, a, b):
    """Distance from point p to the segment a-b (2-D)."""
    ab = b - a
    t = np.clip(np.dot(p - a, ab) / max(np.dot(ab, ab), 1e-12), 0, 1)
    return float(np.linalg.norm(p - (a + t * ab)))

def _footprint(model, body):
    """Radius (m) of a circle around the body's origin that covers its geoms seen from above."""
    r = 0.0
    for g in np.flatnonzero(model.geom_bodyid == body):
        size = model.geom_size[g]
        half = np.hypot(*size[:2]) if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX else size[0]
        r = max(r, np.hypot(*model.geom_pos[g][:2]) + half)
    return r

class GlassPickEnv:
    def __init__(self, max_steps=120, config=None, split=None):
        """config: SceneConfig, YAML path, or None (default = main). split overrides the config's split."""
        cfg = config if isinstance(config, SceneConfig) else load_config(config)
        if split is not None:
            cfg.split = split
        self.cfg = cfg
        self.uses_extras = max(cfg.n_distractors) > 0 or cfg.lookalike_p > 0 or max(cfg.n_absorbers) > 0
        assert cfg.task in ("pick", "place", "align", "two"), cfg.task
        self.align = cfg.task == "align"     # T2: an elongated glass at an angle, a mat with a direction
        self.two = cfg.task == "two"         # T3: two glasses and two coasters
        self.place = cfg.task in ("place", "align", "two")   # the scene has a place target and success is placed()
        # objects an episode can have on the table / every glass body of the scene (ids of the episode records)
        assert cfg.objects in (None, "unseen"), cfg.objects
        unseen = cfg.objects == "unseen"
        assert not unseen or (self.place and not self.two), "the unseen shapes are for the tasks T1 and T2"
        assert not self.two or (cfg.cells and cfg.thermal.show_contrast is not None), "task two needs pair cells of V2"
        self.names = glass_names(cfg)
        # the glasses of this scene beyond the five, parked like them, and every glass body of the scene
        self.extra_park = dict(ALIGN_PARK) if self.align else dict(UNSEEN_PARK) if unseen else {}
        self.all_names = OBJECT_NAMES + list(self.extra_park)
        scene = SCENE_ALIGN if self.align else SCENE_TWO if self.two else SCENE_PLACE_UNSEEN if unseen else SCENE_PLACE if self.place else \
            SCENE_DISTRACTORS if self.uses_extras else SCENE
        self.model = mujoco.MjModel.from_xml_path(scene)
        assert cfg.contacts in ("default", "elliptic"), cfg.contacts
        if cfg.contacts == "elliptic":       # the contact model of place_scene.xml on any scene
            self.model.opt.cone, self.model.opt.impratio = mujoco.mjtCone.mjCONE_ELLIPTIC, 10
        self.data = mujoco.MjData(self.model)
        self.sensors = Sensors(self.model, camera=cfg.camera, config=cfg)
        self.n_sub = int(round(CTRL_DT / self.model.opt.timestep))
        self.obj_q = {n: self.model.jnt_qposadr[self.model.body(n).jntadr[0]] for n in self.all_names}
        self.link_names = [self.model.body(i).name for i in range(self.model.nbody) if "link" in self.model.body(i).name]
        self.max_steps = max_steps
        self.obj = OBJECT_NAMES[0]
        m = self.model
        self.extra_q = {n: m.jnt_qposadr[m.body(n).jntadr[0]] for n in EXTRA_NAMES} if self.uses_extras else {}
        self.extra_r = {n: _footprint(m, m.body(n).id) for n in self.extra_q}
        self.obj_r = {n: _footprint(m, m.body(n).id) for n in self.all_names}
        self.table_geom = int(np.flatnonzero(m.geom_bodyid == m.body("table").id)[0])
        self.glass_geoms = {n: [g for g in np.flatnonzero(m.geom_bodyid == m.body(n).id)
                                if not m.geom(g).name.startswith("liquid_")] for n in self.all_names}
        # Model fields the factors change; restored on every reset
        self._fields = ["geom_rgba", "geom_matid", "geom_size", "geom_pos", "geom_contype", "geom_conaffinity",
                        "body_contype", "body_conaffinity", "body_gravcomp", "light_active", "light_diffuse",
                        "light_dir", "light_pos", "light_specular"]
        if self.place:
            target = "mat" if self.align else "coaster"
            self.coaster = m.body(target).id
            self.coaster_geoms = [m.geom(target).id, m.geom(target + "_top").id]   # the disc and its light centre
            self.obj_dof = {n: m.jnt_dofadr[m.body(n).jntadr[0]] for n in self.all_names}
            self._fields.append("body_pos")
            if self.align:
                self._fields.append("body_quat")
            if self.two:
                self.coasters = [self.coaster, m.body("coaster2").id]
                self.coasters_geoms = [self.coaster_geoms, [m.geom("coaster2").id, m.geom("coaster2_top").id]]
        self._defaults = {f: getattr(m, f).copy() for f in self._fields}
        self._head = (m.vis.headlight.diffuse.copy(), m.vis.headlight.ambient.copy())
        self._head_spec = m.vis.headlight.specular.copy()
        self.dark = False
        self.obj2 = self.cell2 = None

    def reset(self, seed=None, obj=None):
        """obj: object name to put on the table; None picks one at random."""
        self.rng = np.random.default_rng(seed)
        r, m, d, cfg, th = self.rng, self.model, self.data, self.cfg, self.cfg.thermal
        for f, v in self._defaults.items():
            getattr(m, f)[:] = v
        m.vis.headlight.diffuse[:], m.vis.headlight.ambient[:] = self._head
        m.vis.headlight.specular[:] = self._head_spec
        mujoco.mj_resetData(m, d)
        d.qpos[:6] = READY
        d.ctrl[:6] = READY

        # One object upright on the table, the others parked out of view (legacy RNG, main's order)
        cell = self._cell(seed)              # None, or (index, rgb blind, depth blind, thermal blind, object index)
        self.obj = obj if obj is not None else self.names[r.integers(len(self.names))]
        if obj is None and cell is not None:
            self.obj = self.names[cell[4]]   # with cells the object is dealt with the cell (seed % 5), not drawn
        for name, park in zip(OBJECT_NAMES, PARK):
            q = self.obj_q[name]
            pos = [r.uniform(*SPAWN_X), r.uniform(*SPAWN_Y), REST_Z] if name == self.obj else park
            d.qpos[q:q + 7] = [*pos, 1, 0, 0, 0]
        # task "align": the elongated glass lies at a random angle (its long side, 0 to 180 deg)
        self.obj_yaw = float(factor_rng(seed, "align").uniform(0, np.pi)) if self.align else 0.0
        for name, park in self.extra_park.items():   # the glasses of this scene beyond the five (none before V2)
            q = self.obj_q[name]
            if name == self.obj:
                d.qpos[q:q + 7] = [r.uniform(*SPAWN_X), r.uniform(*SPAWN_Y), REST_Z,
                                   np.cos(self.obj_yaw / 2), 0, 0, np.sin(self.obj_yaw / 2)]
            else:
                d.qpos[q:q + 7] = [*park, 1, 0, 0, 0]
        self.obj2 = None
        if self.two:                         # T3: a second glass of another shape, clear of the first
            rb = factor_rng(seed, "two")
            i = self.names.index(self.obj)
            self.obj2 = self.names[(i + 1 + int(rb.integers(len(self.names) - 1))) % len(self.names)]
            qa, qb = self.obj_q[self.obj], self.obj_q[self.obj2]
            first, need = d.qpos[qa:qa + 2].copy(), self.obj_r[self.obj] + self.obj_r[self.obj2] + TWO_GAP
            for _ in range(500):
                xy = np.array([rb.uniform(*SPAWN_X), rb.uniform(*SPAWN_Y)])
                if np.linalg.norm(xy - first) >= need:
                    break
            d.qpos[qb:qb + 7] = [*xy, REST_Z, 1, 0, 0, 0]

        # Thermal conditions (legacy RNG, main's order)
        self.t_env = r.uniform(*th.t_env)
        crossover = r.random() < th.crossover_p
        if cell is not None:
            crossover = cell[3] == 1
        gap = r.uniform(*th.crossover_gap) if crossover else r.choice([-1, 1]) * r.uniform(*th.gap)
        self.temps = {self.obj: self.t_env + gap, "table": self.t_env + r.uniform(*th.table_dt)}
        self.temps.update({n: self.t_env + r.uniform(*th.link_dt) for n in self.link_names})
        self.emis = {self.obj: r.uniform(*th.obj_emis), "table": r.uniform(*th.table_emis)}
        if cell is not None and cell[3]:     # thermal blind: the glass looks as warm as the table behind it
            rc = factor_rng(seed, "cell")
            c = rc.uniform(-th.blind_contrast, th.blind_contrast) if cell[3] == 1 else \
                rc.choice([-1.0, 1.0]) * rc.uniform(*th.faint_contrast)   # 2, faint: a small step against the table
            self.temps[self.obj] = matching_temp(self.temps["table"], self.emis["table"], self.emis[self.obj],
                                                 self.t_env, c)
            gap = self.temps[self.obj] - self.t_env
        elif cell is not None and th.show_contrast is not None:   # V2, thermal shows: a glass at rest against the table
            rc = self._cell_draw(seed, 3)
            c = rc.choice([-1.0, 1.0]) * rc.uniform(*th.show_contrast)
            self.temps[self.obj] = matching_temp(self.temps["table"], self.emis["table"], self.emis[self.obj],
                                                 self.t_env, c)
            gap = self.temps[self.obj] - self.t_env

        # New factors: each has its own RNG stream, so turning one on does not change the others
        rt = factor_rng(seed, "thermal")
        realistic, realistic_gap = rt.random() < th.realistic_p, rt.uniform(*th.realistic_gap)
        if realistic:   # measured on real glass: slightly below ambient
            gap, crossover = realistic_gap, False
            self.temps[self.obj] = self.t_env + gap
        info = dict(split=int(cfg.split == "test"), seed=-1 if seed is None else seed, level=cfg.level,
                    object=GLASS_IDS.index(self.obj), t_env=self.t_env, obj_temp=self.temps[self.obj],
                    thermal_gap=gap, is_crossover=int(crossover), realistic=int(realistic),
                    obj_emis=self.emis[self.obj])
        info.update(self._transparent(seed))
        base_alpha = info["alpha"]
        assert cfg.rgb_blind in ("dark", "clear"), cfg.rgb_blind
        if cell is not None and (cell[1] == 2 or cell[1] == 1 and cfg.rgb_blind == "clear"):
            a = cfg.faint_alpha if cell[1] == 2 else cfg.clear_alpha   # faint glass / not in the colour image at all
            if isinstance(a, (list, tuple)):                           # V2: a range, drawn per episode
                a = float(self._cell_draw(seed, 1).uniform(*a))
            for geoms in self.glass_geoms.values():
                self.model.geom_rgba[geoms, 3] = a
            info["alpha"] = a
        if self.two:                         # the second glass: its own colour, thermal and depth states
            info.update(self._second_glass(seed, base_alpha))
        info.update(self._background(seed))
        self.dark = cell is not None and cell[1] == 1 and cfg.rgb_blind == "dark"
        info.update(self._lighting(seed, self.dark))
        liquid = self._liquid(seed, info)
        extras = self._extras(seed, info) if self.uses_extras else []
        if self.place:
            info.update(self._mat_target(seed, extras) if self.align else self._two_targets(seed, extras) if self.two
                        else self._place_target(seed, extras))
        narrow = cfg.camera == "front_cam_narrow"
        self.sensor_scene = self.sensors.episode(seed, self.obj, liquid, extras, d,
                                                 (EXTRA_X_NARROW, EXTRA_Y_NARROW) if narrow else (EXTRA_X, EXTRA_Y),
                                                 force_seethrough=None if cell is None else cell[2] == 1,
                                                 glass_holes=self._cell_holes(seed, cell),
                                                 second=self._second_depth(seed) if self.two else None)
        if cfg.rgb_glass == "optical":       # the colour state of each glass as a strength of the optical model
            st = {0: 1.0, 1: 0.0, 2: cfg.faint_strength}
            self.sensor_scene["rgb_strength"] = {int(self.model.body(self.obj).id): 1.0 if cell is None else st[cell[1]]}
            if self.two:
                self.sensor_scene["rgb_strength"][int(self.model.body(self.obj2).id)] = st[self.cell2[0]]
        info["n_table_blobs"] = len(self.sensor_scene["blobs"])
        info["glass_seethrough"] = self.sensor_scene["seethrough"]   # 1 in a see-through episode (dots or blobs)
        if cell is not None:
            info.update(cell=cell[0], rgb_dark=int(self.dark), thermal_contrast=apparent_contrast(
                self.temps[self.obj], self.emis[self.obj], self.temps["table"], self.emis["table"], self.t_env))
            if any(2 in c[1:4] for c in cfg.cells):
                info.update(rgb_faint=int(cell[1] == 2), depth_faint=int(cell[2] == 2), thermal_faint=int(cell[3] == 2))
        self.episode_info = info

        for _ in range(300):
            mujoco.mj_step(m, d)
        self.obj_z0 = d.body(self.obj).xpos[2]
        self.z0 = {n: d.body(n).xpos[2] for n in self.glasses()}   # rest height of every glass on the table
        self.extra_z0 = {n: d.body(n).xpos[2] for n in extras}
        self.steps = 0
        return self.get_obs()

    @staticmethod
    def _cell_draw(seed, k):
        """RNG of the k-th per-episode draw of a V2 visibility state (1 alpha, 2 depth holes, 3 thermal shows): streams
        of their own, so no level that does not use them draws anything else than before."""
        return np.random.default_rng(None if seed is None else [seed, FACTORS["cell"], k])

    def _cell_holes(self, seed, cell):
        """Share of the glass's pixels that are depth holes in this episode, or None for the level's glass_holes:
        faint_holes in a cell where depth is faint, show_holes (V2) where it shows the glass; a range is drawn."""
        cfg = self.cfg
        if cell is None:
            return None
        h = cfg.faint_holes if cell[2] == 2 else cfg.show_holes if cell[2] == 0 else None
        if isinstance(h, (list, tuple)):
            h = float(self._cell_draw(seed, 2).uniform(*h))
        return h

    def _transparent(self, seed):
        """Lever A: lower alpha and a slight tint on every transparent object."""
        cfg = self.cfg
        if cfg.alpha is None:
            return dict(alpha=-1.0)
        rng = factor_rng(seed, "transparent")
        alpha = rng.uniform(*cfg.alpha)
        rgb = np.clip(np.array([0.8, 0.9, 1.0]) + rng.uniform(-cfg.tint, cfg.tint, 3), 0, 1)
        for geoms in self.glass_geoms.values():
            self.model.geom_rgba[geoms] = [*rgb, alpha]
        return dict(alpha=alpha)

    def _background(self, seed):
        cfg, m = self.cfg, self.model
        rng = factor_rng(seed, "background")
        name = rng.choice(cfg.pool("background"))
        if name != "xml":
            m.geom_matid[self.table_geom] = m.material("bg_" + name).id
            rgb = rng.uniform(0.15, 0.5, 3) if name == "solid" else np.array(BG_RGBA[name])
            m.geom_rgba[self.table_geom, :3] = np.clip(rgb * (1 + rng.uniform(-1, 1, 3) * cfg.background_jitter), 0, 1)
        return dict(background=BACKGROUNDS.index(name))

    def _cell(self, seed):
        """Visibility cell of the episode (cfg.cells): (index, rgb blind, depth blind, thermal blind, object index),
        or None. The object index is the one the cell was dealt for (seed % number of objects)."""
        cfg = self.cfg
        if not cfg.cells:
            return None
        assert cfg.thermal.realistic_p == 0, "visibility cells set the glass temperature themselves"
        pat = cell_pattern(cfg.cells, len(self.names))
        k = int(factor_rng(None, "cell").integers(len(pat))) if seed is None else seed % len(pat)
        i = int(pat[k])
        flags = tuple(int(x) for x in cfg.cells[i][1:4])   # per sensor: 0 shows, 1 blind, 2 faint
        self.cell2 = tuple(int(x) for x in cfg.cells[i][4:7]) if len(cfg.cells[i]) == 7 else None   # the second glass
        assert all(f in (0, 1, 2) for f in flags) and (flags[0] != 2 or cfg.faint_alpha is not None) and \
            (flags[1] != 2 or cfg.faint_holes is not None) and (flags[2] != 2 or cfg.thermal.faint_contrast is not None), flags
        return (i, *flags, k % len(self.names))

    def _lighting(self, seed, dark=False):
        cfg, m = self.cfg, self.model
        rng = factor_rng(seed, "lighting")
        name = rng.choice(cfg.pool("lighting"))
        if dark:                             # rgb blind cell: the room light is almost off, whatever the pool says
            name = "dark"
        p = LIGHTING[name]
        scale = 1 + rng.uniform(-1, 1) * cfg.light_jitter
        if dark:
            scale *= cfg.dark_light
            m.vis.headlight.specular[:] = self._head_spec * cfg.dark_light
            m.light_specular[0] = self._defaults["light_specular"][0] * cfg.dark_light
        colour = np.array(p.get("colour", (1.0, 1.0, 1.0)))
        if name != "normal" or cfg.light_jitter > 0:
            m.vis.headlight.diffuse[:] = p["head"][0] * scale * colour
            m.vis.headlight.ambient[:] = p["head"][1] * scale * colour
            m.light_diffuse[0] = p["main"] * scale * colour
        if cfg.light_dir_jitter > 0:
            tilt, az = np.radians(rng.uniform(0, cfg.light_dir_jitter)), rng.uniform(0, 2 * np.pi)
            m.light_dir[0] = [np.sin(tilt) * np.cos(az), np.sin(tilt) * np.sin(az), -np.cos(tilt)]
        for e in p["extra"]:
            lid = m.light(e).id
            m.light_active[lid] = 1
            m.light_diffuse[lid] = 0.8 * scale * colour
            if e == "spot":
                m.light_pos[lid] = [rng.uniform(0.3, 0.5), rng.uniform(-0.2, 0.2), 1.0]
        return dict(lighting=LIGHTING_NAMES.index(name), light_scale=scale)

    def _liquid(self, seed, info):
        """Liquid inside the target (if it can hold liquid). Returns (height above the base, temperature) or None."""
        cfg, m = self.cfg, self.model
        rng = factor_rng(seed, "liquid")
        fill, kind = rng.choice(cfg.pool("liquid_fill")), rng.integers(len(LIQUID_KINDS))
        span = cfg.thermal.liquid_temp[LIQUID_KINDS[kind]]
        temp = self.t_env + rng.uniform(-1, 1) if span is None else rng.uniform(*span)
        if self.obj not in LIQUID_OBJECTS or fill <= 0:
            info.update(fill_level=0.0, liquid_kind=-1, liquid_temp=np.nan)
            return None
        h = fill * LIQUID_OBJECTS[self.obj]
        g = m.geom("liquid_" + self.obj).id
        m.geom_size[g, -1] = h / 2           # half height (cylinder size[1], box size[2])
        m.geom_pos[g, 2] = 0.003 + h / 2
        m.geom_rgba[g, 3] = cfg.liquid_alpha
        info.update(fill_level=float(fill), liquid_kind=int(kind), liquid_temp=temp)
        return (0.003 + h, temp)

    def _extras(self, seed, info):
        """Puts distractors, a look-alike (lever B) and absorbers (lever C) on the table with their own temperatures."""
        cfg, m, d, th = self.cfg, self.model, self.data, self.cfg.thermal
        rng = factor_rng(seed, "distractors")
        pool = cfg.pool("distractors")
        n = min(rng.integers(cfg.n_distractors[0], cfg.n_distractors[1] + 1), len(pool))
        names = ["la_" + self.obj] if rng.random() < cfg.lookalike_p else []
        names += list(rng.choice(ABSORBERS, rng.integers(cfg.n_absorbers[0], cfg.n_absorbers[1] + 1), replace=False))
        names += list(rng.choice(pool, n, replace=False))
        xr, yr = (EXTRA_X_NARROW, EXTRA_Y_NARROW) if cfg.camera == "front_cam_narrow" else (EXTRA_X, EXTRA_Y)
        target = d.qpos[self.obj_q[self.obj]:self.obj_q[self.obj] + 2].copy()
        placed = [(target, self.obj_r[self.obj], GAP_TARGET)]
        if self.obj2 is not None:            # task "two": the distractors keep clear of the second glass too
            q2 = self.obj_q[self.obj2]
            placed.append((d.qpos[q2:q2 + 2].copy(), self.obj_r[self.obj2], GAP_TARGET))
        out = []
        for name in names:
            rad = self.extra_r[name]
            for _ in range(200):
                xy = np.array([rng.uniform(*xr), rng.uniform(*yr)])
                if all(np.linalg.norm(xy - p) - rad - pr >= gap for p, pr, gap in placed):
                    break
            else:
                continue    # no room: skip this one (the count actually placed is recorded)
            placed.append((xy, rad, GAP_EXTRA))
            yaw = rng.uniform(0, 2 * np.pi)
            q = self.extra_q[name]
            d.qpos[q:q + 7] = [*xy, REST_Z, np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
            b = m.body(name).id
            m.body_gravcomp[b] = 0
            geoms = np.flatnonzero(m.geom_bodyid == b)
            m.geom_contype[geoms] = m.geom_conaffinity[geoms] = 1
            m.body_contype[b] = m.body_conaffinity[b] = 1   # the broadphase filters bodies by these too
            if not name.startswith("abs_"):
                m.geom_rgba[geoms, :3] = PALETTE[rng.integers(len(PALETTE))]
            ambient = rng.random() < th.distractor_ambient_p
            self.temps[name] = self.t_env + (rng.uniform(-1, 1) if ambient
                                             else rng.choice([-1, 1]) * rng.uniform(*th.distractor_gap))
            self.emis[name] = rng.uniform(*EMISSIVITY.get(name, (0.88, 0.95)))
            out.append(name)
        pad = lambda v: v + [-1] * (MAX_EXTRA - len(v))
        info.update(n_distractors=sum(n in DISTRACTORS for n in out), lookalike=int(any(n in LOOKALIKES for n in out)),
                    n_absorbers=sum(n in ABSORBERS for n in out), extra_ids=pad([EXTRA_NAMES.index(n) for n in out]),
                    extra_temps=pad([self.temps[n] for n in out]), extra_emis=pad([self.emis[n] for n in out]))
        return out

    def _place_target(self, seed, extras):
        """Pick-and-place: puts the coaster on the table. It does not touch the glass or a distractor, and among such
        spots the first of 500 draws is taken that the glass can be carried to in a straight line without passing over
        a distractor (place_clear 1); if none is, the first free spot (place_clear 0); if there is no free spot, the
        draw furthest from everything, overlapping (place_clear -1; about one seed in 3000, none in the test sets).
        Own RNG stream, drawn after everything else, so the rest of the scene is that of the same seed without a
        place target."""
        cfg, m, d = self.cfg, self.model, self.data
        rng = factor_rng(seed, "place")
        self.temps["coaster"] = self.t_env + rng.uniform(-1, 1)
        self.emis["coaster"] = rng.uniform(0.88, 0.95)
        R, r_obj = cfg.place_radius, self.obj_r[self.obj]
        m.geom_size[self.coaster_geoms, 0] = [R, 0.75 * R]
        glass = d.qpos[self.obj_q[self.obj]:self.obj_q[self.obj] + 2].copy()
        others = [(d.qpos[self.extra_q[n]:self.extra_q[n] + 2].copy(), self.extra_r[n]) for n in extras]
        free, best = None, (-np.inf, None)
        for _ in range(500):
            xy = np.array([rng.uniform(*PLACE_X), rng.uniform(*PLACE_Y)])
            if np.hypot(*xy) > PLACE_REACH:
                continue
            room = min([np.linalg.norm(xy - glass) - R - r_obj] + [np.linalg.norm(xy - p) - R - r for p, r in others])
            if room > best[0]:
                best = (room, xy)
            if room < GAP_EXTRA:
                continue
            clear = all(_seg_dist(p, glass, xy) >= r_obj + r + PLACE_CLEAR for p, r in others)
            if clear or free is None:
                free = (xy, int(clear))
            if clear:
                break
        xy, clear = free if free is not None else (best[1], -1)
        m.body_pos[self.coaster] = [*xy, TABLE_TOP]
        self.place_xy = xy
        return dict(place_x=float(xy[0]), place_y=float(xy[1]), place_dist=float(np.linalg.norm(xy - glass)),
                    place_clear=clear, place_temp=self.temps["coaster"], place_emis=self.emis["coaster"])

    def glasses(self):
        """Names of the glasses on the table in this episode (two in task "two")."""
        return [self.obj] if self.obj2 is None else [self.obj, self.obj2]

    def _second_glass(self, seed, base_alpha):
        """Task "two": the colour and thermal states of the second glass (cell2), with RNG streams of their own."""
        cfg, th, f = self.cfg, self.cfg.thermal, self.cell2
        a = base_alpha if f[0] == 0 else cfg.faint_alpha if f[0] == 2 else cfg.clear_alpha
        if isinstance(a, (list, tuple)):
            a = float(self._cell_draw(seed, 4).uniform(*a))
        self.model.geom_rgba[self.glass_geoms[self.obj2], 3] = a
        rc = self._cell_draw(seed, 6)
        self.emis[self.obj2] = rc.uniform(*th.obj_emis)
        c = rc.uniform(-th.blind_contrast, th.blind_contrast) if f[2] == 1 else \
            rc.choice([-1.0, 1.0]) * rc.uniform(*(th.faint_contrast if f[2] == 2 else th.show_contrast))
        self.temps[self.obj2] = matching_temp(self.temps["table"], self.emis["table"], self.emis[self.obj2],
                                              self.t_env, c)
        return dict(object2=GLASS_IDS.index(self.obj2), alpha2=float(a), obj2_temp=self.temps[self.obj2],
                    rgb2=f[0], depth2=f[1], thermal2=f[2], thermal_contrast2=apparent_contrast(
                        self.temps[self.obj2], self.emis[self.obj2], self.temps["table"], self.emis["table"], self.t_env))

    def _second_depth(self, seed):
        """Task "two": the depth state of the second glass for sensors.episode."""
        cfg, f = self.cfg, self.cell2[1]
        h = cfg.faint_holes if f == 2 else cfg.show_holes if f == 0 and cfg.show_holes is not None else cfg.glass_holes
        if isinstance(h, (list, tuple)):
            h = float(self._cell_draw(seed, 5).uniform(*h))
        return dict(geoms=[int(g) for g in self.glass_geoms[self.obj2]], holes=float(h), seethrough=f == 1)

    def _two_targets(self, seed, extras):
        """Task "two": puts the two coasters on the table so that the glasses can be carried to them without passing
        over anything, in an order that can be seen in the scene: the glass nearer to the gripper first, to the
        coaster nearer to that glass if the way there is clear and else to the other; then the second glass to the
        coaster that is left. plan_two = [(glass, coaster index), (glass, coaster index)] is the plan of the scripted
        teacher (a policy may do it in any order). A coaster touches no glass, no distractor and
        not the other coaster; the first of 2000 draws with such a plan is taken (place_clear 1). If there is none,
        the distractors are taken off the table and the coasters drawn again (place_clear 2: every scene of the task
        can be done in the order of the plan); should even that fail, the first free draw (0) or the draw furthest
        from everything (-1)."""
        cfg, m, d = self.cfg, self.model, self.data
        rng = factor_rng(seed, "place")
        R = cfg.place_radius
        for name, geoms in zip(("coaster", "coaster2"), self.coasters_geoms):
            self.temps[name] = self.t_env + rng.uniform(-1, 1)
            self.emis[name] = rng.uniform(0.88, 0.95)
            m.geom_size[geoms, 0] = [R, 0.75 * R]
        names = self.glasses()
        pos = {n: d.qpos[self.obj_q[n]:self.obj_q[n] + 2].copy() for n in names}
        rad = {n: self.obj_r[n] for n in names}
        others = [(d.qpos[self.extra_q[n]:self.extra_q[n] + 2].copy(), self.extra_r[n]) for n in extras]

        def clear(carried, start, goal, blocks):     # the straight path of a carried glass passes over none of them
            return all(_seg_dist(p, start, goal) >= rad[carried] + r + PLACE_CLEAR for p, r in blocks)

        mujoco.mj_kinematics(m, d)                   # the arm is at its start pose; nothing has been stepped yet
        tool = d.body("tool_frame").xpos[:2].copy()
        first, second = sorted(names, key=lambda n: np.linalg.norm(pos[n] - tool))

        def plan(c):
            for i in sorted((0, 1), key=lambda i: np.linalg.norm(c[i] - pos[first])):
                # the first glass, standing on its coaster, is as far from the second as the two are from each other
                # at the start: the open gripper comes down on the second next to it (it knocked it over in 1 of 120)
                if np.linalg.norm(c[i] - pos[second]) - rad[first] - rad[second] < TWO_GAP:
                    continue
                if clear(first, pos[first], c[i], [(pos[second], rad[second])] + others) and \
                        clear(second, pos[second], c[1 - i], [(c[i], rad[first])] + others):
                    return [(first, i), (second, 1 - i)]
            return None

        def search():
            solid = [(pos[n], rad[n]) for n in names] + others
            free, best = None, (-np.inf, None)
            for _ in range(2000):
                c = [np.array([rng.uniform(*PLACE_X), rng.uniform(*PLACE_Y)]) for _ in range(2)]
                if max(np.hypot(*x) for x in c) > PLACE_REACH:
                    continue
                room = min([np.linalg.norm(x - p) - R - r for x in c for p, r in solid]
                           + [np.linalg.norm(c[0] - c[1]) - 2 * R])
                if room > best[0]:
                    best = (room, c)
                if room < GAP_EXTRA:
                    continue
                p = plan(c)
                if p is not None or free is None:
                    free = (c, p)
                if p is not None:
                    break
            c, p = free if free is not None else (best[1], None)
            return c, p, 1 if p is not None else 0 if free is not None else -1

        c, p, ok = search()
        removed = {}
        if ok != 1 and extras:                       # no plan with the distractors where they are: without them
            for name in extras:
                q, b = self.extra_q[name], m.body(name).id
                d.qpos[q:q + 7] = m.qpos0[q:q + 7]
                geoms = np.flatnonzero(m.geom_bodyid == b)
                for field, idx in (("geom_contype", geoms), ("geom_conaffinity", geoms), ("body_contype", b),
                                   ("body_conaffinity", b), ("body_gravcomp", b)):
                    getattr(m, field)[idx] = self._defaults[field][idx]
            extras.clear()
            others.clear()
            c, p, ok = search()
            ok = 2 if ok == 1 else ok
            removed = dict(n_distractors=0, lookalike=0, n_absorbers=0, extra_ids=[-1] * MAX_EXTRA,
                           extra_temps=[-1] * MAX_EXTRA, extra_emis=[-1] * MAX_EXTRA)
        for body, x in zip(self.coasters, c):
            m.body_pos[body] = [*x, TABLE_TOP]
        self.place_xy2, self.place_xy = c, c[0]
        self.plan_two = p if p is not None else [(first, 0), (second, 1)]
        return dict(removed, place_x=float(c[0][0]), place_y=float(c[0][1]), place2_x=float(c[1][0]),
                    place2_y=float(c[1][1]), place_clear=ok, plan_first=GLASS_IDS.index(self.plan_two[0][0]),
                    plan_first_coaster=self.plan_two[0][1])

    def on_coaster(self, name):
        """Task "two": index of the coaster this glass stands on (upright, footprint inside, at rest, let go), or None."""
        d, b = self.data, self.data.body(name)
        dof, tool = self.obj_dof[name], self.data.body("tool_frame").xpos
        if not (abs(b.xpos[2] - (self.z0[name] + COASTER_H)) < PLACE_Z_TOL and b.xmat[8] > PLACE_UPRIGHT
                and np.linalg.norm(d.qvel[dof:dof + 3]) < PLACE_STILL
                and (tool[2] > b.xpos[2] + GRASP_HEIGHT[name] + PLACE_RELEASE
                     or np.linalg.norm(tool[:2] - b.xpos[:2]) > TWO_AWAY)):
            return None
        for i, xy in enumerate(self.place_xy2):
            if np.linalg.norm(b.xpos[:2] - xy) <= self.cfg.place_radius - self.obj_r[name]:
                return i
        return None

    def placed_count(self):
        """Task "two": how many of the two glasses stand on a coaster."""
        return sum(self.on_coaster(n) is not None for n in self.glasses())

    def coaster_top(self, i):
        """Task "two": centre of the top of coaster i."""
        return np.array([*self.place_xy2[i], TABLE_TOP + COASTER_H])

    def _mat_target(self, seed, extras):
        """Task "align": puts the mat on the table at a random angle, as _place_target puts the coaster (free of the
        glass and of the distractors by the circle around the mat; a straight path to it clear of distractors where
        there is one). RNG stream of the place target, then the angle."""
        m, d = self.model, self.data
        rng = factor_rng(seed, "place")
        self.temps["mat"] = self.t_env + rng.uniform(-1, 1)
        self.emis["mat"] = rng.uniform(0.88, 0.95)
        R, r_obj = float(np.hypot(*MAT_HALF)), self.obj_r[self.obj]
        glass = d.qpos[self.obj_q[self.obj]:self.obj_q[self.obj] + 2].copy()
        others = [(d.qpos[self.extra_q[n]:self.extra_q[n] + 2].copy(), self.extra_r[n]) for n in extras]
        free, best = None, (-np.inf, None)
        for _ in range(500):
            xy = np.array([rng.uniform(*PLACE_X), rng.uniform(*PLACE_Y)])
            if np.hypot(*xy) > PLACE_REACH:
                continue
            room = min([np.linalg.norm(xy - glass) - R - r_obj] + [np.linalg.norm(xy - p) - R - r for p, r in others])
            if room > best[0]:
                best = (room, xy)
            if room < GAP_EXTRA:
                continue
            clear = all(_seg_dist(p, glass, xy) >= r_obj + r + PLACE_CLEAR for p, r in others)
            if clear or free is None:
                free = (xy, int(clear))
            if clear:
                break
        xy, clear = free if free is not None else (best[1], -1)
        self.place_yaw = float(rng.uniform(0, np.pi))
        m.body_pos[self.coaster] = [*xy, TABLE_TOP]
        m.body_quat[self.coaster] = [np.cos(self.place_yaw / 2), 0, 0, np.sin(self.place_yaw / 2)]
        self.place_xy = xy
        return dict(place_x=float(xy[0]), place_y=float(xy[1]), place_dist=float(np.linalg.norm(xy - glass)),
                    place_clear=clear, place_temp=self.temps["mat"], place_emis=self.emis["mat"],
                    obj_yaw=float(np.degrees(self.obj_yaw)), place_yaw=float(np.degrees(self.place_yaw)))

    def object_yaw(self):
        """Angle (rad) of the long side of the glass in the table plane now (task "align")."""
        x = self.data.body(self.obj).xmat.reshape(3, 3)[:, 0]
        return float(np.arctan2(x[1], x[0]))

    yaw_tolerance = ALIGN_YAW_TOL

    def yaw_error(self):
        """Degrees between the long side of the glass and that of the mat, either way round (0 to 90)."""
        e = np.degrees(self.object_yaw() - self.place_yaw) % 180.0
        return float(min(e, 180.0 - e))

    def step(self, action):
        self.data.ctrl[:7] = action
        for _ in range(self.n_sub):
            mujoco.mj_step(self.model, self.data)
        self.steps += 1
        done = self.success() or self.steps >= self.max_steps
        return self.get_obs(), done

    def success(self):
        if self.place:
            return self.placed()
        return self.data.body(self.obj).xpos[2] - self.obj_z0 > 0.10   # lifted 10 cm

    def place_offset(self):
        """Horizontal distance (m) between the glass and the centre of the coaster."""
        return float(np.linalg.norm(self.data.body(self.obj).xpos[:2] - self.place_xy))

    def placed(self):
        """Pick-and-place success: the glass stands upright on the coaster with its footprint inside it, at rest, and
        the gripper has let go of it (the tool is PLACE_RELEASE above the glass's grasp point)."""
        if self.two:     # T3: each of the two glasses on a coaster of its own
            return self.placed_count() == 2
        d, b = self.data, self.data.body(self.obj)
        dof = self.obj_dof[self.obj]
        if self.align:   # T2: on the mat, its centre and its direction within the tolerances, upright, at rest, let go
            return bool(self.place_offset() <= ALIGN_POS_TOL and self.yaw_error() <= ALIGN_YAW_TOL
                        and abs(b.xpos[2] - (self.obj_z0 + COASTER_H)) < PLACE_Z_TOL
                        and b.xmat[8] > PLACE_UPRIGHT
                        and np.linalg.norm(d.qvel[dof:dof + 3]) < PLACE_STILL
                        and d.body("tool_frame").xpos[2] > b.xpos[2] + GRASP_HEIGHT[self.obj] + PLACE_RELEASE)
        return bool(self.place_offset() <= self.cfg.place_radius - self.obj_r[self.obj]
                    and abs(b.xpos[2] - (self.obj_z0 + COASTER_H)) < PLACE_Z_TOL
                    and b.xmat[8] > PLACE_UPRIGHT
                    and np.linalg.norm(d.qvel[dof:dof + 3]) < PLACE_STILL
                    and d.body("tool_frame").xpos[2] > b.xpos[2] + GRASP_HEIGHT[self.obj] + PLACE_RELEASE)

    def lifted_extras(self, height=0.05):
        """Distractors lifted more than `height` (for the failure breakdown)."""
        return [n for n, z0 in self.extra_z0.items() if self.data.body(n).xpos[2] - z0 > height]

    @property
    def obj_id(self):
        return GLASS_IDS.index(self.obj)

    def get_obs(self):
        th = self.cfg.thermal
        rgb, depth, thermal, raw = self.sensors.render(self.data, self.temps, self.emis, self.t_env, self.rng,
                                                       self.sensor_scene)
        depth8 = (255 * np.clip(depth, 0, 2.0) / 2.0).astype(np.uint8)
        raw96 = _resize_raw(raw)
        if th.pipeline == "v0" and th.normalize == "legacy":
            thermal96 = _resize((255 * thermal).astype(np.uint8))   # main's path, bit for bit
        else:
            mode = "minmax" if th.normalize == "legacy" else th.normalize
            params = {"range": tuple(th.norm_range)} if th.norm_range is not None else None
            thermal96 = (255 * normalize_thermal_raw(raw96, mode, params)).astype(np.uint8)
        rgb96 = _resize(rgb)
        sigma = self.cfg.dark_rgb_noise if self.dark else self.cfg.rgb_noise
        if sigma > 0:
            noise = self.sensor_scene["rng"].normal(0, sigma, rgb96.shape)
            rgb96 = np.clip(np.round(rgb96 + noise), 0, 255).astype(np.uint8)
        d = self.data
        joints = np.append(d.qpos[:6], d.joint("right_finger_bottom_joint").qpos[0]).astype(np.float32)
        return {"rgb": rgb96,
                "depth": _resize(depth8)[..., None],
                "thermal": thermal96[..., None],
                "thermal_raw": raw96[..., None],
                "joints": joints}

    def close(self):
        self.sensors.r.close()

    # ONLY FOR TEACHER (NOT FOR POLICY)
    def object_position(self):
        return self.data.body(self.obj).xpos.copy()

    def grasp_height(self, name=None):
        return GRASP_HEIGHT[name or self.obj]

    def tool_position(self):
        return self.data.body("tool_frame").xpos.copy()

    def place_position(self):
        """Centre of the coaster's top (pick-and-place levels)."""
        return np.array([*self.place_xy, TABLE_TOP + COASTER_H])
