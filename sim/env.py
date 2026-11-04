"""
Glass-picking environment with randomized scene.
The scene is set by a config (sim/config.py, sim/configs/*.yaml); the default config reproduces main exactly.
Author: Sadhana Kumar
"""
import mujoco
import numpy as np
from PIL import Image

from common.thermal import normalize_thermal_raw
from config import SceneConfig, factor_rng, load_config
from sensors import Sensors

SCENE = "sim/assets/gen3_lite/transparent_scene.xml"
SCENE_DISTRACTORS = "sim/assets/gen3_lite/distractor_scene.xml" # SCENE + parked distractors (only when used)

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

def _resize(img):
    return np.asarray(Image.fromarray(img).resize((IMG, IMG), Image.BILINEAR))

def _resize_raw(raw):
    """Raw counts (float) -> 96x96 uint16, bilinear like the 8-bit images."""
    small = np.asarray(Image.fromarray(raw.astype(np.float32), mode="F").resize((IMG, IMG), Image.BILINEAR))
    return np.clip(np.round(small), 0, 16383).astype(np.uint16)

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
        self.model = mujoco.MjModel.from_xml_path(SCENE_DISTRACTORS if self.uses_extras else SCENE)
        self.data = mujoco.MjData(self.model)
        self.sensors = Sensors(self.model, camera=cfg.camera, config=cfg)
        self.n_sub = int(round(CTRL_DT / self.model.opt.timestep))
        self.obj_q = {n: self.model.jnt_qposadr[self.model.body(n).jntadr[0]] for n in OBJECT_NAMES}
        self.link_names = [self.model.body(i).name for i in range(self.model.nbody) if "link" in self.model.body(i).name]
        self.max_steps = max_steps
        self.obj = OBJECT_NAMES[0]
        m = self.model
        self.extra_q = {n: m.jnt_qposadr[m.body(n).jntadr[0]] for n in EXTRA_NAMES} if self.uses_extras else {}
        self.extra_r = {n: _footprint(m, m.body(n).id) for n in self.extra_q}
        self.obj_r = {n: _footprint(m, m.body(n).id) for n in OBJECT_NAMES}
        self.table_geom = int(np.flatnonzero(m.geom_bodyid == m.body("table").id)[0])
        self.glass_geoms = {n: [g for g in np.flatnonzero(m.geom_bodyid == m.body(n).id)
                                if not m.geom(g).name.startswith("liquid_")] for n in OBJECT_NAMES}
        # Model fields the factors change; restored on every reset
        self._fields = ["geom_rgba", "geom_matid", "geom_size", "geom_pos", "geom_contype", "geom_conaffinity",
                        "body_contype", "body_conaffinity", "body_gravcomp", "light_active", "light_diffuse",
                        "light_dir", "light_pos"]
        self._defaults = {f: getattr(m, f).copy() for f in self._fields}
        self._head = (m.vis.headlight.diffuse.copy(), m.vis.headlight.ambient.copy())

    def reset(self, seed=None, obj=None):
        """obj: object name to put on the table; None picks one at random."""
        self.rng = np.random.default_rng(seed)
        r, m, d, cfg, th = self.rng, self.model, self.data, self.cfg, self.cfg.thermal
        for f, v in self._defaults.items():
            getattr(m, f)[:] = v
        m.vis.headlight.diffuse[:], m.vis.headlight.ambient[:] = self._head
        mujoco.mj_resetData(m, d)
        d.qpos[:6] = READY
        d.ctrl[:6] = READY

        # One object upright on the table, the others parked out of view (legacy RNG, main's order)
        self.obj = obj if obj is not None else OBJECT_NAMES[r.integers(len(OBJECT_NAMES))]
        for name, park in zip(OBJECT_NAMES, PARK):
            q = self.obj_q[name]
            pos = [r.uniform(*SPAWN_X), r.uniform(*SPAWN_Y), REST_Z] if name == self.obj else park
            d.qpos[q:q + 7] = [*pos, 1, 0, 0, 0]

        # Thermal conditions (legacy RNG, main's order)
        self.t_env = r.uniform(*th.t_env)
        crossover = r.random() < th.crossover_p
        gap = r.uniform(*th.crossover_gap) if crossover else r.choice([-1, 1]) * r.uniform(*th.gap)
        self.temps = {self.obj: self.t_env + gap, "table": self.t_env + r.uniform(*th.table_dt)}
        self.temps.update({n: self.t_env + r.uniform(*th.link_dt) for n in self.link_names})
        self.emis = {self.obj: r.uniform(*th.obj_emis), "table": r.uniform(*th.table_emis)}

        # New factors: each has its own RNG stream, so turning one on does not change the others
        rt = factor_rng(seed, "thermal")
        realistic, realistic_gap = rt.random() < th.realistic_p, rt.uniform(*th.realistic_gap)
        if realistic:   # measured on real glass: slightly below ambient
            gap, crossover = realistic_gap, False
            self.temps[self.obj] = self.t_env + gap
        info = dict(split=int(cfg.split == "test"), seed=-1 if seed is None else seed, level=cfg.level,
                    object=OBJECT_NAMES.index(self.obj), t_env=self.t_env, obj_temp=self.temps[self.obj],
                    thermal_gap=gap, is_crossover=int(crossover), realistic=int(realistic),
                    obj_emis=self.emis[self.obj])
        info.update(self._transparent(seed))
        info.update(self._background(seed))
        info.update(self._lighting(seed))
        liquid = self._liquid(seed, info)
        extras = self._extras(seed, info) if self.uses_extras else []
        self.sensor_scene = self.sensors.episode(seed, self.obj, liquid, extras, d)
        info["n_table_blobs"] = len(self.sensor_scene["blobs"])
        self.episode_info = info

        for _ in range(300):
            mujoco.mj_step(m, d)
        self.obj_z0 = d.body(self.obj).xpos[2]
        self.extra_z0 = {n: d.body(n).xpos[2] for n in extras}
        self.steps = 0
        return self.get_obs()

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

    def _lighting(self, seed):
        cfg, m = self.cfg, self.model
        rng = factor_rng(seed, "lighting")
        name = rng.choice(cfg.pool("lighting"))
        p = LIGHTING[name]
        scale = 1 + rng.uniform(-1, 1) * cfg.light_jitter
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

    def step(self, action):
        self.data.ctrl[:7] = action
        for _ in range(self.n_sub):
            mujoco.mj_step(self.model, self.data)
        self.steps += 1
        done = self.success() or self.steps >= self.max_steps
        return self.get_obs(), done

    def success(self):
        return self.data.body(self.obj).xpos[2] - self.obj_z0 > 0.10   # lifted 10 cm

    def lifted_extras(self, height=0.05):
        """Distractors lifted more than `height` (for the failure breakdown)."""
        return [n for n, z0 in self.extra_z0.items() if self.data.body(n).xpos[2] - z0 > height]

    @property
    def obj_id(self):
        return OBJECT_NAMES.index(self.obj)

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
        d = self.data
        joints = np.append(d.qpos[:6], d.joint("right_finger_bottom_joint").qpos[0]).astype(np.float32)
        return {"rgb": _resize(rgb),
                "depth": _resize(depth8)[..., None],
                "thermal": thermal96[..., None],
                "thermal_raw": raw96[..., None],
                "joints": joints}

    def close(self):
        self.sensors.r.close()

    # ONLY FOR TEACHER (NOT FOR POLICY)
    def object_position(self):
        return self.data.body(self.obj).xpos.copy()

    def grasp_height(self):
        return OBJECTS[self.obj]

    def tool_position(self):
        return self.data.body("tool_frame").xpos.copy()
