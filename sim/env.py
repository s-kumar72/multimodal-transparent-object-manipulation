"""
Glass-picking environment with randomized scene.
Scene randomization (background, lighting, liquid, distractors) is set by a scene config (sim/config.py) and a split.
Author: Sadhana Kumar
"""
import mujoco
import numpy as np
from PIL import Image

from config import BACKGROUNDS, LIGHTING, load_scene_config
from sensors import Sensors

SCENE = "sim/assets/gen3_lite/transparent_scene.xml"

# This position is arm above table, gripper down
READY = np.array([-0.118, -0.575, 0.968, 1.571, 1.599, -1.689])
IMG = 96 # image size for Diffusion Policy
CTRL_DT = 0.1 # 10 Hz, same as Diffusion Policy
GLASS_Z = 0.072 # table top (0.02) + glass half height (0.05) + 2 mm, so the glass starts upright
TABLE_TOP = 0.02
TABLE_X, TABLE_Y = (0.22, 0.68), (-0.33, 0.33) # table area where distractors may stand
COLORS = np.array([[0.8, 0.2, 0.2], [0.2, 0.3, 0.8], [0.2, 0.6, 0.3], [0.9, 0.8, 0.2], [0.9, 0.5, 0.1],
                   [0.5, 0.2, 0.6], [0.9, 0.9, 0.88], [0.15, 0.15, 0.15], [0.55, 0.55, 0.6]])

def _resize(img):
    return np.asarray(Image.fromarray(img).resize((IMG, IMG), Image.BILINEAR))

def _resize_raw(raw):
    """uint16 raw thermal counts -> IMG x IMG with the same bilinear filter, kept as counts."""
    small = np.asarray(Image.fromarray(raw.astype(np.float32)).resize((IMG, IMG), Image.BILINEAR))
    return np.round(small).astype(np.uint16)

def _footprint(m, body):
    """Radius in xy and height of the body origin above its lowest point (box / cylinder geoms)."""
    radius, low = 0.0, 0.0
    for g in np.flatnonzero(m.geom_bodyid == body):
        s, p = m.geom_size[g], m.geom_pos[g]
        box = m.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX
        radius = max(radius, np.hypot(p[0], p[1]) + (np.hypot(s[0], s[1]) if box else s[0]))
        low = max(low, (s[2] if box else s[1]) - p[2])
    return radius, low

class GlassPickEnv:
    def __init__(self, max_steps=120, config=None, split="train", render=True):
        assert split in ("train", "test")
        self.cfg = config if isinstance(config, dict) else load_scene_config(config)
        self.split = split
        self.model = mujoco.MjModel.from_xml_path(SCENE)
        self.data = mujoco.MjData(self.model)
        self.sensors = Sensors(self.model) if render else None   # render=False: physics only (teacher checks)
        self.n_sub = int(round(CTRL_DT / self.model.opt.timestep))
        self.glass_q = self.model.jnt_qposadr[self.model.body("glass").jntadr[0]]
        self.link_names = [self.model.body(i).name for i in range(self.model.nbody) if "link" in self.model.body(i).name]
        self.max_steps = max_steps

        m = self.model
        self.table, self.floor = m.geom("table").id, m.geom("floor").id
        self.glass, self.liquid_geom = m.geom("glass").id, m.geom("liquid").id
        self.glass_half, self.glass_r = m.geom_size[self.glass, 1], m.geom_size[self.glass, 0]
        self.lights = {n: m.light(n).id for n in ["key", "back", "spot"]}
        self.rgba0, self.key_dir0 = m.geom_rgba.copy(), m.light_dir[self.lights["key"]].copy()
        self.dist = {}
        for b in range(m.nbody):
            if m.body(b).name.startswith("dist_"):
                radius, low = _footprint(m, b)
                geoms = np.flatnonzero(m.geom_bodyid == b)
                self.dist[m.body(b).name[5:]] = dict(body=m.body(b).name, id=b, q=m.jnt_qposadr[m.body_jntadr[b]], geoms=geoms,
                                                    contype=m.geom_contype[geoms].copy(), conaffinity=m.geom_conaffinity[geoms].copy(),
                                                    radius=radius, low=low)

    def reset(self, seed=None):
        self.rng = np.random.default_rng(seed)
        r, m, d = self.rng, self.model, self.data
        mujoco.mj_resetData(m, d)
        d.qpos[:6] = READY
        d.ctrl[:6] = READY
        d.qpos[self.glass_q:self.glass_q + 3] = [r.uniform(0.32, 0.48), r.uniform(-0.15, 0.15), GLASS_Z]

        # Thermal conditions
        self.t_env = r.uniform(18, 26)
        gap = r.uniform(-1, 1) if r.random() < 0.3 else r.choice([-1, 1]) * r.uniform(5, 35)
        self.temps = {"glass": self.t_env + gap, "table": self.t_env + r.uniform(-1, 1)}
        self.temps.update({n: self.t_env + r.uniform(5, 12) for n in self.link_names})
        self.emis = {"glass": r.uniform(0.85, 0.95), "table": r.uniform(0.90, 0.97)}

        # Scene randomization: one RNG stream per factor, so the draws above and the other factors
        # stay the same when a factor is changed in the config
        def stream(k):
            return np.random.default_rng(None if seed is None else [seed, k])
        self.cond = {"split": self.split}
        self._background(stream(1))
        self._lighting(stream(2))
        self._liquid(stream(3))
        self._distractors(stream(4))
        self._glass_appearance(stream(5))

        for _ in range(300):
            mujoco.mj_step(m, d)
        self.glass_z0 = d.body("glass").xpos[2]
        self.steps = 0
        return self.get_obs()

    def _background(self, rng):
        cfg, m = self.cfg["background"], self.model
        name = str(rng.choice(cfg[self.split]))
        mat = m.material(f"bg_{name}").id
        m.geom_matid[self.table] = mat
        m.geom_rgba[self.table] = np.append(np.clip(m.mat_rgba[mat, :3] * (1 + rng.uniform(-1, 1, 3) * cfg["color_jitter"]), 0, 1), 1)
        if BACKGROUNDS[name] is not None:
            self.emis["table"] = rng.uniform(*BACKGROUNDS[name])
        floor = str(rng.choice(cfg[self.split])) if cfg["randomize_floor"] else None
        m.geom_matid[self.floor] = m.material(f"bg_{floor}").id if floor else -1
        m.geom_rgba[self.floor] = m.mat_rgba[m.geom_matid[self.floor]] if floor else self.rgba0[self.floor]
        self.cond.update(background=name, floor=floor)

    def _lighting(self, rng):
        cfg, m, ids = self.cfg["lighting"], self.model, self.lights
        name = str(rng.choice(cfg[self.split]))
        p, j = LIGHTING[name], cfg["jitter"]
        scale = lambda: 1 + rng.uniform(-1, 1) * j["diffuse"]
        key_dir = self.key_dir0 + rng.uniform(-1, 1, 3) * np.tan(np.radians(j["dir_deg"]))
        key_dir[2] = min(key_dir[2], -0.3)    # keep the key light pointing down
        m.light_dir[ids["key"]] = key_dir / np.linalg.norm(key_dir)
        m.light_diffuse[ids["key"]] = p["key"] * scale()
        m.light_ambient[ids["key"]] = p["key_ambient"]
        for n in ["back", "spot"]:
            m.light_active[ids[n]] = p[n] > 0
            m.light_diffuse[ids[n]] = p[n] * scale()
        m.vis.headlight.diffuse[:] = p["head"] * (1 + rng.uniform(-1, 1) * j["headlight"])
        m.vis.headlight.ambient[:] = np.clip(p["head_ambient"] + rng.uniform(-1, 1) * j["ambient"], 0, 1)
        self.cond.update(lighting=name)

    def _liquid(self, rng):
        """Liquid inside the glass (no fluid sim: a "glass of water" to pick). Thermal blends it into the glass wall."""
        cfg, m, g = self.cfg["liquid"], self.model, self.liquid_geom
        fill = float(rng.choice(cfg["fill_levels"]))
        kind = str(rng.choice(list(cfg["temp"])))
        lo, hi = cfg["temp"][kind]
        temp = self.t_env + rng.uniform(lo, hi) if kind == "room" else rng.uniform(lo, hi)
        half = max(fill * self.glass_half, 1e-3)
        m.geom_size[g, 1] = half
        m.geom_pos[g, 2] = -self.glass_half + half
        m.geom_rgba[g, 3] = rng.uniform(*cfg["alpha"]) if fill > 0 else 0.0
        self.liquid = (fill, temp) if fill > 0 else None
        self.cond.update(fill_level=fill, liquid_kind=kind if fill > 0 else None, liquid_temp=temp if fill > 0 else np.nan)

    def _distractors(self, rng):
        """0..max_count opaque objects on the table, clear of the glass so the teacher's grasp is not blocked."""
        cfg, m, d = self.cfg["distractors"], self.model, self.data
        for o in self.dist.values():    # unused ones stay parked: no contacts and floating, so physics is as without them
            m.geom_contype[o["geoms"]] = m.geom_conaffinity[o["geoms"]] = 0
            m.body_gravcomp[o["id"]] = 1
        pool = cfg[self.split]
        n = min(int(rng.integers(0, cfg["max_count"] + 1)), len(pool))
        gx, gy = d.qpos[self.glass_q:self.glass_q + 2]
        placed, names = [], []
        for name in rng.choice(pool, size=n, replace=False) if n else []:
            o = self.dist[name]
            for _ in range(200):
                x, y = rng.uniform(TABLE_X[0] + o["radius"], TABLE_X[1] - o["radius"]), rng.uniform(TABLE_Y[0] + o["radius"], TABLE_Y[1] - o["radius"])
                if np.hypot(x - gx, y - gy) - o["radius"] - self.glass_r < cfg["min_dist_glass"]:
                    continue
                if all(np.hypot(x - px, y - py) - o["radius"] - pr >= cfg["min_dist_each"] for px, py, pr in placed):
                    break
            else:
                continue    # no room left on the table
            yaw = rng.uniform(0, 2 * np.pi)
            d.qpos[o["q"]:o["q"] + 7] = [x, y, TABLE_TOP + o["low"] + 0.002, np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
            m.geom_contype[o["geoms"]], m.geom_conaffinity[o["geoms"]] = o["contype"], o["conaffinity"]
            m.body_gravcomp[o["id"]] = 0
            placed.append((x, y, o["radius"]))
            names.append(str(name))
            color = COLORS[rng.integers(len(COLORS))]
            for g in o["geoms"]:
                m.geom_rgba[g, :3] = color * (0.6 if m.geom(g).name.endswith("_inner") else 1.0)

            # Thermal: mostly near t_env, sometimes hot or cold, so "the only hot/cold object" is not always the glass
            if rng.random() < cfg["p_off_ambient"]:
                self.temps[o["body"]] = rng.uniform(*cfg["hot"] if rng.random() < 0.5 else cfg["cold"])
            else:
                self.temps[o["body"]] = self.t_env + rng.uniform(*cfg["ambient"])
            self.emis[o["body"]] = rng.uniform(*cfg["emis"])
        self.cond.update(distractors=names)

    def _glass_appearance(self, rng):
        """Optional: vary glass alpha and tint, a little stand-in for the refraction MuJoCo does not render."""
        cfg, m = self.cfg["glass_appearance"], self.model
        m.geom_rgba[self.glass] = self.rgba0[self.glass]
        if cfg["enabled"]:
            m.geom_rgba[self.glass, :3] = np.clip(m.geom_rgba[self.glass, :3] + rng.uniform(-1, 1, 3) * cfg["tint"], 0, 1)
            m.geom_rgba[self.glass, 3] = rng.uniform(*cfg["alpha"])

    def step(self, action):
        self.data.ctrl[:7] = action
        for _ in range(self.n_sub):
            mujoco.mj_step(self.model, self.data)
        self.steps += 1
        done = self.success() or self.steps >= self.max_steps
        return self.get_obs(), done

    def success(self):
        return self.data.body("glass").xpos[2] - self.glass_z0 > 0.10   # lifted 10 cm

    def get_obs(self):
        d = self.data
        joints = np.append(d.qpos[:6], d.joint("right_finger_bottom_joint").qpos[0]).astype(np.float32)
        if self.sensors is None:
            return {"joints": joints}
        rgb, depth, thermal, raw = self.sensors.render(d, self.temps, self.emis, self.t_env, self.rng, self.liquid)
        depth8 = (255 * np.clip(depth, 0, 2.0) / 2.0).astype(np.uint8)
        thermal8 = (255 * thermal).astype(np.uint8)
        return {"rgb": _resize(rgb),
                "depth": _resize(depth8)[..., None],
                "thermal": _resize(thermal8)[..., None],
                "thermal_raw": _resize_raw(raw)[..., None],
                "joints": joints}

    def close(self):
        if self.sensors is not None:
            self.sensors.r.close()

    # ONLY FOR TEACHER (NOT FOR POLICY)
    def glass_position(self):
        return self.data.body("glass").xpos.copy()

    def tool_position(self):
        return self.data.body("tool_frame").xpos.copy()
