"""
Glass-picking environment with randomized scene.
Author: Sadhana Kumar
"""
import mujoco
import numpy as np
from PIL import Image

from sensors import Sensors

SCENE = "sim/assets/gen3_lite/transparent_scene.xml"

# This position is arm above table, gripper down
READY = np.array([-0.118, -0.575, 0.968, 1.571, 1.599, -1.689])
IMG = 96 # image size for Diffusion Policy
CTRL_DT = 0.1 # 10 Hz, same as Diffusion Policy
GLASS_Z = 0.072 # table top (0.02) + glass half height (0.05) + 2 mm, so the glass starts upright

def _resize(img):
    return np.asarray(Image.fromarray(img).resize((IMG, IMG), Image.BILINEAR))

class GlassPickEnv:
    def __init__(self, max_steps=120):
        self.model = mujoco.MjModel.from_xml_path(SCENE)
        self.data = mujoco.MjData(self.model)
        self.sensors = Sensors(self.model)
        self.n_sub = int(round(CTRL_DT / self.model.opt.timestep))
        self.glass_q = self.model.jnt_qposadr[self.model.body("glass").jntadr[0]]
        self.link_names = [self.model.body(i).name for i in range(self.model.nbody) if "link" in self.model.body(i).name]
        self.max_steps = max_steps

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

        for _ in range(300):
            mujoco.mj_step(m, d)
        self.glass_z0 = d.body("glass").xpos[2]
        self.steps = 0
        return self.get_obs()

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
        rgb, depth, thermal = self.sensors.render(self.data, self.temps, self.emis, self.t_env, self.rng)
        depth8 = (255 * np.clip(depth, 0, 2.0) / 2.0).astype(np.uint8)
        thermal8 = (255 * thermal).astype(np.uint8)
        d = self.data
        joints = np.append(d.qpos[:6], d.joint("right_finger_bottom_joint").qpos[0]).astype(np.float32)
        return {"rgb": _resize(rgb),
                "depth": _resize(depth8)[..., None],
                "thermal": _resize(thermal8)[..., None],
                "joints": joints}

    def close(self):
        self.sensors.r.close()

    # ONLY FOR TEACHER (NOT FOR POLICY)
    def glass_position(self):
        return self.data.body("glass").xpos.copy()

    def tool_position(self):
        return self.data.body("tool_frame").xpos.copy()
        