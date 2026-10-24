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

def _resize(img):
    return np.asarray(Image.fromarray(img).resize((IMG, IMG), Image.BILINEAR))

class GlassPickEnv:
    def __init__(self, max_steps=120):
        self.model = mujoco.MjModel.from_xml_path(SCENE)
        self.data = mujoco.MjData(self.model)
        self.sensors = Sensors(self.model)
        self.n_sub = int(round(CTRL_DT / self.model.opt.timestep))
        self.obj_q = {n: self.model.jnt_qposadr[self.model.body(n).jntadr[0]] for n in OBJECT_NAMES}
        self.link_names = [self.model.body(i).name for i in range(self.model.nbody) if "link" in self.model.body(i).name]
        self.max_steps = max_steps
        self.obj = OBJECT_NAMES[0]

    def reset(self, seed=None, obj=None):
        """obj: object name to put on the table; None picks one at random."""
        self.rng = np.random.default_rng(seed)
        r, m, d = self.rng, self.model, self.data
        mujoco.mj_resetData(m, d)
        d.qpos[:6] = READY
        d.ctrl[:6] = READY

        # One object upright on the table, the others parked out of view
        self.obj = obj if obj is not None else OBJECT_NAMES[r.integers(len(OBJECT_NAMES))]
        for name, park in zip(OBJECT_NAMES, PARK):
            q = self.obj_q[name]
            pos = [r.uniform(*SPAWN_X), r.uniform(*SPAWN_Y), REST_Z] if name == self.obj else park
            d.qpos[q:q + 7] = [*pos, 1, 0, 0, 0]

        # Thermal conditions
        self.t_env = r.uniform(18, 26)
        gap = r.uniform(-1, 1) if r.random() < 0.3 else r.choice([-1, 1]) * r.uniform(5, 35)
        self.temps = {self.obj: self.t_env + gap, "table": self.t_env + r.uniform(-1, 1)}
        self.temps.update({n: self.t_env + r.uniform(5, 12) for n in self.link_names})
        self.emis = {self.obj: r.uniform(0.85, 0.95), "table": r.uniform(0.90, 0.97)}

        for _ in range(300):
            mujoco.mj_step(m, d)
        self.obj_z0 = d.body(self.obj).xpos[2]
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
        return self.data.body(self.obj).xpos[2] - self.obj_z0 > 0.10   # lifted 10 cm

    @property
    def obj_id(self):
        return OBJECT_NAMES.index(self.obj)

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
    def object_position(self):
        return self.data.body(self.obj).xpos.copy()

    def grasp_height(self):
        return OBJECTS[self.obj]

    def tool_position(self):
        return self.data.body("tool_frame").xpos.copy()
