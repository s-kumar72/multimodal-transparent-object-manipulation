"""
Scripted teacher for diffusion policy: knows the object position and plans a top-down pick using mink IK.
Author: Sadhana Kumar
"""
import mujoco
import mink
import numpy as np

from env import READY

ARM = "sim/assets/gen3_lite/gen3_lite.xml"
DOWN = np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]], float)
OPEN, CLOSED = 0.0, 0.85
LIFT = 0.16 # pre-grasp / lift height above the grasp
PRE_Z_MAX = 0.30 # the top-down IK cannot reach higher at the far edge of the spawn region, so tall objects lift less

class ScriptedTeacher:
    def __init__(self):
        self.ik_model = mujoco.MjModel.from_xml_path(ARM)
        self.config = mink.Configuration(self.ik_model)
        self.task = mink.FrameTask("tool_frame", "body", position_cost=1.0, orientation_cost=0.5)
        self.posture = mink.PostureTask(self.ik_model, cost=1e-2)
        self.posture.set_target(self._full(READY))
        self.limits = [mink.ConfigurationLimit(self.ik_model)]

    def _full(self, arm_q):
        return np.concatenate([arm_q, np.zeros(self.ik_model.nq - 6)])

    def reset(self, env, rng=None):
        rng = rng or np.random.default_rng()
        self.config.update(self._full(env.data.qpos[:6]))
        obj, tool = env.object_position(), env.tool_position()
        grasp = obj + [0, 0, env.grasp_height()]
        pre = grasp + [0, 0, min(LIFT, PRE_Z_MAX - grasp[2])]
        plan = []

        def segment(a, b, n, grip):
            plan.extend((a + (b - a) * k / n, grip) for k in range(1, n+1))

        segment(tool, pre, int(rng.integers(20, 30)), OPEN)
        segment(pre, grasp, int(rng.integers(12, 18)), OPEN)
        segment(grasp, grasp, 8, CLOSED)
        segment(grasp, pre, 15, CLOSED)
        self.plan, self.t = plan, 0

    def done(self):
        return self.t >= len(self.plan)

    def act(self):
        target, grip = self.plan[self.t]
        self.t += 1
        self.task.set_target(mink.SE3.from_rotation_and_translation(mink.SO3.from_matrix(DOWN), target))
        for _ in range(30):
            vel = mink.solve_ik(self.config, [self.task, self.posture], 0.05, "daqp", 1e-3, limits=self.limits)
            self.config.integrate_inplace(vel, 0.05)

        return np.append(self.config.q[:6], grip).astype(np.float32)