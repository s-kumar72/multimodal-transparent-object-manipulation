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
CARRY_STEP = 0.01 # pick-and-place: the glass is carried about this far (m) per control step
PLACE_DROP = 0.003 # pick-and-place: the gripper opens with the base of the glass this far (m) above the coaster
PRE_Z_MAX = 0.30 # the top-down IK cannot reach higher at the far edge of the spawn region, so tall objects lift less
# Task "align": the gripper closes along the world y axis in the DOWN pose, so it grasps an elongated glass whose long
# side is at angle a (from the world x axis) with the tool turned by a about the vertical. The gripper is the same
# turned by 180 deg, and the last joint has room in one direction only, so the angle is taken from this half turn.
YAW_LOW = np.radians(-170.0)
YAW_STEP = np.radians(6.0) # the tool turns at most this much per control step

def tool_yaw(a):
    """The angle of the half turn [YAW_LOW, YAW_LOW + 180 deg) that grasps a long side at angle a."""
    return YAW_LOW + (a - YAW_LOW) % np.pi

def turned(yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]) @ DOWN

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

        if getattr(env, "two", False):       # T3: pick and place twice, in the order the scene was made for
            self._plan_two(env, rng, tool)
            return
        if getattr(env, "align", False):     # T2: the same moves with the tool turned to the glass, then to the mat
            self._plan_align(env, rng, obj, tool, grasp, pre)
            return

        segment(tool, pre, int(rng.integers(20, 30)), OPEN)
        segment(pre, grasp, int(rng.integers(12, 18)), OPEN)
        segment(grasp, grasp, 8, CLOSED)
        segment(grasp, pre, 15, CLOSED)
        if getattr(env, "place", False):   # pick-and-place: carry to the coaster, put down, let go, back up
            top = env.place_position()
            put = np.array([top[0], top[1], grasp[2] + (top[2] - obj[2]) + PLACE_DROP])
            over = np.array([top[0], top[1], pre[2]])
            n = int(np.ceil(np.linalg.norm(over - pre) / CARRY_STEP)) + int(rng.integers(0, 6))
            segment(pre, over, n, CLOSED)
            segment(over, put, int(rng.integers(12, 18)), CLOSED)
            segment(put, put, 6, OPEN)
            segment(put, over, 12, OPEN)
        self.plan, self.t = plan, 0

    def _plan_two(self, env, rng, tool):
        """Task "two": for each (glass, coaster) of env.plan_two the moves of the pick-and-place, one after the other."""
        plan, at = [], tool

        def segment(a, b, n, grip):
            plan.extend((a + (b - a) * k / n, grip) for k in range(1, n + 1))

        for name, i in env.plan_two:
            obj = env.data.body(name).xpos.copy()
            grasp = obj + [0, 0, env.grasp_height(name)]
            pre = grasp + [0, 0, min(LIFT, PRE_Z_MAX - grasp[2])]
            segment(at, pre, int(rng.integers(20, 30)), OPEN)
            segment(pre, grasp, int(rng.integers(12, 18)), OPEN)
            segment(grasp, grasp, 8, CLOSED)
            segment(grasp, pre, 15, CLOSED)
            top = env.coaster_top(i)
            put = np.array([top[0], top[1], grasp[2] + (top[2] - obj[2]) + PLACE_DROP])
            over = np.array([top[0], top[1], pre[2]])
            n = int(np.ceil(np.linalg.norm(over - pre) / CARRY_STEP)) + int(rng.integers(0, 6))
            segment(pre, over, n, CLOSED)
            segment(over, put, int(rng.integers(12, 18)), CLOSED)
            segment(put, put, 6, OPEN)
            segment(put, over, 12, OPEN)
            at = over
        self.plan, self.t = plan, 0

    def _plan_align(self, env, rng, obj, tool, grasp, pre):
        """Plan entries (position, gripper, tool angle): turn to the glass on the way there, grasp across its narrow
        side, carry it to the mat while turning to the mat's direction, put it down, let go, back up."""
        plan = []

        def segment(a, b, n, grip, y0, y1):
            plan.extend((a + (b - a) * k / n, grip, y0 + (y1 - y0) * k / n) for k in range(1, n + 1))

        yg, yp = tool_yaw(env.object_yaw()), tool_yaw(env.place_yaw)
        turn = lambda dy: int(np.ceil(abs(dy) / YAW_STEP))
        segment(tool, pre, max(int(rng.integers(20, 30)), turn(yg)), OPEN, 0.0, yg)
        segment(pre, grasp, int(rng.integers(12, 18)), OPEN, yg, yg)
        segment(grasp, grasp, 8, CLOSED, yg, yg)
        segment(grasp, pre, 15, CLOSED, yg, yg)
        top = env.place_position()
        put = np.array([top[0], top[1], grasp[2] + (top[2] - obj[2]) + PLACE_DROP])
        over = np.array([top[0], top[1], pre[2]])
        n = int(np.ceil(np.linalg.norm(over - pre) / CARRY_STEP)) + int(rng.integers(0, 6))
        segment(pre, over, max(n, turn(yp - yg)), CLOSED, yg, yp)
        segment(over, put, int(rng.integers(12, 18)), CLOSED, yp, yp)
        segment(put, put, 6, OPEN, yp, yp)
        segment(put, over, 12, OPEN, yp, yp)
        self.plan, self.t = plan, 0

    def done(self):
        return self.t >= len(self.plan)

    def act(self):
        target, grip, *yaw = self.plan[self.t]
        self.t += 1
        rot = turned(yaw[0]) if yaw else DOWN          # task "align": the tool turned about the vertical
        self.task.set_target(mink.SE3.from_rotation_and_translation(mink.SO3.from_matrix(rot), target))
        for _ in range(30):
            vel = mink.solve_ik(self.config, [self.task, self.posture], 0.05, "daqp", 1e-3, limits=self.limits)
            self.config.integrate_inplace(vel, 0.05)

        return np.append(self.config.q[:6], grip).astype(np.float32)