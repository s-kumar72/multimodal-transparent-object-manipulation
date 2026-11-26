"""
The evaluation env with the true grasp point in place of the cue's (results/hard_scenario/cue_gate/README.md, round
6c): a reference, not a method. It reads the simulator.

At reset the cue vector the policy gets is the checkpoint's own cue with its first five entries (the point x, y, z and
its pixel u, v) replaced by the grasp point of this scene: the glass's position in the table plane and the grasp height
of this object, which is where the scripted teacher closes the gripper and so what the hindsight labels of the demos
are (policy/check_oracle_point.py compares the two on the 700 demos). The confidence and the three gate entries stay
those of the checkpoint's cue, so a policy trained with the method's cue is asked one thing only: what it does when
the point is right. With ORACLE_CUE_ONES=1 they are set to 1, the layout of the label cue (hindsight_cue.py --mode
label), for a policy that was trained on the labels themselves.

policy/eval_oracle_cue.py runs policy/eval.py with this env.
"""
import os

import numpy as np

from policy.glass_gym_env import GlassPickGymEnv
from policy.hindsight_cue import S, project


def true_grasp_point(env):
    """(x, y, z) of the grasp point of the scene of a sim/env.py GlassPickEnv right after reset."""
    return np.array([*env.data.body(env.obj).xpos[:2], env.obj_z0 + env.grasp_height()], np.float64)


class OracleCueGymEnv(GlassPickGymEnv):
    def _with_hcue(self, raw, reset=False):
        if "hcue" not in self.obs_meta:
            return raw
        if reset:
            cue = np.array(super()._with_hcue(raw, reset=True)["hcue"], np.float32).copy()   # the checkpoint's own cue
            p = true_grasp_point(self._env)
            cue[:3], cue[3:5] = p, project(self._hcue_model.cam, p[None])[0] / S
            if os.environ.get("ORACLE_CUE_ONES") == "1":
                cue[5:] = 1.0
            self._hcue = cue
        return dict(raw, hcue=self._hcue)
