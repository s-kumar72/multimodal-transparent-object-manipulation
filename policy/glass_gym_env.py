"""
gym 0.21 wrapper around sim/env.py GlassPickEnv, so Diffusion Policy's MultiStepWrapper / AsyncVectorEnv can run it.
reward = 1 once the glass is lifted 10 cm (episode ends there).
"""
import gym
import numpy as np
from gym import spaces

from env import GlassPickEnv, OBJECT_NAMES
from policy.obs_util import to_policy_obs

CTRL_HIGH = np.array([2.69, 2.69, 2.69, 2.59, 2.57, 2.59, 0.85], np.float32)
CTRL_LOW = np.append(-CTRL_HIGH[:6], 0.0).astype(np.float32)

class GlassPickGymEnv(gym.Env):
    metadata = {"render.modes": ["rgb_array"], "video.frames_per_second": 10}

    def __init__(self, obs_meta, max_steps=120):
        self.obs_meta = obs_meta
        self.max_steps = max_steps
        self.observation_space = spaces.Dict({
            k: spaces.Box(0, 1, shape=tuple(a["shape"]), dtype=np.float32) if a.get("type", "low_dim") == "rgb"
            else spaces.Box(-np.inf, np.inf, shape=tuple(a["shape"]), dtype=np.float32)
            for k, a in obs_meta.items()})
        self.action_space = spaces.Box(CTRL_LOW, CTRL_HIGH, dtype=np.float32)
        # Created on first reset, so AsyncVectorEnv's dummy env in the parent never opens a GL context
        self._env = None
        self._seed = None
        self._frame = None

    def seed(self, seed=None):
        self._seed = seed

    def reset(self):
        if self._env is None:
            self._env = GlassPickEnv(max_steps=self.max_steps)
        # The object cycles with the seed, so an evaluation over 5k consecutive seeds has k episodes per object
        obj = OBJECT_NAMES[self._seed % len(OBJECT_NAMES)] if self._seed is not None else None
        raw = self._env.reset(self._seed, obj=obj)
        self._frame = raw["rgb"]
        return to_policy_obs(raw, self.obs_meta)

    def step(self, action):
        raw, done = self._env.step(action)
        self._frame = raw["rgb"]
        reward = float(self._env.success())
        return to_policy_obs(raw, self.obs_meta), reward, done, {}

    def render(self, mode="rgb_array"):
        return self._frame

    @property
    def obj(self):
        """Name of the object on the table in the current episode (for per-object success rates)."""
        return self._env.obj if self._env is not None else None

    def close(self):
        if self._env is not None:
            self._env.close()
            self._env = None
