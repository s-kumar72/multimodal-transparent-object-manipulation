"""
gym 0.21 wrapper around sim/env.py GlassPickEnv, so Diffusion Policy's MultiStepWrapper / AsyncVectorEnv can run it.
reward = 1 once the glass is lifted 10 cm (episode ends there); in a pick-and-place level (sim/config.py, task "place")
once the glass stands on the coaster and the gripper has let go (GlassPickEnv.placed).
The scene is main's unless a level (sim/configs/manifest.json) is given; test episodes use one of its test sets.
Each episode also gets a failure label for the breakdown in results (see `outcome`).
"""
import gym
import numpy as np
from gym import spaces

from config import load_config
from env import ALIGN_POS_TOL, COASTER_H, PLACE_UPRIGHT, PLACE_Z_TOL, GlassPickEnv, OBJECT_NAMES, glass_names
from policy.obs_util import to_policy_obs
from testsets import manifest, test_set

CTRL_HIGH = np.array([2.69, 2.69, 2.69, 2.59, 2.57, 2.59, 0.85], np.float32)
CTRL_LOW = np.append(-CTRL_HIGH[:6], 0.0).astype(np.float32)
CLOSED = 0.4        # gripper command above this counts as closing
NEAR = 0.04         # horizontal distance (m) at which a closing gripper is "at" an object
HEIGHT_TOL = 0.025  # |closing height - grasp height| (m) beyond which a grasp at the target is at the wrong height
PICKED = 0.10       # pick-and-place: the glass counts as picked once it has been this far (m) above the table

class GlassPickGymEnv(gym.Env):
    metadata = {"render.modes": ["rgb_array"], "video.frames_per_second": 10}

    def __init__(self, obs_meta, max_steps=120, level=None, test_set_name="a_in_distribution"):
        self.obs_meta = obs_meta
        self.max_steps = max_steps
        self.level, self.test_set_name = level, test_set_name
        self.observation_space = spaces.Dict({
            k: spaces.Box(0, 1, shape=tuple(a["shape"]), dtype=np.float32) if a.get("type", "low_dim") == "rgb"
            else spaces.Box(-np.inf, np.inf, shape=tuple(a["shape"]), dtype=np.float32)
            for k, a in obs_meta.items()})
        self.action_space = spaces.Box(CTRL_LOW, CTRL_HIGH, dtype=np.float32)
        # Created on first reset, so AsyncVectorEnv's dummy env in the parent never opens a GL context
        self._envs = {}
        self._env = None
        self._seed = None
        self._split = "test"
        self._frame = None
        self._log = None
        self._hcue_model = self._hcue = None

    def _with_hcue(self, raw, reset=False):
        """Adds the hindsight cue (policy/hindsight_cue.py) when the policy has the `hcue` observation: computed from
        the first frame of the episode and kept for all its steps. `ckpt` may be a list of cue files (pick-and-place:
        the grasp cue and the place cue); the vector is then their cues one after the other."""
        if "stage" in self.obs_meta:    # a loss target of training (policy/msbot_encoder.py); no value at evaluation
            raw = dict(raw, stage=np.zeros(2, np.float32))
        if "hcue" not in self.obs_meta:
            return raw
        if reset:
            if self._hcue_model is None:
                import torch
                from policy.hindsight_cue import HindsightCue
                torch.set_num_threads(1)      # env workers are forked processes; one thread each, like DataLoader workers
                ckpt = self.obs_meta["hcue"]["ckpt"]
                self._hcue_model = [HindsightCue(p) for p in ([ckpt] if isinstance(ckpt, str) else list(ckpt))]
            self._hcue = np.concatenate([m(raw) for m in self._hcue_model])
        return dict(raw, hcue=self._hcue)

    def seed(self, seed=None):
        self._seed = seed

    def set_split(self, split):
        """train: the level's training scenes; test: its test set (set before reset)."""
        self._split = split

    def _get_env(self):
        key = self._split if self.level is not None else "main"
        if key not in self._envs:
            if self.level is None:
                cfg, objects = None, OBJECT_NAMES
            elif self._split == "train":
                cfg = load_config(manifest()["levels"][self.level])
                objects = glass_names(cfg)
            else:
                cfg, _, objects = test_set(self.level, self.test_set_name)
            self._envs[key] = (GlassPickEnv(max_steps=self.max_steps, config=cfg), objects)
        return self._envs[key]

    def reset(self):
        self._env, objects = self._get_env()
        # The object cycles with the seed, so an evaluation over 5k consecutive seeds has k episodes per object
        obj = objects[self._seed % len(objects)] if self._seed is not None else None
        raw = self._env.reset(self._seed, obj=obj)
        e = self._env
        self._log = dict(max_lift=0.0, closes=[], z0=e.obj_z0, target0=e.data.body(e.obj).xpos.copy())
        self._frame = raw["rgb"]
        return to_policy_obs(self._with_hcue(raw, reset=True), self.obs_meta)

    def step(self, action):
        e = self._env
        was_closed = e.data.ctrl[6] > CLOSED
        raw, done = e.step(action)
        if action[6] > CLOSED and not was_closed:   # a grasp attempt: where the tool was when closing started
            tool, target = e.data.body("tool_frame").xpos, e.data.body(e.obj).xpos
            extras = {n: np.linalg.norm(e.data.body(n).xpos[:2] - tool[:2]) for n in e.extra_z0}
            self._log["closes"].append(dict(
                d_target=float(np.linalg.norm(target[:2] - tool[:2])),
                dz=float(tool[2] - (self._log["z0"] + e.grasp_height())),   # vs. the right height for this object
                d_extra=float(min(extras.values())) if extras else None))
        self._log["max_lift"] = max(self._log["max_lift"], float(e.data.body(e.obj).xpos[2] - self._log["z0"]))
        self._frame = raw["rgb"]
        reward = float(e.success())
        return to_policy_obs(self._with_hcue(raw), self.obs_meta), reward, done, {}

    @property
    def outcome(self):
        """Why an episode failed, judged from the last grasp (gripper closing):
        wrong_object  a distractor was lifted, or the gripper closed at a distractor rather than the target
        wrong_height  closed at the target but > 2.5 cm from this object's grasp height (object type mistaken),
                      whether or not it then slipped out
        dropped       closed at the right height, lifted the target > 3 cm, lost it
        missed_grasp  closed near the target (< 8 cm) without lifting it
        timeout       never closed the gripper
        other         closed away from every object
        Pick-and-place levels: the labels above when the glass was never lifted 10 cm; once it was, from where the
        glass is at the end:
        place_fell    it does not stand upright
        place_held    it is still in the air (not let go, or the episode ran out on the way)
        place_off     it stands upright, but not with its footprint inside the coaster
        place_other   it stands on the coaster, but the gripper has not let go of it or it still moves
        Task "align": place_off when its centre is further than the tolerance from the centre of the mat, and
        place_turned  when it lies on the mat but not in the mat's direction"""
        e, log = self._env, self._log
        if e is None or log is None:
            return None
        if e.success():
            return "success"
        if e.place and log["max_lift"] >= PICKED:
            b = e.data.body(e.obj)
            if b.xmat[8] <= PLACE_UPRIGHT:
                return "place_fell"
            if b.xpos[2] - log["z0"] > COASTER_H + PLACE_Z_TOL:
                return "place_held"
            if e.align:
                return "place_off" if e.place_offset() > ALIGN_POS_TOL else \
                    "place_turned" if e.yaw_error() > e.yaw_tolerance else "place_other"
            if e.place_offset() > e.cfg.place_radius - e.obj_r[e.obj]:
                return "place_off"
            return "place_other"
        if e.lifted_extras(0.03):
            return "wrong_object"
        if not log["closes"]:
            return "dropped" if log["max_lift"] > 0.03 else "timeout"
        c = log["closes"][-1]
        if c["d_extra"] is not None and c["d_extra"] < NEAR and c["d_extra"] < c["d_target"]:
            return "wrong_object"
        if c["d_target"] < NEAR and abs(c["dz"]) > HEIGHT_TOL:
            return "wrong_height"
        if log["max_lift"] > 0.03:
            return "dropped"
        return "missed_grasp" if c["d_target"] < 2 * NEAR else "other"

    @property
    def episode_record(self):
        """Per-episode summary for eval.py (scene conditions + outcome)."""
        if self._env is None:
            return None
        info = {k: (v if not isinstance(v, (np.floating, np.integer)) else v.item())
                for k, v in self._env.episode_info.items()}
        rec = dict(info, outcome=self.outcome, obj_name=self._env.obj, max_lift=self._log["max_lift"],
                   closes=self._log["closes"])
        if self._env.place:             # pick-and-place: was the glass picked, and how far from the coaster it ended
            rec.update(picked=bool(self._log["max_lift"] >= PICKED), place_offset=self._env.place_offset(),
                       coaster_xy=[float(v) for v in self._env.place_xy])
        if getattr(self._env, "align", False):   # T2: how far the glass is turned from the mat at the end (deg)
            rec.update(yaw_error=self._env.yaw_error())
        if self._hcue is not None:      # the cue the policy was given, and where the glass really was (for the analysis)
            rec.update(hcue=[float(v) for v in self._hcue], target_xyz=[float(v) for v in self._log["target0"]])
        return rec

    def render(self, mode="rgb_array"):
        return self._frame

    @property
    def obj(self):
        """Name of the object on the table in the current episode (for per-object success rates)."""
        return self._env.obj if self._env is not None else None

    def close(self):
        for env, _ in self._envs.values():
            env.close()
        self._envs, self._env = {}, None
