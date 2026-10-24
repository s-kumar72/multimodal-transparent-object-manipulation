"""
Rolls out a policy in parallel GlassPickEnv copies and reports success rate (test/mean_score),
plus the success rate per object (test/mean_score_<object>).
Adapted from diffusion_policy/env_runner/pusht_image_runner.py.
"""
import collections
import math
import pathlib
import uuid

import dill
import numpy as np
import torch
import tqdm
import wandb
from omegaconf import OmegaConf

from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.env_runner.base_image_runner import BaseImageRunner
from diffusion_policy.gym_util.async_vector_env import AsyncVectorEnv
from diffusion_policy.gym_util.multistep_wrapper import MultiStepWrapper
from diffusion_policy.gym_util.video_recording_wrapper import VideoRecorder, VideoRecordingWrapper

from policy.glass_gym_env import GlassPickGymEnv

class GlassPickRunner(BaseImageRunner):
    def __init__(self, output_dir, shape_meta, n_train=4, n_train_vis=1, train_start_seed=0,
                 n_test=30, n_test_vis=3, test_start_seed=100000, max_steps=120,
                 n_obs_steps=2, n_action_steps=8, fps=10, crf=22, past_action=False,
                 tqdm_interval_sec=5.0, n_envs=None):
        super().__init__(output_dir)
        if OmegaConf.is_config(shape_meta):
            shape_meta = OmegaConf.to_container(shape_meta, resolve=True)
        obs_meta = shape_meta["obs"]
        if n_envs is None:
            n_envs = n_train + n_test

        def env_fn():
            return MultiStepWrapper(
                VideoRecordingWrapper(
                    GlassPickGymEnv(obs_meta, max_steps=max_steps),
                    video_recoder=VideoRecorder.create_h264(fps=fps, codec="h264", input_pix_fmt="rgb24",
                                                            crf=crf, thread_type="FRAME", thread_count=1),
                    file_path=None),
                n_obs_steps=n_obs_steps,
                n_action_steps=n_action_steps,
                max_episode_steps=max_steps)

        env_seeds, env_prefixs, env_init_fn_dills = [], [], []
        for prefix, n, n_vis, start_seed in [("train/", n_train, n_train_vis, train_start_seed),
                                             ("test/", n_test, n_test_vis, test_start_seed)]:
            for i in range(n):
                seed = start_seed + i

                def init_fn(env, seed=seed, enable_render=i < n_vis):
                    env.env.video_recoder.stop()
                    env.env.file_path = None
                    if enable_render:
                        filename = pathlib.Path(output_dir).joinpath("media", uuid.uuid4().hex[:12] + ".mp4")  # wandb.util.generate_id is gone in new wandb
                        filename.parent.mkdir(parents=False, exist_ok=True)
                        env.env.file_path = str(filename)
                    env.seed(seed)

                env_seeds.append(seed)
                env_prefixs.append(prefix)
                env_init_fn_dills.append(dill.dumps(init_fn))

        self.env = AsyncVectorEnv([env_fn] * n_envs)
        self.n_envs = n_envs
        self.env_seeds = env_seeds
        self.env_prefixs = env_prefixs
        self.env_init_fn_dills = env_init_fn_dills
        self.n_obs_steps = n_obs_steps
        self.past_action = past_action
        self.max_steps = max_steps
        self.tqdm_interval_sec = tqdm_interval_sec

    def run(self, policy):
        device = policy.device
        env = self.env
        n_envs = self.n_envs
        n_inits = len(self.env_init_fn_dills)
        all_video_paths = [None] * n_inits
        all_rewards = [None] * n_inits
        all_objs = [None] * n_inits

        for chunk_idx in range(math.ceil(n_inits / n_envs)):
            start = chunk_idx * n_envs
            end = min(n_inits, start + n_envs)
            this_init_fns = self.env_init_fn_dills[start:end]
            this_init_fns += [self.env_init_fn_dills[0]] * (n_envs - len(this_init_fns))
            env.call_each("run_dill_function", args_list=[(x,) for x in this_init_fns])

            obs = env.reset()
            past_action = None
            policy.reset()
            pbar = tqdm.tqdm(total=self.max_steps, desc=f"Eval GlassPickRunner {chunk_idx + 1}",
                             leave=False, mininterval=self.tqdm_interval_sec)
            done = False
            while not done:
                np_obs_dict = dict(obs)
                if self.past_action and past_action is not None:
                    np_obs_dict["past_action"] = past_action[:, -(self.n_obs_steps - 1):].astype(np.float32)
                obs_dict = dict_apply(np_obs_dict, lambda x: torch.from_numpy(x).to(device=device))
                with torch.no_grad():
                    action_dict = policy.predict_action(obs_dict)
                action = action_dict["action"].detach().to("cpu").numpy()

                obs, reward, done, info = env.step(action)
                done = np.all(done)
                past_action = action
                pbar.update(action.shape[1])
            pbar.close()

            n_active = end - start
            all_video_paths[start:end] = env.render()[:n_active]
            all_rewards[start:end] = env.call("get_attr", "reward")[:n_active]
            all_objs[start:end] = env.call("get_attr", "obj")[:n_active]
        _ = env.reset()

        max_rewards = collections.defaultdict(list)
        obj_rewards = collections.defaultdict(list)
        log_data = dict()
        for i in range(n_inits):
            seed, prefix = self.env_seeds[i], self.env_prefixs[i]
            max_reward = np.max(all_rewards[i]) if len(all_rewards[i]) > 0 else 0.0
            max_rewards[prefix].append(max_reward)
            obj_rewards[(prefix, all_objs[i])].append(max_reward)
            log_data[prefix + f"sim_max_reward_{seed}"] = max_reward
            if all_video_paths[i] is not None:
                log_data[prefix + f"sim_video_{seed}"] = wandb.Video(all_video_paths[i])
        for prefix, value in max_rewards.items():
            log_data[prefix + "mean_score"] = np.mean(value)    # = success rate
        for (prefix, obj), value in sorted(obj_rewards.items()):
            log_data[prefix + f"mean_score_{obj}"] = np.mean(value)   # success rate on the episodes with this object
            log_data[prefix + f"n_{obj}"] = len(value)
        summary = {obj: f"{int(sum(v))}/{len(v)}" for (prefix, obj), v in sorted(obj_rewards.items()) if prefix == "test/"}
        print(f"test/mean_score {log_data.get('test/mean_score', float('nan')):.3f}  per object {summary}", flush=True)
        return log_data
