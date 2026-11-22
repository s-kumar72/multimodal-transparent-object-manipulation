"""
policy/eval.py for fusion-encoder checkpoints, with --save_w for all of them.

New checkpoints (policy class fusion.attention.fusion_policy.FusionDiffusionPolicy) already work with policy/eval.py --save_w.
Checkpoints trained before that class existed use DP's plain DiffusionUnetImagePolicy, which has no weight trace;
this wrapper adds it (only when the encoder is a FusionObsEncoder) and then runs policy/eval.py unchanged, with the
same options:
  python -m fusion.attention.eval_fusion -c <ckpt> -o <out> --level L0_bench --n_test 200 --n_envs 10 \
      --test_set a_in_distribution --test_set g_heldout_all --save_w
Run from the repo root with the PYTHONPATH of policy/train.sh (repo, sim/, ../diffusion_policy).
"""
from diffusion_policy.policy.diffusion_unet_image_policy import DiffusionUnetImagePolicy

from fusion.attention.fusion_policy import record_w
from fusion.attention.policy_encoder import FusionObsEncoder


def patch():
    if getattr(DiffusionUnetImagePolicy, "_fusion_trace_patched", False):
        return
    reset, predict = DiffusionUnetImagePolicy.reset, DiffusionUnetImagePolicy.predict_action

    def is_fusion(self):
        return type(self) is DiffusionUnetImagePolicy and isinstance(getattr(self, "obs_encoder", None), FusionObsEncoder)

    def reset_(self):
        reset(self)
        if is_fusion(self) and getattr(self, "w_trace", None) is not None:
            self.w_trace.append([])

    def predict_(self, obs_dict):
        out = predict(self, obs_dict)
        if is_fusion(self) and getattr(self, "w_trace", None) is not None:
            record_w(self, obs_dict)
        return out

    DiffusionUnetImagePolicy.reset, DiffusionUnetImagePolicy.predict_action = reset_, predict_
    DiffusionUnetImagePolicy.w_trace = None              
    DiffusionUnetImagePolicy._fusion_trace_patched = True


if __name__ == "__main__":
    patch()
    from policy import eval as policy_eval                 
    policy_eval.main()