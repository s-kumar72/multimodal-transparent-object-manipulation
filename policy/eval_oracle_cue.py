"""
policy/eval.py with the true grasp point in the cue (policy/oracle_cue_env.py; a reference that reads the simulator):

  python policy/eval_oracle_cue.py -c CKPT -o OUT --level L0_vis_ell --test_set j_cells11 ...     (options of eval.py)

The env class of the runner is replaced before the runner is made; the env workers are forked from this process.
"""
import policy.glass_runner as glass_runner
from policy.oracle_cue_env import OracleCueGymEnv

glass_runner.GlassPickGymEnv = OracleCueGymEnv

from policy.eval import main  # noqa: E402

if __name__ == "__main__":
    main()
