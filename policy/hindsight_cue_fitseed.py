"""
policy/hindsight_cue.py with another seed of the fit (results/hard_scenario/cue_gate/README.md, round 6a):

  python policy/hindsight_cue_fitseed.py <k> TRAIN.zarr --out data/hcue/<folder>_fit<k> [--eval ...] [--mode early]

The fit functions of hindsight_cue.py seed torch and numpy themselves (network k of a fold with the fold's number, and
so on), so a fit is the same every time. Here every one of those seeds is shifted by 1000 * k: other initial weights,
other batch order, other augmentation shifts. The folds (which demo is held out of which network) stay those of the
default fit, and k = 0 is the default fit. Everything else, arguments included, is hindsight_cue.py.
"""
import sys

import numpy as np
import torch

K = int(sys.argv.pop(1))
_torch_seed, _numpy_seed = torch.manual_seed, np.random.seed
torch.manual_seed = lambda s: _torch_seed(int(s) + 1000 * K)
np.random.seed = lambda s: _numpy_seed(int(s) + 1000 * K)

from policy import hindsight_cue  # noqa: E402  (after the seeds are shifted)

if __name__ == "__main__":
    print(f"fit seed {K}: the seeds of the fit shifted by {1000 * K}", flush=True)
    hindsight_cue.main()
