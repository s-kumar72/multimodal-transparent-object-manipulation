#!/bin/bash
# One-time setup on PSC Bridges-2 for training and evaluating Diffusion Policy on the sim data.
#
# Builds a fresh Python 3.11 conda env with torch (CUDA 12.6 build: runs on V100, L40S and H100), the MuJoCo sim,
# and Diffusion Policy's dependencies at the versions its code expects. Takes about 30 min, mostly the torch download.
# Run it once from the login node, after cloning this repo into $ROOT:
#   bash scripts/bridges2/setup.sh
#
# Layout under $ROOT (default $PROJECT_DIR/transparent, PROJECT_DIR default /ocean/projects/cis250019p/$USER):
#   multimodal-transparent-object-manipulation/   this repo
#   diffusion_policy/                             real-stanford/diffusion_policy, used as a library (cloned here)
#   envs/dp                                       the conda env (ENV_DIR overrides it)
set -e
PROJECT_DIR=${PROJECT_DIR:-/ocean/projects/cis250019p/$USER}
ROOT=${ROOT:-$PROJECT_DIR/transparent}
ENV=${ENV_DIR:-$ROOT/envs/dp}
REPO=$ROOT/multimodal-transparent-object-manipulation
DP=$ROOT/diffusion_policy
# Conda packages and the pip cache go to /ocean: the home quota is small
export CONDA_PKGS_DIRS=$PROJECT_DIR/.conda_pkgs
export PIP_CACHE_DIR=$PROJECT_DIR/.pip_cache
# torch's CUDA wheels come from pypi.nvidia.com, which sometimes stalls: wait longer and retry. The script is safe to
# rerun after a failure; finished steps are skipped.
export PIP_DEFAULT_TIMEOUT=120 PIP_RETRIES=10

[ -d "$REPO/sim" ] || { echo "clone this repo into $REPO first (or set ROOT)"; exit 1; }
[ -d "$DP" ] || git clone https://github.com/real-stanford/diffusion_policy.git "$DP"

module load anaconda3
eval "$(conda shell.bash hook)"
[ -x "$ENV/bin/python" ] || conda create -y -p "$ENV" python=3.11
conda activate "$ENV"

# gym 0.21 (the env API Diffusion Policy uses) only builds with an old pip, setuptools and wheel
pip install "pip<24.1" "setuptools<66" "wheel<0.40"
pip install --no-build-isolation gym==0.21.0

pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu126
pip install -r "$REPO/sim/requirements.txt"
# diffusers 0.11.1 is the version Diffusion Policy was written for. It imports cached_download, which
# huggingface_hub removed in 0.26, so the hub is pinned below that.
pip install diffusers==0.11.1 "huggingface_hub<0.26" hydra-core omegaconf einops dill numba av imagecodecs wandb \
    pandas tqdm threadpoolctl termcolor psutil scikit-image imageio imageio-ffmpeg pymunk shapely cloudpickle
pip install -e "$DP" --no-deps

cd "$REPO"
PYTHONPATH=$REPO:$REPO/sim:$DP python - <<'PY'
import warnings; warnings.filterwarnings("ignore")
import torch, torchvision, mujoco, mink, zarr, diffusers, gym, hydra
from diffusion_policy.workspace.train_diffusion_unet_image_workspace import TrainDiffusionUnetImageWorkspace
from diffusion_policy.model.common.lr_scheduler import get_scheduler
from policy.glass_dataset import GlassImageDataset
from policy.glass_runner import GlassPickRunner
print("torch", torch.__version__, "| mujoco", mujoco.__version__, "| diffusers", diffusers.__version__,
      "| gym", gym.__version__, "| zarr", zarr.__version__)
PY
echo "setup done: $ENV"
