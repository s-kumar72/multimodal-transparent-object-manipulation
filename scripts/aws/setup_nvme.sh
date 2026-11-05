#!/bin/bash
# Builds the Python env and the Diffusion Policy clone on the instance's NVMe disk (/opt/dlami/nvme).
# The NVMe disk is ephemeral: it is wiped when the instance stops or reboots, so rerun this script after that
# (about 10 min). Everything that must survive lives in the repo (results/) on the root disk.
#   bash scripts/aws/setup_nvme.sh            then: source /opt/dlami/nvme/work/env.sh
set -e
NV=/opt/dlami/nvme
REPO=$(cd "$(dirname "$0")/../.." && pwd)
sudo chown "$USER:$USER" $NV
mkdir -p $NV/envs $NV/cache $NV/work $NV/bin
export UV_CACHE_DIR=$NV/cache/uv UV_PYTHON_INSTALL_DIR=$NV/python
[ -x $NV/bin/uv ] || curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=$NV/bin INSTALLER_NO_MODIFY_PATH=1 sh
U=$NV/bin/uv
[ -x $NV/envs/dp/bin/python ] || $U venv -p 3.11 $NV/envs/dp
source $NV/envs/dp/bin/activate
$U pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
$U pip install -r "$REPO/sim/requirements.txt" pyyaml
# gym 0.21 (Diffusion Policy's env API) only builds with an old pip, setuptools and wheel
$U pip install "pip<24.1" "setuptools<66" "wheel<0.40"
pip install --no-build-isolation gym==0.21.0
$U pip install diffusers==0.11.1 "huggingface_hub<0.26" hydra-core omegaconf einops dill numba av imagecodecs wandb \
    pandas tqdm threadpoolctl termcolor psutil scikit-image imageio imageio-ffmpeg pymunk shapely cloudpickle
[ -d $NV/diffusion_policy ] || git clone -q https://github.com/real-stanford/diffusion_policy.git $NV/diffusion_policy
$U pip install -e $NV/diffusion_policy --no-deps
ln -sfn $NV/diffusion_policy "$REPO/../diffusion_policy"
cat > $NV/work/env.sh <<ENV
source $NV/envs/dp/bin/activate
cd $REPO
export PYTHONPATH=$REPO:$REPO/sim:$NV/diffusion_policy MUJOCO_GL=egl WANDB_MODE=offline SDL_VIDEODRIVER=dummy PYTHONWARNINGS=ignore
ENV
source $NV/work/env.sh
python -c "import torch, mujoco, gym, diffusers; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'mujoco', mujoco.__version__); import diffusion_policy.workspace.train_diffusion_unet_image_workspace; print('setup done')"
