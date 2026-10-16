# multimodal-transparent-object-manipulation
Github repo for CMU 11-785 project on multimodal transparent object manipulation.

**MuJoCo Simulation Setup**
***Requirements***:
1. Python 3.11 recommended
2. OS: Linux (Ubuntu 22.04/24.04) or macOS recommended
3. Packages: see sim/requirements.txt & below

| Package | Version | Used for |
|---|---|---|
| `mujoco` | ≥ 3.13, < 4 | Physics and rendering (3.13+ needed so the gripper fingers stay linked) |
| `robot_descriptions` | ≥ 1.12, < 4 | Downloads Kinova's [ros2_kortex](https://github.com/Kinovarobotics/ros2_kortex) repo (Gen3 lite URDF and meshes) |
| `numpy` | ≥ 1.24, < 3 | Array manipulation |
| `pillow` | ≥ 10, < 13 | Saving rendered RGB and depth images |

To setup the virtual environment:
```bash
git clone https://github.com/s-kumar72/multimodal-transparent-object-manipulation.git
cd multimodal-transparent-object-manipulation
```
***Using a Python venv:***
```bash
python3.11 -m venv .venv
source .venv/bin/activate          
pip install --upgrade pip
pip install -r sim/requirements.txt
```

***Using conda/ miniconda:***
```bash
conda create -n transparent-sim python=3.11 -y
conda activate transparent-sim
pip install -r sim/requirements.txt
```
