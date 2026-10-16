# CONVERTS KINOVA'S GEN3 LITE URDF (from ros2_kortex) INTO A MUJOCO MODEL.
#
# Run from repo root: python sim/prepare_gen3_lite.py
# Output: sim/assets/gen3_lite/ (meshes, urdf, xml)

import re
import shutil
from pathlib import Path

import mujoco
from robot_descriptions import gen3_lite_description as desc

PKG = Path(desc.PACKAGE_PATH)
SRC_URDF = PKG / "robots" / "gen3_lite.urdf"
OUT = Path("sim/assets/gen3_lite")
MESH_OUT = OUT / "meshes"
MESH_OUT.mkdir(parents=True, exist_ok=True)

text = SRC_URDF.read_text()

# Remove ROS only blocks - not needed for MuJoCo
text = re.sub(r"<ros2_control.*?</ros2_control>", "", text, flags=re.S)
text = re.sub(r"<gazebo.*?</gazebo>", "", text, flags=re.S)
text = re.sub(r"<transmission.*?</transmission>", "", text, flags=re.S)

# Copy every mesh next to the model & point URDF at the plain filename
def fix_mesh(match):
    rel = match.group(1)
    src = PKG / rel
    shutil.copy(src, MESH_OUT / src.name)
    return f'filename="{src.name}"'

text = re.sub(r'filename="package://kortex_description/([^"]+)"', fix_mesh, text)

# Point MuJoCo to meshes and keep fixed frames
mujoco_block = (
    "<mujoco><compiler meshdir=\"meshes/\" balanceinertia=\"true\" "
    "discardvisual=\"false\" fusestatic=\"false\"/></mujoco>"
)
text = re.sub(r"(<robot[^>]*>)", r"\1\n " + mujoco_block, text, count=1)

urdf_out = OUT / "gen3_lite.urdf"
urdf_out.write_text(text)

# Load URDF with MuJoCo and save as MJCF
spec = mujoco.MjSpec.from_file(str(urdf_out))
for body in spec.bodies:
    if body.name != "world":
        body.gravcomp = 1.0
for joint in spec.joints:
    joint.armature = 0.01       # add motor inertia for stable position control
model = spec.compile()
(OUT / "gen3_lite.xml").write_text(spec.to_xml())

print(f"SAVED {OUT / 'gen3_lite.xml'}")
print(f"{'joint':28s} range (rad)")
for i in range(model.njnt):
    j = model.joint(i)
    print(f"{j.name:28s} {model.jnt_range[i]}")
print(f"equality constraints (finger coupling): {model.neq}")   # should be 3


