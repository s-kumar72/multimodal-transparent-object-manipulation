"""
Renders RGB/depth/thermal for a hot glass and a glass at thermal crossover
Author: Sadhana Kumar
"""

from pathlib import Path
import mujoco
import numpy as np
from PIL import Image, ImageDraw
from sensors import Sensors

model = mujoco.MjModel.from_xml_path("sim/assets/gen3_lite/transparent_scene.xml")
data = mujoco.MjData(model)
READY = [-0.118, -0.575, 0.968, 1.571, 1.599, -1.689]
data.qpos[:6] = READY
data.ctrl[:6] = READY
gq = model.jnt_qposadr[model.body("glass").jntadr[0]]
data.qpos[gq:gq + 2] = [0.40, 0.0]
for _ in range(500):
    mujoco.mj_step(model, data)

sens = Sensors(model)
emis = {"glass": 0.92, "table": 0.95}
rows = []
for label, t_glass in [("glass 55 C", 55.0), ("crossover 22 C", 22.0)]:
    temps = {"glass": t_glass, "table": 22.0}
    temps.update({model.body(i).name: 30.0 for i in range(model.nbody) if "link" in model.body(i).name})
    rgb, depth, th = sens.render(data, temps, emis, t_env=22.0, rng=np.random.default_rng(0))
    d = np.clip(depth, 0, 2.0) / 2.0
    tiles = [Image.fromarray(rgb),
              Image.fromarray((255 * d).astype(np.uint8)).convert("RGB"),
              Image.fromarray((255 * th).astype(np.uint8)).resize((320, 240), Image.NEAREST).convert("RGB")]

    row = Image.new("RGB", (960, 240))
    for i, t in enumerate(tiles):
        row.paste(t, (320 * i, 0))
    ImageDraw.draw(row).text((8, 8), label, fill=(255, 255, 0))
    rows.append(row)

Path("sim/out").mkdir(exist_ok=True)
grid = Image.new("RGB", (960, 480)); grid.paste(rows[0], (0, 0)); grid.paste(rows[1], (0, 240))
grid.save("sim/out/sensors.png")
print("Saved sim/out/sensors.png")