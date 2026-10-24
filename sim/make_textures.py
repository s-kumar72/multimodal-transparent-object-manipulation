"""
Generates the table background textures used by scene randomization (fixed seed, so output is always the same).

run via python sim/make_textures.py
Output: sim/assets/gen3_lite/textures/*.png (checker textures are MuJoCo builtins, see transparent_scene.xml)
"""
from pathlib import Path

import numpy as np
from PIL import Image

N = 128
OUT = Path("sim/assets/gen3_lite/textures")
rng = np.random.default_rng(0)
y, x = np.mgrid[0:N, 0:N] / N

def two_color(t, c0, c1):
    t = np.clip(t, 0, 1)[..., None]
    return (1 - t) * np.array(c0) + t * np.array(c1)

def blocks(size, colors):
    idx = rng.integers(0, len(colors), (N // size, N // size))
    return np.kron(np.array(colors)[idx], np.ones((size, size, 1)))

def wood(c0, c1, rings, warp):
    noise = np.kron(rng.normal(0, 1, (N // 16, N // 16)), np.ones((16, 16)))
    t = 0.5 + 0.5 * np.sin(2 * np.pi * (rings * x + warp * noise + 0.15 * np.sin(6 * np.pi * y)))
    return two_color(t ** 2, c0, c1) * (1 + 0.05 * rng.normal(0, 1, (N, N, 1)))

textures = {
    "stripes_a": two_color(np.sin(2 * np.pi * 8 * y) > 0, (0.92, 0.88, 0.75), (0.15, 0.35, 0.25)),
    "stripes_b": two_color(np.sin(2 * np.pi * 12 * (x + y)) > 0, (0.95, 0.95, 0.95), (0.75, 0.15, 0.15)),
    "wood_light": wood((0.85, 0.7, 0.5), (0.65, 0.48, 0.3), 6, 0.08),
    "wood_dark": wood((0.4, 0.25, 0.15), (0.22, 0.12, 0.07), 9, 0.12),
    "noise_fine": blocks(2, [(0.9, 0.9, 0.9), (0.2, 0.2, 0.2), (0.7, 0.3, 0.2), (0.2, 0.4, 0.7), (0.8, 0.75, 0.3)]),
    "noise_coarse": blocks(8, [(0.95, 0.95, 0.95), (0.1, 0.1, 0.1), (0.5, 0.5, 0.5), (0.3, 0.6, 0.35), (0.6, 0.3, 0.6)]),
}

OUT.mkdir(parents=True, exist_ok=True)
for name, img in textures.items():
    Image.fromarray((255 * np.clip(img, 0, 1)).astype(np.uint8)).save(OUT / f"{name}.png")
    print("saved", OUT / f"{name}.png")
