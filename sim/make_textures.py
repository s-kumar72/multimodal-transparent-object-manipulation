"""
Generates the table textures in sim/assets/textures/ (128x128 PNG, fixed seed, committed to git).

run via python sim/make_textures.py
"""
from pathlib import Path

import numpy as np
from PIL import Image

N = 128
OUT = Path("sim/assets/textures")

def smooth_noise(rng, cells):
    """Periodic value noise in [0, 1]: a random cells x cells grid, tiled 3x3, upsampled, centre crop (no seams)."""
    g = np.tile(rng.random((cells, cells)).astype(np.float32), (3, 3))
    a = np.asarray(Image.fromarray(g, mode="F").resize((3 * N, 3 * N), Image.BICUBIC))[N:2 * N, N:2 * N]
    return (a - a.min()) / np.ptp(a)

def save(name, gray, c0, c1):
    rgb = np.asarray(c0)[None, None] * (1 - gray[..., None]) + np.asarray(c1)[None, None] * gray[..., None]
    Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8)).save(OUT / f"{name}.png", optimize=True)

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    OUT.mkdir(parents=True, exist_ok=True)
    y, x = np.mgrid[0:N, 0:N] / N
    save("checker", ((x * 8).astype(int) + (y * 8).astype(int)) % 2 * 1.0, (60, 60, 60), (200, 200, 200))
    save("stripes", (np.sin(2 * np.pi * 6 * (x + y)) > 0) * 1.0, (40, 70, 120), (220, 220, 210))
    rings = np.sin(2 * np.pi * (10 * y + 3 * smooth_noise(rng, 6))) * 0.5 + 0.5
    save("wood", 0.7 * rings + 0.3 * smooth_noise(rng, 32), (95, 55, 30), (190, 135, 85))
    save("noise", smooth_noise(rng, 24), (70, 70, 70), (190, 190, 190))
    marble = np.abs(np.sin(2 * np.pi * (3 * x + 4 * smooth_noise(rng, 8))))
    save("marble", marble ** 0.3, (120, 120, 125), (235, 235, 240))
    print(f"saved {sorted(p.name for p in OUT.glob('*.png'))}, "
          f"{sum(p.stat().st_size for p in OUT.glob('*.png')) / 1024:.0f} KB")
