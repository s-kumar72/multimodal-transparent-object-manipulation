"""
Renders RGB, depth, and simulated thermal using MuJoCo camera.
The default config (sim/config.py) reproduces main; the extra depth / thermal effects are off unless a config sets them.
Author: Sadhana Kumar
"""
import mujoco
import numpy as np

from config import SceneConfig, factor_rng

H_PLANCK, C_LIGHT, K_B = 6.626e-34, 2.998e8, 1.381e-23
_WAVE = np.linspace(8e-6, 14e-6, 200) # low wave IR range of most thermal cameras
RAW_MAX = 16383 # 14-bit thermal counts
TABLE_TOP = 0.02

def band_radiance(temp_c):
    t = np.asarray(temp_c, float)[..., None] + 273.15
    spec = 2 * H_PLANCK * C_LIGHT**2 / _WAVE**5 / np.expm1(H_PLANCK * C_LIGHT / (_WAVE * K_B * t))
    return spec.sum(axis=-1) * (_WAVE[1] - _WAVE[0])

def band_radiance_map(temp):
    """band_radiance of an image that has few distinct temperatures: one Planck integral per distinct value."""
    u, inv = np.unique(temp, return_inverse=True)
    return band_radiance(u)[inv].reshape(temp.shape)

def gaussian_blur(img, sigma):
    r = max(1, int(round(3 * sigma)))
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()
    pad = np.pad(img, r, mode="edge")
    rows = sum(w * pad[:, i:i + img.shape[1]] for i, w in enumerate(k))
    return sum(w * rows[i:i + img.shape[0]] for i, w in enumerate(k))

class Sensors:
    TRANSPARENT_GROUP = 5 # geom group of transparent geoms in the scene XML (hidden by default MjvOption)

    def __init__(self, model, camera="front_cam", height=240, width=320, thermal_size=(120, 160), config=None):
        self.model, self.camera = model, camera
        self.cfg = config or SceneConfig()
        self.r = mujoco.Renderer(model, height, width)
        self.thermal_size = thermal_size
        self.transparent_geoms = np.flatnonzero(model.geom_group == self.TRANSPARENT_GROUP)
        self.geom_body = model.geom_bodyid.copy()
        self.show_all = mujoco.MjvOption()
        self.show_all.geomgroup[self.TRANSPARENT_GROUP] = 1
        self.hide_transparent = mujoco.MjvOption()
        self.table_geoms = np.flatnonzero(model.geom_bodyid == model.body("table").id)
        cam = model.camera(camera).id
        self.cam_id, self.focal = cam, 0.5 * height / np.tan(np.radians(model.cam_fovy[cam]) / 2)
        self.scene = None

    def episode(self, seed, target, liquid, extras, data):
        """Per-episode sensor effects: liquid in the target, absorber holes and table hole patches (lever C)."""
        cfg, m = self.cfg, self.model
        rng = factor_rng(seed, "sensor")
        holes = {}
        for name in extras:
            if name.startswith("abs_"):
                frac = rng.uniform(*cfg.absorber_holes)
                holes.update({int(g): frac for g in np.flatnonzero(m.geom_bodyid == m.body(name).id)})
        blobs = [(rng.uniform(0.24, 0.56), rng.uniform(-0.24, 0.24), rng.uniform(0.025, 0.045))
                 for _ in range(rng.integers(cfg.n_table_blobs[0], cfg.n_table_blobs[1] + 1))]
        self.scene = dict(target=target, liquid=liquid, holes=holes, blobs=blobs, rng=rng)
        return self.scene

    def _world_points(self, data, depth):
        """World coordinates (H, W, 3) of every pixel from the linear depth image."""
        h, w = depth.shape
        v, u = np.mgrid[0:h, 0:w]
        x = (u + 0.5 - w / 2) / self.focal * depth
        y = -(v + 0.5 - h / 2) / self.focal * depth
        pc = np.stack([x, y, -depth], -1)
        R = data.cam_xmat[self.cam_id].reshape(3, 3)
        return pc @ R.T + data.cam_xpos[self.cam_id]

    def render(self, data, temps, emissivity, t_env=22.0, rng=None, scene=None):
        """Returns rgb (H, W, 3) uint8, depth (H, W) m, thermal (h, w) in [0, 1] (main's min-max), raw (h, w) counts."""
        rng = rng or np.random.default_rng()
        r, cam, cfg, th = self.r, self.camera, self.cfg, self.cfg.thermal
        scene = scene or dict(liquid=None, holes={}, blobs=[], rng=None)

        r.update_scene(data, camera=cam, scene_option=self.show_all)
        rgb = r.render().copy()

        r.enable_depth_rendering()
        r.update_scene(data, camera=cam, scene_option=self.show_all)
        depth_true = r.render().copy()
        r.update_scene(data, camera=cam, scene_option=self.hide_transparent)
        depth_behind = r.render().copy()
        r.disable_depth_rendering()

        r.enable_segmentation_rendering()
        r.update_scene(data, camera=cam, scene_option=self.show_all)
        seg = r.render()[..., 0].copy()
        r.disable_segmentation_rendering()

        # Depth on transparent objects: sees through them, glass_holes of the pixels are holes
        on_glass = np.isin(seg, self.transparent_geoms)
        depth = depth_true.copy()
        depth[on_glass] = depth_behind[on_glass]
        depth[on_glass & (rng.random(depth.shape) < cfg.glass_holes)] = 0.0

        # Lever C: no reading on IR-absorbing / specular objects, glass-sized hole patches on the table, edge dropout
        need_xyz = scene["blobs"] or scene["liquid"] is not None
        xyz = self._world_points(data, depth_true) if need_xyz else None
        srng = scene["rng"]
        for g, frac in scene["holes"].items():
            mask = seg == g
            depth[mask & (srng.random(depth.shape) < frac)] = 0.0
        if scene["blobs"]:
            on_table = np.isin(seg, self.table_geoms)
            near = np.zeros_like(on_table)
            for bx, by, br in scene["blobs"]:
                near |= (xyz[..., 0] - bx) ** 2 + (xyz[..., 1] - by) ** 2 < br ** 2
            depth[on_table & near & (srng.random(depth.shape) < cfg.glass_holes)] = 0.0
        if cfg.edge_dropout > 0:
            edge = np.zeros(depth.shape, bool)
            edge[:, 1:] |= np.abs(np.diff(depth_true, axis=1)) > 0.03
            edge[1:] |= np.abs(np.diff(depth_true, axis=0)) > 0.03
            depth[edge & (srng.random(depth.shape) < cfg.edge_dropout)] = 0.0

        # Thermal rendering
        temp = np.full(seg.shape, t_env, float)
        eta = np.ones(seg.shape)
        for gid in np.unique(seg[seg >= 0]):
            name = self.model.body(self.geom_body[gid]).name
            mask = seg == gid
            temp[mask] = temps.get(name, t_env)
            eta[mask] = emissivity.get(name, 0.95)
        if scene["liquid"] is not None:   # glass wall below the liquid line takes the liquid's temperature
            height, t_liq = scene["liquid"]
            target = scene["target"]
            body = self.model.body(target).id
            wall = np.isin(seg, np.flatnonzero(self.geom_body == body))
            wall &= xyz[..., 2] < data.xpos[body][2] + height
            temp[wall] = 0.8 * t_liq + 0.2 * temps[target]
        rad = eta * band_radiance_map(temp) + (1 - eta) * band_radiance(t_env)

        # Camera effects
        th_h, th_w = self.thermal_size
        ys = np.linspace(0, rad.shape[0] - 1, th_h).astype(int)
        xs = np.linspace(0, rad.shape[1] - 1, th_w).astype(int)
        if th.pipeline == "raw":   # 14-bit counts: gain / offset, blur, noise in counts
            raw = gaussian_blur(th.gain * rad[np.ix_(ys, xs)] + th.offset, th.blur_sigma)
            raw = np.clip(raw + rng.normal(0, th.noise_counts, raw.shape), 0, RAW_MAX)
            thermal = (raw - raw.min()) / (np.ptp(raw) + 1e-9)
            return rgb, depth, thermal, raw

        small = rad[np.ix_(ys, xs)]
        pad = np.pad(small, 1, mode="edge")

        small = (pad[:-2, :-2] + 2 * pad[:-2, 1:-1] + pad[:-2, 2:] + 2 * pad[1:-1, :-2] + 4 * pad[1:-1, 1:-1] + 2 * pad[1:-1, 2:] + pad[2:, :-2] + 2 * pad[2:, 1:-1] + pad[2:, 2:]) / 16
        small = small + rng.normal(0, th.noise * small.mean(), small.shape)
        thermal = (small - small.min()) / (np.ptp(small) + 1e-9)
        raw = np.clip(th.gain * small + th.offset, 0, RAW_MAX)
        return rgb, depth, thermal, raw
