"""
Renders RGB, depth, and simulated thermal using MuJoCo camera.
Thermal parameters (emissivity default, liquid mix, raw gain/offset, resolution, blur, noise) come from a thermal config.
Author: Sadhana Kumar
"""
import mujoco
import numpy as np

from config import load_thermal_config

H_PLANCK, C_LIGHT, K_B = 6.626e-34, 2.998e8, 1.381e-23
_WAVE = np.linspace(8e-6, 14e-6, 200) # low wave IR range of most thermal cameras
RAW_MAX = 16383 # 14-bit raw counts

def band_radiance(temp_c):
    t = np.asarray(temp_c, float)[..., None] + 273.15
    spec = 2 * H_PLANCK * C_LIGHT**2 / _WAVE**5 / np.expm1(H_PLANCK * C_LIGHT / (_WAVE * K_B * t))
    return spec.sum(axis=-1) * (_WAVE[1] - _WAVE[0])

def blur(img, sigma=None):
    """sigma None: the original 3x3 binomial blur, else Gaussian blur (sigma in pixels). Edges are padded."""
    if sigma is None:
        pad = np.pad(img, 1, mode="edge")
        return (pad[:-2, :-2] + 2 * pad[:-2, 1:-1] + pad[:-2, 2:] + 2 * pad[1:-1, :-2] + 4 * pad[1:-1, 1:-1] + 2 * pad[1:-1, 2:] + pad[2:, :-2] + 2 * pad[2:, 1:-1] + pad[2:, 2:]) / 16
    r = max(1, int(np.ceil(3 * sigma)))
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()
    h, w = img.shape
    pad = np.pad(img, r, mode="edge")
    rows = sum(k[i] * pad[:, i:i + w] for i in range(len(k)))
    return sum(k[i] * rows[i:i + h] for i in range(len(k)))

class Sensors:
    def __init__(self, model, camera="front_cam", height=240, width=320, glass_body="glass", thermal=None):
        self.model, self.camera = model, camera
        self.r = mujoco.Renderer(model, height, width)
        self.th = thermal or load_thermal_config()
        self.glass_id = model.body(glass_body).id
        self.glass_geoms = np.flatnonzero(model.geom_bodyid == self.glass_id)  # glass + liquid
        self.glass_half = model.geom(glass_body).size[1]
        self.cam_id = model.camera(camera).id
        self.geom_body = model.geom_bodyid.copy()
        model.geom_group[self.glass_geoms] = 5
        self.show_all = mujoco.MjvOption()
        self.show_all.geomgroup[5] = 1
        self.hide_glass = mujoco.MjvOption()

    def below_liquid(self, data, depth, mask, fill):
        """Glass pixels whose 3D point (true depth + camera intrinsics) is below the liquid surface."""
        h, w = depth.shape
        f = 0.5 * h / np.tan(np.radians(self.model.cam_fovy[self.cam_id]) / 2)
        v, u = np.nonzero(mask)
        z = depth[v, u]
        p_cam = np.stack([(u + 0.5 - w / 2) * z / f, -(v + 0.5 - h / 2) * z / f, -z], 1)  # MuJoCo camera looks along -z
        p_world = data.cam_xpos[self.cam_id] + p_cam @ data.cam_xmat[self.cam_id].reshape(3, 3).T
        p_glass = (p_world - data.xpos[self.glass_id]) @ data.xmat[self.glass_id].reshape(3, 3)
        out = np.zeros_like(mask)
        out[v, u] = p_glass[:, 2] < self.glass_half * (2 * fill - 1)
        return out

    def render(self, data, temps, emissivity, t_env=22.0, rng=None, liquid=None):
        """liquid: None or (fill_level, liquid_temp). LWIR does not pass glass, so liquid shows only through the wall temperature.
        Returns rgb, depth (m), thermal (per-frame min-max, [0, 1]) and thermal raw counts (uint16) at the thermal resolution."""
        rng = rng or np.random.default_rng()
        rgb, depth_true, depth_behind, seg = self.geometry(data)

        # Depth on glass
        on_glass = np.isin(seg, self.glass_geoms)
        depth = depth_true.copy()
        depth[on_glass] = depth_behind[on_glass]
        depth[on_glass & (rng.random(depth.shape) < 0.4)] = 0.0

        # Thermal rendering
        below = self.below_liquid(data, depth_true, on_glass, liquid[0]) if liquid is not None else None
        raw = self.thermal_raw(seg, temps, emissivity, t_env, rng, below, liquid[1] if liquid is not None else None)

        # Per-frame min-max (v0 behavior), kept until the shared normalization replaces it
        thermal = (raw - raw.min()) / (float(np.ptp(raw)) + 1e-9)

        return rgb, depth, thermal, raw

    def geometry(self, data):
        """rgb, depth with glass, depth behind the glass, segmentation (geom ids) at render size."""
        r, cam = self.r, self.camera
        r.update_scene(data, camera=cam, scene_option=self.show_all)
        rgb = r.render().copy()

        r.enable_depth_rendering()
        r.update_scene(data, camera=cam, scene_option=self.show_all)
        depth_true = r.render().copy()
        r.update_scene(data, camera=cam, scene_option=self.hide_glass)
        depth_behind = r.render().copy()
        r.disable_depth_rendering()

        r.enable_segmentation_rendering()
        r.update_scene(data, camera=cam, scene_option=self.show_all)
        seg = r.render()[..., 0].copy()
        r.disable_segmentation_rendering()
        return rgb, depth_true, depth_behind, seg

    def to_thermal_res(self, img):
        """Render-size image -> thermal resolution, sampled like the thermal image (e.g. the glass mask)."""
        h, w = self.th["resolution"]
        ys = np.linspace(0, img.shape[0] - 1, h).astype(int)
        xs = np.linspace(0, img.shape[1] - 1, w).astype(int)
        return img[np.ix_(ys, xs)]

    def thermal_raw(self, seg, temps, emissivity, t_env, rng, below=None, t_liquid=None):
        """Raw thermal counts (uint16) from a segmentation image, so thermal can be re-made for new parameters
        without rendering again (calibration). below: glass pixels under the liquid surface."""
        th = self.th
        temp = np.full(seg.shape, t_env, float)
        eta = np.ones(seg.shape)
        for gid in np.unique(seg[seg >= 0]):
            name = self.model.body(self.geom_body[gid]).name
            mask = seg == gid
            temp[mask] = temps.get(name, t_env)
            eta[mask] = emissivity.get(name, th["emis"]["other"])
        if below is not None:
            temp[below] = th["liquid_mix"] * t_liquid + (1 - th["liquid_mix"]) * temps.get("glass", t_env)
        # Planck integral only for the few distinct temperatures in the frame (same values, much faster)
        temp_u, inv = np.unique(temp, return_inverse=True)
        rad = eta * band_radiance(temp_u)[inv].reshape(temp.shape) + (1 - eta) * band_radiance(t_env)

        # Camera effects, in raw counts like a radiometric thermal camera
        gain, offset = th["raw"]["gain"], th["raw"]["offset"]
        counts = gain * rad + offset
        small = blur(self.to_thermal_res(counts), th["blur_sigma_px"])
        small = small + rng.normal(0, th["noise"]["rel"] * (small.mean() - offset) + th["noise"]["counts"], small.shape)
        return np.clip(np.round(small), 0, RAW_MAX).astype(np.uint16)
