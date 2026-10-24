"""
Renders RGB, depth, and simulated thermal using MuJoCo camera.
Author: Sadhana Kumar
"""
import mujoco
import numpy as np

H_PLANCK, C_LIGHT, K_B = 6.626e-34, 2.998e8, 1.381e-23
_WAVE = np.linspace(8e-6, 14e-6, 200) # low wave IR range of most thermal cameras
LIQUID_MIX = 0.8 # glass wall below the liquid surface: 0.8 * liquid + 0.2 * glass temperature

def band_radiance(temp_c):
    t = np.asarray(temp_c, float)[..., None] + 273.15
    spec = 2 * H_PLANCK * C_LIGHT**2 / _WAVE**5 / np.expm1(H_PLANCK * C_LIGHT / (_WAVE * K_B * t))
    return spec.sum(axis=-1) * (_WAVE[1] - _WAVE[0])

class Sensors:
    def __init__(self, model, camera="front_cam", height=240, width=320, thermal_size=(120, 160), glass_body="glass"):
        self.model, self.camera = model, camera
        self.r = mujoco.Renderer(model, height, width)
        self.thermal_size = thermal_size
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
        """liquid: None or (fill_level, liquid_temp). LWIR does not pass glass, so liquid shows only through the wall temperature."""
        rng = rng or np.random.default_rng()
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

        # Depth on glass
        on_glass = np.isin(seg, self.glass_geoms)
        depth = depth_true.copy()
        depth[on_glass] = depth_behind[on_glass]
        depth[on_glass & (rng.random(depth.shape) < 0.4)] = 0.0

        # Thermal rendering
        temp = np.full(seg.shape, t_env, float)
        eta = np.ones(seg.shape)
        for gid in np.unique(seg[seg >= 0]):
            name = self.model.body(self.geom_body[gid]).name
            mask = seg == gid
            temp[mask] = temps.get(name, t_env)
            eta[mask] = emissivity.get(name, 0.95)
        if liquid is not None:
            fill, t_liquid = liquid
            below = self.below_liquid(data, depth_true, on_glass, fill)
            temp[below] = LIQUID_MIX * t_liquid + (1 - LIQUID_MIX) * temps.get("glass", t_env)
        rad = eta * band_radiance(temp) + (1 - eta) * band_radiance(t_env)

        # Camera effects
        th, tw = self.thermal_size
        ys = np.linspace(0, rad.shape[0] - 1, th).astype(int)
        xs = np.linspace(0, rad.shape[1] - 1, tw).astype(int)
        small = rad[np.ix_(ys, xs)]
        pad = np.pad(small, 1, mode="edge")

        small = (pad[:-2, :-2] + 2 * pad[:-2, 1:-1] + pad[:-2, 2:] + 2 * pad[1:-1, :-2] + 4 * pad[1:-1, 1:-1] + 2 * pad[1:-1, 2:] + pad[2:, :-2] + 2 * pad[2:, 1:-1] + pad[2:, 2:]) / 16
        small = small + rng.normal(0, 0.002 * small.mean(), small.shape)
        thermal = (small - small.min()) / (np.ptp(small) + 1e-9)
        
        return rgb, depth, thermal
