"""
Renders RGB, depth, and simulated thermal using MuJoCo camera.
Author: Sadhana Kumar
"""
import mujoco
import numpy as np

H_PLANCK, C_LIGHT, K_B = 6.626e-34, 2.998e8, 1.381e-23
_WAVE = np.linspace(8e-6, 14e-6, 200) # low wave IR range of most thermal cameras

def band_radiance(temp_c):
    t = np.asarray(temp_c, float)[..., None] + 273.15
    spec = 2 * H_PLANCK * C_LIGHT**2 / _WAVE**5 / np.expm1(H_PLANCK * C_LIGHT / (_WAVE * K_B * t))
    return spec.sum(axis=-1) * (_WAVE[1] - _WAVE[0])

class Sensors:
    TRANSPARENT_GROUP = 5 # geom group of transparent geoms in the scene XML (hidden by default MjvOption)

    def __init__(self, model, camera="front_cam", height=240, width=320, thermal_size=(120, 160)):
        self.model, self.camera = model, camera
        self.r = mujoco.Renderer(model, height, width)
        self.thermal_size = thermal_size
        self.transparent_geoms = np.flatnonzero(model.geom_group == self.TRANSPARENT_GROUP)
        self.geom_body = model.geom_bodyid.copy()
        self.show_all = mujoco.MjvOption()
        self.show_all.geomgroup[self.TRANSPARENT_GROUP] = 1
        self.hide_transparent = mujoco.MjvOption()

    def render(self, data, temps, emissivity, t_env=22.0, rng=None):
        rng = rng or np.random.default_rng()
        r, cam = self.r, self.camera

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

        # Depth on transparent objects: sees through them, 40% of the pixels are holes
        on_glass = np.isin(seg, self.transparent_geoms)
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
