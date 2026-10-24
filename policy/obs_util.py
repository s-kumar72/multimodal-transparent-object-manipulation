"""
Converts raw sim observations (uint8 HWC images, float joints) into Diffusion Policy inputs.
Shared by the training dataset and the eval env so both see identical inputs.
"""
import numpy as np

def to_policy_obs(raw, obs_meta):
    """raw: dict of (..., H, W, C) uint8 images / (..., D) vectors. obs_meta: shape_meta["obs"]."""
    out = {}
    for key, attr in obs_meta.items():
        x = raw[key]
        if attr.get("type", "low_dim") == "rgb":
            if x.shape[-1] == 1:
                x = np.repeat(x, 3, axis=-1)    # depth / thermal -> 3 channels for the ResNet encoder
            x = np.moveaxis(x, -1, -3).astype(np.float32) / 255
        else:
            x = x.astype(np.float32)
        out[key] = x
    return out
