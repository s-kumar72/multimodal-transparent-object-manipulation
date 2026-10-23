"""
Export the pretrained thermal encoder for Diffusion Policy (state_dict keys = DP's ResNet-18 without fc).

  python -m encoder.export <ckpt> <out.pth> [--in_chans 3]
"""
import argparse
import torch
from encoder.model import make_thermal_resnet18


def encoder_state_dict(ckpt_path, in_chans=1):
    """Encoder weights from a train.py checkpoint. in_chans > 1 repeats conv1 / in_chans, which gives the same
    features on a channel-repeated thermal input (stopgap for DP configs that feed thermal as 3 channels)."""
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)["model"]
    enc = {k.removeprefix("encoder."): v for k, v in sd.items() if k.startswith("encoder.")}
    if in_chans != 1:
        enc["conv1.weight"] = enc["conv1.weight"].repeat(1, in_chans, 1, 1) / in_chans
    return enc


def load_thermal_encoder(path):
    """DP-format ResNet-18 (GroupNorm, fc = Identity, 512-d output) loaded with exported weights."""
    sd = torch.load(path, map_location="cpu")
    net = make_thermal_resnet18("scratch", in_chans=sd["conv1.weight"].shape[1])
    net.load_state_dict(sd)
    return net


def load_into_dp_obs_encoder(obs_encoder, path, key="thermal"):
    """Swap the `key` backbone of a DP MultiImageObsEncoder (share_rgb_model=False) for the pretrained one.
    Call before DP builds its optimizer and EMA copy."""
    obs_encoder.key_model_map[key] = load_thermal_encoder(path).to(obs_encoder.device)
    return obs_encoder


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("out")
    ap.add_argument("--in_chans", type=int, default=1)
    args = ap.parse_args()
    enc = encoder_state_dict(args.ckpt, args.in_chans)
    torch.save(enc, args.out)
    print(f"saved {len(enc)} tensors (conv1 {tuple(enc['conv1.weight'].shape)}) to {args.out}")
