"""
Figures of the fusion architecture with the ablations marked where they act (fusion/ABLATIONS.md).
  python fusion/ablations/make_figures.py          -> fusion/ablations/fig_*.png and .pdf
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

OUT = Path(__file__).resolve().parent
INK, INK2, MUTED, SURF = "#1d1d1b", "#52514e", "#8a8984", "#ffffff"
GRAY_BG, GRAY_EDGE = "#f1f0ec", "#c9c7c0"
BLUE, BLUE_BG = "#2a78d6", "#eaf2fc"            # segmentation pre-training (S)
ORANGE, ORANGE_BG = "#d9572a", "#fdeee7"        # policy (P)
TEAL, TEAL_BG = "#138a63", "#e3f4ed"            # the reliability gate
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})


class Canvas:
    def __init__(self, w, h, scale=0.1):
        self.fig = plt.figure(figsize=(w * scale, h * scale), dpi=200, facecolor=SURF)
        self.ax = self.fig.add_axes([0, 0, 1, 1])
        self.ax.set_xlim(0, w)
        self.ax.set_ylim(0, h)
        self.ax.axis("off")

    def box(self, x, y, w, h, fc=GRAY_BG, ec=GRAY_EDGE, lw=1.0, r=1.2, ls="-", z=1):
        self.ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec,
                                         lw=lw, ls=ls, zorder=z))

    def text(self, x, y, s, size=10, color=INK, weight="normal", ha="center", va="center", z=5, **kw):
        self.ax.text(x, y, s, fontsize=size, color=color, weight=weight, ha=ha, va=va, zorder=z, **kw)

    def node(self, x, y, w, h, title, sub=None, fc=GRAY_BG, ec=GRAY_EDGE, tcolor=INK, lw=1.0, ls="-", size=10):
        self.box(x, y, w, h, fc, ec, lw=lw, ls=ls, z=2)
        if sub:
            self.text(x + w / 2, y + h * 0.64, title, size=size, color=tcolor, weight="bold")
            self.text(x + w / 2, y + h * 0.3, sub, size=size - 2, color=INK2)
        else:
            self.text(x + w / 2, y + h / 2, title, size=size, color=tcolor, weight="bold")

    def arrow(self, x1, y1, x2, y2, color=MUTED, lw=1.3, ls="-", rad=0.0, head=10):
        self.ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=head, color=color,
                                          lw=lw, ls=ls, connectionstyle=f"arc3,rad={rad}", zorder=3))

    def path(self, pts, color=MUTED, lw=1.3, ls="-"):
        xs, ys = zip(*pts)
        self.ax.plot(xs[:-1] + (xs[-1],), ys[:-1] + (ys[-1],), color=color, lw=lw, ls=ls, zorder=3,
                     solid_capstyle="round")

    def tag(self, x, y, label, color):
        """An ablation marker: a filled pill with the ablation id."""
        w = 1.6 + 1.25 * len(label)
        self.box(x - w / 2, y - 1.6, w, 3.2, fc=color, ec="none", r=1.6, z=6)
        self.text(x, y, label, size=8.5, color="white", weight="bold", z=7)

    def save(self, name):
        for ext in ("png", "pdf"):
            self.fig.savefig(OUT / f"{name}.{ext}", dpi=200, facecolor=SURF)
        plt.close(self.fig)
        print("wrote", OUT / f"{name}.png")


# ---------------------------------------------------------------------- figure 1: two stages
def fig_architecture():
    c = Canvas(170, 100)
    c.text(3, 96.5, "Learned attention weighting: where each ablation acts", size=15, weight="bold", ha="left")
    c.text(3, 92.8, "Blue tags S1–S6 change the segmentation pre-training; orange tags P1–P11 change the policy. "
           "Grey: unchanged parts.", size=9.5, color=INK2, ha="left")

    # ---------------- stage 1
    c.box(2, 3, 62, 86, fc=BLUE_BG, ec="none", r=2.5)
    c.text(5, 86, "Stage 1 · segmentation pre-training", size=11.5, weight="bold", color=BLUE, ha="left")
    c.text(5, 82.6, "fusion/train_seg.py · per-frame glass masks", size=8.5, color=INK2, ha="left")
    c.node(6, 70, 54, 9, "Sim frames + real images", "L0_bench (see-through, crossover) · RGB-T Glass · TRansPose …")
    c.node(6, 58, 54, 9, "Training corruptions", "thermal contrast · depth holes · see-through depth · dropout")
    c.tag(10.5, 58, "S1", BLUE)
    c.tag(16.5, 58, "S2", BLUE)
    c.arrow(33, 70, 33, 67.3)
    c.box(6, 21, 54, 34, fc=SURF, ec=GRAY_EDGE, r=1.8)
    c.text(9, 52, "Fusion network", size=9.5, weight="bold", color=INK2, ha="left")
    for i, m in enumerate(("RGB", "depth", "thermal")):
        c.node(9 + i * 17, 42, 14.5, 7, f"{m} encoder", size=8.5)
    c.arrow(33, 58, 33, 49.3)
    c.node(9, 31.5, 48.5, 7.5, "Reliability gate", "weight per sensor and patch, ẑ = Σ w·z", fc=TEAL_BG, ec=TEAL,
           tcolor=TEAL)
    c.tag(53, 39, "S5", BLUE)
    for i in range(3):
        c.arrow(16.25 + i * 17, 42, 16.25 + i * 17, 39.2, head=7)
    c.node(9, 23, 48.5, 6, "Decoder", size=9)
    c.arrow(33, 31.5, 33, 29.2, head=7)
    c.node(6, 11, 25, 7, "Glass mask", "segmentation loss", size=9)
    c.node(35, 11, 25, 7, "Gate loss", "known-bad sensor → low", fc=TEAL_BG, ec=TEAL, tcolor=TEAL, size=9)
    c.tag(41, 18.2, "S3", BLUE)
    c.tag(47.5, 18.2, "S6", BLUE)
    c.arrow(20, 23, 18.5, 18.2, head=7)
    c.path([(57.5, 35), (59, 35), (59, 21)], color=MUTED)
    c.arrow(59, 21, 59, 18.3, head=7)
    c.node(6, 4.5, 54, 4.5, "checkpoint best.pt  (mix_ws; S4 = see-through-aware)", fc=SURF, ec=BLUE, size=8.5)

    # ---------------- stage 2
    c.box(68, 3, 100, 86, fc=ORANGE_BG, ec="none", r=2.5)
    c.text(71, 86, "Stage 2 · Diffusion Policy with the fusion encoder", size=11.5, weight="bold", color=ORANGE,
           ha="left")
    c.text(71, 82.6, "scripts/train_fusion_policy.sh · fusion/policy_encoder.py · fusion/fusion_policy.py",
           size=8.5, color=INK2, ha="left")
    c.node(72, 70, 50, 9, "Observations (96 px, 2 frames)", "RGB · depth · thermal · joints (500 L0_bench demos)")
    c.tag(119, 79, "P10", ORANGE)
    c.text(117.3, 75.0, "no thermal", size=7.5, color=ORANGE)
    c.node(126, 70, 38, 9, "Episode flags (sim only)", "see-through · crossover", fc=SURF)
    c.node(72, 59.5, 50, 7, "Corruption of training frames", size=9, fc=SURF, ls="--")
    c.tag(119, 66.5, "P7", ORANGE)
    c.arrow(97, 70, 97, 66.8, head=7)

    c.box(72, 26, 66, 30, fc=SURF, ec=GRAY_EDGE, r=1.8)
    c.text(75, 53, "Fusion encoder (the stage 1 network)", size=9.5, weight="bold", color=INK2, ha="left")
    c.tag(128.5, 53, "P2", ORANGE)
    c.tag(134.5, 53, "P3", ORANGE)
    c.text(131.5, 49.3, "frozen", size=8, color=ORANGE)
    c.arrow(97, 59.5, 97, 56.2, head=8)
    for i, m in enumerate(("RGB", "depth", "thermal")):
        c.node(75 + i * 20, 43.5, 17, 6, f"{m} enc.", size=8.5)
    c.node(75, 34, 57, 7, "Reliability gate", "weights w", fc=TEAL_BG, ec=TEAL, tcolor=TEAL, size=9.5)
    c.tag(128, 41, "P5", ORANGE)
    for i in range(3):
        c.arrow(83.5 + i * 20, 43.5, 83.5 + i * 20, 41.2, head=7)
    c.node(75, 27.5, 57, 5, "Decoder → glass mask p", size=8.5)
    c.arrow(103.5, 34, 103.5, 32.7, head=6)
    # gate supervision from the flags
    c.path([(145, 70), (145, 37.5)], color=TEAL, ls="--")
    c.arrow(145, 37.5, 132.3, 37.5, color=TEAL, ls="--", head=8)
    c.tag(150, 50, "P6", ORANGE)
    c.text(150, 46.4, "weight", size=7.5, color=TEAL)
    c.text(150, 44.2, "labels", size=7.5, color=TEAL)
    # anchor
    c.node(141, 26, 24, 9, "Frozen copy", "of best.pt", fc=SURF, ec=ORANGE, ls="--", size=8.5)
    c.arrow(141, 30.5, 132.3, 30.5, color=ORANGE, ls="--", head=8)
    c.tag(153, 37, "P4", ORANGE)
    c.text(136.5, 28.3, "match", size=7, color=ORANGE)

    c.node(72, 16, 66, 7, "Readout", "ẑ average · ẑ on the glass · glass centre / size · weights", size=9.5)
    c.tag(126.5, 23, "P8", ORANGE)
    c.tag(132.5, 23, "P9", ORANGE)
    c.arrow(105, 26, 105, 23.2, head=8)
    c.node(72, 5, 40, 7.5, "Diffusion U-Net", "predicts 16 steps, runs 8", size=9.5)
    c.arrow(92, 16, 92, 12.7, head=8)
    c.node(116, 5, 22, 7.5, "Actions", "6 joints + gripper", size=9.5, fc=SURF)
    c.arrow(112, 8.75, 115.8, 8.75, head=8)

    # stage 1 -> stage 2: initialization
    c.path([(60, 6.75), (65, 6.75), (65, 41), (70, 41)], color=BLUE, lw=1.6)
    c.arrow(70, 41, 74.8, 41, color=BLUE, lw=1.6, head=10)
    c.tag(66, 50, "P1", ORANGE)
    c.tag(66, 32, "P11", ORANGE)
    c.text(66, 46.4, "init", size=8, color=BLUE)
    c.save("fig_architecture")


# ---------------------------------------------------------------------- figure 2: gate
def fig_gate():
    c = Canvas(170, 72)
    c.text(3, 68.5, "The reliability gate, one image patch (fusion_mode = weighted_sum)", size=14, weight="bold",
           ha="left")
    c.text(3, 64.8, "What each ablation removes or adds. The output ẑ is the input of the decoder and of the policy readout.",
           size=9.5, color=INK2, ha="left")
    xs = (8, 38, 68)
    for x, m in zip(xs, ("RGB", "depth", "thermal")):
        c.node(x, 50, 26, 8, f"{m} token z", "256 numbers")
    c.node(100, 50, 30, 8, "Cues, added to each token", "brightness · contrast · valid px", fc=SURF, ls="--")
    c.tag(127, 58, "S5", BLUE)
    c.arrow(105, 50, 94.5, 44.4, head=8, rad=-0.2)
    c.node(8, 36, 86, 8, "Joint attention over all sensors and patches", "compares the sensors with each other",
           fc=TEAL_BG, ec=TEAL, tcolor=TEAL)
    for x in xs:
        c.arrow(x + 13, 50, x + 13, 44.3, head=8)
    for x, m in zip(xs, ("rgb", "d", "t")):
        c.node(x, 23, 26, 8, f"score s_{m}", "bounded to ±3", fc=TEAL_BG, ec=TEAL, tcolor=TEAL, size=9.5)
        c.arrow(x + 13, 36, x + 13, 31.3, head=8)
    c.node(8, 10, 86, 8, "Softmax over sensors → weights w", "w_rgb + w_d + w_t = 1;  missing sensor → 0",
           fc=TEAL_BG, ec=TEAL, tcolor=TEAL)
    for x in xs:
        c.arrow(x + 13, 23, x + 13, 18.3, head=8)
    c.node(102, 10, 30, 8, "ẑ = Σ w · z", "uses the unmixed tokens", fc=SURF, ec=INK2)
    c.arrow(94, 14, 101.7, 14, head=8)
    # the switches, on the right
    sx = 138
    c.text(sx, 49, "Ablations here", size=10.5, weight="bold", ha="left")
    rows = [("P5", ORANGE, "fixed w = 1/3: no gate at all"),
            ("S5", BLUE, "no cues: the gate sees features only"),
            ("S6", BLUE, "no gate loss in pre-training"),
            ("S3", BLUE, "gate loss also on see-through"),
            ("P6", ORANGE, "gate loss from episode flags"),
            ("S1", BLUE, "no depth-hole corruption"),
            ("S2", BLUE, "see-through corruption")]
    for i, (t, col, s) in enumerate(rows):
        y = 44 - i * 5.2
        c.tag(sx + 2.6, y, t, col)
        c.text(sx + 6.5, y, s, size=8.3, color=INK2, ha="left")
    c.tag(91, 18.2, "P5", ORANGE)
    c.node(36, 1.5, 58, 5.5, "Gate loss on the scores: known-bad sensor → low", fc=SURF, ec=TEAL, ls="--", size=8.5)
    c.tag(41, 7.2, "S3", BLUE)
    c.tag(47, 7.2, "S6", BLUE)
    c.tag(53, 7.2, "P6", ORANGE)
    c.save("fig_gate")


# ---------------------------------------------------------------------- figure 3: readout
def fig_readout():
    c = Canvas(170, 70)
    c.text(3, 66.5, "What the policy reads from the encoder (readout) and why grasp precision may suffer",
           size=14, weight="bold", ha="left")
    c.text(3, 62.8, "Most failures of the fusion policy are missed grasps (median 2.5 cm from the glass): it finds the glass "
           "but may know its position only coarsely (hypothesis tested by P8 / P9).", size=9.5, color=INK2, ha="left")
    # the 6x6 fused grid
    gx, gy, cell = 8, 18, 6
    c.text(gx + 18, 57, "fused map ẑ: 6 × 6 patches", size=9.5, weight="bold")
    for i in range(6):
        for j in range(6):
            hot = (i, j) in {(3, 2), (3, 3), (2, 2)}
            c.ax.add_patch(Rectangle((gx + i * cell, gy + j * cell), cell, cell, fc=TEAL_BG if hot else GRAY_BG,
                                     ec=GRAY_EDGE, lw=0.8, zorder=2))
    c.text(gx + 18, 14.5, "one patch = 14 × 14 px of the 84 px crop", size=8.5, color=INK2)
    # the full-res mask
    mx = 56
    c.text(mx + 18, 57, "glass mask p: 84 × 84 px", size=9.5, weight="bold")
    c.ax.add_patch(Rectangle((mx, gy), 36, 36, fc=GRAY_BG, ec=GRAY_EDGE, lw=0.8, zorder=2))
    c.ax.add_patch(matplotlib.patches.Ellipse((mx + 20.5, gy + 17), 4.5, 7, fc=ORANGE, ec="none", alpha=0.85, zorder=3))
    c.text(mx + 18, 14.5, "sub-pixel centre available here", size=8.5, color=INK2)
    # parts
    px = 100
    c.text(px, 57, "readout parts → Linear(512) → U-Net conditioning", size=9.5, weight="bold", ha="left")
    parts = [("ẑ average (whole image)", "default", None),
             ("ẑ averaged on the glass", "default", "P8 off"),
             ("glass centre / size on the 6 × 6 grid", "default", "P8 off"),
             ("ẑ max on the glass", "default", "P8 off"),
             ("sensor weights", "default", None),
             ("glass centre, spread, peak at 84 px", "P9 on", "P9"),
             ("32 learned keypoints (spatial softmax)", "P9 on", "P9")]
    for i, (s, state, tag) in enumerate(parts):
        y = 50 - i * 5
        new = state.startswith("P9")
        c.box(px, y - 2, 52, 4, fc=ORANGE_BG if new else SURF, ec=ORANGE if new else GRAY_EDGE, ls="--" if new else "-",
              r=0.8, z=2)
        c.text(px + 1.5, y, s, size=8.5, color=INK, ha="left")
        if tag:
            c.tag(px + 57.5, y, tag, ORANGE)
    c.arrow(gx + 36.5, 36, mx - 0.5, 36, head=8)
    c.text((gx + 36.5 + mx) / 2, 38.5, "decoder", size=8, color=INK2)
    c.arrow(mx + 36.5, 36, px - 0.5, 36, head=8)
    c.text(4, 7.5, "P8 keeps only ẑ average + weights: does the glass-specific readout help at all?", size=9,
           color=INK2, ha="left")
    c.text(4, 3.5, "P9 adds the full-resolution position and keypoints: does grasp precision improve?",
           size=9, color=INK2, ha="left")
    c.save("fig_readout")


if __name__ == "__main__":
    fig_architecture()
    fig_gate()
    fig_readout()