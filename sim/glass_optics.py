"""
The colour image of clear glass as an optical effect on the scene behind it (rgb_glass: optical; section 12 of
results/hard_scenario/bench_v2/README.md). The renderer's own way, a tinted solid blended at some alpha, shows a pale
grey body; real clear glass is the background seen through it, a little dimmed, with a line along its edges and
highlights where its surface mirrors a light.

Per pixel of a glass, from the scene rendered without it (B), its depth image and the lights:

    I = (1 - w_rim * e) * [T0 * (1 - F) * B(p + shift) + F * E] + w_rim * e * rho * E + 255 * k_s * S
    out = (1 - s) * B + s * I

  n      surface normal from the depth image (central differences inside the glass, so that a crease between two
         faces gets the normal of a rounded edge)
  F      Schlick reflectance, F0 + (1 - F0) (1 - cos)^5, cos between n and the ray to the camera
  E      what the glass reflects and refracts from elsewhere: the mean colour of B
  shift  the background behind the glass moved towards its edges by up to `shift` of its radius, times sin^2
  e      the edge, 0 to 1: the wall seen edge-on (sin above sin0), the outline of the glass, the creases between faces
  S      highlights, Blinn-Phong on n for every light of the scene, times the light's level
  s      strength: 1 shows the glass, 0 leaves B untouched pixel for pixel

Not modelled: the inner surfaces and the far wall, the thick base, caustics and shadows, liquid.
"""
import numpy as np

# The numbers (section 12 of the README has how they came about, 11/26):
#   T0, F0   0.92 and 0.08, as first listed: what two thin clean walls let through and reflect. With them the glass's
#            area is about 9 grey levels darker than a plain table of luminance 166, and the regression of real
#            glass on its surroundings (median luminance = 0.68 * surroundings + 43, 2376 windows) says -10 there.
#            (A stronger veil, F0 0.30 with T0 1, was tried in between: it fits the same regression but puts a smooth
#            patch over the glass, which a network that learned real glass counts against glass.)
#   k_s      highlights, from real images: the bright tail inside the glass in excess of its surroundings'
#            (p95 - median) is 18.7 grey levels on real plain surroundings; 1.0 gives 19.0.
#   w_rim    the edge line, from real images: the dark tail in excess of the surroundings' is -3.0 on real glass;
#            0.1 (with rho 0.6) gives -2.3.
# shift, sin0, shininess and rho are as first listed.
PARAMS = dict(T0=0.92, F0=0.08, shift=0.15, sin0=0.85, shininess=30.0, k_s=1.0, w_rim=0.1, rho=0.6)
CREASE = 0.15      # 1 - cos of the angle between neighbouring normals above which there is a crease (about 31 degrees)


def _points(depth, focal):
    """Camera-frame coordinates (x right, y up, -z forward) of every pixel."""
    h, w = depth.shape
    v, u = np.mgrid[0:h, 0:w]
    return np.stack([(u + 0.5 - w / 2) / focal * depth, -(v + 0.5 - h / 2) / focal * depth, -depth], -1)


def _diff(p, m, axis):
    """Derivative of p (h, w, 3) along an axis using only pixels of the mask m: central where both neighbours are
    in it, one-sided where one is. Also the forward difference alone (zero where the next pixel is outside)."""
    nxt, prv = np.roll(p, -1, axis), np.roll(p, 1, axis)
    m_n, m_p = m & np.roll(m, -1, axis), m & np.roll(m, 1, axis)
    edge = [slice(None)] * 2
    edge[axis] = -1
    m_n[tuple(edge)] = False
    edge[axis] = 0
    m_p[tuple(edge)] = False
    f, b = (nxt - p) * m_n[..., None], (p - prv) * m_p[..., None]
    both = (m_n & m_p)[..., None]
    return np.where(both, (f + b) / 2, f + b), f, m_n


def _unit(a):
    return a / np.maximum(np.linalg.norm(a, axis=-1, keepdims=True), 1e-12)


def terms(behind, depth, mask, focal, lights, params=PARAMS):
    """Everything of the formula that does not depend on k_s, w_rim, rho and s, on the bounding box of one glass.
    behind: (h, w, 3) uint8, the scene without the glass; depth: (h, w) m with the glass; mask: its pixels;
    lights: list of (direction to the light in the camera frame or None for the headlight, level)."""
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return None
    y0, y1, x0, x1 = max(ys.min() - 1, 0), min(ys.max() + 2, mask.shape[0]), max(xs.min() - 1, 0), min(xs.max() + 2, mask.shape[1])
    box = np.s_[y0:y1, x0:x1]
    m = mask[box]
    p = _points(depth, focal)[box]
    view = _unit(-p)
    du, fu, mu = _diff(p, m, 1)
    dv, fv, mv = _diff(p, m, 0)
    n = np.cross(du, dv)
    flat = np.linalg.norm(n, axis=-1) < 1e-12
    n = _unit(n)
    n = np.where(((n * view).sum(-1) < 0)[..., None], -n, n)
    n[flat] = view[flat]
    cos = np.clip((n * view).sum(-1), 0, 1)
    sin = np.sqrt(1 - cos ** 2)
    # the edge: grazing view, the outline, creases (forward normals, so that a crease marks one row of pixels)
    inner = m.copy()
    inner[1:] &= m[:-1]
    inner[:-1] &= m[1:]
    inner[:, 1:] &= m[:, :-1]
    inner[:, :-1] &= m[:, 1:]
    inner[0] = inner[-1] = False
    inner[:, 0] = inner[:, -1] = False
    n1 = np.cross(fu, fv)
    ok1 = (np.linalg.norm(n1, axis=-1) > 1e-12) & mu & mv
    n1 = _unit(n1)
    n1 = np.where(((n1 * view).sum(-1) < 0)[..., None], -n1, n1)
    crease = np.zeros(m.shape, bool)
    for axis in (0, 1):
        nb, okb = np.roll(n1, -1, axis), np.roll(ok1, -1, axis)
        crease |= ok1 & okb & (1 - (n1 * nb).sum(-1) > CREASE)
    edge = np.maximum(np.clip((sin - params["sin0"]) / (1 - params["sin0"]), 0, 1), (m & ~inner) | crease) * m
    # the background behind the glass, moved towards the edges
    radius = np.sqrt(m.sum() / np.pi)
    d = np.stack([n[..., 0], -n[..., 1]], -1)                       # image-plane direction of the normal (u right, v down)
    d = d / np.maximum(np.linalg.norm(d, axis=-1, keepdims=True), 1e-12)
    amount = params["shift"] * radius * sin ** 2
    vv, uu = np.mgrid[y0:y1, x0:x1].astype(float)
    us = np.clip(uu + amount * d[..., 0], 0, mask.shape[1] - 1)
    vs = np.clip(vv + amount * d[..., 1], 0, mask.shape[0] - 1)
    u0, v0 = np.floor(us).astype(int), np.floor(vs).astype(int)
    u1, v1 = np.minimum(u0 + 1, mask.shape[1] - 1), np.minimum(v0 + 1, mask.shape[0] - 1)
    a, b = (us - u0)[..., None], (vs - v0)[..., None]
    bf = behind.astype(np.float32)
    seen = (1 - b) * ((1 - a) * bf[v0, u0] + a * bf[v0, u1]) + b * ((1 - a) * bf[v1, u0] + a * bf[v1, u1])
    env = bf.reshape(-1, 3).mean(0)
    fres = (params["F0"] + (1 - params["F0"]) * (1 - cos) ** 5)[..., None]
    base = params["T0"] * (1 - fres) * seen + fres * env
    spec = np.zeros(m.shape, np.float32)
    for direction, level in lights:
        light = np.broadcast_to(np.array([0.0, 0.0, 1.0]) if direction is None else direction, view.shape)
        half = _unit(light + view)
        spec += level * np.clip((n * half).sum(-1), 0, 1) ** params["shininess"]
    return dict(box=box, mask=m, base=base.astype(np.float32), env=env.astype(np.float32), edge=edge.astype(np.float32),
                spec=spec * m)


def compose(behind, t, k_s, w_rim, rho, s=1.0):
    """The glass of `terms` t drawn into a copy of the box of `behind`: (h, w, 3) float32 in 0..255."""
    b = behind[t["box"]].astype(np.float32)
    w = (w_rim * t["edge"])[..., None]
    full = (1 - w) * t["base"] + w * rho * t["env"] + 255.0 * k_s * t["spec"][..., None]
    out = (1 - s) * b + s * np.clip(full, 0, 255)
    return np.where(t["mask"][..., None], out, b)


def optical_rgb(behind, depth, glasses, focal, lights, params=PARAMS):
    """behind with every glass drawn in. glasses: list of (mask, strength)."""
    rgb = behind.copy()
    for mask, s in glasses:
        if s <= 0:
            continue
        t = terms(behind, depth, mask, focal, lights, params)
        if t is not None:
            rgb[t["box"]] = np.rint(compose(behind, t, params["k_s"], params["w_rim"], params["rho"], s)).astype(np.uint8)
    return rgb
