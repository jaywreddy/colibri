"""Simulate the moiré cells: draw the real geometry, then integrate it by eye.

No analytic fringe model. Each panel rasterises the SAME rectangles the mask
writer will emit, at a pitch fine enough to resolve the gratings themselves,
and then block-averages to the eye's 87 um integration cell. The fringes that
appear are therefore the ones the geometry actually produces, not the ones the
beat formula says it should — which is the only way this is a check rather than
an illustration.

    uv run --directory backend python ../tools/dev/render_moire_preview.py OUTDIR
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "backend"))

from app import witness_moire as wm  # noqa: E402

BG = (13, 17, 19)
FG = (214, 224, 227)
DIM = (150, 165, 172)
GOLD = np.array([0.84, 0.66, 0.29], np.float32)
EYE_UM = 87.0


def font(sz: int = 13):
    try:
        return ImageFont.truetype("arialbd.ttf", sz)
    except Exception:
        return ImageFont.load_default()


def _michelson(duty: float) -> float:
    """Michelson contrast of the (2,3) beat at a given duty."""
    def amp(k, c):
        x = math.pi * k * c
        return abs(c * math.sin(x) / x) if x else 0.0
    return (2 * amp(2, duty) * amp(3, duty)) / (1 - duty) ** 2


def raster_raw(art, cx, cy, w, h, px_um=1.0):
    """Geometry as written, with no eye integration."""
    return raster(art, cx, cy, w, h, px_um=px_um, eye_um=px_um)


def raster(art, cx, cy, w, h, px_um=2.0, eye_um=EYE_UM):
    """Rasterise a cell's metal at ``px_um``, then integrate to the eye cell."""
    nx, ny = int(w / px_um), int(h / px_um)
    g = np.zeros((ny, nx), np.float32)
    x0, y1 = cx - w / 2.0, cy + h / 2.0

    def paint(rx0, rx1, ry0, ry1):
        a = max(0, int((rx0 - x0) / px_um)); b = min(nx, int(math.ceil((rx1 - x0) / px_um)))
        c = max(0, int((y1 - ry1) / px_um)); d = min(ny, int(math.ceil((y1 - ry0) / px_um)))
        if b > a and d > c:
            g[c:d, a:b] = 1.0

    for rx0, rx1, ry0, ry1 in np.asarray(art.front, dtype=np.float64):
        paint(rx0, rx1, ry0, ry1)
    for pv in art.polys:
        pv = np.asarray(pv, dtype=np.float64)
        # Scanline fill of a convex polygon, which every clipped rotated line is.
        ys = np.arange(ny) * px_um
        yy = y1 - ys - px_um / 2
        for j, y in enumerate(yy):
            xs = []
            n = len(pv)
            for i in range(n):
                a_, b_ = pv[i], pv[(i + 1) % n]
                if (a_[1] > y) != (b_[1] > y):
                    t = (y - a_[1]) / (b_[1] - a_[1])
                    xs.append(a_[0] + t * (b_[0] - a_[0]))
            if len(xs) >= 2:
                lo, hi = min(xs), max(xs)
                a = max(0, int((lo - x0) / px_um)); b = min(nx, int(math.ceil((hi - x0) / px_um)))
                if b > a:
                    g[j, a:b] = 1.0
    k = max(1, int(round(eye_um / px_um)))
    m, n = g.shape[0] - g.shape[0] % k, g.shape[1] - g.shape[1] % k
    return g[:m, :n].reshape(m // k, k, n // k, k).mean(axis=(1, 3))


def tile(cov, px=260, aspect=1.0, stretch=1.0):
    """Metal coverage -> a gold image. More metal is brighter in reflection.

    ``stretch`` amplifies the modulation about the panel mean. Used only for
    B-HARM, whose whole point is a beat at 3.5% Michelson contrast: real, well
    above the ~0.5% threshold at 9 cycles/deg, and invisible in an 8-bit preview
    at true contrast. The panel says when it has been stretched.
    """
    if stretch != 1.0:
        m = float(np.mean(cov))
        cov = np.clip(m + (cov - m) * stretch, 0, 1)
    img = np.clip(cov, 0, 1)[..., None] * GOLD
    im = Image.fromarray((np.clip(img, 0, 1) ** (1 / 2.2) * 255).astype(np.uint8))
    return im.resize((int(px * aspect), px), Image.LANCZOS)


def strip(tiles, labels, title, sub=""):
    pad = 4
    w = sum(t.width + pad for t in tiles) + pad
    hh = max(t.height for t in tiles)
    im = Image.new("RGB", (w, hh + pad * 2 + (56 if sub else 32)), BG)
    d = ImageDraw.Draw(im)
    x = pad
    for t, l in zip(tiles, labels):
        im.paste(t, (x, pad))
        d.text((x + 2, hh + pad + 3), l, fill=DIM, font=font(12))
        x += t.width + pad
    d.text((pad, hh + pad + 19), title, fill=FG, font=font(13))
    if sub:
        d.text((pad, hh + pad + 36), sub, fill=DIM, font=font(11))
    return im


def stack(rows, path):
    w = max(r.width for r in rows)
    im = Image.new("RGB", (w, sum(r.height + 6 for r in rows) + 6), BG)
    y = 6
    for r in rows:
        im.paste(r, (0, y))
        y += r.height + 6
    im.save(path)
    print("  saved", path, im.size)


def main() -> int:
    out = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
    out.mkdir(parents=True, exist_ok=True)
    rows = []

    print("beat ladder ...")
    tiles, labs = [], []
    for b in (500.0, 1000.0, 1635.0, 3000.0):
        w = max(6000.0, 5.0 * b)
        a = wm.build_beat(0, 0, w, 8000.0, beat_um=b)
        tiles.append(tile(raster(a, 0, 0, w, 8000.0), 150, aspect=min(2.6, w / 8000.0)))
        labs.append(f"{b:g} um  d={a.stats['delta_um']:.2f}")
    rows.append(strip(tiles, labs, "B-BEAT  two pitches on ONE plane",
                      "static fringes, no bond. Fringe count inverts back to the "
                      "pitch error with p/delta = 24.7x gain"))

    print("rotation ...")
    tiles, labs = [], []
    for ang in (1.0, 2.0, 4.0, 8.0):
        a = wm.build_rotation_beat(0, 0, 9000.0, 8000.0, angle_deg=ang)
        tiles.append(tile(raster(a, 0, 0, 9000.0, 8000.0), 150, aspect=1.12))
        labs.append(f"{ang:g} deg -> {a.stats['beat_um']:.0f} um")
    rows.append(strip(tiles, labs, "B-ROT  equal pitch, rotated",
                      "p/(2 sin(a/2)). Fringes run ALONG the lines, not across "
                      "them -- which is why rotation moire looks nothing like pitch moire"))

    print("harmonic ...")
    tiles, labs = [], []
    for c in (0.42, 0.50, 0.58):
        a = wm.build_harmonic(0, 0, 10000.0, 8000.0, duty=c)
        cov = raster(a, 0, 0, 10000.0, 8000.0)
        tiles.append(tile(cov, 150, aspect=1.25, stretch=12.0))
        mich = _michelson(c)
        labs.append(f"duty {c:.2f}   (2,3) at {mich*100:.1f}%")
    rows.append(strip(tiles, labs,
                      "B-HARM  44 um screen over the 63.5 um carrier  "
                      "[contrast stretched 12x]",
                      "the (2,3) beat is 559 um = 9.4 cycles/deg, and its "
                      "contrast is EXACTLY ZERO at 50% duty -- even harmonics "
                      "vanish there. Bias the process and it appears"))

    print("crossed + contrast ...")
    tiles, labs = [], []
    for c in (0.25, 0.50, 0.75):
        a = wm.build_beat_contrast(0, 0, 8200.0, 8000.0, duty=c)
        tiles.append(tile(raster(a, 0, 0, 8200.0, 8000.0), 150, aspect=1.03))
        labs.append(f"duty {c:.2f}  mean {a.stats['coverage_anti']:.2f}")
    for px_, py_ in ((20.0, 20.0), (20.0, 25.0)):
        a = wm.build_crossed(0, 0, 1200.0, 1200.0, period_x_um=px_, period_y_um=py_)
        # NOT eye-integrated: at 87 um a 20 um lattice averages to a flat tone,
        # which is correct and tells you nothing. The structure is the point.
        g = raster_raw(a, 0, 0, 1200.0, 1200.0, px_um=0.5)
        tiles.append(tile(g, 150))
        labs.append(f"cross {px_:g}/{py_:g} um  [1.2 mm, not integrated]")
    rows.append(strip(tiles, labs, "B-CONT duty, and D-CROSS 2-D",
                      "low duty is brighter at similar fringe contrast; the "
                      "crossed pair is the cheapest check of the union identity"))
    stack(rows, out / "witness_moire.png")

    print("swatches ...")
    rows = []
    for spread in (1.20, 1.45, 1.90):
        tiles, labs = [], []
        for bp in (4.0, 5.0, 6.5, 8.0):
            a = wm.build_swatch(0, 0, 6000.0, 4000.0, base_period_um=bp,
                                spread=spread)
            tiles.append(tile(raster(a, 0, 0, 6000.0, 4000.0, px_um=0.5), 110,
                              aspect=1.5))
            ok = "" if a.stats["all_printable"] else "  UNPRINTABLE"
            labs.append(f"base {bp:g} um{ok}")
        rows.append(strip(tiles, labs, f"D-SWATCH  spread {spread:.2f}",
                          "the whole hue ladder side by side -- this one "
                          "instrument replaced eight portrait sweeps"))
    stack(rows, out / "witness_swatch.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
