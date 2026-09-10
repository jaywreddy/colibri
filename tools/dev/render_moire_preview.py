"""Simulate the moiré cells from the plate AS BUILT.

Two figures, both driven by ``export_witness.doe_cells()`` so they cannot drift
from the mask: A — the pitch-beat and rotation ladders at their built widths;
B — B-HARM at true contrast with a column-mean profile under each panel (the
result is a 559 um sine present at 0.42 and 0.58 and absent at 0.50), B-CONT
with its transmission and reflection means, and the crossed lattices raw.

No analytic fringe model: each panel rasterises the rectangles the writer emits
and block-averages to the 87 um eye cell, so the fringes are the geometry's,
not the formula's.

Being driven by the plate cuts both ways: the 2026-09-10 plate gave the moire
ladders' area to the production dies, so most of these panels have no cell to
build and are SKIPPED (named on stdout) rather than drawn from a formula the
mask does not carry. Restore a ladder in export_witness.doe_cells and its figure
comes back. The builders themselves are still in witness_moire/witness_cells.

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
from app.witness_geom import BOX_CARRIER_UM  # noqa: E402
from app.export_witness import doe_cells  # noqa: E402

BG = (13, 17, 19)
FG = (214, 224, 227)
DIM = (150, 165, 172)
WARN = (224, 138, 114)
GOLD = np.array([0.84, 0.66, 0.29], np.float32)
EYE_UM = 87.0


def font(sz=12):
    try:
        return ImageFont.truetype("arialbd.ttf", sz)
    except Exception:
        return ImageFont.load_default()


def raster(art, cx, cy, w, h, px_um=2.0, eye_um=EYE_UM):
    nx, ny = int(w / px_um), int(h / px_um)
    g = np.zeros((ny, nx), np.float32)
    x0, y1 = cx - w / 2.0, cy + h / 2.0
    for rx0, rx1, ry0, ry1 in np.asarray(art.front, dtype=np.float64):
        a = max(0, int((rx0 - x0) / px_um)); b = min(nx, int(math.ceil((rx1 - x0) / px_um)))
        c = max(0, int((y1 - ry1) / px_um)); d = min(ny, int(math.ceil((y1 - ry0) / px_um)))
        if b > a and d > c:
            g[c:d, a:b] = 1.0
    for pv in art.polys:
        pv = np.asarray(pv, dtype=np.float64)
        for j in range(ny):
            y = y1 - (j + 0.5) * px_um
            xs = []
            n = len(pv)
            for i in range(n):
                a_, b_ = pv[i], pv[(i + 1) % n]
                if (a_[1] > y) != (b_[1] > y):
                    xs.append(a_[0] + (y - a_[1]) / (b_[1] - a_[1]) * (b_[0] - a_[0]))
            if len(xs) >= 2:
                a = max(0, int((min(xs) - x0) / px_um)); b = min(nx, int(math.ceil((max(xs) - x0) / px_um)))
                if b > a:
                    g[j, a:b] = 1.0
    k = max(1, int(round(eye_um / px_um)))
    m, n = g.shape[0] - g.shape[0] % k, g.shape[1] - g.shape[1] % k
    return g[:m, :n].reshape(m // k, k, n // k, k).mean(axis=(1, 3))


MIN_TILE_W = 150


def tile(cov, px_h, w_um, h_um):
    """Panel at the cell's true aspect, padded (never stretched) to MIN_TILE_W so
    every label has room. Padding is the page background, so a narrow cell
    still reads as narrow."""
    img = np.clip(cov, 0, 1)[..., None] * GOLD
    im = Image.fromarray((np.clip(img, 0, 1) ** (1 / 2.2) * 255).astype(np.uint8))
    w = max(24, int(px_h * w_um / h_um))
    im = im.resize((w, px_h), Image.LANCZOS)
    if w < MIN_TILE_W:
        pad = Image.new("RGB", (MIN_TILE_W, px_h), BG)
        pad.paste(im, (0, 0))
        im = pad
    return im


def row(tiles, labels, title, sub, W=1180):
    pad = 6
    hh = max(t.height for t in tiles)
    n_lines = max(len(l.split("\n")) for l in labels)
    lab_h = 6 + 13 * n_lines
    im = Image.new("RGB", (W, hh + lab_h + 46), BG)
    d = ImageDraw.Draw(im)
    x = pad
    for t, l in zip(tiles, labels):
        im.paste(t, (x, pad))
        for k, line in enumerate(l.split("\n")):
            d.text((x + 2, hh + pad + 4 + 13 * k), line, fill=DIM, font=font(10))
        x += t.width + pad
    d.text((pad, hh + pad + lab_h + 4), title, fill=FG, font=font(13))
    d.text((pad, hh + pad + lab_h + 22), sub, fill=DIM, font=font(10))
    return im


def stack(rows, path):
    im = Image.new("RGB", (max(r.width for r in rows), sum(r.height + 6 for r in rows) + 6), BG)
    y = 6
    for r in rows:
        im.paste(r, (0, y))
        y += r.height + 6
    im.save(path)
    print("  saved", path, im.size)


def main():
    out = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
    out.mkdir(parents=True, exist_ok=True)
    cells = {c.cid: c for band in doe_cells() for c in band}

    skipped: list[str] = []

    def build(cid):
        """``(cell, art)``, or None when that cell is not on the current plate."""
        c = cells.get(cid)
        if c is None:
            skipped.append(cid)
            return None
        return c, c.build(0.0, 0.0, c.w_um, c.h_um)

    def add_row(tiles, labs, title, sub):
        """Append a row, or note that every cell in it is off-plate."""
        if not tiles:
            print(f"  skipping (no cells on the plate): {title.split(':')[0]}")
            return
        rows.append(row(tiles, labs, title, sub))

    # ---- Figure A: beat + rotation ladders at built widths
    rows = []
    tiles, labs = [], []
    for cid in sorted((k for k in cells if k.startswith("BEAT")), key=lambda k: float(k[4:])):
        c, a = build(cid)
        tiles.append(tile(raster(a, 0, 0, c.w_um, c.h_um), 120, c.w_um, c.h_um))
        labs.append(f"{c.label}\nd = {a.stats['delta_um']:.2f} um\n{c.w_um/1000:g} x {c.h_um/1000:g} mm, {a.stats['fringes_across']:.1f} fringes")
    add_row(tiles, labs, "B-BEAT: two pitches on ONE plane -- fringes ACROSS the lines",
                    "static fringes, no bond. The count inverts to the pitch error through p/delta = 24.7x. Each panel: emitted rectangles, averaged over the 87 um eye cell.")
    tiles, labs = [], []
    for cid in sorted((k for k in cells if k.startswith("ROT")), key=lambda k: float(k[3:])):
        c, a = build(cid)
        tiles.append(tile(raster(a, 0, 0, c.w_um, c.h_um), 120, c.w_um, c.h_um))
        labs.append(f"{c.label}\nbeat {a.stats['beat_um']:.0f} um\n{c.w_um/1000:g} x {c.h_um/1000:g} mm, {a.stats['fringes_across']:.1f} fringes")
    add_row(tiles, labs, "B-ROT: equal pitch, one rotated -- fringes ALONG the lines",
                    "p/(2 sin(a/2)). k1 - k2 is perpendicular to the bisector of the two grating vectors, which is why rotation moire looks nothing like pitch moire.")
    if rows:
        stack(rows, out / "witness_moire.png")

    # ---- Figure B: harmonic with profiles, contrast, crossed
    rows = []
    tiles, labs = [], []
    for cid in ("HARM0.42", "HARM0.50", "HARM0.58"):
        got = build(cid)
        if got is None:
            continue
        c, a = got
        cov = raster(a, 0, 0, c.w_um, c.h_um, px_um=1.0)
        t = tile(cov, 110, c.w_um, c.h_um)
        # The profile must come from the RAW 1 um raster, not the eye-integrated
        # one: at 87 um bins a 559 um period is six samples and the column means
        # are binning noise. At 1 um it is 559 samples, and a 286 um box average
        # (two (1,1) beats) removes the 143 um component and leaves (2,3) standing.
        raw = raster(a, 0, 0, c.w_um, c.h_um, px_um=1.0, eye_um=1.0)
        prof = raw.mean(axis=0)
        # the eye-cell raster still carries the 143 um (1,1) beat; a 286 um
        # moving average removes it and leaves the 559 um component we are after
        # Cascade of integer-length box filters: each nulls its own period and
        # that period's harmonics exactly. 143 um (the (1,1) beat), 64 and 44 um
        # (the two gratings). A single 286 um box left +-3.9% of grating residue,
        # which buries a +-1.75% beat.
        # nulls: the carrier, the screen, and every beat with an ODD carrier
        # harmonic (those exist at every duty and would leak into the profile),
        # each an integer box in um at the 1 um raster. What survives is the
        # even-harmonic family -- the component that must vanish at 0.50.
        odd = {int(round(t["beat_um"])) for t in wm.harmonic_beats(44.0, BOX_CARRIER_UM, 0.5, max_order=4)
               if t["n"] % 2 == 1 and 50.0 < t["beat_um"] < 2000.0}
        for k in sorted(odd | {int(round(BOX_CARRIER_UM)), 44}):
            prof = np.convolve(prof, np.ones(k) / k, mode="valid")
        prof = prof - prof.mean()
        # decimate to ~2 samples per plotted pixel so the line is a curve, not hash
        pw = t.width
        step = max(1, len(prof) // (2 * pw))
        prof = prof[: len(prof) // step * step].reshape(-1, step).mean(axis=1)
        pimg = Image.new("RGB", (pw, 46), BG)
        d = ImageDraw.Draw(pimg)
        xs = np.linspace(0, pw - 1, len(prof))
        amp = max(1e-6, float(np.abs(prof).max()))
        scale = max(amp, 0.01)           # common scale so a flat 0.50 reads flat
        d.line([(0, 15), (pw, 15)], fill=(40, 48, 52))
        pts = [(x, 15 - v / scale * 13) for x, v in zip(xs, prof)]
        d.line(pts, fill=WARN, width=2)
        d.text((2, 32), f"even-harmonic component: +-{amp*100:.2f}% of mean", fill=DIM, font=font(9))
        both = Image.new("RGB", (pw, t.height + 48), BG)
        both.paste(t, (0, 0)); both.paste(pimg, (0, t.height + 2))
        tiles.append(both)
        duty = a.stats["duty"]
        def A(k, cc):
            x = math.pi * k * cc
            return abs(cc * math.sin(x) / x) if x else 0.0
        # the lowest-order visible beat with an EVEN carrier harmonic: that is
        # the one that vanishes at exactly 50% duty
        even = [t for t in wm.harmonic_beats(44.0, BOX_CARRIER_UM, duty) if t["n"] % 2 == 0 and t["beat_um"] > 87.0]
        t0 = max(even, key=lambda t: t["amplitude"]) if even else {"m": 1, "n": 2, "beat_um": 0}
        bias = 2 * A(t0["m"], duty) * A(t0["n"], duty) / (1 - duty) ** 2
        labs.append(f"{c.label}  true contrast\n({t0['m']},{t0['n']}) {t0['beat_um']:.0f} um, bias case {bias*100:.1f}%")
    add_row(tiles, labs, f"B-HARM: 44 um screen over the {BOX_CARRIER_UM:g} um carrier, both at the swept duty -- TRUE contrast, no stretch",
                    "the even-harmonic component is in the profile at 0.42 and 0.58 and at a MINIMUM at 0.50: the (1,2) term vanishes at exactly 50% duty, but 99/44 = 9/4 is commensurate and the odd (3,7) term shares its 396 um period, leaving a floor of about 1% after the cascade (~2% true). The panels themselves carry the (1,3) 132 um texture.")
    tiles, labs = [], []
    for cid in ("BCON0.25", "BCON0.50", "BCON0.75"):
        got = build(cid)
        if got is None:
            continue
        c, a = got
        tiles.append(tile(raster(a, 0, 0, c.w_um, c.h_um), 110, c.w_um, c.h_um))
        s = a.stats
        labs.append(f"{c.label}\nmean  T {s['mean_T']:.2f} / R {s['mean_R']:.2f}\ncontrast  T {s['contrast_T']:.2f} / R {s['contrast_R']:.2f}")
    for cid in ("CROSS5x5", "CROSS5x8"):
        got = build(cid)
        if got is None:
            continue
        c, a = got
        win = 100.0
        aa = wm.build_crossed(0, 0, win, win, period_x_um=a.stats["period_x_um"], period_y_um=a.stats["period_y_um"])
        g = raster(aa, 0, 0, win, win, px_um=0.5, eye_um=0.5)
        tiles.append(tile(g, 110, win, win))
        labs.append(f"{c.label}\n100 um window\nnot eye-integrated")
    add_row(tiles, labs, "B-CONT: duty against brightness (transmission and reflection rank them differently)  |  D-CROSS: the 2-D lattice",
                    "B-CONT rendered as metal coverage. D-CROSS at 0.5 um/px over 100 um: at 87 um a 5 um lattice averages to a flat tone, which is correct and shows nothing.")
    if rows:
        stack(rows, out / "witness_moire_b.png")
    if skipped:
        print("  off-plate cells, nothing drawn for them: " + ", ".join(skipped))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
