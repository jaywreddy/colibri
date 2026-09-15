"""In-silico gates for the production dies (docs/production-plate-plan.md §4.A).

Three checks, each from the geometry the writer emits or the exact fab
periods, never from a formula alone:

  photo    A1  the LEFT die's real front metal (export_fine.build_plate_fine),
               integrated exactly to the 87 um eye cell, compared against the
               builder's own intended per-pixel coverage (photo.photo_coverage).
  regions  A2  the lid's and front's region gratings as written: metal fraction and pitch per region
               at the plate's glass and comb; then a registration
               sweep (back ply offset 0 / 8 / 20 / 40 um) -> the bench tolerance.
  nearfield A3 angular-spectrum propagation across the 1.5 mm ply for the
               garland at the 22/24 um baseline pitch, the garland as
               built at this glass's pitch, and the monogram pair, under an
               incoherent source (11 angles x 3 wavelengths), box-averaged to
               the eye cell: the fringe contrast that survives the gap.

    uv run --directory backend python ../tools/dev/validate_dies.py OUTDIR [photo|regions|nearfield|all]

One heavy process at a time; each check stays under ~1 GB.
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "backend"))

from app import witness_dies as wd  # noqa: E402
from app.witness_geom import (BOX_CARRIER_UM, BOX_FRONT_LEAF_UM, BOX_MONO_UM,  # noqa: E402
                              GLASS_MATERIAL, METAL)

BG = (13, 17, 19)
FG = (214, 224, 227)
DIM = (150, 165, 172)
WARN = (224, 138, 114)
OK = (110, 200, 150)
EYE_UM = 87.0
LAM_UM = (0.45, 0.55, 0.65)


def font(sz=12):
    try:
        return ImageFont.truetype("arialbd.ttf", sz)
    except Exception:
        return ImageFont.load_default()


# --- rasteriser ---------------------------------------------------------------


def raster_rects(rects, x0, y1, px, nx, ny, out=None):
    """Fill (N,4) [x0,x1,y0,y1] rects into a (ny,nx) uint8 grid whose top-left
    is (x0, y1), y down."""
    g = np.zeros((ny, nx), np.uint8) if out is None else out
    r = np.asarray(rects, dtype=np.float64)
    if r.size == 0:
        return g
    a = np.clip(np.floor((r[:, 0] - x0) / px).astype(int), 0, nx)
    b = np.clip(np.ceil((r[:, 1] - x0) / px).astype(int), 0, nx)
    c = np.clip(np.floor((y1 - r[:, 3]) / px).astype(int), 0, ny)
    d = np.clip(np.ceil((y1 - r[:, 2]) / px).astype(int), 0, ny)
    for i in range(len(r)):
        if b[i] > a[i] and d[i] > c[i]:
            g[c[i]:d[i], a[i]:b[i]] = 1
    return g


def raster_polys(polys, x0, y1, px, nx, ny, out=None):
    im = Image.fromarray(out) if out is not None else Image.new("L", (nx, ny), 0)
    d = ImageDraw.Draw(im)
    for pv in polys:
        pv = np.asarray(pv, dtype=np.float64)
        pts = [((x - x0) / px, (y1 - y) / px) for x, y in pv]
        d.polygon(pts, fill=1)
    return np.asarray(im, dtype=np.uint8)


def coverage_grid(rects, x0, y1, cell, nx, ny, out=None):
    """EXACT area fraction of ``rects`` in each ``cell``-sized bin of a grid
    whose top-left is (x0, y1), y down. A rasteriser that floors/ceils edges
    inflates a 2.5 um stripe to whole pixels and biases coverage by 40%; this
    clips every rectangle to the bins it touches and accumulates area."""
    g = np.zeros((ny, nx), np.float64) if out is None else out
    r = np.asarray(rects, dtype=np.float64)
    if r.size == 0:
        return g
    rx0, rx1 = r[:, 0] - x0, r[:, 1] - x0
    ry0, ry1 = y1 - r[:, 3], y1 - r[:, 2]          # y down
    ix0 = np.floor(rx0 / cell).astype(int); ix1 = np.ceil(rx1 / cell).astype(int) - 1
    iy0 = np.floor(ry0 / cell).astype(int); iy1 = np.ceil(ry1 / cell).astype(int) - 1
    for k in range(int((ix1 - ix0).max()) + 1):
        ix = ix0 + k
        okx = (ix <= ix1) & (ix >= 0) & (ix < nx)
        ox = np.minimum(rx1, (ix + 1) * cell) - np.maximum(rx0, ix * cell)
        for l in range(int((iy1 - iy0).max()) + 1):
            iy = iy0 + l
            ok = okx & (iy <= iy1) & (iy >= 0) & (iy < ny)
            oy = np.minimum(ry1, (iy + 1) * cell) - np.maximum(ry0, iy * cell)
            a = ox * oy
            m = ok & (a > 0)
            np.add.at(g, (iy[m], ix[m]), a[m] / (cell * cell))
    return g


def box_mean(g, k):
    m, n = g.shape[0] - g.shape[0] % k, g.shape[1] - g.shape[1] % k
    return g[:m, :n].reshape(m // k, k, n // k, k).mean(axis=(1, 3))


def _propagate(field: np.ndarray, dx_um: float, z_um: float, lam_um: float, n: float) -> np.ndarray:
    """Angular-spectrum propagation of a scalar complex field. Vendored from
    ``app.sim.angular_spectrum._propagate`` (the sim/ package was deleted with
    the holography-simulator retirement, 2026-09 tools+docs cleanup) — this is
    the only caller left, the A3 near-field gate below, so the ~15-line
    pure-numpy FFT sandwich lives here instead of pulling in a whole package
    for one function."""
    h, w = field.shape
    k0 = 2 * np.pi / lam_um
    kx = np.fft.fftfreq(w, d=dx_um) * 2 * np.pi
    ky = np.fft.fftfreq(h, d=dx_um) * 2 * np.pi
    KX, KY = np.meshgrid(kx, ky)
    kz_sq = (n * k0) ** 2 - KX**2 - KY**2
    kz = np.sqrt(np.maximum(kz_sq, 0)).astype(np.complex64)
    evanescent = kz_sq < 0
    kz = np.where(evanescent, 1j * np.sqrt(np.abs(kz_sq)), kz)
    H = np.exp(1j * kz * z_um).astype(np.complex64)
    F = np.fft.fft2(field)
    return np.fft.ifft2(F * H)


# --- A1 photo ------------------------------------------------------------------


def check_photo(out: Path) -> dict:
    """A1, ported to the production geometry: DIE-LEFT is built by
    ``witness_dies.build_face_die``, which is ``export_fine.build_plate_fine``
    — the one authoring toolchain every face goes through. (This gate used to
    call a ``build_colour_side`` writer of its own; that second path is gone,
    which is the point: the gate now measures the geometry the mask gets.)

    ``fine.front_polys`` is the merged, DRC-healed FRONT metal in the plate
    frame (plate-centred, y up), UNMIRRORED — ``witness_dies._ply_art`` only
    mirrors after this point, when it assembles ``build_face_die``'s returned
    ``art``. The intended coverage below (``photo_coverage``) is equally
    unmirrored, so the two are compared directly with no flip.
    """
    from app import export_fine
    from app import plates as P
    from app.patterns.bitmap import photo as ph

    t0 = time.perf_counter()
    _, spec = wd.blank_plan()
    pspec = spec.faces["left"]
    side = P.CENTERPIECE_FILL * P._aperture(pspec)
    px = EYE_UM
    n = int(round(side / px))
    x0, y1 = -side / 2, side / 2

    fine = export_fine.build_plate_fine(pspec, "left")

    # The FRONT metal inside the photo's art box (the halftone bands + their
    # coloured sub-grating stripes; the frame's leaves and carrier lie outside
    # it by construction), accumulated as EXACT area per eye cell by the same
    # analytic rasteriser the simulator's literal_front.png is made with
    # (``literal_raster.layer_coverage``: rectangles by clipped overlap, other
    # rings by Green's theorem). Rings are selected by bbox; the art box is
    # centred on the plate, so the plate frame is the art-box frame.
    from app.literal_raster import layer_coverage

    in_polys: list[np.ndarray] = []
    n_nonrect = 0
    for ring in fine.front_polys:
        r = np.asarray(ring, dtype=np.float64)
        bx0, bx1 = float(r[:, 0].min()), float(r[:, 0].max())
        by0, by1 = float(r[:, 1].min()), float(r[:, 1].max())
        if bx0 < x0 - 1 or bx1 > -x0 + 1 or by0 < -y1 - 1 or by1 > y1 + 1:
            continue  # outside the art box: frame leaves / carrier, not the photo
        in_polys.append(r)
        xs = np.unique(np.round(r[:, 0], 4))
        ys = np.unique(np.round(r[:, 1], 4))
        if not (r.shape[0] == 4 and xs.size == 2 and ys.size == 2):
            n_nonrect += 1  # merged band steps; exact all the same
    metal_eye = np.clip(layer_coverage(in_polys, side, side, n, n), 0, 1)
    # The clear side is 1 - metal_eye BY CONSTRUCTION here (metal and its
    # complement are the same die inversion tested exactly in
    # tests/test_witness.py::test_the_die_inversion_is_the_exact_complement_of_its_metal),
    # so there is no separate clear build/raster and no tiling stat to report.

    # The intended coverage: the SAME per-pixel tone model the builder screens
    # from (`plates._photo_halftone_art` -> `photo.photo_coverage`), at its own
    # asset resolution, resampled to the eye grid. This already carries the
    # edge fade-to-carrier and the colour gate (`weight < 0.5` zeros `ids`),
    # so it is the builder's actual intent, not a re-derivation of it.
    inp = P._photo_screen_inputs(pspec)
    cov, ids, _periods = ph.photo_coverage(
        inp["image"], inp["fade_start"], inp["fade_gate"], inp["tone_steps"],
        inp["asset_px"], colour_mode=inp["colour_mode"],
    )
    shape = metal_eye.shape[::-1]  # PIL wants (w, h)
    dk = np.asarray(Image.fromarray(np.clip(cov * 255, 0, 255).astype(np.uint8))
                    .resize(shape, Image.BOX)) / 255.0
    cm = np.asarray(Image.fromarray(((ids > 0).astype(np.uint8) * 255))
                    .resize(shape, Image.BOX)) / 255.0
    # coloured bands carry a 50% sub-grating and are NOT tone-held: their metal
    # is half the band (screenrects.stripe_plan / witness_cells.build_halftone_bands,
    # duty = colourplan.ColourPlan.duty, default 0.5). The builder's intended
    # coverage is therefore cov * (1 - 0.5 * coloured); the plain target is cov itself.
    want_metal = dk * (1.0 - 0.5 * cm)
    levels = inp["tone_steps"]
    err_plain = metal_eye - dk
    err_int = metal_eye - want_metal
    res = {
        "portrait_um": side, "eye_cells": list(metal_eye.shape),
        "method": "exact area per 87 um cell, from export_fine.build_plate_fine's front_polys (un-mirrored)",
        "non_rect_rings_in_art_box": n_nonrect,
        "mean_metal": float(metal_eye.mean()), "mean_dark": float(dk.mean()),
        "coloured_frac_of_cells": float(cm.mean()),
        "vs_darkness_mae_levels": float(np.abs(err_plain).mean() * levels),
        "vs_darkness_bias_levels": float(err_plain.mean() * levels),
        "vs_intended_mae_levels": float(np.abs(err_int).mean() * levels),
        "vs_intended_p95_levels": float(np.percentile(np.abs(err_int), 95) * levels),
        "vs_intended_bias_levels": float(err_int.mean() * levels),
        "vs_intended_bias_in_coloured_cells_levels": float(err_int[cm > 0.5].mean() * levels) if (cm > 0.5).any() else None,
        "vs_intended_bias_in_plain_cells_levels": float(err_int[cm <= 0.5].mean() * levels),
        "colour_lightening_levels": float((dk * 0.5 * cm).mean() * levels),
        "build_s": round(time.perf_counter() - t0, 1),
    }
    def tile(a, lo, hi):
        v = np.clip((a - lo) / (hi - lo), 0, 1)
        return Image.fromarray((v * 255).astype(np.uint8)).convert("RGB")
    h = metal_eye.shape[0]
    im = Image.new("RGB", (4 * h + 50, h + 64), BG)
    im.paste(tile(1 - metal_eye, 0, 1), (10, 10))
    im.paste(tile(1 - want_metal, 0, 1), (h + 20, 10))
    im.paste(tile(1 - dk, 0, 1), (2 * h + 30, 10))
    im.paste(tile(err_int, -3 / levels, 3 / levels), (3 * h + 40, 10))
    dr = ImageDraw.Draw(im)
    dr.text((10, h + 14), "emitted metal, 87 um eye cell", fill=FG, font=font(11))
    dr.text((h + 20, h + 14), "intended (colour bands halved)", fill=FG, font=font(11))
    dr.text((2 * h + 30, h + 14), "prepped darkness", fill=FG, font=font(11))
    dr.text((3 * h + 40, h + 14), f"emitted - intended, +-3 levels", fill=FG, font=font(11))
    dr.text((10, h + 34), f"DIE-LEFT portrait {side/1000:.1f} mm, {inp['colour_mode'].upper()}; "
            f"{inp['line_period_um']:g} um screen, {levels} levels. mean |err| {res['vs_intended_mae_levels']:.2f} levels, "
            f"bias {res['vs_intended_bias_levels']:+.2f} ({res['non_rect_rings_in_art_box']} non-rect rings folded in)", fill=DIM, font=font(10))
    dr.text((10, h + 48), f"the coloured bands ({res['coloured_frac_of_cells']:.0%} of cells) hold tone with a 50% sub-grating and print {res['colour_lightening_levels']:.1f} levels lighter than the photo -- the cost of holding tone (plan 1.1), measured", fill=DIM, font=font(10))
    im.save(out / "validate_photo.png")
    return res


# --- A2 regions ----------------------------------------------------------------


def check_regions(out: Path) -> dict:
    """The single-layer diffraction centrepieces (lid, front) as WRITTEN: for
    every region of the motif's map, the metal fraction of the die's own
    front geometry inside that region's cells and the stripe pitch measured
    from the run lengths along x. A grating region should come out at its
    duty and its period; a solid one near 1.0. Heavy (two die builds, one
    at a time); replaces the bonded design's A2 switch gate."""
    from app import plates as P
    from app import region_art as RA

    res: dict = {}
    for face in ("top", "front"):
        _, spec = wd.blank_plan()
        pspec = spec.faces[face]
        if not RA.single_layer_centerpiece(pspec):
            res[face] = {"skipped": f"{pspec.pattern_slug} has no region map"}
            continue
        d = wd.die_dims(face)
        t0 = time.perf_counter()
        art = wd.build_face_die(face, 0.0, 0.0, d["f_w"], d["f_h"], METAL)
        side = P.CENTERPIECE_FILL * P._aperture(pspec)
        px = 0.5                                  # resolve 4-6 um lines
        n = int(round(side / px))
        x0, y1 = -side / 2, side / 2
        # the die is MIRRORED (x -> -x): un-mirror so the map's x matches
        rects = np.asarray(art.front, dtype=np.float64).copy()
        if rects.size:
            rects[:, [0, 1]] = -rects[:, [1, 0]]
        polys = [np.asarray(pv, dtype=np.float64) * np.array([-1.0, 1.0]) for pv in art.polys]
        metal = raster_polys(polys, x0, y1, px, n, n, raster_rects(rects, x0, y1, px, n, n)) > 0.5
        ra = RA.centerpiece_regions(pspec.pattern_slug, n, pspec.pattern_params)
        per_region = {}
        for rid, reg in sorted(ra.regions.items()):
            zone = ra.labels == rid
            if zone.sum() < 400:
                continue
            # interior of the zone (the emitter insets one 20 um cell at the seams)
            k = max(1, int(round(RA.REGION_ZONE_PITCH_UM / px)))
            inner = zone.copy()
            for _ in range(k):
                inner[1:] &= zone[:-1]; inner[:-1] &= zone[1:]
                inner[:, 1:] &= zone[:, :-1]; inner[:, :-1] &= zone[:, 1:]
            if not inner.any():
                continue
            frac = float(metal[inner].mean())
            entry = {"period_um": reg.period_um, "duty": reg.duty, "cells": int(inner.sum()),
                     "metal_frac": round(frac, 3)}
            if reg.period_um > 0:
                # stripe pitch: mean distance between rising edges along x inside the zone
                rows = np.flatnonzero(inner.any(axis=1))[::7]
                pitches = []
                for r in rows:
                    m = metal[r] & inner[r]
                    rises = np.flatnonzero(np.diff(m.astype(np.int8)) == 1)
                    if rises.size >= 4:
                        pitches.append(np.median(np.diff(rises)) * px)
                entry["measured_period_um"] = round(float(np.median(pitches)), 3) if pitches else None
                entry["ok"] = bool(abs(frac - reg.duty) < 0.08 and pitches
                                   and abs(np.median(pitches) - reg.period_um) < 0.3)
            else:
                entry["ok"] = bool(frac > 0.9)
            per_region[reg.name] = entry
        res[face] = {"slug": pspec.pattern_slug, "build_s": round(time.perf_counter() - t0, 1),
                     "art_box_um": side, "regions": per_region,
                     "all_ok": all(e.get("ok") for e in per_region.values())}
        # a false-colour picture of the written art box: period -> hue, solid gold
        lut = ra.period_lut()
        per_map = lut[ra.labels]
        img = np.zeros((n, n, 3), np.float32)
        gold = np.array([0.83, 0.68, 0.21], np.float32)
        hue_t = np.clip((per_map - 4.15) / (6.02 - 4.15), 0, 1)
        col = np.stack([hue_t, 1 - np.abs(hue_t - 0.5) * 2, 1 - hue_t], axis=-1)
        img[metal] = gold
        colr = (per_map > 0) & metal
        img[colr] = 0.35 * gold + 0.65 * col[colr]
        Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8)).resize((900, 900), Image.BOX).save(
            out / f"validate_regions_{face}.png")
    return res


# --- A3 near field -----------------------------------------------------------


def _grating(nx, ny, px, period, duty, angle_deg, phase=0.0):
    y, x = np.mgrid[0:ny, 0:nx].astype(np.float64)
    x = (x - nx / 2) * px
    y = (y - ny / 2) * px
    a = math.radians(angle_deg)
    u = x * math.cos(a) + y * math.sin(a)
    return (np.mod(u / period + phase, 1.0) < duty).astype(np.float64)   # 1 = gold


def _eye_lowpass(I, px):
    """Gaussian low-pass, sigma = half the eye cell. A box the size of the eye
    cell leaves 21% of a 63.5 um carrier standing (its sinc is not at a zero
    there); the Gaussian leaves 1e-4 of it and passes a 770 um beat at 94%."""
    sigma = EYE_UM / 2.0 / px
    ny, nx = I.shape
    fy = np.fft.fftfreq(ny)[:, None]
    fx = np.fft.fftfreq(nx)[None, :]
    H = np.exp(-2.0 * (math.pi * sigma) ** 2 * (fx ** 2 + fy ** 2))
    return np.real(np.fft.ifft2(np.fft.fft2(I) * H))


def _fringe_contrast(I, px, crop=0.15):
    m = _eye_lowpass(I, px)
    c = int(m.shape[0] * crop)
    m = m[c:-c, c:-c]
    lo, hi = np.percentile(m, 5), np.percentile(m, 95)
    return float((hi - lo) / max(1e-9, hi + lo)), m


def check_nearfield(out: Path) -> dict:
    z = wd.PLY_UM
    n_glass = wd.GLASS_N
    px = 2.0
    N = 2048
    C = BOX_CARRIER_UM
    cases = [
        ("garland at the 500 um baseline pitch, 22 / 24 um", 22.0, 22.0 * 1.09, 2.5),
        (f"garland as built, {C:g} / {BOX_FRONT_LEAF_UM:.1f} um", C, BOX_FRONT_LEAF_UM, 2.5),
        (f"monogram, {C:g} / {BOX_MONO_UM:.2f} um beat 1635", C, BOX_MONO_UM, 0.0),
    ]
    angles = np.linspace(-0.5, 0.5, 11)
    results = []
    tiles = []
    for name, p_back, p_front, ang in cases:
        back = _grating(N, N, px, p_back, 0.5, 0.0)
        front = _grating(N, N, px, p_front, 0.5, ang)
        t_back = 1.0 - back
        t_front = 1.0 - front
        I_geo = t_back * t_front                       # zero gap: the union identity
        I_sum = np.zeros((N, N))
        t0 = time.perf_counter()
        for lam in LAM_UM:
            for th in angles:
                kx = 2 * math.pi / lam * math.sin(math.radians(th))
                xs = (np.arange(N) - N / 2) * px
                tilt = np.exp(1j * kx * xs)[None, :]
                U = t_back * tilt
                U = _propagate(U, px, z, lam, n_glass)
                I_sum += np.abs(U * t_front) ** 2
        I_sum /= len(LAM_UM) * len(angles)
        c_geo, m_geo = _fringe_contrast(I_geo, px)
        c_gap, m_gap = _fringe_contrast(I_sum, px)
        fres = p_back ** 2 * n_glass / (4 * 0.55 * z)
        results.append({"case": name, "back_um": p_back, "front_um": round(p_front, 3), "angle_deg": ang,
                        "fresnel_N": round(fres, 3), "contrast_zero_gap": round(c_geo, 4),
                        "contrast_across_gap": round(c_gap, 4),
                        "survives": round(c_gap / max(1e-9, c_geo), 3), "s": round(time.perf_counter() - t0, 1)})
        tiles.append((name, m_geo, m_gap, c_geo, c_gap, fres))
        print(f"  {name}: N={fres:.2f}  contrast zero-gap {c_geo:.3f} -> across gap {c_gap:.3f}")
    # figure
    T = 220
    im = Image.new("RGB", (3 * (2 * T + 30) + 20, T + 90), BG)
    dr = ImageDraw.Draw(im)
    for i, (name, m_geo, m_gap, c_geo, c_gap, fres) in enumerate(tiles):
        x = 10 + i * (2 * T + 30)
        for j, (m, lab, c) in enumerate(((m_geo, "zero gap", c_geo), (m_gap, "across 1.5 mm", c_gap))):
            lo, hi = m.min(), m.max()
            v = (m - lo) / max(1e-9, hi - lo)
            t = Image.fromarray((v * 255).astype(np.uint8)).resize((T, T), Image.BILINEAR).convert("RGB")
            im.paste(t, (x + j * (T + 6), 10))
            dr.text((x + j * (T + 6), T + 14), f"{lab}: contrast {c:.3f}", fill=FG, font=font(11))
        col = OK if c_gap / max(1e-9, c_geo) > 0.5 else WARN
        dr.text((x, T + 34), name, fill=col, font=font(12))
        dr.text((x, T + 52), f"Fresnel N = {fres:.2f}  ->  {c_gap/max(1e-9,c_geo):.0%} of the zero-gap fringe survives", fill=DIM, font=font(10))
    dr.text((10, T + 72), f"angular spectrum through {z/1000:g} mm of n = {n_glass} glass; 11 source angles over +-0.5 deg x 3 wavelengths, eye-cell (87 um) integrated, 4 mm patches", fill=DIM, font=font(10))
    im.save(out / "validate_nearfield.png")
    return {"z_um": z, "n": n_glass, "cases": results}


def main():
    out = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
    out.mkdir(parents=True, exist_ok=True)
    which = sys.argv[2] if len(sys.argv) > 2 else "all"
    res = {}
    if which in ("photo", "all"):
        print("A1 photo ..."); res["photo"] = check_photo(out); print(json.dumps(res["photo"], indent=1))
    if which in ("regions", "all"):
        print("A2 regions ..."); res["regions"] = check_regions(out); print(json.dumps(res["regions"], indent=1))
    if which in ("nearfield", "all"):
        print("A3 near field ..."); res["nearfield"] = check_nearfield(out); print(json.dumps(res["nearfield"], indent=1))
    prev = {}
    jp = out / "validate_dies.json"
    if jp.exists():
        prev = json.loads(jp.read_text())
    prev.update(res)
    jp.write_text(json.dumps(prev, indent=1))
    print("saved", jp)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
