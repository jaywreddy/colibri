"""Simulate how the witness plate's halftone cells will look, at several tilts.

The appearance model is the corrected one, and it is worth restating because an
earlier version of these previews was wrong in a way that changed a design
recommendation:

    appearance = GOLD * eta_0 * coverage  +  spectrum(d, view) * eta_1 * coverage

Both terms, always. A binary amplitude grating puts eta_0 = duty^2 = 0.25 into
the SPECULAR order against eta_1 = 0.101 into the first, so a gratinged band is
mostly still a gold mirror with a spectrum riding on it — which is exactly how
``plate.frag`` composites it (``color += sheen``). Rendering only the diffracted
part collapses a region to black wherever the first order leaves the visible
band, and that black is what made hue-mapping look garish and unusable when it
is neither.

Coverage is the halftone's own tone, so nothing here needs the rectangles: the
geometry is fully determined by the tone map and the period field, and
``test_screenrects.py`` is what checks that the emitted rectangles agree with
this. Previewing from the source rather than from a raster of two million
rectangles is also the only version that runs on this host.

    uv run --directory backend python ../tools/dev/render_witness_preview.py OUTDIR
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "backend"))

from app import witness_cells as wc  # noqa: E402
from app.diffraction import bake_lut, lamellar_order_efficiency  # noqa: E402
from app.export_witness import PORTRAIT_MM, _plan  # noqa: E402
from app.patterns.bitmap import colourplan as cp  # noqa: E402
from app.patterns.bitmap import imageprep as ip  # noqa: E402

GOLD = np.array([0.84, 0.66, 0.29], np.float32)
BG = (13, 17, 19)
FG = (214, 224, 227)
DIM = (150, 165, 172)
EYE_UM = 87.0
DUTY = 0.5
ETA0 = lamellar_order_efficiency(0, DUTY)
ETA1 = lamellar_order_efficiency(1, DUTY)
LUT = bake_lut(duty=DUTY, u_max_um=10.0, size=1024)
K0 = 0.11
"""Geometry factor sin(theta_v) - sin(theta_i) at rest. lambda = d * k, so at
k = 0.11 a 5 um period diffracts 550 nm — green — straight at the viewer."""
TILT_GAIN = 0.42
SOURCE_SPREAD = 0.06
"""Fractional width of the source. Narrow: at 4-5 um the visible band spans
about 5 deg, so a room-wide source washes the colour to white. These previews
assume a lamp."""


def font(sz: int = 13):
    try:
        return ImageFont.truetype("arialbd.ttf", sz)
    except Exception:
        return ImageFont.load_default()


def sheen(u: np.ndarray, n: int = 5) -> np.ndarray:
    """Spectral term, averaged over an extended source."""
    acc = np.zeros(u.shape + (3,), np.float32)
    for w in np.linspace(1.0 - SOURCE_SPREAD, 1.0 + SOURCE_SPREAD, n):
        i = np.clip(np.abs(u * w) / 10.0, 0.0, 1.0) * (LUT.shape[0] - 1)
        acc += LUT[i.astype(np.int32)]
    return acc / n


def block_mean(a: np.ndarray, k: int) -> np.ndarray:
    m = a.shape[0] - a.shape[0] % k
    n = a.shape[1] - a.shape[1] % k
    b = a[:m, :n]
    return b.reshape(m // k, k, n // k, k).mean(axis=(1, 3))


def block_mode(a: np.ndarray, k: int, n_ids: int) -> np.ndarray:
    """Majority id per block — ids are nominal, so a mean would invent rungs."""
    m = a.shape[0] - a.shape[0] % k
    n = a.shape[1] - a.shape[1] % k
    b = a[:m, :n]
    best = np.zeros((m // k, n // k), np.float32)
    out = np.zeros((m // k, n // k), np.int32)
    for j in range(n_ids):
        c = block_mean((b == j).astype(np.float32), k)
        win = c > best
        best, out = np.where(win, c, best), np.where(win, j, out)
    return out


def render_cell(
    plan: cp.ColourPlan, side_mm: float, tilt_deg: float, *,
    line_period_um: float = 44.0, tone_steps: int = 22,
    prep: ip.PrepSpec | None = None, px: int = 320,
) -> Image.Image:
    """One cell at one tilt, integrated over the eye's 87 um cell."""
    side_um = side_mm * 1000.0
    src_px = wc.asset_px_for(side_um, line_period_um)
    gray, rgb = wc.portrait_source(src_px)
    spec = prep or ip.PrepSpec(tone_steps=tone_steps)
    if spec.tone_steps != tone_steps:
        spec = ip.PrepSpec(**{**spec.__dict__, "tone_steps": tone_steps})
    dark = ip.prep_darkness(gray, spec)
    ids, periods, _ = cp.build_period_field(rgb, plan)

    # Integrate over the eye cell: that is what makes a halftone a photograph.
    k = max(1, int(round(EYE_UM / (side_um / src_px))))
    cov = np.clip(block_mean(dark, k), 0.0, 1.0)
    n_ids = int(max(ids.max(), 0)) + 1
    pid = block_mode(ids, k, n_ids) if n_ids > 1 else np.zeros(cov.shape, np.int32)

    lut = np.zeros(n_ids, np.float32)
    for i, d in periods.items():
        if i < n_ids:
            lut[i] = d
    d_map = lut[pid]

    u = d_map * (K0 + math.sin(math.radians(tilt_deg)) * TILT_GAIN)
    spec_rgb = sheen(u.astype(np.float32))
    out = GOLD * ETA0 * cov[..., None]
    out = out + np.where((d_map > 0)[..., None], spec_rgb * ETA1 * cov[..., None], 0.0)
    out = out / ETA0                      # normalise so plain gold reads as gold
    img = np.clip(ip.linear_to_srgb(np.clip(out, 0, 1)), 0, 1)
    return Image.fromarray((img * 255).astype(np.uint8)).resize(
        (px, px), Image.LANCZOS
    )


def strip(tiles, labels, title, tile_px, sub=""):
    pad, lab_h = 4, (56 if sub else 30)
    w = len(tiles) * (tile_px + pad) + pad
    im = Image.new("RGB", (w, tile_px + pad * 2 + lab_h), BG)
    d = ImageDraw.Draw(im)
    for i, t in enumerate(tiles):
        im.paste(t, (pad + i * (tile_px + pad), pad))
        d.text((pad + i * (tile_px + pad) + 2, tile_px + pad + 3), labels[i],
               fill=DIM, font=font(12))
    d.text((pad, tile_px + pad + 19), title, fill=FG, font=font(13))
    if sub:
        d.text((pad, tile_px + pad + 36), sub, fill=DIM, font=font(11))
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
    return im


def main() -> int:
    out = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
    out.mkdir(parents=True, exist_ok=True)
    TILTS = (0.0, 1.0, 2.0)
    TL = [f"tilt {t:+.0f}°" for t in TILTS]

    print("three variants at %g mm ..." % PORTRAIT_MM)
    rows = []
    for cid, mode, title, sub in [
        ("PORT-P", "plain", "PORT PLAIN  — the control",
         "identical code path, empty period field: any difference below is the colour"),
        ("PORT-H", "hue", "PORT HUE  — period from every pixel's own hue",
         "a gold photograph that shifts colour with view; 12 rungs over the whole frame"),
        ("PORT-Z", "zones", "PORT ZONES  — flowers by hue, sweater and glasses authored",
         "carpet takes one rung per petal colour; sweater mid-ladder; frames at the blue end"),
    ]:
        plan = _plan(mode)
        tiles = [render_cell(plan, PORTRAIT_MM, t, px=300) for t in TILTS]
        rows.append(strip(tiles, TL, title, 300, sub))
    stack(rows, out / "witness_variants.png")

    print("DoE axes ...")
    rows = []
    for title, sub, cells in [
        ("colour base period — the C2 gate",
         "below 4.82 um the ladder's blue end needs sub-floor lines; wider period = narrower colour",
         [(f"{d:g} um", _plan("zones", base_period_um=d), {}) for d in (4.0, 5.0, 6.5, 8.0)]),
        ("ladder spread — how far apart two zones read",
         "red/blue period ratio; 1.9 separates more but pushes the blue end under the floor",
         [(f"{s:.2f}", _plan("zones", spread=s), {}) for s in (1.20, 1.45, 1.90)]),
        ("colour coarsening — noise versus intent",
         "patch size of the period field; too fine and the colour averages to a warm cast",
         [(f"{p} px", _plan("zones", coarsen_px=p), {}) for p in (3, 7, 14)]),
        ("screen pitch — tone depth against line visibility",
         "tone steps capped at period/2 um by the litho floor",
         [(f"{p:g} um", _plan("zones"), {"line_period_um": p,
                                         "tone_steps": min(22, int(p / 2))})
          for p in (20.0, 30.0, 44.0, 60.0)]),
    ]:
        tiles, labs = [], []
        for lab, plan, kw in cells:
            tiles.append(render_cell(plan, 14.0, 0.0, px=210, **kw))
            labs.append(lab)
        rows.append(strip(tiles, labs, title, 210, sub))
    stack(rows, out / "witness_doe.png")

    print("scale ladder ...")
    tiles, labs = [], []
    for mm in (4.0, 8.0, 14.0, 30.0):
        tiles.append(render_cell(_plan("zones"), mm, 0.0, px=210))
        labs.append(f"{mm:g} mm · {int(mm*1000/87)} eye-cells")
    stack([strip(tiles, labs, "image scale — when it stops being a thumbnail",
                 210, "shown at equal size; on the plate they differ 7.5x")],
          out / "witness_scale.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
