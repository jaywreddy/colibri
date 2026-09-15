"""Re-render the AUTHORED colour plans that ship beside the prepared photographs.

Every photo on the box carries ``<image>.colour.json`` — a ``ColourPlan``
(``colourplan.plan_to_json``) written for that picture, saying which REGIONS
take which rung of the period ladder. ``photo.colour_mode="authored"`` is what
loads it on the fab path; this tool is how the plan is judged before it gets
there.

What it draws, per photo: the prepared source, and beside it a FALSE-COLOUR
proof of the plan — gold where the bands stay solid, spectral hue where they are
sub-grated (hue from the rung's period, red end to blue end), and dark where the
picture has dissolved to bare glass. It is a map of the plan, not a simulation
of the plate: the real thing is a gold mirror with a spectrum riding on it and
only shows its hue at the right view (see ``colourzone`` on lighting), so a
literal render would hide exactly the thing you are trying to review.

The coverage comes from ``photo.photo_coverage`` — the shipping tone model, the
same numbers the SVG and GDS bakes write — so the proof cannot drift from the
mask. That also means the edge fade is included, and colour is already zeroed
wherever the fade weight fell below a half.

    uv run --directory backend python ../tools/dev/render_colour_plans.py OUTDIR [IMAGE ...]

Outputs OUTDIR/<image>_plan.png per photo, colour_plans.json (the per-rule and
per-rung report for each), and colour_plans_contact.png. Light: one 700 px
image at a time, a few seconds each.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "backend"))

from app.patterns.bitmap import colourplan as cp  # noqa: E402
from app.patterns.bitmap import photo as ph  # noqa: E402

N_PX = 700
BG = (13, 17, 19)
FG = (214, 224, 227)
DIM = (150, 165, 172)

# What a solid (un-sub-grated) gold band looks like in the proof.
GOLD = np.array([1.00, 0.78, 0.34], dtype=np.float32)
# How much of a coloured band's look is still the specular gold. At 50% duty
# eta_0 = 0.25 against eta_1 = 0.101, so the spectrum is a minority partner —
# drawing it at full saturation would promise a colour the plate cannot give.
SPECULAR_MIX = 0.42


def font(sz: int = 12):
    try:
        return ImageFont.truetype("arialbd.ttf", sz)
    except Exception:
        return ImageFont.load_default()


def _hsv_to_rgb(h: np.ndarray, s: float, v: float) -> np.ndarray:
    """Vectorised HSV -> RGB for a hue field in degrees."""
    h = (np.asarray(h, np.float32) % 360.0) / 60.0
    i = np.floor(h).astype(np.int32)
    f = h - i
    p, q, t = v * (1 - s), v * (1 - s * f), v * (1 - s * (1 - f))
    full = np.full_like(h, v, dtype=np.float32)
    table = [
        (full, t, np.full_like(h, p)),
        (q, full, np.full_like(h, p)),
        (np.full_like(h, p), full, t),
        (np.full_like(h, p), q, full),
        (t, np.full_like(h, p), full),
        (full, np.full_like(h, p), q),
    ]
    out = np.zeros(h.shape + (3,), np.float32)
    for k, (r, g, b) in enumerate(table):
        m = i % 6 == k
        for c, ch in enumerate((r, g, b)):
            out[..., c] = np.where(m, ch, out[..., c])
    return out


def period_to_hue(period_um: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Rung period -> display hue. Long period = red, short = blue.

    The same direction ``colourplan.hue_to_scale`` maps in, so the proof shows
    the photograph's own colour ORDER rather than inverting it.
    """
    t = np.clip((np.asarray(period_um, np.float32) - lo) / max(1e-6, hi - lo), 0, 1)
    return (1.0 - t) * 250.0


def proof(image: str, fade_start: float = 0.50, fade_gate: float = 0.94) -> Image.Image:
    """False-colour map of what the authored plan does to this picture."""
    cov, ids, periods = ph.photo_coverage(
        image, fade_start, fade_gate, ph.DEFAULT_TONE_STEPS, N_PX,
        colour_mode="authored",
    )
    if periods:
        lut = np.zeros(max(periods) + 1, np.float32)
        for i, d in periods.items():
            lut[i] = d
        per = lut[np.clip(ids, 0, lut.size - 1)]
        lo, hi = min(periods.values()), max(periods.values())
    else:
        per = np.zeros_like(cov)
        lo = hi = 1.0

    spectral = _hsv_to_rgb(period_to_hue(per, lo, hi), 0.95, 1.0)
    tint = SPECULAR_MIX * GOLD + (1.0 - SPECULAR_MIX) * spectral
    colour = np.where((ids > 0)[..., None], tint, GOLD[None, None, :])
    img = np.clip(colour * cov[..., None], 0, 1) ** (1 / 2.2)
    return Image.fromarray((img * 255).astype(np.uint8))


def source(image: str) -> Image.Image:
    im = Image.open(ph.photo_path(image)).convert("RGB")
    return im.resize((N_PX, N_PX), Image.LANCZOS)


def report(image: str) -> dict:
    plan = ph.colour_plan("authored", image)
    rgb = cp.load_rgb(ph.photo_path(image), size=N_PX)
    _ids, periods, rep = cp.build_period_field(rgb, plan)
    # What the FADE leaves, which is the number that matters on the plate.
    _cov, faded, _ = ph.photo_coverage(
        image, 0.50, 0.94, ph.DEFAULT_TONE_STEPS, N_PX, colour_mode="authored"
    )
    rep["frac_coloured_after_fade"] = round(float((faded > 0).mean()), 4)
    rep["min_period_um"] = round(min(periods.values()), 4)
    rep["name"] = plan.name
    return rep


def pair(image: str, rep: dict) -> Image.Image:
    src, prf = source(image), proof(image)
    im = Image.new("RGB", (10 + 2 * (N_PX + 10), N_PX + 54), BG)
    d = ImageDraw.Draw(im)
    im.paste(src, (10, 10))
    im.paste(prf, (20 + N_PX, 10))
    d.text((10, N_PX + 16), f"{image} — prepared source", fill=DIM, font=font(13))
    d.text(
        (20 + N_PX, N_PX + 16),
        f"{rep['name']}: {rep['frac_coloured_after_fade']*100:.1f}% coloured, "
        f"{rep['rungs_used']} rungs, blue end {rep['min_period_um']:g} um",
        fill=FG,
        font=font(13),
    )
    d.text(
        (20 + N_PX, N_PX + 34),
        "  ".join(
            f"{r['name']}={r['frac']*100:.1f}%" for r in rep["rules"] if r["frac"] > 0.001
        ) or "(no rules fired)",
        fill=DIM,
        font=font(11),
    )
    return im


def contact(out: Path, images: list[str], cols: int = 2) -> Path:
    tiles = [Image.open(out / f"{im}_plan.png") for im in images]
    tw, th = tiles[0].size
    rows = (len(tiles) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * tw, rows * th), BG)
    for i, t in enumerate(tiles):
        sheet.paste(t, ((i % cols) * tw, (i // cols) * th))
    p = out / "colour_plans_contact.png"
    sheet.save(p)
    return p


def main() -> int:
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    images = sys.argv[2:] or [
        n for n in ph.available_photos() if ph.authored_plan_path(n).is_file()
    ]
    reports = {}
    for image in images:
        rep = report(image)
        reports[image] = rep
        pair(image, rep).save(out / f"{image}_plan.png")
        print(
            f"{image}: {rep['frac_coloured_after_fade']*100:5.1f}% coloured, "
            f"{rep['rungs_used']} rungs, printable={rep['all_used_rungs_printable']}"
        )
        for r in rep["rules"]:
            print(f"    {r['name']:<22} {r['mode']:<6} {r['frac']*100:6.2f}%")
    (out / "colour_plans.json").write_text(
        json.dumps(reports, indent=1), encoding="utf-8"
    )
    print("contact sheet:", contact(out, images))
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
