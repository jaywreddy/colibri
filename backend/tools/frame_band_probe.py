"""Standalone band/frame probe for tuning the perimeter foliage frame.

Renders (a) a full-face frame at production-ish scale and (b) a wide crop of
one edge band, so we can judge whether the perimeter reads as a deliberate,
densely-filled engraved FRAME with readable motifs.

Motifs are tinted by their angle bucket (via _frame_level_for) so the layered
understory reads distinctly from the primary vine layer in the probe image —
this is a DIAGNOSTIC tint only; the real plate encodes the bucket as graylevel
for the moiré shader.

Usage (one-shot, tiny; safe under the machine budget):
    uv run python tools/frame_band_probe.py <out_prefix> [seed]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.patterns.frames import RectFrame, generate_frame  # noqa: E402
from app.patterns.frames.api import FrameParams, render_scene_to_image  # noqa: E402
from app import plates as P  # noqa: E402


# Production-ish geometry: 30 mm plate, ~9% weld inset each side already baked
# into active dims; band ~12% of the shorter side.
ACTIVE_UM = 27000.0
BAND_UM = 0.13 * ACTIVE_UM
PITCH_UM = ACTIVE_UM / 1500.0  # ~18 um, matches plate_pitch order


# Gold-ish tint per bucket so the layered read is visible in the probe. Warm
# golds for the canopy, cooler/greener for the understory buckets.
BUCKET_TINT = {
    0: (232, 196, 92),
    1: (214, 176, 74),
    2: (198, 158, 66),
    3: (176, 150, 80),
    4: (150, 170, 96),   # greener — understory ferns/tendrils tend here
    5: (128, 158, 84),
    6: (110, 150, 76),   # (if a 7th bucket is ever added)
}


def _colorize(gray: Image.Image) -> Image.Image:
    """Map the bucket-graylevel L image to an RGB gold-on-black probe frame."""
    g = np.asarray(gray, dtype=np.int32)
    h, w = g.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    # background dark
    out[:] = (14, 12, 10)
    # decode bucket from level: b = round((L-BUCKET0)/STEP)
    nz = g > 0
    b = np.round((g - P.FRAME_BUCKET0) / P.FRAME_BUCKET_STEP).astype(np.int32)
    b = np.clip(b, 0, 6)
    for bucket, tint in BUCKET_TINT.items():
        m = nz & (b == bucket)
        out[m] = tint
    return Image.fromarray(out, "RGB")


# Named recipes: distinct density/gradient personalities for the frame band.
RECIPES = {
    # A — Lush: heavy understory, strong outer gradient, full corners.
    "A": dict(density=1.0, bloom=0.5, foliage=0.75, edge_gradient=1.0,
              understory=1.15, border_vine=1.0, corner_fans=1.0),
    # B — Tailored: cleaner inner edge (lighter understory, firmer border
    #     line), moderate gradient — reads as a crisp engraved border.
    "B": dict(density=0.85, bloom=0.55, foliage=0.55, edge_gradient=0.8,
              understory=0.75, border_vine=1.25, corner_fans=0.9),
    # C — Ornate corners: airier edges, bold corner fans, gentle gradient.
    "C": dict(density=0.7, bloom=0.7, foliage=0.6, edge_gradient=0.55,
              understory=0.6, border_vine=0.9, corner_fans=1.35),
    # PROD — exactly the production FrameSpec defaults (density/bloom/foliage
    # = 1.0/0.6/0.6) with the new-knob DEFAULTS applied, so we can see what a
    # box face actually renders with no per-face overrides.
    "PROD": dict(density=1.0, bloom=0.6, foliage=0.6, edge_gradient=0.8,
                 understory=0.85, border_vine=1.15, corner_fans=1.0),
}


def _params(seed: int, recipe: str) -> FrameParams:
    r = RECIPES[recipe]
    return FrameParams(
        algorithm="colonize", theme_slug="esmeralda", seed=seed,
        frame_width_um=BAND_UM, **r,
    )


def render_recipe(seed: int, recipe: str, prefix: Path) -> None:
    rect = RectFrame(width_um=ACTIVE_UM, height_um=ACTIVE_UM)
    params = _params(seed, recipe)
    scene = generate_frame(rect, params)
    print(f"[{recipe} seed={seed}] segs={len(scene.segments)} "
          f"flowers={len(scene.flowers)} leaves={len(scene.leaves)}")
    img = render_scene_to_image(scene, rect, params, PITCH_UM,
                                level_fn=P._frame_level_for)
    rgb = _colorize(img)
    rgb.save(prefix.with_name(prefix.name + "_full.png"))
    # Top band strip.
    W, _H = rgb.size
    strip_h_px = int(round(BAND_UM * 1.3 / PITCH_UM))
    strip = rgb.crop((0, 0, W, strip_h_px))
    strip = strip.resize((2200, int(2200 * strip.size[1] / strip.size[0])),
                         Image.LANCZOS)
    strip.save(prefix.with_name(prefix.name + "_strip.png"))
    print(f"  wrote {prefix.name}_full.png + _strip.png")


def main() -> int:
    prefix = Path(sys.argv[1])
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 7
    recipe = sys.argv[3] if len(sys.argv) > 3 else "A"
    prefix.parent.mkdir(parents=True, exist_ok=True)
    render_recipe(seed, recipe, prefix)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
