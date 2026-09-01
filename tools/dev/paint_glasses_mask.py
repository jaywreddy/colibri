"""Author the spectacle-frame mask for the reference portrait.

Why this exists rather than a colour rule: the frames are teal only where they
are lit. The shadowed left rim measures val 0.09, which is exactly where the
hair sits, so no threshold can hold one and drop the other — a colour rule
catches the two top rims and leaves the picture looking like a half-finished
selection. A spectacle frame is a closed shape, so the honest thing is to say
where it is.

The ELLIPSES are not eyeballed. The rule's own detection is run first, its
connected components give the rims, and each lens ellipse is fitted to a rim's
bounding box; the bridge is the span between them. So the mask is derived from
the photograph and reproducible, and the hand-authored part is only the
knowledge that a spectacle frame is two rings and a bar.

    uv run --directory backend python ../tools/dev/paint_glasses_mask.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "backend"))

from app.patterns.bitmap.colourplan import (  # noqa: E402
    ASSET_DIR, PAULA_ZONES, _hsv, load_rgb,
)

SRC = Path(__file__).resolve().parents[2] / "PXL_20250920_201250581.jpg"
CROP = (0.040, 0.165, 0.825)
SIZE = 1400
OUT = ASSET_DIR / "portrait-paula.glasses.png"
STROKE_FRAC = 0.055
"""Rim thickness as a fraction of the lens width. The frames measure about 4 px
of 900 in the source, which is under one halftone line at any cell size on the
plate; drawn at 5.5% of the lens they are 3-4 lines and can actually carry the
sub-grating that makes them a colour."""


def main() -> int:
    rgb = load_rgb(SRC, crop=CROP, size=SIZE)
    hue, sat, val = _hsv(rgb)
    rule = next(r for r in PAULA_ZONES.rules if r.name == "glasses")

    h, w = hue.shape
    x0, y0, x1, y1 = rule.bbox
    box = np.zeros(hue.shape, dtype=bool)
    box[int(y0 * h): int(y1 * h), int(x0 * w): int(x1 * w)] = True
    lo, hi = rule.hue_deg
    m = (box & (hue >= lo) & (hue <= hi)
         & (sat >= rule.sat[0]) & (val >= rule.val[0]))
    m = ndimage.binary_closing(m, structure=np.ones((5, 5), dtype=bool))
    lab, n = ndimage.label(m)
    if n == 0:
        print("no rim detected — cannot fit")
        return 1
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    keep = np.argsort(sizes)[::-1][:2]
    boxes = []
    for k in keep:
        if sizes[k] < 40:
            continue
        ys, xs = np.nonzero(lab == k)
        boxes.append((xs.min(), ys.min(), xs.max(), ys.max()))
    if len(boxes) < 2:
        print(f"only {len(boxes)} rim(s) found; need two")
        return 1
    boxes.sort(key=lambda b: b[0])
    print("fitted rims:", boxes)

    # A detected rim is only the LIT arc, so its own width under-measures its
    # lens (60 and 35 px against a real ~90). What IS well measured is the
    # OUTER span of the pair, so the two lenses are derived from that: a
    # spectacle front is two equal lenses either side of a short bridge, and
    # 0.43 / 0.14 / 0.43 of the span is that proportion.
    span_x0 = min(b[0] for b in boxes)
    span_x1 = max(b[2] for b in boxes)
    span = span_x1 - span_x0
    lens_w = span * 0.43
    lens_h = lens_w * 0.92
    top = float(np.mean([b[1] for b in boxes]))
    cy = top + lens_h * 0.42
    stroke = max(3, int(round(lens_w * STROKE_FRAC)))
    print(f"span {span} px -> lens {lens_w:.0f} x {lens_h:.0f}, stroke {stroke}")

    img = Image.new("L", (w, h), 0)
    d = ImageDraw.Draw(img)
    lx = span_x0 + lens_w / 2.0
    rx = span_x1 - lens_w / 2.0
    for cx in (lx, rx):
        d.ellipse([cx - lens_w / 2, cy - lens_h / 2,
                   cx + lens_w / 2, cy + lens_h / 2], outline=255, width=stroke)
    d.line([lx + lens_w / 2, cy - lens_h * 0.22,
            rx - lens_w / 2, cy - lens_h * 0.22], fill=255, width=stroke)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    img.save(OUT)
    a = np.asarray(img) > 127
    print(f"wrote {OUT}  ({a.sum()} px, {a.mean()*100:.3f}% of frame)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
