"""Draw the demo bitmaps consumed by the bitmap-halftone pattern.

Outputs two ~512x512 grayscale PNGs under backend/assets/bitmaps/ that act as
tasteful stand-ins for the user's own photos:

  andes-dawn.png  layered mountain ridges under a dawn sky with a sun disk
  capybara.png    chunky capybara side profile on a soft radial-gradient ground

Pure Pillow drawing, runs in milliseconds. Run once and commit the outputs:

    cd backend && uv run python ../tools/dev/gen_demo_bitmaps.py

The halftone generator auto-contrasts whatever it loads, so exact gray values
here are aesthetic, not load-bearing.
"""
from __future__ import annotations

import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

SIZE = 512
OUT_DIR = Path(__file__).resolve().parents[2] / "backend" / "assets" / "bitmaps"


def _ridge_points(
    y_base: float,
    amp: float,
    freq: float,
    phase: float,
) -> list[tuple[float, float]]:
    """A mountain ridgeline as a two-harmonic sine, closed to the bottom edge."""
    pts = [
        (
            x,
            y_base
            - amp
            * (
                0.62 * math.sin(freq * x / SIZE * 2 * math.pi + phase)
                + 0.38 * math.sin(2.7 * freq * x / SIZE * 2 * math.pi + 1.9 * phase)
            ),
        )
        for x in range(0, SIZE + 1, 4)
    ]
    return pts + [(SIZE, SIZE), (0, SIZE)]


def andes_dawn() -> Image.Image:
    img = Image.new("L", (SIZE, SIZE), 0)
    draw = ImageDraw.Draw(img)

    # Sky: dark zenith brightening toward the horizon (dawn glow).
    horizon = 0.52 * SIZE
    for y in range(SIZE):
        t = min(1.0, y / horizon)
        draw.line([(0, y), (SIZE, y)], fill=int(80 + 155 * t))

    # Sun disk with a soft halo (the final blur feathers both edges).
    cx, cy = 0.63 * SIZE, 0.36 * SIZE
    draw.ellipse([cx - 70, cy - 70, cx + 70, cy + 70], fill=215)
    draw.ellipse([cx - 42, cy - 42, cx + 42, cy + 42], fill=252)

    # Three ridges, darker toward the foreground.
    draw.polygon(_ridge_points(0.58 * SIZE, 30, 1.6, 0.7), fill=150)
    draw.polygon(_ridge_points(0.70 * SIZE, 45, 1.1, 2.6), fill=105)
    draw.polygon(_ridge_points(0.85 * SIZE, 55, 0.8, 5.1), fill=55)

    return img.filter(ImageFilter.GaussianBlur(1.4))


def capybara() -> Image.Image:
    img = Image.new("L", (SIZE, SIZE), 205)
    draw = ImageDraw.Draw(img)

    # Soft radial-gradient ground, brightest under the animal.
    gx, gy = 256, 400
    for r in range(520, 0, -8):
        v = int(150 + 70 * (1 - r / 520))
        draw.ellipse([gx - r, gy - r * 0.55, gx + r, gy + r * 0.55], fill=v)

    # Ground shadow, then the capybara itself (side profile, facing right).
    draw.ellipse([120, 390, 410, 435], fill=130)
    body = 70
    draw.rounded_rectangle([150, 340, 192, 415], radius=16, fill=body)   # hind leg
    draw.rounded_rectangle([225, 345, 262, 415], radius=14, fill=body)   # mid leg
    draw.rounded_rectangle([300, 345, 338, 415], radius=14, fill=body)   # fore leg
    draw.rounded_rectangle([120, 210, 390, 375], radius=80, fill=body)   # chunky body
    draw.rounded_rectangle([330, 190, 445, 300], radius=45, fill=body)   # head
    draw.rounded_rectangle([400, 230, 462, 300], radius=25, fill=body)   # blunt snout
    draw.ellipse([352, 172, 390, 212], fill=60)                          # small ear
    draw.ellipse([396, 230, 412, 246], fill=190)                         # eye
    draw.ellipse([444, 252, 456, 266], fill=45)                          # nostril

    return img.filter(ImageFilter.GaussianBlur(1.2))


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, fn in [("andes-dawn.png", andes_dawn), ("capybara.png", capybara)]:
        path = OUT_DIR / name
        fn().save(path)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
