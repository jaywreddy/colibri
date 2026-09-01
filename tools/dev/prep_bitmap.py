"""Prepare a photograph as a halftone-ready bitmap asset.

    uv run --directory backend python ../tools/dev/prep_bitmap.py \
        ../my-photo.jpg portrait-flores --crop 0.44,0.14,0.46 --steps 22

Writes ``backend/assets/bitmaps/<name>.png``, after which the name appears as a
choice on the ``bitmap-halftone`` pattern.

Why a prep step exists at all: a line screen throws away everything below its
own pitch, cannot print 0% or 100% duty, and quantizes what is left into ~22
levels. See ``app.patterns.bitmap.imageprep`` for what each stage undoes.

Pick the crop with ``--contact`` first: it writes a sheet of candidate square
crops so you can choose one by eye rather than by arithmetic.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "backend"))

from app.patterns.bitmap.imageprep import (  # noqa: E402
    PrepSpec,
    load_gray,
    local_contrast,
    prep_darkness,
    prepare_asset,
    screen_period_for,
)

ASSETS = REPO / "backend" / "assets" / "bitmaps"


def _crop(text: str) -> tuple[float, float, float]:
    parts = [float(p) for p in text.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("--crop wants x,y,side as fractions of the WIDTH")
    return parts[0], parts[1], parts[2]


def _contact_sheet(src: Path, out: Path) -> None:
    """Candidate square crops, so the choice is made by eye."""
    from PIL import Image, ImageDraw

    im = Image.open(src)
    w, h = im.size
    cands = {
        "A head+shoulders": (0.44, 0.14, 0.46),
        "B upper body": (0.38, 0.13, 0.58),
        "C tight": (0.52, 0.17, 0.30),
        "D wide": (0.10, 0.15, 0.85),
    }
    tiles = []
    for name, (fx, fy, fs) in cands.items():
        x0, y0, side = int(fx * w), int(fy * h), int(fs * w)
        tiles.append((name, (fx, fy, fs),
                      im.crop((x0, y0, min(w, x0 + side), min(h, y0 + side)))
                        .resize((260, 260), Image.LANCZOS)))
    sheet = Image.new("RGB", (len(tiles) * 264 + 4, 292), (13, 17, 19))
    d = ImageDraw.Draw(sheet)
    for i, (name, frac, t) in enumerate(tiles):
        sheet.paste(t, (4 + i * 264, 4))
        d.text((6 + i * 264, 268), f"{name}  --crop {frac[0]},{frac[1]},{frac[2]}",
               fill=(214, 224, 227))
    sheet.save(out)
    print(f"contact sheet -> {out}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src", type=Path)
    ap.add_argument("name", nargs="?", help="asset name (no extension)")
    ap.add_argument("--crop", type=_crop, default=None,
                    help="x,y,side as fractions of the source WIDTH")
    ap.add_argument("--steps", type=int, default=22,
                    help="tone steps of the target screen (default 22 -> a 44um screen)")
    ap.add_argument("--size", type=int, default=1400)
    ap.add_argument("--unsharp", type=float, default=0.75, help="0 disables")
    ap.add_argument("--falloff", type=float, default=0.30, help="0 disables")
    ap.add_argument("--gain", type=float, default=0.0,
                    help="dot-gain compensation, measured off the witness plate's duty ladder")
    ap.add_argument("--no-dither", action="store_true")
    ap.add_argument("--contact", action="store_true",
                    help="write a candidate-crop sheet and stop")
    a = ap.parse_args()

    if not a.src.is_file():
        print(f"no such file: {a.src}", file=sys.stderr)
        return 2

    if a.contact:
        _contact_sheet(a.src, a.src.with_name(a.src.stem + "-crops.png"))
        return 0

    if not a.name:
        print("give an asset name (or use --contact to pick a crop first)", file=sys.stderr)
        return 2

    spec = PrepSpec(
        tone_steps=a.steps,
        unsharp_amount=a.unsharp,
        falloff=a.falloff,
        gain=a.gain,
        dither=not a.no_dither,
    )
    dst = ASSETS / f"{a.name}.png"
    stats = prepare_asset(a.src, dst, crop=a.crop, size=a.size, spec=spec)

    raw = load_gray(a.src, crop=a.crop, size=a.size)
    before, after = local_contrast(raw), local_contrast(prep_darkness(raw, spec))
    print(f"wrote {dst}")
    print(f"  screen        {screen_period_for(a.steps):.0f} um / {a.steps} steps "
          f"({screen_period_for(a.steps) / a.steps:.2f} um finest band)")
    print(f"  duty window   {stats['window_lo']:.3f} .. {stats['window_hi']:.3f}")
    print(f"  tone          mean {stats['mean']:.3f}  sd {stats['std']:.3f}  "
          f"range {stats['min']:.3f}-{stats['max']:.3f}")
    print(f"  local contrast {before:.4f} -> {after:.4f}  "
          f"({(after / before - 1.0) * 100:+.0f}%)")
    print(f"\nnow selectable as image='{a.name}.png' on the bitmap-halftone pattern.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
