"""Standalone wreath probe — renders the ordered garland wreath at plate scale.

Renders (a) a full-face frame and (b) a wide top-edge strip so we can judge the
wreath read: continuous guiding vine, shingled leaf ranks, mirror symmetry,
resolved corners. Motifs are tinted by angle bucket (diagnostic only).

Usage (one-shot, tiny; safe under the machine budget):
    uv run python tools/wreath_probe.py <out_prefix> <style> [seed]
    style in: laurel | garland | clusters
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


ACTIVE_UM = 27000.0
BAND_UM = 0.13 * ACTIVE_UM
PITCH_UM = ACTIVE_UM / 1500.0

BUCKET_TINT = {
    0: (232, 196, 92),
    1: (214, 176, 74),
    2: (198, 158, 66),
    3: (176, 150, 80),
    4: (150, 170, 96),
    5: (128, 158, 84),
    6: (110, 150, 76),
}


def _colorize(gray: Image.Image) -> Image.Image:
    g = np.asarray(gray, dtype=np.int32)
    h, w = g.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    out[:] = (14, 12, 10)
    nz = g > 0
    b = np.round((g - P.FRAME_BUCKET0) / P.FRAME_BUCKET_STEP).astype(np.int32)
    b = np.clip(b, 0, 6)
    for bucket, tint in BUCKET_TINT.items():
        m = nz & (b == bucket)
        out[m] = tint
    return Image.fromarray(out, "RGB")


def main() -> int:
    prefix = Path(sys.argv[1])
    style = sys.argv[2] if len(sys.argv) > 2 else "garland"
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 7
    prefix.parent.mkdir(parents=True, exist_ok=True)

    rect = RectFrame(width_um=ACTIVE_UM, height_um=ACTIVE_UM)
    params = FrameParams(
        algorithm="wreath", theme_slug="esmeralda", seed=seed,
        frame_width_um=BAND_UM, density=1.0, bloom=0.6, foliage=0.6,
        wreath_style=style,
    )
    scene = generate_frame(rect, params)
    print(f"[wreath {style} seed={seed}] segs={len(scene.segments)} "
          f"flowers={len(scene.flowers)} leaves={len(scene.leaves)}")
    img = render_scene_to_image(scene, rect, params, PITCH_UM,
                                level_fn=P._frame_level_for)
    rgb = _colorize(img)
    rgb.save(prefix.with_name(prefix.name + "_full.png"))

    W, _H = rgb.size
    strip_h_px = int(round(BAND_UM * 1.5 / PITCH_UM))
    strip = rgb.crop((0, 0, W, strip_h_px))
    strip = strip.resize((2200, int(2200 * strip.size[1] / strip.size[0])),
                         Image.LANCZOS)
    strip.save(prefix.with_name(prefix.name + "_strip.png"))
    print(f"  wrote {prefix.name}_full.png + _strip.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
