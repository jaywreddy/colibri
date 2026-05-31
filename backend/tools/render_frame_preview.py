"""Smoke-test renderer: generate a frame on a rectangular plate and save a PNG.

Useful for eyeballing whether the algorithm produces gold engraving that
actually reads as vines + foliage rather than tangled noise. Not used by the
service — pure dev tool.

Usage:
    uv run python tools/render_frame_preview.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# Allow running from anywhere via `uv run python tools/...` without
# package installation.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.patterns.frames import RectFrame, generate_frame, scene_to_multipolygon
from app.patterns.frames.api import FrameParams
from app.rasterize import rasterize


def main() -> int:
    rect = RectFrame(width_um=2400.0, height_um=1600.0)
    params = FrameParams(
        algorithm="colonize",
        theme_slug="esmeralda",
        seed=2024,
        density=1.0,
        bloom=0.7,
        foliage=0.6,
    )
    scene = generate_frame(rect, params)
    print(f"scene: {len(scene.segments)} segs, {len(scene.flowers)} flowers, "
          f"{len(scene.leaves)} leaves, max_t={scene.max_t}")

    mask = scene_to_multipolygon(scene, rect, params)
    print(f"mask area: {mask.area:.0f} um^2, bounds: {mask.bounds}")

    img = rasterize(mask, (rect.width_um, rect.height_um), pitch_um=2.0)
    out = Path(__file__).resolve().parent / "frame_preview.png"
    img.save(out)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
