"""Track B leaf/wreath probe: render band, measure gold fill fraction.

Renders a full-face frame + a top-edge strip for a chosen algorithm/style,
tints motifs by angle bucket (diagnostic only), and reports the gold-area
fraction WITHIN the perimeter band annulus (the number the brief wants:
before / broken / fixed).

Usage:
    uv run python tools/leaf_probe.py <out_prefix> <mode> [seed]

where <mode> is one of:
    colonize            -- pre-wreath look (old default dials)
    wreath:laurel       -- current default wreath
    wreath:garland
    wreath:clusters
    wreath:garland2     -- (the reworked lush garland, once wired)
"""
from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
from PIL import Image

BACKEND = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND))

# The full ``app.patterns`` package __init__ pulls in unrelated modules (one of
# which may be mid-edit in a sibling track and fail to import). We only need the
# frames subtree, so we register lightweight namespace packages for ``app`` and
# ``app.patterns`` WITHOUT running their real __init__, then load the frames
# package normally (its relative imports resolve against that chain).
def _ns(name: str, path: Path) -> types.ModuleType:
    mod = types.ModuleType(name)
    mod.__path__ = [str(path)]
    mod.__package__ = name
    sys.modules[name] = mod
    return mod

_ns("app", BACKEND / "app")
_ns("app.patterns", BACKEND / "app" / "patterns")
import app.patterns.frames as frames  # noqa: E402
from app.patterns.frames import RectFrame, generate_frame  # noqa: E402
from app.patterns.frames.api import FrameParams, render_scene_to_image  # noqa: E402

# Frame-band angle-bucket constants (mirror app.plates so the probe tints motifs
# by bucket exactly as the plate compositor would). Kept local so the probe does
# not import the heavy app.plates module.
_FRAME_BUCKET0 = 96
_FRAME_BUCKET_STEP = 14
_N_FRAME_BUCKETS = 6
_MOTIF_ANGLE_BUCKET = {
    "vine": 2,
    "heliconia": 4, "orchid": 0, "anthurium": 3, "coffee": 1,
    "wax_palm": 5, "plantain": 1, "fern": 4, "philodendron": 2,
}


def _frame_level_for(key: str) -> int:
    b = max(0, min(_N_FRAME_BUCKETS - 1,
                   _MOTIF_ANGLE_BUCKET.get(key, _MOTIF_ANGLE_BUCKET["vine"])))
    return _FRAME_BUCKET0 + b * _FRAME_BUCKET_STEP


class P:  # tiny shim so existing references keep working
    FRAME_BUCKET0 = _FRAME_BUCKET0
    FRAME_BUCKET_STEP = _FRAME_BUCKET_STEP
    _frame_level_for = staticmethod(_frame_level_for)


ACTIVE_UM = 27000.0
BAND_UM = 0.13 * ACTIVE_UM
PITCH_UM = ACTIVE_UM / 1500.0  # ~18 um


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


def _band_fill_fraction(gray: Image.Image) -> float:
    """Gold-area fraction inside the perimeter band annulus (0..1)."""
    g = np.asarray(gray, dtype=np.int32)
    h, w = g.shape
    band_px = BAND_UM / PITCH_UM
    yy, xx = np.mgrid[0:h, 0:w]
    dist = np.minimum.reduce([xx, w - 1 - xx, yy, h - 1 - yy]).astype(np.float64)
    annulus = dist <= band_px
    gold = (g > 0) & annulus
    return float(gold.sum()) / float(max(1, annulus.sum()))


def _make_params(mode: str, seed: int) -> FrameParams:
    if mode == "colonize":
        # Pre-wreath default dials (the frame_band_probe PROD recipe = the
        # production FrameSpec defaults that the band shipped with).
        return FrameParams(
            algorithm="colonize", theme_slug="esmeralda", seed=seed,
            frame_width_um=BAND_UM,
            density=1.0, bloom=0.6, foliage=0.6, edge_gradient=0.8,
            understory=0.85, border_vine=1.15, corner_fans=1.0,
        )
    if mode.startswith("wreath:"):
        style = mode.split(":", 1)[1]
        return FrameParams(
            algorithm="wreath", theme_slug="esmeralda", seed=seed,
            frame_width_um=BAND_UM,
            density=1.0, bloom=0.6, foliage=0.6,
            understory=0.85, border_vine=1.15, corner_fans=1.0,
            wreath_style=style,
        )
    raise SystemExit(f"unknown mode {mode!r}")


def render(mode: str, seed: int, prefix: Path) -> None:
    rect = RectFrame(width_um=ACTIVE_UM, height_um=ACTIVE_UM)
    params = _make_params(mode, seed)
    scene = generate_frame(rect, params)
    img = render_scene_to_image(scene, rect, params, PITCH_UM,
                                level_fn=P._frame_level_for)
    frac = _band_fill_fraction(img)
    print(f"[{mode} seed={seed}] segs={len(scene.segments)} "
          f"flowers={len(scene.flowers)} leaves={len(scene.leaves)} "
          f"band_fill={frac:.3f}")
    rgb = _colorize(img)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    rgb.save(prefix.with_name(prefix.name + "_full.png"))
    W, _H = rgb.size
    strip_h_px = int(round(BAND_UM * 1.35 / PITCH_UM))
    strip = rgb.crop((0, 0, W, strip_h_px))
    strip = strip.resize((2200, int(2200 * strip.size[1] / strip.size[0])),
                         Image.LANCZOS)
    strip.save(prefix.with_name(prefix.name + "_strip.png"))
    print(f"  wrote {prefix.name}_full.png + _strip.png  band_fill={frac:.3f}")


def main() -> int:
    prefix = Path(sys.argv[1])
    mode = sys.argv[2] if len(sys.argv) > 2 else "wreath:laurel"
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 7
    render(mode, seed, prefix)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
