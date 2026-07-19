"""Bitmap -> gold halftone plate.

Turns an arbitrary grayscale photo/bitmap into a fabbable amplitude
line-screen: the front layer carries horizontal gold lines whose local duty
(band height within each line period) is proportional to image darkness, so
tone survives binarization the way newspaper halftones do. The back layer is
chosen per `back_mode` to make the substrate parallax ANIMATE the picture:

  none          front halftone only — a static gold photograph.
  carrier       uniform 50% grating at the same period/phase; tilting slides
                the carrier under the halftone so the whole image's tone
                shimmers/breathes with viewing angle.
  complement    halftone of the INVERTED image; head-on transmission is
                uniformly dark, tilting de-registers the two screens and the
                photo emerges bright-on-dark from nothing.
  phase_reveal  the front halftone again but with the carrier phase shifted
                half a period; parallax biases which copy dominates, so the
                image pulses/slides as the box is tilted.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from .._helpers import check_lattice_budget, empty_layer, raster_to_polygons
from ..base import GeneratedPattern, ParamSpec, Pattern, register

# Demo bitmaps live in the repo (drawn by tools/dev/gen_demo_bitmaps.py).
# Module-level Path so tests can monkeypatch the directory; every reader goes
# through the functions below, which look the global up at call time.
ASSETS_DIR = Path(__file__).resolve().parents[3] / "assets" / "bitmaps"

_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}

# Working-grid sizing: 8 cells per halftone line resolves the triangular
# duty profile (band heights quantize to 1/8 of the period), capped so the
# resampled darkness grid never exceeds ~2M float32 cells of numpy work.
CELLS_PER_LINE = 8
MAX_GRID = 1400


def available_bitmaps() -> list[str]:
    """Sorted image filenames under ASSETS_DIR (call-time lookup so tests can
    monkeypatch the directory)."""
    if not ASSETS_DIR.is_dir():
        return []
    return sorted(
        p.name for p in ASSETS_DIR.iterdir() if p.suffix.lower() in _IMAGE_SUFFIXES
    )


def _load_darkness(image: str, n_grid: int, angle_deg: str, invert: bool) -> np.ndarray:
    """Load `image` as an (n_grid, n_grid) float32 darkness map in [0, 1].

    Grayscale + auto-contrast so arbitrary photos use the full tonal range,
    optional invert, then the requested screen angle. The angle rotates the
    SOURCE IMAGE (np.rot90 for 90°, PIL rotate for 45° with expand=False and
    white fill = no gold at the corners) so the halftone lines themselves stay
    axis-aligned rasters — rotating the polygons instead would break the
    row-run merge in raster_to_polygons and explode the rectangle count.
    """
    path = ASSETS_DIR / image
    if not path.is_file():
        raise ValueError(
            f"bitmap {image!r} not found under {ASSETS_DIR} "
            f"(available: {', '.join(available_bitmaps()) or 'none'})"
        )
    img = ImageOps.autocontrast(Image.open(path).convert("L"))
    if invert:
        img = ImageOps.invert(img)
    if angle_deg == "45":
        img = img.rotate(45, resample=Image.BILINEAR, expand=False, fillcolor=255)
    img = img.resize((n_grid, n_grid), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    if angle_deg == "90":
        arr = np.rot90(arr)
    return 1.0 - arr  # darker = more gold


def _min_gold_band_duty(mask: np.ndarray, cell_um: float, line_period_um: float) -> float:
    """Smallest realized vertical gold band as a duty fraction of the line
    period (0.0 for an empty mask). Same pad/diff run trick as
    raster_to_polygons, applied down columns instead of along rows."""
    g = mask.T
    padded = np.zeros((g.shape[0], g.shape[1] + 2), dtype=np.int8)
    padded[:, 1:-1] = g
    d = np.diff(padded, axis=1)
    starts = np.nonzero(d == 1)[1]
    ends = np.nonzero(d == -1)[1]
    if starts.size == 0:
        return 0.0
    return float((ends - starts).min()) * cell_um / line_period_um


_BITMAP_CHOICES = available_bitmaps()


@register
class BitmapHalftone(Pattern):
    slug = "bitmap-halftone"
    name = "Bitmap halftone plate"
    description = (
        "Any grayscale photo becomes a gold line-screen halftone: each stripe "
        "of the front layer carries a gold band whose height tracks local "
        "image darkness, so the picture reads in transmitted light. The back "
        "layer turns substrate parallax into animation — a uniform carrier "
        "makes the tone shimmer with tilt, a complementary screen makes the "
        "photo appear out of darkness, and a half-period phase copy makes it "
        "pulse between registrations."
    )
    tags = ["halftone", "bitmap", "photo", "tilt-reveal"]
    tier = 1
    theme = "Global Travel"
    # Front x parallax-shifted back through the quartz: the back grating
    # de-registers against the halftone as the camera tilts, which is the
    # whole show for every non-none back_mode.
    render_recipe = "moire_interactive"
    params = [
        ParamSpec(
            "image",
            "Source bitmap",
            "choice",
            _BITMAP_CHOICES[0] if _BITMAP_CHOICES else "",
            choices=_BITMAP_CHOICES,
        ),
        ParamSpec("line_period_um", "Line period", "float", 20.0, 8.0, 80.0, 0.5, "μm"),
        ParamSpec("angle_deg", "Screen angle", "choice", "0", choices=["0", "45", "90"]),
        ParamSpec("extent_um", "Extent", "float", 2000.0, 500.0, 5000.0, 100.0, "μm"),
        ParamSpec(
            "back_mode",
            "Back layer",
            "choice",
            "carrier",
            choices=["none", "carrier", "complement", "phase_reveal"],
        ),
        ParamSpec("invert", "Invert tones", "bool", False),
    ]

    @classmethod
    def generate(
        cls,
        image: str = _BITMAP_CHOICES[0] if _BITMAP_CHOICES else "",
        line_period_um: float = 20.0,
        angle_deg: str = "0",
        extent_um: float = 2000.0,
        back_mode: str = "carrier",
        invert: bool = False,
    ) -> GeneratedPattern:
        extent = (extent_um, extent_um)

        # Working grid: CELLS_PER_LINE cells per halftone line (duty resolves
        # in 1/8 steps), capped at MAX_GRID for coarse-but-huge requests.
        n_grid = int(round(extent_um / (line_period_um / CELLS_PER_LINE)))
        n_grid = max(64, min(MAX_GRID, n_grid))
        cell_um = extent_um / n_grid
        n_lines = int(math.ceil(extent_um / line_period_um))
        n_halftone_layers = 2 if back_mode in ("complement", "phase_reveal") else 1

        # Budget on the halftone lattice — one duty cell per (line, column)
        # per halftoned layer. That is the order of the rectangle count the
        # run-length merge emits for detailed content (each line is several
        # cell rows, but horizontal merging collapses a band to ~one rect per
        # tonal feature), and it refuses the pathological corner BEFORE any
        # image work: extent 5000 um at period 8 um is 625 lines x 1400 cols
        # = 875k cells, well over the 400k cap.
        check_lattice_budget(
            n_lines * n_grid * n_halftone_layers,
            "Bitmap halftone",
            line_period_um=line_period_um,
            extent_um=extent_um,
        )

        darkness = _load_darkness(image, n_grid, angle_deg, invert)

        # Triangular carrier profile across each line period: tri = |2t-1| is
        # 0 at the stripe center and 1 at its edges, so `darkness > tri`
        # opens a centered gold band whose height fraction EQUALS the local
        # darkness — the amplitude halftone, fully vectorized.
        t = (((np.arange(n_grid) + 0.5) * cell_um) / line_period_um) % 1.0
        tri = np.abs(2.0 * t - 1.0).astype(np.float32)
        front_mask = darkness > tri[:, None]

        if back_mode == "none":
            back_mask = None
        elif back_mode == "carrier":
            # Uniform 50% grating, same period/phase as the halftone screen.
            back_mask = np.tile((tri < 0.5)[:, None], (1, n_grid))
        elif back_mode == "complement":
            back_mask = (1.0 - darkness) > tri[:, None]
        elif back_mode == "phase_reveal":
            tri_shift = np.abs(2.0 * ((t + 0.5) % 1.0) - 1.0).astype(np.float32)
            back_mask = darkness > tri_shift[:, None]
        else:
            raise ValueError(f"unknown back_mode {back_mode!r}")

        front = raster_to_polygons(front_mask.astype(np.uint8), cell_um, extent)
        back = (
            empty_layer()
            if back_mask is None
            else raster_to_polygons(back_mask.astype(np.uint8), cell_um, extent)
        )

        min_duty = _min_gold_band_duty(front_mask, cell_um, line_period_um)
        return GeneratedPattern(
            front=front,
            back=back,
            extent_um=extent,
            pixel_pitch_um=cell_um,
            min_feature_um=max(2.0, line_period_um * min_duty),
            extra={
                "image": image,
                "coverage_front": float(front_mask.mean()),
                "coverage_back": float(back_mask.mean()) if back_mask is not None else 0.0,
                "n_grid": int(n_grid),
                "cell_um": float(cell_um),
                "n_lines": int(n_lines),
                "min_duty_realized": float(min_duty),
            },
        )
