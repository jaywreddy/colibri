"""Plate composition — central optical pattern + decorative gold frame.

A ``PlateSpec`` is the user-level recipe for one glass plate:
  * a central pattern slug + params (any existing Pattern subclass), and
  * a frame spec (theme, algorithm, dials, seed) wrapping it.

``compose_plate`` runs the central pattern's generator at the aperture extent,
then unions the rendered frame onto the front gold layer. The back layer is
untouched — frames are decorative metallization on the viewer-facing side.

The plate's total extent is ``(width_um, height_um)``; the inner aperture is
``(width - 2·band, height - 2·band)``. The central pattern always generates
square at ``min(aperture)`` so existing patterns (all of which take a single
``extent_um`` scalar) drop in without changes.
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image
from shapely.geometry import MultiPolygon, Polygon

from .export_svg import to_svg
from .patterns.base import GeneratedPattern, Substrate, ensure_multipolygon, registry
from .patterns.frames import RectFrame, generate_frame, render_scene_to_image, scene_to_multipolygon
from .patterns.frames.api import FrameParams
from .rasterize import make_thumbnail, rasterize
from .service import DATA_ROOT, materialize as materialize_pattern


def _concat_polygons(*sources: MultiPolygon) -> MultiPolygon:
    """Combine MultiPolygons by concatenation, not union.

    For the lithography raster we don't care that overlapping front-layer
    polygons appear twice — ``ImageDraw.polygon`` paints them with the same
    gold color and the result is visually identical. Skipping ``unary_union``
    on tens of thousands of small polygons (a dense frame's stroke buffer
    output) is what keeps compose_plate inside GEOS memory budgets.
    """
    geoms: list[Polygon] = []
    for src in sources:
        if src is None or src.is_empty:
            continue
        if isinstance(src, MultiPolygon):
            geoms.extend(g for g in src.geoms if not g.is_empty)
        elif isinstance(src, Polygon):
            geoms.append(src)
    return MultiPolygon(geoms)


_log = logging.getLogger("optics.plates")

PLATES_ROOT = DATA_ROOT / "plates"


@dataclass
class FrameSpec:
    """Frame generation knobs, serializable across the API."""

    algorithm: str = "colonize"
    theme: str = "esmeralda"
    density: float = 1.0
    bloom: float = 0.6
    foliage: float = 0.6
    seed: int = 1
    band_um: float | None = None  # default: 12% of shorter plate side

    def to_frame_params(self) -> FrameParams:
        return FrameParams(
            algorithm=self.algorithm,
            theme_slug=self.theme,
            density=self.density,
            bloom=self.bloom,
            foliage=self.foliage,
            seed=self.seed,
            frame_width_um=self.band_um,
        )


@dataclass
class GlassSpec:
    thickness_um: float = 500.0
    material: str = "fused silica"
    n: float = 1.46


@dataclass
class PlateSpec:
    """User-level recipe for one glass plate."""

    pattern_slug: str
    pattern_params: dict[str, Any] = field(default_factory=dict)
    frame: FrameSpec = field(default_factory=FrameSpec)
    glass: GlassSpec = field(default_factory=GlassSpec)
    width_um: float = 3000.0
    height_um: float = 3000.0
    label: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "pattern_slug": self.pattern_slug,
            "pattern_params": dict(self.pattern_params),
            "frame": asdict(self.frame),
            "glass": asdict(self.glass),
            "width_um": self.width_um,
            "height_um": self.height_um,
            "label": self.label,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PlateSpec":
        frame_data = data.get("frame", {}) or {}
        glass_data = data.get("glass", {}) or {}
        return cls(
            pattern_slug=data["pattern_slug"],
            pattern_params=dict(data.get("pattern_params", {})),
            frame=FrameSpec(**frame_data),
            glass=GlassSpec(**glass_data),
            width_um=float(data.get("width_um", 2400.0)),
            height_um=float(data.get("height_um", 1600.0)),
            label=data.get("label", ""),
        )


def plate_hash(spec: PlateSpec) -> str:
    raw = json.dumps(spec.to_dict(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(raw.encode()).hexdigest()[:12]


def _band_um(spec: PlateSpec) -> float:
    if spec.frame.band_um is not None:
        return spec.frame.band_um
    return 0.12 * min(spec.width_um, spec.height_um)


# In-process cache of (slug, params-hash) -> GeneratedPattern. The full
# materialize() path persists PNG/SVG/JSON to disk for the standalone
# /patterns endpoints; here we just need the shapely polygons quickly. A
# composed plate that re-uses the same central pattern across frame edits
# is the common case, so even a tiny LRU keeps the dev loop responsive.
_central_cache: dict[str, GeneratedPattern] = {}
_CENTRAL_CACHE_MAX = 24


def _generate_central_cached(cls: type, params: dict[str, Any]) -> GeneratedPattern:
    key = cls.slug + ":" + hashlib.sha1(
        json.dumps(params, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:10]
    hit = _central_cache.get(key)
    if hit is not None:
        return hit
    if len(_central_cache) >= _CENTRAL_CACHE_MAX:
        # Drop one arbitrary entry — exact LRU isn't worth the bookkeeping
        # at this cache size.
        _central_cache.pop(next(iter(_central_cache)))
    result = cls.generate(**params)
    _central_cache[key] = result
    return result


def _aperture(spec: PlateSpec) -> float:
    """Side length (μm) of the square central aperture inside the frame band."""
    band = _band_um(spec)
    return max(0.0, min(spec.width_um, spec.height_um) - 2.0 * band)


def compose_plate(spec: PlateSpec) -> GeneratedPattern:
    """Run the central pattern + frame and return the union as polygons.

    NOTE: kept around for tests and the SVG export path. The runtime
    materialize_plate uses the raster compositor (`_compose_plate_raster`)
    which is far faster for the dense-frame common case.
    """
    if spec.pattern_slug not in registry:
        raise KeyError(f"Unknown pattern: {spec.pattern_slug}")

    central_cls = registry[spec.pattern_slug]
    merged_params = {**central_cls.defaults(), **spec.pattern_params}

    aperture = _aperture(spec)
    if aperture <= 0:
        raise ValueError(
            f"Aperture is non-positive ({aperture}μm) for plate "
            f"{spec.width_um}×{spec.height_um}μm with band {_band_um(spec)}μm — "
            "shrink the frame band or grow the plate."
        )

    central = _generate_central_cached(central_cls, merged_params)

    rect = RectFrame(width_um=spec.width_um, height_um=spec.height_um)
    frame_params = spec.frame.to_frame_params()
    scene = generate_frame(rect, frame_params)
    frame_mask = scene_to_multipolygon(scene, rect, frame_params)

    central_front = ensure_multipolygon(central.front)
    central_back = ensure_multipolygon(central.back)
    combined_front = _concat_polygons(central_front, frame_mask)

    return GeneratedPattern(
        front=combined_front,
        back=central_back,
        extent_um=(spec.width_um, spec.height_um),
        pixel_pitch_um=central.pixel_pitch_um,
        min_feature_um=central.min_feature_um,
        substrate=Substrate(
            thickness_um=spec.glass.thickness_um,
            material=spec.glass.material,
            n=spec.glass.n,
        ),
        extra={
            **central.extra,
            "central_pattern": spec.pattern_slug,
            "aperture_um": aperture,
        },
        recipe_data={**central.recipe_data, "frame_scene": scene.to_dict()},
        extra_layers=central.extra_layers,
    )


def _raster_compose_plate(spec: PlateSpec, out_dir: Path) -> dict[str, Any]:
    """Raster-space compose — much faster than the polygon path.

    Pipeline:
      1. materialize_pattern → cached central PNGs on disk.
      2. generate_frame → Scene of vines + motif sprites.
      3. scene_to_multipolygon → frame-only MultiPolygon at plate scale.
      4. Composite all three into front/back PIL images at plate extent.

    Returns the *partial* manifest dict (files + extras + recipe data); the
    caller stamps the spec/labels and writes manifest.json.
    """
    central_cls = registry[spec.pattern_slug]
    merged_params = {**central_cls.defaults(), **spec.pattern_params}

    central_manifest = materialize_pattern(spec.pattern_slug, merged_params)
    central_extent = central_manifest["extent_um"]
    central_pitch = central_manifest["pixel_pitch_um"]
    central_front_path = DATA_ROOT / central_manifest["files"]["front_png"].lstrip("/").removeprefix("data/")
    central_back_path = DATA_ROOT / central_manifest["files"]["back_png"].lstrip("/").removeprefix("data/")

    # Plate canvas — pitch chosen so the longest plate side is ≤ 4000 px,
    # which keeps the raster under the rasterize cap (16M px) at any aspect.
    plate_pitch = max(central_pitch, max(spec.width_um, spec.height_um) / 4000.0)
    plate_w = max(1, int(round(spec.width_um / plate_pitch)))
    plate_h = max(1, int(round(spec.height_um / plate_pitch)))

    central_front = Image.open(central_front_path).convert("L")
    central_back = Image.open(central_back_path).convert("L")

    # Resize central from its native pitch into plate-canvas pixels — at the
    # aperture size (smaller than the full plate by the frame band).
    aperture_w = max(1, int(round(central_extent[0] / plate_pitch)))
    aperture_h = max(1, int(round(central_extent[1] / plate_pitch)))
    central_front_r = central_front.resize((aperture_w, aperture_h), Image.Resampling.LANCZOS)
    central_back_r = central_back.resize((aperture_w, aperture_h), Image.Resampling.LANCZOS)

    front = Image.new("L", (plate_w, plate_h), 0)
    back = Image.new("L", (plate_w, plate_h), 0)
    # Center the central pattern in the plate canvas.
    paste_x = (plate_w - aperture_w) // 2
    paste_y = (plate_h - aperture_h) // 2
    front.paste(central_front_r, (paste_x, paste_y))
    back.paste(central_back_r, (paste_x, paste_y))

    # Generate frame scene + paint directly into a PIL image (skips Shapely).
    rect = RectFrame(width_um=spec.width_um, height_um=spec.height_um)
    frame_params = spec.frame.to_frame_params()
    scene = generate_frame(rect, frame_params)
    frame_png = render_scene_to_image(scene, rect, frame_params, plate_pitch)
    # Composite frame ON TOP of the central pattern on the front layer.
    front.paste(255, mask=frame_png)

    front.save(out_dir / "front.png")
    back.save(out_dir / "back.png")
    thumb = make_thumbnail(front, back, size=320)
    thumb.save(out_dir / "thumbnail.png")

    return {
        "central_manifest": central_manifest,
        "frame_scene": scene.to_dict(),
        "pixel_pitch_um": plate_pitch,
        "min_feature_um": central_manifest["min_feature_um"],
    }


def materialize_plate(spec: PlateSpec, force: bool = False) -> dict[str, Any]:
    """Compose the plate, cache PNG/SVG/manifest under ``data/plates/<hash>/``."""
    PLATES_ROOT.mkdir(parents=True, exist_ok=True)
    pid = plate_hash(spec)
    out = PLATES_ROOT / pid
    manifest_path = out / "manifest.json"
    if manifest_path.exists() and not force:
        cached = json.loads(manifest_path.read_text())
        _log.info("materialize_plate cache_hit id=%s slug=%s", pid, spec.pattern_slug)
        return cached

    t0 = time.perf_counter()
    out.mkdir(parents=True, exist_ok=True)
    raster_result = _raster_compose_plate(spec, out)
    central_manifest = raster_result["central_manifest"]
    pitch = raster_result["pixel_pitch_um"]

    # SVG is built on demand by the /export endpoint (see ensure_plate_svg)
    # — the polygon path is ~40× slower than raster and the interactive UI
    # never needs it. Manifest carries empty svg paths until requested.
    svg_ok = (out / "front.svg").exists() and (out / "back.svg").exists()

    central_cls = registry[spec.pattern_slug]
    manifest = {
        "kind": "plate",
        "id": pid,
        "spec": spec.to_dict(),
        "name": spec.label or central_cls.name,
        "description": central_cls.description,
        "tags": central_cls.tags,
        "substrate": {
            "thickness_um": spec.glass.thickness_um,
            "material": spec.glass.material,
            "n": spec.glass.n,
        },
        "extent_um": [spec.width_um, spec.height_um],
        "pixel_pitch_um": pitch,
        "min_feature_um": raster_result["min_feature_um"],
        "extra": {
            **central_manifest.get("extra", {}),
            "central_pattern": spec.pattern_slug,
            "aperture_um": _aperture(spec),
        },
        "render_recipe": central_cls.render_recipe,
        "recipe_data": {
            **central_manifest.get("recipe_data", {}),
            "frame_scene": raster_result["frame_scene"],
        },
        "files": {
            "front_png": f"/data/plates/{pid}/front.png",
            "back_png": f"/data/plates/{pid}/back.png",
            "front_svg": f"/data/plates/{pid}/front.svg" if svg_ok else "",
            "back_svg": f"/data/plates/{pid}/back.svg" if svg_ok else "",
            "thumbnail": f"/data/plates/{pid}/thumbnail.png",
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))
    dt_ms = int((time.perf_counter() - t0) * 1000)
    _log.info(
        "materialize_plate done id=%s slug=%s %dms pitch=%.3f svg=%s",
        pid,
        spec.pattern_slug,
        dt_ms,
        pitch,
        svg_ok,
    )
    return manifest


def ensure_plate_svg(plate_id: str) -> tuple[Path, Path] | None:
    """Lazily build the SVG fab pair for an existing plate.

    Called by the export endpoint when the user hits "download bundle". Skips
    if SVGs already exist (subsequent exports are instant). Returns (front,
    back) paths or None if the plate is unknown.
    """
    plate_dir = PLATES_ROOT / plate_id
    manifest_path = plate_dir / "manifest.json"
    if not manifest_path.exists():
        return None
    front_svg = plate_dir / "front.svg"
    back_svg = plate_dir / "back.svg"
    if front_svg.exists() and back_svg.exists():
        return front_svg, back_svg

    manifest = json.loads(manifest_path.read_text())
    spec = PlateSpec.from_dict(manifest["spec"])
    composed = compose_plate(spec)  # polygon path is the SVG source
    front_svg.write_text(to_svg(composed.front, composed.extent_um), encoding="utf-8")
    back_svg.write_text(to_svg(composed.back, composed.extent_um), encoding="utf-8")
    # Update manifest with the now-real SVG paths.
    manifest["files"]["front_svg"] = f"/data/plates/{plate_id}/front.svg"
    manifest["files"]["back_svg"] = f"/data/plates/{plate_id}/back.svg"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return front_svg, back_svg


def list_plates() -> list[dict[str, Any]]:
    if not PLATES_ROOT.exists():
        return []
    out: list[dict[str, Any]] = []
    for d in sorted(PLATES_ROOT.iterdir()):
        m = d / "manifest.json"
        if m.exists():
            try:
                out.append(json.loads(m.read_text()))
            except Exception:  # noqa: BLE001 — skip corrupt manifests
                continue
    return out


def get_plate(plate_id: str) -> dict[str, Any] | None:
    m = PLATES_ROOT / plate_id / "manifest.json"
    if not m.exists():
        return None
    return json.loads(m.read_text())
