"""Plate composition — central optical pattern + decorative gold frame.

A ``PlateSpec`` is the user-level recipe for one glass plate:
  * a central pattern slug + params (any existing Pattern subclass), and
  * a frame spec (theme, algorithm, dials, seed) wrapping it.

The runtime path is ``materialize_plate`` -> ``_raster_compose_plate``: the
cached central-pattern PNGs and a RasterPen-rendered frame composite straight
into PIL images (no Shapely on the hot path). ``compose_plate`` is the
polygon-space equivalent, kept for tests; it CONCATENATES the frame polygons
onto the front gold layer (never ``unary_union`` — see ``_concat_polygons``).
The back layer is untouched — frames are decorative metallization on the
viewer-facing side.

SVGs are NOT built at materialize time: ``ensure_plate_svg`` builds the fab
pair lazily on first export request (the polygon/SVG path is ~40× slower
than raster and the interactive UI never needs it).

Plate layout: the ``weld_margin_um`` rim around the plate edge stays blank
glass (reserved for copper foil + solder); the frame band sits inside that
on the *active rect*; the square central aperture sits inside the band at
``min(active) - 2·band``. The central pattern always generates square so
existing patterns (all of which take a single ``extent_um`` scalar) drop in
without changes.
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw
from shapely.geometry import MultiPolygon, Polygon

from .export_svg import to_svg
from .patterns.base import GeneratedPattern, Substrate, ensure_multipolygon, registry
from .patterns.frames import (
    RectFrame,
    generate_frame,
    render_scene_to_image,
    render_scene_to_svg,
    scene_to_multipolygon,
)
from .patterns.frames.api import FrameParams
from .rasterize import make_thumbnail
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
    width_um: float = 30000.0
    height_um: float = 30000.0
    # Blank rim around the plate edges reserved for assembly welds — no
    # gold patterned anywhere inside this border. Default 1 mm matches a
    # typical UV-cure / solder joint allowance.
    weld_margin_um: float = 1000.0
    label: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "pattern_slug": self.pattern_slug,
            "pattern_params": dict(self.pattern_params),
            "frame": asdict(self.frame),
            "glass": asdict(self.glass),
            "width_um": self.width_um,
            "height_um": self.height_um,
            "weld_margin_um": self.weld_margin_um,
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
            width_um=float(data.get("width_um", 30000.0)),
            height_um=float(data.get("height_um", 30000.0)),
            weld_margin_um=float(data.get("weld_margin_um", 1000.0)),
            label=data.get("label", ""),
        )

    def active_dims(self) -> tuple[float, float]:
        """(W, H) of the patterned area inside the weld border, in μm."""
        return (
            max(0.0, self.width_um - 2.0 * self.weld_margin_um),
            max(0.0, self.height_um - 2.0 * self.weld_margin_um),
        )


def plate_hash(spec: PlateSpec) -> str:
    raw = json.dumps(spec.to_dict(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(raw.encode()).hexdigest()[:12]


def _band_um(spec: PlateSpec) -> float:
    if spec.frame.band_um is not None:
        return spec.frame.band_um
    aw, ah = spec.active_dims()
    return 0.12 * max(0.0, min(aw, ah))


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
    """Side length (μm) of the square central aperture inside the frame band.

    Sits inside *both* the weld margin and the frame band, so the central
    optical pattern has clean glass on every side.
    """
    band = _band_um(spec)
    aw, ah = spec.active_dims()
    return max(0.0, min(aw, ah) - 2.0 * band)


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

    # Frame is generated on the inset active rect so vines stop at the
    # weld boundary. The polygons come out in *active-rect coords* (origin
    # at active-rect center, which happens to be the same as plate center
    # since the inset is symmetric), so they drop into the plate frame
    # without any translation.
    active_w, active_h = spec.active_dims()
    rect = RectFrame(width_um=active_w, height_um=active_h)
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
    central_pitch = central_manifest["pixel_pitch_um"]
    central_front_path = DATA_ROOT / central_manifest["files"]["front_png"].lstrip("/").removeprefix("data/")
    central_back_path = DATA_ROOT / central_manifest["files"]["back_png"].lstrip("/").removeprefix("data/")

    # Plate canvas — pitch chosen so the longest plate side caps at ~1500 px.
    # We deliberately stay below the rasterize cap (16M px) by a wide margin
    # so the *six* face textures uploaded together to the WebGL preview don't
    # exhaust VRAM (3 mm cube was fine at 4000 px; 30 mm cube needs the cap).
    # The fab SVG export goes through the polygon path at full precision, so
    # this only bounds the live raster, not the actual mask.
    plate_pitch = max(central_pitch, max(spec.width_um, spec.height_um) / 1500.0)
    plate_w = max(1, int(round(spec.width_um / plate_pitch)))
    plate_h = max(1, int(round(spec.height_um / plate_pitch)))

    central_front = Image.open(central_front_path).convert("L")
    central_back = Image.open(central_back_path).convert("L")

    # Resize the central pattern's raster up to fill the plate's aperture —
    # the optically-active opening between the weld margin and the frame
    # band. The central pattern itself was generated at its own native μm
    # extent (small, fast, cached); we upscale here so a 3 cm plate doesn't
    # render a 2 mm dot in the middle of empty glass.
    aperture_um = _aperture(spec)
    aperture_w_px = max(1, int(round(aperture_um / plate_pitch)))
    aperture_h_px = max(1, int(round(aperture_um / plate_pitch)))
    # Preserve aspect ratio of the native central raster — if a pattern ever
    # ships non-square output, scale-to-fit and center on the long axis.
    src_aspect = central_front.size[0] / max(1, central_front.size[1])
    if src_aspect >= 1:
        aperture_h_px = max(1, int(round(aperture_w_px / src_aspect)))
    else:
        aperture_w_px = max(1, int(round(aperture_h_px * src_aspect)))
    central_front_r = central_front.resize((aperture_w_px, aperture_h_px), Image.Resampling.LANCZOS)
    central_back_r = central_back.resize((aperture_w_px, aperture_h_px), Image.Resampling.LANCZOS)
    # Reuse the same names below for the paste-offset math.
    aperture_w, aperture_h = aperture_w_px, aperture_h_px

    front = Image.new("L", (plate_w, plate_h), 0)
    back = Image.new("L", (plate_w, plate_h), 0)
    # Center the central pattern in the plate canvas.
    paste_x = (plate_w - aperture_w) // 2
    paste_y = (plate_h - aperture_h) // 2
    front.paste(central_front_r, (paste_x, paste_y))
    back.paste(central_back_r, (paste_x, paste_y))

    # Generate frame on the *active rect* — inset from the plate boundary by
    # the weld margin on every side, so vines + motifs can't bleed into the
    # weld zone. The render then composes onto a buffer the same size as the
    # active area; we paste that into the plate canvas centered.
    active_w, active_h = spec.active_dims()
    if active_w > 0 and active_h > 0:
        rect = RectFrame(width_um=active_w, height_um=active_h)
        frame_params = spec.frame.to_frame_params()
        scene = generate_frame(rect, frame_params)
        frame_png = render_scene_to_image(scene, rect, frame_params, plate_pitch)
        active_w_px = max(1, int(round(active_w / plate_pitch)))
        active_h_px = max(1, int(round(active_h / plate_pitch)))
        weld_x_px = (plate_w - active_w_px) // 2
        weld_y_px = (plate_h - active_h_px) // 2
        # `paste(255, mask=…)` only writes where the mask is non-zero, so
        # we offset the frame buffer by the weld border and the rest of the
        # plate stays untouched.
        front.paste(255, box=(weld_x_px, weld_y_px), mask=frame_png)
    else:
        scene = None  # weld swallowed the active area — degenerate but legal

    # Belt-and-suspenders: zero out the weld border on BOTH layers, so any
    # central-pattern overshoot (the raster step rounds half-pixels) can't
    # leak into the weld zone. The central pattern is paste-centered above
    # at aperture size, which is already inside the active rect, so this is
    # only defense in depth.
    if spec.weld_margin_um > 0:
        weld_px = max(1, int(round(spec.weld_margin_um / plate_pitch)))
        for img in (front, back):
            # top, bottom, left, right rectangles
            ImageDraw.Draw(img).rectangle((0, 0, plate_w, weld_px), fill=0)
            ImageDraw.Draw(img).rectangle((0, plate_h - weld_px, plate_w, plate_h), fill=0)
            ImageDraw.Draw(img).rectangle((0, 0, weld_px, plate_h), fill=0)
            ImageDraw.Draw(img).rectangle((plate_w - weld_px, 0, plate_w, plate_h), fill=0)

    front.save(out_dir / "front.png")
    back.save(out_dir / "back.png")
    thumb = make_thumbnail(front, back, size=320)
    thumb.save(out_dir / "thumbnail.png")

    # The frame scene (potentially MBs of segment dicts) goes to a sidecar,
    # NOT into manifest.json: embedding it made every plate manifest 0.4-2.3
    # MB and dominated both the warm box regen (6 × json decode) and the cold
    # compose's manifest encode. Nothing at runtime consumes it — the
    # frontend never reads it, export strips it, and ensure_plate_svg
    # regenerates the scene deterministically from the spec — so the sidecar
    # is purely for debugging/inspection.
    frame_scene = scene.to_dict() if scene is not None else {"segments": [], "flowers": [], "leaves": [], "max_t": 0.0}
    (out_dir / "scene.json").write_text(json.dumps(frame_scene))

    return {
        "central_manifest": central_manifest,
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
        # frame_scene deliberately NOT embedded — see _raster_compose_plate
        # (scene.json sidecar keeps the manifest ~20 KB instead of ~MBs).
        "recipe_data": dict(central_manifest.get("recipe_data", {})),
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

    Uses the SVG-direct pen (no Shapely) so 30 mm plates with 500+ vines
    finish in <1 s. The central pattern's polygons still go through to_svg
    (small, cached), but the frame — the expensive layer — emits SVG
    fragments straight from the motif vertex lists.

    Returns (front, back) paths or None if the plate is unknown.
    """
    plate_dir = PLATES_ROOT / plate_id
    manifest_path = plate_dir / "manifest.json"
    if not manifest_path.exists():
        return None
    front_svg = plate_dir / "front.svg"
    back_svg = plate_dir / "back.svg"
    if front_svg.exists() and back_svg.exists() and _svg_is_current(front_svg):
        return front_svg, back_svg

    manifest = json.loads(manifest_path.read_text())
    spec = PlateSpec.from_dict(manifest["spec"])

    # 1. Central pattern via existing polygon path (small, cached).
    central_cls = registry[spec.pattern_slug]
    merged_params = {**central_cls.defaults(), **spec.pattern_params}
    central = _generate_central_cached(central_cls, merged_params)

    # 2. Frame scene via the fast SVG pen.
    active_w, active_h = spec.active_dims()
    rect = RectFrame(width_um=active_w, height_um=active_h)
    frame_params = spec.frame.to_frame_params()
    scene = generate_frame(rect, frame_params)
    frame_body = render_scene_to_svg(scene, rect, frame_params)

    # 3. Central pattern as SVG paths, scaled up to fill the plate aperture —
    # the SAME target the raster compositor hits (_raster_compose_plate), so
    # the fab SVG geometry matches the preview PNG. Both spaces share a
    # centered origin, so a uniform scale about the origin is exact; patterns
    # generate square, and a non-square one scales-to-fit on its long axis
    # just like the raster path.
    aperture_um = _aperture(spec)
    central_long_um = max(central.extent_um[0], central.extent_um[1])
    central_scale = aperture_um / central_long_um if central_long_um > 0 else 1.0
    central_front_body = to_svg(central.front, central.extent_um, background=None)
    central_back_body = to_svg(central.back, central.extent_um, background=None)

    # 4. Stamp the surrounding <svg> wrapper at plate extent (μm units).
    W, H = spec.width_um, spec.height_um
    front_svg.write_text(
        _wrap_svg(
            W,
            H,
            [
                _inner_svg_paths(central_front_body, scale=central_scale),
                _group("frame", frame_body),
            ],
        ),
        encoding="utf-8",
    )
    back_svg.write_text(
        _wrap_svg(W, H, [_inner_svg_paths(central_back_body, scale=central_scale)]),
        encoding="utf-8",
    )

    manifest["files"]["front_svg"] = f"/data/plates/{plate_id}/front.svg"
    manifest["files"]["back_svg"] = f"/data/plates/{plate_id}/back.svg"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return front_svg, back_svg


# Bump when the SVG compose geometry changes: cached plate SVGs are only
# reused if they carry the current marker, so a formula fix (e.g. the
# aperture-scaling fix) invalidates stale files under unchanged spec hashes.
PLATE_SVG_VERSION = "plate-svg-v2"


def _svg_is_current(svg_path: Path) -> bool:
    try:
        head = svg_path.read_text(encoding="utf-8", errors="ignore")[:256]
    except OSError:
        return False
    return PLATE_SVG_VERSION in head


def _wrap_svg(width_um: float, height_um: float, groups: list[str]) -> str:
    # width/height carry an explicit physical unit (mm) so importers that
    # honor them place the plate at true scale; the viewBox keeps user units
    # = μm, matching every nested path coordinate.
    body = "".join(groups)
    return (
        f'<?xml version="1.0" encoding="UTF-8"?>'
        f'<!--{PLATE_SVG_VERSION}-->'
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{width_um / 1000.0:.4f}mm" height="{height_um / 1000.0:.4f}mm" '
        f'viewBox="{-width_um/2:.2f} {-height_um/2:.2f} {width_um:.2f} {height_um:.2f}">'
        f'{body}</svg>'
    )


def _group(layer_id: str, inner: str) -> str:
    return f'<g id="{layer_id}">{inner}</g>'


def _inner_svg_paths(full_svg: str, scale: float = 1.0) -> str:
    """Strip the outer <svg>...</svg> wrapper off a drawsvg output so we can
    nest it inside our wrapper, optionally scaling the group about the shared
    centered origin (used to blow the central pattern up to the aperture).
    Drawsvg emits a fixed prelude that we don't want twice in one document.
    """
    # Drawsvg emits something like:
    #   <?xml ...?><svg ...><defs>...</defs><rect ...>...<path .../></svg>
    # Find the first '>' after '<svg' and the closing '</svg>'.
    open_idx = full_svg.find("<svg")
    if open_idx == -1:
        return full_svg
    open_end = full_svg.find(">", open_idx)
    close_idx = full_svg.rfind("</svg>")
    if open_end == -1 or close_idx == -1:
        return full_svg
    inner = full_svg[open_end + 1 : close_idx]
    if abs(scale - 1.0) > 1e-9:
        return f'<g id="central" transform="scale({scale:.8g})">{inner}</g>'
    return _group("central", inner)


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
