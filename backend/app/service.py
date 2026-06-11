from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any

from .export_svg import to_svg
from .patterns.base import registry
from .rasterize import make_thumbnail, rasterize


DATA_ROOT = Path(__file__).resolve().parent.parent / "data"

_log = logging.getLogger("optics.service")


def _params_hash(params: dict[str, Any]) -> str:
    raw = json.dumps(params, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(raw.encode()).hexdigest()[:10]


def pattern_dir(slug: str, variant: str) -> Path:
    return DATA_ROOT / slug / variant


def catalog() -> list[dict]:
    return [cls.descriptor() for cls in registry.values()]


def list_variants(slug: str) -> list[str]:
    root = DATA_ROOT / slug
    if not root.exists():
        return []
    return sorted(p.name for p in root.iterdir() if p.is_dir())


def materialize(
    slug: str,
    params: dict[str, Any] | None = None,
    force: bool = False,
) -> dict:
    """Generate (or load from cache) a variant of a pattern and return its manifest."""
    if slug not in registry:
        raise KeyError(f"Unknown pattern: {slug}")

    cls = registry[slug]
    merged = {**cls.defaults(), **(params or {})}
    variant = _params_hash(merged)
    out = pattern_dir(slug, variant)

    manifest_path = out / "manifest.json"
    if manifest_path.exists() and not force:
        cached = json.loads(manifest_path.read_text())
        if cached.get("render_recipe") == cls.render_recipe:
            _log.info("materialize cache_hit slug=%s variant=%s", slug, variant)
            return cached
        _log.info(
            "materialize regenerate_for_recipe slug=%s variant=%s recipe=%s",
            slug,
            variant,
            cls.render_recipe,
        )

    t0 = time.perf_counter()
    gp = cls.generate(**merged)
    out.mkdir(parents=True, exist_ok=True)

    # Rasterize both layers
    front_png = rasterize(gp.front, gp.extent_um, gp.pixel_pitch_um)
    back_png = rasterize(gp.back, gp.extent_um, gp.pixel_pitch_um)
    front_png.save(out / "front.png")
    back_png.save(out / "back.png")

    # SVG is built on demand (see ensure_pattern_svg) — eager to_svg cost
    # ~4-8 s per cold variant and nothing at runtime ever fetched the files.
    svg_ok = (out / "front.svg").exists() and (out / "back.svg").exists()

    # Recipe-specific extra layers (e.g. view_a / view_b for stereo_lenticular).
    # Each gets rasterized and its URL stamped into recipe_data[<name>_png] so
    # the frontend can load them without the generator doing path bookkeeping.
    extra_layer_urls: dict[str, str] = {}
    for layer_name, layer_poly in gp.extra_layers.items():
        layer_png = rasterize(layer_poly, gp.extent_um, gp.pixel_pitch_um)
        layer_png.save(out / f"{layer_name}.png")
        extra_layer_urls[f"{layer_name}_png"] = f"/data/{slug}/{variant}/{layer_name}.png"

    # Thumbnail
    thumb = make_thumbnail(front_png, back_png, size=256)
    thumb.save(out / "thumbnail.png")

    # Manifest
    manifest = {
        "slug": slug,
        "variant": variant,
        "name": cls.name,
        "description": cls.description,
        "tags": cls.tags,
        "params": merged,
        "substrate": {
            "thickness_um": gp.substrate.thickness_um,
            "material": gp.substrate.material,
            "n": gp.substrate.n,
        },
        "extent_um": list(gp.extent_um),
        "pixel_pitch_um": gp.pixel_pitch_um,
        "min_feature_um": gp.min_feature_um,
        "extra": gp.extra,
        # Render-recipe hints for the frontend shader. The class-level default
        # is "stylized_amplitude" (flat-mask composite); patterns that want a
        # physics-correct recipe override Pattern.render_recipe on the class
        # and can stash per-variant file refs in GeneratedPattern.recipe_data.
        "render_recipe": cls.render_recipe,
        "recipe_data": {**gp.recipe_data, **extra_layer_urls},
        "files": {
            "front_png": f"/data/{slug}/{variant}/front.png",
            "back_png": f"/data/{slug}/{variant}/back.png",
            # Lazy (mirrors the plate manifest contract): empty until
            # ensure_pattern_svg builds the files on demand.
            "front_svg": f"/data/{slug}/{variant}/front.svg" if svg_ok else "",
            "back_svg": f"/data/{slug}/{variant}/back.svg" if svg_ok else "",
            "thumbnail": f"/data/{slug}/{variant}/thumbnail.png",
        },
    }
    (manifest_path).write_text(json.dumps(manifest, indent=2))
    dt_ms = int((time.perf_counter() - t0) * 1000)
    pixels = front_png.size[0] * front_png.size[1]
    _log.info(
        "materialize done slug=%s variant=%s pixels=%d pitch_um=%.3f %dms",
        slug,
        variant,
        pixels,
        gp.pixel_pitch_um,
        dt_ms,
    )
    return manifest


def ensure_pattern_svg(slug: str, variant: str) -> tuple[Path, Path] | None:
    """Lazily build the SVG pair for an existing pattern variant.

    Mirrors ``plates.ensure_plate_svg``: the materialize path skips the
    eager ``to_svg`` (it cost seconds per cold variant with zero runtime
    consumers); callers that genuinely want the authoritative vector form
    (fab/export tooling) hit this instead. Regenerates the polygons from the
    manifest's params (deterministic), writes front.svg/back.svg, stamps the
    URLs back into the manifest.

    Returns (front, back) paths or None if the variant is unknown.
    """
    out = pattern_dir(slug, variant)
    manifest_path = out / "manifest.json"
    if slug not in registry or not manifest_path.exists():
        return None
    front_svg = out / "front.svg"
    back_svg = out / "back.svg"
    if front_svg.exists() and back_svg.exists():
        return front_svg, back_svg

    manifest = json.loads(manifest_path.read_text())
    cls = registry[slug]
    gp = cls.generate(**{**cls.defaults(), **manifest.get("params", {})})
    front_svg.write_text(to_svg(gp.front, gp.extent_um), encoding="utf-8")
    back_svg.write_text(to_svg(gp.back, gp.extent_um), encoding="utf-8")

    manifest["files"]["front_svg"] = f"/data/{slug}/{variant}/front.svg"
    manifest["files"]["back_svg"] = f"/data/{slug}/{variant}/back.svg"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return front_svg, back_svg


def seed_defaults() -> list[dict]:
    """Materialize every registered pattern's default variant.

    NOT called at startup anymore (it cost minutes of cold wall clock before
    the first request could be answered) — the box/plate path materializes
    central patterns lazily with a disk cache. Kept as the explicit warm
    command (``just seed``).
    """
    return [materialize(slug) for slug in registry]
