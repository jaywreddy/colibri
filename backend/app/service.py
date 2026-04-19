from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .export_svg import to_svg
from .patterns.base import GeneratedPattern, Pattern, registry
from .rasterize import make_thumbnail, rasterize


DATA_ROOT = Path(__file__).resolve().parent.parent / "data"


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
        return json.loads(manifest_path.read_text())

    gp = cls.generate(**merged)
    out.mkdir(parents=True, exist_ok=True)

    # Rasterize both layers
    front_png = rasterize(gp.front, gp.extent_um, gp.pixel_pitch_um)
    back_png = rasterize(gp.back, gp.extent_um, gp.pixel_pitch_um)
    front_png.save(out / "front.png")
    back_png.save(out / "back.png")

    # SVG exports
    (out / "front.svg").write_text(to_svg(gp.front, gp.extent_um), encoding="utf-8")
    (out / "back.svg").write_text(to_svg(gp.back, gp.extent_um), encoding="utf-8")

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
        "files": {
            "front_png": f"/data/{slug}/{variant}/front.png",
            "back_png": f"/data/{slug}/{variant}/back.png",
            "front_svg": f"/data/{slug}/{variant}/front.svg",
            "back_svg": f"/data/{slug}/{variant}/back.svg",
            "thumbnail": f"/data/{slug}/{variant}/thumbnail.png",
        },
    }
    (manifest_path).write_text(json.dumps(manifest, indent=2))
    return manifest


def seed_defaults() -> list[dict]:
    return [materialize(slug) for slug in registry]
