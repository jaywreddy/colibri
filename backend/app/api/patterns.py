from __future__ import annotations

import json
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from ..patterns.base import registry
from ..rasterize import (
    DEFAULT_THUMBNAIL_METAL,
    THUMBNAIL_PALETTES,
    make_thumbnail,
)
from ..service import (
    catalog,
    ensure_pattern_svg,
    materialize,
    pattern_dir,
    save_png_atomic,
    variant_key,
)

router = APIRouter(prefix="/patterns", tags=["patterns"])


class GenerateRequest(BaseModel):
    slug: str
    params: dict[str, Any] = {}
    force: bool = False


@router.get("")
def list_patterns() -> list[dict]:
    return catalog()


@router.get("/{slug}")
def describe_pattern(slug: str) -> dict:
    if slug not in registry:
        raise HTTPException(404, f"Unknown pattern: {slug}")
    return registry[slug].descriptor()


@router.post("/generate")
def generate(req: GenerateRequest) -> dict:
    if req.slug not in registry:
        raise HTTPException(404, f"Unknown pattern: {req.slug}")
    try:
        return materialize(req.slug, req.params, force=req.force)
    except ValueError as e:
        # ParamSpec bounds (service.validate_params), the litho floor and the
        # lattice budget all raise ValueError with text written FOR the user
        # ("period_um must be >= 4 um ..."). Pass it through verbatim — a repr
        # would reach the UI as ValueError('...') with escaped quotes.
        raise HTTPException(422, str(e)) from e
    except Exception as e:  # noqa: BLE001 — surface generation errors to UI
        raise HTTPException(400, f"Generation failed: {e!r}") from e


@router.get("/{slug}/default")
def default_variant(slug: str) -> dict:
    if slug not in registry:
        raise HTTPException(404, f"Unknown pattern: {slug}")
    return materialize(slug)


@router.get("/{slug}/thumbnail")
def default_thumbnail(slug: str, metal: str = DEFAULT_THUMBNAIL_METAL) -> FileResponse:
    """Serve the DEFAULT variant's CACHED thumbnail, or 404. Never generates.

    The pattern picker fires one request per catalog slug on mount. Pointed at
    ``/{slug}/default`` that sweep was sixteen full cold generates holding the
    very per-slot locks a box regen wants, which is the "two long computes at
    once" CLAUDE.md forbids. This route is read-only: a cold slug 404s, the
    picker shows a placeholder, and the thumbnail appears once the variant is
    materialized for a real reason.
    """
    if slug not in registry:
        raise HTTPException(404, f"Unknown pattern: {slug}")
    if metal not in THUMBNAIL_PALETTES:
        raise HTTPException(400, f"Unknown metal: {metal}")
    # Same merge + hash materialize(slug) would use for the default variant.
    variant = variant_key(registry[slug].defaults())
    vdir = pattern_dir(slug, variant)
    name = (
        "thumbnail.png"
        if metal == DEFAULT_THUMBNAIL_METAL
        else f"thumbnail_{metal}.png"
    )
    thumb = vdir / name
    if not thumb.is_file():
        # Lazily TINT a variant cached before per-metal chips existed. This
        # stays inside the route's cache-only contract: it never runs a
        # generator (a cold slug still 404s below) — it only re-colours the
        # front/back masks this variant already has, which is two pastes on a
        # 256 px image, not a compute that could collide with a box regen.
        front_p, back_p = vdir / "front.png", vdir / "back.png"
        if not (front_p.is_file() and back_p.is_file()):
            raise HTTPException(404, f"No cached thumbnail for {slug} yet")
        from PIL import Image

        with Image.open(front_p) as fim, Image.open(back_p) as bim:
            tinted = make_thumbnail(
                fim.convert("L"), bim.convert("L"), size=256, metal=metal
            )
        save_png_atomic(tinted, thumb)
    return FileResponse(thumb, media_type="image/png")


@router.get("/{slug}/{variant}/svg")
def build_svg(slug: str, variant: str) -> dict:
    """Build (or reuse) the lazy SVG pair for a materialized variant.

    materialize() no longer writes pattern-level SVGs eagerly; this is the
    on-demand hook (mirrors the plate export path's ensure_plate_svg). The
    returned manifest carries the now-populated files.front_svg/back_svg
    URLs, servable from the /data static mount.
    """
    pair = ensure_pattern_svg(slug, variant)
    if pair is None:
        raise HTTPException(404, f"Unknown variant: {slug}/{variant}")
    return json.loads((pattern_dir(slug, variant) / "manifest.json").read_text())
