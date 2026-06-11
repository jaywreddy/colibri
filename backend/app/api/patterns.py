from __future__ import annotations

import json
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ..patterns.base import registry
from ..service import catalog, ensure_pattern_svg, materialize, pattern_dir

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
    except Exception as e:  # noqa: BLE001 — surface generation errors to UI
        raise HTTPException(400, f"Generation failed: {e!r}") from e


@router.get("/{slug}/default")
def default_variant(slug: str) -> dict:
    if slug not in registry:
        raise HTTPException(404, f"Unknown pattern: {slug}")
    return materialize(slug)


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
