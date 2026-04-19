from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ..patterns.base import registry
from ..service import catalog, materialize

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
