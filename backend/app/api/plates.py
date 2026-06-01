from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from ..patterns.base import registry
from ..plates import PlateSpec, get_plate, list_plates, materialize_plate

router = APIRouter(prefix="/plates", tags=["plates"])


class FrameSpecBody(BaseModel):
    algorithm: str = "colonize"
    theme: str = "esmeralda"
    density: float = 1.0
    bloom: float = 0.6
    foliage: float = 0.6
    seed: int = 1
    band_um: float | None = None


class GlassSpecBody(BaseModel):
    thickness_um: float = 500.0
    material: str = "fused silica"
    n: float = 1.46


class PlateSpecBody(BaseModel):
    pattern_slug: str
    pattern_params: dict[str, Any] = Field(default_factory=dict)
    frame: FrameSpecBody = Field(default_factory=FrameSpecBody)
    glass: GlassSpecBody = Field(default_factory=GlassSpecBody)
    width_um: float = 30000.0
    height_um: float = 30000.0
    weld_margin_um: float = 1000.0
    label: str = ""

    def to_spec(self) -> PlateSpec:
        return PlateSpec.from_dict(self.model_dump())


class GeneratePlateRequest(BaseModel):
    spec: PlateSpecBody
    force: bool = False


@router.get("")
def list_all() -> list[dict[str, Any]]:
    return list_plates()


@router.get("/{plate_id}")
def get_one(plate_id: str) -> dict[str, Any]:
    m = get_plate(plate_id)
    if m is None:
        raise HTTPException(404, f"Unknown plate: {plate_id}")
    return m


@router.post("/generate")
def generate(req: GeneratePlateRequest) -> dict[str, Any]:
    spec = req.spec.to_spec()
    if spec.pattern_slug not in registry:
        raise HTTPException(404, f"Unknown pattern: {spec.pattern_slug}")
    try:
        return materialize_plate(spec, force=req.force)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(400, f"Plate generation failed: {e!r}") from e
