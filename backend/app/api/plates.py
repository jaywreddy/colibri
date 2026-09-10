from __future__ import annotations

from typing import Any, Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from ..patterns.base import registry
from ..plates import PlateSpec, get_plate, list_plates, materialize_plate

router = APIRouter(prefix="/plates", tags=["plates"])


class FrameSpecBody(BaseModel):
    algorithm: str = "wreath"
    theme: str = "esmeralda"
    density: float = 1.0
    bloom: float = 0.6
    foliage: float = 0.6
    seed: int = 1
    band_um: float | None = None
    # Band-composition dials — passed through to the frame grower so each
    # face can carry its own engraved-border recipe (see plates.FrameSpec).
    edge_gradient: float = 0.8
    understory: float = 0.85
    border_vine: float = 1.15
    corner_fans: float = 1.0
    # Wreath composition preset (wreath algorithm only): "laurel" | "garland"
    # | "clusters". Ignored by colonize. See frames/algorithms/wreath.py.
    wreath_style: str = "laurel"
    # Motif-only size dial (wreath only): scales leaf/bloom/understory sizes and
    # their spacing along the vine; band width and vine gauge stay put. Declared
    # here so the knob survives ``model_dump`` into FrameSpec — an undeclared
    # field would be silently dropped from the request.
    motif_scale: float = 1.0


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
    # Fabricated grating pitch (μm) of the back carrier + leaf louvre family.
    carrier_pitch_um: float = 22.0
    # Per-face carrier scaling vs the real glass: "gap" tracks t/n (controlled
    # reveal on any stock, no-op at the 500 µm baseline); "fixed" keeps the
    # literal fine pitch (refraction shimmer on thick stock).
    carrier_scale_mode: Literal["gap", "fixed"] = "gap"
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
