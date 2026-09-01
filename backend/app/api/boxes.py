from __future__ import annotations

from typing import Any, Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from ..boxes import BoxSpec, delete_box, get_box, list_boxes, materialize_box
from .plates import GlassSpecBody, PlateSpecBody

router = APIRouter(prefix="/boxes", tags=["boxes"])


class FoilSpecBody(BaseModel):
    tape_width_um: float = 6350.0
    safety_um: float = 500.0
    bead_um: float = 2000.0
    finish: str = "bright"


class HingeSpecBody(BaseModel):
    style: str = "tube"
    tube_od_um: float = 2400.0
    rod_od_um: float = 1600.0
    segments: int = 5
    coverage: float = 0.8


class BoxSpecBody(BaseModel):
    """BoxSpec v2 — defaults MUST mirror the backend dataclasses and the
    frontend ``defaultBoxSpec()`` exactly."""

    width_um: float = 50000.0
    depth_um: float = 50000.0
    height_um: float = 40000.0
    glass: GlassSpecBody = Field(default_factory=GlassSpecBody)
    foil: FoilSpecBody = Field(default_factory=FoilSpecBody)
    hinge: HingeSpecBody = Field(default_factory=HingeSpecBody)
    faces: dict[str, PlateSpecBody] = Field(default_factory=dict)
    # Box-level grating pitch (μm) — stamped onto every face by
    # normalize_face_dims. Default 22 µm (litho floor 4 µm, enforced UI-side).
    carrier_pitch_um: float = 22.0
    # Bonded (two-ply) construction: glass.thickness_um is then the PLY, the
    # wall is 2x, and cut dims / foil margins follow the nested-shell math.
    bonded: bool = False
    # Litho metal (preview material only — masks are identical):
    # gold | chrome | chrome-ar.
    metal: Literal["gold", "chrome", "chrome-ar"] = "gold"
    label: str = ""

    def to_spec(self) -> BoxSpec:
        return BoxSpec.from_dict(self.model_dump())


class GenerateBoxRequest(BoxSpecBody):
    box_id: str | None = None
    force: bool = False


@router.get("")
def list_all() -> list[dict[str, Any]]:
    return list_boxes()


@router.get("/{box_id}")
def get_one(box_id: str) -> dict[str, Any]:
    m = get_box(box_id)
    if m is None:
        raise HTTPException(404, f"Unknown box: {box_id}")
    return m


@router.post("/generate")
def generate(req: GenerateBoxRequest) -> dict[str, Any]:
    spec = req.to_spec()
    try:
        return materialize_box(spec, box_id=req.box_id, force=req.force)
    except ValueError as e:
        # Assembly validation failures carry actionable, user-facing text.
        raise HTTPException(400, str(e)) from e
    except Exception as e:  # noqa: BLE001
        raise HTTPException(400, f"Box generation failed: {e!r}") from e


@router.delete("/{box_id}")
def delete_one(box_id: str) -> dict[str, Any]:
    ok = delete_box(box_id)
    if not ok:
        raise HTTPException(404, f"Unknown box: {box_id}")
    return {"ok": True, "id": box_id}
