from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from ..boxes import BoxSpec, delete_box, get_box, list_boxes, materialize_box
from .plates import PlateSpecBody

router = APIRouter(prefix="/boxes", tags=["boxes"])


class BoxSpecBody(BaseModel):
    width_um: float = 30000.0
    height_um: float = 30000.0
    depth_um: float = 30000.0
    weld_margin_um: float = 1000.0
    faces: dict[str, PlateSpecBody] = Field(default_factory=dict)
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
    except Exception as e:  # noqa: BLE001
        raise HTTPException(400, f"Box generation failed: {e!r}") from e


@router.delete("/{box_id}")
def delete_one(box_id: str) -> dict[str, Any]:
    ok = delete_box(box_id)
    if not ok:
        raise HTTPException(404, f"Unknown box: {box_id}")
    return {"ok": True, "id": box_id}
