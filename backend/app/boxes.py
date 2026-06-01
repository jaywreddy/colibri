"""Box model — six glass plates assembled into a rectangular cuboid.

A ``BoxSpec`` is the user-level recipe for a complete box. Each face is its
own ``PlateSpec`` with independent central pattern, frame, and dimensions,
but face dimensions are *derived* from the shared ``(W, H, D)`` box size so
opposing faces stay congruent.

Face-id → derived plate dimensions (W = width / x, H = height / y, D = depth / z):
    front, back   →  W × H
    top, bottom   →  W × D
    left, right   →  D × H
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any

from .plates import PlateSpec, materialize_plate
from .service import DATA_ROOT


_log = logging.getLogger("optics.boxes")

BOXES_ROOT = DATA_ROOT / "boxes"

FACE_IDS = ("front", "back", "top", "bottom", "left", "right")


def face_dimensions(face_id: str, width_um: float, height_um: float, depth_um: float) -> tuple[float, float]:
    """Return (W, H) for the given face — the local coords on that face."""
    if face_id in ("front", "back"):
        return width_um, height_um
    if face_id in ("top", "bottom"):
        return width_um, depth_um
    if face_id in ("left", "right"):
        return depth_um, height_um
    raise ValueError(f"Unknown face id: {face_id}")


@dataclass
class BoxSpec:
    """A full box: 6 plate specs + shared box dimensions.

    The weld margin is a *box-level* constraint — every face shares the same
    blank rim around its edges so the assembly bonds register correctly.
    ``normalize_face_dims`` collapses the box value onto each face's PlateSpec
    before materialize so the per-face cache key reflects the chosen weld.
    """

    width_um: float = 30000.0   # 30 mm cube — a fist-size desk object
    height_um: float = 30000.0
    depth_um: float = 30000.0
    weld_margin_um: float = 1000.0
    faces: dict[str, PlateSpec] = field(default_factory=dict)
    label: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "width_um": self.width_um,
            "height_um": self.height_um,
            "depth_um": self.depth_um,
            "weld_margin_um": self.weld_margin_um,
            "faces": {fid: p.to_dict() for fid, p in self.faces.items()},
            "label": self.label,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BoxSpec":
        return cls(
            width_um=float(data.get("width_um", 30000.0)),
            height_um=float(data.get("height_um", 30000.0)),
            depth_um=float(data.get("depth_um", 30000.0)),
            weld_margin_um=float(data.get("weld_margin_um", 1000.0)),
            faces={fid: PlateSpec.from_dict(p) for fid, p in data.get("faces", {}).items()},
            label=data.get("label", ""),
        )

    def normalize_face_dims(self) -> None:
        """Stamp box dims + weld margin onto every face's plate spec.

        Per-face dimensions and welds are not an independent degree of freedom
        — opposing faces must stay congruent and welds register at assembly.
        We collapse the values here so the per-face cache key reflects what
        actually gets materialized.
        """
        for fid in FACE_IDS:
            if fid not in self.faces:
                continue
            w, h = face_dimensions(fid, self.width_um, self.height_um, self.depth_um)
            self.faces[fid].width_um = w
            self.faces[fid].height_um = h
            self.faces[fid].weld_margin_um = self.weld_margin_um


def box_hash(spec: BoxSpec) -> str:
    raw = json.dumps(spec.to_dict(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(raw.encode()).hexdigest()[:12]


def materialize_box(spec: BoxSpec, *, box_id: str | None = None, force: bool = False) -> dict[str, Any]:
    """Fan out per face → materialize each plate → assemble a box manifest.

    ``box_id`` is the persistent identifier for the *saved preset* (user-named
    box). If omitted, a fresh UUID is allocated. Per-face plates are cached by
    their content hash under ``data/plates/`` and shared across boxes — so
    swapping one face's frame seed re-renders only that face.
    """
    BOXES_ROOT.mkdir(parents=True, exist_ok=True)
    spec.normalize_face_dims()
    bid = box_id or uuid.uuid4().hex[:12]
    out = BOXES_ROOT / bid
    out.mkdir(parents=True, exist_ok=True)
    box_path = out / "box.json"

    t0 = time.perf_counter()
    face_manifests: dict[str, dict[str, Any]] = {}
    for fid in FACE_IDS:
        plate_spec = spec.faces.get(fid)
        if plate_spec is None:
            continue
        face_manifests[fid] = materialize_plate(plate_spec, force=force)

    box_manifest = {
        "kind": "box",
        "id": bid,
        "spec": spec.to_dict(),
        "name": spec.label or f"Box {bid}",
        "faces": face_manifests,
        "dimensions_um": {
            "width": spec.width_um,
            "height": spec.height_um,
            "depth": spec.depth_um,
        },
        "content_hash": box_hash(spec),
    }
    box_path.write_text(json.dumps(box_manifest, indent=2))
    dt_ms = int((time.perf_counter() - t0) * 1000)
    _log.info("materialize_box id=%s faces=%d %dms", bid, len(face_manifests), dt_ms)
    return box_manifest


def list_boxes() -> list[dict[str, Any]]:
    if not BOXES_ROOT.exists():
        return []
    out: list[dict[str, Any]] = []
    for d in sorted(BOXES_ROOT.iterdir()):
        m = d / "box.json"
        if m.exists():
            try:
                out.append(json.loads(m.read_text()))
            except Exception:  # noqa: BLE001 — skip corrupt
                continue
    return out


def get_box(box_id: str) -> dict[str, Any] | None:
    m = BOXES_ROOT / box_id / "box.json"
    if not m.exists():
        return None
    return json.loads(m.read_text())


def delete_box(box_id: str) -> bool:
    box_path = BOXES_ROOT / box_id / "box.json"
    if not box_path.exists():
        return False
    box_path.unlink()
    # Leave the data dir + per-face plate caches; just remove the saved preset.
    try:
        (BOXES_ROOT / box_id).rmdir()
    except OSError:
        pass
    return True
