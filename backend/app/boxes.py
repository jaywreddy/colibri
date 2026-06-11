"""Ring box model — six glass plates assembled stained-glass style.

A ``BoxSpec`` (v2) is the user-level recipe for a complete ring box: outer
``(W, D, H)`` dimensions plus box-level glass, copper-foil, and hinge specs.
Each face is its own ``PlateSpec`` carrying the central pattern + frame, but
its cut dimensions, glass, and keep-out rim are *stamped* from the box level
by ``normalize_face_dims`` — they are not independent degrees of freedom.

Construction (see assembly.py for the full math):
  * Body = bottom + 4 walls, copper-foiled and fully soldered (8 seams).
  * Lid = flat top plate on a brass tube-and-rod hinge along the back top
    edge; its rim is tinned foil only, never soldered.
  * The foil fold-over + safety margin becomes each face's ``weld_margin_um``
    (the existing blank-rim mechanism in plates.py), keeping gold clear of
    the copper tape.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from .assembly import (
    FACE_IDS,
    FoilSpec,
    HingeSpec,
    assembly_summary,
    face_cut_dims,
    keepout_um,
    validate_assembly,
)
from .plates import FrameSpec, GlassSpec, PlateSpec, materialize_plate
from .service import DATA_ROOT


_log = logging.getLogger("optics.boxes")

BOXES_ROOT = DATA_ROOT / "boxes"

DEFAULT_FACE_PATTERN_SLUG = "wayuu-kanasu-moire"

# Anonymous (live-preview) generates all land in this single scratch slot so
# debounced edits never pile up saved boxes. The leading underscores keep it
# out of reach of UI-generated preset slugs (those are [a-z0-9-] only).
SCRATCH_BOX_ID = "__scratch"

# Filesystem-safe box ids: 1-64 chars, letters/digits/._- with a first char
# that can't start a traversal ('..' and absolute/illegal paths are rejected).
_BOX_ID_RE = re.compile(r"[A-Za-z0-9_][A-Za-z0-9._-]{0,63}")


def _box_dir(box_id: str) -> Path | None:
    """``BOXES_ROOT/<box_id>`` if the id is filesystem-safe, else ``None``.

    ``box_id`` arrives verbatim from the API; without this check a name like
    ``'../../evil'`` would read/write outside ``data/boxes``.
    """
    if not _BOX_ID_RE.fullmatch(box_id):
        return None
    return BOXES_ROOT / box_id


@dataclass
class BoxSpec:
    """A full ring box: 6 plate specs + box-level glass/foil/hinge.

    Glass, foil keep-out, and per-face cut dims are box-level constraints —
    every face shares them so the assembly registers correctly. The keep-out
    (foil overlap + safety) collapses onto each face's ``weld_margin_um``
    in ``normalize_face_dims`` before materialize, so the per-face cache key
    reflects the actual blank rim.
    """

    width_um: float = 50000.0   # outer X — default 50 mm ring box
    depth_um: float = 50000.0   # outer Z
    height_um: float = 40000.0  # outer Y
    glass: GlassSpec = field(default_factory=GlassSpec)
    foil: FoilSpec = field(default_factory=FoilSpec)
    hinge: HingeSpec = field(default_factory=HingeSpec)
    faces: dict[str, PlateSpec] = field(default_factory=dict)
    label: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "width_um": self.width_um,
            "depth_um": self.depth_um,
            "height_um": self.height_um,
            "glass": {
                "thickness_um": self.glass.thickness_um,
                "material": self.glass.material,
                "n": self.glass.n,
            },
            "foil": self.foil.to_dict(),
            "hinge": self.hinge.to_dict(),
            "faces": {fid: p.to_dict() for fid, p in self.faces.items()},
            "label": self.label,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BoxSpec":
        """Tolerant parse: any missing v2 field falls back to its default."""
        glass_data = data.get("glass") or {}
        return cls(
            width_um=float(data.get("width_um", 50000.0)),
            depth_um=float(data.get("depth_um", 50000.0)),
            height_um=float(data.get("height_um", 40000.0)),
            glass=GlassSpec(
                thickness_um=float(glass_data.get("thickness_um", 500.0)),
                material=str(glass_data.get("material", "fused silica")),
                n=float(glass_data.get("n", 1.46)),
            ),
            foil=FoilSpec.from_dict(data.get("foil")),
            hinge=HingeSpec.from_dict(data.get("hinge")),
            faces={fid: PlateSpec.from_dict(p) for fid, p in (data.get("faces") or {}).items()},
            label=str(data.get("label", "")),
        )

    def normalize_face_dims(self) -> None:
        """Stamp derived cut dims + foil keep-out + box glass onto every face.

        Per-face dims follow the stained-glass cut list (walls sit on the
        bottom; left/right walls fit between front/back), the keep-out rim
        lands in ``weld_margin_um``, and the box glass spec replaces whatever
        the face carried — none of these are face-level choices in a box.
        """
        ko = keepout_um(self.foil, self.glass.thickness_um)
        for fid in FACE_IDS:
            plate = self.faces.get(fid)
            if plate is None:
                continue
            w, h = face_cut_dims(
                fid, self.width_um, self.depth_um, self.height_um, self.glass.thickness_um
            )
            plate.width_um = w
            plate.height_um = h
            plate.weld_margin_um = ko
            plate.glass = replace(self.glass)


def default_box_spec() -> BoxSpec:
    """The default ring box — MUST match the frontend's ``defaultBoxSpec()``.

    50 x 50 x 40 mm, 1/4" foil, 5-segment tube hinge, Wayuu kanasü moiré on
    all six faces with frame seeds 100..105 so each face renders distinctly.
    """
    spec = BoxSpec()
    for i, fid in enumerate(FACE_IDS):
        spec.faces[fid] = PlateSpec(
            pattern_slug=DEFAULT_FACE_PATTERN_SLUG,
            frame=FrameSpec(seed=100 + i),
        )
    spec.normalize_face_dims()
    return spec


def box_hash(spec: BoxSpec) -> str:
    raw = json.dumps(spec.to_dict(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(raw.encode()).hexdigest()[:12]


def _lean_face_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    """Shallow-copy a plate manifest and drop ``recipe_data['frame_scene']``.

    The frame scene is megabytes of segment dicts the box-level consumers
    never need. New plate manifests keep it in a ``scene.json`` sidecar
    (see plates.py), but old cached manifests may still embed it — strip
    defensively. A shallow copy suffices: ``materialize_plate`` returns a
    fresh ``json.loads`` per call and ``materialize_box`` only serializes the
    result (the old ``json.loads(json.dumps(...))`` round-trip was ~60% of a
    warm 6-face box regen).
    """
    lean = dict(manifest)
    rd = dict(manifest.get("recipe_data") or {})
    rd.pop("frame_scene", None)
    lean["recipe_data"] = rd
    return lean


def materialize_box(spec: BoxSpec, *, box_id: str | None = None, force: bool = False) -> dict[str, Any]:
    """Validate, fan out per face, and assemble the v2 box manifest.

    ``box_id`` is the persistent identifier for the *saved preset* (user-named
    box). Anonymous generates (live-preview regens with no id) all reuse the
    single ``SCRATCH_BOX_ID`` slot — they overwrite each other, never appear
    in ``list_boxes``, and can't grow ``data/boxes`` without bound. Per-face
    plates are cached by their content hash under ``data/plates/`` and shared
    across boxes — so swapping one face's frame seed re-renders only that face.
    """
    box_id = box_id or None  # treat "" like absent
    saved = box_id is not None
    if box_id is not None and _box_dir(box_id) is None:
        raise ValueError(
            f"Invalid box_id {box_id!r}: use 1-64 letters, digits, '.', '_' or '-' "
            "(must not start with '.' or '-')."
        )

    spec.normalize_face_dims()
    validate_assembly(
        spec.width_um, spec.depth_um, spec.height_um,
        spec.glass.thickness_um, spec.foil, spec.hinge,
    )

    BOXES_ROOT.mkdir(parents=True, exist_ok=True)
    bid = box_id or SCRATCH_BOX_ID
    out = BOXES_ROOT / bid
    out.mkdir(parents=True, exist_ok=True)
    box_path = out / "box.json"

    t0 = time.perf_counter()
    face_manifests: dict[str, dict[str, Any]] = {}
    for fid in FACE_IDS:
        plate_spec = spec.faces.get(fid)
        if plate_spec is None:
            continue
        face_manifests[fid] = _lean_face_manifest(materialize_plate(plate_spec, force=force))

    box_manifest = {
        "kind": "box",
        "id": bid,
        # Anonymous live-preview boxes are not presets — list_boxes skips them.
        "saved": saved,
        "spec": spec.to_dict(),
        "name": spec.label or f"Box {bid}",
        "faces": face_manifests,
        "dimensions_um": {
            "width": spec.width_um,
            "height": spec.height_um,
            "depth": spec.depth_um,
        },
        "assembly": assembly_summary(
            spec.width_um, spec.depth_um, spec.height_um,
            spec.glass.thickness_um, spec.foil, spec.hinge,
        ),
        "content_hash": box_hash(spec),
    }
    box_path.write_text(json.dumps(box_manifest, indent=2))
    dt_ms = int((time.perf_counter() - t0) * 1000)
    _log.info("materialize_box id=%s faces=%d %dms", bid, len(face_manifests), dt_ms)
    return box_manifest


def list_boxes() -> list[dict[str, Any]]:
    """All saved boxes, skipping corrupt or old-format (pre-v2) box.json files.

    Anonymous live-preview manifests (``saved: false``, the scratch slot) are
    excluded — only explicitly saved presets belong in the dropdown.
    """
    if not BOXES_ROOT.exists():
        return []
    out: list[dict[str, Any]] = []
    for d in sorted(BOXES_ROOT.iterdir()):
        m = d / "box.json"
        if m.exists():
            try:
                manifest = json.loads(m.read_text())
                # v2 manifests always carry the derived assembly block —
                # anything without it predates the ring-box redesign.
                if manifest.get("kind") != "box" or "assembly" not in manifest:
                    _log.warning("list_boxes skipping old-format box.json: %s", m)
                    continue
                if manifest.get("saved") is False:
                    continue  # scratch / anonymous regen — not a preset
                out.append(manifest)
            except Exception:  # noqa: BLE001 — skip corrupt
                continue
    return out


def get_box(box_id: str) -> dict[str, Any] | None:
    d = _box_dir(box_id)
    if d is None:
        return None
    m = d / "box.json"
    if not m.exists():
        return None
    return json.loads(m.read_text())


def delete_box(box_id: str) -> bool:
    d = _box_dir(box_id)
    if d is None:
        return False
    box_path = d / "box.json"
    if not box_path.exists():
        return False
    box_path.unlink()
    # Leave the data dir + per-face plate caches; just remove the saved preset.
    try:
        d.rmdir()
    except OSError:
        pass
    return True
