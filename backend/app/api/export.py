"""Fab export — zips per-plate masks (PNG + SVG + manifest) into one archive.

Endpoints:
  GET /export/plate/{plate_id}/fab.zip — single plate bundle
  GET /export/box/{box_id}/fab.zip     — all 6 plate bundles + box manifest
"""
from __future__ import annotations

import io
import json
import zipfile

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from ..boxes import BOXES_ROOT, get_box
from ..plates import PLATES_ROOT, ensure_plate_svg, get_plate

router = APIRouter(prefix="/export", tags=["export"])


def _plate_files_to_zip(zf: zipfile.ZipFile, plate_id: str, prefix: str = "") -> None:
    """Stream every file in a plate's data dir into ``zf`` under ``prefix``.

    Builds the SVGs lazily if they haven't been generated yet — the
    interactive materialize path skips the slow polygon SVG to keep edits
    snappy, so the fab bundle is the first time we pay that cost.
    """
    plate_dir = PLATES_ROOT / plate_id
    if not plate_dir.exists():
        raise HTTPException(404, f"Plate data missing on disk: {plate_id}")
    ensure_plate_svg(plate_id)
    for f in sorted(plate_dir.iterdir()):
        if not f.is_file():
            continue
        zf.write(f, arcname=f"{prefix}{f.name}")


@router.get("/plate/{plate_id}/fab.zip")
def plate_fab_zip(plate_id: str) -> StreamingResponse:
    manifest = get_plate(plate_id)
    if manifest is None:
        raise HTTPException(404, f"Unknown plate: {plate_id}")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        _plate_files_to_zip(zf, plate_id)
        # Plain-text README for the engraver
        zf.writestr(
            "README.txt",
            (
                f"Optics Pattern Studio — plate {plate_id}\n"
                f"Pattern: {manifest['spec']['pattern_slug']}\n"
                f"Plate size: {manifest['extent_um'][0]} × {manifest['extent_um'][1]} μm\n"
                f"Substrate: {manifest['substrate']['material']}, "
                f"{manifest['substrate']['thickness_um']} μm thick (n={manifest['substrate']['n']})\n\n"
                "Files:\n"
                "  front.svg / front.png — gold mask, viewer-facing side (includes frame)\n"
                "  back.svg  / back.png  — gold mask, far side\n"
                "  manifest.json         — full plate spec + generation parameters\n"
                "  thumbnail.png         — composite preview\n"
            ),
        )
    buf.seek(0)
    return StreamingResponse(
        buf,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="plate-{plate_id}.zip"'},
    )


@router.get("/box/{box_id}/fab.zip")
def box_fab_zip(box_id: str) -> StreamingResponse:
    box_manifest = get_box(box_id)
    if box_manifest is None:
        raise HTTPException(404, f"Unknown box: {box_id}")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        # Per-face folder per plate so the engraver can register and cut each
        # independently. Box manifest lives at the root.
        for face_id, face in box_manifest.get("faces", {}).items():
            plate_id = face["id"]
            _plate_files_to_zip(zf, plate_id, prefix=f"{face_id}/")
        zf.writestr("box.json", json.dumps(box_manifest, indent=2))
        dims = box_manifest["dimensions_um"]
        zf.writestr(
            "README.txt",
            (
                f"Optics Pattern Studio — box {box_id}\n"
                f"Outer dimensions: {dims['width']} × {dims['height']} × {dims['depth']} μm\n\n"
                "Each subdirectory ({front,back,top,bottom,left,right}/) is a fab-ready\n"
                "plate bundle (SVG + PNG + manifest). The top-level box.json describes\n"
                "the assembly and ties the per-face plate ids together.\n"
            ),
        )
    buf.seek(0)
    return StreamingResponse(
        buf,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="box-{box_id}.zip"'},
    )
