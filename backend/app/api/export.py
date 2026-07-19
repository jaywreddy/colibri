"""Fab export — zips per-plate masks (PNG + SVG + manifest) into one archive.

Endpoints:
  GET /export/plate/{plate_id}/fab.zip — single plate bundle
  GET /export/box/{box_id}/fab.zip     — all 6 plate bundles + box manifest
                                         + CUTLIST.csv + ASSEMBLY.md

The box archive is everything the builder needs at the bench: lithography
masks per face, a glass cut list, and numbered stained-glass (copper foil +
solder) assembly steps with real dimensions.
"""
from __future__ import annotations

import io
import json
import zipfile
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from ..assembly import FoilSpec, HingeSpec, assembly_summary
from ..boxes import BoxSpec, get_box
from ..plates import PLATES_ROOT, ensure_plate_svg, get_plate

router = APIRouter(prefix="/export", tags=["export"])


def _plate_files_to_zip(zf: zipfile.ZipFile, plate_id: str, prefix: str = "") -> None:
    """Stream every file in a plate's data dir into ``zf`` under ``prefix``.

    Builds the SVGs lazily if they haven't been generated yet, and rewrites
    the manifest to drop ``frame_scene`` (the live-preview blob; ~MB of
    segment dicts that the engraver doesn't need).
    """
    plate_dir = PLATES_ROOT / plate_id
    if not plate_dir.exists():
        raise HTTPException(404, f"Plate data missing on disk: {plate_id}")
    ensure_plate_svg(plate_id)
    for f in sorted(plate_dir.iterdir()):
        if not f.is_file():
            continue
        if f.name == "scene.json":
            continue  # frame_scene sidecar — live-preview blob, not fab data
        if f.name == "manifest.json":
            try:
                m = json.loads(f.read_text())
            except Exception:  # noqa: BLE001
                zf.write(f, arcname=f"{prefix}{f.name}")
                continue
            rd = m.get("recipe_data") or {}
            rd.pop("frame_scene", None)
            m["recipe_data"] = rd
            zf.writestr(f"{prefix}{f.name}", json.dumps(m, indent=2))
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


def _box_assembly_block(box_manifest: dict[str, Any]) -> dict[str, Any]:
    """The manifest's assembly block, recomputed from spec if absent.

    Boxes saved before the v2 redesign lack the block; the formulas are
    deterministic from the spec, so we can always rebuild it.
    """
    block = box_manifest.get("assembly")
    if block:
        return block
    spec = BoxSpec.from_dict(box_manifest.get("spec", {}))
    return assembly_summary(
        spec.width_um, spec.depth_um, spec.height_um,
        spec.glass.thickness_um, spec.foil, spec.hinge,
    )


def _cutlist_csv(assembly: dict[str, Any]) -> str:
    """CUTLIST.csv — one glass plate per row, mm first for the saw bench."""
    t_mm = assembly["glass_thickness_um"] / 1000.0
    lines = ["face,width_mm,height_mm,thickness_mm,width_um,height_um"]
    for entry in assembly["cut_list"]:
        lines.append(
            f"{entry['face']},{entry['width_mm']},{entry['height_mm']},"
            f"{round(t_mm, 3)},{entry['width_um']},{entry['height_um']}"
        )
    return "\n".join(lines) + "\n"


def _assembly_md(box_manifest: dict[str, Any], assembly: dict[str, Any]) -> str:
    """ASSEMBLY.md — numbered stained-glass build steps with real numbers."""
    spec = box_manifest.get("spec", {})
    foil = FoilSpec.from_dict(spec.get("foil"))
    hinge = HingeSpec.from_dict(spec.get("hinge"))
    dims = box_manifest["dimensions_um"]
    w_mm = dims["width"] / 1000.0
    d_mm = dims["depth"] / 1000.0
    h_mm = dims["height"] / 1000.0
    t_mm = assembly["glass_thickness_um"] / 1000.0
    overlap_mm = assembly["overlap_um"] / 1000.0
    keepout_mm = assembly["keepout_um"] / 1000.0
    tape_mm = foil.tape_width_um / 1000.0
    safety_mm = foil.safety_um / 1000.0

    cut_rows = "\n".join(
        f"| {e['face']} | {e['width_mm']:.1f} | {e['height_mm']:.1f} | {t_mm:.1f} |"
        for e in assembly["cut_list"]
    )
    # Total foil = sum of plate perimeters (the tape wraps every plate edge).
    foil_total_mm = sum(
        2.0 * (e["width_um"] + e["height_um"]) for e in assembly["cut_list"]
    ) / 1000.0

    seams = assembly["seams"]
    bottom_seams = [s for s in seams if s["kind"] == "bottom"]
    corner_seams = [s for s in seams if s["kind"] == "corner"]
    seam_rows = "\n".join(
        f"| {s['id']} | {s['kind']} | {s['length_um'] / 1000.0:.1f} |" for s in seams
    )
    solder_total_mm = sum(s["length_um"] for s in seams) / 1000.0

    hg = assembly["hinge"]
    seg_mm = hg["segment_length_um"] / 1000.0
    run_mm = hg["run_length_um"] / 1000.0
    gap_mm = hg["gap_um"] / 1000.0
    rod_mm = hg["rod_length_um"] / 1000.0

    return f"""# {box_manifest.get('name', 'Ring box')} — stained-glass assembly

Outer dimensions: {w_mm:.1f} x {d_mm:.1f} x {h_mm:.1f} mm (W x D x H),
{t_mm:.1f} mm fused-silica plates, copper-foil construction with a brass
tube-and-rod lid hinge. Work the body first; the lid is hinged, never soldered.

## 1. Cut the glass

| face | width (mm) | height (mm) | thickness (mm) |
|---|---|---|---|
{cut_rows}

Walls sit ON the bottom plate and the lid rests on the wall rim, so the
walls are cut 2 x {t_mm:.1f} mm short of the box height, and the left/right
walls fit BETWEEN front/back ({2 * t_mm:.1f} mm narrower than the depth).

## 2. Check the gold keep-out

Each mask leaves a blank rim of {keepout_mm:.3f} mm on every edge:
{overlap_mm:.3f} mm of copper-foil fold-over plus a {safety_mm:.3f} mm
safety margin for hand alignment. The foil will exactly cover this rim
(minus the safety) — no gold may sit under the tape. Verify the etched
plates before foiling; re-fab any plate with gold inside the rim.

## 3. Foil every plate

Wrap {tape_mm:.3f} mm copper foil tape around all four edges of all six
plates, folding {overlap_mm:.3f} mm onto each face. Burnish flat.
Estimated total foil length: {foil_total_mm:.0f} mm (sum of plate perimeters).

## 4. Assemble the body

Place the 4 walls ON the bottom plate (outer surfaces flush with the
{w_mm:.1f} x {d_mm:.1f} mm footprint), left/right walls between front/back.
Tack-solder the corners to hold the geometry square.

## 5. Solder the 8 body seams

| seam | kind | length (mm) |
|---|---|---|
{seam_rows}

{len(bottom_seams)} bottom-perimeter seams + {len(corner_seams)} vertical
corner seams; about {solder_total_mm:.0f} mm of beaded seam total. Run a
continuous bead; the lid and the wall top rim get NO solder bead.

## 6. Tin the lid and rim

Tin (thin solder wipe, no bead) the lid's 4 foiled edges and the top rim of
the 4 walls. This is the finished contact surface the lid closes onto.

## 7. Build the hinge

- Cut brass tube ({hinge.tube_od_um / 1000.0:.2f} mm OD) into {hinge.segments}
  segments of {seg_mm:.2f} mm, leaving {gap_mm:.1f} mm clearance between
  segments ({run_mm:.1f} mm total run, centered on the back top edge).
- Cut the brass rod ({hinge.rod_od_um / 1000.0:.2f} mm OD) to {rod_mm:.1f} mm.
- Segments alternate body, lid, body, ... (both end segments on the BODY).
  Solder {hg['body_segments']} segments to the back wall's top rim and
  {hg['lid_segments']} to the lid's back edge, keeping all bores collinear.

## 8. Hang the lid

Thread the rod through all {hinge.segments} segments and cap the ends.
The lid should swing up and back over the hinge through ~120 degrees
without striking the back wall. Apply patina/finish ("{foil.finish}") last.
"""


@router.get("/wafer/plan")
def wafer_plan() -> dict[str, Any]:
    """The 4-inch-wafer packing plan for the largest fitting box.

    Pure math (no pattern generation): runs the deterministic
    rectangle-in-circle packer + max-scale solver from ``export_wafer`` and
    returns the mini-box dimensions plus each plate's wafer-centered footprint.
    The full GDS is produced offline via ``python -m app.export_wafer``.
    """
    from ..export_wafer import (
        DICING_STREET_UM,
        EDGE_EXCLUSION_UM,
        WAFER_DIAMETER_UM,
        solve_max_scale,
    )

    result = solve_max_scale()
    if result is None:
        raise HTTPException(500, "No box size fits the wafer — check constants.")
    return {
        "wafer_diameter_um": WAFER_DIAMETER_UM,
        "edge_exclusion_um": EDGE_EXCLUSION_UM,
        "dicing_street_um": DICING_STREET_UM,
        "mini_box": result.dims_mm(),
        "placements": [
            {
                "face": p.face,
                "cx_um": round(p.cx, 1),
                "cy_um": round(p.cy, 1),
                "width_um": round(p.width_um, 1),
                "height_um": round(p.height_um, 1),
                "rotated": p.rotated,
            }
            for p in result.placements
        ],
    }


@router.get("/box/{box_id}/fab.zip")
def box_fab_zip(box_id: str) -> StreamingResponse:
    box_manifest = get_box(box_id)
    if box_manifest is None:
        raise HTTPException(404, f"Unknown box: {box_id}")
    assembly = _box_assembly_block(box_manifest)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        # Per-face folder per plate so the engraver can register and cut each
        # independently. Box manifest lives at the root.
        for face_id, face in box_manifest.get("faces", {}).items():
            plate_id = face["id"]
            _plate_files_to_zip(zf, plate_id, prefix=f"{face_id}/")
        # Strip frame_scene from each face manifest in the box snapshot too
        # (v2 manifests are already lean; this also covers older saves).
        bm_lean = json.loads(json.dumps(box_manifest))  # cheap deep copy
        for face in bm_lean.get("faces", {}).values():
            rd = face.get("recipe_data") or {}
            rd.pop("frame_scene", None)
            face["recipe_data"] = rd
        zf.writestr("box.json", json.dumps(bm_lean, indent=2))
        zf.writestr("CUTLIST.csv", _cutlist_csv(assembly))
        zf.writestr("ASSEMBLY.md", _assembly_md(box_manifest, assembly))
        dims = box_manifest["dimensions_um"]
        zf.writestr(
            "README.txt",
            (
                f"Optics Pattern Studio — ring box {box_id}\n"
                f"Outer dimensions: {dims['width']} × {dims['height']} × {dims['depth']} μm\n\n"
                "Each subdirectory ({front,back,top,bottom,left,right}/) is a fab-ready\n"
                "plate bundle (SVG + PNG + manifest). At the root:\n"
                "  box.json    — assembly geometry + per-face plate ids\n"
                "  CUTLIST.csv — glass cut list (mm + μm)\n"
                "  ASSEMBLY.md — numbered copper-foil build steps\n"
            ),
        )
    buf.seek(0)
    return StreamingResponse(
        buf,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="box-{box_id}.zip"'},
    )
