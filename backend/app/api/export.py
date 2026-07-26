"""Fab export — zips per-plate masks (fine GDS + PNG + SVG + manifest) into one
archive.

Endpoints:
  GET  /export/plate/{plate_id}/fab.zip   — single plate bundle
  POST /export/box/{box_id}/fab/start     — 202 {job_id}: build in a SUBPROCESS
  GET  /export/jobs/{job_id}              — {status, progress, error}
  GET  /export/jobs/{job_id}/fab.zip      — the finished archive
  GET  /export/box/{box_id}/fab.zip       — legacy synchronous build (same body)

The box archive is everything the builder needs at the bench: lithography
masks per face, a glass cut list, and numbered stained-glass (copper foil +
solder) assembly steps with real dimensions. It does not need a warm cache:
``data/`` is disposable, so any face whose plate dir is gone is recomposed
from the saved box spec (sequentially) before zipping.

A cold box export is minutes of plate composition and klayout DRC, so the
preferred path is the JOB path: ``fab/start`` spawns ``python -m
app.export_job`` and returns immediately, the client polls
``/export/jobs/{job_id}`` for a named phase + faces_done/faces_total, and
downloads the archive when the status flips to ``done``. That keeps the heavy
compute out of the request thread (it used to hold the GIL long enough to
starve the live preview) and gives the UI something true to show instead of a
spinner that looks hung. ``build_box_fab_zip`` is the ONE build body — the
worker and the legacy synchronous endpoint both call it, so they cannot drift.

Exports are the heaviest thing this app does, so exactly ONE runs at a time
across the host: the in-process slot (``_export_slot``, backed by
``export_job.claim_slot``) and job subprocess liveness are the same slot, and a
request that arrives while either is busy gets 429 instead of silently queueing
behind minutes of compute. Polling a job is not heavy work and takes no slot.

Two geometries ship per face and they are NOT interchangeable: ``fine.gds`` is
the fab-grade mask, written by ``export_fine`` as exact vector rectangles at the
TRUE optical periods, while ``front.svg``/``back.svg`` are the budget-rasterized
previews whose baked periods may be coarsened to fit the cell budget (the
manifest's ``svg_bake_*`` fields record what actually got baked). The README in
each archive says the same thing to the engraver.
"""
from __future__ import annotations

import io
import json
import logging
import math
import os
import re
import tempfile
import zipfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, NamedTuple

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, StreamingResponse

from .. import export_job
from ..assembly import FoilSpec, HingeSpec, assembly_summary
from ..boxes import BoxSpec, get_box
from ..export_job import ExportBusy, Progress
from ..plates import (
    PLATES_ROOT,
    PlateSpec,
    ensure_plate_svg,
    get_plate,
    materialize_plate,
)

_log = logging.getLogger("optics.export")

router = APIRouter(prefix="/export", tags=["export"])

# Per-face fab-grade mask + its QA sidecar.
FINE_GDS_NAME = "fine.gds"
FINE_SUMMARY_NAME = "FINE_MASKS.json"


def _json_finite(obj: Any) -> Any:
    """Replace ±inf/NaN with None recursively.

    ``min_width_um``/``min_space_um`` are ``inf`` when a layer has zero
    violations; Python's ``json.dumps`` emits bare ``Infinity`` for those,
    which strict JSON parsers (any JS consumer of FINE_MASKS.json) reject.
    ``null`` is the honest strict-JSON spelling of "no sub-floor feature".
    """
    if isinstance(obj, dict):
        return {k: _json_finite(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_finite(v) for v in obj]
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj

# The fine builders are the heaviest compute in the app (per-plate vector
# composition + a merged klayout DRC heal, one wafer-writer face each). FastAPI
# runs these sync endpoints in a worker thread, so two concurrent fab.zip
# downloads would put two heavy compute processes on a host that kernel-bugchecks
# under exactly that (CLAUDE.md). One HOST-wide slot, taken for the WHOLE
# export body — plate rebuilds, the lazy SVG bakes inside _plate_files_to_zip,
# and the per-face fine loop are all heavy compute, and gating only the fine
# loop let box A's SVG bake run beside box B's fine build.
#
# The slot itself lives in ``export_job`` because it is no longer only about
# threads: a live export SUBPROCESS holds it too, so an in-process export cannot
# start beside a job (and vice versa). It stays reentrant per thread — the inner
# claims in _FineMaskRun.write_face / _rebuild_face_plate nest inside the build
# body's, and one export runs entirely in one worker thread.
_EXPORT_LOCK_TIMEOUT_S = export_job.SLOT_WAIT_S


@contextmanager
def _export_slot() -> Iterator[None]:
    """Hold the host-wide heavy-compute slot for one in-process export, or 429.

    Reentrant, so a nested claim by the same request is free. A blocked caller
    is told to retry rather than queued: exports run minutes, and a queued
    HTTP download has no way to say "waiting" — which is what the job routes
    exist to fix.
    """
    try:
        export_job.claim_slot(_EXPORT_LOCK_TIMEOUT_S)
    except ExportBusy as exc:
        raise HTTPException(429, str(exc), headers={"Retry-After": "10"}) from exc
    try:
        yield
    finally:
        export_job.release_slot()


# Plate ids are content hashes (12 hex chars from ``plate_hash``), but they
# reach us out of a box.json that may have been hand-edited — keep them from
# addressing anything but a direct child of PLATES_ROOT.
_PLATE_ID_RE = re.compile(r"[A-Za-z0-9_-]{1,64}")


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


def _fine_gds_bytes(fine: Any, spec: PlateSpec, face: str) -> bytes:
    """One plate's healed fine geometry → GDS bytes, plate-centered.

    Emits the DRC-healed plate-frame rings ``build_plate_fine`` produced on the
    same layer map the wafer writer uses (front gold 10/0, back gold 20/0,
    outline 1/0, label 3/0) so a plate bundle and the wafer export overlay. µm
    user units at a 1 nm database unit, origin at the plate center — the same
    frame the SVGs use, minus the SVG's period coarsening. No packing rotation:
    a plate bundle is written as the plate is cut.
    """
    import klayout.db as kdb

    from ..export_fine import (
        DBU_UM,
        LAYER_BACK,
        LAYER_FRONT,
        LAYER_LABEL,
        LAYER_OUTLINE,
        insert_polys_deduped,
    )

    ly = kdb.Layout()
    ly.dbu = DBU_UM
    top = ly.create_cell(f"plate_{face}")
    L_front = ly.layer(*LAYER_FRONT)
    L_back = ly.layer(*LAYER_BACK)
    L_outline = ly.layer(*LAYER_OUTLINE)
    L_label = ly.layer(*LAYER_LABEL)

    # Hierarchical emission: one cell per unique shape + SREFs (a grating is
    # thousands of copies of one line — flat replication is what fab tools and
    # GDS itself exist to avoid). See insert_polys_deduped.
    for polys, layer, prefix in (
        (fine.front_polys, L_front, "f"),
        (fine.back_polys, L_back, "b"),
    ):
        st = insert_polys_deduped(top, layer, polys, cell_prefix=f"{face}_{prefix}")
        _log.info(
            "fine gds face=%s layer=%s cells=%d refs=%d flat=%d",
            face, prefix, st["cells"], st["refs"], st["flat"],
        )

    hw, hh = spec.width_um / 2.0, spec.height_um / 2.0
    top.shapes(L_outline).insert(kdb.DBox(-hw, -hh, hw, hh))
    top.shapes(L_label).insert(kdb.DText(face, kdb.DTrans(kdb.DVector(0.0, 0.0))))

    # klayout writes to a path only; the zip wants bytes and we deliberately do
    # NOT cache the GDS under data/plates/<id>/ — that dir is keyed by the spec
    # hash alone, so a cached fab mask would survive an export_fine geometry
    # change (the staleness class PLATE_SVG_VERSION exists to prevent).
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / f"{face}.gds"
        ly.write(str(out))
        return out.read_bytes()


def _fine_face_summary(fine: Any) -> dict[str, Any]:
    """The QA block for one face in FINE_MASKS.json — realized periods, polygon
    counts, and the merged-geometry DRC report of the geometry actually emitted."""
    st = fine.stats
    drc = st.get("drc") or {}
    block: dict[str, Any] = {
        "status": "ok",
        "file": FINE_GDS_NAME,
        "slug": fine.slug,
        "periods_um": st.get("periods", {}),
        "zone_boundary_pitch_um": st.get("pitch_um"),
        "front_polygons": st.get("front_polys"),
        "back_polygons": st.get("back_polys"),
        "drc_merged_after": {
            "front": drc.get("front_merged_after"),
            "back": drc.get("back_merged_after"),
        },
    }
    if st.get("scanimation") is not None:
        block["scanimation"] = st["scanimation"]
    return block


# Bump whenever export_fine geometry changes so cached fine masks regenerate —
# the fine.gds cache is keyed (plate_hash, FINE_GDS_VERSION), same staleness
# contract as PLATE_SVG_VERSION / PLATE_COMPOSE_VERSION (see CLAUDE.md).
#   v2: hierarchical emission (cell-per-unique-shape + SREFs, ≤1 nm placement
#       rounding) replaced flat per-polygon writes.
#   v3: the capybara water band excludes the SUBMERGED BODY — the exact vector
#       slit-barrier comb and ripple interleave are carved around the animal (and
#       the back carrier is kept over it), matching the composed preview. Capybara
#       faces only; every other face's geometry is byte-identical.
FINE_GDS_VERSION = 3


def _fine_cache_dir(spec: PlateSpec) -> Path:
    from .. import plates as P

    return P.PLATES_ROOT / P.plate_hash(spec) / f"fine_v{FINE_GDS_VERSION}"


def _fine_cache_read(spec: PlateSpec) -> tuple[bytes, dict[str, Any]] | None:
    """Cached (gds bytes, FINE_MASKS face block) for this spec, or None.

    The version-suffixed dir name IS the staleness check; a corrupt slot is a
    miss, matching the corrupt-manifest-as-miss rule in service/plates.
    """
    d = _fine_cache_dir(spec)
    try:
        data = (d / FINE_GDS_NAME).read_bytes()
        block = json.loads((d / "summary.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not data:
        return None
    return data, block


def _fine_cache_write(spec: PlateSpec, data: bytes, block: dict[str, Any]) -> None:
    """Publish atomically: gds first, summary.json last (its presence = complete)."""
    d = _fine_cache_dir(spec)
    try:
        d.mkdir(parents=True, exist_ok=True)
        tmp_gds = d / (FINE_GDS_NAME + ".tmp")
        tmp_gds.write_bytes(data)
        os.replace(tmp_gds, d / FINE_GDS_NAME)
        tmp_sum = d / "summary.json.tmp"
        tmp_sum.write_text(json.dumps(_json_finite(block), indent=2, default=float), encoding="utf-8")
        os.replace(tmp_sum, d / "summary.json")
    except OSError:
        _log.exception("fine mask cache write failed dir=%s", d)


@contextmanager
def _drc_phase_probe(progress: Progress, face_id: str) -> Iterator[None]:
    """Report the merged-region DRC heal as its own progress phase.

    ``build_plate_fine`` composes vector geometry and then heals it in ONE call
    and takes no progress hook, yet the heal is the slow half (klayout on the
    merged region) — a UI that cannot name it shows a frozen "compose" for
    minutes. The honest transition marker is the build's FIRST call to
    ``drc_clean_region``, so we wrap that name in ``export_fine``'s namespace
    for the duration of one face and restore it after. Exports are serialized
    host-wide, so there is exactly one user of the patched name at a time, and
    if the name ever moves the probe degrades to "no heal sub-phase" instead of
    breaking the build.
    """
    from .. import export_fine

    original = getattr(export_fine, "drc_clean_region", None)
    if original is None or not callable(original):
        yield
        return
    fired = False

    def probed(*args: Any, **kwargs: Any) -> Any:
        nonlocal fired
        if not fired:
            fired = True
            progress.step("fine mask (DRC heal)", face=face_id, detail="merged-region heal")
        return original(*args, **kwargs)

    export_fine.drc_clean_region = probed
    try:
        yield
    finally:
        export_fine.drc_clean_region = original


class _FineMaskRun:
    """One export's fab-mask pass: per-face ``fine.gds`` + the root FINE_MASKS.json.

    Faces go in ONE AT A TIME through ``write_face`` — interleaved with each
    face's plate-file copy so progress is reported per face — and the root
    summary is published once by ``finish``. Never parallelize: per-plate vector
    composition plus a merged klayout DRC heal is the heaviest compute in the
    app and two at once bugchecks this host (CLAUDE.md).

    A face whose build fails gets a loud ``FINE_BUILD_FAILED.txt`` marker and a
    ``failed`` status instead of silently shipping a bundle that looks fab-grade
    with no mask in it — the SVGs alone are not releasable geometry. If the
    klayout/DRC bridge is missing entirely, ``finish`` writes an ``unavailable``
    summary and the archive carries no mask at all.
    """

    def __init__(self) -> None:
        self.summary: dict[str, Any] = {}
        self._build: Any = None
        self._unavailable: str | None = None
        try:
            from ..export_fine import (
                LAYER_BACK,
                LAYER_FRONT,
                LAYER_OUTLINE,
                LITHO_FLOOR_UM,
                build_plate_fine,
            )
        except Exception as exc:  # noqa: BLE001 — klayout / DRC bridge unavailable
            _log.exception("fine mask writer unavailable")
            self._unavailable = f"{type(exc).__name__}: {exc}"
            return
        self._build = build_plate_fine
        self.summary["litho_floor_um"] = LITHO_FLOOR_UM
        self.summary["layers"] = {
            "front_gold": list(LAYER_FRONT),
            "back_gold": list(LAYER_BACK),
            "outline": list(LAYER_OUTLINE),
        }
        self.summary["faces"] = {}

    @property
    def available(self) -> bool:
        return self._build is not None

    def write_face(
        self,
        zf: zipfile.ZipFile,
        face_id: str,
        prefix: str,
        spec: PlateSpec,
        progress: Progress,
    ) -> None:
        """Build (or serve from cache) one face's TRUE-pitch mask into ``zf``.

        Claims the export slot per face: normally reentrant free (the build body
        already holds it), but it keeps the guarantee local to the one function
        that starts a heavy compute.
        """
        if self._build is None:
            return
        with _export_slot():
            cached = _fine_cache_read(spec)
            if cached is not None:
                data, block = cached
                zf.writestr(f"{prefix}{FINE_GDS_NAME}", data)
                self.summary["faces"][face_id] = block
                progress.step(
                    "fine mask (cached)", face=face_id, detail=f"{len(data)} bytes"
                )
                return
            progress.step("fine mask (compose)", face=face_id, detail=spec.pattern_slug)
            try:
                with _drc_phase_probe(progress, face_id):
                    fine = self._build(spec, face_id)
                data = _fine_gds_bytes(fine, spec, face_id)
            except Exception as exc:  # noqa: BLE001 — one bad face must not hide the rest
                _log.exception(
                    "fine mask build failed face=%s slug=%s", face_id, spec.pattern_slug
                )
                detail = f"{type(exc).__name__}: {exc}"
                self.summary["faces"][face_id] = {
                    "status": "failed",
                    "slug": spec.pattern_slug,
                    "error": detail,
                }
                zf.writestr(
                    f"{prefix}FINE_BUILD_FAILED.txt",
                    (
                        f"Fab-grade mask generation FAILED for face '{face_id}'.\n"
                        f"{detail}\n\n"
                        "This face is NOT fab-releasable: front.svg/back.svg are\n"
                        "budget-rasterized previews whose optical periods may be\n"
                        "coarsened, so they cannot be used to write gold.\n"
                    ),
                )
                return
            zf.writestr(f"{prefix}{FINE_GDS_NAME}", data)
            block = _fine_face_summary(fine)
            self.summary["faces"][face_id] = block
            _fine_cache_write(spec, data, block)
            _log.info(
                "fine mask face=%s slug=%s front=%d back=%d bytes=%d",
                face_id, fine.slug, fine.stats.get("front_polys", 0),
                fine.stats.get("back_polys", 0), len(data),
            )

    def finish(self, zf: zipfile.ZipFile) -> None:
        """Publish the root ``FINE_MASKS.json`` for the faces written so far."""
        if self._build is None:
            zf.writestr(
                FINE_SUMMARY_NAME,
                json.dumps(
                    {
                        "status": "unavailable",
                        "error": self._unavailable,
                        "note": (
                            "No fab-grade mask in this archive. front.svg/back.svg are "
                            "coarsened previews — do NOT write gold from them."
                        ),
                        "faces": {},
                    },
                    indent=2,
                ),
            )
            return
        zf.writestr(
            FINE_SUMMARY_NAME,
            json.dumps(_json_finite(self.summary), indent=2, default=float),
        )


# Shared README paragraph: which files are masks and which are previews. The
# svg_bake_* manifest fields record the periods the SVG raster actually baked.
_FINE_VS_PREVIEW_TEXT = (
    "Which files are the mask:\n"
    "  fine.gds is THE fab-grade mask — exact vector geometry at the TRUE\n"
    "  optical periods (back carrier / front louvre / centerpiece switch /\n"
    "  diffraction accent as designed), DRC-healed to the 2 μm litho floor.\n"
    "  Pattern the part from fine.gds. FINE_MASKS.json lists each face's\n"
    "  realized periods, polygon counts and merged-geometry DRC report.\n"
    "  front.svg / back.svg (and the PNGs) are PREVIEWS, not masks. They are\n"
    "  rasterized under a fixed cell budget, and when the true period is finer\n"
    "  than that raster can carry the bake COARSENS the optical periods to fit\n"
    "  — by a large factor on a big plate. The effective baked values are in\n"
    "  manifest.json under the svg_bake_* fields; compare them against the\n"
    "  design fab_*_period_um fields in the same file before trusting any\n"
    "  period you measure in an SVG. Use the SVG/PNG pair for layout,\n"
    "  registration and keep-out checks only — never to write gold.\n"
)


@router.get("/plate/{plate_id}/fab.zip")
def plate_fab_zip(plate_id: str) -> StreamingResponse:
    """One plate's bundle. Lookup first (an unknown id must 404, not 429), then
    the whole build under the export slot — the lazy SVG bake is heavy too."""
    manifest = get_plate(plate_id)
    if manifest is None:
        raise HTTPException(404, f"Unknown plate: {plate_id}")
    buf = io.BytesIO()
    with _export_slot(), zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        _plate_files_to_zip(zf, plate_id)
        fine = _FineMaskRun()
        fine.write_face(
            zf, "plate", "", PlateSpec.from_dict(manifest["spec"]), Progress()
        )
        fine.finish(zf)
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
                "  fine.gds              — FAB-GRADE MASK: both gold layers at their\n"
                "                          true optical periods (front 10/0, back 20/0,\n"
                "                          plate outline 1/0)\n"
                "  FINE_MASKS.json       — fine.gds QA: realized periods + DRC report\n"
                "  front.svg / front.png — PREVIEW of the viewer-facing gold (incl. frame)\n"
                "  back.svg  / back.png  — PREVIEW of the far-side gold\n"
                "  manifest.json         — full plate spec + generation parameters\n"
                "  thumbnail.png         — composite preview\n\n"
                f"{_FINE_VS_PREVIEW_TEXT}\n"
                "Mask orientation & scale:\n"
                "  Every file is drawn as seen from the plate's FRONT side (pattern\n"
                "  toward the viewer, +x right, +y up). The back layer is the\n"
                "  far-surface gold PROJECTED through the glass onto that same view,\n"
                "  so front and back register 1:1 without any flip. If your writer\n"
                "  patterns the back surface with the plate flipped about its vertical\n"
                "  edge, mirror the back layer in X first — it is NOT pre-mirrored.\n"
                "  SVG width/height are true millimeters; user units (viewBox and\n"
                "  every path coordinate) are micrometers with the origin at the\n"
                "  plate center. fine.gds is micrometers at a 1 nm database unit,\n"
                "  same origin.\n"
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

Orientation: every plate mounts with its patterned FRONT surface facing
OUT of the box. The masks are drawn as viewed from outside (see
README.txt for the back-layer mirroring convention).

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
The lid should swing up and back over the hinge freely to ~110 degrees
(max travel 120) without striking the back wall. Apply patina/finish
("{foil.finish}") last.
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


class _ExportFace(NamedTuple):
    """One face of a box export: which plate dir to zip and how it was found.

    ``spec`` is ``None`` only for a legacy face entry that carries no plate
    spec anywhere — its files still ship, but it gets no fine.gds.
    ``rebuilt`` holds a fresh plate manifest when recomposing landed under a
    DIFFERENT id than box.json recorded, so the archive's box.json can be
    corrected to point at the plate actually inside it.
    """

    face_id: str
    plate_id: str
    spec: PlateSpec | None
    rebuilt: dict[str, Any] | None


def _face_plate_spec(
    face_id: str, face: dict[str, Any], box_faces: dict[str, PlateSpec]
) -> PlateSpec | None:
    """The PlateSpec that regenerates one face, or ``None`` if unrecoverable.

    Prefers the face's own recorded spec — its hash IS the plate id, so a
    rebuild from it lands back in the same cache dir. Falls back to the
    box-level face spec (already stamped through ``normalize_face_dims``) for
    pre-v2 or hand-edited manifests whose face entries carry no spec block.
    """
    spec_data = face.get("spec")
    if isinstance(spec_data, dict):
        try:
            return PlateSpec.from_dict(spec_data)
        except (KeyError, TypeError, ValueError) as exc:  # hand-edited manifest
            _log.warning(
                "face %s spec unusable (%s: %s) — falling back to the box spec",
                face_id, type(exc).__name__, exc,
            )
    return box_faces.get(face_id)


def _rebuild_face_plate(
    box_id: str, face_id: str, spec: PlateSpec, progress: Progress
) -> dict[str, Any]:
    """Recompose one wiped face plate into the cache, SEQUENTIALLY.

    Serialized on the export slot: plate composition is heavy compute and two at
    once bugchecks this host (CLAUDE.md). The claim is reentrant, so a six-face
    rebuild under an endpoint that already holds the slot is still six serial
    composes, never a fan-out.
    """
    _log.info(
        "box %s face %s: plate cache miss — rebuilding slug=%s",
        box_id, face_id, spec.pattern_slug,
    )
    progress.step(
        "rebuild plate",
        face=face_id,
        detail=f"plate cache miss — recomposing {spec.pattern_slug}",
    )
    with _export_slot():
        try:
            return materialize_plate(spec)
        except Exception as exc:  # noqa: BLE001 — report which face, not a bare 500
            _log.exception("plate rebuild failed box=%s face=%s", box_id, face_id)
            raise HTTPException(
                500,
                f"Box {box_id} face '{face_id}': its plate is not in the cache and "
                f"recomposing it from the saved spec failed "
                f"({type(exc).__name__}: {exc}).",
            ) from exc


def _resolve_box_faces(
    box_id: str, box_manifest: dict[str, Any], progress: Progress | None = None
) -> list[_ExportFace]:
    """Locate every face's plate dir, rebuilding any the cache no longer holds.

    ``backend/data/`` is a disposable cache and 'everything regenerates lazily'
    (CLAUDE.md), so a wiped ``data/plates`` must not break fab export — the one
    artifact that has to keep working, since it is what goes to the mask shop.
    The saved box.json carries a complete spec per face and ``materialize_plate``
    is idempotent, so we recompose the missing ones here (one at a time, see
    ``_rebuild_face_plate``) instead of 404ing.

    Face entries too broken to act on raise 409 with the regenerate instruction
    rather than a bare ``KeyError`` 500.
    """
    progress = progress if progress is not None else Progress()
    faces = box_manifest.get("faces")
    if not isinstance(faces, dict) or not faces:
        raise HTTPException(
            409,
            f"Box {box_id} has no face plates to export — regenerate it "
            "(POST /boxes/generate) and export again.",
        )
    try:
        box_spec = BoxSpec.from_dict(box_manifest.get("spec") or {})
        box_spec.normalize_face_dims()
    except Exception as exc:  # noqa: BLE001 — corrupt/hand-edited spec block
        raise HTTPException(
            409,
            f"Box {box_id} has an unreadable spec block "
            f"({type(exc).__name__}: {exc}) — regenerate the box and export again.",
        ) from exc

    resolved: list[_ExportFace] = []
    for face_id, face in faces.items():
        if not isinstance(face, dict):
            raise HTTPException(
                409,
                f"Box {box_id} face '{face_id}' is not a plate manifest — "
                "regenerate the box and export again.",
            )
        plate_id = face.get("id") or None  # treat "" like absent
        if plate_id is not None and not _PLATE_ID_RE.fullmatch(str(plate_id)):
            raise HTTPException(
                409,
                f"Box {box_id} face '{face_id}' records an illegal plate id "
                f"{plate_id!r} — regenerate the box and export again.",
            )
        spec = _face_plate_spec(face_id, face, box_spec.faces)
        rebuilt: dict[str, Any] | None = None
        cached = plate_id is not None and (PLATES_ROOT / str(plate_id) / "manifest.json").exists()
        if not cached:
            if spec is None:
                raise HTTPException(
                    409,
                    f"Box {box_id} face '{face_id}' cannot be exported: plate "
                    f"{plate_id or '(no id recorded)'} is not in the cache and the box "
                    "manifest carries no spec to rebuild it. Regenerate the box "
                    "(POST /boxes/generate) and export again.",
                )
            rebuilt = _rebuild_face_plate(box_id, face_id, spec, progress)
            new_id = str(rebuilt.get("id") or "")
            if not _PLATE_ID_RE.fullmatch(new_id):
                raise HTTPException(
                    500, f"Rebuilt plate for face '{face_id}' has no usable id."
                )
            if new_id == plate_id:
                rebuilt = None  # same slot — the saved box.json is still accurate
            else:
                _log.warning(
                    "box %s face %s rebuilt under id %s (box.json recorded %r)",
                    box_id, face_id, new_id, plate_id,
                )
            plate_id = new_id
        resolved.append(_ExportFace(face_id, str(plate_id), spec, rebuilt))
    return resolved


class PreparedBox(NamedTuple):
    """A box that has passed every cheap check and is ready to build.

    Split out of the endpoint so the manifest checks (404/409) happen BEFORE
    anything heavy: a bad box id must answer 404 while another export is in
    flight, and ``fab/start`` must not spawn a worker that is doomed to fail.
    Reading a manifest and recomputing the assembly block is pure math — no
    plate composition, no bake, no slot.
    """

    box_id: str
    manifest: dict[str, Any]
    assembly: dict[str, Any]
    faces_total: int


def prepare_box_export(box_id: str) -> PreparedBox:
    """Validate a box for export, or raise the HTTP error explaining why not."""
    box_manifest = get_box(box_id)
    if box_manifest is None:
        raise HTTPException(404, f"Unknown box: {box_id}")
    dims = box_manifest.get("dimensions_um")
    if not isinstance(dims, dict) or not {"width", "height", "depth"} <= dims.keys():
        raise HTTPException(
            409,
            f"Box {box_id} predates the v2 manifest format (no dimensions_um) — "
            "regenerate it (POST /boxes/generate) and export again.",
        )
    try:
        assembly = _box_assembly_block(box_manifest)
    except (KeyError, TypeError, ValueError) as exc:
        raise HTTPException(
            409,
            f"Box {box_id} carries assembly geometry the cut list cannot be derived "
            f"from ({type(exc).__name__}: {exc}) — regenerate the box and export again.",
        ) from exc
    faces = box_manifest.get("faces")
    faces_total = len(faces) if isinstance(faces, dict) else 0
    return PreparedBox(box_id, box_manifest, assembly, faces_total)


def build_box_fab_zip(
    prepared: PreparedBox,
    zf: zipfile.ZipFile,
    *,
    progress: Progress | None = None,
) -> None:
    """Compose the full box bundle into ``zf``: six plate folders + fab masks +
    cut list + assembly steps.

    THE build body — the subprocess worker (``app.export_job``) and the legacy
    synchronous endpoint both call this, so the two paths cannot drift and the
    archive is byte-identical whichever produced it.

    Rebuilds any face whose plate cache dir is gone before zipping (see
    ``_resolve_box_faces``) — ``backend/data/`` is disposable, so ``just clean``
    must not make a saved box un-exportable.

    Everything here is heavy (plate rebuilds, the lazy SVG bakes inside
    ``_plate_files_to_zip``, the per-face fine masks), so the whole body runs
    inside the host-wide export slot; a second in-process export gets 429 and a
    job start gets 429 while this runs. Faces are walked ONE AT A TIME, plate
    files then fab mask, so ``progress`` reports a real per-face position
    instead of a spinner.
    """
    progress = progress if progress is not None else Progress()
    box_id = prepared.box_id
    box_manifest = prepared.manifest
    assembly = prepared.assembly
    dims = box_manifest["dimensions_um"]
    progress.faces_total = prepared.faces_total

    with _export_slot():
        progress.step("resolve faces", detail=f"{prepared.faces_total} face(s)")
        faces = _resolve_box_faces(box_id, box_manifest, progress)
        progress.faces_total = len(faces)
        # Per-face folder per plate so the engraver can register and cut each
        # independently. Box manifest lives at the root.
        fine = _FineMaskRun()
        for f in faces:
            progress.step("svg bake", face=f.face_id, detail="preview SVG pair")
            _plate_files_to_zip(zf, f.plate_id, prefix=f"{f.face_id}/")
            # TRUE-pitch fab mask for this face (see _FineMaskRun).
            if f.spec is not None:
                fine.write_face(zf, f.face_id, f"{f.face_id}/", f.spec, progress)
            progress.face_finished(f.face_id)
        fine.finish(zf)
        progress.step("zip", detail="box.json + cut list + assembly steps")
        # Strip frame_scene from each face manifest in the box snapshot too
        # (v2 manifests are already lean; this also covers older saves).
        bm_lean = json.loads(json.dumps(box_manifest))  # cheap deep copy
        for f in faces:
            if f.rebuilt is not None:
                # Rebuild landed in a different slot: point box.json at the
                # plate this archive actually ships.
                bm_lean["faces"][f.face_id] = f.rebuilt
        for face in bm_lean.get("faces", {}).values():
            rd = face.get("recipe_data") or {}
            rd.pop("frame_scene", None)
            face["recipe_data"] = rd
        zf.writestr("box.json", json.dumps(bm_lean, indent=2))
        zf.writestr("CUTLIST.csv", _cutlist_csv(assembly))
        zf.writestr("ASSEMBLY.md", _assembly_md(box_manifest, assembly))
        zf.writestr(
            "README.txt",
            (
                f"Optics Pattern Studio — ring box {box_id}\n"
                f"Outer dimensions: {dims['width']} × {dims['height']} × {dims['depth']} μm\n\n"
                "Each subdirectory ({front,back,top,bottom,left,right}/) is one plate:\n"
                "  <face>/fine.gds  — FAB-GRADE MASK for that face (front gold 10/0,\n"
                "                     back gold 20/0, plate outline 1/0)\n"
                "  <face>/front.svg, back.svg, *.png — PREVIEWS (see below)\n"
                "  <face>/manifest.json — that plate's spec + generation parameters\n"
                "At the root:\n"
                "  FINE_MASKS.json — per-face fine.gds QA: realized periods + DRC\n"
                "  box.json    — assembly geometry + per-face plate ids\n"
                "  CUTLIST.csv — glass cut list (mm + μm)\n"
                "  ASSEMBLY.md — numbered copper-foil build steps\n\n"
                f"{_FINE_VS_PREVIEW_TEXT}\n"
                "Mask orientation & scale (applies to every face folder):\n"
                "  Plates mount with the patterned FRONT surface facing OUT of the\n"
                "  box; every file is drawn as viewed from outside (+x right, +y up).\n"
                "  The back layer is the far-surface gold projected through the glass\n"
                "  onto the same view — it registers 1:1 with the front layer and is\n"
                "  NOT pre-mirrored. Mirror it in X if your writer patterns the back\n"
                "  surface with the plate flipped about its vertical edge.\n"
                "  SVG width/height are true millimeters; viewBox user units are\n"
                "  micrometers, origin at the plate center. fine.gds is micrometers\n"
                "  at a 1 nm database unit, same origin.\n"
            ),
        )


def build_box_fab_archive(
    box_id: str, out_path: Path, *, progress: Progress | None = None
) -> Path:
    """Build one box's fab archive to a FILE. The subprocess worker's entry.

    Published by rename like every other cache artifact (CLAUDE.md): a poller
    that sees ``fab.zip`` sees a complete zip, never the tail of one still being
    deflated.
    """
    out_path = Path(out_path)
    prepared = prepare_box_export(box_id)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zf:
        build_box_fab_zip(prepared, zf, progress=progress)
    os.replace(tmp, out_path)
    return out_path


@router.get("/box/{box_id}/fab.zip")
def box_fab_zip(box_id: str) -> StreamingResponse:
    """The full box bundle, built synchronously in the request thread (LEGACY).

    Kept for existing tools and tests. It is the same ``build_box_fab_zip``
    body the subprocess worker runs, and it still holds the host-wide export
    slot for the whole build — which is exactly why it is legacy: a cold
    six-face export takes minutes, during which this connection is silent and
    the GIL-heavy stretches (klayout DRC) starve the rest of the app. New
    clients should POST ``/export/box/{box_id}/fab/start`` and poll.
    """
    prepared = prepare_box_export(box_id)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        build_box_fab_zip(prepared, zf)
    buf.seek(0)
    return StreamingResponse(
        buf,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="box-{box_id}.zip"'},
    )


# --- the subprocess job path --------------------------------------------------


@router.post("/box/{box_id}/fab/start", status_code=202)
def box_fab_start(box_id: str) -> dict[str, Any]:
    """Start the box export in a SUBPROCESS; answer 202 with a job id.

    Validates the box first, so an unknown or unexportable box still answers
    404/409 instead of spawning a worker that would only fail. Refuses with 429
    while any other export is building — one export at a time across the host
    (CLAUDE.md), whether the other one is a job or the legacy synchronous route.

    The parent does NOT hold the heavy slot for the job's lifetime: the live
    subprocess IS the claim, so this process stays free to serve previews and
    the poll route while minutes of DRC run next door.
    """
    prepared = prepare_box_export(box_id)
    try:
        job = export_job.start_job(box_id, faces_total=prepared.faces_total)
    except ExportBusy as exc:
        raise HTTPException(429, str(exc), headers={"Retry-After": "10"}) from exc
    except RuntimeError as exc:  # worker could not be spawned at all
        _log.exception("export job spawn failed box=%s", box_id)
        raise HTTPException(500, str(exc)) from exc
    status = export_job.get_status(job.job_id) or {}
    return {
        "job_id": job.job_id,
        "box_id": box_id,
        "status": status.get("status", "running"),
        "progress": status.get("progress"),
        "poll_url": f"/export/jobs/{job.job_id}",
        "download_url": f"/export/jobs/{job.job_id}/fab.zip",
    }


@router.get("/jobs/{job_id}")
def export_job_status(job_id: str) -> dict[str, Any]:
    """Poll one export job: ``{status: running|done|failed, progress, error}``.

    Cheap and side-effect free — poll it once a second while the UI shows the
    phase. ``status`` merges the worker's progress file with the subprocess exit
    code, so a worker that died reports ``failed`` here, never ``running``
    forever (see ``export_job.get_status``).
    """
    status = export_job.get_status(job_id)
    if status is None:
        raise HTTPException(
            404,
            f"Unknown export job: {job_id}. Job ids live in this server process "
            "only — start a new export (POST /export/box/{box_id}/fab/start).",
        )
    return status


@router.get("/jobs/{job_id}/fab.zip")
def export_job_zip(job_id: str) -> FileResponse:
    """The finished archive for a job. 409 while it is still building."""
    status = export_job.get_status(job_id)
    if status is None:
        raise HTTPException(404, f"Unknown export job: {job_id}")
    if status["status"] != "done":
        if status.get("error"):
            detail = f"Export job {job_id} failed: {status['error']}"
        else:
            detail = (
                f"Export job {job_id} is still {status['status']} — poll "
                f"GET /export/jobs/{job_id} until its status is 'done'."
            )
        raise HTTPException(409, detail)
    path = export_job.result_path(job_id)
    if path is None:
        raise HTTPException(
            404,
            f"Export job {job_id} reported done but its archive is gone — "
            "data/export_jobs is a disposable cache; start the export again.",
        )
    return FileResponse(
        path,
        media_type="application/zip",
        filename=f"box-{status['box_id']}.zip",
    )
