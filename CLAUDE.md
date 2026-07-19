# CLAUDE.md — operator notes for Ring Box Studio

## Machine constraint (non-negotiable)
13.7 GB host that has kernel-bugchecked under heavy parallel compute.
- NEVER run two long compute processes at once (no parallel pytest, no test run
  alongside a dev server build, no concurrent pattern generation).
- Run backend tests in small chunks: max 2 test files per pytest invocation,
  one invocation at a time. Budget: ~1 GB peak, seconds-to-a-minute per chunk.
- Never generate patterns beyond their default extents. The lattice-budget
  guard (`app/patterns/_helpers.py`, 400k-cell cap) is the last line of
  defense, not permission to push it.

## Commands
- `just dev` — backend :8765 + frontend :5173 together (don't start servers
  during automated sessions unless asked).
- `just test-backend` — backend suite in the proven safe 5-chunk order.
  Manual equivalent, from `backend/`, one at a time:
  ```
  uv run --extra dev pytest tests/test_assembly.py tests/test_plates_and_boxes.py -q
  uv run --extra dev pytest tests/test_motifs.py tests/test_rasterize.py tests/test_export_svg.py tests/test_export_gds_stub.py tests/test_theme_metadata.py tests/test_variant_hash.py -q
  uv run --extra dev pytest tests/test_api_patterns.py tests/test_frames.py -q
  uv run --extra dev pytest tests/test_patterns_roundtrip.py tests/test_sim_numerics.py -q
  uv run --extra dev pytest tests/test_sim2d.py tests/test_showcase_patterns.py tests/test_bitmap_halftone.py -q
  ```
  (Chunk 2 is six files but they are all light/fast; the heavy files —
  plates_and_boxes, api_patterns, frames, patterns_roundtrip — never share a
  chunk with more than one other file.)
- `just test-unit` / `just test-e2e` — frontend vitest / Playwright. E2E starts
  both servers itself; treat it as a heavy process.
- `just test-effects` — physical-honesty pixel-metric suite for the 3D
  renderer (@effects specs; part of test-e2e too). Heavy process, run alone.
  `just test-effects-verify` adds Claude vision grading of the frame dumps.
- `just seed` — pre-warm all pattern default variants; `just clean` — wipe
  `backend/data/`.

## Rules
- **assembly contract:** `backend/app/assembly.py` and `frontend/src/assembly.ts`
  implement identical formulas (cut dims, foil keep-out, seams, hinge,
  validation). Any change to one must land in the other in the same change,
  AND regenerate the shared golden fixture
  (`uv run --directory backend python ../tools/dev/gen_assembly_golden.py`) —
  both test suites pin their implementation to it.
- **renderer honesty:** every optical effect must be view-dependent,
  time-invariant, litho-mask-driven, and parameterized by the substrate
  (Snell parallax). No procedural/time-animated shader fakes. The @effects
  suite (`frontend/tests/e2e/effectsPhysical.spec.ts`) enforces this on the
  live WebGL buffer — keep it green when touching shaders or BoxScene.
- **image-switch patterns are parallax barriers:** BOTH images interlaced in
  the BACK layer, slit/phase mask in FRONT. A front-layer image can never
  vanish under parallax (the front mask does not move with tilt) — the old
  two-image "phase overlay" construction is a banned anti-pattern. Barrier
  switches peak at a back shift of ±p/4 and alias every full period; carrier
  reveals peak at ±p/2. Type metrics live in `app/sim2d.py`
  (`switch_metrics`/`reveal_metrics`/`fringe_metrics`, pinned by
  `tests/test_pattern_types.py`).
- **fab SVG = preview PNG:** `plates.py::ensure_plate_svg` must compose the
  same aperture-scaled geometry as `_raster_compose_plate`. Bump
  `PLATE_SVG_VERSION` whenever SVG compose geometry changes so stale cached
  SVGs regenerate.
- **Units:** micrometers (um) everywhere in code, specs, manifests, and the API.
  mm appears only in UI display and in derived `_mm` cut-list fields.
- **Caches:** `backend/data/` is a disposable on-disk cache — pattern variants
  (`data/<slug>/<variant>/`), composed plates (`data/plates/<hash>/`), boxes
  (`data/boxes/<id>/`). Safe to delete; everything regenerates lazily.
- **Geometry perf:** no `unary_union` / whole-geometry GEOS booleans on hot
  paths — concatenate parts and clip with `crop_parts`. The why lives in
  docstrings: `app/plates.py::_concat_polygons`,
  `app/patterns/_helpers.py::crop_parts`,
  `app/patterns/frames/shapely_pen.py::ShapelyPen.finish`.
