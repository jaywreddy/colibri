# CLAUDE.md — operator notes for Ring Box Studio

## Machine constraint (non-negotiable)
13.7 GB host that has kernel-bugchecked under heavy parallel compute.
- NEVER run two long compute processes at once (no parallel pytest, no test run
  alongside a dev server build, no concurrent pattern generation).
- Run backend tests in the proven justfile chunk order; never more than 2
  HEAVY test files per invocation, one invocation at a time. Budget: ~1 GB
  peak, seconds-to-a-minute per chunk.
- Never generate patterns beyond their default extents. The lattice-budget
  guard (`app/patterns/_helpers.py`, 400k-cell cap) is the last line of
  defense, not permission to push it.

## Commands
- `just dev` — backend :8765 + frontend :5173 together (don't start servers
  during automated sessions unless asked).
- `just test-backend` — backend suite in the proven safe 10-chunk order.
  Manual equivalent, from `backend/`, one at a time:
  ```
  uv run --extra dev pytest tests/test_assembly.py tests/test_plates_and_boxes.py -q
  uv run --extra dev pytest tests/test_motifs.py tests/test_rasterize.py tests/test_export_svg.py tests/test_theme_metadata.py tests/test_variant_hash.py -q
  uv run --extra dev pytest tests/test_api_patterns.py tests/test_frames.py -q
  uv run --extra dev pytest tests/test_patterns_roundtrip.py tests/test_sim_numerics.py -q
  uv run --extra dev pytest tests/test_sim2d.py tests/test_pattern_types.py tests/test_showcase_patterns.py tests/test_bitmap_halftone.py -q
  uv run --extra dev pytest tests/test_param_validation.py tests/test_sim_bounds.py tests/test_grating_phase.py tests/test_barrier_registration.py tests/test_drc_tiling.py tests/test_diffraction.py tests/test_readability.py -q
  uv run --extra dev pytest tests/test_collage.py tests/test_api_collage.py tests/test_shimmer_moire.py -q
  uv run --extra dev pytest tests/test_imageprep.py tests/test_colourzone.py -q
  uv run --extra dev pytest tests/test_screenrects.py tests/test_colourplan.py tests/test_witness.py tests/test_ply_cuts.py tests/test_export_svg_rects.py -q
  uv run --extra dev pytest tests/test_cache_integrity.py -q
  ```
  (The witness chunk is light — every cell in it is built at a couple of mm,
  never at the shipping 30 mm; the real plate is a 30 s standalone build.
  The collage chunk is light — it sweeps warm variant rasters, ~6 s.
  Chunk 2 is six files and chunk 5 is four, but all of them are light/fast —
  chunk 5's files are synthetic/small-extent, ~2 s total; chunk 6 is validation/
  registration tests, ~6 s. The heavy files — plates_and_boxes, api_patterns,
  frames, patterns_roundtrip, cache_integrity (~80 s, does a real fab-zip
  rebuild) — never share a chunk with more than one other file.)
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
  time-invariant, litho-mask-driven, and GEOMETRIC — the back gold layer
  renders on a real inner plane at the paraxial T/n air gap below the outer
  plane, cross-layer illusions emerge from perspective across that gap, and
  every composed plate binds render_recipe `foliage_moire`. No
  procedural/time-animated shader fakes. The @effects suite
  (`frontend/tests/e2e/effectsPhysical.spec.ts`) enforces this on the live
  WebGL buffer by scaling the actual plane gap
  (`effectsHelpers.ts::scaleBackPlaneGap`) — keep it green when touching
  shaders or BoxScene.
- **the box is single-ply (2026-09-15):** every production face is ONE written
  ply with single-layer diffraction effects (`region_art` centrepieces,
  per-family leaf gratings, photo colour zones); no face carries a carrier, a
  beat or a switch. The two-ply moiré and two-way switch code paths are kept
  only as hidden exemplars. Design and face selection live in code
  (`boxes.default_box_spec`); the frontend is a visualizer.
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
- **cache versions:** cache keys hash user spec/params ONLY, so a version bump
  is the only thing that invalidates a warm `backend/data/` after a code or
  constant change. Three markers, all checked on the cache-hit path:
  `plates.PLATE_SVG_VERSION` (fab SVG geometry), `plates.PLATE_COMPOSE_VERSION`
  (`_raster_compose_plate`, `_paste_centerpiece`, `_centerpiece_masks`, the mask
  level palette, `_carrier_recipe_data`), and `service.PATTERN_GEN_VERSION`
  (pattern generators, `patterns/base.py`, the rasterizer, manifest shape).
  Bump the ones your change touches in the same commit.
- **Units:** micrometers (um) everywhere in code, specs, manifests, and the API.
  mm appears only in UI display and in derived `_mm` cut-list fields.
- **Caches:** `backend/data/` is a disposable on-disk cache — pattern variants
  (`data/<slug>/<variant>/`), composed plates (`data/plates/<hash>/`), boxes
  (`data/boxes/<id>/`). Safe to delete; everything regenerates lazily. Every
  slot is published payload-files-first, then the manifest renamed into place
  (`service.write_json_atomic` / `write_text_atomic` / `save_png_atomic`), and
  read back through `service.read_json_cache` so a truncated manifest is a MISS,
  not a permanently poisoned slot. Generates serialize per cache id on
  `service.cache_lock` (variant hash / plate hash / box id) — that lock is also
  what keeps two identical in-flight requests from breaking the
  single-heavy-compute rule. New cache writers must use all of it.
- **Geometry perf:** no `unary_union` / whole-geometry GEOS booleans on hot
  paths — concatenate parts and clip with `crop_parts`. The why lives in
  docstrings: `app/plates.py::_concat_polygons`,
  `app/patterns/_helpers.py::crop_parts`,
  `app/patterns/frames/shapely_pen.py::ShapelyPen.finish`.
