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
- `just test-backend` — backend suite in the proven safe 8-chunk order.
  Manual equivalent, from `backend/`, one at a time:
  ```
  uv run --extra dev pytest tests/test_assembly.py tests/test_plates_and_boxes.py -q
  uv run --extra dev pytest tests/test_motifs.py tests/test_rasterize.py tests/test_export_svg.py tests/test_theme_metadata.py tests/test_variant_hash.py tests/test_face_kind.py tests/test_production_constants.py tests/test_cache_fingerprint.py -q
  uv run --extra dev pytest tests/test_api_patterns.py tests/test_frames.py -q
  uv run --extra dev pytest tests/test_patterns_roundtrip.py tests/test_showcase_patterns.py -q
  uv run --extra dev pytest tests/test_param_validation.py tests/test_grating_phase.py tests/test_barrier_registration.py tests/test_drc_tiling.py tests/test_diffraction.py tests/test_shimmer_moire.py -q
  uv run --extra dev pytest tests/test_imageprep.py tests/test_colourzone.py -q
  uv run --extra dev pytest tests/test_screenrects.py tests/test_colourplan.py tests/test_witness.py tests/test_optics_math.py tests/test_ply_cuts.py tests/test_export_svg_rects.py -q
  uv run --extra dev pytest tests/test_cache_integrity.py -q
  ```
  (The heavy files — plates_and_boxes, api_patterns, frames, patterns_roundtrip,
  cache_integrity (~80 s, does a real fab-zip rebuild) — never share a chunk
  with more than one other file. Everything else is light: chunk 2 and chunk 7
  are six files each but run in seconds, chunk 5 is validation/registration
  at small extents, and the witness chunk builds every cell at a couple of mm,
  never at the shipping 30 mm — the real plate is `just plate`, an 8 min
  standalone build.)
- `just test-unit` / `just test-e2e` — frontend vitest / Playwright. E2E starts
  both servers itself; treat it as a heavy process.
- `just test-effects` — physical-honesty pixel-metric suite for the 3D
  renderer (@effects specs; part of test-e2e too). Heavy process, run alone.
  `just test-effects-verify` adds Claude vision grading of the frame dumps.
- `just plate` — write the 5" production plate (GDS + OASIS + map + DICING.md).
  HEAVY (~8 min); run alone. This is the fab deliverable and the regression
  gate: rebuild it and compare per layer against the previous mask.
- `just seed` — pre-warm all pattern default variants; `just clean` — wipe
  `backend/data/`.

## Rules
- **assembly contract:** `backend/app/assembly.py` and `frontend/src/assembly.ts`
  implement identical formulas (cut dims, foil keep-out, seams, hinge,
  validation). Any change to one must land in the other in the same change,
  AND regenerate the shared golden fixture
  (`uv run --directory backend python ../tools/dev/gen_assembly_golden.py`) —
  both test suites pin their implementation to it.
- **production constants:** `backend/app/production.py` is the ONE holder of the
  process (litho floor, finish radius, DBU, polarity, layer map), stock (ply,
  index, blank, street) and box (dims, tape, motif scale, band, art rim, ID-tick
  offset) constants. Never re-type one of those numbers anywhere — import it.
  `witness_geom` derives the plate's optics from it; `frontend/src/production.ts`
  is GENERATED from it (`uv run --directory backend python
  ../tools/dev/gen_production_constants.py`) and `api.ts::defaultBoxSpec` reads
  the generated file, so regenerate in the same change
  (`tests/test_production_constants.py` fails if it is stale). Six of the
  constants are PINNED to the 2026-09-15 plate — changing one is a mask change,
  and the witness rebuild is the gate.
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
  switches peak at a back shift of ±p/4 and alias every full period. No
  production face carries one; `globe-duo-phase` is the hidden exemplar that
  keeps the construction built, and `app/sim2d.py::switch_metrics` measures it
  on the real generated geometry (`tests/test_barrier_registration.py`).
- **fab SVG = preview PNG:** `plates/svg.py::ensure_plate_svg` must compose the
  same aperture-scaled geometry as `plates/compose.py::_raster_compose_plate`.
  Bump `PLATE_SVG_VERSION` whenever SVG compose geometry changes so stale
  cached SVGs regenerate.
- **cache fingerprints:** cache keys hash user spec/params ONLY, so a marker on
  the hit path is the only thing that invalidates a warm `backend/data/` after a
  code, constant or asset change. The three markers are COMPUTED, not bumped
  (`app/cache_fingerprint.py`): each is a digest of its cache's module closure
  plus the bytes of the prepared photos —
  `plates.svg.PLATE_SVG_FINGERPRINT` (fab SVG geometry),
  `plates.compose.PLATE_COMPOSE_FINGERPRINT` (`_raster_compose_plate`,
  `_paste_centerpiece`, `_centerpiece_masks`, the mask level palette,
  `recipe._carrier_recipe_data`, the literal rasters) and
  `service.PATTERN_GEN_FINGERPRINT` (pattern generators, `patterns/base.py`,
  the rasterizer, manifest shape). The digest is over a normalized `ast.dump`:
  comments, blank lines and REFLOWED docstrings are free, everything else —
  a constant, an expression, a reworded docstring — moves the marker and
  regenerates that cache once. There is nothing to remember and nothing to
  bump; when you ADD a module that writes cached bytes, add it to the closure.
  `CACHE_EPOCH` is the manual lever for what no closure can see (a Pillow
  resampling change, a cache-poisoning bug) — bump it and say why.
- **Units:** micrometers (um) everywhere in code, specs, manifests, and the API.
  mm appears only in UI display and in derived `_mm` cut-list fields.
- **the plates package:** `app/plates/` is six modules in dependency order —
  `spec` (the specs, FaceKind, plate_hash, PLATES_ROOT), `recipe` (mask level
  palette, grating constants, `_carrier_recipe_data`), `photo` (the halftone
  centrepiece), `compose` (masks, paste, `_raster_compose_plate`,
  `materialize_plate`), `literal` (the fabricated-chrome rasters) and `svg`
  (the fab SVG). `__init__.py` is a pure facade re-exporting all of them, so
  `from app import plates as P` still resolves every name; import the submodule
  directly in new code. `PLATES_ROOT` is read through `spec` (never bound by
  value) because the test fixtures repoint it.
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
  docstrings: `app/plates/compose.py::_concat_polygons`,
  `app/patterns/_helpers.py::crop_parts`,
  `app/patterns/frames/shapely_pen.py::ShapelyPen.finish`.
