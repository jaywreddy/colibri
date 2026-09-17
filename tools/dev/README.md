# tools/dev

Ad-hoc developer utilities. Not part of any automated test or build. Run them
one at a time — the machine constraint in `CLAUDE.md` applies to these too.

## Surviving scripts (2026-09-15 tools+docs cleanup)

Every remaining tool's imports were checked against the modules deleted in
this pass (`export_wafer`, `export_blank`, `export_gds`, `export_witness_gf`,
`app.sim*`, `sim2d`, `readability`, `collage`, the retired `witness_moire`
cell-builders, `capybara`). One dependency was load-bearing and has been
vendored rather than dropped — see `validate_dies.py` below.

| Script | What it does |
|---|---|
| `add_dice_marks.py` | Stamp (or refresh) the saw-lane marks onto an already-written plate GDS/OASIS without re-running the 8-minute build. `uv run --directory backend python ../tools/dev/add_dice_marks.py [stem]` |
| `build_photo_page.py` | Build the side-photo review page (crop, screen preview, the four candidate colour treatments) for the six side-plate photographs. `uv run --directory backend python ../tools/dev/build_photo_page.py PHOTOS_DIR OUT_HTML [NOTES_JSON...]` |
| `build_witness_page.py` | Assemble the historical whitepaper page from `docs/archived/witness-physics-plan.md` plus the figures the surviving renderers still produce. `uv run --directory backend python ../tools/dev/build_witness_page.py OUTDIR` |
| `gen_assembly_golden.py` | Regenerate `tools/fixtures/assembly_golden.json` from the backend assembly math — the backend↔frontend contract fixture. `uv run --directory backend python ../tools/dev/gen_assembly_golden.py` |
| `paint_glasses_mask.py` | Author the spectacle-frame mask for the reference portrait from the rule's own detected rims. `uv run --directory backend python ../tools/dev/paint_glasses_mask.py` |
| `probe_gds_sanity.py` | GDS sanity: per-die unclipped clear DRC (width/space at the 2 um floor), zero-area shapes, vertex counts, data extent. Supersedes the deleted `probe_clear_drc.py` (same clear-DRC check, less complete — no zero-area/vertex/whole-plate stats). `uv run --directory backend python ../tools/dev/probe_gds_sanity.py PLATE.gds` |
| `profile_die.py` | cProfile one face's `witness_dies.build_face_die`, then time a bare GDS write of its polygons. Kept (not on the deletion list) as the perf-triage tool for the heaviest builds on a memory-constrained host — cheap to keep (52 lines, only imports `witness_dies`), and exactly the kind of check CLAUDE.md's machine-constraint section wants available before a change is suspected of a regression. `uv run --directory backend python ../tools/dev/profile_die.py left [garden]` |
| `render_box_faces.py` | Compose the production box and lay its six faces out from the literal coverage rasters. Its `1 - (1-A)(1-B)` stack formula degrades correctly on today's all-single-ply box (B is zeroed whenever `recipe_data.single_ply` or `.blank`, so the composite is just A). `uv run --directory backend python ../tools/dev/render_box_faces.py OUTDIR` |
| `render_colour_plans.py` | Re-render the AUTHORED colour plans that ship beside the prepared photographs, from the same `photo.photo_coverage` the SVG/GDS bakes use. This is the live review tool for what actually ships — see `docs/plan.md` §1. `uv run --directory backend python ../tools/dev/render_colour_plans.py OUTDIR [IMAGE ...]` |
| `render_photo_colour.py` | Explore colour TREATMENTS (plain / hue / hue-eq / auto-zones) for CANDIDATE side-plate photographs, before a plan is authored — different inputs (raw crop notes, not shipped images) and a different appearance model (borrowed from `render_witness_preview.py`) than `render_colour_plans.py`. Kept separate rather than merged: they answer different questions (which treatment should this photo get vs. what does the authored plan actually do) and share only a few constants. `uv run --directory backend python ../tools/dev/render_photo_colour.py NOTES_JSON [NOTES_JSON...] OUTDIR` |
| `render_side_plate.py` | Render each candidate side photo as the finished single-ply plate: portrait in its art box, dissolving to bare glass inside the leaf garland, as the eye sees it under a lamp. `uv run --directory backend --with rembg --with onnxruntime python ../tools/dev/render_side_plate.py PHOTOS_DIR OUTDIR [FADE_DIR]` |
| `render_witness_figures.py` | Whitepaper figures that survive the single-ply pivot: the diffraction colour-ladder swatch (4.2), the H-WEDGE tone-linearisation figure (4.5, a surviving bench cell), and the generic cell-anatomy diagram (4.6). The two-ply figures (screen/carrier harmonic beat chart, P-SWAP barrier-comb plot, the parallax-stack and union-identity diagrams) were removed with the bond. `uv run --directory backend python ../tools/dev/render_witness_figures.py OUTDIR` |
| `render_witness_plate.py` | Render the written witness GDS to PNG (whole plate + die zooms) with klayout's headless LayoutView, straight from the file. `uv run --directory backend python ../tools/dev/render_witness_plate.py OUTDIR [GDS]` |
| `render_witness_preview.py` | Shared appearance-model helpers (`ETA0`/`ETA1`/`sheen`/`block_mean`/`block_mode`, imported by `render_photo_colour.py`) plus its own tilt-simulation previews. | `uv run --directory backend python ../tools/dev/render_witness_preview.py OUTDIR` |
| `validate_dies.py` | The in-silico production gates (`docs/plan.md` §4.A): A1 photo tone, A2 region gratings as written, A3 near-field fringe survival. Vendors `_propagate` (the angular-spectrum FFT sandwich) directly rather than importing it from `app.sim.angular_spectrum`, since the whole `app.sim` package was deleted with the holography-simulator retirement and this is the only remaining caller of that one function. `uv run --directory backend python ../tools/dev/validate_dies.py OUTDIR [photo|regions|nearfield|all]` |

`backend/tools/frame_band_probe.py` also survives (not on the deletion list):
a standalone probe for tuning the perimeter foliage frame band (density,
bloom, understory, border-vine, corner-fan knobs — the "frame dials with no
UI" from the cleanup review), independent of anything retired. `uv run python
tools/frame_band_probe.py <out_prefix> [seed]` (run with cwd `backend/`).

### Deleted this pass

`write_verdicts.py`, `shot_server.py`, `render_moire_preview.py` (two-ply
moiré ladders, driven by `witness_moire`/`export_witness.doe_cells` cells
already absent from the plate), `nearfield_ladder.py` (two-ply carrier/leaf
exploration, superseded by the A3 gate in `validate_dies.py`),
`gen_demo_bitmaps.py` (demo bitmaps for a pattern the box no longer ships),
`probe_clear_drc.py` (subsumed by `probe_gds_sanity.py`); and from
`backend/tools/`: `drc_selfcheck.py`, `leaf_probe.py`, `wreath_probe.py`,
`render_frame_preview.py`, `scan_sliver_probe.py` (capybara scanimation,
deleted with the artistic catalogue) and the tracked `frame_preview.png`.
None had a caller left, and no unique check in them was worth folding
elsewhere beyond what `probe_gds_sanity.py` and `validate_dies.py` already
cover.

## Headless inspection of the live app

The old `window.__debug` surface (and its cheat-sheet, `tools/preview_inspect.md`)
died with the pre-box studio UI. Current hooks, registered on every page load:

- `window.__store` — the zustand store (`frontend/src/main.tsx`); read/write
  state deterministically without fighting React-controlled inputs.
- `window.__studio` — three.js handles from `BoxScene`
  (`scene`, `camera`, `controls`, `renderer`, `faces`, `store`,
  `setLid(deg)`, `getLidDeg()`, `autoRotate`, `lidPivot`).

Render recipes (shader `uRecipe`): `0 stereo_lenticular`,
`1 moire_interactive`, `3 foliage_moire` (id 2, `phase_shift_overlay`, is
retired). Every composed box plate binds `foliage_moire`: the back gold layer
renders on a REAL inner plane at the paraxial T/n air gap, so substrate
checks manipulate the actual plane gap
(`frontend/tests/e2e/effectsHelpers.ts::scaleBackPlaneGap`) — the legacy
`uThicknessUm` uniform is dead on the foliage path and poking it proves
nothing.

For scripted screenshots use `frontend/scripts/shot.mjs` /
`moireshot.mjs`; for automated verification use `just test-visual` /
`just test-effects` (see the justfile). The old `shot_server.py` collector
(for when the MCP preview tab is hidden) is gone — `preview_screenshot`
timeouts are now rarer than they were, and `just test-visual` needs no
external collector either way.
