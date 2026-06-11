# Ring Box Studio

A single-screen designer for a stained-glass-style glass ring box: six fused-silica
plates carrying dual-layer gold-on-quartz moire patterns framed by generative gold
engraving (vines, Colombian flora), assembled Tiffany-style with copper-foil tape and
soldered seams, and hinged with a brass tube-and-rod hinge along the back top edge.
The browser preview renders the full box in 3D — glass slabs, pattern shaders, foil
strips, solder beads, hinge — with an animated opening lid.

**Design workflow:** size the box → tune the solder joints (foil tape, safety margin,
bead, finish) → configure the hinge → assign a pattern + frame to each of the six
faces → check the open-lid preview → **Export fab bundle**: a zip with per-face
SVG/PNG lithography masks, `CUTLIST.csv` (glass cut list), and `ASSEMBLY.md`
(numbered copper-foil build steps with real dimensions).

## Quickstart

Prereqs: [uv](https://docs.astral.sh/uv/), [pnpm](https://pnpm.io/),
[just](https://github.com/casey/just) (`winget install Casey.Just`).

```sh
cd frontend && pnpm install && cd ..   # first run only; uv syncs itself
just dev                               # backend :8765 + frontend -> http://localhost:5173
```

`just` with no arguments lists all targets.

## Architecture map

```
backend/   FastAPI (Python, uv). ALL geometry in micrometers (um).
  app/assembly.py        assembly math: cut list, foil keep-out, seams, hinge —
                         pure geometry/validation, no I/O
  app/boxes.py           BoxSpec (6 faces + glass/foil/hinge) -> box manifest
  app/plates.py          PlateSpec (central pattern + frame) -> raster compose,
                         lazy SVG (ensure_plate_svg), cache under data/plates/<hash>
  app/service.py         pattern materialize + disk cache under data/<slug>/<variant>
  app/patterns/          pattern registry, moire motifs, frame engine (frames/)
  app/api/               routers: /patterns /plates /boxes /export /sim
frontend/  Vite + React + TypeScript + vanilla three.js. UI displays mm.
  src/assembly.ts        client-side MIRROR of backend/app/assembly.py
  src/scene/BoxScene.tsx 3D preview; src/store.ts zustand state
tools/     visual verification harness (visual_verifier.py + tools/dev/)
```

**Contract pair — keep in sync:** `backend/app/assembly.py` and
`frontend/src/assembly.ts` implement the same formulas (cut dims, keep-out,
seam/hinge layout, validation). Any change to one MUST be mirrored in the other;
both sides must produce identical numbers for the same spec.

**Lazy caches:** nothing materializes at startup. Pattern variants, composed
plates, and boxes are generated on first request and cached on disk under
`backend/data/` (content-addressed by spec hash). The whole directory is
disposable — `just clean` wipes it, `just seed` pre-warms pattern defaults.
Plate SVGs are also lazy: built on first fab-export request, not at compose time.

## Tests

```sh
just test-backend   # backend pytest, run as 4 sequential chunks (see below)
just test-unit      # frontend vitest
just test-e2e       # Playwright E2E (spins up both servers itself)
just test-all       # all three layers, sequentially
just test-visual    # @visual signature capture (PNGs + meta, no LLM)
```

## Performance & safety

This codebase runs on memory-constrained dev machines and has hard guards born
from real kernel bugchecks (2026-06-10). Read these before touching geometry code.

- **Lattice budget guard** — `app/patterns/_helpers.py::check_lattice_budget`
  refuses any pattern build above `MAX_LATTICE_CELLS = 400,000` cells (raises
  `ValueError`, surfaced as HTTP 400). GEOS polygon memory is ~4-5 kB per cell and
  cell count grows quadratically with extent/period; parameters well inside the UI
  slider ranges could otherwise commit tens of GB. The plate compositor upscales
  pattern rasters to fill the aperture, so patterns never need multi-mm extents at
  micron periods.
- **No `unary_union` / whole-geometry booleans on hot paths.** Buffering or
  unioning thousands of mutually-overlapping polygons makes GEOS node the entire
  overlay in one pass — this is what froze and bugchecked the host. Hot paths use
  *concatenation* instead: overlapping members raster and SVG-fill identically,
  and the only consumers are fill-only. The authoritative in-code docstrings:
  - `app/plates.py::_concat_polygons` — why plate composition concatenates
  - `app/patterns/_helpers.py::crop_parts` — per-part clip instead of a
    whole-geometry intersection (and the OGC-validity caveat)
  - `app/patterns/frames/api.py::scene_to_multipolygon` — frame mask concat + crop
  - `app/patterns/frames/shapely_pen.py::ShapelyPen.finish` — per-part buffering;
    `merge=True` is an opt-in escape hatch with no runtime callers
  - `app/patterns/motifs/wayuu.py` / `emerald.py` — bulk lattices built via the
    shapely array API, combined by concatenation
  The remaining `unary_union` calls in `_helpers.py` live only in caller-less
  helpers (`dot_array`, `zone_plate`, `ring_grating`, `chevron_stripes`) — each
  carries a docstring stating its concat-safety class if revived.
- **Chunked test runs.** The backend suite is run as 4 sequential pytest
  invocations (`just test-backend`), each peaking around ~1 GB. Never run the
  whole suite in one process or chunks in parallel; never run two heavy compute
  processes concurrently. See `CLAUDE.md` for operator rules.

Reference timings (warm cache): 6-face box regen ~550 ms; cold pattern ~2 s.
