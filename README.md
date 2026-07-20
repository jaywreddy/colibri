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
  app/patterns/          pattern registry, moire motifs, frame engine (frames/),
                         bitmap/ (photo -> halftone plates, assets/bitmaps/),
                         geo/ (Natural Earth orthographic globe silhouettes,
                         rotation via lon0)
  app/export_fine.py / export_gds.py / export_wafer.py
                         fine-pitch GDS + 4-inch (100 mm) wafer packing
                         (GET /export/wafer/plan)
  app/sim2d.py           headless 2D dual-layer parallax compositor + contrast
                         metrics (the lightweight alternative to the 3D preview)
  app/api/               routers: /patterns /plates /boxes /export /sim
                         (incl. /sim/parallax2d composite + curve endpoints)
frontend/  Vite + React + TypeScript + vanilla three.js. UI displays mm.
  src/assembly.ts        client-side MIRROR of backend/app/assembly.py
  src/scene/BoxScene.tsx 3D preview — two-plane geometric renderer: back gold
                         on a real inner plane at the paraxial T/n gap, every
                         composed plate binds foliage_moire, PBR metal
                         finishes; src/store.ts zustand state
  src/lab/composite2d.ts + src/ui/PatternLab.tsx
                         2D Pattern Lab: canvas dual-layer preview with tilt
                         sliders/drag, illumination, and live param regen
tools/     visual verification harness (visual_verifier.py + tools/dev/)
```

**Contract pair — keep in sync:** `backend/app/assembly.py` and
`frontend/src/assembly.ts` implement the same formulas (cut dims, keep-out,
seam/hinge layout, validation). Any change to one MUST be mirrored in the other;
both sides must produce identical numbers for the same spec.

**Default box:** the six-face default plan (`backend/app/boxes.py`) is
front = `globe-duo-phase` (California/Colombia globe switch),
right = `gear-quill-switch`, back = `capybara-scanimation`,
left = `jamon-tray`, top = `monogram-jp`, bottom = `inscription-line`.
The catalog registers 17 patterns (16 artistic showpieces + the
bitmap-halftone photo pipeline).

**Lazy caches:** nothing materializes at startup. Pattern variants, composed
plates, and boxes are generated on first request and cached on disk under
`backend/data/` (content-addressed by spec hash). The whole directory is
disposable — `just clean` wipes it, `just seed` pre-warms pattern defaults.
Plate SVGs are also lazy: built on first fab-export request, not at compose time.

## Tests

```sh
just test-backend   # backend pytest, run as 5 sequential chunks (see below)
just test-unit      # frontend vitest
just test-e2e       # Playwright E2E (spins up both servers itself)
just test-all       # all three layers, sequentially
just test-visual    # @visual signature capture (PNGs + meta, no LLM)
just test-effects   # @effects physical-honesty suite (pixel metrics, no LLM)
```

**Assembly contract golden fixture** — `tools/fixtures/assembly_golden.json`
is generated from the backend math (`tools/dev/gen_assembly_golden.py`) and
consumed by BOTH `backend/tests/test_assembly.py` and
`frontend/tests/unit/assemblyGolden.test.ts`, so any drift between
`app/assembly.py` and `src/assembly.ts` fails a suite instead of silently
diverging. Regenerate it only when the contract intentionally changes.

**Physical-honesty effects suite** (`frontend/tests/e2e/effectsPhysical.spec.ts`
+ `effectsCatalog.ts` + `effectsHelpers.ts`) is the anti-cheat gate for the 3D
renderer. It verifies four axioms on the live WebGL buffer with deterministic
pixel metrics:

1. **View-dependent** — moiré fringes flow under camera orbit; the stereo
   lenticular flips scenes across the slit axis; the carrier reveal
   de-registers across the switch axis (front figure on a stripe carrier,
   uniform anti-phase carrier on the inner plane).
2. **Time-invariant** — a parked camera yields pixel-identical frames (no
   time-animated shader fakes).
3. **Texture-driven** — imagery binds from the backend litho masks; stereo
   view textures must really bind (no front-mask fallback).
4. **Substrate physics (geometric gap)** — the back gold layer lives on a
   REAL inner plane at the paraxial T/n air gap below the outer plane. At a
   fixed oblique view, collapsing that two-plane gap to zero registers the
   layers and moves the fringes; a partial collapse moves them proportionally
   less; recapturing at the same gap is pixel-identical. The suite manipulates
   the actual plane gap (`effectsHelpers.ts::scaleBackPlaneGap`) — the legacy
   `uThicknessUm` uniform is dead on the foliage path.

It also covers the lid transition (monotonic hinge rotation + closed-frame
round-trip), illumination modes (distinct + laser-colored), and turntable
flow. Every scenario dumps a PNG frame sequence + metrics sidecar under
`frontend/test-results/visual/latest/effects/`; `just test-effects-verify`
additionally grades the sequences with Claude vision
(`tools/visual_verifier.py`, needs `ANTHROPIC_API_KEY`).

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
- **Chunked test runs.** The backend suite is run as 5 sequential pytest
  invocations (`just test-backend`), each peaking around ~1 GB. Never run the
  whole suite in one process or chunks in parallel; never run two heavy compute
  processes concurrently. See `CLAUDE.md` for operator rules.

Reference timings (warm cache): 6-face box regen ~550 ms; cold pattern ~2 s.
