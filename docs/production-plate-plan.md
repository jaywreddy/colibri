# Production Plate Plan — the box's own plies on the 5″ witness

Status: **BUILT**, 2026-09-10 (second rebuild), on **2.25 mm fused quartz** (n = 1.4585). The plate is
`backend/data/witness/witness-5in.{gds,oas}` (52.3 MB / 3.1 MB, 23 cells, 115.3 of 117.9 mm); the physics argument and the
as-built numbers are in `witness-physics-plan.md` (§4.7 for the dies). This file
records the decisions and how they were validated. The 2026-09-10 rebuild
follows a three-part review (faces, renderer, mechanics) — §6 lists what it
found and what changed.

## 0. Decisions, as taken

| # | Decision | Taken |
|---|---|---|
| 1 | **Blank** | 5″ × 0.090″ fused-quartz mask blank, 2.25 mm, chrome + resist. The dies ARE box plies, so the box is built from 2.25 mm quartz plies (wall 4.5 mm) and every witness number is the box's. The glass lives in ONE place, `witness_geom.PLY_UM / GLASS_N`; comb, parallax rate, near-field limit and swap angles derive from it and a test pins them to the plate compositor. |
| 1b | **Box size** | **32 × 32 × 35 mm**, sized by the RING (`boxes.RING_*`): a 21 mm outer-diameter band standing 24 mm tall in a 1 mm liner over a 2 mm base pad needs a 23 × 23 × 26 mm interior, which is what 4.5 mm walls leave. The old 29.1 mm box was the largest whose twelve plies packed one blank; its 20.1 mm interior held no adult ring. **Jay to confirm the ring envelope** — the four RING constants and a rebuild are the whole change. |
| 1c | **Carrier pitch** | **65.5 µm**, sized by the eye (`witness_geom.CARRIER_ARCMIN` = 0.75′ at 300 mm): the lines are invisible in hand, only the beat shows. Gap-scaling the 22 µm design pitch gave 99 µm — 1.13′, a hatch the eye resolves, which read as chunks. Cost, from `tools/dev/nearfield_ladder.py`: the fringes keep 51% of zero-gap contrast across the ply at 65.5 µm (75% at 99, 34% at 55, 7% at 44). Leaf 71.4 µm, monogram 68.23 µm (beat 1635), comb 270.5 unchanged. Production faces run `carrier_scale_mode` **fixed**. |
| 2 | **Polarity** | CLEAR, as the whole plate. Pre-plated chrome, positive resist: exposed = etched = clear. Dies are inverted inside their own rectangle by one klayout Region boolean (`witness_dies.clear_field`) and decomposed to trapezoids; never the GEOS inverter. |
| 3 | **Faces** | top monogram-jp + garland (F + B), front globe-duo-phase + garland (F + B), and **all six** prepared photographs as single-ply side dies (`witness_dies.SIDE_PHOTOS`): beach (faces in diffraction colour) and sunset (plain) are the box's left and right and carry the box's own garland seeds 104 / 105; garden, paris, night-group and porch-group ride along as candidates with seeds 106–109. Ten written plies in eight dies. Every die is a face's own `PlateSpec` from `boxes.default_box_spec` through `export_fine.build_plate_fine` — one authoring path, the one the box bake uses. Sides' backing plies are bare glass. |
| 4 | **Gap-scaled pitches (first build, superseded)** | The first build ran `plates._carrier_recipe_data` at `carrier_scale_mode="gap"`: at t/n = 1543 µm (4.50× the 500 µm design point) the frame pair was 99 / 107.9 µm, the monogram carrier 105.38 µm (beat 1635 µm), the barrier comb 270.5 µm — confirmed by the near-field gate (75% of fringe contrast survives vs 11% at 22 µm). Decision 1c replaced this with the eye-sized `fixed` carrier (65.5 µm); the comb was unaffected. |
| 5 | **Capybara** | Cut from the box (back and bottom are `blank`: bare quartz on both plies). Its 15 µm slots sit at Fresnel N = 0.07 on this glass. The P-SCAN ladder left the plate with it. |
| 5b | **Single-ply sides** | photo + leaf frame on bare glass, **no carrier**. A carrier on the same ply as the leaves is a static union moiré: at the two-ply 2.5° offset its bright fringes were 1.5 mm apart and 0.4 mm wide, a few solid diagonal bars across every photo frame (the 2 µm heal welded them solid). One sheet cannot make a moiré that moves, so Jay chose the honest version: the picture dissolves to glass (`photo.CARRIER_COV` = 0) inside a garland whose leaves are **fine 50% diffractive gratings, one PERIOD per motif family** from the colour ladder 4.15–6.02 µm (`plates.SINGLE_PLY_LEAF_FILL = "hue"`, `SINGLE_PLY_LEAF_HUE_PERIODS_UM`, `leaf_fills.py`; the frame's angle buckets pick the family). Zero order is flat gold; the first order leaves at λ/p — 5.2° to 7.6° at green — so under a lamp each family shows its own hue and flashes at its own tilt: the photographs' colour-zone physics applied to foliage, verified by M-CD, M-DUTY and D-PER on this plate. Lines 2.1–3 µm, under 0.05′: the leaves read as smooth gold. Chosen over angled lines at a common pitch because axis-aligned stripes leave rectangular gaps: 13× faster to finish, half the polygons and zero DRC flags per side (angled 10 µm lines: 98 s, 195k polygons, 506 flags; 6 µm: 324 s, 293k, 3462). |
| 6 | **Experiments** | cut to the bench essentials for THIS box (23 cells in all): M-POL, M-CD dense/iso, M-DUTY 10/5, D-PER, WEDGE 44, H-ACU, M-VERN, P-RULE, B-MOVE, NF 20/64, SWAP 173/270.5. The dropped ladders are in git history (0799344). 115.3 of 117.9 mm used, no spares. |
| 7 | **Foil** | **3/8″ copper (9525 µm)**, `boxes.PRODUCTION_TAPE_UM`. The bonded stack wraps a 3-ply stepped edge, 6.75 mm; 1/4″ tape is 0.4 mm short of it and left no fold at all — the seam had no lap, the hinge nothing to solder to, and `vernier_blocks` (which needs a fold ≥ 0.48 mm to live in) wrote **no verniers**. `validate_bonded_assembly` now rejects tape narrower than 3 plies. Fold 1.39 mm per face. |
| 8 | **Art rim** | Front art starts at `assembly.bonded_art_keepout_um` = ply + fold = **3.64 mm** from the outer edge, not at the 1.89 mm foil rim: between the two the outer ply sits over the tape on the ledge and the inner ply's interior fold, with no back grating behind it, so a garland there was leaves on copper (52% of the old ring). Same rim on the single-ply sides so all six borders start on one line. Band **2.4 mm** on every face (`PRODUCTION_BAND_UM`), foliage at `motif_scale` **0.68**. Art boxes on the 32 mm box: lid 17.1 mm, front 15.8 mm, sides 13.3 mm. |
| 9 | **Written-data DRC** | `witness_dies.clear_field` opens the metal and then the CLEAR complement to the 2 µm floor (1.0 µm radius in the art box, 1.2 µm in the frame where the single-ply garland's two gratings cross at 2.5°) and settles whatever the shop-style width/space check still flags; convex decomposition (`PO_htrapezoids`), whose cut-point rounding leaves none of the notches the plain trapezoid one did. Every die's manifest carries `drc_written_front/back` — the check on the polygons actually in the file. |

What the thicker ply costs and buys: parallax is 26.9 µm/° (was 17.2), the
comb that swaps at 2.5° has a 135 µm lane — 1.55′ at 300 mm, a barrier the eye
resolves. At the eye-sized 65.5 µm carrier of decision 1c the moiré gain is back
to p/Δ = 24.0× (it was 15.5× at the gap-scaled 99 µm) and the screen-to-carrier
ratio is 65.5/44 = 1.49, no longer the commensurate 9/4 of the 99 µm build — so
the tone-dependent (2,3) banding of the 1.5 mm design stays gone and B-HARM's
50%-duty null is not floored by a shared period (§2.4 of the whitepaper).

## 1. What was built

`app/witness_dies.py` — `build_face_die` (F + B from `export_fine.build_plate_fine`,
mirrored, verniers + ID + dicing ticks, finished and inverted per die). The
legacy `build_colour_side` portrait path is gone: the photo sides are
`photo-halftone` faces of the box like any other. `production_cells()` sizes its
eight cells from the same blank solve `export_blank` uses: box
32 × 32 × 35 mm, F plies 32 × 32 (lid) / 32 × 30.5 (front) / 27.5 × 30.5 (sides), B plies
inset one ply (4.5 mm smaller). Ten written plies on the plate: lid F + B,
front F + B, and one each for the six photo sides.

`export_witness.layout` learned unequal pairs (a smaller B), pockets (the
leftover width of a tall row becomes a small shelf) and columns (cells much
shorter than the open row stack at one x). Gutter 1.0 mm = the blank's
scribe street. The DoE was then cut to the cells this box's bench actually
reads — polarity, CD, duty, the two-layer registration and switch cells at the
box's own pitches, one near-field pair either side of the design, one halftone
wedge at the photo screen, and single rungs of the diffraction and moiré
ladders. Gone: the portraits and their scale/steps/duty/coarsen/unsharp
ladders, the whole single-layer moiré block (BEAT, BCON, ROT, VEC, HARM, SCR),
the D-SWATCH and D-CROSS and D-BAND swatches, P-SCAN and B-MAG. **23 cells**,
115.3 of 117.9 mm of height; the eight production plies are 81% of the written
cell area and 61% of the usable square.

The single-ply sides carry **no carrier at all** (`photo.CARRIER_COV` = 0):
the photograph dissolves straight to bare glass inside a garland whose leaves
are written as **50% diffractive gratings with one PERIOD per motif family**
off the photographs' own 4.15–6.02 µm colour ladder
(`plates.SINGLE_PLY_LEAF_FILL` = `"hue"`, `SINGLE_PLY_LEAF_HUE_PERIODS_UM`,
`leaf_fills.py`) — so a garland family and a colour zone in the picture beside
it flash the same hue at the same tilt. (Angled lines at one common pitch, one
ORIENTATION per family, were the first construction; they cost 13× the finish
time and 506–3462 DRC flags per side against zero, because axis-aligned stripes
leave rectangular gaps — see decision 5b.) A
carrier on the same ply as the leaves made a static union moiré — solid
diagonal bars welded together by the 2 µm heal (decision 5b) — and one sheet
cannot make a moiré that moves, so the carrier was dropped rather than fake
motion a single ply cannot deliver. The monogram is set at glyph interlock
0.76 (was 0.68), fills 0.92 of its art box (was 0.82) and its glyphs are
stroked so the hairlines print at half again their width.

## 2. Validation, as run (2.25 mm quartz)

| Gate | Tool | Result |
|---|---|---|
| A1 photo tone | `tools/dev/validate_dies.py photo` | the beach side's front metal (the same `build_plate_fine` polygons the die is written from), exact area per 87 µm eye cell, vs the coverage the photo module asked for: MAE **0.52 of 22 levels**, bias −0.40 (p95 1.3); vs the photograph's own darkness 0.74 levels, −0.63. Coloured cells (6.9% of the picture) sit −0.18, plain −0.42: the 2 µm finish shaves a little from every band edge |
| A2 switch | `validate_dies.py switch` | sim2d on the die's own F/B, comb 270.5 µm over the 13.3 mm art box: separation 48.0 at zero error, 12.9 at ±8 µm, 5.3 at ±20 µm, 3.1 at 34 µm, 1.0 (blend) at 68 µm = p/4, inverted beyond. Swap ±2.51° |
| A3 near field | `validate_dies.py nearfield` | angular spectrum, 2.25 mm, n 1.4585, incoherent source, eye cell: the 500 µm design baseline (22/24 µm) keeps 10.7%; garland as built (65.5/71.4 µm) keeps 52.7%; monogram (65.5/68.23 µm, beat 1635) keeps 48.1% of zero-gap contrast |
| Geometry (metal) | `export_fine` merged DRC per die | 0 width / 0 space violations at the 2 µm floor on the authored metal of every face |
| Geometry (written) | `witness_dies.clear_field` (decompose, then the tiled checker on the pieces) + `tools/dev/probe_gds_sanity.py` on the GDS | the CLEAR data as written, merged, Euclidian, zero-distance touches excluded: lid 0 width / 2 space on F, 0 / 0 on B; front 0 / 0; beach, sunset, Paris, night, porch 0 / 0; garden (colour zones) 9 / 1 — **12 sub-micron spots on the plate**, listed in each die's `drc_written_*`. Max 22 vertices, no zero-area shapes, data 4.0 mm inside the blank's edge. Build 303 s (garden 163 s, every other die under 40 s), write 7 s; GDS 52.3 MB, OASIS 3.1 MB |
| GDS read-back | klayout probe | every die's data bbox equals its outline |
| Renderer | `just test-effects` | 8/8 on the literal composite; the gap-scaling probe moved from 80% to 95% of the gap because on 1543 µm of paraxial gap a 20% collapse is already past the garland beat's correlation length (§5) |
| Tiling | `test_witness` | analytic inverses tile their cells; coloured-band clear/metal tile by area to 1e-5 (centre-in stripe rule) |
| Glass | `test_witness` | `BOX_CARRIER_UM`, `BOX_COMB_UM`, `BOX_MONO_UM` equal the plate compositor's values for the blank spec; parallax 26.93 µm/°, p_min 41.2 µm |
| Layout | `test_witness` | dies inside the field, no overlaps, pair spans, back dims in the manifest, area budget |
| Box spec | `test_plates_and_boxes`, all ten backend chunks, frontend unit (111) | green on 2026-09-10 |

## 3. Left for the bench

Read M-POL, M-CD, M-DUTY, H-WEDGE; bond M-VERN + P-RULE first; bond
DIE-FRONT on its 80/88 µm verniers to ±8 µm; then DIE-TOP; square a bare
2.25 mm ply to each side on its F comb. The back and bottom faces are bare
glass and are not on this plate; nor are the sides' bare inner plies — sized
to each side's own F-ply cut dimensions (§1), inset one ply per the nested-shell
convention — which come from a second, uncoated 2.25 mm quartz blank. Dicing:
2.25 mm fused quartz on 1 mm streets is a saw job, not a hand scribe; the die
edge is the box's cut dimension, so specify the kerf inside the street and
±50 µm on the edge.

## 4. Photos for the sides

All six ride the plate (decision 3); the notes below are the review that chose
their treatments.

Six candidates in `photos/` were prepared to the sides' 13.3 mm art box
(inside a 15.4 mm aperture; contact
sheets in the session scratchpad, notes in `photo_notes.json`). Strongest:
the beach selfie (PXL_20240807, crop 0.20/0.03/0.62) and the sunset couple
(IMG_1827, crop 0.147/0/0.763). The garden pair (PXL_20240803) and the Paris
picnic (IMG_6584) want a segmented or lightened background; the two group
shots (IMG_1290, signal-2026-01-05) share one box between four or five faces
and read as scenes, not portraits. The night group was mirror-padded to
square by the prep and should not be — extend with a darkened blur instead.

## 5. The box in the simulator (2026-09-09)

The default box in Ring Box Studio is now the production box, and the renderer
draws it **literally**. Bonded two-ply faces (top, front) publish BOTH
`literal_front.png` and `literal_back.png` (2048 px coverage rasters of the
DRC-healed fine polygons, the same geometry the GDS bake writes) and the WebGL
scene samples both on the two real pattern planes at the paraxial t/n gap.
Single-ply faces (left/right `photo-halftone`, beach/sunset) publish
`literal_front.png` only — there is no back layer to sample, the manifest's
back raster is empty, and the inner glass ply is untouched. No procedural
gratings: the globe swap, the monogram shimmer and the garland fringes all
emerge from perspective. Faces: top monogram-jp, front globe-duo-phase, left
and right `photo-halftone` (beach / sunset, `single_ply` — leaf gratings on
the outer ply, **no carrier** — `photo.CARRIER_COV` = 0 — and no back layer),
back and bottom `blank`. Glass 2.25 mm fused quartz, n 1.4585, bonded; frame
`motif_scale` 0.68 in a 2.4 mm band.

`period_front.png` carries the fabricated sub-litho pitch at every pixel that
has one: the photo faces' 4.15–6.02 µm colour-zone sub-gratings, AND, on
single-ply faces, the garland's leaf diffraction gratings — which since the
`"hue"` fill are rungs of that SAME ladder, one period per motif family, so a
leaf and a colour zone beside it flash the same hue at the same tilt.

Two approximations remain, both flagged in the `uPeriodMap` shader comment. The
map carries ONE value over the leaves — the ladder's mean, 5.04 µm, which is
what `recipe_data.single_ply_leaf_period_um` advertises — rather than the
per-family split, so the garland sheens as one hue instead of six. And the
sheen itself is a stand-in: no texture can resolve a 5 µm grating, so the shader
hands the pitch to `diffractionSheen()` (the same function the procedural
RAINBOW accent uses) at one fixed angle instead of rendering the real grating.
Everything else on the face is the fabricated geometry.

Two renderer defects found by the 2026-09-10 review are fixed. The coverage
rasters are now EXACT area per texel (`literal_raster.py`: analytic rectangle
overlap for the halftone bands and colour stripes, Green's-theorem edge
accumulation for the angled gratings) — the PIL fill they replaced was
boundary-inclusive and phase-quantised, reading a 50% carrier at 0.53–0.57 and
the 5 µm colour stripes as solid gold. And the two chrome layers are composited
in ONE pass on the outer plane: the fragment samples the outer raster at its
own uv and the inner raster at the paraxially refracted uv, forms
1 − (1 − A)(1 − B) per sub-sample on a stratified 4×4…8×8 grid over the pixel
footprint, and averages — the eye integrates the PRODUCT of the two layers, and
filtering each layer first (mipmaps, then blend) had averaged every cross-layer
effect to a flat quarter tone at the default zoom. The inner gap is read from
the live inner-plane position, so the gap-scaling honesty check still bites.
Both layers see the glass transmission (the outer chrome is at the bond).

Costs and caveats found on the way:

* A photo face's fine bake takes about two minutes cold (66k healed polygons,
  because the colour stripes are expanded before the DRC heal); the cache hides
  it after the first compose. Keeping the stripes as array plans for the raster
  would remove most of it.
* With 2.25 mm plies the stepped bonded edge wraps 3 × 2.25 = 6.75 mm of tape,
  wider than 1/4″ copper foil — resolved: 3/8″ is the production preset
  (decision 7), the assembly contract (backend, frontend, golden fixture) carries
  it, and the keep-out is 1.89 mm of foil rim under a 3.64 mm art rim.
* Pattern-picker thumbnails for `blank` and `photo-halftone` 404 until their
  default variants are seeded (`just seed`); that is the existing
  never-generate-on-GET design, not a fault.
* The `@effects` Playwright suite runs on the literal path. The harness knows
  that a literal face binds `literal_front/back.png` on the OUTER material
  (`uBackCoverage` for the inner layer), that a blank face binds nothing, and
  that a single-ply face has no inner layer; the renderer assertions themselves
  (time-invariance, substrate-driven parallax, orbit flow) are unchanged.

## 6. The 2026-09-10 review, and what it changed

Three reviewers looked at the 2026-09-09 plate — every face for framing and
beauty, the renderer for honesty, the file and the mechanics for anything that
would stop a first-try build. What they found, and the fix:

| Finding | Severity | Fix |
|---|---|---|
| No verniers in the file: 1/4″ tape gave a zero fold, and the combs live in the fold band | blocker | 3/8″ tape (decision 7); the 80/88 µm combs and ID ticks are back on every die |
| Tape 0.4 mm narrower than the stepped edge, clamped to zero overlap instead of rejected | blocker | `validate_bonded_assembly` / `validateBondedBox` reject tape < 3 plies |
| The WRITTEN clear data had never been checked: 250 sub-2 µm clear slits (down to 1 nm) and 3 zero-area triangles across the four dies; the manifest's `drc_*` measured the pre-inversion metal | blocker | decision 9; `drc_written_*` in every die's stats; `tools/dev/probe_gds_sanity.py` runs the same check on the GDS |
| Outer 52% of the lid/front garland had no back-ply carrier (the B ply is inset one ply) — flat gold over copper | risk | decision 8: art starts at the inner ply's window |
| Dice ticks reached 3.45 mm from the plate edge, inside the 4 mm margin | risk | the packer's usable square is 0.55 mm smaller |
| Solder bead (2 mm) over a 0.5 mm keep-out | risk | 3/8″ tape: 1.89 mm rim |
| Renderer: per-layer mipmaps before the blend erased every cross-layer effect at the default zoom; raster fill inflated coverage 6–43%; outer chrome drawn without glass transmission | blocker for the audit | §5 |
| Side carrier field as dark as the picture; sunset framing loose; monogram small (0.70 of its box) | taste | carrier dropped to zero (decision 5b: leaves rewritten as 6 µm diffractive gratings instead), re-crop, 0.92 fill / 0.76 interlock / stroked glyphs |
| Stale: `build_colour_side` dead code, docs still describing ZONES/HUE portrait sides and 39k array bands | nit | removed; this file and the whitepaper §4.7 |
