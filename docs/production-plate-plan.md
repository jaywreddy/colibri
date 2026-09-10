# Production Plate Plan — four usable faces on the 5″ witness

Status: **BUILT**, 2026-09-10, on **2.25 mm fused quartz** (n = 1.4585). The plate is
`backend/data/witness/witness-5in.{gds,oas}`; the physics argument and the
as-built numbers are in `witness-physics-plan.md` (§4.7 for the dies). This file
records the decisions and how they were validated. The 2026-09-10 rebuild
follows a three-part review (faces, renderer, mechanics) — §6 lists what it
found and what changed.

## 0. Decisions, as taken

| # | Decision | Taken |
|---|---|---|
| 1 | **Blank** | 5″ × 0.090″ fused-quartz mask blank, 2.25 mm, chrome + resist. The dies ARE box plies, so the box is built from 2.25 mm quartz plies (wall 4.5 mm) and every witness number is the box's. The glass lives in ONE place, `witness_geom.PLY_UM / GLASS_N`; carrier, comb, monogram pitch, parallax rate, near-field limit and swap angles derive from it and a test pins them to the plate compositor. |
| 2 | **Polarity** | CLEAR, as the whole plate. Pre-plated chrome, positive resist: exposed = etched = clear. Dies are inverted inside their own rectangle by one klayout Region boolean (`witness_dies.clear_field`) and decomposed to trapezoids; never the GEOS inverter. |
| 3 | **Faces** | top monogram-jp + garland (F + B), front globe-duo-phase + garland (F + B), left = the beach photograph as a 44 µm line screen with the faces in diffraction colour (F, single ply), right = the sunset photograph, plain, re-cropped tight on the couple (F, single ply). Every die is the face's own `PlateSpec` from `boxes.default_box_spec` through `export_fine.build_plate_fine` — one authoring path, the one the box bake uses. Sides' backing plies are bare glass. |
| 4 | **Gap-scaled pitches** | `plates._carrier_recipe_data` runs `carrier_scale_mode="gap"`: at t/n = 1543 µm (4.50× the 500 µm design point) the frame pair is 99 / 107.9 µm, the monogram carrier 105.38 µm (beat 1635 µm), the barrier comb 270.5 µm. Confirmed by the near-field gate (75% of fringe contrast survives vs 11% at 22 µm). |
| 5 | **Capybara** | Cut from the box (back and bottom are `blank`: bare quartz on both plies). Its 15 µm slots sit at Fresnel N = 0.07 on this glass. The P-SCAN ladder left the plate with it. |
| 6 | **Spares** | none; 114.7 of 117.9 mm used (the usable square is drawn 0.55 mm inside the blank's 4 mm margin so the corner dicing ticks stay out of it). |
| 7 | **Foil** | **3/8″ copper (9525 µm)**, `boxes.PRODUCTION_TAPE_UM`. The bonded stack wraps a 3-ply stepped edge, 6.75 mm; 1/4″ tape is 0.4 mm short of it and left no fold at all — the seam had no lap, the hinge nothing to solder to, and `vernier_blocks` (which needs a fold ≥ 0.48 mm to live in) wrote **no verniers**. `validate_bonded_assembly` now rejects tape narrower than 3 plies. Fold 1.39 mm per face. |
| 8 | **Art rim** | Front art starts at `assembly.bonded_art_keepout_um` = ply + fold = **3.64 mm** from the outer edge, not at the 1.89 mm foil rim: between the two the outer ply sits over the tape on the ledge and the inner ply's interior fold, with no back grating behind it, so a garland there was leaves on copper (52% of the old ring). Same rim on the single-ply sides so all six borders start on one line. Band **2.4 mm** on every face (`PRODUCTION_BAND_UM`), foliage at `motif_scale` **0.68**. Art boxes: lid 14.6 mm, front 13.3 mm, sides 10.8 mm. |
| 9 | **Written-data DRC** | `witness_dies.clear_field` opens the metal and then the CLEAR complement to the 2 µm floor (1.0 µm radius in the art box, 1.2 µm in the frame where the single-ply garland's two gratings cross at 2.5°) and settles whatever the shop-style width/space check still flags; convex decomposition (`PO_htrapezoids`), whose cut-point rounding leaves none of the notches the plain trapezoid one did. Every die's manifest carries `drc_written_front/back` — the check on the polygons actually in the file. |

What the thicker ply costs and buys: parallax is 26.9 µm/° (was 17.2), the
moiré gain p/Δ falls to 15.5× (was 24.7×), the comb that swaps at 2.5° has a
135 µm lane — 1.55′ at 300 mm, a barrier the eye resolves — and the 99/44 = 9/4
screen-to-carrier ratio removes the tone-dependent (2,3) banding of the 1.5 mm
design but is commensurate, so B-HARM reads a minimum at 50% duty rather than a
null (§2.4 of the whitepaper).

## 1. What was built

`app/witness_dies.py` — `build_face_die` (F + B from `export_fine.build_plate_fine`,
mirrored, verniers + ID + dicing ticks, finished and inverted per die). The
legacy `build_colour_side` portrait path is gone: the photo sides are
`photo-halftone` faces of the box like any other. `production_cells()` sizes the
four cells from the same blank solve `export_blank` uses: box
29.1 × 29.1 × 32.0 mm, F plies 29.1 / 29.1 × 27.51 / 24.6 × 27.51 mm, B plies inset
one ply (4.5 mm smaller).

`export_witness.layout` learned unequal pairs (a smaller B), pockets (the
leftover width of a tall row becomes a small shelf) and columns (cells much
shorter than the open row stack at one x). Gutter 1.0 mm = the blank's
scribe street. Cuts: portraits and scale rungs, BEAT 6000, the VEC 3° column,
P-SCAN. 70 cells; production dies 32% of the field.

The single-ply sides write their carrier at **35% duty** (`photo.CARRIER_COV`,
re-exported as `plates.SINGLE_PLY_CARRIER_DUTY`): on one ply the leaves lie ON
the carrier, and against a 50% field they had 25 points of contrast and the
picture faded into a mid-grey slab; at 35% the field is a veil the leaves stand
off by 33 points and the photograph's edge visibly dissolves into. The
monogram is set at glyph interlock 0.76 (was 0.68), fills 0.92 of its art box
(was 0.82) and its glyphs are stroked so the hairlines print at half again
their width.

## 2. Validation, as run (2.25 mm quartz)

| Gate | Tool | Result |
|---|---|---|
| A1 photo tone | `tools/dev/validate_dies.py photo` | the beach side's front metal (the same `build_plate_fine` polygons the die is written from), exact area per 87 µm eye cell, vs the coverage the photo module asked for: MAE **0.64 of 22 levels**, bias −0.43 (p95 3.6); vs the photograph's own darkness 0.84 levels, −0.65. Coloured cells (6% of the picture) sit −0.21, plain −0.45: the 2 µm finish shaves a little from every band edge |
| A2 switch | `validate_dies.py switch` | sim2d on the die's own F/B, comb 270.5 µm over the 13.3 mm art box: separation 33 at zero error, 11.7 at ±8 µm, 5.1 at ±20 µm, 3.0 at 34 µm, 1.0 (blend) at 68 µm = p/4, inverted beyond. Swap ±2.51° |
| A3 near field | `validate_dies.py nearfield` | angular spectrum, 2.25 mm, n 1.4585, incoherent source, eye cell: garland at 99/107.9 keeps 75%, monogram 105.38/99 keeps 72.5% of zero-gap contrast |
| Geometry (metal) | `export_fine` merged DRC per die | 0 width / 0 space violations at the 2 µm floor on the authored metal of every face |
| Geometry (written) | `witness_dies.clear_field` + `tools/dev/probe_gds_sanity.py` on the GDS | the CLEAR data as written, merged, Euclidian, zero-distance touches excluded: lid 0 width / 1 space (a 1.16 µm chrome filament) on F, 0 / 1 on B; front 2 width (0.54 µm) / 0; beach side 28 / 9; sunset side 9 / 16 — 66 sub-micron spots on the plate, listed in each die's `drc_written_*`, down from ~250 nanometre slits and 3 zero-area polygons. Max 22 vertices, no zero-area shapes, data 4.0 mm inside the blank's edge |
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
glass and are not on this plate; nor are the sides' bare inner plies — four
plies (two 24.6 × 24.6, two 20.1 × 23.01 mm) come from a second, uncoated
2.25 mm quartz blank. Dicing: 2.25 mm fused quartz on 1 mm streets is a saw
job, not a hand scribe; the die edge is the box's cut dimension, so specify
the kerf inside the street and ±50 µm on the edge.

## 4. Photos for the sides

Six candidates in `photos/` were prepared to the 15.4 mm art box (contact
sheets in the session scratchpad, notes in `photo_notes.json`). Strongest:
the beach selfie (PXL_20240807, crop 0.20/0.03/0.62) and the sunset couple
(IMG_1827, crop 0.147/0/0.763). The garden pair (PXL_20240803) and the Paris
picnic (IMG_6584) want a segmented or lightened background; the two group
shots (IMG_1290, signal-2026-01-05) share one box between four or five faces
and read as scenes, not portraits. The night group was mirror-padded to
square by the prep and should not be — extend with a darkened blur instead.

## 5. The box in the simulator (2026-09-09)

The default box in Ring Box Studio is now the production box, and the renderer
draws it **literally**: every composed face publishes `literal_front.png` /
`literal_back.png` (2048 px coverage rasters of the DRC-healed fine polygons,
the same geometry the GDS bake writes) and the WebGL scene samples those on the
two real pattern planes at the paraxial t/n gap. No procedural gratings: the
globe swap, the monogram shimmer and the garland fringes all emerge from
perspective. Faces: top monogram-jp, front globe-duo-phase, left and right
`photo-halftone` (beach / sunset, `single_ply` — leaf gratings and carrier on
the outer ply, inner ply bare), back and bottom `blank`. Glass 2.25 mm fused
quartz, n 1.4585, bonded; frame `motif_scale` 0.68 in a 2.4 mm band. The one
non-literal term is the diffraction sheen for the 4–6 µm colour sub-gratings
(`period_front.png`), which no texture can resolve; it is flagged in the shader.

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
| Side carrier field as dark as the picture; sunset framing loose; monogram small (0.70 of its box) | taste | 35% carrier, re-crop, 0.92 fill / 0.76 interlock / stroked glyphs |
| Stale: `build_colour_side` dead code, docs still describing ZONES/HUE portrait sides and 39k array bands | nit | removed; this file and the whitepaper §4.7 |
