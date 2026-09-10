# Production Plate Plan — four usable faces on the 5″ witness

Status: **BUILT**, 2026-09-09, on **2.25 mm fused quartz** (n = 1.4585). The plate is
`backend/data/witness/witness-5in.{gds,oas}`; the physics argument and the
as-built numbers are in `witness-physics-plan.md` (§4.7 for the dies). This file
records the decisions and how they were validated.

## 0. Decisions, as taken

| # | Decision | Taken |
|---|---|---|
| 1 | **Blank** | 5″ × 0.090″ fused-quartz mask blank, 2.25 mm, chrome + resist. The dies ARE box plies, so the box is built from 2.25 mm quartz plies (wall 4.5 mm) and every witness number is the box's. The glass lives in ONE place, `witness_geom.PLY_UM / GLASS_N`; carrier, comb, monogram pitch, parallax rate, near-field limit and swap angles derive from it and a test pins them to the plate compositor. |
| 2 | **Polarity** | CLEAR, as the whole plate. Pre-plated chrome, positive resist: exposed = etched = clear. Dies are inverted inside their own rectangle by one klayout Region boolean (`witness_dies.clear_field`) and decomposed to trapezoids; never the GEOS inverter. |
| 3 | **Faces** | top monogram-jp + garland (F + B), front globe-duo-phase + garland (F + B), left = portrait ZONES + colour garland (F), right = portrait HUE + colour garland (F). Sides' backing plies are bare glass. |
| 4 | **Gap-scaled pitches** | `plates._carrier_recipe_data` runs `carrier_scale_mode="gap"`: at t/n = 1543 µm (4.50× the 500 µm design point) the frame pair is 99 / 107.9 µm, the monogram carrier 105.38 µm (beat 1635 µm), the barrier comb 270.5 µm. Confirmed by the near-field gate (75% of fringe contrast survives vs 11% at 22 µm). |
| 5 | **Capybara** | Cut from the box (`boxes.BACK_PATTERN_SLUG` → inscription-line, frontend mirrored). Its 15 µm slots sit at Fresnel N = 0.07 on this glass. The P-SCAN ladder left the plate with it. |
| 6 | **Spares** | none; 114.7 of 119 mm used. |

What the thicker ply costs and buys: parallax is 26.9 µm/° (was 17.2), the
moiré gain p/Δ falls to 15.5× (was 24.7×), the comb that swaps at 2.5° has a
135 µm lane — 1.55′ at 300 mm, a barrier the eye resolves — and the 99/44 = 9/4
screen-to-carrier ratio removes the tone-dependent (2,3) banding of the 1.5 mm
design but is commensurate, so B-HARM reads a minimum at 50% duty rather than a
null (§2.4 of the whitepaper).

## 1. What was built

`app/witness_dies.py` — `build_face_die` (F + B from `export_fine.build_plate_fine`,
mirrored, verniers + ID + dicing ticks, inverted per die) and `build_colour_side`
(portrait via `witness_cells.build_halftone`, colour garland with a diffraction
sub-grating per motif family from the frame scene, F vernier). `production_cells()`
sizes the four cells from the same blank solve `export_blank` uses: box
29.1 × 29.1 × 32.0 mm, F plies 29.1 / 29.1 × 27.51 / 24.6 × 27.51 mm, B plies inset
one ply (4.5 mm smaller).

`export_witness.layout` learned unequal pairs (a smaller B), pockets (the
leftover width of a tall row becomes a small shelf) and columns (cells much
shorter than the open row stack at one x). Gutter 1.0 mm = the blank's
scribe street. Cuts: portraits and scale rungs, BEAT 6000, the VEC 3° column,
P-SCAN. 70 cells; production dies 32% of the field.

## 2. Validation, as run (2.25 mm quartz)

| Gate | Tool | Result |
|---|---|---|
| A1 photo tone | `tools/dev/validate_dies.py photo` | emitted metal vs intended coverage: mean 1.27 of 22 levels, bias −0.34; clear + metal tile to 1.0000. Coloured bands print 0.9 levels lighter than the photo on average (≈2.8 inside the zones): the cost of holding tone |
| A2 switch | `validate_dies.py switch` | sim2d on the die's own F/B, comb 270.5 µm: separation 101 at zero error, 20 at ±8 µm, 6.4 at ±20 µm, 3.5 at 34 µm, 1.0 (blend) at 68 µm = p/4, inverted beyond. Swap ±2.51° |
| A3 near field | `validate_dies.py nearfield` | angular spectrum, 2.25 mm, n 1.4585, incoherent source, eye cell: garland at 22/24 keeps 11%, at 99/107.9 keeps 75%, monogram 105.38/99 keeps 73% of zero-gap contrast |
| Geometry | `export_fine` merged DRC per die | 0 width / 0 space violations at the 2 µm floor on every layer of both bonded faces |
| GDS read-back | klayout probe | every die's data bbox equals its outline |
| Tiling | `test_witness` | analytic inverses tile their cells; coloured-band clear/metal tile by area to 1e-5 (centre-in stripe rule) |
| Glass | `test_witness` | `BOX_CARRIER_UM`, `BOX_COMB_UM`, `BOX_MONO_UM` equal the plate compositor's values for the blank spec; parallax 26.93 µm/°, p_min 41.2 µm |
| Layout | `test_witness` | dies inside the field, no overlaps, pair spans, back dims in the manifest, area budget |
| Box spec | `test_plates_and_boxes` | 44 pass with the back face on inscription-line |

## 3. Left for the bench

Read M-POL, M-CD, M-DUTY, H-WEDGE; bond M-VERN + P-RULE first; bond
DIE-FRONT on its verniers to ±8 µm; then DIE-TOP; square a bare 2.25 mm ply to
each side. The back and bottom faces are not on this plate.

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
quartz, n 1.4585, bonded; frame `motif_scale` 0.75. The one non-literal term
is the diffraction sheen for the 4–6 µm colour sub-gratings (`period_front.png`),
which no texture can resolve; it is flagged in the shader.

Costs and caveats found on the way:

* A photo face's fine bake takes about two minutes cold (66k healed polygons,
  because the colour stripes are expanded before the DRC heal); the cache hides
  it after the first compose. Keeping the stripes as array plans for the raster
  would remove most of it.
* With 2.25 mm plies the stepped bonded edge wraps 3 × 2.25 = 6.75 mm of tape,
  wider than the 1/4″ (6.35 mm) copper foil, so the foil overlap floors at zero
  and the keep-out is the 0.5 mm safety margin alone. The build needs 3/8″
  tape; adding the preset is an assembly-contract change (backend + frontend
  + golden fixture) not yet made.
* Pattern-picker thumbnails for `blank` and `photo-halftone` 404 until their
  default variants are seeded (`just seed`); that is the existing
  never-generate-on-GET design, not a fault.
* The `@effects` Playwright suite passes (8/8) on the literal path. Two
  harness checks were taught that a literal face binds `literal_front/back.png`
  rather than the level-coded masks, that a blank face binds nothing, and that a
  single-ply face has no inner plane; the renderer assertions themselves
  (time-invariance, substrate-driven parallax, orbit flow) are unchanged.
