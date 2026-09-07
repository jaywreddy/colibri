# Production Plate Plan — four usable faces on the 5″ witness

Status: **BUILT**, 2026-09-03. The plate is `backend/data/witness/witness-5in.{gds,oas}`;
the physics argument and the as-built numbers are in `witness-physics-plan.md`
(§4.7 for the dies). This file records the decisions and how they were validated.

## 0. Decisions, as taken

| # | Decision | Taken |
|---|---|---|
| 1 | **Blank** | 5″ × 0.060″ (1.5 mm) soda lime, chrome + resist. The dies ARE box plies; every witness number is the box's. |
| 2 | **Polarity** | CLEAR, as the whole plate. Pre-plated chrome, positive resist: exposed = etched = clear. Dies are inverted inside their own rectangle by one klayout Region boolean (`witness_dies.clear_field`) and decomposed to trapezoids; never the GEOS inverter. |
| 3 | **Faces** | top monogram-jp + garland (F + B), front globe-duo-phase + garland (F + B), left = portrait ZONES + colour garland (F), right = portrait HUE + colour garland (F). Sides' backing plies are bare glass. |
| 4 | **Garland pitch** | Already gap-scaled: `plates._carrier_recipe_data` runs `carrier_scale_mode="gap"`, so on 1.5 mm stock the frame pair is 63.5 / 69.2 µm, not the 22 / 24 µm baseline. Confirmed by the near-field gate (74% of fringe contrast survives vs 12% at 22 µm). No code change was needed. |
| 5 | **Capybara** | Cut from the box (`boxes.BACK_PATTERN_SLUG` → inscription-line, frontend mirrored). Its 15 µm slots sit at Fresnel N = 0.03. The P-SCAN ladder left the plate with it. |
| 6 | **Spares** | none; 115.4 of 119 mm used. |

## 1. What was built

`app/witness_dies.py` — `build_face_die` (F + B from `export_fine.build_plate_fine`,
mirrored, verniers + ID + dicing ticks, inverted per die) and `build_colour_side`
(portrait via `witness_cells.build_halftone`, colour garland with a diffraction
sub-grating per motif family from the frame scene, F vernier). `production_cells()`
sizes the four cells from the same blank solve `export_blank` uses.

`export_witness.layout` learned unequal pairs (a smaller B), pockets (the
leftover width of a tall row becomes a small shelf) and columns (cells much
shorter than the open row stack at one x). Gutter 1.0 mm = the blank's
scribe street. Cuts: portraits and scale rungs, BEAT 6000, the VEC 3° column,
P-SCAN. 70 cells, 9,770 mm² written, 31% of it dies.

## 2. Validation, as run

| Gate | Tool | Result |
|---|---|---|
| A1 photo tone | `tools/dev/validate_dies.py photo` | emitted metal vs intended coverage: mean 0.80 of 22 levels, bias −0.36; clear + metal tile to 0.995. Coloured bands print 0.9 levels lighter than the photo on average (≈2.7 inside the zones): the cost of holding tone |
| A2 switch | `validate_dies.py switch` | sim2d on the die's own F/B: separation 54 at zero error, 12 at ±8 µm, 3.5 at ±20 µm, 1.0 (blend) at 43 µm = p/4, inverted beyond. Swap ±2.51° |
| A3 near field | `validate_dies.py nearfield` | angular spectrum, 1.5 mm, incoherent source, eye cell: garland 22/24 keeps 12%, 63.5/69.2 keeps 74%, monogram 69% of zero-gap contrast |
| Geometry | `export_fine` merged DRC per die | 0 width / 0 space violations at the 2 µm floor on every layer of both bonded faces |
| GDS read-back | klayout probe | every die's data bbox equals its outline; clear fractions 88.5 / 57.3 / 72.9 / 58.6 / 84.6 / 86.3 % |
| Tiling | `test_witness` | analytic inverses tile their cells; coloured-band clear/metal tile by area to 1e-5 (centre-in stripe rule) |
| Layout | `test_witness` | dies inside the field, no overlaps, pair spans, back dims in the manifest, area budget |
| Box spec | `test_plates_and_boxes` | 44 pass with the back face on inscription-line |

## 3. Left for the bench

Read M-POL, M-CD, M-DUTY, H-WEDGE; bond M-VERN + P-RULE first; bond
DIE-FRONT on its verniers to ±8 µm; then DIE-TOP; square a bare ply to each
side. The back and bottom faces are not on this plate.
