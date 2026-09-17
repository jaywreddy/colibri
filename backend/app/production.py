"""The production constants — process, stock, and box — in ONE place.

Every number here is a fact about what Advanced Optronics actually writes and
what the ring box actually is. Before this module they were spread across
``export_fine`` / ``witness_dies`` / ``witness_geom`` / ``export_witness`` /
``ply_cuts`` / ``boxes`` / ``patterns.base``, several of them under two or
three names each, and ``witness_dies.blank_plan`` carried a runtime assert
whose whole job was to notice when two copies of the glass had drifted apart.
A constant that needs a runtime equality check is a constant that lives in the
wrong number of places.

This module imports NOTHING from the rest of ``app``. That is what lets
``patterns.base`` read the litho floor from here without the import cycle that
forced it to mirror the number by hand (patterns -> effects.drc -> export_fine
-> patterns). Keep it that way: constants only, no geometry, no I/O.

The frontend's copy is GENERATED from this module —
``tools/dev/gen_production_constants.py`` writes ``frontend/src/production.ts``
and ``tests/test_production_constants.py`` fails if the checked-in file is
stale. Do not hand-edit the TypeScript.

Units are micrometres unless the name says otherwise.

WHAT A CHANGE HERE COSTS. These numbers are the mask. Six of them
(``ART_RIM_UM``, ``ID_TICK_OFFSET_UM``, ``MOTIF_SCALE``, ``BAND_UM``, the box
dims) are PINNED to the plate written on 2026-09-15 and are documented as such
below: they are not derived any more, because the plate is already written and
re-deriving them would move gold. Editing any constant in this module must be
followed by a witness-plate rebuild compared per layer against the previous
mask (``python -m app.export_witness``), and a bump of whichever cache
fingerprint covers the writer you touched.
"""
from __future__ import annotations

# =========================================================================== #
#  PROCESS — the gold-on-quartz line                                          #
# =========================================================================== #

LITHO_FLOOR_UM = 2.0
"""Hard process floor, BOTH senses: no written line and no gap narrower than
this, anywhere, in any polarity. The pattern generators refuse geometry under
it (``patterns.base.check_litho_floor``), the fine writer heals what survives
the rasteriser (``export_fine``, ``drc_clean_rects``), and the die inverter
checks the clear field against it after inversion (``witness_dies``, where it
used to be spelled ``CLEAR_FLOOR_UM``). 2 um line / 2 um gap is the MLA 150 +
AZ 400K + Cr-etch recipe's floor, not a safety margin on top of one."""

FINISH_RADIUS_UM = 1.0
"""Half-width of the morphological OPENs that finish a die (``witness_dies``).

The METAL is opened so no chrome filament survives, then the CLEAR complement
is opened so no slit, pinch or DBU sliver does. One radius covers both the art
box and the frame band: an open of radius r deletes every line under 2r, and
the single-ply garland writes 4.15 um leaf gratings (2.075 um lines) in the
frame, so the frame cannot be finished harder than the art. 1.2 um — chosen
when the frame held only 36 um vernier bars — erased the two finest leaf
families outright on the 07:37 plate; 1.0 leaves 75 nm of core on the finest
line and it regrows. Exactly half the litho floor, which is the most an open
can be without eating geometry that is legally on the floor."""

DBU_UM = 0.001
"""GDS/OASIS database unit: 1 nm. Coordinates are written as
``round(um / DBU_UM)``; both mask writers must agree or the same design lands
at two scales."""

POLARITY = "clear"
"""The plate is a DARKFIELD write with positive resist: the file holds the
OPENINGS and chrome stays wherever the file is empty. A face's art is authored
as METAL (gold where the rectangles are), so every die is inverted — die box
minus metal — on the way out. Getting this backwards writes the negative of
the box."""

# --- layer map --------------------------------------------------------------
# (layer, datatype) pairs. Only these four are written; the bonded design's
# back-carrier layer (20/0) and the blank-outline layer (99/0) went with the
# pairs and the blank solver, and are deliberately NOT reserved here — if a
# second written layer ever comes back it should be named for what it is.

LAYER_GOLD = (10, 0)
"""The one written layer: every die's clear field. (Was ``LAYER_FRONT`` when
the box had a back ply to be the front of.)"""

LAYER_OUTLINE = (1, 0)
"""Annotation: die outlines. Never chrome."""

LAYER_DICE = (2, 0)
"""Annotation: the saw's street centrelines. Never chrome."""

LAYER_LABEL = (3, 0)
"""Annotation: cell-ID text under each cell. Never chrome."""


# =========================================================================== #
#  STOCK — the glass and the blank                                            #
# =========================================================================== #

PLY_UM = 2250.0
"""2.25 mm fused quartz. The witness plate IS the box stock, so this is both
the blank's thickness and every face's wall: ONE ply per face since
2026-09-16, which makes the paraxial gap t/n = 1543 um the whole slab (and is
why no face can carry a parallax effect — see ``witness_geom.P_MIN_UM``)."""

GLASS_N = 1.4585
"""Fused quartz at the d-line. Every gap-scaled family of the design — the
carrier, the comb, the monogram pitch, the parallax rate, the near-field limit,
the swap angles — follows this and ``PLY_UM``."""

GLASS_MATERIAL = "fused quartz"

BLANK_SIDE_UM = 127_000.0
"""5 inch square mask blank."""

BLANK_EDGE_MARGIN_UM = 4_000.0
"""Handling / chuck exclusion: nothing written, not even a dicing tick, within
4 mm of the blank edge."""

STREET_UM = 1_000.0
"""Saw street between dies, and between rows — the blank's hand-scribe
street. Every row boundary on this plate is a real cut."""


# =========================================================================== #
#  THE BOX                                                                    #
# =========================================================================== #

WIDTH_UM = 32_000.0
DEPTH_UM = 32_000.0
HEIGHT_UM = 35_000.0
"""Outer box dimensions. Square in plan; the height carries the lid.

Set when the walls were 4.5 mm bonded stacks (a 23 x 23 x 26 mm interior). At
the 2.25 mm single-ply wall the same outer box holds 27.5 x 27.5 x 30.5 mm,
which is room to spare for the ring rather than a reason to shrink: the die
sizes are what the 2026-09-15 plate was cut to."""

TAPE_UM = 6350.0
"""1/4" copper foil. A single 2.25 mm ply is a 2.25 mm edge, so 6.35 mm of tape
leaves a 2.05 mm fold on each face — a real lap for the solder, and clear of
the 3.64 mm art rim below. (The bonded build needed 3/8": its stepped edge was
three plies, 6.75 mm, and 1/4" tape did not reach across it. 3/8" on a single
ply would fold 3.64 mm onto the face — exactly onto the art.)"""

MOTIF_SCALE = 0.68
"""Motif size dial for every face's frame. 0.68 makes the foliage read finer
and lacier without narrowing the band — the leaves simply come more often —
and keeps a flower under a millimetre inside the 2.4 mm band. The production
plate is written at this value."""

BAND_UM = 2400.0
"""One garland band width for all six faces. The old default (12% of the
shorter side) gave the lid a 3.5 mm band and the walls 2.95 mm; inside the
pinned art rim a fixed 2.4 mm keeps the lid's art box at 17 mm and the photo
walls' at 12.5 mm."""

ART_RIM_UM = 3637.5
"""THE ART RIM, PINNED. Every face's gold starts 3.6375 mm in from its edge.

This number is NOT derived, and that is the point: it is what the 2026-09-15
plate was WRITTEN with, back when the box was bonded and the rim was
``assembly.bonded_art_keepout_um`` = one ply plus the interior foil fold
(2250 + 1387.5). The box stopped being bonded on 2026-09-16; the mask did not
change, because a written plate is a written plate. Deriving the rim from the
new single-ply foil instead would move every face's garland in by ~1.1 mm and
invalidate the plate for the sake of tidiness.

It is also still a GOOD rim on the new build: the 1/4" fold lands at 2.05 mm,
so the art keeps 1.59 mm of clear glass between itself and the copper."""

ID_TICK_OFFSET_UM = 2943.75
"""Mark-centre distance from the die edge for the tick-code plate ID, PINNED.

Written on the 2026-09-15 plate; chosen when the box was bonded, where it was
``ply_cuts.fold_band_offset_um(2250, 1387.5)`` — centred in the interior
foil-fold band of a 3/8"-taped three-ply edge. Kept so the plate does not
change.

NOTE what the 2026-09-16 tape change did to it: on a single ply with 1/4" tape
the fold reaches only 2.05 mm, so a mark centred at 2.94 mm is NO LONGER under
the copper. It lands in the blank ring between the fold and the art rim —
still outside the garland, still off every optical surface, but visible on the
finished box unless the solder bead covers it. A 0.15 x 0.4 mm gold tick at
the extreme edge of a face is a bench-legible plate ID, which is what it is
for; if that is not wanted on the shipping box the fix is a mask change, not a
constant change here."""
