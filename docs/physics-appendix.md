# Physics Appendix — the single-layer box

Extracted from `docs/archived/witness-physics-plan.md` in the 2026-09-15
tools+docs cleanup, trimmed to the physics the box actually stands on today
(decision 10 of `docs/decisions.md`: every face is one written ply). The full
whitepaper — including the two-ply moiré, parallax-barrier and bonded-pair
mechanics it also covers — is archived there in full as the record of the
first (bonded) design; this appendix pulls forward only what still applies,
verbatim except where a cross-reference needed adjusting, plus a glossary at
the end for names you'll still meet in code, tests and git history.

---

## Conventions

| Term | Meaning here |
|---|---|
| **eye cell** | 87 µm: what 1 arcmin subtends at 300 mm. Structure finer than this averages; coarser is seen. At the ~200 mm a box is held, every visibility figure here is 1.5× conservative. |
| **litho floor** | 2.0 µm minimum line *and* minimum gap. A 50%-duty grating therefore needs a pitch of at least 4.0 µm. |
| **pitch** *p*, **duty** *c* | period of a grating, and its metal fraction. Efficiencies and harmonic content follow from *c*. |
| **polarity** | the mask is **darkfield**: chrome everywhere the file is empty. The drawn geometry is the **clear data** — the openings where chrome is removed — for positive resist. |
| **chrome / metal** | the opaque layer on the mask; the box's is gold. Optically interchangeable here. |
| **cell families** | named by mechanism: `M-` metrology, `D-` diffraction, `H-` halftone. (`B-`, `P-`, `E-` were moiré/parallax/edge-of-envelope families for the retired bonded mechanics — see the glossary.) |

---

## Diffraction — the only source of colour

*(witness-physics-plan.md §1.1, verbatim)*

A lamellar amplitude grating of pitch *p* and duty *c*. The grating equation
sets which wavelength leaves in which direction and the order efficiencies set
how bright each order is:

> sin θ_m = sin θ_i + m λ / p
> η_m = (c · sinc(m c))²  →  η₀ = c²

Two consequences drive the colour design. First, **η₀ = 0.25 against
η₁ = 0.101 at 50% duty**: a gratinged band is mostly still a mirror with a
spectrum riding on it, and any model that renders only the diffracted part is
wrong by 2.5× and shows black where the first order leaves the visible band.
Second, **even harmonics vanish at exactly 50% duty** — sinc(m c) = 0 for even
m when c = ½.

Angular dispersion is dλ/dθ = p cos θ. A coarse grating packs the visible band
into few degrees and any wide source mixes it back to white: at the 4–5 µm the
litho floor allows, the visible spans ~5°, and a 6° room source covers all of
it. **Colour needs a lamp, not a window**, and finer pitches are better twice —
a wider fan, and colour that survives softer light. Chrome reflectance is ~0.6
and roughly flat across the visible, so it scales the spectrum without tinting
it.

On the box the grating is chopped into the halftone's bands. A band *b* wide
holds N = b/p periods and its first order is Δλ/λ ≈ 1/N wide: a full 44 µm band
at 5 µm is N = 8.8, an 11% linewidth — clean; a 0.2-coverage band is 8.8 µm,
N = 1.8, no spectrum. A 50% sub-grating also removes half the band's metal, so a
coloured band must be widened to tone/duty of the pitch to hold its tone — and
that saturates at tone = duty: above the midtone a held-tone band is the whole
period, a continuous grating with a phase reset every 44 µm, and the tone clips.
That is what holding tone costs: colour only below the midtone, or tone lost
above it. (`D-BAND` measured it; cut from the current plate for space — see
the glossary.)

*Single layer. No bond, no registration, no gap.*

## Halftone — the only source of continuous tone

*(witness-physics-plan.md §1.4, verbatim)*

Tone from the *width* of a metal band at fixed pitch. Two bounds meet:

> steps ≤ p / 2 µm  (the finest band must clear the litho floor)
> p ≤ 43.5 µm  (the eye cell must span two periods, or the screen is seen as lines)

which is why the design sits at 44 µm and 22 levels. Coverage is linear in area
and the eye re-encodes it, so matching an sRGB source needs
coverage = linearise(source); feeding sRGB in directly renders a 0.25 midtone
as 0.54.

## One ply makes no moiré — the single-layer colour-zone mechanism

*(witness-physics-plan.md §2.4b, adapted: the cross-reference to §2.5b's
specular-double-pass discussion is dropped below since that section is
two-ply-only and no longer applies to any box face — see the glossary.)*

Two gratings on ONE chrome layer still make a moiré — the static union of the
two-plane identity, coverage from the carrier's duty where lines coincide to
near-solid where they interleave — but it is a printed texture, not a
shimmer, and at a 2.5° offset between two same-plane lattices the fringes
were 1.5 mm apart: one or two solid bars across a 2.4 mm band, which the
first (bonded) build showed and the 2 µm heal welded solid. This is why the
box's photo sides and region-art centrepieces carry **no carrier at all**:
the picture or map dissolves to bare glass (`photo.CARRIER_COV` = 0,
`region_art`'s ungratinged regions solid gold) inside leaves/regions written
as 50% gratings.

What one ply CAN do is diffract. The garland leaves (and the region-art
gratings) are written as fine 50% gratings with one PERIOD per family from
the colour ladder (4.15–6.02 µm, vertical lines). The zero order of a flat
50% grating is plain gold, but the first order returns, with the lamp behind
the viewer (**Littrow** condition), at sin θ = λ/2p — 2.6° to 3.8° of tilt at
green across the ladder — so at one tilt every family lights at once, each in
its own colour (at 3.0°: 434 nm at 4.15 µm to 630 nm at 6.02 µm), and rocking
the box sweeps the hues through the families. The lamp must be narrow enough
for that colour to stay saturated (≲5° retro, since dλ/dα = 2p cos α at
retro-reflection). Angled lines at a common pitch (one orientation per
family) were measured first and rejected on cost: axis-aligned stripes leave
rectangular gaps that need no convex decomposition and have no acute tips,
so the period fill finishes 13× faster with half the polygons and no DRC
flags. It is the colour-zone physics above applied to foliage (and, via
`region_art`, to the monogram and the globe), honest for a single layer and
view-dependent without a gap; its price is that it needs a directional light
(a window smears the orders into a flat 50% sheen) and that its colour is
spectral, not white. The finest rung, 4.15 µm, keeps its lines at 2.07 µm —
on the floor by design, the same rung the photographs' colour zones already
print — and a one-cell glass gutter keeps neighbouring families' (or
region-art regions') gratings apart.

---

## Surviving bench cells

The full experiment tables are in the archived whitepaper (§4.1–4.6); these
are the cells decision 6 of `docs/decisions.md` keeps on the current plate.

### Metrology

| Cell | Structure | Pass |
|---|---|---|
| **M-POL** | a square with a square hole, beside a 1:3 bar pair | reads as chrome with a clear centre. Inverted, the reverse. Read first, before anything else is interpreted |
| **M-CD** (dense/iso) | line/space pairs at 50% duty, and the same line widths isolated at 10× the pitch, 0.8 → 8 µm, 10 rungs | the finest rung that resolves, in each. Iso and dense do not print alike, and the colour ladder assumes the difference is small |
| **M-DUTY** (10/5 µm) | one pitch, duty stepped 0.30 → 0.70 in 0.05, at 10 µm and 5 µm | the 0.50 rung shows **no second order** (η₂ = 0 there). If it does, 0.40 vs 0.60 gives the sign of the bias |

### Diffraction

| Cell | Structure | Pass |
|---|---|---|
| **D-PER** | bare gratings, 50% duty, one strip, 2 → 20 µm, 11 rungs | hue at fixed geometry and how wide the fan is. Read under a lamp *and* room light: the difference is the source-width result |

D-CHIRP, D-CROSS, D-SWATCH and D-BAND (the graded-pitch fan, the crossed
lattice, the full 12-rung hue-swatch matrix, and the held-tone band pair)
were cut from the current plate for space, not because the physics stopped
mattering — `render_witness_figures.py`'s colour-swatch figure still
exercises the same ladder D-SWATCH would have shown. Full tables in the
archived whitepaper.

### Halftone

| Cell | Structure | Pass |
|---|---|---|
| **H-WEDGE** (screens 20/44/60 µm) | 16-patch tone wedge, duties on the screen's own ladder | measured coverage against designed: the dot-gain number. At 20 µm only 9 distinct duties fit (10 levels at the 2 µm floor), so seven patches repeat; 44 and 60 µm give 16 distinct |
| **H-ACU** (20-60 µm) | bare screens, no image | at what pitch the lines are *seen*; 43.5 µm (the eye-cell bound above) is the calculation, this is the measurement |

---

## Glossary of retired terms

Two-ply mechanics survive in the codebase only as hidden exemplars
(`monogram` two-ply, `globe-duo-phase`) — never entered for a production
face — and in git history. If you meet one of these names in code, a test,
or an old doc, this is what it meant:

| Term | Meant |
|---|---|
| **bonded pair / ply pair** | two glass plies separated by a gap, glued together — the mechanism every parallax/moiré-motion effect needed. The box has none: every face is one written ply over a bare inner ply. |
| **carrier** | the back-layer grating a front pattern beat against for a travelling moiré, or the barrier comb a front slit switched against. No face carries one; `photo.CARRIER_COV = 0` everywhere. |
| **comb / parallax barrier** | a two-lane grating on the back ply that shows one of two interlaced images depending on view angle (a "switch"). Retired with `globe-duo-phase`; the front is now the single-layer `globe-atlantic`. |
| **beat / moiré (bonded sense)** | the large fringe pattern from two gratings at slightly different pitch across a gap, moving with tilt (parallax). Distinct from the *static* same-plane union moiré in "One ply makes no moiré" above, which the box does use (as the reason it avoids a carrier). |
| **swap angle** | the tilt at which a parallax barrier flips from showing one interlaced image to the other. |
| **scanimation / kinegram** | a barrier with N frames in 1/N lanes — an animated version of the switch. The capybara scanimation was cut from the box (near-field cost) before the single-ply pivot. |
| **vernier (M-VERN)** | a comb pair used to register two plies to ~1 µm by eye. Meaningless with nothing to register. |
| **B-, P-, E- cell families** | moiré (`B-`), parallax (`P-`) and edge-of-envelope near-field (`E-`) bench cells, all bonded-only. Cut from the current plate; `M-`, `D-` and `H-` families survive (see above). |
| **`sim2d`, `readability`, `collage`, `app.sim*`** | the holography/parallax simulators and their contrast-metric routes, deleted in the 2026-09-15 tools+docs cleanup along with the UI views built on them (Pattern Lab, Collage, Progression). The one physics function still needed outside them (`_propagate`, the angular-spectrum FFT sandwich for the A3 near-field gate) is now vendored directly in `tools/dev/validate_dies.py`. |
| **`witness_moire` / `witness_cells` builders** | the bonded bench-cell geometry generators (beat, rotation, harmonic, near-field, parallax-ruler, vernier, swap ladders). Most were deleted with the cells they built; what the surviving `M-`/`D-`/`H-` cells and tests still need was kept. |
