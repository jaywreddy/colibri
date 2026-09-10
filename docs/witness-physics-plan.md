# Witness Plate — Physics Validation Plan

**Subject:** one 127 mm (5″) chrome plate on **2.25 mm fused quartz — the box's own
stock** — written darkfield with positive resist. **Purpose:** measure every
physical mechanism the ring box depends on, and carry four of the box's faces
as finished plies on the same write (§4.7).

---

## 0. What this plate is for

It is not a sampler. A sampler shows that effects exist; this plate has to
return **numbers that change the box design**, from a single write. The
organising question for every cell is therefore *what would make this fail on
the box, and what measurement would have told me?* Cells that answer nothing
were cut.

Photographs are the wrong instrument for most of that. A portrait is a
subjective read of many coupled variables at once; a step wedge and a swatch
matrix return the same information objectively in a twentieth of the area.
There are no portrait *cells* on this plate at all: the two colour **sides** of
the box (§4.7) are the portraits, at 15 mm, and the one subjective question —
zone-mapped or hue-mapped — is answered on the box itself.

The plate is also the box's glass. Since it is written on the 2.25 mm fused quartz
the bonded box is built from, a rectangle of it carrying a face's fine geometry
IS that face's ply once diced. Four faces ride along — the lid and the front as
bonded pairs, the two colour sides as single plies — and take
31% of the usable field; the experiments fill the rest.

### The box as designed

Every derived number below uses these.

| | |
|---|---|
| **glass** | 2.25 mm fused quartz, n = 1.4585 — box and plate alike (the 5″ × 0.090″ mask blank) |
| **witness pair** | two dies of this plate stacked: the same 2.25 mm gap as the box, so every angle below is one number |
| **production dies** | top F 29.1 mm + B 24.6 mm; front F 29.1 × 27.51 + B 24.6 × 23.01; left and right F 24.6 × 27.51 (single ply). Cut dims from the blank solve at this ply, so a die here is interchangeable with one from the full panel |
| **carrier** | 99 µm back-layer grating, 50% duty — the lattice every lid moiré beats against (22 µm at the 500 µm design point, × the gap ratio 4.50) |
| **screen** | 44 µm halftone, 22 levels (2 µm step) |
| **comb** | 270.5 µm parallax-barrier comb; lane = p/2 = 135.25 µm — 1.55′ at 300 mm, a barrier the eye resolves |
| **colour sub-grating** | base 5 µm, 12-rung ladder, spread 1.45 (red/blue pitch ratio), 50% duty |
| **viewing** | D = 300 mm; a held box is ~200 mm |
| **λ** | 550 nm for every Fresnel number and coherence width |

### Conventions

| Term | Meaning here |
|---|---|
| **eye cell** | 87 µm: what 1 arcmin subtends at 300 mm. Structure finer than this averages; coarser is seen. At the ~200 mm a box is held, every visibility figure below is 1.5× conservative. |
| **litho floor** | 2.0 µm minimum line *and* minimum gap. A 50%-duty grating therefore needs a pitch of at least 4.0 µm. |
| **pitch** *p*, **duty** *c* | period of a grating, and its metal fraction. Efficiencies and harmonic content follow from *c* (§1.1). |
| **beat** | the resolvable difference component m f₁ + n f₂ of two superposed gratings — the moiré. |
| **polarity** | the plate is **darkfield**: chrome everywhere the file is empty. The drawn geometry is the **clear data** — the openings where chrome is removed — for positive resist. The mask file and the `M-POL CLEAR` label use those two words. |
| **chrome / metal** | the opaque layer on this plate; the box's is gold. Optically interchangeable here. |
| **ply / die** | a ply is one glass sheet of a bonded pair; a die is the piece cut from this plate to become one. |
| **bonded** | a cell that needs two plies and a gap to work. Everything not so marked is single-layer. Experiment pairs stack **chrome-up on chrome-up, unmirrored**; the production dies are **mirrored** for the box's chrome-down stack (§2.3). |
| **perimeter frame** | the moiré frame around the lid: two lattices at unequal pitch *and* a small angle — hence B-VEC. |
| **dot gain / prep gain** | realised minus designed coverage; and the correction the image pipeline applies for it. |
| **cell families** | named by mechanism: `M-` metrology, `D-` diffraction, `B-` beat/moiré, `P-` parallax, `H-` halftone, `E-` edge of envelope, and `DIE-` for the four production faces. E-NF is packed with the moiré block and the appendix lists it there. |

---

## 1. The four mechanisms

Everything the box does is one of four things, each with a governing relation,
its own free parameters, and its own way of failing.

### 1.1 Diffraction — the only source of colour

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
m when c = ½ — which matters for moiré (§2.4).

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
above it. D-BAND measures it.

*Single layer. No bond, no registration, no gap.*

### 1.2 Moiré — large structure out of invisible gratings

Two periodic structures superposed. The product of two transmittances
contains every component m f₁ + n f₂; the ones coarse enough to resolve are the
beats.

> pitch beat:  p_beat = p₁ p₂ / |p₁ − p₂|
> two gratings at angle α:  p_beat = p₁ p₂ / √(p₁² + p₂² − 2 p₁ p₂ cos α)
> equal pitch, rotated:  p_beat = p / (2 sin(α/2))

The second line is the general case; the other two are its limits. Its
fringes run perpendicular to **k₁ − k₂**: for a pitch difference that is across
the lines, for a rotation it is nearly *along* them, which is why rotational
moiré looks nothing like pitch moiré though the algebra is one line. The box's
perimeter frame is the general case — unequal pitches at a small angle — and
B-VEC checks it.

Moiré is an **amplifier**. Against the 99 µm carrier, a 105.38 µm front
grating (Δ = p₂ − p₁ = 6.38 µm) gives a 1635 µm beat. The beat is p₁p₂/Δ, so
it is p₁/Δ = 15.5 times the carrier pitch and scales as 1/Δ: a 0.1 µm pitch
error is 1.6% of Δ and moves the beat 1.6%. (On the 1.5 mm design the
carrier was 63.5 µm, Δ 2.57 µm and the gain 24.7×; the thicker ply coarsens the
carrier and halves the sensitivity.) That is the same mathematics as a vernier, it
is a free metrology instrument, and it is why the lid's band spacing is the
most process-sensitive number in the design.

Sampling one lattice through another — a pinhole array over a motif array — is
the same superposition read a different way: the pinholes pick one point of
each motif cell and the picked points drift cell to cell, so the eye sees the
motif magnified by M = p_s/(p_s − p_m). B-MAG is that mechanism, and the only
one here that needs two planes.

### 1.3 Parallax — the only thing that needs two plies

Two patterns separated by the substrate. The viewer's line of sight refracts
into the glass and crosses the gap at θ′ = asin(sin θ / n), so the back pattern
appears displaced against the front by

> shift(θ) = t · tan(asin(sin θ / n))

— **26.93 µm/°**. A bonded pair cut from this plate is the box's glass, so
every angle in §4.4 is one number.

For a parallax barrier the switch completes when the shift reaches a quarter of
the comb pitch p, and the lanes are invisible while their subtense p/2D stays
under one eye cell. Dividing the two:

> θ_swap / α_lane = n D / 2 t

The pitch cancels. At 300 mm the ratio is **97**: swap angle and barrier
visibility are locked together by the glass alone, and only a thinner substrate
or a closer viewer moves them. It is the price of a 2.25 mm ply: the comb that
swaps at 2.5° is 270.5 µm, whose 135.25 µm lane is 1.55′ at 300 mm — a
visible barrier — where the 1.5 mm design's 173 µm comb sat exactly at the 1′
threshold. (A kinegram is the same barrier with N frames in
1/N lanes and wants N times the registration; the capybara scanimation was cut
from the box for the near-field reason in §2.2, and its P-SCAN ladder left with
it.)

### 1.4 Halftone — the only source of continuous tone

Tone from the *width* of a metal band at fixed pitch. Two bounds meet:

> steps ≤ p / 2 µm  (the finest band must clear the litho floor)
> p ≤ 43.5 µm  (the eye cell must span two periods, or the screen is seen as lines)

which is why the design sits at 44 µm and 22 levels. Coverage is linear in area
and the eye re-encodes it, so matching an sRGB source needs
coverage = linearise(source); feeding sRGB in directly renders a 0.25 midtone
as 0.54.

### 1.5 Metrology

Not a mechanism: verniers, CD ladders, duty ladders, a polarity witness. They
do not appear on the box. They are what turns the other four sections from
opinions into numbers, and they are read first (§6).

---

## 2. What needs the bond

This decides the area budget, so it comes before the experiments.

### 2.1 The union identity

For indicator functions, inclusion–exclusion gives

> 1 − 1[A₁ ∪ A₂] = (1 − 1[A₁])(1 − 1[A₂]) = 1 − 1[A₁] − 1[A₂] + 1[A₁]·1[A₂]

The left side is the transmission of **one plane** carrying both gratings as
metal; the right side is the transmission of **two plies in contact**, each
carrying one. They are the same function, and the 1[A₁]·1[A₂] term is the
beat. So the *static* moiré — beat spacing, fringe direction, contrast — can be
measured on a single layer for free: 1,770 mm² of this plate, 26% of the
written area, is single-layer moiré for that reason.

The identity is optical, not just algebraic, under three assumptions worth
naming: binary amplitude masks; geometric optics, so no diffraction between the
planes; and zero gap, so the two lattices hold a fixed relative phase. In
reflection off chrome the bright regions are the metal *union* rather than the
clear *intersection* — the same cross term, with brightness inverted — so every
contrast in this document is given for both.

The gap buys two things and costs two:

| The gap gives | The gap costs |
|---|---|
| relative phase that rides on the line of sight — the fringes **travel** | registration: a lateral error is a phase error |
| depth, and every cross-layer illusion that depends on it | **near-field decay** (§2.2) |

### 2.2 The near-field limit

A grating does not cast a sharp shadow across millimetres. Take λ = 550 nm.
Under coherent light a 50% grating's odd orders carry phase
exp(−2πi m² z / z_T) with z_T = 2p²n/λ. At z = z_T/4 that phase is −i for every
odd m, so u = ½ − i(g − ½) and |u|² = ½ everywhere: the shadow has **zero**
contrast. Define

> N = p² n / (4 λ z) = z_T / (8 z)

N = 1 is z_T/8, where |u|² = ½ ± 1/(2√2), contrast 0.71: intact. N = ½ is the
null, and the pitch that puts the gap there is

> p_min = √(2 λ z / n) — **41 µm** at the 2.25 mm ply

N = ¼ is z_T/2, where coherent light would revive an inverted image. It does
not here: the pupil's coherence width at the plate, λD/a = 33 µm, is under one
period, and the revival needs the ±1 orders — sheared by λz/(np) = p at that
distance — to stay coherent. So: **N ≥ 1 intact, ¼ < N < 1 degraded, N < ¼
gone.**

| pitch | N at 2.25 mm | verdict | simulated (§4.7) |
|---|---|---|---|
| 5 µm | 0.01 | gone |  |
| 15 µm | 0.07 | gone — the capybara's slot; the face was cut |  |
| 20 µm | 0.12 | gone |  |
| 22 µm | 0.14 | gone — the garland at the 500 µm design pitch | 11% of its fringe survives |
| 30 µm | 0.27 | degraded |  |
| 44 µm | 0.57 | degraded |  |
| 63.5 µm | 1.19 | intact — the 1.5 mm design's carrier |  |
| 99 µm | 2.89 | intact — the garland and monogram as built | 75% (garland), 72% (monogram) |
| 270.5 µm | 21.56 | intact — the comb |  |

The pupil adds a geometric blur of the same order: a 5 mm pupil at 300 mm is a
16.7 mrad cone in air, 16.7/n inside the glass — 26 µm across the gap. The
boundary is therefore soft, and E-NF measures it rather than trusting the
constant. The right-hand column is the angular-spectrum result of §4.7: the
fraction of the zero-gap fringe contrast that survives 2.25 mm of glass under an
incoherent source, eye-cell integrated. It puts the frame's two gratings where
the constant said: at the 500 µm design pitch of 22 / 24 µm the garland would keep 11% of its
contrast (0.46 → 0.05) and would not shimmer; at 99 / 107.9 µm — the gap-scaled pitch the plate compositor applies on this glass —
it keeps 75% (0.85 → 0.64), and the monogram pair 72%.
The 270.5 µm comb is at N = 22. A 5 µm colour grating is two orders of magnitude
past the boundary, which is fine because colour is single-layer by
construction. **No two-layer effect may be designed at a fine pitch.**

### 2.3 Two stacking conventions on one plate

The plate is the box's glass, so a bonded pair of dies reproduces the box's
gap exactly: 2.25 mm plus the adhesive, n = 1.4585, 26.93 µm/°. What differs
between the two kinds of pair on this plate is orientation.

| | stack | mirrored in the file | gap |
|---|---|---|---|
| **experiment pairs** (M-VERN, P-RULE, B-MOVE, NF, SWAP, MAG) | front die chrome-up on top of the back die, also chrome-up; adhesive between the back chrome and the front die's bare glass | no — a die must **not** be flipped | 2.25 mm |
| **production dies** (DIE-TOP, DIE-FRONT) | the box's chrome-down stack: outer ply F with its chrome at the bond, inner ply B with its chrome facing the interior; the viewer looks through the F glass | **yes**, both plies, x → −x about the die centre, exactly as `export_blank` writes the full panel | 2.25 mm |

In both cases one ply of glass separates the two chrome layers, so the angles
are the same; the labels and the 80 / 88 µm verniers tell the two conventions
apart at the bench. The B die of a production pair is 4.5 mm smaller than its F
(inset one ply per edge — the nested-shell corner of the bonded box), and its
verniers sit at identical stack coordinates so the pair beats when it is
aligned.

### 2.4 The moiré the halftone carries

The box puts the 44 µm halftone screen over the 99 µm carrier — a ratio of
2.25. Screen harmonic m has amplitude a_m = |sin(π m c)| / (π m); the 50%
carrier's b_n = |sin(π n / 2)| / (π n), zero for even n. The (m, n) beat rides on
the mean at amplitude 2 a_m b_n. The columns below are 2A/Ī — the modulation of
that one component against the mean; in transmission the mean is (1 − c)/2, in
reflection off gold (1 + c)/2, so every reflection figure is the transmission
figure × (1 − c)/(1 + c). This is the *tone* case: screen at local duty c over
a carrier held at 0.50. The last column is the *bias* case B-HARM is built to —
both gratings at c — for the beats that only exist under bias.

| (m, n) | beat | at 300 mm | 2A/Ī transmission, c = 0.20 / 0.42 / 0.50 / 0.65 | reflection off gold | bias case at 0.42 / 0.50 / 0.58 |
|---|---|---|---|---|---|
| (1, 2) | 396 µm | 4.55′ | **0** at every tone (even carrier harmonic) | **0** | 14.1 / 0 / 26.8% |
| (2, 4) | 198 µm | 2.28′ | **0** at every tone (even carrier harmonic) | **0** | 3.1 / 0 / 5.8% |
| (1, 3) | 132 µm | 1.52′ | 10 / 23 / 27 / 34% | 7 / 9 / 9 / 7% | — |
| (1, 1) | 79 µm (under the eye cell) | 0.91′ | 30 / 68 / 81 / 103% | 20 / 28 / 27 / 22% | — |
| (2, 3) | 66 µm (under the eye cell) | 0.76′ | 8 / 6 / 0 / 16% | 5 / 2 / 0 / 3% | — |
| (1, 4) | 57 µm (under the eye cell) | 0.65′ | **0** at every tone (even carrier harmonic) | **0** | 12.3 / 0 / 23.5% |

This is where the thicker ply pays back. On the 1.5 mm design the carrier was
63.5 µm and the visible even-harmonic beat was (2, 3) at 559 µm: its amplitude
carried a₂(c) of the *screen*, so it vanished only where the picture's own tone
was exactly 0.50 and returned as tone-dependent banding everywhere else. Here
the 2.25 ratio puts the even harmonic on the *carrier* side: the
(1, 2) beat at 396 µm (4.5′) carries b_2, which a carrier
at exactly 50% duty makes **zero at every tone of the photograph**. It exists
only through process bias, and B-HARM reads that null and its sign. What
remains at every duty is the odd family — the (1, 3) beat at 132 µm
(1.5′) is the largest — a fine, tone-following texture rather than a
band, and B-SCREEN measures how far a perpendicular screen suppresses it.

One caveat the ratio brings with it. 99/44 is exactly 2.25 = 9/4, so the two
lattices are commensurate: the whole superposition repeats every 396 µm, and
the odd-n term (3, 7) lands on the same 396 µm as (1, 2). It does not vanish
at 50% — its amplitude is a₃ b₇, about 3.9% of the mean by the sinusoid
estimate — so B-HARM will read a *minimum* at 0.50, a few percent deep, not a
clean null; the simulation of the built cells finds the same floor. The screen
pitch is the knob that moves this (43 µm breaks the 9/4 ratio but puts (3, 7)
at 1.06 mm), and B-SCREEN's 45 and 90° rungs are the other way out.

### 2.5 The plate may never be bonded

Bonding is a separate operation. Sections 2.1–2.2 are what make that
survivable: the entire static moiré programme is single-layer, and only
*motion*, *sampling* and *registration* need two plies. Among the experiments
those are **32% of the written area**, so if the bond never
happens 68% of the experiment still returns its numbers. Of the four
production faces, the two colour sides need no bond at all.

---

## 3. Failure modes, ranked by what they would cost

| # | Failure | Consequence | Measured by |
|---|---|---|---|
| 1 | CD fine or coarse of design | colour pitch wrong; finest gratings may not print | **M-CD**, **D-PER** |
| 2 | duty biased off 0.50 | efficiencies and tone wrong; the (2, 3) null moves off the midtone | **M-DUTY**, **B-HARM** |
| 3 | plies misregister | every parallax effect degrades or inverts | **M-VERN** |
| 4 | bond gap off design | parallax angles wrong; near-field limit moves | **P-RULE** |
| 5 | pitch error × 24.7 into the beat | lid band spacing visibly wrong | **B-BEAT** |
| 6 | dot gain | picture too dark or too light | **H-WEDGE** |
| 7 | source too wide | colour washes to white | protocol, **D-PER** |
| 8 | screen beats with the carrier | banding across the photograph | **B-HARM**, **B-SCREEN** |
| 9 | polarity inverted | everything is its own negative | **M-POL** |
| 10 | fine pitch across the gap | two-layer effect absent | **E-NF** |
| 11 | frame lattice angle or pitch off | perimeter-frame fringe spacing wrong | **B-VEC** |
| 12 | sampler/motif pitch ratio off | magnified motif wrong size or upright | **B-MAG** |
| 13 | a production pair bonded off register | the globe swap goes asymmetric or vanishes | the die's own verniers, and **DIE-FRONT** is `SWAP 270.5` at 29.1 mm |

---

## 4. The experiments

Sizes and sweeps are those of the plate as built; the **etched** column is what
is read under the microscope.

### 4.1 Metrology — read these first

| Cell | Etched | Structure | Sweep | Pass |
|---|---|---|---|---|
| **M-POL** | `M-POL CLEAR` | a square with a square hole, beside a 1:3 bar pair | — | reads as chrome with a clear centre. Inverted, the reverse. Ten seconds, before anything else is interpreted |
| **M-CD** | `M-CD 0.8-8um dense / iso` | line/space pairs at 50% duty, and the same line widths isolated at 10× the pitch | 0.8 → 8 µm, 10 rungs, both environments | the finest rung that resolves, in each. Iso and dense do not print alike, and the colour ladder assumes the difference is small |
| **M-DUTY** | `M-DUTY .30-.70 @10um / @5um` | one pitch, duty stepped | 0.30 → 0.70 in 0.05, at 10 µm and at 5 µm | the 0.50 rung shows **no second order** (η₂ = 0 there). If it does, 0.40 vs 0.60 gives the sign of the bias |
| **M-VERN** | `M-VERN` | 80 / 88 µm comb pairs | — | coincidence read to 1 µm; beat 880 µm, gain p/Δ = 80/8 = 10×. *Bonded* |

### 4.2 Diffraction

| Cell | Etched | Structure | Sweep | Pass |
|---|---|---|---|---|
| **D-PER** | `D-PER 2-20um` | bare gratings, 50% duty, one strip | 2 → 20 µm, 11 rungs | hue at fixed geometry and how wide the fan is. Read under a lamp *and* room light: the difference is the source-width result |
| **D-CHIRP** | `D-CHIRP` | pitch swept along the patch | 22 → 3 µm | a graded fan, not one flashing hue; where it stops diffracting is a continuous CD read, and the sweep crosses the 4 µm floor |
| **D-CROSS** | `CROSS 5/5`, `5/8` | two orthogonal gratings on one layer | 2 cells | orders on a 2-D grid; the cheapest check that same-plane superposition behaves as §2.1 says |
| **D-SWATCH** | `SW base/spread` | the whole 12-rung hue ladder as adjacent stripes | base 4 / 5 / 6.5 / 8 µm × spread 1.20 / 1.45 / 1.90 | which base and spread separate cleanly under a lamp. `SW 4/*` and `SW 5/1.90` put the blue end under the floor and are on the plate *to be seen failing* |
| **D-BAND** | `BAND pitch`, `H` = tone held | a flat quarter-tone halftone whose bands carry a sub-grating | `BAND 5H / 4.15H / 6.02H`: 22 µm bands, tone held (N = 4.4 periods at 5 µm); `BAND 5`: 10 µm bands, not held (N = 2.0) | the held cells still show a hue at N = 4.4 (~23% linewidth); the unheld one sits at N = 2.0, the two-period boundary where a spectrum should just fail. The pair is the cost of holding tone, measured (§1.1) |

### 4.3 Moiré — the largest block

| Cell | Etched | Structure | Sweep | Pass |
|---|---|---|---|---|
| **B-BEAT** | `BEAT beat` | two pitches on one plane | beat 500 / 1000 / 1635 / 3000 µm; each cell holds ≥ 5 fringes | count fringes; the count inverts to the true pitch error through p/Δ |
| **B-CONT** | `BCON duty` | B-BEAT at 1635 µm, duty swept | 0.25 / 0.50 / 0.75 | fringe contrast and mean brightness. Local metal coverage runs from c (lines coincident) to min(2c, 1) (interleaved). Transmission: mean (1−c)² = 0.56 / 0.25 / 0.06, contrast 0.20 / 1.0 / 1.0 — low duty is brighter, at a fifth of the contrast. Reflection: mean 0.44 / 0.75 / 0.94, contrast 0.33 / 0.33 / 0.14. Which reads better on a lid is not a calculation |
| **B-ROT** | `ROT angle` | equal 99 µm pitch, one rotated | 1 / 2 / 4 / 8° → 5.67 / 2.84 / 1.42 / 0.71 mm | fringes along the lines, at the predicted spacing |
| **B-VEC** | `VEC p2/angle` | 99 µm against 99 / 105.38 / 108.9 µm at 1 / 6° | 3 × 2 | the general \|k₁ − k₂\| formula the perimeter frame relies on. The 1° column holds under 5 fringes at 5 mm and checks fringe *direction*, not spacing |
| **B-HARM** | `HARM duty` | 44 µm over 99 µm, both on one plane, **both at the swept duty** | 0.42 / 0.50 / 0.58 | the uniform-bias case: the (1, 2) beat at 396 µm, 2 a_1(c) a_2(c)/(1−c)² — zero at 0.50 and nonzero either side, with the sign of the bias. With the carrier held at 0.50 this beat is absent at every tone (§2.4), so the cell tests the null and its sign, not a lid amplitude |
| **B-SCREEN** | `SCR angle` | 44 µm screen over the 99 µm carrier, screen rotated | 0 / 45 / 90° | how far the perpendicular screen suppresses the (1, 3) 132 µm texture and the (1, 2) 396 µm beat |
| **B-MAG** | `MAG 10x`, `31.6x` | pinhole array (sampler p_s = 60 µm) over a motif array (p_m = 66.0 / 61.9 µm) | M = p_s/(p_s − p_m) = −10 / −31.6, inverted | a floating lattice at the predicted magnification, upside-down. *Bonded* — sampling needs two planes |
| **B-MOVE** | `B-MOVE` | the 1635 µm beat across the gap | — | the twin of `BEAT 1635`: same geometry, so the only difference is whether the fringes **travel** — (1635/99) × 26.93 µm/° ≈ 445 µm per degree. *Bonded*; DIE-TOP is the same experiment at 28 mm |

### 4.4 Parallax — bonded pairs

| Cell | Etched | Structure | Sweep | Pass |
|---|---|---|---|---|
| **P-SWAP** | `SWAP comb` | barrier switch, straddle-registered: the slit sits on a lane boundary head-on | comb 100 / 173 / 270.5 / 350 µm | a 50/50 blend head-on, clean B at +p/4 and clean A at −p/4: swap at **0.93 / 1.61 / 2.51 / 3.25°**; lanes at 0.57 / 0.99 / 1.55 / 2.01′ — the box comb (270.5 µm) is the third rung and is a visible barrier by construction |
| **P-RULE** | `P-RULE` | 60 µm comb on one ply, index line on the other; a tooth edge sits under the index head-on | — | tilt until the index sits over tooth k: 2.23° per tooth (60 µm / 26.93 µm per °), 9 teeth inside ±10°. Reads t/n directly — the number every other bonded cell rests on |

### 4.5 Halftone

| Cell | Etched | Structure | Sweep | Pass |
|---|---|---|---|---|
| **H-WEDGE** | `WEDGE screen` | 16-patch tone wedge, duties on the screen's own ladder | screens 20 / 44 / 60 µm | measured coverage against designed: **the dot-gain number**, which inverts straight into the prep gain. At 20 µm only 9 distinct duties fit (10 levels at the 2 µm floor), so seven patches repeat — read the repeats as within-cell uniformity; 44 and 60 µm give 16 distinct |
| **H-ACU** | `H-ACU 20-60um` | bare screens, no image | 20 / 30 / 44 / 60 µm | at what pitch the lines are *seen*; 43.5 µm is the calculation, this is the measurement |

The portrait cells and the scale ladder are gone: **DIE-LEFT** (zones) and
**DIE-RIGHT** (hue) are the portraits, at 15 mm on the box's own sides, and the
plain control is the H-WEDGE. The subjective call is made on the box.

### 4.6 Edge of envelope

| Cell | Etched | Structure | Sweep | Pass |
|---|---|---|---|---|
| **E-NF** | `NF pitch` | the beat pair split across two plies, pitch swept through the §2.2 boundary | 20 / 30 / 44 / 64 / 100 µm | where the two-layer fringes die: N = ½ null at 41 µm — 20 gone (N = 0.12), 30 degraded (N = 0.27), 44 degraded (N = 0.57), 64 intact (N = 1.21), 100 intact (N = 2.95). *Bonded*; its single-layer twins are the B-BEAT cells |

### 4.7 Production dies

Four faces of the box, written on this plate as the plies they will be. Each is
also an experiment: the two bonded faces are `B-MOVE` and `SWAP 270.5` at 29.1 mm,
and the sides are the portraits.

| Die | Etched | What it is | Its own witness role | Written |
|---|---|---|---|---|
| **DIE-TOP** | `TOP F monogram-jp`, `TOP B` | the lid: J+P monogram on a 105.38 µm carrier beating the inner ply's 99 µm carrier at 1635 µm; foliage garland at 99 / 107.9 µm | travelling two-layer moiré at full size — does the 1635 µm beat move at 445 µm/° and hold contrast (§2.2)? | F 29.1 × 29.1, B 24.6 × 24.6 mm; 17,506 polygons |
| **DIE-FRONT** | `FRONT F globe-duo-phase`, `FRONT B` | the front: California ↔ Colombia globe as a parallax barrier, comb 270.5 µm, both globes interlaced on B under a neutral slit comb on F; garland | the barrier switch at 29.1 mm: 50/50 head-on, clean swap at ±2.51°, and the bench registration tolerance | F 29.1 × 27.51, B 24.6 × 23.01 mm; 19,701 polygons |
| **DIE-LEFT** | `LEFT F colour ZONES` | a side: the portrait at 15.4 mm, 44 µm screen, 22 levels, zone-mapped colour; a colour garland with a diffraction sub-grating per motif family (orchid 4.15, philodendron 4.7, fern 4.85, plantain 5, vine 5, wax_palm 5.16, heliconia 5.66, coffee 5.84, anthurium 6.02 µm) | the halftone and the colour ladder at production size; single-layer, no bond | F 24.6 × 27.51 mm; 202,134 band rectangles + 39,209 coloured bands as arrays |
| **DIE-RIGHT** | `RIGHT F colour HUE` | the same portrait, hue-mapped; same garland | the subjective comparison with DIE-LEFT, on the box | F 24.6 × 27.51 mm; 204,472 + 69,867 |

Every die is mirrored for the chrome-down stack and carries the 80 µm (F) or
88 µm (B) assembly verniers and a tick-code ID in the interior foil-fold band,
plus corner scribe ticks in the street. The sides' backing plies are bare glass
and take no plate area. The polarity is CLEAR like the rest of the plate: each
die's gold geometry (the fine builder's merged, DRC-healed polygons) is
inverted inside its own rectangle by one klayout Region boolean and decomposed
to trapezoids, so no polygon carries a hole. The chrome that stays is the gold
of the box; the clear is its bare glass.

**In-silico gates, run before ordering** (`tools/dev/validate_dies.py`):

*A1 — the photo.* The left side's metal, with the sub-gratings flattened and its area accumulated exactly per 87 µm eye cell, matches the coverage the builder intends to **1.27 of the 22 tone levels** on average (95th percentile 4.9, bias -0.34); clear and metal tile the portrait to 1.000. Against the photograph's darkness itself the die reads 1.3 levels light, all of it in the 35% of cells that carry colour: a coloured band holds tone with a 50% sub-grating, so it prints at half the metal — the cost of holding tone from §1.1, here 0.9 levels averaged over the picture and about 2.7 inside the coloured zones. That is the number the prep gain has to carry if the zones are to sit at the photograph's tone.

*A2 — the switch.* `sim2d.switch_metrics` on DIE-FRONT's own front and back
metal, rasterised at 4 µm over the art box, comb 270.5 µm, t = 2.25 mm, n = 1.4585
(swap at ±2.51°). Then the back ply is slid by a registration error against
the design lanes and the metric repeated:

| back-ply error | lane shown at +p/4 | lane that should vanish | separation |
|---|---|---|---|
| 0 µm | 0.01 | 0.97 | 101.0 |
| 8 µm | 0.04 | 0.94 | 20.5 |
| 20 µm | 0.13 | 0.85 | 6.4 |
| 34 µm | 0.22 | 0.76 | 3.5 |
| 51 µm | 0.36 | 0.62 | 1.7 |
| 68 µm | 0.48 | 0.50 | 1.0 |
| 85 µm | 0.60 | 0.38 | 1.5 |

The lane pitch is p/2 = 135.25 µm, so an error of p/4 = 68 µm is a 50/50 blend
at every tilt and past it the two globes trade places. The table is what the
verniers have to hold: **±8 µm keeps a 20:1 swap, ±20 µm a 6.4:1 one.** The
80/88 µm combs beat at 880 µm and read coincidence to about 1 µm, so the bond
has margin of an order of magnitude — if it is set on the verniers and not by
eye. The coarser comb of the 2.25 mm ply is more forgiving than the 173 µm one
was, by exactly the ratio of the pitches.

*A3 — the near field.* The §2.2 column: angular-spectrum propagation through
2.25 mm of n = 1.4585 glass, an incoherent source (11 angles over ±0.5°, three
wavelengths), eye-cell integrated, on 4 mm patches of the exact grating pairs.
A garland at the 500 µm design pitch (22 / 23.98 µm at 2.5°) would keep 11% of
its zero-gap fringe contrast; as built (99 / 107.9 µm) it keeps 75%; the
monogram pair (99 / 105.38 µm) keeps 72%. That is the number that decided
the garland pitch and cut the capybara.

Iso-dense bias is measured by the two M-CD strips and needs no cell of its own.

---

## 5. Area budget, as built

The table is generated from the plate manifest at page-build time, so it
cannot disagree with the plate. Written area counts a bonded cell twice, once
per die.

*(table inserted from the manifest)*

---

## 6. Protocol

1. **M-POL.** If the polarity is inverted nothing else means what it appears
   to mean.
2. **M-CD**, **M-DUTY**, under a microscope. These gate the colour design and
   feed the dot-gain correction.
3. **D-PER** and **D-SWATCH**, under a lamp *and* under room light. The
   difference is the source-width result.
4. The **B-** block, single-layer. Count fringes; back out the pitch error
   through p/Δ; read B-HARM's null and sign against §2.4.
5. **H-WEDGE** → the dot gain. Then look at **DIE-LEFT** and **DIE-RIGHT**
   under the lamp: they are the portraits, and they need no bond.
6. Dice. Bond **M-VERN** + **P-RULE** first, chrome-up on chrome-up; read
   registration and t/n. Then **B-MOVE**, **E-NF**, the **P-** block.
7. Bond **DIE-FRONT** (mirrored, chrome-down, B nested inside F) on its
   verniers: it is `SWAP 270.5` at 29.1 mm, and A2's table says how much error the
   swap forgives. Then **DIE-TOP**: `B-MOVE` at 29.1 mm.
8. Square a bare 2.25 mm ply to each side on its F vernier.

---

## 7. What this plate does not test

* **Phase gratings.** Everything here is binary amplitude. An etched depth is
  a real colour control and roughly quadruples first-order efficiency, but
  needs a process step this build does not have.
* **The bond itself** — adhesive thickness uniformity, index, voids. P-RULE
  measures the gap you got; not whether it repeats.
* **Durability.** No abrasion, adhesion or environmental exposure.
* **The other two faces.** Back and bottom are not on this plate; the
  capybara scanimation that was the back is cut from the design (§2.2), and
  the back defaults to the inscription line until something replaces it.
* **The real viewing distance.** All subtenses assume 300 mm; a held box is
  examined at ~200 mm, which makes every structure ~1.5× more visible than the
  figures say. That is a margin, and it is a margin of 1.5 and not more.
