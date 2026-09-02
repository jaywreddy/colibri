# Witness Plate — Physics Validation Plan

**Subject:** one 127 mm (5″) chrome-on-quartz plate, written darkfield with
positive resist. **Purpose:** measure every physical mechanism the ring box
depends on, before any box glass is written.

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
Portraits survive here only for the one question that genuinely is subjective —
*which colour treatment goes on the lid* — and they take 3% of the plate.

### Conventions used throughout

| Term | Meaning here |
|---|---|
| **eye cell** | 87 µm: what 1 arcmin subtends at a 300 mm viewing distance. Structure finer than this averages; structure coarser is seen. A held ring box is nearer, ~200 mm, so every visibility figure below is 1.5× conservative. |
| **litho floor** | 2.0 µm minimum line *and* minimum gap. A 50%-duty grating therefore needs a period of at least 4.0 µm. |
| **duty** *c* | metal fraction of a period. Efficiencies and harmonic content follow from it (§1.1). |
| **clear field** | the polarity of this write: the file holds the regions where chrome is *removed*. |
| **chrome / metal** | the opaque layer on this plate; "gold" is the box's layer. Optically interchangeable here. |
| **ply** | one of the two glass sheets of a bonded pair; **die** is a cell cut from this plate to become a ply. |
| **bonded** | a cell that needs two plies and a gap to work. Everything not so marked is single-layer. |

---

## 1. The four mechanisms

Everything the box does is one of four things, each with a governing relation,
its own free parameters, and its own way of failing.

### 1.1 Diffraction — the only source of colour

A lamellar amplitude grating of period *d* and duty *c*. The grating equation
sets which wavelength leaves in which direction and the order efficiencies set
how bright each order is:

> sin θ_m = sin θ_i + m λ / d
> η_m = (c · sinc(m c))²  →  η₀ = c²

Two consequences drive the colour design. First, **η₀ = 0.25 against
η₁ = 0.101 at 50% duty**: a gratinged band is mostly still a mirror with a
spectrum riding on it, and any model that renders only the diffracted part is
wrong by 2.5× and shows black where the first order leaves the visible band.
Second, **even harmonics vanish at exactly 50% duty** — sinc(m c) = 0 for even
m when c = ½ — which matters for moiré (§2.4).

Angular dispersion is dλ/dθ = d cos θ. A coarse grating packs the visible band
into few degrees and any wide source mixes it back to white: at the 4–5 µm the
litho floor allows, the visible spans ~5°, and a 6° room source covers all of
it. **Colour needs a lamp, not a window**, and finer periods are better twice —
a wider fan, and colour that survives softer light. Reflectance of chrome is
~0.6 and roughly flat across the visible, so it scales the spectrum without
tinting it.

*Single layer. No bond, no registration, no gap.*

### 1.2 Moiré — large structure out of invisible gratings

Two periodic structures superposed. The product of two transmittances
contains every component m f₁ + n f₂; the ones coarse enough to resolve are the
moiré.

> fundamental beat:  p_beat = p₁ p₂ / |p₁ − p₂|
> two gratings at angle α:  p_beat = p₁ p₂ / √(p₁² + p₂² − 2 p₁ p₂ cos α)
> equal pitch, rotated:  p_beat = p / (2 sin(α/2))

The second line is the general case; the other two are its limits. Its
fringes run perpendicular to **k₁ − k₂**: for a pitch difference that is across
the lines, for a rotation it is nearly *along* them, which is why rotational
moiré looks nothing like pitch moiré though the algebra is one line.

Moiré is an **amplifier**. Against the 63.5 µm carrier, a 66.07 µm front gives
a 1635 µm beat: the beat is p/Δ = 24.7× the pitch difference, so a pitch error
of 0.1 µm moves the beat by 4%. That is the same mathematics as a vernier, it
is a free metrology instrument, and it is why the lid's band spacing is the
most process-sensitive number in the design.

### 1.3 Parallax — the only thing that needs two plies

Two patterns separated by the substrate. The viewer's line of sight refracts
into the glass and crosses the gap at θ′ = asin(sin θ / n), so the back pattern
appears displaced against the front by

> shift(θ) = t · tan(asin(sin θ / n))

— 17.22 µm/° for the box (1.5 mm soda lime, n = 1.52) and **27.40 µm/° for a
bonded pair cut from this plate** (2.29 mm quartz, n = 1.4585). Every
angle in §4.4 is quoted for both.

For a parallax barrier the switch completes when the shift reaches a quarter of
the comb pitch p, and the lanes are invisible while their subtense p/2D stays
under one eye cell. Dividing the two:

> θ_swap / α_lane = n D / 2 t

The pitch cancels. At 300 mm the ratio is **152 for the box and 96 for this
plate**: swap angle and barrier visibility are locked together by the glass
alone, and only a thinner substrate or a closer viewer moves them.

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

## 2. What actually needs the bond

This decides the area budget, so it comes before the experiments.

### 2.1 The union identity

For indicator functions, inclusion–exclusion gives

> 1 − 1[A₁ ∪ A₂] = (1 − 1[A₁])(1 − 1[A₂]) = 1 − 1[A₁] − 1[A₂] + 1[A₁]·1[A₂]

The left side is the transmission of **one plane** carrying both gratings as
metal; the right side is the transmission of **two plies in contact**, each
carrying one. They are the same function, and the 1[A₁]·1[A₂] term is the
moiré. So the *static* moiré — beat spacing, fringe direction, contrast — can
be measured on a single layer for free, and that is where a third of this plate
went.

The identity is optical, not just algebraic, under three assumptions worth
naming: binary amplitude masks; geometric optics, so no diffraction between the
planes; and zero gap, so the two lattices hold a fixed relative phase. In
reflection off chrome the bright regions are the metal *union* rather than the
clear *intersection* — the same cross term, with brightness inverted — which is
why B-CONT's brightness claims are stated for transmission.

The gap buys two things and costs two:

| The gap gives | The gap costs |
|---|---|
| relative phase that rides on the line of sight — the fringes **travel** | registration: a lateral error is a phase error |
| depth, and every cross-layer illusion that depends on it | **near-field decay** (§2.2) |

### 2.2 The near-field limit

A grating does not cast a sharp shadow across millimetres. The box is lit by
incoherent white light and viewed through a pupil, so the eye is the
collimator and the question is how far a slit of width p/2 spreads by
diffraction over the gap, inside glass of index n: roughly λ z / (n p). The
shadow is gone once that spread reaches the slit width. The Fresnel number puts
the same statement on one axis:

> N = p² n / (4 λ z)  —  N ≥ 1 the shadow is intact; N < ¼ it is gone
> p_min = √(2 λ z / n)  —  **33 µm for the box, 42 µm for this plate**

| pitch | N at box (1.5 mm) | N at witness (2.29 mm) | verdict |
|---|---|---|---|
| 5 µm | 0.012 | 0.007 | washed out |
| 20 µm | 0.18 | 0.12 | washed out |
| 30 µm | 0.41 | 0.26 | degraded |
| 44 µm | 0.89 | 0.56 | degraded |
| 63.5 µm | 1.86 | 1.17 | intact |
| 100 µm | 4.6 | 2.9 | intact |

The pupil adds a second blur of the same order — a 5 mm pupil at 300 mm is a
25 µm ray bundle across the box gap, 38 µm across this plate's — so the
boundary is soft, and E-NF measures it rather than trusting the constant. The
shipping 63.5 µm carrier and 173 µm comb clear it; a 5 µm colour grating is two
orders of magnitude past it, which is fine because colour is single-layer by
construction. **No two-layer effect may be designed at a fine pitch.**

(Coherent Talbot self-imaging, z_T = 2 p² n / λ, is not the mechanism here: it
needs spatial and temporal coherence the box will never have, and its quarter
distance is where a 50% grating's shadow *vanishes*, not survives.)

### 2.3 The witness is quartz; the box is soda lime

A 5″ mask blank is 0.090″ = 2.29 mm of fused quartz. Bonding two diced dies
therefore does not reproduce the box:

| | thickness | n | parallax | n D / 2 t | swap at 173 µm |
|---|---|---|---|---|---|
| box | 1.50 mm | 1.520 | 17.22 µm/° | 152 | 2.51° |
| witness pair | 2.29 mm | 1.4585 | 27.40 µm/° | 96 | 1.58° |

The witness is the harsher case on both counts, which is the useful direction:
a barrier whose lanes are invisible here is safely invisible on the box. But
**angles do not transfer, only verdicts do**, and every bonded cell is labelled
with the witness angle.

### 2.4 The moiré the halftone carries

The box puts a 44 µm halftone screen over the 63.5 µm carrier. Their beats,
with the screen at local duty *c* over a 50% carrier:

| (m, n) | beat | at 300 mm | Michelson contrast, c = 0.20 | c = 0.42 | c = 0.50 | c = 0.65 |
|---|---|---|---|---|---|---|
| (1, 1) | 143 µm | 1.65′ | 45% | 57% | 81% | 108% |
| (2, 3) | **559 µm** | **6.4′** | 8.0% | 5.6% | **0** | 15.6% |
| (1, 3) | 41 µm | 0.47′ | — | — | — | — |

The (1,1) beat is at the edge of visibility and is what the screen-angle cell
B-SCREEN attacks. The (2,3) beat is at **9.4 cycles/degree — the peak of the
eye's contrast sensitivity, where 0.5% is visible** — and its contrast is zero
only where the screen's local duty is exactly 0.50. A halftone spans duty
0.05–0.95 *by design*, so on a photograph this beat is a **tone-dependent
banding**: absent in the midtones, 8% in the quarter-tones, 30% in the
highlights. Process bias does not create it; bias moves the null off the
midtone. B-HARM measures the contrast at three duties; B-SCREEN measures how
far running the screen perpendicular to the carrier suppresses both beats.

### 2.5 The plate may never be bonded

Bonding is a separate operation. Sections 2.1–2.2 are what make that
survivable: the entire static moiré programme is single-layer, and only
*motion*, *sampling* and *registration* need two plies. Those are confined to
one band of the plate — **34% of the written area** — so if the bond never
happens two-thirds of the plate still returns its numbers.

---

## 3. Failure modes, ranked by what they would cost

| # | Failure | Consequence | Measured by |
|---|---|---|---|
| 1 | CD fine or coarse of design | colour period wrong; finest gratings may not print | **M-CD**, **D-PER** |
| 2 | duty biased off 0.50 | efficiencies and tone wrong; the (2,3) null moves off the midtone | **M-DUTY**, **B-HARM** |
| 3 | plies misregister | every parallax effect degrades or inverts | **M-VERN** |
| 4 | bond gap off design | parallax angles wrong; near-field limit moves | **P-RULE** |
| 5 | pitch error × 25 into the beat | lid band spacing visibly wrong | **B-BEAT** |
| 6 | dot gain | picture too dark or too light | **H-WEDGE** |
| 7 | source too wide | colour washes to white | protocol, **D-PER** |
| 8 | screen beats with the carrier | banding across the photograph | **B-HARM**, **B-SCREEN** |
| 9 | polarity inverted | everything is its own negative | **M-POL** |
| 10 | fine pitch across the gap | two-layer effect simply absent | **E-NF** |

---

## 4. The experiments

Cell families are named by block — `M-` metrology, `D-` diffraction, `B-`
beat/moiré, `P-` parallax, `H-` halftone, `E-` edge of envelope. The **etched
label** on the plate is shorter and carries the parameter value; the tables
below give both. Sizes and sweeps are those of the plate as built.

### 4.1 Metrology — read these first

| Cell | Etched | Structure | Sweep | Pass |
|---|---|---|---|---|
| **M-POL** | `M-POL CLEAR` | a square with a square hole, beside a 1:3 bar pair | — | reads as chrome with a clear centre. Inverted, the reverse. Ten seconds, before anything else is interpreted |
| **M-CD** | `M-CD 0.8-8um dense / iso` | line/space pairs at 50% duty, and the same line widths isolated at 10× the pitch | 0.8 → 8 µm, 10 rungs, both environments | the finest rung that resolves, in each. Iso and dense do not print alike, and the colour ladder assumes the difference is small |
| **M-DUTY** | `M-DUTY .30-.70 @10um / @5um` | one period, duty stepped | 0.30 → 0.70 in 0.05, at 10 µm and at 5 µm | the 0.50 rung shows **no second order** (η₂ = 0 there). If it does, 0.40 vs 0.60 gives the sign of the bias |
| **M-VERN** | `M-VERN` | 80 / 88 µm comb pairs | — | coincidence read to 1 µm, 11× amplification. *Bonded* |

### 4.2 Diffraction

| Cell | Etched | Structure | Sweep | Pass |
|---|---|---|---|---|
| **D-PER** | `D-PER 2-20um` | bare gratings, 50% duty, one strip | 2 → 20 µm, 11 rungs | hue at fixed geometry and how wide the fan is. Read under a lamp *and* room light: the difference is the source-width result |
| **D-CHIRP** | `D-CHIRP` | period swept along the patch | 22 → 3 µm | a graded fan, not one flashing hue; where it stops diffracting is a continuous CD read, and the sweep crosses the 4 µm floor |
| **D-CROSS** | `CROSS 5/5`, `5/8` | two orthogonal gratings on one layer | 2 cells | orders on a 2-D grid; the cheapest check that same-plane superposition behaves as §2.1 says |
| **D-SWATCH** | `SW base/spread` | the whole 12-rung hue ladder as adjacent stripes | base 4 / 5 / 6.5 / 8 µm × spread 1.20 / 1.45 / 1.90 | which base and spread separate cleanly under a lamp. Base 4 µm and spread 1.90 put the blue end under the floor and are on the plate *to be seen failing* |
| **D-BAND** | `BAND period`, `H` = tone held | a flat-tone halftone whose bands carry a sub-grating | 5.0 / 4.15 / 6.02 µm; tone held or not | does a *banded* grating still diffract, and what does holding tone cost? Separates "the colour works" from "the picture works" |

### 4.3 Moiré — the largest block

| Cell | Etched | Structure | Sweep | Pass |
|---|---|---|---|---|
| **B-BEAT** | `BEAT beat` | two pitches on one plane | beat 500 / 1000 / 1635 / 3000 / 6000 µm; each cell holds ≥ 5 fringes | count fringes; the count inverts to the true pitch error through p/Δ |
| **B-CONT** | `BCON duty` | B-BEAT at 1635 µm, duty swept | 0.25 / 0.50 / 0.75 | fringe contrast and mean brightness, in transmission. Union coverage runs from c to 2c − c², so low duty is brighter at similar contrast |
| **B-ROT** | `ROT angle` | equal 63.5 µm pitch, one rotated | 1 / 2 / 4 / 8° → 3.6 / 1.8 / 0.91 / 0.46 mm | fringes along the lines, at the predicted spacing |
| **B-VEC** | `VEC p2/angle` | 63.5 µm against 63.5 / 66.07 / 70 µm at 1 / 3 / 6° | 3 × 3 | the general |k₁ − k₂| formula the perimeter frame relies on |
| **B-HARM** | `HARM duty` | 44 µm over 63.5 µm, both on one plane, duty swept | 0.42 / 0.50 / 0.58 | the 559 µm banding at the predicted contrast (§2.4) — 5.6%, 0, 7.7% |
| **B-SCREEN** | `SCR angle` | 44 µm screen over the 63.5 µm carrier, screen rotated | 0 / 45 / 90° | how far the perpendicular screen suppresses the 143 µm and 559 µm beats |
| **B-MAG** | `MAG 10x`, `30x` | pinhole array over a motif array | M = 10, 30 | M = p_r / (p_r − p_s). *Bonded* — sampling genuinely needs two planes |
| **B-MOVE** | `B-MOVE` | the 1635 µm beat across the gap | — | the twin of `BEAT 1635`: same geometry, so the only difference is whether the fringes **travel** — about 700 µm per degree on this plate. *Bonded* |

### 4.4 Parallax — bonded pairs

| Cell | Etched | Structure | Sweep | Pass |
|---|---|---|---|---|
| **P-SWAP** | `SWAP comb` | barrier switch, quarter-period registered | comb 100 / 173 / 250 / 350 µm | swap at **0.91 / 1.58 / 2.28 / 3.19°** on this plate (1.45 / 2.51 / 3.63 / 5.08° on the box); lanes at 0.57 / 0.99 / 1.43 / 2.01′, so the last two are visible barriers. Where does the barrier become visible, and does it swap cleanly head-on? |
| **P-SCAN** | `SCAN N` | N-phase kinegram | N = 2 / 4 / 6 | crests travel one way; wants N× the registration of a 2-phase switch |
| **P-RULE** | `P-RULE` | 60 µm comb on one ply, index line on the other | — | tilt until the index sits over tooth k: 2.2° per tooth on this plate. Reads t·n directly — the number every other bonded cell rests on |

### 4.5 Halftone

| Cell | Etched | Structure | Sweep | Pass |
|---|---|---|---|---|
| **H-WEDGE** | `WEDGE screen` | 16-patch tone wedge, duties on the screen's own ladder | screens 20 / 44 / 60 µm | measured coverage against designed: **the dot-gain number**, which inverts straight into the prep's `gain` |
| **H-ACU** | `H-ACU 20-60um` | bare screens, no image | 20 / 30 / 44 / 60 µm | at what pitch the lines are *seen*; 43.5 µm is the calculation, this is the measurement |
| **H-PORT** | `PORT PLAIN / HUE / ZONES` | the three colour treatments, 12 mm | plain / hue-mapped / zone-mapped | the one subjective call: which goes on the lid |
| **H-SCALE** | `SZ size` | one treatment at three sizes | 4 / 8 mm, with the 12 mm portraits as the third rung | where a halftone stops reading as a screen |

### 4.6 Edge of envelope

| Cell | Etched | Structure | Sweep | Pass |
|---|---|---|---|---|
| **E-NF** | `NF pitch` | the beat pair split across two plies, pitch swept through the §2.2 boundary | 20 / 30 / 44 / 64 / 100 µm | where the two-layer fringes die: predicted between 44 and 64 µm on this plate. *Bonded*; its single-layer twins are the B-BEAT cells |

Iso-dense bias, which earlier drafts listed as its own cell, is measured by the
two M-CD strips and needs nothing more.

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
3. **M-VERN**, on a bonded pair. Decides whether §4.4 is worth pursuing.
4. **D-PER** and **D-SWATCH**, under a lamp *and* under room light. The
   difference is the source-width result.
5. The **B-** block, single-layer. Count fringes; back out the pitch error
   through p/Δ; read B-HARM's contrast against §2.4.
6. **B-MOVE**, **E-NF**, the **P-** block, bonded. Read by rocking the part.
7. **H-WEDGE**, then the portraits last, with the dot gain already known.

---

## 7. What this plate does not test

* **Phase gratings.** Everything here is binary amplitude. An etched depth is
  a real colour control and roughly quadruples first-order efficiency, but
  needs a process step this build does not have.
* **The bond itself** — adhesive thickness uniformity, index, voids. P-RULE
  measures the gap you got; not whether it repeats.
* **Durability.** No abrasion, adhesion or environmental exposure.
* **The real viewing distance.** All subtenses assume 300 mm; a held box is
  examined at ~200 mm, which makes every structure ~1.5× more visible than the
  figures say. That is a margin, and it is a margin of 1.5 and not more.
