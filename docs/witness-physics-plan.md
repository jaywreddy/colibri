# Witness Plate — Physics Validation Plan

**Subject:** one 127 mm (5″) chrome-on-quartz plate, written darkfield, positive
resist. **Purpose:** measure every physical mechanism the ring box depends on,
before any box glass is written.

---

## 0. What this plate is for, and what it is not

It is not a sampler. A sampler shows that effects exist; this plate has to
return **numbers that change the box design**, and it has to return them from a
single write, because a second write costs a cycle.

The organising question is therefore not "does the effect work" but "**what
would make it fail on the box, and what measurement would have told me?**" Every
cell below exists because it answers one of those, and cells that answer nothing
have been cut regardless of how good they look.

An earlier revision of this plate spent **40% of its area on photographic
portraits** — one per parameter, thirty of them. That is the wrong instrument.
A portrait is a subjective read of many coupled variables at once; a step wedge
and a swatch matrix give the same information objectively in a twentieth of the
area. Portraits survive here only where the question genuinely is subjective
(*which of these three do I want on the lid*), and they are small.

---

## 1. The four mechanisms

Everything the box does is one of four things. Each has a governing relation,
its own free parameters, and its own way of failing.

### 1.1 Diffraction — the only source of colour

A lamellar amplitude grating of period `d` and duty `c`. The grating equation

> **sin θ_m = sin θ_i + m λ / d**

sets *which* wavelength leaves in *which* direction, and the order efficiencies

> **η_m = (c · sinc(mc))²**,  so **η₀ = c²**

set how bright each order is. Two consequences drive the whole colour design:

* **η₀ = 0.25 against η₁ = 0.101 at 50% duty.** A gratinged band is *mostly still
  a mirror* with a spectrum riding on it. Any model that renders only the
  diffracted part is wrong by a factor of 2.5 and will show black where the
  first order leaves the visible band.
* **Even harmonics vanish at exactly 50% duty.** This is not a curiosity — see
  §2.4, where it becomes a failure mode.

Angular dispersion is `dλ/dθ = d·cos θ`, so a *coarse* grating packs the visible
band into few degrees and any wide source mixes it back to white. At the 4–5 µm
the litho floor allows, the visible band spans ~5°, and a 6° room source covers
all of it. **Colour needs a lamp, not a window.** Finer periods are better
twice over: wider fan *and* colour that survives softer light.

*Single layer. No bond, no registration, no gap.*

### 1.2 Moiré — large structure out of invisible gratings

Two periodic structures superposed. The product of two transmittances contains
every component `m·f₁ + n·f₂`; the ones small enough to resolve are the moiré.

> **fundamental beat:  p_beat = p₁p₂ / |p₁ − p₂|**
> **with rotation:     p_beat = p₁p₂ / √(p₁² + p₂² − 2p₁p₂ cos α)**
> **equal pitch:       p_beat = p / (2 sin(α/2))**

The important framing is that **moiré is an amplifier**. A 63.5 µm carrier
against a 66.07 µm front gives a 1635 µm beat: the beat is `p/Δ = 24.7×` the
pitch difference, so it magnifies a pitch error by 25. That is the same
mathematics as the vernier in §1.5 and as moiré magnification, and it cuts both
ways — it is a free metrology instrument *and* the reason the lid's band spacing
is the most process-sensitive number in the design (§3.5).

### 1.3 Parallax — the only thing that needs two plies

Two patterns separated by the substrate. Tilting changes their relative phase:

> **shift(θ) = t · tan(asin(sin θ / n))**  ≈ 17.22 µm/° at 1.5 mm soda lime

That is the entire argument for a second plate. Everything else on this plate is
single-layer.

### 1.4 Halftone — the only source of continuous tone

Tone from the *width* of a gold band at fixed pitch. Two bounds meet:

> **steps ≤ p / 2 µm** (the finest band must clear the litho floor)
> **p ≤ 43.5 µm** (the eye's 87 µm cell must span two periods or the screen
> stops averaging and you see lines)

which is why the design sits at 44 µm and 22 levels. Coverage is linear in area
and the eye re-encodes, so matching an sRGB source needs
`coverage = linearize(source)`; feeding sRGB in directly renders a 0.25 midtone
as 0.54.

### 1.5 (Not a mechanism) Metrology

Verniers, CD ladders, duty ladders, polarity witnesses. These do not appear on
the box. They are what turn the other four sections from opinions into numbers.

---

## 2. The structural result: what actually needs the bond

This decides the area budget, so it comes before the experiments.

For binary masks, **a single-layer superposition is optically identical to a
two-layer stack at zero gap.** The complement of a union is the intersection of
complements:

> **1 − (A₁ ∪ A₂) ≡ (1 − A₁)(1 − A₂)**

Draw both gratings on one plane and the transmission is *exactly* the product
you would get from two plies in contact. So the **static** moiré — the beat
geometry, the fringe spacing, the contrast — can be measured on a single layer
for free.

The gap buys exactly two things, and costs two:

| The gap gives | The gap costs |
|---|---|
| relative phase that moves with view — the fringes **travel** | registration: a lateral error is a phase error |
| depth, and the cross-layer illusions that come with it | **near-field decay** (below) |

### 2.1 The near-field limit — a hard constraint, previously unstated

A grating does not cast a sharp shadow across 1.5 mm. Self-imaging revives at
the Talbot distance `z_T = 2p²/λ`, and between revivals the shadow washes out.
Taking `z_T/4` as the point where a usable shadow survives:

| pitch | z_T | gap / z_T | verdict |
|---|---|---|---|
| 5 µm | 0.09 mm | 16.5 | **washed out** |
| 10 µm | 0.36 mm | 4.1 | **washed out** |
| 20 µm | 1.45 mm | 1.03 | **washed out** |
| 30 µm | 3.27 mm | 0.46 | degraded |
| 44 µm | 7.04 mm | 0.21 | intact |
| 63.5 µm | 14.7 mm | 0.10 | intact |
| 173 µm | 109 mm | 0.01 | intact |

> **Two-layer effects need a pitch of roughly 40 µm or coarser at a 1.5 mm gap.**

The shipping 63.5 µm carrier and 173 µm comb clear it comfortably. A 5 µm
colour grating is sixteen Talbot lengths away and could never work across the
gap — which is fine, because colour is single-layer by construction, but it
means **no two-layer effect may ever be designed at a fine pitch.** This is
worth a cell of its own (E-NF) because the constant `z_T/4` is a rule of thumb
and the real boundary is what we want.

### 2.2 The witness is quartz; the box is soda lime

A 5" mask blank is 0.090" = 2.29 mm of fused quartz, n = 1.4585. The box is
1.5 mm soda lime, n = 1.52. Bonding a diced witness pair therefore does **not**
reproduce the box's geometry:

| | thickness | n | parallax | nD/2t | swap at 173 um |
|---|---|---|---|---|---|
| box | 1.50 mm | 1.520 | 17.22 um/deg | **152** | 2.51 deg |
| witness pair | 2.29 mm | 1.4585 | 27.40 um/deg | **96** | 1.58 deg |

The witness is the **harsher** case on both counts: 1.6x the parallax per degree
and two-thirds the swap-to-visibility ratio. That is the useful direction - a
barrier whose lanes read invisible on the witness is safely invisible on the
box - but it means **the angles do not transfer, only the verdicts do.** Any
number read off a bonded witness cell must be rescaled by `n/t` before it is
compared with a box prediction, and the cells are labelled with the *witness*
angle for that reason.

It also shifts the near-field table in 2.1: at 2.29 mm the boundary moves from
about 40 um to about 50 um, so **E-NF is calibrated for the witness, not the
box.**

### 2.3 The plate may never be bonded

Bonding is a separate operation and this is a single-plate write. The result in
2 is what makes that survivable: the entire **static** moire programme - beat
geometry, rotation, the vector formula, harmonic beats, contrast - is
single-layer and costs nothing. Only *motion*, *sampling* and *registration*
genuinely need two plies, and those are confined to a clearly marked minority of
the area. If the bond never happens, about 80% of the plate still returns its
numbers.

---

## 3. Failure modes, ranked by what they would cost

| # | Failure | Consequence | Measured by |
|---|---|---|---|
| 3.1 | CD comes out fine/coarse of design | colour period wrong; finest gratings may not print at all | **M-CD**, **D-PER** |
| 3.2 | Duty biased off 0.50 | efficiencies wrong, tone wrong, **and a new moiré appears** (§2.4 below) | **M-DUTY** |
| 3.3 | Plies misregister | every parallax effect degrades or inverts | **M-VERN** |
| 3.4 | Bond gap off design | parallax angles wrong; near-field limit moves | **P-RULE** |
| 3.5 | Pitch error × 25 into the beat | lid band spacing visibly wrong | **B-BEAT** |
| 3.6 | Dot gain | picture too dark or too light | **H-WEDGE** |
| 3.7 | Source too wide in the room | colour washes to white | viewing protocol, **D-PER** |
| 3.8 | Screen beats with the carrier | banding across the photograph | **B-HARM** |
| 3.9 | Polarity inverted | everything is its own negative | **M-POL** |
| 3.10 | Fine pitch across the gap | two-layer effect simply absent | **E-NF** |

### 2.4 / 3.2 in detail — the moiré that only exists when the process is off

The halftone screen (44 µm) sits above the carrier (63.5 µm). Their beats:

| (m,n) | beat | at 300 mm | amplitude, duty 0.50 | duty 0.42 |
|---|---|---|---|---|
| (1,1) | 143 µm | 1.65′ | 0.1013 | 0.0951 |
| (2,3) | **559 µm** | **6.42′** | **0.0000** | **0.0059** |
| (1,3) | 41 µm | 0.47′ | 0.0338 | 0.0238 |
| (3,3) | 48 µm | 0.55′ | 0.0113 | 0.0060 |

The (2,3) beat is **6.4 arcmin — plainly visible** — and its amplitude is
*identically zero at 50% duty* because even harmonics of a square wave vanish
there. Bias the duty to 0.42 and it appears.

So a duty error does not merely dim the plate: it **conjures a half-millimetre
banding across the photograph that the nominal design does not have.** This is
the single best argument for the duty ladder, and it is why M-DUTY is read
before any picture is judged.

---

## 4. The experiments

Naming: `M-` metrology, `D-` diffraction, `B-` beat/moiré, `P-` parallax,
`H-` halftone, `E-` edge-of-envelope. Every cell's gold label carries its own
parameter value, not an index.

### 4.1 Metrology — read these first

| Cell | Structure | Sweep | Pass |
|---|---|---|---|
| **M-VERN** | 80/88 µm comb pairs, 4 corners | — | coincidence read to 1 µm; 11× amplification. Four corners so rotation separates from shift. *Bonded.* |
| **M-CD** | line/space pairs, iso **and** dense | 0.8 → 6 µm, 10 rungs | finest rung that resolves, in both environments. Iso-dense split is a real bias and the colour ladder assumes it is small |
| **M-DUTY** | one period, duty stepped | 0.30 → 0.70, 9 rungs, at 10 µm **and** 5 µm | the 0.50 rung shows **no second order**. If it does, read which way from 0.40 vs 0.60 |
| **M-POL** | solid square with a square hole, plus the word CLEAR | — | unmistakable at a glance. Catches an inverted write before anything else is interpreted |

### 4.2 Diffraction

| Cell | Structure | Sweep | Pass |
|---|---|---|---|
| **D-PER** | bare gratings, 50% duty | 2 → 20 µm, 12 rungs | hue at fixed geometry, and how wide the fan is. Read twice: under a lamp and under room light — the difference *is* the source-width result |
| **D-CHIRP** | period swept along the patch | 22 → 3 µm | a graded fan, not one flashing hue; and where it stops diffracting is a continuous CD read |
| **D-CROSS** | two orthogonal gratings, one layer | 2 periods | 2-D orders. Also the cheapest check that same-plane superposition behaves |
| **D-SWATCH** | **colour swatch matrix**, not portraits | base period × ladder spread, 4 × 3 | which base and which spread separate cleanly. Replaces eight portrait cells at a twentieth of the area |
| **D-BAND** | sub-grating inside a halftone band | 3 periods × tone held / not held | does a *banded* grating still diffract? Separates "the colour works" from "the picture works" |

### 4.3 Moiré — the largest block

| Cell | Structure | Sweep | Pass |
|---|---|---|---|
| **B-BEAT** | two pitches, same plane | beat 500 → 6000 µm, 5 rungs | count fringes against prediction. Sized to hold ≥ 3 fringes each, which is why these cells are large |
| **B-CONT** | as B-BEAT, duty swept | 0.25 / 0.50 / 0.75 | fringe contrast and mean brightness. Low duty is brighter at the same contrast |
| **B-ROT** | equal pitch, rotated | 1 / 2 / 4 / 8° | `p/(2 sin(α/2))`: 3.6 / 1.8 / 0.91 / 0.46 mm. Fringes run **along** the lines, not across |
| **B-VEC** | pitch **and** angle together | 3 × 3 matrix | the general vector formula `|k₁ − k₂|`, which the repo already uses. Cheap to check, expensive to get wrong |
| **B-HARM** | 44 vs 63.5 µm, duty stepped | 0.42 / 0.50 / 0.58 | **the 0.50 cell should show no 559 µm banding and the others should.** §3.2 |
| **B-SCREEN** | halftone screen over a carrier | screen angle 0 / 45 / 90° to the carrier | the 1.65′ (1,1) beat, and how far the perpendicular screen suppresses it |
| **B-MAG** | pinhole array over motif array | M = 10 / 30 | `M = p_r/(p_r − p_s)`. *Bonded* — sampling genuinely needs two planes |
| **B-MOVE** | one B-BEAT rung, **bonded** | — | the paired experiment: same geometry as its single-layer twin, so the only difference is whether the fringes **travel**. Band velocity ≈ (beat/p) × 17.22 µm/° ≈ 443 µm/° |

### 4.4 Parallax — bonded pairs

| Cell | Structure | Sweep | Pass |
|---|---|---|---|
| **P-SWAP** | barrier image switch, quarter-period registered | comb 100/173/250/350 µm | swap at 1.45 / 2.51 / 3.63 / 5.08°, lanes at 0.57 / 0.99 / 1.43 / 2.01′. Where does the barrier become visible? `θ_swap/α = nD/2t = 152`, pitch cancels |
| **P-SCAN** | N-phase kinegram | N = 2 / 4 / 6 | crests travel one way. Wants N× the registration of P-SWAP |
| **P-RULE** | fine comb on one ply, index line on the other | — | reads `t·n` directly by tilt. The gap metrology the whole parallax section rests on |

### 4.5 Halftone

| Cell | Structure | Sweep | Pass |
|---|---|---|---|
| **H-WEDGE** | 16-patch tone step wedge | at 20 / 44 / 60 µm screens | measured coverage vs designed. **This is the dot-gain number**, and it inverts straight into the prep's `gain`. Objective where a portrait is not |
| **H-ACU** | screen with no image, pitch swept | 20 → 80 µm | at what pitch do you *see the lines*? The 43.5 µm bound is a calculation; this is the measurement |
| **H-PORT** | the three colour variants | plain / hue / zones, 12 mm | the one genuinely subjective call: which goes on the lid |
| **H-SCALE** | one variant at three sizes | 4 / 8 / 16 mm | where a halftone stops being a thumbnail |

### 4.6 Edge of envelope

| Cell | Structure | Sweep | Pass |
|---|---|---|---|
| **E-NF** | two-layer moiré, pitch swept | 20 / 30 / 44 / 64 / 100 µm | **where the near-field shadow dies.** §2.1 predicts ~40 µm; this measures it. *Bonded* |
| **E-DENSE** | fine grating in isolated vs dense surround | 3 pitches | iso-dense CD bias, which sets whether the hue ladder's *ratios* survive |

---

## 5. Area budget

Usable area is 119 × 119 mm ≈ 14,200 mm²; shelf packing realises about 80%.

| Block | Area | Share | Why |
|---|---|---|---|
| Moiré, single-layer | ~3,200 mm² | 23% | fringes are millimetres, so cells must be large enough to hold three of them; and it is the mechanism with the most unmeasured parameters |
| Moiré + parallax, **bonded** | ~2,300 mm² | 16% | each cell is written twice, front die and back die. Everything the bond is actually needed for |
| Diffraction (D-) | ~1,600 mm² | 11% | small cells; a grating needs only enough area to fill the pupil |
| Halftone (H-) | ~1,150 mm² | 8% | of which **portraits are 770 mm², 5% of the plate** — against 40% before |
| Metrology (M-) | ~700 mm² | 5% | small, high value |
| Edge of envelope (E-) | ~300 mm² | 2% | |
| **content** | **~9,250 mm²** | **65%** | |
| Packing overhead, streets, margin | ~4,950 mm² | 35% | shelf packing realises about 80% of a row; 4 mm edge exclusion all round |

Single-layer content is about **80% of the total**, which is the point of §2.3.

---

## 6. Protocol

1. **M-POL.** Ten seconds. If the polarity is inverted nothing else means what
   it appears to mean.
2. **M-CD**, **M-DUTY**. Microscope. These two numbers gate the colour design
   and are the inputs to the prep's dot-gain correction.
3. **M-VERN** on a bonded pair. Decides whether §4.4 is worth pursuing at all.
4. **D-PER** and **D-SWATCH**, under a lamp *and* under diffuse room light. The
   difference between the two readings is the source-width result and it is not
   obtainable any other way.
5. **B-** block, single-layer first. Count fringes, compare with prediction,
   and back out the true pitch error from `p/Δ`.
6. **B-MOVE** and the **P-** block, bonded. Everything here is about motion, so
   it is read by rocking the part, not by looking at it.
7. **H-WEDGE**, then the portraits last — by then you know the dot gain, and
   the subjective call is made with the objective numbers already in hand.

---

## 7. What this plate does not test

Stated so nobody assumes otherwise:

* **Phase gratings.** Everything here is binary amplitude. An etched depth would
  be a real colour control (the etch is wavelength-dependent) and would roughly
  quadruple first-order efficiency, but it needs a process step this build does
  not have.
* **The bond itself** — adhesive thickness uniformity, index, voids. P-RULE
  measures the gap you got; it does not tell you whether it is repeatable.
* **Durability.** No abrasion, adhesion or environmental exposure.
* **Anything at the real viewing distance of a held object.** All subtense
  numbers assume 300 mm; a ring box is picked up and examined at 200 mm or
  closer, which makes every structure ~1.5× more visible than these figures say.
  That is a deliberate margin, not an oversight, but it is a margin of 1.5 and
  not more.
