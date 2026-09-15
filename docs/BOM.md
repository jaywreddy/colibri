# Ring Box — Bill of Materials (bonded-pair build)

> **Current build (2026-09-15), superseding the numbers below where they
> differ:** 32 × 32 × 35 mm box, six faces, each a bonded pair of **2.25 mm
> fused-quartz** plies (4.5 mm walls; 5″ × 0.090″ chrome + resist mask blank,
> the same stock for the written outer ply and the bare inner ply). Only the
> OUTER ply of each face is written; the mask carries 11 dies in a **dicing
> grid** (`app.export_witness`, `DICING.md`) and is cut on a saw — the first
> plate was hand-cleaved and broke. Copper foil is **3/8″** (decision 7 of
> `docs/decisions.md`, decision 7 — extracted from `docs/plan.md`): the 6.75 mm three-ply stepped edge leaves
> 1/4″ tape with no fold. Two blanks per box: one written, one bare.

Target build (original plan): ~30.5 × 30.5 × 33.6 mm upright box, six faces, each a BONDED PAIR
of 1.5 mm soda-lime plies (3 mm walls, nested-shell bevel-step corners —
`assembly.bonded_cut_list`). Copper-foil/solder assembly, brass tube-and-rod
hinge, UV-cure optical bond.

Prices/links verified live 2026-08-29 (background research run); everything
listed ships in days, US. Re-verify at order time.

## 1 · Substrate (the litho consumable)

The plate assumes a 5″ × 5″ chrome (+ resist) coated blank, written on the
maskless aligner, wet-etched, DICED into the plies (dicing saw, edge cuts).

| Item | Vendor | Price | Lead | Notes |
|---|---|---|---|---|
| **5″×5″ chrome+resist photomask blank (1.5 mm soda lime)** — baseline | Telic (US) / Nanofilm (US) / JD Photo Data (UK) | ~$20–60/blank (Nanofilm ~$20–25 at standard thickness; Telic quote; [JD blanks](https://jd-photodata.co.uk/custom-photomask-blank.html)) | days (US) / 1–2 wk (UK) | Standard mask stock **is** 1.5 mm soda lime — the baseline needs nothing custom. Buy ≥3 (practice cleaves + a rewrite). |
| 1.1 mm soda-lime 100×100 mm, 50-pack, uncoated | [MSE Supplies](https://www.msesupplies.com/products/1-1-mm-uncoated-soda-lime-glass-substrates) | $136.95/50 (~$2.74/pc) | ships next business day | Thinner drop-in (×0.73 on every derived pitch/angle) — needs a coating step (Telic/Nanofilm custom-coat, or in-house Cr sputter+spin). |
| 0.5 mm JGS1 fused-silica 4″×4″ DSP, uncoated | [MSE Supplies](https://www.msesupplies.com/products/mse-pro-4-x-4-inch-square-shaped-jgs1-ultraviolet-uv-grade-fused-silica-wafer-500-um-dsp) | $127.95/pc | ships next business day | True quartz at the ORIGINAL 500 µm design gap — restores the native 60 µm barrier pitch. Expensive per box; needs coating; 4″ (smaller box). |
| 0.5/0.7/1.0 mm recycled soda-lime mask plates 4″/5″ | [JD Photo Data](https://www.jd-photodata.co.uk/5-x-5-x-0060-soda-lime-plate.html) | $16–30/pc | UK, 1–2 wk | Cheapest thin squares; “recycled” = cosmetic scratches possible. Uncoated. |

**Thickness is a first-class parameter everywhere** (`GlassSpec.thickness_um`,
`--plate-thickness`): barrier/scanimation fab periods derive from t/n
(`plates.fab_center_period_um`), foil margins and the nested cut list follow,
and the renderer previews it. Changing supplier = changing one number.

## 2 · Bonding (ply pairs)

| Item | Vendor | Price | Lead | Notes |
|---|---|---|---|---|
| **Norland NOA 61, 1 oz** — primary | [Edmund #37-322](https://www.edmundoptics.com/p/1-oz-application-bottle-of-noa-61/4209/) | $38.00 | in stock, 2–4 d | n=1.56, cures 320–380 nm, 300 cps — wicks a tight glass-glass bondline by capillarity; reposition freely until UV. |
| NOA 68, 1 oz — optional 2nd bottle | [Edmund](https://www.edmundoptics.com/p/1-oz-application-bottle-of-noa-68/4178/) | $38.00 | in stock | n=1.54, 5000 cps, cures FLEXIBLE — better where the solder frame thermal-cycles the 3 mm pane. |
| Loctite AA 349, 50 mL — Prime-fast alternative | [Amazon](https://www.amazon.com/Loctite-Impruv-Amber-Acrylic-Adhesive/dp/B00M0QG844) | $48.11 | Prime 1–2 d | Real glass/metal light-cure acrylic if Edmund’s ground shipping is too slow. |
| **LIGHTFE UV301D 365 nm / 3000 mW lamp** | [Amazon](https://www.amazon.com/LIGHTFE-Flashlight-blacklight-UV301D-Max-3000mW/dp/B07BC8L581) | $24.99 | Prime 1–2 d | Curing-grade 365 nm (matches NOA’s band) — not a novelty blacklight. |
| Bondic kit (glue + built-in LED) | [Amazon](https://www.amazon.com/Bondic-BONDING-SK001-Plastic-Complete/dp/B01MR7J5ZX) | $13.95 | Prime | Bench backup / spot tacks only — uncharacterized optics. |

Bond procedure: stack chrome-down per the export manifest, align on the moiré
vernier combs (80/88 µm pair → 11× amplification, 880 µm beat) + the live
effect, wick NOA61 from the edges, re-check, cure. Fully reversible until the
lamp comes on.

## 3 · Hinge (brass tube-and-rod, back top edge)

| Item | Vendor | Price | Lead | Notes |
|---|---|---|---|---|
| **K&S 1/16″ brass rod (3-pack)** — pin | [Amazon B0044F6QFK](https://www.amazon.com/16-Dia-12-Brass-Rod/dp/B0044F6QFK) | ~$5–8 | Prime 1–2 d | 0.0625″ OD — telescopes inside the 3/32″ tube below. Solderable to the foil seam. |
| **K&S 3/32″ brass tube (3-pack)** — barrel | [Amazon B079VL3XD1](https://www.amazon.com/Round-Brass-Tube-014-Wall/dp/B079VL3XD1) | ~$6–9 | Prime 1–2 d | 0.066″ ID / 0.014″ wall — the exact pairing every “stained-glass box hinge” kit is made of. Matches the modeled 2.4 mm tube / 1.6 mm rod. |
| K&S #3400 telescoping assortment (12 tubes) | [Amazon](https://www.amazon.com/Precision-Metals-Telescopic-Tubing-Assortment/dp/B0GMRT28FZ) | ~$25–33 | Prime | Only if you want to dial barrel/pin clearance across sizes. |
| Delphi 3/32″ tube 5-pack + rods (stained-glass supplier) | [Delphi](https://www.delphiglass.com/dimensional-projects/boxes-clocks/3-32-brass-tubes-5-pack) | ~$18–20 | 3–5 d | Same brass, pre-matched for box hinges, slower/pricier than raw K&S. |

Skip ready-made barrel/piano hinges (Rockler/McMaster) — brass and solderable
but built for mortise/screw mounts, more rework than tube+rod.

## 4 · Cleaving tools (see verdict below)

| Item | Vendor | Price | Lead | Notes |
|---|---|---|---|---|
| **Swpeet 3-pc kit** (oil cutter + running pliers + grozer) | [Amazon](https://www.amazon.com/Swpeet-Running-Breaker-Professional-Stained/dp/B07JMWMC2Z) | $25.68 | Prime 2 d | The whole cleave workflow in one order. |
| **Luminbo 12″ plastic L-square** | [Amazon](https://www.amazon.com/Plastic-Cutting-Square-Stained-Glass/dp/B01K1OXDDA) | $13.99 | Prime 1 d | Plastic won’t scratch chrome; squares the 27–31 mm scores. |
| TOYO TC-600 pistol-grip oil cutter — upgrade | [Amazon](https://www.amazon.com/Toyo-Pistol-Cutter-Assorted-Colors/dp/B016ABHVWW) | $36.95 | Prime 2 d | The stained-glass community’s consistent-wheel pick for repeat scores. |
| Wet tile saw (VEVOR 7″ ~$74 / QEP 700XT ~$100) | Amazon | — | Prime | FALLBACK only, if cleave rejects run high and the friend’s dicing saw is unavailable. |

### Cleave vs dice verdict

**Hand cleaving 1.5 mm soda lime into ~27–31 mm squares is feasible** — this is
bread-and-butter stained-glass work. Expect ±0.3–0.5 mm on straight scores
(±0.5–1 mm worst case with some corner-chip rejects; small squares are the
hardest geometry — budget practice pieces). The design absorbs this by
construction: optical registration happens at BOND time (verniers + live
moiré), never at the cut edge, and the copper-foil/solder seams swallow
±0.5 mm like they were designed to (they were).

Chrome-specific handling: **score the BARE glass side, never the chrome**
(flakes and gums the wheel); chrome face DOWN on felt for both score and snap
so the coating sits on the compression side. Practice the full motion on
coated scrap — coating adhesion at the break line is the untested variable.
The panelizer’s street width (1 mm) and corner L-ticks are sized for exactly
this workflow.

Call in the dicing-saw favor only if (a) the reject rate stays high after
practice, or (b) chips reach the foil keep-out. The 12-plate panel is one
afternoon on a dicing saw — a small favor — so it’s a genuine plan B, not a
requirement.

## 5 · Already-planned consumables (unchanged)

Copper foil tape **3/8″** (9.525 mm; the 2.25 mm bonded stepped edge consumes
3 plies = 6.75 mm of tape width and leaves a 1.39 mm fold per face — 1/4″ tape
is 0.4 mm short of the edge and `validate_bonded_assembly` rejects it), solder
+ flux, Cr etchant (Cr etch 1020 / CR-7), AZ 400K developer (CMU recipe: dose
125 on the MLA 150, AZ 400K:DI 1:4 80 s, Cr etch 80 s, Nanostrip 60 °C 20 min).

## Order-now shortlist (~$120 + blanks)

1. Swpeet kit + Luminbo square + TOYO — $77 (Prime)
2. K&S rod + tube — ~$14 (Prime)
3. NOA 61 + UV lamp — $63 (Edmund + Prime)
4. 3× 5″ chrome+resist blanks — quote Telic/Nanofilm (US, days)
