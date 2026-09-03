# Quick-turn photomask vendors for the 5″ plate

Researched 2026-09-03. Everything below is a list price or a vendor claim on
that date; re-quote at order time. The plate is 6,695 mm² of written area,
39.9 MB GDSII / 1 MB OASIS, everything ≥ 2 µm except the two M-CD strips,
whose finest rungs are 0.8 / 1.0 / 1.5 µm. **A 1–2 µm laser class writes the
whole plate; the 0.8 µm rung is best-effort at that tier, which is what it is
for.** Only a guaranteed sub-µm rung needs a 0.5 µm shop or e-beam, at
roughly double the cost and lead time.

## Tier 1 — published configurator, ~$830–1,080, 4 days

[JD Photo Data](https://www.jd-photodata.co.uk/catalog/product/view/id/198)
(UK). 5″ base $373.73 + quartz 0.090″ +$182.21 + Class 4 (1 µm) +$261.61 +
GDSII +$13.30 = **$831** at the 4-day schedule. 2-day +$93.43, same-day
+$373.73, 8-day −$37. Their listed sub-micron 5″ quartz plate is
[$1,075.97](https://jd-photodata.co.uk/catalogsearch/result/?404=1&q=photomask+special+material+photomasks+5+0090+qtz+lr+chrome+od3).
The only vendor found publishing prices; add international shipping.

## Tier 2 — US quick-turn laser shops, quote only, likely $600–1,500, 1–5 days

* [HTA Photomask](https://www.htaphotomask.com/faq.php) — 0.5 µm lines/spaces,
  1–3 day standard, most masks in 24 h, quick-turn often at no premium.
* [Photo Sciences](https://photosciences.com/faq/) — 5-day after plot
  approval, 0.5 µm smallest feature, quartz at 90 nm chrome.
* [Front Range Photomask](https://www.frontrangephotomask.com/pricing) — the
  budget shop; quartz is a premium; 1 µm corner radius (fine for gratings).
* [Advance Reproductions](https://advancerepro.com/photomasking/) — 0.75 µm,
  ±0.15 µm CD, quartz available.
* [UniversityWafer](https://www.universitywafer.com/photomasks.html) — 1 µm
  laser / 0.25 µm e-beam, 5–7 d standard, 48–72 h rush, 24 h super-rush;
  broker, add margin.

## Tier 3 — e-beam mask houses, likely $2,500–6,000+, 1–3 weeks

[Design Realized](https://design-realized.com/pages/optical-photomask),
[Benchmark Technologies](https://benchmarktech.com/photomasks/),
[Compugraphics](https://www.macdermidalpha.com/products/semiconductor-assembly/photomasks),
[Photronics](https://www.photronics.com/integrated-circuits-ic/leading-edge-advanced-photomasks/).
Only if the sub-µm CD rungs must be metrology rather than a where-does-it-break probe.

## Tier 4 — write it yourself at a nanofab: the expensive route

[Harvard CNS](https://cns1.rc.fas.harvard.edu/documents/2018/06/cns-consolidated-rate-sheet.pdf)
(rates effective 04/01/26): $225/h industry, $72.50/h non-Harvard academic.
Their [DWL66 write-time note](https://cns1.rc.fas.harvard.edu/facilities/docs/TM021_r1_0_DWL66%20write%20time%20document.pdf):
0.3 min/mm² for 1–5 µm features, 0.9 min/mm² below 1 µm, +25% overhead.

| head forced by | write time | industry | academic |
|---|---|---|---|
| 1–5 µm (ignore the 0.8 rung) | ~42 h | ~$9,400 | ~$3,000 |
| < 1 µm (honour it) | ~125 h | ~$28,000 | ~$9,100 |

A full-field 127 mm plate is where a DWL66-class tool loses to a commercial
writer by an order of magnitude. DIY only wins for small dies.

## Two strategic notes

**Soda lime, not quartz.** A 1.5 mm (0.060″) soda-lime blank saves the quartz
premium and its index (1.52) matches the box's soda lime, so dies cut from
the plate ARE box plies and the witness pair differs from the box in nothing.
Quartz is only needed for deep-UV exposure through the plate.

**Order at the 1 µm class and let the 0.8 µm rung fall where it may.** That
is the measurement; paying the sub-µm tier to guarantee a rung whose purpose
is to find the floor inverts the logic of the CD ladder.

Recommended path: quotes from HTA (fastest, 0.5 µm capable) and Front Range
(cheapest), with the JD configurator as the price anchor. Expect $600–1,100
and under a week.
