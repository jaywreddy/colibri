# Vendored reference images — sources & licenses

Every file in this directory is used **only** as a tracing reference for
`motifs/trace.py` and must be public-domain / CC0 (this repo ships publicly).
Record each new reference here with its source URL and license.

| File | Subject | Source | Author | License |
|------|---------|--------|--------|---------|
| `capybara_skye_source.png` | Capybara (*Hydrochoerus hydrochaeris*) side profile, faces left | PhyloPic — https://www.phylopic.org/images/9c234021-ce53-45d9-8fdd-b0ca3115a451 (source: https://images.phylopic.org/images/9c234021-ce53-45d9-8fdd-b0ca3115a451/source.png) | Skye M | CC0 1.0 (Public Domain Dedication) |
| `capybara_traver_source.svg` | Capybara (*Hydrochoerus hydrochaeris*) side profile | PhyloPic — https://www.phylopic.org/images/097336ef-db5f-4edc-a2e4-2c29a10aa512 (source: https://images.phylopic.org/images/097336ef-db5f-4edc-a2e4-2c29a10aa512/source.svg) | Steven Traver | CC0 1.0 (Public Domain Dedication) |
| `quill_pen_bwcny_source.png` | A single writing feather quill (grey feather, near-white ground), full length with curved rachis, asymmetric vane, tapering bare shaft to a nib | Wikimedia Commons — https://commons.wikimedia.org/wiki/File:Quill_pen.PNG (file: https://upload.wikimedia.org/wikipedia/commons/8/87/Quill_pen.PNG) | BWCNY | Public domain (author's PD release: "grants anyone the right to use this work for any purpose, without any conditions") |

## Notes
- **In use:** `capybara_skye_source.png` is the traced reference for
  `lab/capybara.py` (the `capybara-scanimation` back face). It was chosen over
  the Traver SVG for its more naturalistic proportions (blocky muzzle, deep
  barrel, correct leg stance) and because a clean alpha raster feeds the
  `trace.py` alpha path directly.
- **Backup:** `capybara_traver_source.svg` is kept as an alternate CC0 reference.
- **In use:** `quill_pen_bwcny_source.png` is the traced reference for the quill in
  `lab/gear_quill.py::quill_book_silhouette` (the `gear-quill-switch` historian
  emblem). It replaces the earlier hand-drawn vane, which read as a leaf; the PD
  photo-illustration traces into a genuine writing quill (asymmetric barbed vane,
  curved rachis, nib). The source has no usable alpha (opaque grey-on-white), so
  the trace uses the luma path (`threshold≈0.92`).
- CC0 1.0 imposes no attribution requirement, but authors are credited here as a
  courtesy and for provenance.
- PhyloPic content licensing: https://www.phylopic.org/ (each image page lists
  its individual license; both entries above are CC0).
