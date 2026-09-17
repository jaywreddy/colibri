Prepared side-plate photographs, 1400 px square, sRGB.

<name>.png          the cropped (and for night-group, matted and ghost-grounded) source
<name>.subject.png  0..255 matte of the people (rembg u2net), used to push the edge fade outward
<name>.fade.png     (night-group only) the fade field its ground was built with
paris.png           the right tenth of the square dropped and the left tenth extended (the
                    mirrored edge band, blurred progressively toward the edge, slightly
                    darkened) so the couple sits at mid-frame with their full bodies

Produced by tools/dev/render_side_plate.py + the fade study; originals live in photos/.
{
 "beach": {
  "source": "PXL_20240807_233638672.MP",
  "prepared": true
 },
 "sunset": {
  "source": "IMG_1827~2",
  "prepared": true
 },
 "garden": {
  "source": "PXL_20240803_232128378.MP",
  "prepared": true
 },
 "paris": {
  "source": "IMG_6584",
  "prepared": true
 },
 "night-group": {
  "source": "IMG_1290-EDIT",
  "prepared": true
 },
 "porch-group": {
  "source": "signal-2026-01-05-11-40-52-562",
  "prepared": true
 }
}

sunset.png (and its .subject.png) is a second crop of the IMG_1827~2 square: x 0.19-0.81, y 0.05-0.67,
resampled back to 1400 px, so the couple fills the frame — the sun and the coast are gone and the
faces are 1.6x larger (the picture is 12.5 mm wide on the box).

Tracking (2026-09-16): the six photographs (`<name>.png`) are PERSONAL and are
NOT in git — they were purged from the history and are ignored by
`.gitignore`. They live on disk here and in `photos/prepared/` (also ignored);
a fresh clone needs them copied in before the photo faces or the plate can be
built. The mattes (`.subject.png`), the fade field (`.fade.png`) and the colour
plans (`.colour.json`) are derived and stay tracked.

Tests: the tests that screen a real picture (`tests/test_plates_and_boxes.py`
photo tests, marked `requires_photo("beach")`; `tests/test_witness.py`'s
portrait complement test, marked `requires_source_photo`) SKIP when the asset
is absent, with the reason printed (`pytest -rs`) — so a fresh clone and CI
see "skipped: personal asset", not a failure. The band builder's complement
property is still exercised everywhere on a synthetic image
(`test_the_halftone_inverse_is_a_complement_on_any_image`).
