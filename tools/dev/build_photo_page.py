"""Build the side-photo review page: each candidate photograph, its crop and
screen preview, and the four colour treatments the pipeline can give it.

    uv run --directory backend python ../tools/dev/build_photo_page.py PHOTOS_DIR OUT_HTML [NOTES_JSON...]
"""
from __future__ import annotations

import base64
import html as H
import io
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "backend"))
from app.witness_geom import GLASS_MATERIAL, PLY_UM  # noqa: E402

photos = Path(sys.argv[1])
out = Path(sys.argv[2])
notes = {}
for nj in sys.argv[3:]:
    d = json.load(open(nj, encoding="utf-8"))
    items = d if isinstance(d, list) else d.get("photos", list(d.values()))
    for e in items:
        if isinstance(e, dict) and "file" in e:
            notes[Path(e["file"]).stem] = e
colour = json.load(open(photos / "photo_colour.json", encoding="utf-8"))
CSS = io.open(HERE / "witness_page.css", encoding="utf-8").read()


def b64(p: Path) -> str:
    return base64.b64encode(p.read_bytes()).decode()


def fig(p: Path, alt: str, cap: str) -> str:
    return (f'<figure><img class="preview" alt="{H.escape(alt)}" src="data:image/png;base64,{b64(p)}">'
            f'<figcaption>{cap}</figcaption></figure>')


def as_text(v) -> str:
    if isinstance(v, dict):
        return "; ".join(f"{k}: {as_text(x)}" for k, x in v.items() if x not in (None, False, ""))
    if isinstance(v, list):
        return "; ".join(as_text(x) for x in v)
    return H.escape(str(v))


ORDER = ["PXL_20240807_233638672.MP", "IMG_1827~2", "PXL_20240803_232128378.MP", "IMG_6584", "IMG_1290-EDIT",
         "signal-2026-01-05-11-40-52-562"]
READING = {
    "PXL_20240807_233638672.MP": "Faces are the only saturated thing in the frame, so <b>auto-zones</b> colours skin and nothing else (8% of the field); <b>hue</b> adds the sky as a blue rung and the shirt as a scatter. This is the one where zone-mapping earns its keep: a warm face against plain gold sea and sky.",
    "IMG_1827~2": "The sunset owns the hue map: sky and sea go to the red-orange end, his shirt to blue, and 77% of the field is coloured. <b>Hue-equalised</b> spreads that into the green rungs and looks busier, not better. The authored version would be: shirt blue, sun disc red, everything else plain.",
    "PXL_20240803_232128378.MP": "The dress is the picture. Hue colours the foliage with it (80% of the field) and the dress loses its identity; <b>auto-zones</b> at 41% keeps the dress and the leaves both. A zone plan that colours the dress alone and leaves the garden plain would be the strongest side of the six for colour.",
    "IMG_6584": "Grass and trees swamp the hue map at 71% coloured; auto-zones brings it to 20%, mostly the dress and the tower. Colour will not rescue the busy background here, segmentation will. Plain, with a segmented background, is the honest recommendation.",
    "IMG_1290-EDIT": "Four faces and the floral top give hue and hue-eq 83% coverage, all of it fragmentary at the eye cell. The mirror-padded top edge doubles the heads and must be replaced by a darkened extension before this is usable at all.",
    "signal-2026-01-05-11-40-52-562": "Backlit foliage colours 86% of the frame in hue mode; auto-zones halves it and still lands mostly on the trellis. With five faces at ~35 cells each this is a scene, and a scene wants plain gold.",
}

sections = []
for stem in ORDER + [c for c in (Path(e["file"]).stem for e in colour) if c not in ORDER]:
    entry = next((e for e in colour if Path(e["file"]).stem == stem), None)
    if entry is None:
        continue
    n = notes.get(stem, {})
    prep = photos / f"{stem}_prep.png"
    col = photos / entry["sheet"]
    st = entry["stats"]
    rows = "".join(
        f'<tr><td class="mono"><b>{k}</b></td><td class="n mono">{v["rungs_used"]}</td>'
        f'<td class="n mono">{v["coloured_fraction"]*100:.0f}%</td>'
        f'<td class="muted-cell">{"rungs spread over the picture" if v.get("hue_equalize") else ("saturated regions only, coarse patches" if k == "auto-zones" else ("every pixel its own hue" if k == "hue" else "control: no sub-grating"))}</td></tr>'
        for k, v in st.items())
    sections.append(f"""
<div class="group"><h2>{H.escape(stem)}</h2>
  <p>{as_text(n.get("subject", ""))}</p></div>
<div class="tbl-wrap"><table><tbody>
<tr><td><b>Crop</b></td><td class="mono">{tuple(round(c, 3) for c in entry["crop"])}</td><td class="muted-cell">x, y offset and side as fractions of the source width; the same convention the builder uses</td></tr>
<tr><td><b>Extension</b></td><td colspan="2">{as_text(n.get("extended", "none"))}</td></tr>
<tr><td><b>Segmentation</b></td><td colspan="2">{as_text(n.get("segmentation_recommended", ""))}</td></tr>
<tr><td><b>Risks</b></td><td colspan="2">{as_text(n.get("risks", ""))}</td></tr>
<tr class="hi"><td><b>Verdict</b></td><td colspan="2">{as_text(n.get("verdict", ""))}</td></tr>
</tbody></table></div>
{fig(prep, stem + " crop and screen", "Original with the crop, the square source, the tone the eye integrates at 87 µm (177 × 177 cells over the 15.4 mm art box), and a true 3 mm patch of the 44 µm / 22-level line screen.") if prep.exists() else ""}
{fig(col, stem + " colour treatments", "Top row: the period field each plan assigns — red is the long-period end of the ladder (6.0 µm), violet the short (4.15 µm), grey stays plain gold. Middle and bottom rows: the rendered plate at 0° and 1° of tilt under a lamp, specular gold plus first-order sheen, eye-cell integrated.")}
<div class="tbl-wrap"><table><thead><tr><th>treatment</th><th class="n">rungs</th><th class="n">coloured</th><th>what it does</th></tr></thead><tbody>{rows}</tbody></table></div>
<p>{READING.get(stem, "")}</p>
""")

BODY = f"""<title>Side Photos</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Archivo:wght@500;600;700&family=IBM+Plex+Mono:wght@400;500;600&family=Source+Serif+4:opsz,wght@8..60,400;8..60,600&display=swap">
<style>{CSS}
figure {{ margin: 18px 0 30px; }}
figcaption {{ font-size: 13px; color: var(--muted); margin-top: 8px; max-width: 76ch; line-height: 1.5; }}
</style>
<div class="wrap">
  <p class="eyebrow">Ring box &middot; side plates</p>
  <h1>Side Photos</h1>
  <p class="standfirst">
    Six candidate photographs for the two colour sides, each cropped to the 15.4 mm art box of a
    {PLY_UM/1000:g} mm {GLASS_MATERIAL} side plate and put through the four colour treatments the pipeline
    can apply: plain gold, hue-mapped, hue-mapped and equalised, and an automatic zone map that colours
    only saturated regions. The period fields show what each plan would engrave; the renders show what
    the eye would see under a lamp.
  </p>
  <div class="callout"><h4>How to read the sheets</h4>
    <p>The <b>period field</b> is the plan's decision per patch of the picture: which rung of the 12-step
    diffraction ladder (4.15 – 6.0 µm sub-grating) that patch gets, or none. A coloured patch prints at half
    the metal of a plain one (the sub-grating is 50% duty), so more colour means a lighter picture — about
    2.8 tone levels inside the coloured zones. The renders integrate over the 87 µm eye cell, so what you
    see is the picture, not the screen; the 3 mm patch on the crop sheet is the screen itself.</p>
    <p>An <b>authored zone plan</b> — like the reference portrait's, where the sweater, glasses and each
    flower colour are named — is what turns the auto-zones map into a design. The reading under each
    photo says what that plan would be.</p></div>
{''.join(sections)}
  <footer>Crops and prep by the photo agents (<code>photo_notes.json</code>); colour fields and renders by
  <code>tools/dev/render_photo_colour.py</code> with the witness previews' appearance model; page by
  <code>build_photo_page.py</code>.</footer>
</div>"""
out.write_text(BODY, encoding="utf-8")
print(f"wrote {out}  {len(BODY)/1024/1024:.2f} MB")
