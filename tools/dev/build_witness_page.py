"""Build the witness-plate page: the plan, with figures where the argument needs them.

Refuses to embed a render older than its renderer, so a stale PNG cannot reach
the page — two did once.
"""
import base64
import html as H
import io
import json
import os
import re
import sys
from collections import Counter, defaultdict

SP = sys.argv[1]
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'backend'))
from app.witness_geom import GLASS_MATERIAL, GLASS_N, PLY_UM  # noqa: E402
ROOT = r"C:/Users/jaywr/OneDrive/Desktop/optics"
PLAN = f"{ROOT}/docs/witness-physics-plan.md"
MAN = f"{ROOT}/backend/data/witness/witness-5in.json"
MAP = f"{ROOT}/backend/data/witness/witness-5in-map.svg"
OUT = f"{SP}/witness-5in.html"
RENDERERS = {
    "witness_swatch.png": "render_witness_figures.py", "witness_harmonic.png": "render_witness_figures.py",
    "witness_swap.png": "render_witness_figures.py", "witness_wedge.png": "render_witness_figures.py",
    "fig_stack.svg": "render_witness_figures.py", "fig_union.svg": "render_witness_figures.py",
    "fig_cell.svg": "render_witness_figures.py",
    "witness_moire.png": "render_moire_preview.py", "witness_moire_b.png": "render_moire_preview.py",
    "witness_variants.png": "render_witness_preview.py",
    "witness_plate.png": "render_witness_plate.py", "witness_die_top.png": "render_witness_plate.py",
    "witness_die_leaf.png": "render_witness_plate.py", "witness_die_portrait.png": "render_witness_plate.py",
    "witness_die_garland.png": "render_witness_plate.py",
    "validate_photo.png": "validate_dies.py", "validate_switch.png": "validate_dies.py",
    "validate_nearfield.png": "validate_dies.py",
}
# Figures of experiment blocks that were cut from the plate on 2026-09-10 are
# kept as the record of the first design; their renderer can no longer draw
# them from the plate, so they are exempt from the staleness rule.
ARCHIVED = {"witness_moire.png", "witness_moire_b.png", "witness_variants.png"}
for name, r in RENDERERS.items():
    fig_path, r_path = f"{SP}/{name}", f"{ROOT}/tools/dev/{r}"
    if not os.path.exists(fig_path):
        sys.exit(f"missing figure {name}: run tools/dev/{r}")
    if name not in ARCHIVED and os.path.getmtime(fig_path) < os.path.getmtime(r_path):
        sys.exit(f"stale figure {name} is older than tools/dev/{r}: re-run it")

man = json.load(io.open(MAN, encoding="utf-8"))
svg_map = re.sub(r"^.*?<svg", "<svg", io.open(MAP, encoding="utf-8").read(), flags=re.S)
md = io.open(PLAN, encoding="utf-8").read()
CSS = io.open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "witness_page.css"), encoding="utf-8").read()


def b64(name):
    return base64.b64encode(open(f"{SP}/{name}", "rb").read()).decode()


def svg(name):
    return io.open(f"{SP}/{name}", encoding="utf-8").read()


def fig(name, alt, caption):
    return (f'<figure><img class="preview" alt="{H.escape(alt)}" '
            f'src="data:image/png;base64,{b64(name)}"><figcaption>{caption}</figcaption></figure>')


def diagram(name, caption):
    return f'<figure class="diagram">{svg(name)}<figcaption>{caption}</figcaption></figure>'


# ---------------------------------------------------------------- markdown
def inline(t):
    t = H.escape(t)
    t = re.sub(r"`([^`]+)`", r"<code>\1</code>", t)
    t = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", t, flags=re.S)
    t = re.sub(r"(?<![*\w])\*(?!\s)(.+?)(?<!\s)\*(?![*\w])", r"<em>\1</em>", t, flags=re.S)
    return t.replace("\\|", "|")


def md_to_blocks(src):
    out, lines, i = [], src.split("\n"), 0
    while i < len(lines):
        ln = lines[i]
        if ln.startswith("|") and i + 1 < len(lines) and set(
                lines[i + 1].replace("|", "").strip()) <= set("-: "):
            def split(row):
                return [c.strip() for c in re.split(r"(?<!\\)\|", row.strip().strip("|"))]
            head = split(ln)
            i += 2
            body = []
            while i < len(lines) and lines[i].startswith("|"):
                body.append(split(lines[i]))
                i += 1
            th = "".join(f"<th>{inline(c)}</th>" for c in head)
            tr = "".join("<tr>" + "".join(f"<td>{inline(c)}</td>" for c in r) + "</tr>" for r in body)
            out.append(("table", f'<div class="tbl-wrap"><table><thead><tr>{th}</tr></thead><tbody>{tr}</tbody></table></div>'))
            continue
        m = re.match(r"^(#{1,4})\s+(.*)", ln)
        if m:
            out.append((f"h{len(m.group(1))}", inline(m.group(2))))
            i += 1
            continue
        if ln.startswith(">"):
            buf = []
            while i < len(lines) and lines[i].startswith(">"):
                buf.append(lines[i].lstrip("> ").rstrip())
                i += 1
            out.append(("quote", '<div class="eq">' + "<br>".join(inline(b) for b in buf) + "</div>"))
            continue
        if re.match(r"^\s*[*-]\s+", ln):
            items = []
            while i < len(lines) and lines[i].strip():
                if re.match(r"^\s*[*-]\s+", lines[i]):
                    items.append(re.sub(r"^\s*[*-]\s+", "", lines[i]).strip())
                else:
                    items[-1] += " " + lines[i].strip()
                i += 1
            out.append(("ul", "<ul>" + "".join(f"<li>{inline(b)}</li>" for b in items) + "</ul>"))
            continue
        if re.match(r"^\s*\d+\.\s+", ln):
            items = []
            while i < len(lines) and lines[i].strip():
                if re.match(r"^\s*\d+\.\s+", lines[i]):
                    items.append(re.sub(r"^\s*\d+\.\s+", "", lines[i]).strip())
                else:
                    items[-1] += " " + lines[i].strip()
                i += 1
            out.append(("ol", "<ol>" + "".join(f"<li>{inline(b)}</li>" for b in items) + "</ol>"))
            continue
        if ln.strip() == "---":
            out.append(("hr", "<hr>"))
            i += 1
            continue
        if not ln.strip():
            i += 1
            continue
        buf = []
        while i < len(lines) and lines[i].strip() and not lines[i].startswith(("#", "|", ">", "---")) \
                and not re.match(r"^\s*([*-]|\d+\.)\s+", lines[i]):
            buf.append(lines[i].strip())
            i += 1
        out.append(("p", f"<p>{inline(' '.join(buf))}</p>"))
    return out


blocks = md_to_blocks(md)

# ---------------------------------------------------------------- as-built numbers
cells = man["cells"]
wr, n_by = Counter(), Counter()
def _wr(c):
    return c["w_mm"] * c["h_mm"] + (c.get("back_w_mm", c["w_mm"]) * c.get("back_h_mm", c["h_mm"]) if c["two_layer"] else 0)
for c in cells:
    wr[c["block"]] += _wr(c)
    n_by[c["block"]] += 1
tot_wr = sum(wr.values())
prod_wr = wr["production"]
exp_wr = tot_wr - prod_wr
two_wr = sum(_wr(c) for c in cells if c["two_layer"] and c["block"] != "production")
port = sum(c["w_mm"] * c["h_mm"] for c in cells if c["cid"].startswith(("PORT", "SZ")))
usable = man["layout"]["height_available_mm"] ** 2
lay, gds = man["layout"], man["gds"]
BLOCK_NAME = {"production": "Production dies", "moire": "Moiré", "diffraction": "Diffraction",
              "parallax": "Parallax", "halftone": "Halftone", "metrology": "Metrology"}
BLOCK_WHY = {
    "production": f"four box faces as plies: lid F+B, front F+B, two colour sides F only — the box's own {PLY_UM/1000:g} mm {GLASS_MATERIAL}",
    "moire": "fringes are millimetres, so cells must be large to hold five of them; and it had the least evidence behind it",
    "parallax": "each cell is written twice, front die and back die",
    "diffraction": "a grating needs only enough area to fill the pupil",
    "halftone": "wedges and the acuity ladder; the portraits are the two sides",
    "metrology": "small, and read first",
}
budget_rows = "".join(
    f'<tr><td><b>{BLOCK_NAME[b]}</b></td><td class="n mono">{n_by[b]}</td>'
    f'<td class="n mono">{wr[b]:,.0f}</td><td class="n mono">{wr[b]/tot_wr*100:.0f}%</td>'
    f'<td class="muted-cell">{BLOCK_WHY[b]}</td></tr>'
    for b in sorted(wr, key=lambda k: -wr[k]))
n_port = sum(1 for c in cells if c["cid"].startswith(("PORT", "SZ")))
BUDGET = f"""
<div class="tbl-wrap"><table>
<thead><tr><th>Block</th><th class="n">cells</th><th class="n">written mm²</th><th class="n">share</th><th>why</th></tr></thead>
<tbody>{budget_rows}
<tr class="hi"><td><b>total written</b></td><td class="n mono">{len(cells)}</td><td class="n mono">{tot_wr:,.0f}</td><td class="n mono">100%</td>
  <td class="muted-cell">{tot_wr/usable*100:.0f}% of the {usable:,.0f} mm² usable field; {lay['height_used_mm']:.1f} of {lay['height_available_mm']:.0f} mm of height</td></tr>
<tr><td>experiments that need a bond</td><td class="n mono">{sum(1 for c in cells if c['two_layer'] and c['block'] != 'production')}</td><td class="n mono">{two_wr:,.0f}</td><td class="n mono">{two_wr/exp_wr*100:.0f}%</td>
  <td class="muted-cell">of the experiment area; the other {100-two_wr/exp_wr*100:.0f}% returns its numbers without one</td></tr>
<tr><td>portrait cells</td><td class="n mono">{n_port}</td><td class="n mono">{port:,.0f}</td><td class="n mono">0%</td>
  <td class="muted-cell">none — the six photo side dies are the portraits, at a 13.3 mm art box</td></tr>
</tbody></table></div>
<div class="platewrap" style="background:#0d1113;border-radius:3px;padding:10px;overflow-x:auto;margin:18px 0 8px">{svg_map}</div>
<p class="dim" style="margin:0 0 26px">The plate as packed, from the manifest. Colour is block; gold is the four production dies; dashed outlines are back dies; the shaded band is the bonded experiments. Rows are packed by height with pockets and columns beside the tall dies, so a ladder reads left to right and a family may wrap.</p>
"""

# ---------------------------------------------------------------- cell tables (appendix)
by_block = defaultdict(list)
for c in cells:
    by_block[c["block"]].append(c)


def units(s):
    return s.replace(" um", " µm").replace("um,", "µm,").replace(" deg", "°")


tabs = []
for blk in sorted(wr, key=lambda k: -wr[k]):
    rows = []
    for c in sorted(by_block[blk], key=lambda c: (c["axis"], c["cid"])):
        two = '<span class="two">bonded</span>' if c["two_layer"] else ""
        rows.append(f'<tr><td class="mono"><b>{H.escape(c["label"])}</b>{two}</td>'
                    f'<td>{H.escape(units(c["title"]))}</td>'
                    f'<td class="n mono">{c["w_mm"]:g}&times;{c["h_mm"]:g}</td>'
                    f'<td class="dim">{H.escape(units(c["level"]))}</td></tr>')
    tabs.append(f"""<h3>{BLOCK_NAME[blk]} &middot; {n_by[blk]} cells &middot; {wr[blk]:,.0f} mm²</h3>
<div class="tbl-wrap"><table><thead><tr><th style="width:26%">Etched label</th><th>Cell</th><th class="n">mm</th><th style="width:22%">Level</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table></div>""")

FAB = f"""
<div class="tbl-wrap"><table><tbody>
<tr class="hi"><td><b>GDSII</b></td><td class="mono n">{gds['size_mb_by_format'].get('gds',0):.1f} MB</td><td><code>witness-5in.gds</code> — the primary deliverable</td></tr>
<tr><td>OASIS</td><td class="mono n">{gds['size_mb_by_format'].get('oas',0):.2f} MB</td><td><code>witness-5in.oas</code>, the same geometry; verified shape-for-shape and to the last database unit of area</td></tr>
<tr><td>Polarity</td><td class="mono n">clear data</td><td>darkfield plate (§0): the file holds the openings where chrome comes off. <b>M-POL</b> is the witness; metal + clear is checked to tile every cell box. The production dies are inverted inside their own rectangles by a klayout Region boolean and decomposed to trapezoids (no polygon carries a hole)</td></tr>
<tr><td>Written clear</td><td class="mono n">{sum(c['n_polys'] for c in cells if c['block']=='production'):,}</td><td>trapezoids over the four dies; every die's data bbox equals its outline to the database unit</td></tr>
<tr><td>Array references</td><td class="mono n">{gds['n_array_instances']:,}</td><td>every periodic structure — colour sub-gratings, ladders, swatches, combs — over {gds['n_unit_cells']:,} unit cells</td></tr>
<tr><td>Flattened</td><td class="mono n">{man['flat_rect_count']:,}</td><td>shapes if every array is expanded; <code>--flat</code> writes it that way</td></tr>
<tr><td>gdsfactory hierarchy</td><td class="mono n">31 MB</td><td><code>--writer gf</code>: <code>@gf.cell</code>-cached (width, height) cells turn 467k halftone rectangles into references. Measured 39.9 → 31.2 MB in 104 s; the ~470k references are the floor no hierarchy removes</td></tr>
</tbody></table></div>"""

# ---------------------------------------------------------------- assemble
AFTER = {
    "1.2 Moiré": fig("witness_moire.png", "beat and rotation ladders, simulated from the emitted geometry",
                      "Top: pitch beat at five spacings, at the widths built — fringes across the lines. Bottom: rotation at four angles — fringes along the lines. Each panel rasterises the rectangles the writer emits and averages over the 87 µm eye cell, so the fringes are the geometry's, not the formula's."),
    "1.3 Parallax": diagram("fig_stack.svg", "The two plies and the line of sight. Head-on the slit straddles the A/B lane boundary (a 50/50 blend); a back shift of p/4 uncovers one lane class cleanly. The plies do not move; tilting the part moves where the refracted ray crosses the back ply."),
    "2.1 The union identity": diagram("fig_union.svg", "One plane carrying both gratings against two plies in contact. The transmission functions are identical, so the static moiré is measured single-layer. In reflection the same cross term appears with inverted brightness."),
    "2.4 The moiré the halftone carries": fig("witness_harmonic.png", "(2,3) beat modulation vs screen duty, three cases",
                      "Modulation of the 559 µm beat against the screen's local duty. Tone case in transmission (solid): zero at 0.50, 5.6% at 0.42, 15.6% at 0.65, 40% at 0.90. The built HARM cells realise the bias case (dashed): 3.5 / 0 / 6.7%. In reflection off gold (thin) the lid never exceeds 5.5%."),
    "4.2 Diffraction": fig("witness_swatch.png", "colour swatch matrix, simulated at four tilts",
                      "D-SWATCH simulated: specular gold plus first-order sheen at tilts of 0 / 0.5 / 1 / 2°, blue end left; the bottom row strips the specular to show the ordering. The spread sets how much of the visible the ladder covers. Δλ per degree of tilt is p(cos θ_out − cos θ_in): the renderer assumes a lamp about 65° off the normal with the eye near it, so the hue walks with tilt; with the lamp behind the viewer it would not. SW 4/* and SW 5/1.90 have their blue end below the 2 µm line and are on the plate to be seen failing."),
    "4.3 Moiré": fig("witness_moire_b.png", "harmonic, contrast and crossed cells, simulated",
                      "B-HARM at true contrast with a column-mean profile under each panel: the 559 µm component is in the profile at 0.42 and 0.58 and absent at 0.50 — the panels themselves are dominated by the 143 µm (1,1) beat. B-CONT with its transmission and reflection means. D-CROSS raw over a 100 µm window: at the eye cell a 5 µm lattice averages to a flat tone."),
    "4.4 Parallax": fig("witness_swap.png", "swap angle and lane visibility vs comb pitch",
                      "Swap angle (left) and lane subtense (right) against comb pitch, for the witness pair and the box. The 1′ eye-cell line is crossed at p = 174 µm at 300 mm: the 173 µm comb sits on the threshold and is 1.5′ at the 200 mm a box is held; SWAP 250 and 350 are visible barriers by construction."),
    "4.5 Halftone": fig("witness_wedge.png", "tone wedges and the linearisation curve",
                      "The three H-WEDGE strips rendered from the emitted rectangles and column-averaged to coverage, with the designed duty under each patch (repeats in orange at 20 µm, where only 10 levels fit). Below: coverage against source value — the linearisation curve, with the classic error marked."),
    "4.6 Edge of envelope": diagram("fig_cell.svg", "Anatomy of a cell as written: the drawn geometry is the clear data; the label is chrome at the bottom-left and carries the value."),
    "4.7 Production dies": (
        fig("witness_plate.png", "the written GDS, whole plate",
            "The plate as written, rendered from the GDS by klayout: light is data, i.e. where the chrome comes off. Top row: TOP F, TOP B, FRONT F, FRONT B; second row: LEFT F and RIGHT F, with the metrology ladders stacked in the column beside them. The dies read as light because a box face is mostly bare glass; the chrome that stays is the gold.")
        + fig("witness_die_top.png", "the lid pair, F and B",
              "DIE-TOP: outer ply F (left) with the foliage garland and the J+P monogram as chrome gratings in a clear field, inner ply B (right) with the uniform carrier. Both mirrored for the chrome-down stack; the 80 / 88 µm vernier combs and the tick-code ID sit in the fold band at identical stack coordinates.")
        + fig("witness_die_leaf.png", "4 mm of the lid's garland",
              f"Four millimetres of the lid's garland: each leaf is a chrome grating at its own angle (the carrier × 1.09), which beats against the B ply's carrier across {PLY_UM/1000:g} mm of glass. The gaps between leaves are clear — on the box, bare glass.")
        + fig("witness_die_portrait.png", "3 mm of the beach side's line screen",
              "Three millimetres of DIE-LEFT: the 44 µm line screen with the 4.15–6.0 µm colour sub-gratings inside the coloured bands. The die is mirrored, so the picture reads correctly through the glass once it is the box's outer ply.")
        + fig("witness_die_garland.png", "3 mm of the beach side's garland: 6 µm per-family diffractive leaf gratings on bare glass",
              "The garland at the same scale: each motif family is filled with a 6 µm, 50% grating at its own angle, fanned over the half-turn — flat gold at normal incidence, spectral colour at the family's diffraction angle under a lamp. There is no carrier on this ply; the picture itself has dissolved to bare glass, so the whole side is single-layer and needs no bond.")
        + fig("validate_nearfield.png", "near-field simulation across 2.25 mm",
              f"A3, the gate: angular-spectrum propagation through {PLY_UM/1000:g} mm of glass under an incoherent source, eye-cell integrated. Left: a garland at the 500 µm design pitch of 22 / 24 µm; middle: the garland as built at this glass's carrier; right: the monogram pair. The surviving fraction of the zero-gap fringe contrast is printed under each. Predicted by §2.2's Fresnel number, and the number that decided the garland pitch and cut the capybara.")
        + fig("validate_switch.png", "registration tolerance of the globe switch",
              f"A2: DIE-FRONT's own front and back metal through sim2d at t = {PLY_UM/1000:g} mm at this glass's comb. The design lanes are fixed and the back ply slid; at p/4 the two globes are a 50/50 blend at every tilt and beyond it they trade places. The verniers read to about 1 µm; the table in §4.7 has the separation at each error.")
        + fig("validate_photo.png", "tone check of the beach side",
              "A1: the emitted metal of DIE-LEFT accumulated exactly per 87 µm eye cell against the coverage the builder intends. Mean error 0.8 of 22 levels; the coloured bands, holding tone with a 50% sub-grating, print about 2.7 levels lighter than the photograph inside the zones — the cost of holding tone, measured on the real geometry.")
    ),
    "5. Area budget, as built": BUDGET,
}
VARIANTS = fig("witness_variants.png", "three portrait treatments at three tilts",
               "The three colour treatments of the same photograph on the 44 µm / 22-level screen — plain, hue-mapped and zone-mapped — at 0 / 1 / 2°, simulated. Plain does not change with tilt; that is the control. On this plate the sides carry ZONES (left) and HUE (right) at 15 mm, and the choice between them is made on the box.")

parts = []
for kind, html_ in blocks:
    if kind == "h1":
        continue
    if kind == "p" and "(table inserted from the manifest)" in html_:
        continue
    if kind == "h2":
        parts.append(f'<div class="group"><h2>{html_}</h2></div>')
    elif kind in ("h3", "h4"):
        parts.append(f"<{kind}>{html_}</{kind}>")
    else:
        parts.append(html_)
    if kind in ("h2", "h3"):
        key = H.unescape(re.sub(r"<[^>]+>", "", html_))
        for k, v in AFTER.items():
            if key.startswith(k):
                parts.append(("__AFTER__", v))
        if key.startswith("4.5 Halftone"):
            parts.append(("__AFTER__", VARIANTS))

final, pending = [], []
for p in parts:
    if isinstance(p, tuple):
        pending.append(p[1])
        continue
    if pending and (p.startswith('<div class="group"><h2>') or p.startswith("<h3>") or p.startswith("<hr>")):
        final.extend(pending)
        pending = []
    final.append(p)
final.extend(pending)
plan_html = "\n".join(final)

BODY = f"""<title>Witness Plate</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Archivo:wght@500;600;700&family=IBM+Plex+Mono:wght@400;500;600&family=Source+Serif+4:opsz,wght@8..60,400;8..60,600&display=swap">
<style>{CSS}
figure {{ margin: 18px 0 30px; }}
figure.diagram {{ color: var(--ink-2); max-width: 780px; }}
figure.diagram svg {{ width: 100%; height: auto; display: block; }}
figcaption {{ font-size: 13px; color: var(--muted); margin-top: 8px; max-width: 76ch; line-height: 1.5; }}
ol {{ max-width: 72ch; color: var(--ink-2); }}
</style>

<div class="wrap">
  <p class="eyebrow">Ring box &middot; physics validation plan</p>
  <h1>Witness Plate</h1>
  <p class="standfirst">
    One 127 mm chrome plate on {PLY_UM/1000:g} mm {GLASS_MATERIAL} — the box's own stock — written darkfield with
    positive resist, carrying {len(cells)} cells: four of them are box faces that come off the plate
    as finished plies, the rest are the experiments. Every cell returns a number that could change
    the box design; the ones that could not were cut. What follows is the argument, from the four
    mechanisms down to the protocol for reading the glass, with the plate as built at the end.
  </p>

  <dl class="givens">
    <div class="given"><dt>Plate</dt><dd>127 mm<small>5&Prime; {GLASS_MATERIAL}, {PLY_UM/1000:g} mm, n = {GLASS_N}</small></dd></div>
    <div class="given"><dt>Polarity</dt><dd>darkfield<small>clear data, positive resist</small></dd></div>
    <div class="given"><dt>Cells</dt><dd>{len(cells)}<small>{lay['height_used_mm']:.0f} of {lay['height_available_mm']:.0f} mm packed</small></dd></div>
    <div class="given"><dt>Production dies</dt><dd>{prod_wr/usable*100:.0f}%<small>of the field: lid + front pairs, two colour sides</small></dd></div>
    <div class="given"><dt>Needs a bond</dt><dd>{two_wr/exp_wr*100:.0f}%<small>of the experiments; the rest is single-layer</small></dd></div>
    <div class="given"><dt>Mask</dt><dd>{gds['size_mb_by_format'].get('gds',0):.0f} MB<small>GDSII; {gds['size_mb_by_format'].get('oas',0):.1f} MB as OASIS</small></dd></div>
  </dl>

{plan_html}

  <div class="group" style="margin-top:44px"><span class="group-tag">Appendix A</span><h2>Every cell</h2>
    <p>Grouped by block, sorted by family. The etched label is what you will read under the microscope.</p></div>
{''.join(tabs)}

  <div class="group" style="margin-top:44px"><span class="group-tag">Appendix B</span><h2>The mask file</h2></div>
{FAB}

  <footer>
    Built by <code>app.export_witness</code>; cell geometry in <code>app.witness_cells</code> and
    <code>app.witness_moire</code>, the colour pipeline in <code>patterns.bitmap.colourplan</code> +
    <code>screenrects</code>. The plan is <code>docs/witness-physics-plan.md</code>, rendered here with
    the as-built tables inserted from the manifest. Simulated figures are rendered from the emitted
    geometry by <code>tools/dev/render_moire_preview.py</code>, <code>render_witness_preview.py</code>
    and <code>render_witness_figures.py</code>; the production dies live in <code>app.witness_dies</code>,
    their renders come from <code>render_witness_plate.py</code> (klayout, from the GDS) and their gates
    from <code>validate_dies.py</code>. The page refuses a figure older than its renderer.
  </footer>
</div>"""

io.open(OUT, "w", encoding="utf-8").write(BODY)
print(f"wrote {OUT}  {len(BODY)/1024/1024:.2f} MB")
