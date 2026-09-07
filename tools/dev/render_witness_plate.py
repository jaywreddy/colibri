"""Render the written witness GDS to PNG — the whole plate and zooms into the
production dies — with klayout's headless LayoutView, so what is shown is the
data in the file, not a preview drawn from the manifest.

    uv run --directory backend python ../tools/dev/render_witness_plate.py OUTDIR [GDS]

Only layer 10/0 (the written CLEAR data) is drawn, light on dark: light is
where the chrome comes off. Zoom windows are in plate micrometres.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import klayout.db as db
import klayout.lay as lay

DATA = "#e8d9a0"


def render(src: str, out: Path, size: int, box: tuple[float, float, float, float] | None) -> None:
    lv = lay.LayoutView()
    lv.load_layout(src, True)
    lv.max_hier()
    lv.set_config("background-color", "#101418")
    lv.set_config("grid-visible", "false")
    it = lv.begin_layers()
    while not it.at_end():
        lp = it.current()
        lp.visible = lp.source_layer == 10
        lp.fill_color = int(DATA[1:], 16)
        lp.frame_color = int(DATA[1:], 16)
        lp.dither_pattern = 0
        it.next()
    if box is None:
        lv.zoom_fit()
    else:
        lv.zoom_box(db.DBox(*box))
    lv.save_image(str(out), size, size)
    print("  saved", out)


def main() -> int:
    out = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
    out.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[2] / "backend" / "data" / "witness"
    gds = sys.argv[2] if len(sys.argv) > 2 else str(root / "witness-5in.gds")
    man = json.loads((root / "witness-5in.json").read_text(encoding="utf-8"))
    dies = {c["cid"]: c for c in man["cells"] if c["block"] == "production"}
    render(gds, out / "witness_plate.png", 1800, None)
    # the top pair: F and B side by side
    t = dies["DIE-TOP"]
    x0 = (t["x_mm"] - t["w_mm"] / 2) * 1000 - 800
    x1 = (t["x_mm"] + t["w_mm"] / 2 + 1.0 + t["back_w_mm"]) * 1000 + 800
    yc = t["y_mm"] * 1000
    half = max(t["h_mm"], (x1 - x0) / 1000) * 500 + 800
    render(gds, out / "witness_die_top.png", 1600, (x0, yc - half, x1, yc + half))
    # a leaf of the lid's garland, 4 mm
    lx = (t["x_mm"] - t["w_mm"] / 2 + 2.5) * 1000
    ly_ = (t["y_mm"] + t["h_mm"] / 2 - 4.5) * 1000
    render(gds, out / "witness_die_leaf.png", 1200, (lx, ly_ - 2000, lx + 4000, ly_ + 2000))
    # the left side: portrait centre 3 mm, and a garland corner 3 mm
    s = dies["DIE-LEFT"]
    cx, cy = s["x_mm"] * 1000, s["y_mm"] * 1000
    render(gds, out / "witness_die_portrait.png", 1200, (cx - 1500, cy - 1500, cx + 1500, cy + 1500))
    # The colour garland: 4.15-6 um vertical gratings, which a 3 mm window
    # cannot resolve. Find the densest 0.6 mm patch of thin shapes in the left
    # band from the GDS itself and zoom there at ~0.5 um/px.
    ly = db.Layout(); ly.read(gds)
    top_cell = ly.top_cell(); L = ly.layer(10, 0); dbu = ly.dbu
    x_edge = (s["x_mm"] - s["w_mm"] / 2) * 1000
    band = db.DBox(x_edge + 1500, cy - s["h_mm"] * 500 + 3000, x_edge + 4200, cy + s["h_mm"] * 500 - 3000)
    reg = db.Region(top_cell.begin_shapes_rec_touching(L, db.Box(band.to_itype(dbu))))
    bins: dict[tuple[int, int], int] = {}
    for poly in reg.each():
        b = poly.bbox().to_dtype(dbu)
        if b.width() < 4.0 and b.height() > 50.0:
            key = (int(b.center().x // 600), int(b.center().y // 600))
            bins[key] = bins.get(key, 0) + 1
    if bins:
        kx, ky = max(bins, key=bins.get)
        gx, gy = (kx + 0.5) * 600, (ky + 0.5) * 600
    else:
        gx, gy = x_edge + 2800, cy
    render(gds, out / "witness_die_garland.png", 1200, (gx - 300, gy - 300, gx + 300, gy + 300))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
