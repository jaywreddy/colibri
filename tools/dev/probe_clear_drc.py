"""Probe: DRC the WRITTEN clear data of each production die in witness-5in.gds.

    uv run --directory backend python ../tools/dev/probe_clear_drc.py backend/data/witness/witness-5in.gds
"""
import json
import sys

import klayout.db as kdb

gds = sys.argv[1]
ly = kdb.Layout()
ly.read(gds)
top = ly.top_cell()
man = json.load(open(gds.replace(".gds", ".json"), encoding="utf-8"))
dbu = ly.dbu
layers = {(ly.get_info(i).layer, ly.get_info(i).datatype): i for i in ly.layer_indexes()}
print("layers", sorted(layers))
data_layer = layers[(10, 0)]


def region_in(box_um, layer):
    b = kdb.Box(*[int(round(v / dbu)) for v in box_um])
    r = kdb.Region(top.begin_shapes_rec_touching(layer, b))
    r &= kdb.Region(b)
    r.merge()
    return r


min_um = float(sys.argv[2]) if len(sys.argv) > 2 else 2.0
for c in man["cells"]:
    if not str(c["cid"]).startswith("DIE-"):
        continue
    x, y, w, h = (c[k] * 1000.0 for k in ("x_mm", "y_mm", "w_mm", "h_mm"))
    boxes = [("F", (x - w / 2, y - h / 2, x + w / 2, y + h / 2))]
    if c.get("two_layer"):
        bw, bh = c["back_w_mm"] * 1000, c["back_h_mm"] * 1000
        # back die sits at the pair position: search the manifest layout for it
        bx = c.get("back_x_mm"); by = c.get("back_y_mm")
        if bx is not None:
            boxes.append(("B", (bx * 1000 - bw / 2, by * 1000 - bh / 2, bx * 1000 + bw / 2, by * 1000 + bh / 2)))
    for tag, box in boxes:
        r = region_in(box, data_layer)
        wv = r.width_check(int(round(min_um / dbu)), False, kdb.Region.Euclidian, 80)
        sv = r.space_check(int(round(min_um / dbu)), False, kdb.Region.Euclidian, 80)
        print(f"{c['cid']} {tag}: merged polys {r.count()}  clear-width<{min_um} {wv.count()}  clear-space<{min_um} {sv.count()}")
        seen = 0
        for e in wv.each():
            if seen >= 6:
                break
            print(f"    width viol at ({e.first.p1.x*dbu:.3f},{e.first.p1.y*dbu:.3f}) um  d={e.distance()*dbu:.3f}")
            seen += 1
        seen = 0
        for e in sv.each():
            if seen >= 6:
                break
            print(f"    space viol at ({e.first.p1.x*dbu:.3f},{e.first.p1.y*dbu:.3f}) um  d={e.distance()*dbu:.3f}")
            seen += 1
