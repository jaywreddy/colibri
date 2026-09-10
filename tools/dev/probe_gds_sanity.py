"""GDS sanity: per-die unclipped clear DRC, zero-area shapes, vertex counts, data extent."""
import json, sys
import klayout.db as kdb
gds = sys.argv[1]
ly = kdb.Layout(); ly.read(gds); top = ly.top_cell(); dbu = ly.dbu
man = json.load(open(gds.replace(".gds", ".json"), encoding="utf-8"))
layers = {(ly.get_info(i).layer, ly.get_info(i).datatype): i for i in ly.layer_indexes()}
L1 = layers[(10, 0)]
# vertex counts + zero-area over the whole plate, layer by layer
for (l, d), li in sorted(layers.items()):
    maxv = 0; zero = 0; n = 0; bbox = kdb.Box()
    it = top.begin_shapes_rec(li)
    while not it.at_end():
        sh = it.shape()
        if sh.is_box() or sh.is_polygon() or sh.is_path():
            p = sh.polygon.transformed(it.trans())
            n += 1
            maxv = max(maxv, p.num_points())
            if p.area() == 0: zero += 1
            bbox += p.bbox()
        it.next()
    print(f"layer {l}/{d}: shapes {n} max_vertices {maxv} zero_area {zero} bbox_um {[round(v*dbu,1) for v in (bbox.left,bbox.bottom,bbox.right,bbox.top)]}")
# unclipped die DRC: all shapes whose bbox is inside the die box (no clip)
for c in man["cells"]:
    if not str(c["cid"]).startswith("DIE-"): continue
    x, y, w, h = (c[k]*1000 for k in ("x_mm","y_mm","w_mm","h_mm"))
    b = kdb.Box(*[int(round(v/dbu)) for v in (x-w/2-1, y-h/2-1, x+w/2+1, y+h/2+1)])
    r = kdb.Region(top.begin_shapes_rec_touching(L1, b))
    raw_n = r.count()
    wv_raw = r.width_check(int(round(2.0/dbu)), False, kdb.Region.Euclidian, 80, None, None, True, False, kdb.Region.IgnoreProperties, kdb.Region.NeverIncludeZeroDistance)
    r.merge()
    wv = r.width_check(int(round(2.0/dbu)), False, kdb.Region.Euclidian, 80, None, None, True, False, kdb.Region.IgnoreProperties, kdb.Region.NeverIncludeZeroDistance)
    sv = r.space_check(int(round(2.0/dbu)), False, kdb.Region.Euclidian, 80)
    print(f"{c['cid']}: raw shapes {raw_n} raw-width<2 {wv_raw.count()} | merged polys {r.count()} holes {sum(p.holes() for p in r.each())} width<2 {wv.count()} space<2 {sv.count()}")
    k = 0
    for e in wv_raw.each():
        if k >= 4: break
        print(f"   raw w viol ({e.first.p1.x*dbu:.3f},{e.first.p1.y*dbu:.3f}) d={e.distance()*dbu:.4f}"); k += 1
