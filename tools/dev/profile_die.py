"""Where a production die's build time goes.

    uv run --directory backend python ../tools/dev/profile_die.py left [garden]

Runs ``witness_dies.build_face_die`` for the face under cProfile and prints the
top cumulative-time entries, then times a GDS write of its polygons the way
``export_witness.build_plate`` does it. Heavy (one die build); run alone.
"""
from __future__ import annotations

import cProfile
import pstats
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "backend"))

from app import witness_dies as wd  # noqa: E402

faces = sys.argv[1:] or ["left"]
for face in faces:
    pspec = None
    fid = face
    if face not in ("top", "front", "left", "right"):
        image, mode, _cid, seed = next(t for t in wd.SIDE_PHOTOS if t[0] == face)
        pspec = wd.side_photo_spec(image, mode, seed)
        fid = "left"
    d = wd.die_dims(fid)
    pr = cProfile.Profile()
    t0 = time.perf_counter()
    pr.enable()
    art = wd.build_face_die(fid, 0.0, 0.0, d["f_w"], d["f_h"], wd.CLEAR, pspec=pspec)
    pr.disable()
    print(f"\n=== {face}: build_face_die {time.perf_counter() - t0:.1f} s, "
          f"{len(art.polys)} F polys ===")
    st = pstats.Stats(pr)
    st.sort_stats("cumulative").print_stats(28)

    import klayout.db as kdb
    t1 = time.perf_counter()
    ly = kdb.Layout(); ly.dbu = 0.001
    top = ly.create_cell("T"); L = ly.layer(10, 0)
    for pv in art.polys:
        top.shapes(L).insert(kdb.DPolygon([kdb.DPoint(float(x), float(y)) for x, y in pv]))
    t2 = time.perf_counter()
    out = Path(HERE.parents[1] / "backend" / "data" / "witness" / "_profile_die.gds")
    ly.write(str(out))
    t3 = time.perf_counter()
    print(f"GDS: python insert of {len(art.polys)} polys {t2 - t1:.1f} s; write {t3 - t2:.1f} s; {out.stat().st_size / 1e6:.1f} MB")
    out.unlink(missing_ok=True)
