"""Add (or refresh) the saw-lane marks on an already-written plate.

    uv run --directory backend python ../tools/dev/add_dice_marks.py [stem]

``export_witness.build_plate`` writes the marks itself (``_dice_edge_marks``);
this tool exists for a plate written before they existed, so the 8-minute
build need not be repeated to get them. It reads the layout's cut list from
``<stem>.json``, computes the identical rectangles, inserts them on the data
layer of ``<stem>.gds`` / ``<stem>.oas`` (skipping any already present) and
rewrites both files with the same save options the writer uses.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "backend"))
sys.stdout.reconfigure(encoding="utf-8")

import klayout.db as kdb  # noqa: E402

from app.export_witness import (DICE_MARK_LEN_UM, DICE_MARK_W_UM, LAYER_DICE,  # noqa: E402
                                LAYER_FRONT, _dice_edge_marks, _dice_line_rects,
                                _insert_rects, _save_options, doe_cells, layout,
                                write_dicing_md, write_map_svg)


def main() -> int:
    stem = Path(sys.argv[1] if len(sys.argv) > 1 else "data/witness/witness-5in")
    man = json.loads(stem.with_suffix(".json").read_text(encoding="utf-8"))
    # The layout is deterministic and needs no die build: recompute it so the
    # cut list is the CURRENT scheme, and check it still describes this plate.
    placed, lay = layout(doe_cells())
    pos = {c["cid"]: (c["x_mm"], c["y_mm"]) for c in man["cells"]}
    for p_ in placed:
        x, y = pos[p_.cell.cid]
        assert abs(p_.cx / 1000 - x) < 1e-3 and abs(p_.cy / 1000 - y) < 1e-3, p_.cell.cid
    man["layout"] = lay
    stem.with_suffix(".json").write_text(json.dumps(man, indent=2), encoding="utf-8")
    plate = {"layout": lay, "manifest": man["cells"]}
    write_dicing_md(plate, stem.with_name("DICING.md"))
    write_map_svg(plate, stem.with_name(stem.name + "-map.svg"))
    marks = _dice_edge_marks(lay)
    mark_w = int(round(DICE_MARK_W_UM * 1000))
    mark_l = int(round(DICE_MARK_LEN_UM * 1000))
    for suffix in (".gds", ".oas"):
        p = stem.with_suffix(suffix)
        if not p.exists():
            continue
        ly = kdb.Layout()
        ly.read(str(p))
        top = ly.top_cell()
        li = ly.layer(*LAYER_FRONT)
        # drop every earlier saw-lane mark (a bare 80 x 500 um bar is nothing
        # else on this plate), then insert the current set
        removed = 0
        for sh in top.shapes(li).each():
            if sh.is_box():
                b = sh.box
                if sorted((b.width(), b.height())) == [mark_w, mark_l]:
                    top.shapes(li).erase(sh)
                    removed += 1
        import numpy as np
        _insert_rects(top, li, np.asarray(marks), kdb)
        # the annotation layer: street centre-lines -> edge lines
        ld = ly.layer(*LAYER_DICE)
        top.shapes(ld).clear()
        _insert_rects(top, ld, _dice_line_rects(lay), kdb)
        ly.write(str(p), _save_options(suffix, kdb))
        print(f"{p.name}: {removed} old marks removed, {len(marks)} saw-lane marks written; "
              f"{p.stat().st_size / 1e6:.1f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
