"""Compose the production box and lay its six faces out as the eye would see
them, from the SAME literal coverage rasters the simulator samples.

    uv run --directory backend python ../tools/dev/render_box_faces.py OUTDIR

For every face: literal_front.png and literal_back.png (255 = chrome) are read
from the plate directory, the bonded stack's metal coverage 1 - (1-A)(1-B) is
formed per texel (head-on, so no parallax shift), and painted gold on glass
over a dark interior. A second row shows the front raster alone (what the
outer ply carries). Heavy: a cold compose of the two photo faces takes minutes;
run alone.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "backend"))

from app.boxes import default_box_spec, materialize_box  # noqa: E402
from app.plates import PLATES_ROOT  # noqa: E402

GOLD = np.array([227, 181, 59], dtype=np.float32)
GLASS = np.array([26, 28, 34], dtype=np.float32)   # bare glass over a dark interior
COPPER = np.array([120, 74, 40], dtype=np.float32)  # foil rim, for scale only


def load(pid: str, name: str) -> np.ndarray | None:
    p = PLATES_ROOT / pid / f"{name}.png"
    if not p.exists():
        return None
    return np.asarray(Image.open(p).convert("L"), dtype=np.float32) / 255.0


def paint(cov: np.ndarray) -> Image.Image:
    rgb = GLASS[None, None, :] * (1 - cov[..., None]) + GOLD[None, None, :] * cov[..., None]
    return Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8), "RGB")


def main() -> int:
    out = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    spec = default_box_spec()
    man = materialize_box(spec)
    print(f"box composed in {time.perf_counter() - t0:.0f}s")
    faces = ["top", "front", "left", "right", "back", "bottom"]
    tiles: list[tuple[str, Image.Image, Image.Image, str]] = []
    for fid in faces:
        fm = man["faces"][fid]
        pid = fm["id"]
        rd = fm.get("recipe_data", {})
        A = load(pid, "literal_front")
        B = load(pid, "literal_back")
        if A is None:
            print(fid, pid, "no literal raster")
            continue
        if B is None or rd.get("single_ply") or rd.get("blank"):
            B = np.zeros_like(A)
        # the back raster is composed in the outer frame at the same size
        if B.shape != A.shape:
            B = np.asarray(Image.fromarray((B * 255).astype(np.uint8)).resize(A.shape[::-1], Image.BILINEAR), dtype=np.float32) / 255.0
        stack = 1.0 - (1.0 - A) * (1.0 - B)
        ps = spec.faces[fid]
        cap = (f"{fid}: {ps.pattern_slug}{' ' + str(ps.pattern_params.get('image')) if ps.pattern_params.get('image') else ''}"
               f"  {ps.width_um/1000:.1f}x{ps.height_um/1000:.2f} mm  rim {ps.weld_margin_um/1000:.2f} mm"
               f"  {'single ply' if rd.get('single_ply') else ('blank' if rd.get('blank') else 'F+B')}"
               f"  front cov {A.mean():.3f} stack cov {stack.mean():.3f}")
        print(cap, "| plate", pid)
        tiles.append((fid, paint(stack), paint(A), cap))
    if not tiles:
        return 1
    tw = 700
    font = ImageFont.load_default()
    rows = []
    for fid, stack_im, front_im, cap in tiles:
        w, h = stack_im.size
        th = int(round(tw * h / w))
        a = stack_im.resize((tw, th), Image.LANCZOS)
        b = front_im.resize((tw, th), Image.LANCZOS)
        row = Image.new("RGB", (2 * tw + 30, th + 40), (12, 12, 14))
        row.paste(a, (10, 30)); row.paste(b, (tw + 20, 30))
        ImageDraw.Draw(row).text((12, 8), cap + "   |  left: both plies head-on, right: outer ply alone", fill=(230, 230, 230), font=font)
        rows.append(row)
    W = max(r.width for r in rows); H = sum(r.height for r in rows)
    sheet = Image.new("RGB", (W, H), (12, 12, 14))
    y = 0
    for r in rows:
        sheet.paste(r, (0, y)); y += r.height
    sheet.save(out / "production_box_faces.png")
    # individual faces at full raster size
    for fid, stack_im, front_im, _ in tiles:
        stack_im.save(out / f"face_{fid}_stack.png")
    print("wrote", out / "production_box_faces.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
