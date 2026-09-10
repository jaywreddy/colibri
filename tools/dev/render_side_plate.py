"""Prepare the candidate side-plate photographs and render each as the finished
plate: portrait in the 15.4 mm art box, edge-faded, inside the colour garland,
as the eye sees it under a lamp.

Per photo a RECIPE says how it is prepared:
  crop        (x, y, side) fractions in the load_gray convention
  segment     cut the people out with rembg (u2net) and put them on a clean
              light ground -- the ground prints as bare glass, so there is
              nothing to extend or invent behind them
  contrast    stretch the subject's luminance (2-98 percentile) and open the
              midtones a little
  square      "pad" places a non-square source on the square ground (after
              segmentation there is no background to continue); "crop" uses
              the crop as given
  plan        the colour plan: plain / auto-zones / an authored rule list
  fade        fraction of the art box over which the picture fades to glass at
              its edge, so it dissolves into the frame instead of ending at a
              square

Outputs per photo: <stem>_source.png (the prepared square), <stem>_plate.png
(the die at 0 and 1 degrees) and side_plates.json with the numbers.

    uv run --directory backend --with rembg --with onnxruntime python ../tools/dev/render_side_plate.py PHOTOS_DIR OUTDIR

Light: one image at a time at 1400 px; the matting model runs once per photo.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "backend"))
sys.path.insert(0, str(HERE))

from app import witness_dies as wd  # noqa: E402
from app.patterns.bitmap import colourplan as cp  # noqa: E402
from app.patterns.bitmap import imageprep as ip  # noqa: E402
from render_witness_preview import ETA0, ETA1, EYE_UM, GOLD, K0, TILT_GAIN, sheen  # noqa: E402

ROOT = HERE.parents[1]
SRC_PX = 1400
GROUND = (236, 233, 226)   # prints as near-clear glass
BG = (13, 17, 19)
FG = (214, 224, 227)
DIM = (150, 165, 172)


def font(sz=12):
    try:
        return ImageFont.truetype("arialbd.ttf", sz)
    except Exception:
        return ImageFont.load_default()


def _rule(name, hue, sat=(0.0, 1.0), val=(0.0, 1.0), bbox=None, scale=1.0):
    return cp.Rule(name=name, hue_deg=hue, sat=sat, val=val, bbox=bbox, mode="fixed",
                   period_scale=scale, min_blob_px=40, close_px=3)


BLUE = 1.45 ** -0.5      # 4.15 um: the blue end of the ladder
RED = 1.45 ** 0.5        # 6.02 um: the red end
GREEN = 1.0              # 5.0 um

RECIPES = {
    "PXL_20240807_233638672.MP": dict(
        file="photos/PXL_20240807_233638672.MP.jpg", crop=(0.20, 0.03, 0.62), segment=False, contrast=False,
        square="crop", fade=0.12,
        plan=cp.ColourPlan(name="faces", mode="hue", coarsen_px=14, hue_equalize=False, hue_min_sat=0.28, hue_min_value=0.22),
        why="auto-zones: the faces are the only saturated thing, so they alone take colour; sea and sky stay gold"),
    "IMG_1827~2": dict(
        file="photos/IMG_1827~2.jpg", crop=(0.147, 0.0, 0.763), segment=False, contrast=False, square="crop", fade=0.12,
        plan=cp.ColourPlan(name="sunset", mode="zones", coarsen_px=7, rules=(
            _rule("shirt", (185.0, 250.0), sat=(0.12, 1.0), val=(0.35, 1.0), bbox=(0.45, 0.15, 1.0, 0.95), scale=BLUE),
            _rule("sun", (15.0, 55.0), sat=(0.35, 1.0), val=(0.85, 1.0), bbox=(0.0, 0.15, 0.5, 0.5), scale=RED),
        )),
        why="authored zones: his shirt at the blue end, the sun disc at the red end, everything else plain gold"),
    "PXL_20240803_232128378.MP": dict(
        file="photos/PXL_20240803_232128378.MP.jpg", crop=(0.19, 0.24, 0.62), segment=False, contrast=False, square="crop", fade=0.12,
        plan=cp.ColourPlan(name="dress", mode="zones", coarsen_px=7, rules=(
            _rule("dress", (195.0, 255.0), sat=(0.28, 1.0), val=(0.25, 1.0), bbox=(0.4, 0.3, 1.0, 1.0), scale=BLUE),
        )),
        why="authored zones: the dress alone at the blue end; the garden stays plain so it cannot compete"),
    "IMG_6584": dict(
        file="photos/IMG_6584.jpg", crop=(0.0, 0.02626, 1.0), segment=True, contrast=True, square="crop", fade=0.12,
        plan=cp.ColourPlan(name="plain", mode="plain"),
        why="segmented: the couple cut from the park and set on glass; plain gold, the busy background is gone rather than coloured"),
    "IMG_1290-EDIT": dict(
        file="photos/IMG_1290-EDIT.jpg", crop=None, segment=True, contrast=True, square="pad", fade=0.14,
        plan=cp.ColourPlan(name="plain", mode="plain"),
        why="segmented, contrast-stretched, and placed on a square glass ground: with the marina gone there is nothing to extend, so the canvas is simply the ground"),
    "signal-2026-01-05-11-40-52-562": dict(
        file="photos/signal-2026-01-05-11-40-52-562.jpg", crop=(0.125, 0.0, 0.75), segment=True, contrast=True, square="crop", fade=0.12,
        plan=cp.ColourPlan(name="plain", mode="plain"),
        why="segmented: five people on glass, the backlit trellis removed; plain gold, read as a group scene"),
}


# --- preparation --------------------------------------------------------------


def load_square(path: Path, crop, square: str) -> Image.Image:
    im = Image.open(path)
    im.draft("RGB", (SRC_PX * 2, SRC_PX * 2))
    im = ImageOps.exif_transpose(im).convert("RGB")
    if crop is not None:
        w, h = im.size
        fx, fy, fs = crop
        x0, y0, side = int(fx * w), int(fy * h), int(fs * w)
        im = im.crop((x0, y0, min(w, x0 + side), min(h, y0 + side)))
    if square == "pad":
        w, h = im.size
        s = max(w, h)
        canvas = Image.new("RGB", (s, s), GROUND)
        canvas.paste(im, ((s - w) // 2, (s - h) // 2))
        im = canvas
    return im.resize((SRC_PX, SRC_PX), Image.LANCZOS)


def segment(im: Image.Image) -> np.ndarray:
    """Alpha matte of the people, 0..1, via rembg's u2net (human-oriented)."""
    from rembg import new_session, remove
    sess = new_session("u2net_human_seg")
    out = remove(im, session=sess, alpha_matting=True, alpha_matting_foreground_threshold=240,
                 alpha_matting_background_threshold=15, alpha_matting_erode_size=8)
    a = np.asarray(out.convert("RGBA"), dtype=np.float32)[..., 3] / 255.0
    return a


def stretch_contrast(rgb: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
    lum = rgb @ np.array([0.2126, 0.7152, 0.0722], np.float32)
    sel = lum[mask > 0.5] if mask is not None and (mask > 0.5).sum() > 100 else lum.ravel()
    lo, hi = np.percentile(sel, 2), np.percentile(sel, 98)
    out = np.clip((rgb - lo) / max(1e-3, hi - lo), 0, 1)
    return out ** 0.9                      # open the midtones a little


def prepare(stem: str, r: dict, out: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    path = ROOT / r["file"]
    im = load_square(path, r["crop"], r["square"])
    rgb = np.asarray(im, dtype=np.float32) / 255.0
    info = {"segmented": False, "contrast": False}
    mask = None
    if r["segment"]:
        mask = segment(im)
        info["segmented"] = True
        info["subject_fraction"] = round(float((mask > 0.5).mean()), 3)
    if r["contrast"]:
        rgb = stretch_contrast(rgb, mask)
        info["contrast"] = True
    if mask is not None:
        g = np.array(GROUND, np.float32)[None, None, :] / 255.0
        rgb = rgb * mask[..., None] + g * (1 - mask[..., None])
        Image.fromarray((mask * 255).astype(np.uint8)).save(out / f"{stem}_mask.png")
    src = Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8))
    src.save(out / f"{stem}_source.png")
    gray = np.asarray(src.convert("L"), dtype=np.float32) / 255.0
    return gray, rgb, info


# --- the plate ----------------------------------------------------------------


def edge_fade(n: int, frac: float) -> np.ndarray:
    """1 inside, smooth to 0 over the outer ``frac`` of the square (rounded
    square distance, so corners fade like edges)."""
    if frac <= 0:
        return np.ones((n, n), np.float32)
    y, x = np.mgrid[0:n, 0:n].astype(np.float32)
    u = np.abs((x + 0.5) / n - 0.5) * 2      # 0 centre .. 1 edge
    v = np.abs((y + 0.5) / n - 0.5) * 2
    d = np.maximum(u, v)
    # soften the corners: blend max-norm with 2-norm
    d = 0.7 * d + 0.3 * np.sqrt(u * u + v * v) / math.sqrt(2)
    t = np.clip((1.0 - d) / frac, 0, 1)
    return (t * t * (3 - 2 * t)).astype(np.float32)


def render_plate(stem: str, gray: np.ndarray, rgb: np.ndarray, r: dict, out: Path, face: str = "left") -> dict:
    _, spec = wd.blank_plan()
    pspec = spec.faces[face]
    d = wd.die_dims(face)
    W, H = d["f_w"], d["f_h"]
    from app import plates as P
    side = P.CENTERPIECE_FILL * P._aperture(pspec)
    pitch = max(10.0, math.sqrt(W * H / (0.9 * 400000)))
    levels, periods = wd._garland_levels(pspec, pitch)
    fh, fw = levels.shape

    # portrait darkness at the garland raster, faded at the edge
    dark = ip.prep_darkness(gray, ip.PrepSpec(tone_steps=22))
    ids, pmap, _ = cp.build_period_field(rgb, r["plan"])
    n_art = max(8, int(round(side / pitch)))
    dark_s = np.asarray(Image.fromarray((np.clip(dark, 0, 1) * 255).astype(np.uint8)).resize((n_art, n_art), Image.BOX)) / 255.0
    dark_s = dark_s * edge_fade(n_art, r["fade"])
    ids_s = np.asarray(Image.fromarray(ids.astype(np.int32)).resize((n_art, n_art), Image.NEAREST))
    lut = np.zeros(int(max(ids.max(), 0)) + 1, np.float32)
    for i, per in pmap.items():
        if i < lut.size:
            lut[i] = per
    per_s = lut[np.clip(ids_s, 0, lut.size - 1)] if lut.size > 1 else np.zeros_like(dark_s)

    # compose: garland cells at 50% coverage with their motif period, art box centred
    cov = np.zeros((fh, fw), np.float32)
    per = np.zeros((fh, fw), np.float32)
    g_mask = levels > 0
    cov[g_mask] = 0.5
    for i, p_ in enumerate(periods):
        per[levels == wd.GARLAND_LEVEL0 + i] = p_
    ox, oy = (fw - n_art) // 2, (fh - n_art) // 2
    cov[oy:oy + n_art, ox:ox + n_art] = dark_s
    per[oy:oy + n_art, ox:ox + n_art] = per_s

    # eye integration (pitch ~45 um -> 87 um cells)
    k = max(1, int(round(EYE_UM / pitch)))
    def bm(a):
        m, n = a.shape[0] - a.shape[0] % k, a.shape[1] - a.shape[1] % k
        return a[:m, :n].reshape(m // k, k, n // k, k).mean(axis=(1, 3))
    cov_e = bm(cov)
    per_e = bm(per * (per > 0)) / np.maximum(1e-6, bm((per > 0).astype(np.float32)))
    per_e[bm((per > 0).astype(np.float32)) < 0.5] = 0.0

    tiles = []
    for tilt in (0.0, 1.0):
        u = per_e * (K0 + math.sin(math.radians(tilt)) * TILT_GAIN)
        spec_rgb = sheen(u.astype(np.float32))
        o = GOLD * ETA0 * cov_e[..., None]
        o = o + np.where((per_e > 0)[..., None], spec_rgb * ETA1 * cov_e[..., None], 0.0)
        o = o / ETA0
        img = np.clip(ip.linear_to_srgb(np.clip(o, 0, 1)), 0, 1)
        scale = 700 / img.shape[0]
        tiles.append(Image.fromarray((img * 255).astype(np.uint8)).resize(
            (int(img.shape[1] * scale), 700), Image.LANCZOS))
    src = Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8)).resize((int(700 * side / H), int(700 * side / H)), Image.LANCZOS)
    Wp = 10 + src.width + 10 + sum(t.width + 10 for t in tiles)
    sheet = Image.new("RGB", (Wp, 700 + 70), BG)
    dr = ImageDraw.Draw(sheet)
    x = 10
    sheet.paste(src, (x, 10 + (700 - src.height) // 2)); dr.text((x, 718), "prepared source (art box scale)", fill=DIM, font=font(11)); x += src.width + 10
    for t, tilt in zip(tiles, (0.0, 1.0)):
        sheet.paste(t, (x, 10)); dr.text((x, 718), f"the {face} plate, tilt {tilt:g}°  ({W/1000:.1f} × {H/1000:.2f} mm, art box {side/1000:.1f} mm)", fill=FG, font=font(11)); x += t.width + 10
    dr.text((10, 740), f"{stem}: {r['why']}. Edge fade {r['fade']:.0%} of the box; garland motifs at their hue rungs; eye-cell integrated, lamp.", fill=DIM, font=font(10))
    p = out / f"{stem}_plate.png"
    sheet.save(p)
    print("  saved", p)
    return {"plate": p.name, "art_box_um": round(side, 1), "fade": r["fade"],
            "coloured_fraction": round(float((ids > 0).mean()), 3), "plan": r["plan"].name}


def main() -> int:
    photos = Path(sys.argv[1]); out = Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
    results = []
    for stem, r in RECIPES.items():
        print(stem)
        gray, rgb, info = prepare(stem, r, out)
        res = render_plate(stem, gray, rgb, r, out)
        results.append({"stem": stem, "file": r["file"], "crop": r["crop"], "square": r["square"], "why": r["why"], **info, **res})
    (out / "side_plates.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
