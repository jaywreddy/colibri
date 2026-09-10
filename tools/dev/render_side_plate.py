"""Prepare the candidate side-plate photographs and render each as the finished
plate: portrait in the 13.3 mm art box (inside a 15.4 mm aperture), dissolving
to bare glass inside the
garland, as the eye sees it under a lamp.

Per photo a RECIPE says how it is prepared:
  crop        (x, y, side) fractions in the load_gray convention
  source_png  a source prepared outside this script (the fade study's ground for
              IMG_1290), with fade_png its fade field, used instead of crop
  plan        the colour plan: plain / auto-zones / an authored rule list
  fade        (start, gate) of the edge fade as fractions of the half-width

The garland on a single-ply side is the leaf gratings and NOTHING else (the
single-ply block in export_fine.build_plate_fine): one sheet has no second plane
for a carrier to beat against, and a carrier on the SAME ply printed as static
bars, so it went on 2026-09-10. Each leaf family is a fine 50% diffractive
grating at its own PERIOD from the colour ladder (plates.SINGLE_PLY_LEAF_FILL =
"hue"), which is what the sheen below is driven by. The portrait therefore
dissolves to BARE GLASS (photo.CARRIER_COV = 0), and the fade is done in
COVERAGE space after the tone prep (the prep is image-relative, so a source-level
target grey does not exist) by photo.context_fade -- the shipping function, so
this preview and the fab path cannot drift:

  stage 1  detail: cross-fade coverage to its 8%-of-width blur over the edge
           distance d in [s, s + 0.30] -- detail dies before level
  stage 2  level: cross-fade to 0.5 over d in [s + 0.25, gate]; hard 0.5 past gate
  where the per-pixel start s is pushed outward where the people are (rembg
  matte), so faces survive further into the margin than the ground does.

    uv run --directory backend --with rembg --with onnxruntime python ../tools/dev/render_side_plate.py PHOTOS_DIR OUTDIR [FADE_DIR]

Outputs per photo: <stem>_source.png, <stem>_plate.png (0 and 1 deg), and
side_plates.json + side_plates_grid.png. Light: one image at a time at 1400 px.
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
# The fade lives in the package now (photo.py's "fade primitives", ported OUT of
# this file). Import it back rather than keeping a second copy: this tool is what
# a photograph is chosen on, so it has to dissolve exactly the way the fab path
# will. CARRIER_COV came with it -- 0.0 since 2026-09-10, bare glass.
from app.patterns.bitmap.photo import (  # noqa: E402
    CARRIER_COV, blur, box_edge, context_fade, smootherstep,
)
from render_witness_preview import ETA0, ETA1, EYE_UM, GOLD, K0, TILT_GAIN, sheen  # noqa: E402

ROOT = HERE.parents[1]
SRC_PX = 1400
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

RECIPES = {
    "PXL_20240807_233638672.MP": dict(
        file="photos/PXL_20240807_233638672.MP.jpg", crop=(0.20, 0.03, 0.62), fade=(0.40, 0.94),
        plan=cp.ColourPlan(name="faces", mode="hue", coarsen_px=14, hue_equalize=False, hue_min_sat=0.28, hue_min_value=0.22),
        why="auto-zones: the faces are the only saturated thing, so they alone take colour; sea and sky stay gold"),
    "IMG_1827~2": dict(
        file="photos/IMG_1827~2.jpg", crop=(0.147, 0.0, 0.763), fade=(0.40, 0.94),
        plan=cp.ColourPlan(name="sunset", mode="zones", coarsen_px=7, rules=(
            _rule("shirt", (185.0, 250.0), sat=(0.12, 1.0), val=(0.35, 1.0), bbox=(0.45, 0.15, 1.0, 0.95), scale=BLUE),
            _rule("sun", (15.0, 55.0), sat=(0.35, 1.0), val=(0.85, 1.0), bbox=(0.0, 0.15, 0.5, 0.5), scale=RED),
        )),
        why="authored zones: his shirt at the blue end, the sun disc at the red end, everything else plain gold"),
    "PXL_20240803_232128378.MP": dict(
        file="photos/PXL_20240803_232128378.MP.jpg", crop=(0.19, 0.24, 0.62), fade=(0.35, 0.94),
        plan=cp.ColourPlan(name="dress", mode="zones", coarsen_px=7, rules=(
            _rule("dress", (195.0, 255.0), sat=(0.28, 1.0), val=(0.25, 1.0), bbox=(0.4, 0.3, 1.0, 1.0), scale=BLUE),
        )),
        why="authored zones: the dress alone at the blue end; the garden stays plain so it cannot compete"),
    "IMG_6584": dict(
        file="photos/IMG_6584.jpg", crop=(0.0, 0.02626, 1.0), fade=(0.40, 0.94),
        plan=cp.ColourPlan(name="plain", mode="plain"),
        why="plain gold with the park kept; the wider fade lets the busy background dissolve first"),
    "IMG_1290-EDIT": dict(
        file="photos/IMG_1290-EDIT.jpg", crop=None, source_png="IMG_1290_v3_source.png", fade_png="IMG_1290_fade_field.png",
        fade=(0.35, 0.94), lift=0.65,
        plan=cp.ColourPlan(name="plain", mode="plain"),
        why="the fade study's ground: people matted out, contrast stretched on them, the marina kept as a low-contrast ghost that decays with distance from the people and continues the bodies past the frame edge — no mirror, no box; re-targeted here so it dissolves into the carrier field"),
    "signal-2026-01-05-11-40-52-562": dict(
        file="photos/signal-2026-01-05-11-40-52-562.jpg", crop=(0.125, 0.0, 0.75), fade=(0.40, 0.94),
        plan=cp.ColourPlan(name="plain", mode="plain"),
        why="plain gold with the porch kept, as a group scene"),
}


# --- small image helpers ---------------------------------------------------------


def resize(a, n, mode=Image.BOX):
    return np.asarray(Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8)).resize((n, n), mode)) / 255.0


# --- preparation -----------------------------------------------------------------


def load_square(path, crop):
    im = Image.open(path)
    im.draft("RGB", (SRC_PX * 2, SRC_PX * 2))
    im = ImageOps.exif_transpose(im).convert("RGB")
    if crop is not None:
        w, h = im.size
        fx, fy, fs = crop
        x0, y0, side = int(fx * w), int(fy * h), int(fs * w)
        im = im.crop((x0, y0, min(w, x0 + side), min(h, y0 + side)))
    return im.resize((SRC_PX, SRC_PX), Image.LANCZOS)


def people_matte(im):
    """0..1 matte of the people, used only to push the fade start outward where
    they stand (no background is removed here)."""
    from rembg import new_session, remove
    sess = new_session("u2net_human_seg")
    out = remove(im, session=sess, alpha_matting=False)
    a = np.asarray(out.convert("RGBA"), np.float32)[..., 3] / 255.0
    return blur(a, 0.02 * a.shape[1])


def prepare(stem, r, photos, fade_dir):
    if r.get("source_png"):
        im = Image.open(fade_dir / r["source_png"]).convert("RGB").resize((SRC_PX, SRC_PX), Image.LANCZOS)
        rgb = np.asarray(im, np.float32) / 255.0
        # the study's feathered people matte (G channel) drives the fade push;
        # its own late-starting fade field is not used (the reviewer measured the
        # ramp starting at ~80% of the half-width -- too late)
        m = np.asarray(Image.open(fade_dir / "IMG_1290_matte.png").convert("RGB").resize((SRC_PX, SRC_PX), Image.BILINEAR), np.float32)[..., 1] / 255.0
        return rgb, blur(m, 0.02 * SRC_PX), None, {"external_source": r["source_png"]}
    im = load_square(ROOT / r["file"], r["crop"])
    rgb = np.asarray(im, np.float32) / 255.0
    subj = people_matte(im)
    return rgb, subj, None, {"subject_fraction": round(float((subj > 0.5).mean()), 3)}


# --- the plate ------------------------------------------------------------------


def garland_field(pspec, w, h, pitch):
    """Coverage and period maps of the single-ply garland at the boundary raster.

    The shipping construction (plates.SINGLE_PLY_LEAF_FILL = "hue",
    leaf_fills.bucket_layers, and the single-ply block in
    export_fine.build_plate_fine): NO carrier — the window outside the art box is
    bare glass — and every motif family filled with a fine 50%-duty grating at
    its own PERIOD from the colour ladder. Zero order is flat 50% gold whatever
    the period, so the coverage map is the duty; the family's period goes in the
    PERIOD map, which is what makes each one flash its own hue in the sheen
    below. (This used to draw a 0.5 carrier and paint each leaf with its beat
    against it: the pre-2026-09-10 design, and a second copy of geometry the
    package already owns.)
    """
    from app import plates as P
    from app.export_fine import _build_zone_masks
    from app import leaf_fills as LF

    rd = P._carrier_recipe_data(pspec)
    zm = _build_zone_masks(pspec, pitch)
    fh, fw = zm.frame_level.shape
    cov = np.zeros((fh, fw), np.float32)
    per = np.zeros((fh, fw), np.float32)
    duty = float(rd["grating_duty"])
    n_b = int(round(float(rd["frame_bucket_count"])))
    fill = P.SINGLE_PLY_LEAF_FILL
    for b in range(n_b):
        lo = P.FRAME_BUCKET0 + b * P.FRAME_BUCKET_STEP - P.FRAME_BUCKET_STEP // 2
        hi = lo + P.FRAME_BUCKET_STEP
        m = (zm.frame_level > max(0, lo)) & (zm.frame_level <= hi)
        if not m.any():
            continue
        layers = LF.bucket_layers(fill, b, n_b, 0.0, P.SINGLE_PLY_LEAF_PERIOD_UM,
                                  P.SINGLE_PLY_LEAF_HUE_PERIODS_UM)
        cov[m] = duty * len(layers)          # "crossed" writes two sets, hence the count
        per[m] = layers[0][0] if layers else 0.0
    return cov, per, zm


def render_plate(stem, rgb, subj, fade_field, r, out, face="left"):
    _, spec = wd.blank_plan()
    pspec = spec.faces[face]
    d = wd.die_dims(face)
    W, H = d["f_w"], d["f_h"]
    from app import plates as P
    side = P.CENTERPIECE_FILL * P._aperture(pspec)
    pitch = max(10.0, math.sqrt(W * H / (0.9 * 400000)))
    cov, per, zm = garland_field(pspec, W, H, pitch)
    fh, fw = cov.shape

    # portrait coverage, faded into the field
    gray = rgb @ np.array([0.2126, 0.7152, 0.0722], np.float32)
    dark = ip.prep_darkness(gray, ip.PrepSpec(tone_steps=22))
    n_art = max(8, int(round(side / pitch)))
    cov_art = resize(dark, n_art)
    if fade_field is not None:
        # the study's ground was faded to black; re-target it to the carrier field
        ff = resize(fade_field, n_art, Image.BILINEAR)
        cov_art = cov_art + (1 - ff) * (CARRIER_COV - cov_art)
        fw_map = ff
    else:
        subj_s = resize(subj, n_art)
        if r.get("lift"):
            # the study's ground decays to black around the people; move it
            # most of the way to the carrier field and keep the remainder as
            # the marina's ghost, so the group stands in a slightly darker
            # field rather than in a dark slab inside the frame
            cov_art = np.clip(cov_art + r["lift"] * (CARRIER_COV - cov_art) * (1 - subj_s), 0, 1)
        cov_art, fw_map = context_fade(cov_art, subj_s, *r["fade"])
    ids, pmap, _ = cp.build_period_field(rgb, r["plan"])
    ids_s = np.asarray(Image.fromarray(ids.astype(np.int32)).resize((n_art, n_art), Image.NEAREST))
    lut = np.zeros(int(max(ids.max(), 0)) + 1, np.float32)
    for i, p_ in pmap.items():
        if i < lut.size:
            lut[i] = p_
    per_art = lut[np.clip(ids_s, 0, lut.size - 1)] if lut.size > 1 else np.zeros_like(cov_art)
    per_art = per_art * (fw_map > 0.5)          # colour dies with the detail
    ox, oy = (fw - n_art) // 2, (fh - n_art) // 2
    cov[oy:oy + n_art, ox:ox + n_art] = cov_art
    per[oy:oy + n_art, ox:ox + n_art] = per_art

    k = max(1, int(round(EYE_UM / pitch)))
    def bm(a):
        m, n = a.shape[0] - a.shape[0] % k, a.shape[1] - a.shape[1] % k
        return a[:m, :n].reshape(m // k, k, n // k, k).mean(axis=(1, 3))
    cov_e = bm(cov)
    has = bm((per > 0).astype(np.float32))
    per_e = np.where(has > 0.5, bm(per) / np.maximum(1e-6, has), 0.0)
    tiles = []
    for tilt in (0.0, 1.0):
        u = per_e * (K0 + math.sin(math.radians(tilt)) * TILT_GAIN)
        spec_rgb = sheen(u.astype(np.float32))
        o = GOLD * ETA0 * cov_e[..., None]
        o = o + np.where((per_e > 0)[..., None], spec_rgb * ETA1 * cov_e[..., None], 0.0)
        o = o / ETA0
        img = np.clip(ip.linear_to_srgb(np.clip(o, 0, 1)), 0, 1)
        scale = 700 / img.shape[0]
        tiles.append(Image.fromarray((img * 255).astype(np.uint8)).resize((int(img.shape[1] * scale), 700), Image.LANCZOS))
    src = Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8)).resize((int(700 * side / H),) * 2, Image.LANCZOS)
    Wp = 10 + src.width + 10 + sum(t.width + 10 for t in tiles)
    sheet = Image.new("RGB", (Wp, 700 + 70), BG)
    dr = ImageDraw.Draw(sheet)
    x = 10
    sheet.paste(src, (x, 10 + (700 - src.height) // 2)); dr.text((x, 718), "prepared source (art box scale)", fill=DIM, font=font(11)); x += src.width + 10
    for t, tilt in zip(tiles, (0.0, 1.0)):
        sheet.paste(t, (x, 10)); dr.text((x, 718), f"the {face} plate, tilt {tilt:g}°  ({W/1000:.1f} × {H/1000:.2f} mm, art box {side/1000:.1f} mm)", fill=FG, font=font(11)); x += t.width + 10
    dr.text((10, 740), f"{stem}: {r['why']}. Garland: fine leaf gratings on the one ply, one period per motif family, no carrier; the portrait dissolves to bare glass; eye-cell integrated, lamp.", fill=DIM, font=font(10))
    p = out / f"{stem}_plate.png"
    sheet.save(p)
    print("  saved", p)
    Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8)).save(out / f"{stem}_source.png")
    return {"plate": p.name, "plate_px": (tiles[0].width, 700), "src_px": src.width, "art_box_um": round(side, 1),
            "fade": r["fade"], "coloured_fraction": round(float((ids > 0).mean()), 3), "plan": r["plan"].name}


def grid(out, results, cols=3):
    tiles = []
    for res in results:
        sheet = Image.open(out / res["plate"])
        x0 = 10 + res["src_px"] + 10
        tiles.append((res["stem"], sheet.crop((x0, 10, x0 + res["plate_px"][0], 710))))
    tw, th = tiles[0][1].size
    rows = (len(tiles) + cols - 1) // cols
    im = Image.new("RGB", (10 + cols * (tw + 10), 10 + rows * (th + 34)), BG)
    d = ImageDraw.Draw(im)
    for i, (stem, t) in enumerate(tiles):
        x = 10 + (i % cols) * (tw + 10); y = 10 + (i // cols) * (th + 34)
        im.paste(t, (x, y)); d.text((x, y + th + 8), stem, fill=FG, font=font(12))
    p = out / "side_plates_grid.png"
    im.save(p); print("  saved", p)


def main():
    photos = Path(sys.argv[1]); out = Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
    fade_dir = Path(sys.argv[3]) if len(sys.argv) > 3 else out.parent / "fade"
    results = []
    for stem, r in RECIPES.items():
        print(stem)
        rgb, subj, ff, info = prepare(stem, r, photos, fade_dir)
        res = render_plate(stem, rgb, subj, ff, r, out)
        results.append({"stem": stem, "file": r["file"], "crop": r["crop"], "why": r["why"], **info, **res})
    (out / "side_plates.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    grid(out, results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
