"""Colour treatments for candidate side-plate photographs.

For each photo (crop from the prep notes) this renders, with the same appearance
model as the witness previews (specular gold + first-order sheen, eye-cell
integrated):

  * the cropped square,
  * the period field each colour plan assigns — a false-colour map of ladder
    rungs, grey where a region stays plain gold,
  * the rendered plate at tilt 0 and 1 deg for PLAIN, HUE (every pixel's own
    hue), HUE EQUALISED (the rungs spread over the picture's hue range) and
    AUTO ZONES (only saturated regions coloured, coarse patches — the closest
    thing to an authored zone plan without authoring one).

    uv run --directory backend python ../tools/dev/render_photo_colour.py NOTES_JSON [NOTES_JSON...] OUTDIR

One photo at a time, 1000 px working size; nothing here is heavy.
"""
from __future__ import annotations

import colorsys
import json
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "backend"))
sys.path.insert(0, str(HERE))

from app.patterns.bitmap import colourplan as cp  # noqa: E402
from app.patterns.bitmap import imageprep as ip  # noqa: E402
from render_witness_preview import ETA0, ETA1, EYE_UM, GOLD, K0, TILT_GAIN, block_mean, block_mode, sheen  # noqa: E402

ROOT = HERE.parents[1]
SIDE_UM = 15425.0          # the side plate's art box
SRC_PX = 1000
BG = (13, 17, 19)
FG = (214, 224, 227)
DIM = (150, 165, 172)

PLANS = {
    "plain": cp.ColourPlan(name="plain", mode="plain"),
    "hue": cp.ColourPlan(name="hue", mode="hue", base_period_um=5.0, ladder_steps=12, spread=1.45,
                         coarsen_px=7, hue_equalize=False),
    "hue-eq": cp.ColourPlan(name="hue-eq", mode="hue", base_period_um=5.0, ladder_steps=12, spread=1.45,
                            coarsen_px=7, hue_equalize=True),
    "auto-zones": cp.ColourPlan(name="auto-zones", mode="hue", base_period_um=5.0, ladder_steps=12, spread=1.45,
                                coarsen_px=14, hue_equalize=False, hue_min_sat=0.28, hue_min_value=0.22),
}


def font(sz=12):
    try:
        return ImageFont.truetype("arialbd.ttf", sz)
    except Exception:
        return ImageFont.load_default()


def rung_palette(n: int) -> np.ndarray:
    """Rung 0 = red end (long period) ... rung n-1 = blue end. Index 0 of the
    returned table is 'plain' (grey)."""
    pal = np.zeros((n + 1, 3), np.float32)
    pal[0] = (0.35, 0.35, 0.35)
    for i in range(n):
        h = 0.0 + 0.72 * i / max(1, n - 1)       # red -> violet
        pal[i + 1] = colorsys.hsv_to_rgb(h, 0.85, 0.95)
    return pal


def render(dark: np.ndarray, ids: np.ndarray, periods: dict, tilt_deg: float, px: int) -> Image.Image:
    k = max(1, int(round(EYE_UM / (SIDE_UM / dark.shape[0]))))
    cov = np.clip(block_mean(dark, k), 0.0, 1.0)
    n_ids = int(max(ids.max(), 0)) + 1
    pid = block_mode(ids, k, n_ids) if n_ids > 1 else np.zeros(cov.shape, np.int32)
    lut = np.zeros(n_ids, np.float32)
    for i, d in periods.items():
        if i < n_ids:
            lut[i] = d
    d_map = lut[pid]
    u = d_map * (K0 + math.sin(math.radians(tilt_deg)) * TILT_GAIN)
    spec_rgb = sheen(u.astype(np.float32))
    out = GOLD * ETA0 * cov[..., None]
    out = out + np.where((d_map > 0)[..., None], spec_rgb * ETA1 * cov[..., None], 0.0)
    out = out / ETA0
    img = np.clip(ip.linear_to_srgb(np.clip(out, 0, 1)), 0, 1)
    return Image.fromarray((img * 255).astype(np.uint8)).resize((px, px), Image.LANCZOS)


def field_image(ids: np.ndarray, periods: dict, px: int) -> tuple[Image.Image, dict]:
    n = max(len(periods), 1)
    # map period id -> rung rank by period (long = red)
    order = sorted(periods.items(), key=lambda kv: -kv[1])
    rank = {pid_: r + 1 for r, (pid_, _) in enumerate(order)}
    pal = rung_palette(len(order))
    idx = np.zeros(ids.shape, np.int32)
    for pid_, r in rank.items():
        idx[ids == pid_] = r
    im = Image.fromarray((pal[idx] * 255).astype(np.uint8)).resize((px, px), Image.NEAREST)
    cover = float((ids > 0).mean()) if periods else 0.0
    return im, {"rungs_used": len(order), "coloured_fraction": round(cover, 3),
                "periods_um": [round(v, 2) for _, v in order]}


def source_for(entry: dict, notes_dir: Path) -> tuple[Path, tuple[float, float, float]]:
    f = entry["file"]
    stem = Path(f).stem
    sq = notes_dir / f"{stem}_square.png"
    if entry.get("extended") and (entry["extended"] is True or (isinstance(entry["extended"], dict) and
                                                               entry["extended"].get("extended") or entry["extended"].get("value"))) and sq.exists():
        return sq, (0.0, 0.0, 1.0)
    path = ROOT / f if not f.startswith("photos/") else ROOT / f
    if not path.exists():
        path = ROOT / "photos" / Path(f).name
    return path, tuple(entry["crop"])


def sheet(entry: dict, notes_dir: Path, out: Path) -> dict:
    path, crop = source_for(entry, notes_dir)
    stem = Path(entry["file"]).stem
    gray = ip.load_gray(path, crop=crop, size=SRC_PX)
    rgb = cp.load_rgb(path, crop=crop, size=SRC_PX)
    dark = ip.prep_darkness(gray, ip.PrepSpec(tone_steps=22))
    P = 300
    cols = ["plain", "hue", "hue-eq", "auto-zones"]
    fields, renders, stats = {}, {}, {}
    for key in cols:
        ids, periods, rep = cp.build_period_field(rgb, PLANS[key])
        fields[key], st = field_image(ids, periods, P)
        st["hue_equalize"] = PLANS[key].hue_equalize
        stats[key] = st
        renders[key] = [render(dark, ids, periods, t, P) for t in (0.0, 1.0)]
    W = 10 + (len(cols) + 1) * (P + 10)
    H = 10 + 3 * (P + 26) + 40
    im = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(im)
    src = Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8)).resize((P, P), Image.LANCZOS)
    im.paste(src, (10, 10)); d.text((10, P + 14), f"{stem}  crop {tuple(round(c, 3) for c in crop)}", fill=FG, font=font(11))
    eye = Image.fromarray((np.clip(1 - block_mean(dark, 5), 0, 1) * 255).astype(np.uint8)).resize((P, P), Image.NEAREST).convert("RGB")
    im.paste(eye, (10, 10 + P + 26)); d.text((10, 2 * P + 40), "tone as the eye sees it (87 um cells)", fill=DIM, font=font(11))
    for j, key in enumerate(cols):
        x = 10 + (j + 1) * (P + 10)
        im.paste(fields[key], (x, 10))
        st = stats[key]
        d.text((x, P + 14), f"{key.upper()}: period field, {st['rungs_used']} rungs, {st['coloured_fraction']:.0%} coloured", fill=FG, font=font(11))
        im.paste(renders[key][0], (x, 10 + P + 26)); d.text((x, 2 * P + 40), f"{key} · tilt 0°", fill=DIM, font=font(11))
        im.paste(renders[key][1], (x, 10 + 2 * (P + 26))); d.text((x, 3 * P + 66), f"{key} · tilt 1°", fill=DIM, font=font(11))
    d.text((10, H - 26), "period field: red = long period (6.0 um) ... violet = short (4.15 um), grey = plain gold. Renders: specular gold + first-order sheen, lamp, eye-cell integrated, 15.4 mm art box.",
           fill=DIM, font=font(10))
    p = out / f"{stem}_colour.png"
    im.save(p)
    print("  saved", p)
    return {"file": entry["file"], "crop": list(crop), "source": str(path), "sheet": p.name, "stats": stats}


def main() -> int:
    args = sys.argv[1:]
    out = Path(args[-1]); out.mkdir(parents=True, exist_ok=True)
    entries = []
    for nj in args[:-1]:
        d = json.load(open(nj, encoding="utf-8"))
        items = d if isinstance(d, list) else d.get("photos", list(d.values()))
        for e in items:
            if isinstance(e, dict) and "file" in e:
                e["_dir"] = str(Path(nj).parent)
                entries.append(e)
    seen, results = set(), []
    for e in entries:
        stem = Path(e["file"]).stem
        if stem in seen:
            continue
        seen.add(stem)
        results.append(sheet(e, Path(e["_dir"]), out))
    (out / "photo_colour.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
