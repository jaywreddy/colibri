/**
 * Pattern Lab 2D compositor — the pure-math half of the lab preview.
 *
 * Mirrors the physical model of the 3D renderer in flat 2D: the
 * Snell-refracted parallax of src/shaders/lib/parallax.glsl slides the back
 * gold mask under the front one, and simplified moire_interactive
 * illumination rules from src/shaders/plate.frag shade the resulting
 * (transmission, reflected) pair. Deliberately DOM-free (no canvas, no
 * ImageData) so vitest exercises the exact formulas headless;
 * PatternLab.tsx owns the canvas plumbing.
 */

export type ImageDataLike = {
  width: number;
  height: number;
  /** Single-channel gold mask, row-major, 0..255 where 255 = gold. */
  data: Uint8ClampedArray | number[];
};

export type Illumination2D = 'ambient' | 'laser' | 'backlight';

export type Rgb = readonly [number, number, number];

/** Gold tints — keep in sync with GOLD / GOLD_BACK in plate.frag. */
export const GOLD: Rgb = [0.902, 0.737, 0.314];
export const GOLD_BACK: Rgb = [0.4, 0.32, 0.12];
/** Default laser color (green) — matches the 3D scene's default uLaserColor. */
export const DEFAULT_LASER_COLOR: Rgb = [0.267, 1.0, 0.533];

/**
 * Snell-refracted lateral shift through a slab, one axis at a time:
 *   sinSub = sin(tilt) / n;  shift = t · sinSub / max(0.05, cosSub)
 * Identical to parallax_offset in src/shaders/lib/parallax.glsl (including
 * the 0.05 cosine floor that clamps grazing angles). Sign follows the tilt.
 */
function axisShiftUm(tiltDeg: number, thicknessUm: number, n: number): number {
  const sinV = Math.sin((tiltDeg * Math.PI) / 180);
  const sinSub = sinV / n;
  const cosSub = Math.sqrt(Math.max(0, 1 - sinSub * sinSub));
  return (thicknessUm * sinSub) / Math.max(0.05, cosSub);
}

/**
 * Parallax shift of the back layer for a plate tilted by (tiltX, tiltY)
 * degrees: x tilt shifts along x and y tilt along y, each through the shared
 * one-axis formula. Returns micrometers; positive tilt -> positive shift.
 */
export function parallaxShiftUm(
  tiltXDeg: number,
  tiltYDeg: number,
  thicknessUm: number,
  n: number
): { dxUm: number; dyUm: number } {
  return {
    dxUm: axisShiftUm(tiltXDeg, thicknessUm, n),
    dyUm: axisShiftUm(tiltYDeg, thicknessUm, n),
  };
}

/**
 * Inverse of the one-axis parallax formula: the exterior tilt (degrees) that
 * produces a given back-layer shift through a slab of thickness t and index
 * n. Inside the substrate shift = t · tan(θ_sub), and Snell gives
 * sin(tilt) = n · sin(θ_sub), so
 *   tilt = asin(n · sin(atan(shift / t)))
 * Exact inverse of parallaxShiftUm per axis wherever the 0.05 cosine floor
 * is inactive (|shift|/t below ~20 — every practical zone shift; the floor
 * only bites within a fraction of a degree of grazing). Zone calibration
 * lives on this: switch/reveal patterns complete within the FIRST
 * half-period of shift and alias with period p, so the lab converts p/2 to
 * an exact tilt instead of letting users overshoot into repeated zones.
 *
 * The asin argument is clamped to ±1: shifts beyond the internal grazing
 * limit (|shift| > t/√(n²−1)) are unreachable from outside the plate and
 * clamp to ±90°. thickness ≤ 0 returns 0 (every tilt produces zero shift;
 * 0 is the canonical preimage).
 */
export function tiltForShiftUm(shiftUm: number, thicknessUm: number, n: number): number {
  if (thicknessUm <= 0) return 0;
  const sinTilt = n * Math.sin(Math.atan(shiftUm / thicknessUm));
  return (Math.asin(Math.max(-1, Math.min(1, sinTilt))) * 180) / Math.PI;
}

/** Bilinear sample of a single-channel mask, normalized to 0..1. Out-of-bounds reads 0 (no gold). */
export function sampleBilinear(img: ImageDataLike, x: number, y: number): number {
  const { width, height, data } = img;
  const x0 = Math.floor(x);
  const y0 = Math.floor(y);
  const fx = x - x0;
  const fy = y - y0;
  const at = (px: number, py: number): number =>
    px >= 0 && px < width && py >= 0 && py < height ? data[py * width + px] / 255 : 0;
  return (
    at(x0, y0) * (1 - fx) * (1 - fy) +
    at(x0 + 1, y0) * fx * (1 - fy) +
    at(x0, y0 + 1) * (1 - fx) * fy +
    at(x0 + 1, y0 + 1) * fx * fy
  );
}

/**
 * Composite the two gold masks into an RGBA buffer (Uint8ClampedArray of
 * length 4·w·h, alpha 255) under one of the three illumination modes.
 *
 * Sign convention matches plate.frag (uBack sampled at vUv - shift): the
 * back layer is sampled at (x - dxPx, y - dyPx), so a positive shift slides
 * back-mask features toward +x/+y on screen. Per pixel, with f/b the front
 * and shifted-back gold coverage in 0..1:
 *   transmission = (1 - f) · (1 - b)
 *   reflected    = max(f, 0.55 · b)
 *   ambient  : (GOLD · reflected · 0.85 + 0.04 · transmission) · (1 - 0.35 · f · b)
 *   laser    : laserColor · transmission + GOLD · 0.12 · reflected
 *   backlight: (1,1,1) · transmission + GOLD_BACK · reflected · 0.25
 *
 * The ambient overlap darkening is plate.frag's ONLY back-layer dependence
 * wherever the front mask is gold — there `reflected` saturates at 1 and
 * transmission is 0, so without it every f=1 pixel is a constant and the
 * lab's ambient tilt preview shows no fringe modulation over the gold figure
 * (exactly where the designer looks). Laser and backlight carry no overlap
 * factor in the shader; do not add one here. The 0.85 / 0.12 / 0.25 factors
 * stand in for the shader's head-on Lambert terms.
 */
export function compositeParallax(
  front: ImageDataLike,
  back: ImageDataLike,
  dxPx: number,
  dyPx: number,
  illum: Illumination2D,
  laserColor: Rgb = DEFAULT_LASER_COLOR
): Uint8ClampedArray {
  if (front.width !== back.width || front.height !== back.height) {
    throw new Error(
      `compositeParallax: mask dims differ ` +
        `(${front.width}x${front.height} vs ${back.width}x${back.height})`
    );
  }
  const { width, height, data: fdata } = front;
  const bdata = back.data;
  const out = new Uint8ClampedArray(width * height * 4);
  // Whole-pixel shifts (the head-on 0/0 case and every integer slide) make the
  // bilinear weights exactly 1 and 0, so the nearest read is not an
  // approximation — it is the same arithmetic with the three zero-weighted
  // terms elided.
  const nearest = Number.isInteger(dxPx) && Number.isInteger(dyPx);
  // Loop-invariant tint constants. Each keeps the SAME multiplication grouping
  // as the documented formulas above: this function's output is pinned to the
  // byte in tests/unit/composite2d.test.ts and paired with the backend's
  // test_sim2d.py, so re-associating float products is not a free rewrite.
  const [gr, gg, gb] = GOLD;
  const [br, bg, bb] = GOLD_BACK;
  const [lr, lg, lb] = laserColor;
  const grLaser = gr * 0.12;
  const ggLaser = gg * 0.12;
  const gbLaser = gb * 0.12;
  for (let y = 0; y < height; y++) {
    // The back-mask row and its bilinear y weight are the same for the whole
    // scanline; only the x term moves.
    const sy = y - dyPx;
    const y0 = Math.floor(sy);
    const fy = sy - y0;
    const omfy = 1 - fy;
    const y0ok = y0 >= 0 && y0 < height;
    const y1ok = y0 + 1 >= 0 && y0 + 1 < height;
    const row0 = y0 * width;
    const row1 = row0 + width;
    for (let x = 0; x < width; x++) {
      const i = y * width + x;
      const f = fdata[i] / 255;
      // Inlined sampleBilinear (same term order, same grouping, same
      // out-of-bounds-reads-0 rule) — the closure it allocates per pixel is
      // the single hottest cost of a lab recomposite.
      const sx = x - dxPx;
      const x0 = Math.floor(sx);
      let b: number;
      if (nearest) {
        b = y0ok && x0 >= 0 && x0 < width ? bdata[row0 + x0] / 255 : 0;
      } else {
        const fx = sx - x0;
        const omfx = 1 - fx;
        const x0ok = x0 >= 0 && x0 < width;
        const x1ok = x0 + 1 >= 0 && x0 + 1 < width;
        const s00 = y0ok && x0ok ? bdata[row0 + x0] / 255 : 0;
        const s10 = y0ok && x1ok ? bdata[row0 + x0 + 1] / 255 : 0;
        const s01 = y1ok && x0ok ? bdata[row1 + x0] / 255 : 0;
        const s11 = y1ok && x1ok ? bdata[row1 + x0 + 1] / 255 : 0;
        b = s00 * omfx * omfy + s10 * fx * omfy + s01 * omfx * fy + s11 * fx * fy;
      }
      const transmission = (1 - f) * (1 - b);
      const reflected = Math.max(f, 0.55 * b);
      let r: number;
      let g: number;
      let bl: number;
      if (illum === 'laser') {
        r = lr * transmission + grLaser * reflected;
        g = lg * transmission + ggLaser * reflected;
        bl = lb * transmission + gbLaser * reflected;
      } else if (illum === 'backlight') {
        r = transmission + br * reflected * 0.25;
        g = transmission + bg * reflected * 0.25;
        bl = transmission + bb * reflected * 0.25;
      } else {
        // plate.frag darkens the whole ambient color (gold shade AND the
        // transmission floor) where both layers are gold.
        const overlapDark = 1 - 0.35 * f * b;
        r = (gr * reflected * 0.85 + 0.04 * transmission) * overlapDark;
        g = (gg * reflected * 0.85 + 0.04 * transmission) * overlapDark;
        bl = (gb * reflected * 0.85 + 0.04 * transmission) * overlapDark;
      }
      const o = i * 4;
      out[o] = r * 255; // Uint8ClampedArray assignment clamps + rounds
      out[o + 1] = g * 255;
      out[o + 2] = bl * 255;
      out[o + 3] = 255;
    }
  }
  return out;
}

/**
 * Collapse an RGBA pixel buffer (e.g. canvas getImageData().data) into a
 * single-channel gold mask. Uses the red channel like the shaders
 * (texture2D(...).r), weighted by alpha so transparent pixels read as
 * no-gold regardless of their color.
 */
export function grayFromRgba(
  rgba: Uint8ClampedArray | number[],
  width: number,
  height: number
): ImageDataLike {
  const data = new Uint8ClampedArray(width * height);
  for (let i = 0; i < width * height; i++) {
    data[i] = (rgba[i * 4] * rgba[i * 4 + 3]) / 255;
  }
  return { width, height, data };
}
