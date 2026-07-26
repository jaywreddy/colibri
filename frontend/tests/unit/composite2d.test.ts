/**
 * Pattern Lab 2D compositor — formula parity + compositing rules.
 *
 * parallaxShiftUm must match src/shaders/lib/parallax.glsl exactly:
 *   sinSub = sin(tilt)/n; shift = t * sinSub / max(0.05, cosSub)
 * compositeParallax must match the simplified plate.frag rules (back layer
 * sampled at uv - shift; transmission/reflected shading per illumination).
 */
import { describe, it, expect } from 'vitest';
import {
  GOLD,
  GOLD_BACK,
  compositeParallax,
  grayFromRgba,
  parallaxShiftUm,
  sampleBilinear,
  tiltForShiftUm,
  type ImageDataLike,
} from '../../src/lab/composite2d';

/** Uniform single-channel grid. */
const grid = (w: number, h: number, fill = 0): ImageDataLike => ({
  width: w,
  height: h,
  data: new Uint8ClampedArray(w * h).fill(fill),
});

/** Grid with one bright dot at (x, y). */
const dotGrid = (w: number, h: number, x: number, y: number): ImageDataLike => {
  const g = grid(w, h, 0);
  (g.data as Uint8ClampedArray)[y * w + x] = 255;
  return g;
};

/** RGBA of output pixel (x, y). */
const px = (out: Uint8ClampedArray, w: number, x: number, y: number) => {
  const o = (y * w + x) * 4;
  return [out[o], out[o + 1], out[o + 2], out[o + 3]] as const;
};

describe('parallaxShiftUm', () => {
  it('matches the shared Snell formula at 20 deg, t=500 um, n=1.46', () => {
    // Expected value computed independently with the shared formula.
    const sinV = Math.sin((20 * Math.PI) / 180);
    const sinSub = sinV / 1.46;
    const cosSub = Math.sqrt(Math.max(0, 1 - sinSub * sinSub));
    const expected = (500 * sinSub) / Math.max(0.05, cosSub);

    const { dxUm, dyUm } = parallaxShiftUm(20, 0, 500, 1.46);
    expect(dxUm).toBeCloseTo(expected, 6);
    // Hard numeric anchor so a formula typo can't hide in both sides.
    expect(dxUm).toBeCloseTo(120.48275, 4);
    expect(dyUm).toBe(0);
  });

  it('zero tilt -> zero shift', () => {
    const { dxUm, dyUm } = parallaxShiftUm(0, 0, 500, 1.46);
    expect(dxUm).toBe(0);
    expect(dyUm).toBe(0);
  });

  it('y tilt shifts along y only; sign follows the tilt', () => {
    const y = parallaxShiftUm(0, 20, 500, 1.46);
    expect(y.dxUm).toBe(0);
    expect(y.dyUm).toBeCloseTo(120.48275, 4);

    const neg = parallaxShiftUm(-20, 0, 500, 1.46);
    expect(neg.dxUm).toBeCloseTo(-120.48275, 4);
  });

  it('scales linearly with thickness', () => {
    const t500 = parallaxShiftUm(15, 0, 500, 1.46).dxUm;
    const t1000 = parallaxShiftUm(15, 0, 1000, 1.46).dxUm;
    expect(t1000).toBeCloseTo(2 * t500, 9);
  });
});

describe('tiltForShiftUm', () => {
  it('round-trips parallaxShiftUm to 1e-6 over 0..200 um at t=500 um, n=1.46', () => {
    // Zone calibration depends on this inverse being exact: the Pattern Lab
    // quick-sets convert ±p/2 shifts to tilts, and the composite must land
    // on the requested shift, not merely near it. toBeCloseTo(x, 6) asserts
    // |err| < 5e-7 um.
    for (let s = 0; s <= 200; s += 2.5) {
      const tiltPos = tiltForShiftUm(s, 500, 1.46);
      expect(parallaxShiftUm(tiltPos, 0, 500, 1.46).dxUm).toBeCloseTo(s, 6);
      // Sign symmetry: negative shifts invert cleanly too.
      const tiltNeg = tiltForShiftUm(-s, 500, 1.46);
      expect(tiltNeg).toBeCloseTo(-tiltPos, 9);
      expect(parallaxShiftUm(tiltNeg, 0, 500, 1.46).dxUm).toBeCloseTo(-s, 6);
    }
  });

  it('anchors the first-zone table: p/2 = 20 um -> 3.35 deg (t=500, n=1.46)', () => {
    // theta(s) = asin(n * sin(atan(s / t))); cross-checked against
    // app.sim2d.parallax_shift_um (20.03 um at 3.35 deg).
    expect(tiltForShiftUm(20, 500, 1.46)).toBeCloseTo(3.345, 2);
    expect(tiltForShiftUm(40, 500, 1.46)).toBeCloseTo(6.688, 2);
    expect(tiltForShiftUm(84, 500, 1.46)).toBeCloseTo(13.99, 1);
  });

  it('degenerate inputs: zero shift and zero thickness map to zero tilt', () => {
    expect(tiltForShiftUm(0, 500, 1.46)).toBe(0);
    expect(tiltForShiftUm(50, 0, 1.46)).toBe(0);
  });

  it('clamps shifts beyond the grazing limit to +/-90 deg', () => {
    // |shift| > t/sqrt(n^2 - 1) is unreachable from outside the plate.
    expect(tiltForShiftUm(1e6, 500, 1.46)).toBeCloseTo(90, 6);
    expect(tiltForShiftUm(-1e6, 500, 1.46)).toBeCloseTo(-90, 6);
  });
});

describe('compositeParallax', () => {
  it('all-gold front kills transmission (laser light blocked)', () => {
    const w = 4;
    const open = compositeParallax(grid(w, w, 0), grid(w, w, 0), 0, 0, 'laser');
    const blocked = compositeParallax(grid(w, w, 255), grid(w, w, 0), 0, 0, 'laser');
    for (let y = 0; y < w; y++) {
      for (let x = 0; x < w; x++) {
        // Blank pair: full transmission -> laser green saturates.
        expect(px(open, w, x, y)[1]).toBe(255);
        // Gold front: only GOLD*0.12*reflected remains — green collapses.
        expect(px(blocked, w, x, y)[1]).toBeLessThan(30);
      }
    }
  });

  it('backlight on a blank pair -> pure white pixels', () => {
    const w = 2;
    const out = compositeParallax(grid(w, w, 0), grid(w, w, 0), 0, 0, 'backlight');
    for (let i = 0; i < out.length; i++) expect(out[i]).toBe(255);
  });

  it('+x shift moves a back dot in +x (back sampled at x - dx)', () => {
    const w = 4;
    const front = grid(w, w, 0);
    const back = dotGrid(w, w, 1, 1);
    const out = compositeParallax(front, back, 1, 0, 'backlight');
    // Dot lands at (2,1): transmission closes -> dark against a white field.
    expect(px(out, w, 2, 1)[0]).toBeLessThan(60);
    // Original position and the rest of the row stay white.
    expect(px(out, w, 1, 1)[0]).toBe(255);
    expect(px(out, w, 0, 1)[0]).toBe(255);
    expect(px(out, w, 3, 1)[0]).toBe(255);
  });

  it('laser tint: blank pair renders the default green laser color', () => {
    const w = 2;
    const out = compositeParallax(grid(w, w, 0), grid(w, w, 0), 0, 0, 'laser');
    const [r, g, b] = px(out, w, 0, 0);
    expect(g).toBe(255); // 1.0 * 255
    expect(Math.abs(r - 0.267 * 255)).toBeLessThanOrEqual(1);
    expect(Math.abs(b - 0.533 * 255)).toBeLessThanOrEqual(1);
  });

  it('ambient: all-gold front reflects the GOLD tint', () => {
    const w = 2;
    const out = compositeParallax(grid(w, w, 255), grid(w, w, 0), 0, 0, 'ambient');
    const [r, g, b] = px(out, w, 0, 0);
    expect(Math.abs(r - GOLD[0] * 0.85 * 255)).toBeLessThanOrEqual(1);
    expect(Math.abs(g - GOLD[1] * 0.85 * 255)).toBeLessThanOrEqual(1);
    expect(Math.abs(b - GOLD[2] * 0.85 * 255)).toBeLessThanOrEqual(1);
    expect(r).toBeGreaterThan(g);
    expect(g).toBeGreaterThan(b);
  });

  it('ambient: f=1 pixels still track the back layer (overlap darkening)', () => {
    // Paired pin with backend tests/test_sim2d.py
    // ::test_ambient_overlap_darkening_is_the_back_layer_dependence.
    // Hand-computed from plate.frag's ambient formula at front = 1:
    //   reflected = 1, transmission = 0, overlap = back
    //   rgb = GOLD * 0.85 * (1 - 0.35 * back)
    //   back=0 -> (195.5, 159.7, 68.1)   back=1 -> (127.1, 103.8, 44.2)
    // Drop the overlap term and both states collapse to the same constant:
    // the lab's ambient tilt preview goes flat over the gold figure, which is
    // precisely where the fringe modulation is supposed to be read.
    const w = 2;
    const clear = compositeParallax(grid(w, w, 255), grid(w, w, 0), 0, 0, 'ambient');
    const covered = compositeParallax(grid(w, w, 255), grid(w, w, 255), 0, 0, 'ambient');
    for (let c = 0; c < 3; c++) {
      expect(Math.abs(px(clear, w, 0, 0)[c] - GOLD[c] * 0.85 * 255)).toBeLessThanOrEqual(1);
      expect(
        Math.abs(px(covered, w, 0, 0)[c] - GOLD[c] * 0.85 * 0.65 * 255)
      ).toBeLessThanOrEqual(1);
    }
    // The 35% swing itself, not just the endpoints.
    expect(px(covered, w, 0, 0)[0] / px(clear, w, 0, 0)[0]).toBeCloseTo(0.65, 2);
  });

  it('ambient: the clear-pair floor is the shader 0.04 transmission term', () => {
    // Blank pair: reflected = 0, transmission = 1, overlap = 0 ->
    // 0.04 * 255 = 10.2 on every channel (the shader constant; both 2D paths
    // used to carry 0.06 = 15.3). Pinned identically in test_sim2d.py.
    const w = 2;
    const out = compositeParallax(grid(w, w, 0), grid(w, w, 0), 0, 0, 'ambient');
    for (const v of px(out, w, 0, 0).slice(0, 3)) expect(v).toBe(10);
  });

  it('out-of-bounds back samples read as no gold', () => {
    const w = 4;
    // All-gold back shifted fully off-grid: backlight goes fully white.
    const out = compositeParallax(grid(w, w, 0), grid(w, w, 255), w, 0, 'backlight');
    for (let y = 0; y < w; y++) {
      for (let x = 0; x < w; x++) expect(px(out, w, x, y)[0]).toBe(255);
    }
  });

  it('backlight gold-on-gold uses GOLD_BACK * 0.25 reflectance', () => {
    const w = 2;
    const out = compositeParallax(grid(w, w, 255), grid(w, w, 255), 0, 0, 'backlight');
    const [r] = px(out, w, 0, 0);
    // reflected = max(1, 0.55) = 1 -> r = GOLD_BACK[0] * 0.25 * 255 = 25.5
    expect(Math.abs(r - GOLD_BACK[0] * 0.25 * 255)).toBeLessThanOrEqual(1);
  });

  it('rejects mismatched mask dimensions', () => {
    expect(() =>
      compositeParallax(grid(2, 2), grid(4, 4), 0, 0, 'ambient')
    ).toThrow(/mask dims differ/);
  });
});

describe('sampleBilinear / grayFromRgba', () => {
  it('interpolates between neighbors and reads 0 out of bounds', () => {
    const g = dotGrid(4, 4, 1, 1);
    expect(sampleBilinear(g, 1, 1)).toBe(1);
    expect(sampleBilinear(g, 1.5, 1)).toBeCloseTo(0.5, 9);
    expect(sampleBilinear(g, -1, 1)).toBe(0);
    expect(sampleBilinear(g, 1, 10)).toBe(0);
  });

  it('grayFromRgba takes red weighted by alpha', () => {
    const rgba = new Uint8ClampedArray([
      255, 0, 0, 255, // opaque red -> 255
      100, 0, 0, 127, // half-transparent -> ~50
    ]);
    const g = grayFromRgba(rgba, 2, 1);
    expect(g.data[0]).toBe(255);
    expect(Math.abs((g.data[1] as number) - 50)).toBeLessThanOrEqual(1);
  });
});
