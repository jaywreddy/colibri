/**
 * Contract math for the Ring Box assembly (v2). These numbers are pinned to
 * the design contract — the backend's assembly.py must produce identical
 * values for the same spec, so a red test here means the two sides drifted.
 */
import { describe, it, expect } from 'vitest';
import { defaultBoxSpec, type BoxSpec } from '../../src/api';
import {
  copperTapeLengthCm,
  cutList,
  hingeLayout,
  keepoutUm,
  overlapUm,
  platePlacements,
  seamSegments,
  stampFaces,
  validateBox,
  SEGMENT_GAP_UM,
} from '../../src/assembly';

/**
 * The 50 x 50 x 40 mm / 0.5 mm single-plate geometry every number in this file
 * is pinned to.
 *
 * Written out here rather than taken from `defaultBoxSpec()`: this suite tests
 * the ASSEMBLY FORMULAS against the design contract, and it used to ride on the
 * default box, so the day the default became the PRODUCTION box (32 x 32 x
 * 35 mm, bonded 2.25 mm quartz plies) every pinned number here went red
 * without a single formula changing. The default box's own values are pinned in
 * api.test.ts, where they belong; the golden-fixture suite
 * (assemblyGolden.test.ts) is what cross-checks the formulas against the
 * backend, including the bonded cases.
 */
const spec = (): BoxSpec => {
  const s = defaultBoxSpec();
  return {
    ...s,
    width_um: 50000,
    depth_um: 50000,
    height_um: 40000,
    glass: { thickness_um: 500, material: 'fused silica', n: 1.46 },
    // The classic single-ply numbers below assume 1/4" tape (the production
    // default is now 3/8" for the bonded quartz stack).
    foil: { ...s.foil, tape_width_um: 6350 },
    bonded: false,
  };
};

describe('foil overlap + keep-out', () => {
  it('defaults (1/4" tape, 500 um glass, 500 um safety): overlap 2925, keepout 3425', () => {
    expect(overlapUm(spec())).toBe(2925);
    expect(keepoutUm(spec())).toBe(3425);
  });

  it('overlap floors at zero for tape narrower than the glass', () => {
    const s = spec();
    s.foil.tape_width_um = 400;
    expect(overlapUm(s)).toBe(0);
    expect(keepoutUm(s)).toBe(500);
  });
});

describe('cut list (50 x 50 x 40 mm, t = 0.5 mm)', () => {
  it('matches the contract formulas exactly', () => {
    const cuts = Object.fromEntries(cutList(spec()).map((c) => [c.face, c]));
    expect(cuts.bottom.width_mm).toBe(50.0);
    expect(cuts.bottom.height_mm).toBe(50.0);
    expect(cuts.top.width_mm).toBe(50.0);
    expect(cuts.top.height_mm).toBe(50.0);
    expect(cuts.front.width_mm).toBe(50.0);
    expect(cuts.front.height_mm).toBe(39.0);
    expect(cuts.back.width_mm).toBe(50.0);
    expect(cuts.back.height_mm).toBe(39.0);
    expect(cuts.left.width_mm).toBe(49.0);
    expect(cuts.left.height_mm).toBe(39.0);
    expect(cuts.right.width_mm).toBe(49.0);
    expect(cuts.right.height_mm).toBe(39.0);
    // exact um values
    expect(cuts.front.height_um).toBe(39000);
    expect(cuts.left.width_um).toBe(49000);
  });

  it('produces exactly 6 plates', () => {
    expect(cutList(spec())).toHaveLength(6);
  });

  it('rounds mm to 3 decimals, matching backend round(x, 3)', () => {
    const s = spec();
    s.glass.thickness_um = 525; // walls: 40000 - 2*525 = 38950 um
    const cuts = Object.fromEntries(cutList(s).map((c) => [c.face, c]));
    expect(cuts.front.height_mm).toBe(38.95);
    expect(cuts.left.width_mm).toBe(48.95); // 50000 - 2*525
  });
});

describe('copper tape length (sum of plate perimeters)', () => {
  it('defaults: 2*(200 + 178 + 176) mm = 110.8 cm', () => {
    // bottom/top: 2*(50+50) = 200 mm each; front/back: 2*(50+39) = 178 mm;
    // left/right: 2*(49+39) = 176 mm. Total 1108 mm = 110.8 cm.
    expect(copperTapeLengthCm(spec())).toBeCloseTo(110.8, 9);
  });

  it('tracks glass thickness (thicker glass -> shorter wall plates)', () => {
    const s = spec();
    s.glass.thickness_um = 1000; // walls lose 1 mm per side vs 0.5 mm
    // bottom/top unchanged (400 mm total). front/back: 2*(50+38) = 176;
    // left/right: 2*(48+38) = 172. Total = 400 + 352 + 344 = 1096 mm.
    expect(copperTapeLengthCm(s)).toBeCloseTo(109.6, 9);
  });

  it('is pure — does not mutate the spec', () => {
    const s = spec();
    const before = JSON.stringify(s);
    copperTapeLengthCm(s);
    expect(JSON.stringify(s)).toBe(before);
  });
});

describe('plate placements (Y up, origin at box center)', () => {
  it('outer surfaces are flush with the box envelope', () => {
    const p = Object.fromEntries(platePlacements(spec()).map((x) => [x.face, x]));
    expect(p.bottom.center_um).toEqual([0, -19750, 0]);
    expect(p.top.center_um).toEqual([0, 19750, 0]);
    expect(p.front.center_um).toEqual([0, 0, 24750]);
    expect(p.back.center_um).toEqual([0, 0, -24750]);
    expect(p.left.center_um).toEqual([-24750, 0, 0]);
    expect(p.right.center_um).toEqual([24750, 0, 0]);
  });

  it('outward normals point out of the box', () => {
    const p = Object.fromEntries(platePlacements(spec()).map((x) => [x.face, x]));
    expect(p.front.outward).toEqual([0, 0, 1]);
    expect(p.back.outward).toEqual([0, 0, -1]);
    expect(p.top.outward).toEqual([0, 1, 0]);
    expect(p.bottom.outward).toEqual([0, -1, 0]);
    expect(p.left.outward).toEqual([-1, 0, 0]);
    expect(p.right.outward).toEqual([1, 0, 0]);
  });
});

describe('solder seams', () => {
  it('8 beads: 4 bottom at y = -hh + t, 4 vertical corners spanning the walls', () => {
    const seams = seamSegments(spec());
    expect(seams).toHaveLength(8);

    const bottom = seams.filter((s) => s.id.startsWith('bottom'));
    expect(bottom).toHaveLength(4);
    for (const s of bottom) {
      expect(s.start_um[1]).toBe(-19500); // -20000 + 500
      expect(s.end_um[1]).toBe(-19500);
    }

    const corners = seams.filter((s) => s.id.startsWith('corner'));
    expect(corners).toHaveLength(4);
    for (const s of corners) {
      expect(s.axis).toBe('y');
      expect(s.start_um[1]).toBe(-19500);
      expect(s.end_um[1]).toBe(19500); // hh - t
      expect(Math.abs(s.start_um[0])).toBe(25000);
      expect(Math.abs(s.start_um[2])).toBe(25000);
    }
  });
});

describe('hinge layout (defaults: tube 2400, rod 1600, 5 segments, 0.8 coverage)', () => {
  it('contract segment math', () => {
    const h = hingeLayout(spec());
    expect(h.run_length_um).toBe(40000); // 0.8 * 50000
    expect(h.gap_um).toBe(SEGMENT_GAP_UM);
    expect(h.segment_length_um).toBe(7680); // (40000 - 4*400) / 5
    expect(h.rod_length_um).toBe(43200); // 40000 + 2*1600
    expect(h.tube_r_um).toBe(1200);
    expect(h.rod_r_um).toBe(800);
  });

  it('axis sits at lid mid-thickness, just outside the back face', () => {
    const h = hingeLayout(spec());
    expect(h.axis_y_um).toBe(19750); // hh - t/2
    expect(h.axis_z_um).toBe(-26200); // -hd - tube_od/2
  });

  it('segments alternate body,lid,... with both ends body, centered run', () => {
    const h = hingeLayout(spec());
    expect(h.segments.map((s) => s.owner)).toEqual(['body', 'lid', 'body', 'lid', 'body']);
    expect(h.segments[0].center_x_um).toBeCloseTo(-16160, 6);
    expect(h.segments[2].center_x_um).toBeCloseTo(0, 6);
    expect(h.segments[4].center_x_um).toBeCloseTo(16160, 6);
  });
});

describe('validation', () => {
  it('default ring box is valid', () => {
    expect(validateBox(spec())).toEqual([]);
  });

  it('the PRODUCTION default box is valid (bonded 2.25 mm quartz plies)', () => {
    // The box the app boots on and the live preview POSTs. `spec()` above is the
    // legacy single-plate geometry the formula pins are written against, so
    // without this the shipped default would go unvalidated by this suite.
    expect(validateBox(defaultBoxSpec())).toEqual([]);
  });

  it('flags a box too small to keep a patternable aperture', () => {
    const s = { ...spec(), width_um: 9000, depth_um: 9000, height_um: 9000 };
    const errors = validateBox(s);
    expect(errors.length).toBeGreaterThan(0);
    expect(errors[0]).toMatch(/aperture/);
  });

  it('flags even or too-few hinge segments', () => {
    const s = spec();
    s.hinge.segments = 4;
    expect(validateBox(s).some((e) => /odd/.test(e))).toBe(true);
  });

  it('rejects the exact aperture boundary like the backend (<=)', () => {
    // left/right width = depth - 2t = 9850 um -> aperture exactly 3000 um.
    const s = spec();
    s.width_um = 30000;
    s.height_um = 30000;
    s.depth_um = 10850;
    expect(validateBox(s).some((e) => /aperture/.test(e))).toBe(true);
  });

  it('mirrors backend hinge checks: coverage range, rod vs tube, uncuttable segments', () => {
    const c0 = spec();
    c0.hinge.coverage = 0;
    expect(validateBox(c0).some((e) => /coverage/.test(e))).toBe(true);
    const c2 = spec();
    c2.hinge.coverage = 1.5;
    expect(validateBox(c2).some((e) => /coverage/.test(e))).toBe(true);

    const rod = spec();
    rod.hinge.rod_od_um = rod.hinge.tube_od_um; // rod must be < tube
    expect(validateBox(rod).some((e) => /rod/i.test(e))).toBe(true);

    const seg = spec();
    seg.hinge.segments = 99; // ~8 um pieces, shorter than the tube OD
    expect(validateBox(seg).some((e) => /uncuttable/.test(e))).toBe(true);
  });

  it('mirrors backend degenerate-input guards: glass, dims, tape, safety', () => {
    // Backend validate_assembly rejects all four; the frontend must too
    // (a spec that passes here but 400s on POST /boxes/generate is a
    // contract bug).
    const glass = spec();
    glass.glass.thickness_um = 0;
    expect(validateBox(glass).some((e) => /thickness must be positive/i.test(e))).toBe(true);

    const dims = spec();
    dims.width_um = -1;
    expect(validateBox(dims).some((e) => /dimensions must be positive/i.test(e))).toBe(true);

    const tape = spec();
    tape.foil.tape_width_um = 0;
    expect(validateBox(tape).some((e) => /tape width must be positive/i.test(e))).toBe(true);

    const safety = spec();
    safety.foil.safety_um = -100;
    expect(validateBox(safety).some((e) => /safety margin cannot be negative/i.test(e))).toBe(
      true
    );
  });
});

describe('stampFaces (box-level normalization)', () => {
  it('stamps glass, cut dims and keep-out into every face', () => {
    const s = spec();
    s.faces.front = { ...s.faces.front!, width_um: 1, height_um: 2, weld_margin_um: 0 };
    const stamped = stampFaces(s);
    expect(stamped.faces.front!.width_um).toBe(50000);
    expect(stamped.faces.front!.height_um).toBe(39000);
    expect(stamped.faces.front!.weld_margin_um).toBe(3425);
    expect(stamped.faces.front!.glass).toEqual(s.glass);
    // every face agrees with the cut list
    for (const cut of cutList(s)) {
      const f = stamped.faces[cut.face]!;
      expect(f.width_um).toBe(cut.width_um);
      expect(f.height_um).toBe(cut.height_um);
      expect(f.weld_margin_um).toBe(3425);
    }
  });
});
