/**
 * Assembly-contract golden test — the frontend half of the shared fixture.
 *
 * tools/fixtures/assembly_golden.json is GENERATED from the backend math
 * (tools/dev/gen_assembly_golden.py); backend/tests/test_assembly.py pins
 * the backend to it, and this file pins src/assembly.ts to the identical
 * numbers. If either implementation drifts, exactly one suite fails and
 * points at the divergence — no more hand-synced magic literals.
 */
import { describe, it, expect } from 'vitest';
import { defaultBoxSpec, type BoxSpec } from '../../src/api';
import {
  cutList,
  hingeLayout,
  keepoutUm,
  overlapUm,
  seamSegments,
  validateBox,
} from '../../src/assembly';
import golden from '../../../tools/fixtures/assembly_golden.json';

type GoldenCase = {
  name: string;
  spec: {
    width_um: number;
    depth_um: number;
    height_um: number;
    glass_thickness_um: number;
    foil: BoxSpec['foil'];
    hinge: BoxSpec['hinge'];
  };
  expected: {
    valid: boolean;
    overlap_um?: number;
    keepout_um?: number;
    cut_list?: { face: string; width_um: number; height_um: number; width_mm: number; height_mm: number }[];
    seams?: Record<string, number>;
    hinge?: {
      run_length_um: number;
      segment_length_um: number;
      gap_um: number;
      rod_length_um: number;
      body_segments: number;
      lid_segments: number;
    };
  };
};

function toBoxSpec(g: GoldenCase['spec']): BoxSpec {
  const s = defaultBoxSpec();
  return {
    ...s,
    width_um: g.width_um,
    depth_um: g.depth_um,
    height_um: g.height_um,
    glass: { ...s.glass, thickness_um: g.glass_thickness_um },
    foil: { ...g.foil },
    hinge: { ...g.hinge },
  };
}

const segLenUm = (seg: { start_um: number[]; end_um: number[] }): number =>
  Math.hypot(
    seg.end_um[0] - seg.start_um[0],
    seg.end_um[1] - seg.start_um[1],
    seg.end_um[2] - seg.start_um[2]
  );

describe('assembly contract golden fixture (shared with backend)', () => {
  for (const c of (golden as { cases: GoldenCase[] }).cases) {
    it(`case '${c.name}' matches the backend numbers`, () => {
      const spec = toBoxSpec(c.spec);
      const errors = validateBox(spec);
      expect(errors.length === 0, `validity: backend=${c.expected.valid}, errors=${errors}`).toBe(
        c.expected.valid
      );
      if (!c.expected.valid) return;

      expect(overlapUm(spec)).toBeCloseTo(c.expected.overlap_um!, 6);
      expect(keepoutUm(spec)).toBeCloseTo(c.expected.keepout_um!, 6);

      const cuts = new Map(cutList(spec).map((x) => [x.face, x]));
      for (const e of c.expected.cut_list!) {
        const got = cuts.get(e.face as never)!;
        expect(got, `cut list missing face ${e.face}`).toBeDefined();
        expect(got.width_um).toBeCloseTo(e.width_um, 6);
        expect(got.height_um).toBeCloseTo(e.height_um, 6);
        expect(got.width_mm).toBe(e.width_mm);
        expect(got.height_mm).toBe(e.height_mm);
      }
      expect(cuts.size).toBe(c.expected.cut_list!.length);

      const seams = new Map(seamSegments(spec).map((s) => [s.id, segLenUm(s)]));
      for (const [id, len] of Object.entries(c.expected.seams!)) {
        expect(seams.get(id), `seam ${id} missing`).toBeDefined();
        expect(seams.get(id)!).toBeCloseTo(len, 6);
      }
      expect(seams.size).toBe(Object.keys(c.expected.seams!).length);

      const h = hingeLayout(spec);
      const eh = c.expected.hinge!;
      expect(h.run_length_um).toBeCloseTo(eh.run_length_um, 6);
      expect(h.segment_length_um).toBeCloseTo(eh.segment_length_um, 6);
      expect(h.gap_um).toBeCloseTo(eh.gap_um, 6);
      expect(h.rod_length_um).toBeCloseTo(eh.rod_length_um, 6);
      expect(h.segments.filter((s) => s.owner === 'body')).toHaveLength(eh.body_segments);
      expect(h.segments.filter((s) => s.owner === 'lid')).toHaveLength(eh.lid_segments);
    });
  }
});
