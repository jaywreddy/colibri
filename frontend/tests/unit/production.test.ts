import { describe, it, expect } from 'vitest';
import * as api from '../../src/api';
import * as P from '../../src/production';

/**
 * src/production.ts is GENERATED from backend/app/production.py
 * (tools/dev/gen_production_constants.py; backend/tests/test_production_constants.py
 * fails when it is stale). These tests pin the OTHER half of that contract:
 * that `defaultBoxSpec` actually reads the generated constants instead of
 * retyping the numbers, which is what the generator exists to prevent.
 *
 * They deliberately compare against `P.*` rather than against literals — a
 * literal here would just be a fourth copy of the number. The one place
 * literals belong is the handful of PINNED values below, which are pinned
 * precisely because the 2026-09-15 plate is already written.
 */
describe('production constants', () => {
  it('feeds every number of defaultBoxSpec', () => {
    const s = api.defaultBoxSpec();
    expect(s.width_um).toBe(P.WIDTH_UM);
    expect(s.depth_um).toBe(P.DEPTH_UM);
    expect(s.height_um).toBe(P.HEIGHT_UM);
    expect(s.glass).toEqual({
      thickness_um: P.PLY_UM,
      material: P.GLASS_MATERIAL,
      n: P.GLASS_N,
    });
    expect(s.foil.tape_width_um).toBe(P.TAPE_UM);
    expect(s.foil.safety_um).toBe(P.FOIL_SAFETY_UM);
    expect(s.foil.bead_um).toBe(P.FOIL_BEAD_UM);
    expect(s.art_rim_um).toBe(P.ART_RIM_UM);
    expect(s.carrier_pitch_um).toBe(P.CARRIER_UM);
  });

  it('feeds every face frame the same garland dials', () => {
    const s = api.defaultBoxSpec();
    const ids = Object.keys(s.faces);
    expect(ids).toHaveLength(6);
    for (const fid of ids) {
      const face = s.faces[fid as keyof typeof s.faces];
      expect(face!.frame.motif_scale).toBe(P.MOTIF_SCALE);
      expect(face!.frame.band_um).toBe(P.BAND_UM);
      // The art rim is stamped onto every face as its keep-out (both rims).
      expect(face!.weld_margin_um).toBe(P.ART_RIM_UM);
    }
  });

  it('holds the PINNED values the written plate depends on', () => {
    // These four are not derived and must not become derived: the mask exists.
    // Changing one here is a mask change, and the witness rebuild is the gate.
    expect(P.ART_RIM_UM).toBe(3637.5);
    expect(P.MOTIF_SCALE).toBe(0.68);
    expect(P.BAND_UM).toBe(2400);
    expect(P.CARRIER_UM).toBe(65.5);
    // Single ply, 1/4" tape: a 2.05 mm fold, clear of the art rim.
    expect((P.TAPE_UM - P.PLY_UM) / 2).toBeCloseTo(2050, 6);
    expect((P.TAPE_UM - P.PLY_UM) / 2).toBeLessThan(P.ART_RIM_UM);
  });
});
