/**
 * Guard the PatternManifest + BoxManifest (v2) type shapes at runtime so the
 * backend can't silently change a field name without a red test.
 */
import { describe, it, expect } from 'vitest';
import {
  RECIPE_IDS,
  defaultBoxSpec,
  type BoxManifest,
  type PatternManifest,
  type RenderRecipe,
} from '../../src/api';

// A pattern the box actually carries. The fixture used to name a retired
// catalogue slug, which made a shape guard read like a catalogue claim.
const SAMPLE: PatternManifest = {
  slug: 'globe-atlantic',
  variant: 'abcdef0123',
  name: 'Atlantic globe',
  description: '',
  tags: ['region_art'],
  params: { period_um: 4.0, duty: 0.5 },
  substrate: { thickness_um: 2250, material: 'fused quartz', n: 1.4585 },
  extent_um: [2000, 2000],
  pixel_pitch_um: 0.5,
  min_feature_um: 2.0,
  extra: {},
  files: {
    front_png: '/data/x/front.png',
    back_png: '/data/x/back.png',
    front_svg: '/data/x/front.svg',
    back_svg: '/data/x/back.svg',
    thumbnail: '/data/x/thumbnail.png',
  },
};

const REQUIRED_FILE_KEYS = [
  'front_png',
  'back_png',
  'front_svg',
  'back_svg',
  'thumbnail',
] as const;

function isManifest(m: unknown): m is PatternManifest {
  if (typeof m !== 'object' || m === null) return false;
  const obj = m as Record<string, unknown>;
  if (typeof obj.slug !== 'string') return false;
  if (typeof obj.variant !== 'string') return false;
  if (typeof obj.pixel_pitch_um !== 'number' || obj.pixel_pitch_um <= 0) return false;
  if (!Array.isArray(obj.extent_um) || obj.extent_um.length !== 2) return false;
  if (typeof obj.substrate !== 'object' || obj.substrate === null) return false;
  const sub = obj.substrate as Record<string, unknown>;
  if (typeof sub.thickness_um !== 'number' || sub.thickness_um <= 0) return false;
  if (typeof sub.n !== 'number' || sub.n <= 0) return false;
  const files = obj.files as Record<string, unknown> | undefined;
  if (!files) return false;
  for (const k of REQUIRED_FILE_KEYS) {
    if (typeof files[k] !== 'string') return false;
  }
  return true;
}

describe('PatternManifest shape', () => {
  it('a valid manifest passes the guard', () => {
    expect(isManifest(SAMPLE)).toBe(true);
  });

  it('missing pixel_pitch_um fails the guard', () => {
    const bad = { ...SAMPLE, pixel_pitch_um: undefined } as unknown;
    expect(isManifest(bad)).toBe(false);
  });

  it('missing a files entry fails the guard', () => {
    const bad = {
      ...SAMPLE,
      files: { ...SAMPLE.files, front_png: undefined },
    } as unknown;
    expect(isManifest(bad)).toBe(false);
  });

  it('substrate with zero thickness fails the guard', () => {
    const bad = {
      ...SAMPLE,
      substrate: { ...SAMPLE.substrate, thickness_um: 0 },
    } as unknown;
    expect(isManifest(bad)).toBe(false);
  });

  it('accepts a manifest with render_recipe + recipe_data', () => {
    const withRecipe: unknown = {
      ...SAMPLE,
      render_recipe: 'foliage_moire' as RenderRecipe,
      recipe_data: { carrier_period_um: 20.0 },
    };
    expect(isManifest(withRecipe)).toBe(true);
  });
});

// ----------------------------------------------------------------------------
// BoxManifest v2 — the contract's server response shape. The frontend's
// scene/cut-list code reads spec + assembly; pin the required keys here.
// ----------------------------------------------------------------------------

const SAMPLE_BOX: BoxManifest = {
  kind: 'box',
  id: 'ring-box-1',
  spec: defaultBoxSpec(),
  name: 'Ring box',
  faces: {},
  dimensions_um: { width: 50000, height: 40000, depth: 50000 },
  assembly: {
    keepout_um: 3425,
    overlap_um: 2925,
    // Back-carrier window = overlap only (drops the safety margin).
    back_window_um: 2925,
    glass_thickness_um: 500,
    cut_list: [
      { face: 'bottom', width_um: 50000, height_um: 50000, width_mm: 50.0, height_mm: 50.0 },
      { face: 'top', width_um: 50000, height_um: 50000, width_mm: 50.0, height_mm: 50.0 },
      { face: 'front', width_um: 50000, height_um: 39000, width_mm: 50.0, height_mm: 39.0 },
      { face: 'back', width_um: 50000, height_um: 39000, width_mm: 50.0, height_mm: 39.0 },
      { face: 'left', width_um: 49000, height_um: 39000, width_mm: 49.0, height_mm: 39.0 },
      { face: 'right', width_um: 49000, height_um: 39000, width_mm: 49.0, height_mm: 39.0 },
    ],
    seams: 8,
    hinge: {
      style: 'tube',
      tube_od_um: 2400,
      rod_od_um: 1600,
      segments: 5,
      coverage: 0.8,
      run_length_um: 40000,
      segment_length_um: 7680,
    },
  },
  content_hash: 'abc123',
};

function isBoxManifest(m: unknown): m is BoxManifest {
  if (typeof m !== 'object' || m === null) return false;
  const obj = m as Record<string, unknown>;
  if (obj.kind !== 'box') return false;
  if (typeof obj.id !== 'string') return false;
  if (typeof obj.content_hash !== 'string') return false;
  const spec = obj.spec as Record<string, unknown> | undefined;
  if (!spec) return false;
  for (const k of ['width_um', 'depth_um', 'height_um']) {
    if (typeof spec[k] !== 'number' || (spec[k] as number) <= 0) return false;
  }
  for (const k of ['glass', 'foil', 'hinge', 'faces']) {
    if (typeof spec[k] !== 'object' || spec[k] === null) return false;
  }
  const dims = obj.dimensions_um as Record<string, unknown> | undefined;
  if (!dims) return false;
  for (const k of ['width', 'height', 'depth']) {
    if (typeof dims[k] !== 'number') return false;
  }
  const asm = obj.assembly as Record<string, unknown> | undefined;
  if (!asm) return false;
  for (const k of ['keepout_um', 'overlap_um', 'glass_thickness_um']) {
    if (typeof asm[k] !== 'number') return false;
  }
  if (!Array.isArray(asm.cut_list) || asm.cut_list.length !== 6) return false;
  const hinge = asm.hinge as Record<string, unknown> | undefined;
  if (!hinge) return false;
  if (typeof hinge.run_length_um !== 'number') return false;
  if (typeof hinge.segment_length_um !== 'number') return false;
  return true;
}

describe('BoxManifest v2 shape', () => {
  it('a valid box manifest passes the guard', () => {
    expect(isBoxManifest(SAMPLE_BOX)).toBe(true);
  });

  it('missing the assembly block fails the guard', () => {
    const bad = { ...SAMPLE_BOX, assembly: undefined } as unknown;
    expect(isBoxManifest(bad)).toBe(false);
  });

  it('a cut list without all 6 plates fails the guard', () => {
    const bad = {
      ...SAMPLE_BOX,
      assembly: { ...SAMPLE_BOX.assembly, cut_list: SAMPLE_BOX.assembly.cut_list.slice(0, 4) },
    } as unknown;
    expect(isBoxManifest(bad)).toBe(false);
  });

  it('hinge echo must include run/segment lengths', () => {
    const { run_length_um: _drop, ...hingeNoRun } = SAMPLE_BOX.assembly.hinge;
    const bad = {
      ...SAMPLE_BOX,
      assembly: { ...SAMPLE_BOX.assembly, hinge: hingeNoRun },
    } as unknown;
    expect(isBoxManifest(bad)).toBe(false);
  });

  it('v1 manifests (no glass/foil/hinge in spec) fail the guard', () => {
    const v1spec = {
      width_um: 30000,
      height_um: 30000,
      depth_um: 30000,
      weld_margin_um: 1000,
      faces: {},
      label: '',
    };
    const bad = { ...SAMPLE_BOX, spec: v1spec } as unknown;
    expect(isBoxManifest(bad)).toBe(false);
  });
});

// ----------------------------------------------------------------------------
// RECIPE_IDS — the manifest's recipe vocabulary. plate.frag no longer switches
// on it (one recipe is implemented), but the backend's RECIPE_NAMES is a
// POSITIONAL list, so the id must stay 3 and the retired names must stay out.
// ----------------------------------------------------------------------------
describe('RECIPE_IDS', () => {
  it('maps foliage_moire to the backend id 3', () => {
    expect(RECIPE_IDS.foliage_moire).toBe(3);
    expect(Object.keys(RECIPE_IDS)).toEqual(['foliage_moire']);
  });

  it('never resurrects a retired recipe name', () => {
    // 0 stereo_lenticular and 1 moire_interactive previewed standalone
    // patterns on a single plane; 2 phase_shift_overlay is the banned
    // two-image phase split. Their ids are burnt, not free.
    for (const retired of [
      'stereo_lenticular',
      'moire_interactive',
      'phase_shift_overlay',
      'iridescent_grating',
      'near_field_carpet',
      'far_field_hologram',
      'stylized_amplitude',
    ]) {
      expect(retired in RECIPE_IDS).toBe(false);
    }
    expect(Object.values(RECIPE_IDS)).toEqual([3]);
  });
});
