/**
 * Guard the PatternManifest type shape at runtime so the backend can't
 * silently change a field name without a red test.
 */
import { describe, it, expect } from 'vitest';
import { RECIPE_IDS, type PatternManifest, type RenderRecipe } from '../../src/api';

const SAMPLE: PatternManifest = {
  slug: 'wayuu-kanasu-moire',
  variant: 'abcdef0123',
  name: 'Wayuu kanasü moiré',
  description: '',
  tags: ['moire'],
  params: { period_um: 4.0, duty: 0.5 },
  substrate: { thickness_um: 500, material: 'fused silica', n: 1.46 },
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
      render_recipe: 'moire_interactive' as RenderRecipe,
      recipe_data: { carrier_period_um: 20.0 },
    };
    expect(isManifest(withRecipe)).toBe(true);
  });
});

// ----------------------------------------------------------------------------
// RECIPE_IDS numeric mapping must match the uRecipe switch order in
// plate.frag. Pin the values so the shader and the TS never drift apart.
// ----------------------------------------------------------------------------
describe('RECIPE_IDS', () => {
  it('maps each recipe name to the plate.frag switch constant', () => {
    expect(RECIPE_IDS.stereo_lenticular).toBe(0);
    expect(RECIPE_IDS.moire_interactive).toBe(1);
    expect(RECIPE_IDS.phase_shift_overlay).toBe(2);
  });

  it('covers every RenderRecipe name and has no stale entries', () => {
    const names: RenderRecipe[] = [
      'stereo_lenticular',
      'moire_interactive',
      'phase_shift_overlay',
    ];
    for (const n of names) {
      expect(typeof RECIPE_IDS[n]).toBe('number');
    }
    expect('iridescent_grating' in RECIPE_IDS).toBe(false);
    expect('near_field_carpet' in RECIPE_IDS).toBe(false);
    expect('far_field_hologram' in RECIPE_IDS).toBe(false);
    expect('stylized_amplitude' in RECIPE_IDS).toBe(false);
    const ids = Object.values(RECIPE_IDS).sort();
    expect(ids).toEqual([0, 1, 2]);
  });
});
