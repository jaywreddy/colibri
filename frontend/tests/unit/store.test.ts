import { describe, it, expect, beforeEach } from 'vitest';
import { useStore } from '../../src/store';
import type { PatternManifest } from '../../src/api';

function baseManifest(overrides: Partial<PatternManifest> = {}): PatternManifest {
  return {
    slug: 'wayuu-kanasu-moire',
    variant: 'abcdef0123',
    name: 'Wayuu kanasü moiré',
    description: '',
    tags: [],
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
      thumbnail: '/data/x/thumb.png',
    },
    ...overrides,
  };
}

beforeEach(() => {
  useStore.setState({
    catalog: [],
    activeSlug: null,
    manifest: null,
    params: {},
    illumination: 'ambient',
    laserColor: 'green',
  });
});

describe('store', () => {
  it('selectPattern updates slug + manifest + params atomically', () => {
    const m = baseManifest();
    useStore.getState().selectPattern('wayuu-kanasu-moire', m);
    const s = useStore.getState();
    expect(s.activeSlug).toBe('wayuu-kanasu-moire');
    expect(s.manifest).toEqual(m);
    expect(s.params).toEqual(m.params);
  });

  it('patchParams merges without dropping other params', () => {
    useStore.setState({ params: { a: 1, b: 2 } });
    useStore.getState().patchParams({ b: 3, c: 4 });
    expect(useStore.getState().params).toEqual({ a: 1, b: 3, c: 4 });
  });

  it('selectPatternIfCurrent commits only when pendingSlug matches', () => {
    const m = baseManifest({ slug: 'colibri-globe-moire' });
    useStore.getState().beginSelect('colibri-globe-moire');
    const ok = useStore.getState().selectPatternIfCurrent('colibri-globe-moire', m);
    expect(ok).toBe(true);
    expect(useStore.getState().activeSlug).toBe('colibri-globe-moire');

    // A stale response for a different slug must be dropped.
    useStore.getState().beginSelect('wayuu-kanasu-moire');
    const stale = useStore.getState().selectPatternIfCurrent(
      'colibri-globe-moire',
      m,
    );
    expect(stale).toBe(false);
  });
});
