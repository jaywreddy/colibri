import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { useStore } from '../../src/store';
import { FACE_IDS, defaultBoxSpec } from '../../src/api';

beforeEach(() => {
  useStore.setState({
    catalog: [],
    thumbnails: {},
    boxSpec: defaultBoxSpec(),
    boxManifest: null,
    selectedFaceId: 'front',
    lidTargetDeg: 0,
    layout: 'assembled',
    autoRotate: false,
    illumination: 'ambient',
    laserColor: 'green',
  });
});

afterEach(() => {
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
});

describe('store (Ring Box Studio v2)', () => {
  it('boots with the production default box spec', () => {
    const s = useStore.getState().boxSpec;
    expect(s.width_um).toBe(29100);
    expect(s.depth_um).toBe(29100);
    expect(s.height_um).toBe(32010);
    expect(s.bonded).toBe(true);
    expect(s.foil.tape_width_um).toBe(6350);
    expect(s.hinge.segments).toBe(5);
  });

  it('patchFoil merges and re-stamps keep-out into every face', () => {
    // Thin single-ply glass first. On the PRODUCTION box the bonded 2.25 mm
    // plies make the tape wrap a stepped edge that consumes 3p = 6.75 mm, so no
    // preset tape folds over at all and every keep-out collapses to the bare
    // safety margin — a correct number, but one that cannot show that the
    // re-stamp happened at all. Move to a geometry where the tape does fold.
    useStore.getState().patchBoxSpec({ bonded: false });
    useStore.getState().patchGlass({ thickness_um: 500 });
    useStore.getState().patchFoil({ tape_width_um: 4763 });
    const s = useStore.getState().boxSpec;
    expect(s.foil.tape_width_um).toBe(4763);
    expect(s.foil.safety_um).toBe(500); // untouched
    // keepout = (4763 - 500)/2 + 500 = 2631.5
    expect(s.faces.front!.weld_margin_um).toBe(2631.5);
    expect(s.faces.left!.weld_margin_um).toBe(2631.5);
  });

  it('patchGlass re-stamps cut dims into the faces', () => {
    useStore.getState().patchGlass({ thickness_um: 1000 });
    const s = useStore.getState().boxSpec;
    expect(s.faces.front!.height_um).toBe(30010); // H - 2t
    expect(s.faces.left!.width_um).toBe(27100); // D - 2t
    expect(s.faces.front!.glass.thickness_um).toBe(1000);
  });

  it('patchBoxSpec on dimensions re-stamps the faces', () => {
    useStore.getState().patchBoxSpec({ width_um: 60000 });
    const s = useStore.getState().boxSpec;
    expect(s.faces.front!.width_um).toBe(60000);
    expect(s.faces.top!.width_um).toBe(60000);
  });

  it('patchHinge merges without touching faces', () => {
    const before = useStore.getState().boxSpec.faces;
    useStore.getState().patchHinge({ segments: 7, coverage: 0.6 });
    const s = useStore.getState().boxSpec;
    expect(s.hinge.segments).toBe(7);
    expect(s.hinge.coverage).toBe(0.6);
    expect(s.hinge.tube_od_um).toBe(2400); // untouched
    expect(s.faces).toBe(before);
  });

  it('patchFace merges one face without dropping the others', () => {
    useStore.getState().patchFace('top', { pattern_slug: 'emerald-facet-moire' });
    const s = useStore.getState().boxSpec;
    expect(s.faces.top!.pattern_slug).toBe('emerald-facet-moire');
    expect(s.faces.front!.pattern_slug).toBe('globe-duo-phase');
  });

  it('patchFaceFrame merges frame fields', () => {
    useStore.getState().patchFaceFrame('front', { seed: 42, density: 1.5 });
    const f = useStore.getState().boxSpec.faces.front!.frame;
    expect(f.seed).toBe(42);
    expect(f.density).toBe(1.5);
    expect(f.bloom).toBe(0.6); // untouched
  });

  it('setLidTargetDeg clamps to [0, 120]', () => {
    useStore.getState().setLidTargetDeg(90);
    expect(useStore.getState().lidTargetDeg).toBe(90);
    useStore.getState().setLidTargetDeg(500);
    expect(useStore.getState().lidTargetDeg).toBe(120);
    useStore.getState().setLidTargetDeg(-20);
    expect(useStore.getState().lidTargetDeg).toBe(0);
  });

  it('setLayout switches assembled <-> flat', () => {
    useStore.getState().setLayout('flat');
    expect(useStore.getState().layout).toBe('flat');
    useStore.getState().setLayout('assembled');
    expect(useStore.getState().layout).toBe('assembled');
  });

  it('setAutoRotate toggles the turntable flag (default off)', () => {
    expect(useStore.getState().autoRotate).toBe(false);
    useStore.getState().setAutoRotate(true);
    expect(useStore.getState().autoRotate).toBe(true);
    useStore.getState().setAutoRotate(false);
    expect(useStore.getState().autoRotate).toBe(false);
  });
});

describe('applyFaceToAll', () => {
  it('copies slug + params + frame dials to all faces, preserving each seed', () => {
    const st = useStore.getState();
    st.patchFace('top', {
      pattern_slug: 'emerald-facet-moire',
      pattern_params: { period_um: 7.5 },
    });
    st.patchFaceFrame('top', { density: 1.4, bloom: 0.9, foliage: 0.2, band_um: 2200 });
    const seedsBefore = FACE_IDS.map(
      (fid) => useStore.getState().boxSpec.faces[fid]!.frame.seed
    );

    useStore.getState().applyFaceToAll('top');

    const s = useStore.getState().boxSpec;
    FACE_IDS.forEach((fid, i) => {
      const f = s.faces[fid]!;
      expect(f.pattern_slug).toBe('emerald-facet-moire');
      expect(f.pattern_params).toEqual({ period_um: 7.5 });
      expect(f.frame.density).toBe(1.4);
      expect(f.frame.bloom).toBe(0.9);
      expect(f.frame.foliage).toBe(0.2);
      expect(f.frame.band_um).toBe(2200);
      // Seeds stay individual — faces remain distinct variations.
      expect(f.frame.seed).toBe(seedsBefore[i]);
    });
  });

  it('copies params by value — later edits to the source do not leak', () => {
    const st = useStore.getState();
    st.patchFace('front', { pattern_params: { duty: 0.4 } });
    st.applyFaceToAll('front');
    st.patchFace('front', { pattern_params: { duty: 0.9 } });
    expect(useStore.getState().boxSpec.faces.back!.pattern_params).toEqual({ duty: 0.4 });
  });
});

describe('shuffleFaceSeed', () => {
  it('randomizes only the chosen face seed (31-bit non-negative int)', () => {
    vi.spyOn(Math, 'random').mockReturnValue(0.5);
    const beforeBack = useStore.getState().boxSpec.faces.back!.frame.seed;
    useStore.getState().shuffleFaceSeed('front');
    const s = useStore.getState().boxSpec;
    expect(s.faces.front!.frame.seed).toBe(Math.floor(0.5 * 0x7fffffff));
    expect(s.faces.back!.frame.seed).toBe(beforeBack); // untouched
  });

  it('leaves the other frame dials alone', () => {
    vi.spyOn(Math, 'random').mockReturnValue(0.123);
    const before = useStore.getState().boxSpec.faces.front!.frame;
    useStore.getState().shuffleFaceSeed('front');
    const after = useStore.getState().boxSpec.faces.front!.frame;
    expect(after.density).toBe(before.density);
    expect(after.bloom).toBe(before.bloom);
    expect(after.foliage).toBe(before.foliage);
    expect(after.band_um).toBe(before.band_um);
  });
});

describe('loadThumbnails', () => {
  const descriptor = (slug: string) => ({
    slug,
    name: slug,
    description: '',
    tags: [],
    tier: 1 as const,
    theme: 'Colombia' as const,
    params: [],
  });

  it('probes /patterns/{slug}/thumbnail once per slug and stores the probe URL', async () => {
    // The sweep is a read-only HEAD-style probe of the cached-thumbnail route
    // (never a /default materialize — that could start a multi-second
    // generation at boot); on 200 the probed URL itself is the <img> src.
    const fetchSpy = vi.fn().mockImplementation(async () => ({
      ok: true,
      status: 200,
      text: async () => '',
    }));
    vi.stubGlobal('fetch', fetchSpy);
    useStore.setState({ catalog: [descriptor('aaa'), descriptor('bbb')] });

    await useStore.getState().loadThumbnails();
    expect(fetchSpy).toHaveBeenCalledTimes(2);
    expect(useStore.getState().thumbnails).toEqual({
      aaa: '/patterns/aaa/thumbnail',
      bbb: '/patterns/bbb/thumbnail',
    });

    // Second call is a no-op — already-loaded slugs are skipped.
    await useStore.getState().loadThumbnails();
    expect(fetchSpy).toHaveBeenCalledTimes(2);
  });

  it('clears the loading marker on failure so a later open retries', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue({ ok: false, status: 500, json: async () => ({}), text: async () => '' })
    );
    useStore.setState({ catalog: [descriptor('broken')] });
    await useStore.getState().loadThumbnails();
    expect(useStore.getState().thumbnails).toEqual({});
  });
});
