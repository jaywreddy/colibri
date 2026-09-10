import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import * as api from '../../src/api';

function mockFetchOk(body: unknown) {
  return vi.fn().mockResolvedValue({
    ok: true,
    status: 200,
    json: async () => body,
    text: async () => JSON.stringify(body),
  });
}

beforeEach(() => {
  vi.stubGlobal('fetch', mockFetchOk({}));
});
afterEach(() => {
  vi.unstubAllGlobals();
});

describe('api', () => {
  it('listPatterns GETs /patterns', async () => {
    const spy = mockFetchOk([{ slug: 'wayuu-kanasu-moire' }]);
    vi.stubGlobal('fetch', spy);
    const out = await api.listPatterns();
    expect(spy).toHaveBeenCalledWith('/patterns');
    expect(out).toEqual([{ slug: 'wayuu-kanasu-moire' }]);
  });

  it('generatePattern POSTs slug+params as JSON', async () => {
    const spy = mockFetchOk({ slug: 'x', variant: 'v' });
    vi.stubGlobal('fetch', spy);
    await api.generatePattern('wayuu-kanasu-moire', { period_um: 4.0 });
    const [url, opts] = spy.mock.calls[0] as [string, RequestInit];
    expect(url).toBe('/patterns/generate');
    expect(opts.method).toBe('POST');
    expect((opts.headers as Record<string, string>)['Content-Type']).toBe('application/json');
    expect(JSON.parse(opts.body as string)).toEqual({
      slug: 'wayuu-kanasu-moire',
      params: { period_um: 4.0 },
    });
  });

  it('throws on non-ok response', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue({
        ok: false,
        status: 500,
        json: async () => ({}),
        text: async () => 'boom',
      })
    );
    await expect(api.listPatterns()).rejects.toThrow(/500/);
  });

  it('generateBox POSTs the v2 spec envelope with box_id and force', async () => {
    const spy = mockFetchOk({ kind: 'box', id: 'x' });
    vi.stubGlobal('fetch', spy);
    await api.generateBox(api.defaultBoxSpec(), { boxId: 'my-box', force: true });
    const [url, opts] = spy.mock.calls[0] as [string, RequestInit];
    expect(url).toBe('/boxes/generate');
    expect(opts.method).toBe('POST');
    const body = JSON.parse(opts.body as string);
    expect(body.box_id).toBe('my-box');
    expect(body.force).toBe(true);
    expect(body.width_um).toBe(29100);
    expect(body.glass.n).toBe(1.4585);
    expect(body.foil.tape_width_um).toBe(9525);
    expect(body.hinge.style).toBe('tube');
    expect(Object.keys(body.faces)).toHaveLength(6);
  });
});

describe('defaultBoxSpec (contract defaults)', () => {
  it('matches the PRODUCTION box exactly (= backend default_box_spec)', () => {
    const s = api.defaultBoxSpec();
    expect(s.width_um).toBe(29100);
    expect(s.depth_um).toBe(29100);
    expect(s.height_um).toBe(32010);
    expect(s.bonded).toBe(true);
    expect(s.glass).toEqual({ thickness_um: 2250, material: 'fused quartz', n: 1.4585 });
    expect(s.foil).toEqual({
      tape_width_um: 9525,
      safety_um: 500,
      bead_um: 2000,
      finish: 'bright',
    });
    expect(s.hinge).toEqual({
      style: 'tube',
      tube_od_um: 2400,
      rod_od_um: 1600,
      segments: 5,
      coverage: 0.8,
    });
    expect(s.label).toBe('');
  });

  it('applies the production six-face plan with per-face frame seeds', () => {
    // Production plan: top = J+P monogram, front = colibri<->globe duo switch,
    // left/right = the two halftone photos on SINGLE-PLY walls, back + bottom =
    // bare glass. Per-face frame profiles stay seeded 100..105 in profile order,
    // now at the production motif scale.
    const s = api.defaultBoxSpec();
    const expectedSeed: Record<string, number> = {
      front: 100, back: 101, top: 102, bottom: 103, left: 104, right: 105,
    };
    expect(s.faces.front!.pattern_slug).toBe('globe-duo-phase');
    expect(s.faces.top!.pattern_slug).toBe('monogram-jp');
    expect(s.faces.back!.pattern_slug).toBe('blank');
    expect(s.faces.bottom!.pattern_slug).toBe('blank');
    expect(s.faces.left!.pattern_slug).toBe('photo-halftone');
    expect(s.faces.right!.pattern_slug).toBe('photo-halftone');
    expect(s.faces.left!.pattern_params).toEqual({ image: 'beach', colour_mode: 'faces' });
    expect(s.faces.right!.pattern_params).toEqual({ image: 'sunset', colour_mode: 'plain' });
    api.FACE_IDS.forEach((fid) => {
      const f = s.faces[fid]!;
      expect(f.frame.seed).toBe(expectedSeed[fid]);
      expect(f.frame.motif_scale).toBe(0.68);
      expect(f.frame.band_um).toBe(2400);
      expect(f.glass).toEqual(s.glass);
      // Only the two photo walls leave their inner ply bare.
      expect(f.single_ply).toBe(fid === 'left' || fid === 'right');
      // Stamped front-art rim (bondedArtKeepoutUm). The bonded 2.25 mm plies
      // make the 3/8" (9525 um) tape wrap a stepped edge that consumes 3p =
      // 6750 um, leaving a 1387.5 um fold; the foil rim is 1887.5 but the art
      // starts at the inner ply's window, 2250 + 1387.5 = 3637.5 from the edge,
      // so every front feature has the back carrier behind it.
      expect(f.weld_margin_um).toBe(3637.5);
      expect(f.back_margin_um).toBe(3637.5);
    });
  });
});
