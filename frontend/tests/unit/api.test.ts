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
    expect(body.width_um).toBe(50000);
    expect(body.glass.n).toBe(1.46);
    expect(body.foil.tape_width_um).toBe(6350);
    expect(body.hinge.style).toBe('tube');
    expect(Object.keys(body.faces)).toHaveLength(6);
  });
});

describe('defaultBoxSpec (contract defaults)', () => {
  it('matches the v2 contract exactly', () => {
    const s = api.defaultBoxSpec();
    expect(s.width_um).toBe(50000);
    expect(s.depth_um).toBe(50000);
    expect(s.height_um).toBe(40000);
    expect(s.glass).toEqual({ thickness_um: 500, material: 'fused silica', n: 1.46 });
    expect(s.foil).toEqual({
      tape_width_um: 6350,
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

  it('puts wayuu-kanasu-moire with frame seeds 100+i on all six faces', () => {
    const s = api.defaultBoxSpec();
    api.FACE_IDS.forEach((fid, i) => {
      const f = s.faces[fid]!;
      expect(f.pattern_slug).toBe('wayuu-kanasu-moire');
      expect(f.frame.seed).toBe(100 + i);
      expect(f.glass).toEqual(s.glass);
      // stamped keep-out: (6350-500)/2 + 500
      expect(f.weld_margin_um).toBe(3425);
    });
  });
});
