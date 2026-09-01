import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import * as api from '../../src/api';

const METRICS = {
  effect_strength: 0.42,
  mean_swing: 0.01,
  changed_frac: 0.77,
  peak_pair_deg: [-3, 3],
  tile_means: [1, 2],
  axis: 'y',
  axis_mode: 'auto',
};

function mockSheet(metrics: unknown = METRICS, ok = true) {
  return vi.fn().mockResolvedValue({
    ok,
    status: ok ? 200 : 500,
    headers: {
      get: (k: string) =>
        k === 'X-Collage-Metrics' && metrics !== null ? JSON.stringify(metrics) : null,
    },
    blob: async () => new Blob([new Uint8Array([1, 2, 3])], { type: 'image/png' }),
    text: async () => 'boom',
  });
}

beforeEach(() => {
  vi.stubGlobal('URL', { ...URL, createObjectURL: () => 'blob:stub', revokeObjectURL: () => {} });
});
afterEach(() => vi.unstubAllGlobals());

describe('fetchCollage', () => {
  it('asks for the auto axis by default', async () => {
    const spy = mockSheet();
    vi.stubGlobal('fetch', spy);
    await api.fetchCollage('wayuu-kanasu-moire');
    const url = spy.mock.calls[0][0] as string;
    expect(url).toContain('/sim/collage/wayuu-kanasu-moire/default');
    expect(url).toContain('axis=auto');
  });

  it('passes the sweep dials through', async () => {
    const spy = mockSheet();
    vi.stubGlobal('fetch', spy);
    await api.fetchCollage('x', { spanDeg: 12, steps: 13, illum: 'backlight', axis: 'y', cols: 13 });
    const url = spy.mock.calls[0][0] as string;
    expect(url).toContain('span_deg=12');
    expect(url).toContain('steps=13');
    expect(url).toContain('illum=backlight');
    expect(url).toContain('axis=y');
    expect(url).toContain('cols=13');
  });

  it('suppresses the in-image caption only when asked', async () => {
    const spy = mockSheet();
    vi.stubGlobal('fetch', spy);
    await api.fetchCollage('x', { showTitle: false });
    expect(spy.mock.calls[0][0] as string).toContain('show_title=false');
    await api.fetchCollage('x');
    expect(spy.mock.calls[1][0] as string).not.toContain('show_title');
  });

  it('reads the metrics off the header so the sweep is not composited twice', async () => {
    vi.stubGlobal('fetch', mockSheet());
    const out = await api.fetchCollage('x');
    expect(out.metrics.effect_strength).toBe(0.42);
    // The axis the backend actually chose must survive — a caller showing 'x'
    // when the effect was found on 'y' is worse than showing nothing.
    expect(out.metrics.axis).toBe('y');
    expect(out.metrics.peak_pair_deg).toEqual([-3, 3]);
    expect(out.url).toBe('blob:stub');
  });

  it('fails loudly when the metrics header is missing', async () => {
    vi.stubGlobal('fetch', mockSheet(null));
    await expect(api.fetchCollage('x')).rejects.toThrow(/no metrics/);
  });

  it('surfaces a non-2xx as an error rather than a blank sheet', async () => {
    vi.stubGlobal('fetch', mockSheet(METRICS, false));
    await expect(api.fetchCollage('x')).rejects.toThrow();
  });
});
