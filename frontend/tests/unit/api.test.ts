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
});
