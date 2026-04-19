/**
 * Pins regression #7 adjacent: the backend's variant hash must be stable
 * across two identical POSTs. If we ever changed the hash algorithm or
 * introduced non-deterministic param normalization, this goes red.
 */
import { test, expect } from '@playwright/test';

test('same params -> same variant hash', async ({ request }) => {
  const params = { period_um: 20.0, duty: 0.5 };
  const r1 = await request.post('http://127.0.0.1:8765/patterns/generate', {
    data: { slug: 'wayuu-kanasu-moire', params },
  });
  const r2 = await request.post('http://127.0.0.1:8765/patterns/generate', {
    data: { slug: 'wayuu-kanasu-moire', params },
  });
  expect(r1.ok()).toBe(true);
  expect(r2.ok()).toBe(true);
  const m1 = await r1.json();
  const m2 = await r2.json();
  expect(m1.variant).toBe(m2.variant);
});

test('param key ordering does not affect variant hash', async ({ request }) => {
  const r1 = await request.post('http://127.0.0.1:8765/patterns/generate', {
    data: { slug: 'wayuu-kanasu-moire', params: { period_um: 20.0, duty: 0.5 } },
  });
  const r2 = await request.post('http://127.0.0.1:8765/patterns/generate', {
    data: { slug: 'wayuu-kanasu-moire', params: { duty: 0.5, period_um: 20.0 } },
  });
  const m1 = await r1.json();
  const m2 = await r2.json();
  expect(m1.variant).toBe(m2.variant);
});
