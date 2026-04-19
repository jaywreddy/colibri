/**
 * Changing a parameter should:
 *   1. fire POST /patterns/generate
 *   2. return a new variant (or the same one if params identical)
 *   3. re-bind the texture so a sampled canvas pixel changes
 *
 * No setTimeout — waitForResponse only.
 */
import { test, expect } from '@playwright/test';
import { waitForThree, waitForTexturesBound, sampleMany } from './helpers';

test('changing a param triggers /patterns/generate and re-renders', async ({ page }) => {
  await page.goto('/');
  await waitForThree(page);
  await waitForTexturesBound(page);

  const sampleGrid: [number, number][] = [
    [120, 120],
    [200, 180],
    [260, 220],
    [320, 260],
    [380, 300],
    [180, 260],
    [260, 160],
    [340, 220],
  ];
  const before = await sampleMany(page, sampleGrid);
  const beforeVariant = await page.evaluate(async () => {
    const mod = await import('/src/store.ts');
    return mod.useStore.getState().manifest!.variant;
  });

  const respP = page.waitForResponse(
    (r) => r.url().includes('/patterns/generate') && r.status() === 200,
    { timeout: 30_000 }
  );

  // Mutate a param in the store and call generatePattern like the UI does.
  await page.evaluate(async () => {
    const api = await import('/src/api.ts');
    const store = await import('/src/store.ts');
    const slug = store.useStore.getState().activeSlug!;
    const current = store.useStore.getState().params;
    // Pick a parameter that exists on the default pattern (period_um) and nudge it.
    const nextParams = { ...current, period_um: (current.period_um as number) * 1.5 };
    store.useStore.setState({ params: nextParams });
    const m = await api.generatePattern(slug, nextParams);
    store.useStore.getState().selectPattern(slug, m);
  });

  const resp = await respP;
  const m = await resp.json();
  expect(m.variant).not.toBe(beforeVariant);

  // Wait for texture to re-bind to the new variant.
  await page.waitForFunction(
    (v) => {
      const img = (window as any).__three.material.uniforms.uFront.value?.image;
      return img?.complete && img?.currentSrc?.includes(v);
    },
    m.variant,
    { timeout: 30_000 }
  );
  // Give Three one RAF to upload the new texture to the GPU.
  await page.evaluate(
    () =>
      new Promise<void>((resolve) => requestAnimationFrame(() => requestAnimationFrame(() => resolve())))
  );

  const after = await sampleMany(page, sampleGrid);
  // Aggregate delta across the grid — any individual pixel might coincide
  // between two gratings, but the sum of |Δ| across 8 points is essentially
  // guaranteed to be nonzero when the pattern actually re-rendered.
  const totalDelta = before.reduce((s, b, i) => s + Math.abs(after[i] - b), 0);
  expect(totalDelta).toBeGreaterThan(0);
});
