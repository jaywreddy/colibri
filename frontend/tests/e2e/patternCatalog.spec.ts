/**
 * Parameterized: every pattern in the catalog selects, loads a manifest,
 * and renders a non-blank canvas. Uses the store directly (rather than
 * clicking a thumbnail) so the spec is robust to Gallery DOM changes.
 */
import { test, expect } from '@playwright/test';
import { waitForThree, waitForTexturesBound, expectCanvasNotBlank } from './helpers';

const SLUGS = [
  'wayuu-kanasu-moire',
  'emerald-facet-moire',
  'colibri-globe-lenticular',
  'colibri-globe-moire',
  'colibri-globe-phase',
];

for (const slug of SLUGS) {
  test(`pattern ${slug} selects and renders`, async ({ page }) => {
    await page.goto('/');
    await waitForThree(page);
    await waitForTexturesBound(page);

    // Drive selection via the store so the spec is robust to Gallery DOM
    // changes. The texture-load side-effect is observed on the material.
    await page.evaluate(async (s) => {
      const r = await fetch(`/patterns/${s}/default`);
      const manifest = await r.json();
      const mod = await import('/src/store.ts');
      mod.useStore.getState().selectPattern(s, manifest);
    }, slug);

    // Wait until the material actually points at the new pattern's front image.
    await page.waitForFunction(
      (s) => {
        const t = (window as any).__three;
        const img = t.material.uniforms.uFront.value?.image;
        return img && typeof img.currentSrc === 'string' && img.currentSrc.includes(`/data/${s}/`);
      },
      slug,
      { timeout: 30_000 }
    );

    const activeSlug = await page.evaluate(async () => {
      const mod = await import('/src/store.ts');
      return mod.useStore.getState().activeSlug;
    });
    expect(activeSlug).toBe(slug);
    await expectCanvasNotBlank(page);
  });
}
