/**
 * Canvas-renders suite. Pins the two silent-black-canvas regressions:
 *   #3 canvas style drift (missing display:block; width:100%; height:100%)
 *   #4 NPOT texture with default LinearMipmapLinear filter
 *
 * We assert on pixel brightness inequalities, NOT screenshot diffs.
 */
import { test, expect } from '@playwright/test';
import { waitForThree, waitForTexturesBound, expectCanvasNotBlank, sampleBrightness } from './helpers';

test.describe('canvas renders', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
    await waitForThree(page);
    await waitForTexturesBound(page);
  });

  test('canvas has non-zero dimensions', async ({ page }) => {
    const dims = await page.evaluate(() => {
      const t = (window as any).__three;
      const c = t.renderer.domElement as HTMLCanvasElement;
      return { w: c.offsetWidth, h: c.offsetHeight };
    });
    expect(dims.w).toBeGreaterThan(100);
    expect(dims.h).toBeGreaterThan(100);
  });

  test('canvas is not entirely black', async ({ page }) => {
    await expectCanvasNotBlank(page);
  });

  test('center and corner pixels are distinguishable (texture binds, NPOT guard)', async ({
    page,
  }) => {
    const dims = await page.evaluate(() => {
      const t = (window as any).__three;
      const c = t.renderer.domElement as HTMLCanvasElement;
      return [c.width, c.height];
    });
    const [w, h] = dims;
    const bCenter = await sampleBrightness(page, Math.floor(w / 2), Math.floor(h / 2));
    const corners = [
      await sampleBrightness(page, 10, 10),
      await sampleBrightness(page, w - 10, 10),
      await sampleBrightness(page, 10, h - 10),
      await sampleBrightness(page, w - 10, h - 10),
    ];
    // At least one sample should differ meaningfully from another — else
    // we're looking at a flat-color canvas (the "silent black" or "silent
    // gray" failure mode).
    const all = [bCenter, ...corners];
    const spread = Math.max(...all) - Math.min(...all);
    expect(
      spread,
      `canvas appears flat, all samples=${JSON.stringify(all)}`
    ).toBeGreaterThan(5);
  });
});
