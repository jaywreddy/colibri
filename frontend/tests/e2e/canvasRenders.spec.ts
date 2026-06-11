/**
 * Canvas-renders suite. Pins the silent-black-canvas regressions:
 *   - canvas style drift (missing display:block; width:100%; height:100%)
 *   - NPOT texture with default LinearMipmapLinear filter
 *
 * We assert on pixel brightness inequalities, NOT screenshot diffs.
 */
import { test, expect } from '@playwright/test';
import {
  waitForStudio,
  waitForBoxTextures,
  expectCanvasNotBlank,
  sampleBrightness,
} from './helpers';

test.describe('canvas renders', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
    await waitForStudio(page);
    await waitForBoxTextures(page);
  });

  test('canvas has non-zero dimensions', async ({ page }) => {
    const dims = await page.evaluate(() => {
      const s = (window as any).__studio;
      const c = s.renderer.domElement as HTMLCanvasElement;
      return { w: c.offsetWidth, h: c.offsetHeight };
    });
    expect(dims.w).toBeGreaterThan(100);
    expect(dims.h).toBeGreaterThan(100);
  });

  test('canvas is not entirely black', async ({ page }) => {
    await expectCanvasNotBlank(page);
  });

  test('center and corner pixels are distinguishable (box visible, not flat)', async ({
    page,
  }) => {
    const dims = await page.evaluate(() => {
      const s = (window as any).__studio;
      const c = s.renderer.domElement as HTMLCanvasElement;
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
    const all = [bCenter, ...corners];
    const spread = Math.max(...all) - Math.min(...all);
    expect(
      spread,
      `canvas appears flat, all samples=${JSON.stringify(all)}`
    ).toBeGreaterThan(5);
  });

  test('flat layout re-renders without blanking the canvas', async ({ page }) => {
    await page.getByTestId('layout-flat').click();
    await page.waitForTimeout(600);
    await expectCanvasNotBlank(page);
    await page.getByTestId('layout-assembled').click();
    await page.waitForTimeout(600);
    await expectCanvasNotBlank(page);
  });
});
