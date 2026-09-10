import { test, expect } from '@playwright/test';
import { waitForStudio, waitForBoxTextures } from './helpers';

test.describe('smoke', () => {
  test('app loads, studio boots, box manifest textures bind', async ({ page }) => {
    // Register the listener BEFORE goto — on a warm backend the /patterns
    // response can land before the post-navigation assertions run.
    const patternsResponse = page.waitForResponse(
      (r) => r.url().includes('/patterns') && r.status() === 200
    );
    await page.goto('/');
    await expect(page.getByText('Ring Box Studio')).toBeVisible();
    await patternsResponse;
    await waitForStudio(page);
    await waitForBoxTextures(page);
  });

  test('window.__studio exposes renderer/scene/camera/controls/faces/setLid', async ({
    page,
  }) => {
    await page.goto('/');
    await waitForStudio(page);
    const keys = await page.evaluate(() => Object.keys((window as any).__studio).sort());
    for (const k of ['camera', 'controls', 'faces', 'renderer', 'root', 'scene', 'setLid', 'store']) {
      expect(keys).toContain(k);
    }
    // All six face runtimes are present with a shader + glass material.
    const faceOk = await page.evaluate(() => {
      const s = (window as any).__studio;
      const ids = ['front', 'back', 'top', 'bottom', 'left', 'right'];
      return ids.every((fid) => !!s.faces[fid]?.shader && !!s.faces[fid]?.glassMat);
    });
    expect(faceOk).toBe(true);
  });

  test('build panel shows the live cut list and keep-out readout', async ({ page }) => {
    await page.goto('/');
    await waitForStudio(page);
    await expect(page.getByTestId('cut-list')).toBeVisible();
    // Production defaults: front outer ply 29.1 x 27.5 mm; the art rim is the
    // inner ply's window, 2.25 mm ply + 1.4 mm fold of 3/8" tape = 3.6 mm.
    await expect(page.getByTestId('cut-front')).toContainText('27.5');
    await expect(page.getByTestId('keepout-readout')).toContainText('3.6 mm');
  });
});
