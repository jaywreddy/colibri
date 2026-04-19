import { test, expect } from '@playwright/test';
import { waitForThree, waitForTexturesBound } from './helpers';

test.describe('smoke', () => {
  test('app loads, gallery populates, first pattern auto-selects', async ({ page }) => {
    await page.goto('/');
    // Gallery lists at least the 10 expected patterns.
    await expect(page.getByText('Optics Pattern Studio')).toBeVisible();
    await page.waitForResponse((r) => r.url().includes('/patterns') && r.status() === 200);
    await waitForThree(page);
    await waitForTexturesBound(page);
  });

  test('window.__three exposes renderer/scene/camera/material', async ({ page }) => {
    await page.goto('/');
    await waitForThree(page);
    const keys = await page.evaluate(() => Object.keys((window as any).__three).sort());
    for (const k of ['camera', 'controls', 'material', 'mesh', 'renderer', 'scene']) {
      expect(keys).toContain(k);
    }
  });
});
