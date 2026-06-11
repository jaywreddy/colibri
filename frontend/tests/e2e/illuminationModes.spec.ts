/**
 * Illumination path: ambient/laser/backlight each map to a specific integer
 * in uIllumination on every face's pattern ShaderMaterial.
 */
import { test, expect } from '@playwright/test';
import { waitForStudio, readFaceUniform } from './helpers';

async function setStore(page: import('@playwright/test').Page, patch: Record<string, unknown>) {
  await page.evaluate((p) => {
    (window as any).__studio.store.setState(p);
  }, patch);
}

test.describe('illumination', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
    await waitForStudio(page);
  });

  test('ambient -> uIllumination=0 on all faces', async ({ page }) => {
    await setStore(page, { illumination: 'ambient' });
    await page.waitForFunction(
      () => (window as any).__studio.faces.front.shader.uniforms.uIllumination.value === 0
    );
    expect(await readFaceUniform<number>(page, 'top', 'uIllumination')).toBe(0);
  });

  test('laser -> uIllumination=1', async ({ page }) => {
    await setStore(page, { illumination: 'laser' });
    await page.waitForFunction(
      () => (window as any).__studio.faces.front.shader.uniforms.uIllumination.value === 1
    );
    expect(await readFaceUniform<number>(page, 'left', 'uIllumination')).toBe(1);
  });

  test('backlight -> uIllumination=2', async ({ page }) => {
    await setStore(page, { illumination: 'backlight' });
    await page.waitForFunction(
      () => (window as any).__studio.faces.front.shader.uniforms.uIllumination.value === 2
    );
    expect(await readFaceUniform<number>(page, 'right', 'uIllumination')).toBe(2);
  });

  test('laser color updates uLaserColor uniform', async ({ page }) => {
    await setStore(page, { illumination: 'laser', laserColor: 'red' });
    await page.waitForFunction(() => {
      const c = (window as any).__studio.faces.front.shader.uniforms.uLaserColor.value;
      return c.r > 0.5 && c.g < 0.5;
    });
    await setStore(page, { laserColor: 'green' });
    await page.waitForFunction(() => {
      const c = (window as any).__studio.faces.front.shader.uniforms.uLaserColor.value;
      return c.g > 0.5;
    });
  });
});
