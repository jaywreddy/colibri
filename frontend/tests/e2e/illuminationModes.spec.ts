/**
 * Illumination path: ambient/laser/backlight each map to a specific integer
 * in uIllumination. The three render recipes all branch on uIllumination to
 * pick their tinting model.
 */
import { test, expect } from '@playwright/test';
import { waitForThree, waitForTexturesBound, readUniform } from './helpers';

async function setStore(page: import('@playwright/test').Page, patch: Record<string, unknown>) {
  await page.evaluate(async (p) => {
    const mod = await import('/src/store.ts');
    mod.useStore.setState(p as any);
  }, patch);
}

test.describe('illumination', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
    await waitForThree(page);
    await waitForTexturesBound(page);
  });

  test('ambient -> uIllumination=0', async ({ page }) => {
    await setStore(page, { illumination: 'ambient' });
    await page.waitForFunction(
      () => (window as any).__three.material.uniforms.uIllumination.value === 0
    );
    expect(await readUniform<number>(page, 'uIllumination')).toBe(0);
  });

  test('laser -> uIllumination=1', async ({ page }) => {
    await setStore(page, { illumination: 'laser' });
    await page.waitForFunction(
      () => (window as any).__three.material.uniforms.uIllumination.value === 1
    );
    expect(await readUniform<number>(page, 'uIllumination')).toBe(1);
  });

  test('backlight -> uIllumination=2', async ({ page }) => {
    await setStore(page, { illumination: 'backlight' });
    await page.waitForFunction(
      () => (window as any).__three.material.uniforms.uIllumination.value === 2
    );
    expect(await readUniform<number>(page, 'uIllumination')).toBe(2);
  });

  test('laser color updates uLaserColor uniform', async ({ page }) => {
    await setStore(page, { illumination: 'laser', laserColor: 'red' });
    await page.waitForFunction(() => {
      const c = (window as any).__three.material.uniforms.uLaserColor.value;
      return c.r > 0.5 && c.g < 0.5;
    });
    await setStore(page, { laserColor: 'green' });
    await page.waitForFunction(() => {
      const c = (window as any).__three.material.uniforms.uLaserColor.value;
      return c.g > 0.5;
    });
  });
});
