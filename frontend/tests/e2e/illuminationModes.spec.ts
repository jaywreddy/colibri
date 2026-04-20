/**
 * Illumination path: ambient/laser/backlight each map to a specific integer
 * in uIllumination. Laser color choice maps to a specific wavelength (μm)
 * on uLaserWavelengthUm so the iridescent_grating recipe can gate its
 * diffraction order against the active laser.
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

  test('laser color red/green/blue maps to uLaserWavelengthUm 0.65/0.55/0.45', async ({ page }) => {
    await setStore(page, { laserColor: 'red' });
    await page.waitForFunction(
      () => (window as any).__three.material.uniforms.uLaserWavelengthUm.value === 0.65
    );
    await setStore(page, { laserColor: 'green' });
    await page.waitForFunction(
      () => (window as any).__three.material.uniforms.uLaserWavelengthUm.value === 0.55
    );
    await setStore(page, { laserColor: 'blue' });
    await page.waitForFunction(
      () => (window as any).__three.material.uniforms.uLaserWavelengthUm.value === 0.45
    );
  });
});
