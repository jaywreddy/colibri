/**
 * Illumination path: ambient/laser/backlight each map to a specific integer
 * in uIllumination. Laser color choice maps to a specific wavelength slot.
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

  test('laser color red/green/blue maps to wavelength slot 0/1/2', async ({ page }) => {
    await setStore(page, { laserColor: 'red' });
    await page.waitForFunction(
      () => (window as any).__three.material.uniforms.uWavelengthSlot.value === 0
    );
    await setStore(page, { laserColor: 'green' });
    await page.waitForFunction(
      () => (window as any).__three.material.uniforms.uWavelengthSlot.value === 1
    );
    await setStore(page, { laserColor: 'blue' });
    await page.waitForFunction(
      () => (window as any).__three.material.uniforms.uWavelengthSlot.value === 2
    );
  });
});
