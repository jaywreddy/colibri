/**
 * Pins regression #5: `currentEngine` in the store and `uUseFft` on the
 * material must stay in sync. Uses the store's setEngine directly rather
 * than chasing DOM selectors — this is a wiring test, not a UI test.
 */
import { test, expect } from '@playwright/test';
import { waitForThree, waitForTexturesBound, readUniform } from './helpers';

async function setEngine(page: import('@playwright/test').Page, engine: string) {
  await page.evaluate((e) => {
    (window as any).__setEngine?.(e) ??
      // Fall back to reaching into the zustand store global that leva/React uses.
      // PlateScene subscribes via useStore, so this triggers the useEffect.
      (function () {
        const mod = (window as any).__store;
        if (mod) mod.setState({ engine: e });
      })();
  }, engine);
}

test.describe('engine switching', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
    await waitForThree(page);
    await waitForTexturesBound(page);
  });

  test('stylized engine sets uUseFft=0', async ({ page }) => {
    // Expose store setter in page for deterministic engine flips.
    await page.evaluate(() => {
      // zustand stores expose setState via the hook return; PlateScene uses
      // `useStore.getState().setEngine(...)` — re-use via the store module.
      (window as any).__setEngine = async (e: string) => {
        const mod = await import('/src/store.ts');
        mod.useStore.setState({ engine: e as 'stylized' });
      };
    });
    await setEngine(page, 'stylized');
    await page.waitForFunction(
      () => (window as any).__three.material.uniforms.uUseFft.value === 0.0,
      null,
      { timeout: 10_000 }
    );
    const v = await readUniform<number>(page, 'uUseFft');
    expect(v).toBe(0.0);
  });

  test('fraunhofer engine fetches /sim/fft and sets uUseFft=1', async ({ page }) => {
    await page.evaluate(() => {
      (window as any).__setEngine = async (e: string) => {
        const mod = await import('/src/store.ts');
        mod.useStore.setState({ engine: e as 'fraunhofer' });
      };
    });
    const respP = page.waitForResponse((r) => r.url().includes('/sim/fft') && r.status() === 200, {
      timeout: 30_000,
    });
    await setEngine(page, 'fraunhofer');
    await respP;
    await page.waitForFunction(
      () => (window as any).__three.material.uniforms.uUseFft.value === 1.0,
      null,
      { timeout: 10_000 }
    );
  });

  test('waveprop engine fetches /sim/propagate and sets uUseFft=1', async ({ page }) => {
    await page.evaluate(() => {
      (window as any).__setEngine = async (e: string) => {
        const mod = await import('/src/store.ts');
        mod.useStore.setState({ engine: e as 'waveprop' });
      };
    });
    const respP = page.waitForResponse(
      (r) => r.url().includes('/sim/propagate') && r.status() === 200,
      { timeout: 60_000 }
    );
    await setEngine(page, 'waveprop');
    await respP;
    await page.waitForFunction(
      () => (window as any).__three.material.uniforms.uUseFft.value === 1.0,
      null,
      { timeout: 10_000 }
    );
  });
});
