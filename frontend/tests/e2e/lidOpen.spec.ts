/**
 * Lid hinge animation. The lid + its foil + its tube segments live under a
 * pivot group at the hinge axis (back top edge); opening rotates the pivot
 * about +X by -angle so the front edge swings UP and BACK over the hinge.
 */
import { test, expect } from './fixtures';
import { waitForStudio } from './helpers';

test.describe('lid', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
    await waitForStudio(page);
    await page.waitForFunction(() => !!(window as any).__studio.lidPivot);
  });

  test('setLid(100) drives the pivot to ~-100 deg about +X (up and back)', async ({
    page,
  }) => {
    const before = await page.evaluate(
      () => (window as any).__studio.lidPivot.rotation.x as number
    );
    expect(Math.abs(before)).toBeLessThan(0.01);

    await page.evaluate(() => (window as any).__studio.setLid(100));
    // Damped animation — wait until the pivot approaches -100° = -1.745 rad.
    await page.waitForFunction(
      () => (window as any).__studio.lidPivot.rotation.x < -1.6,
      null,
      { timeout: 10_000 }
    );
    const after = await page.evaluate(
      () => (window as any).__studio.lidPivot.rotation.x as number
    );
    // NEGATIVE rotation about +X = lid front edge swings up/back, never
    // through the box.
    expect(after).toBeLessThan(-1.6);
    expect(after).toBeGreaterThan(-1.8);

    await page.evaluate(() => (window as any).__studio.setLid(0));
    await page.waitForFunction(
      () => Math.abs((window as any).__studio.lidPivot.rotation.x) < 0.05,
      null,
      { timeout: 10_000 }
    );
  });

  test('lid slider sets the store target (clamped 0..120)', async ({ page }) => {
    await page.getByTestId('lid-slider').fill('90');
    await page.waitForFunction(
      () => (window as any).__studio.store.getState().lidTargetDeg === 90
    );
    // Pivot starts moving toward the target.
    await page.waitForFunction(
      () => (window as any).__studio.lidPivot.rotation.x < -0.5,
      null,
      { timeout: 10_000 }
    );
  });

  test('Open/Close toggle button round-trips', async ({ page }) => {
    await page.getByTestId('lid-toggle').click();
    await page.waitForFunction(
      () => (window as any).__studio.store.getState().lidTargetDeg > 0
    );
    await page.getByTestId('lid-toggle').click();
    await page.waitForFunction(
      () => (window as any).__studio.store.getState().lidTargetDeg === 0
    );
  });
});
