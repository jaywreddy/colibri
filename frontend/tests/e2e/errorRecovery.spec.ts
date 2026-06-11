/**
 * Ensure the app doesn't wedge itself after a failed box regenerate. We
 * force a 500 on /boxes/generate, nudge the spec to trigger a regen, verify
 * the failure is logged + surfaced, then unroute and verify the next spec
 * change recovers — no page reload required.
 */
import { test, expect } from './fixtures';
import { clearLog, expectLogEvent, waitForStudio } from './helpers';

test.describe('error recovery', () => {
  test('500 on /boxes/generate surfaces in the log and the app recovers', async ({
    page,
  }) => {
    await page.goto('/');
    await waitForStudio(page);

    await page.route('**/boxes/generate', (route) =>
      route.fulfill({ status: 500, body: 'forced' })
    );
    await clearLog(page);
    // Nudge a mask-affecting field to trigger the debounced regen.
    await page.evaluate(() => {
      (window as any).__studio.store.getState().patchBoxSpec({ width_um: 51000 });
    });
    await expectLogEvent(page, 'box_regen_failed');
    await expect(page.getByTestId('regen-error')).toBeVisible();

    // The scene still updated instantly from assembly.ts despite the 500.
    const cutFront = await page.getByTestId('cut-front').textContent();
    expect(cutFront).toContain('51.0');

    await page.unroute('**/boxes/generate');
    await clearLog(page);
    await page.evaluate(() => {
      (window as any).__studio.store.getState().patchBoxSpec({ width_um: 52000 });
    });
    await expectLogEvent(page, 'box_regen_done', undefined, { timeout: 90_000 });
  });
});
