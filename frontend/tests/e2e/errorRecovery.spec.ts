/**
 * Ensure the app doesn't wedge itself after a failed pattern load. We
 * force a 500 on the next `/patterns/{slug}/default` call, click a card,
 * verify the failure is logged (not swallowed), then unroute and verify
 * the next click recovers — no page reload required.
 */
import { test, expect } from './fixtures';
import {
  clearLog,
  expectLogEvent,
  readLog,
  waitForTexturesBound,
  waitForThree,
} from './helpers';

test.describe('error recovery', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
    await waitForThree(page);
    await waitForTexturesBound(page);
  });

  test('500 on /default surfaces in the log and the app recovers on next click', async ({
    page,
  }) => {
    // Pick two distinct slugs so the failure and the recovery target
    // different cards — this also proves the recovery isn't just
    // re-reading whatever was already bound.
    const slugs = await page.locator('button[data-slug]').evaluateAll((els) =>
      els.slice(0, 2).map((el) => el.getAttribute('data-slug')!)
    );
    expect(slugs.length).toBe(2);
    const [failSlug, recoverSlug] = slugs;

    // Route exactly one future default call to 500. Use a one-shot route
    // so the second click (on a different slug) hits the real server.
    const failRoute = `**/patterns/${failSlug}/default`;
    await page.route(failRoute, (route) => route.fulfill({ status: 500, body: 'forced' }));

    await clearLog(page);
    await page.locator(`button[data-slug="${failSlug}"]`).click();

    await expectLogEvent(
      page,
      'pattern_load_failed',
      (e) => e.slug === failSlug
    );
    // `fetch_error` should also fire (belt-and-suspenders on the wire layer).
    await expectLogEvent(page, 'fetch_error', (e) => e.status === 500);

    // Canvas may still show the previous pattern — that's fine. What
    // matters is that the app didn't crash and the next click works.
    await page.unroute(failRoute);
    await clearLog(page);
    await page.locator(`button[data-slug="${recoverSlug}"]`).click();
    await expectLogEvent(
      page,
      'pattern_selected',
      (e) => e.slug === recoverSlug && e.committed !== false,
      { timeout: 15_000 }
    );
    await expectLogEvent(page, 'texture_bound', (e) => e.slug === recoverSlug);

    // Sanity: no stray `pattern_load_failed` for the recovery slug.
    const log = await readLog(page);
    const recoverFailure = log.find(
      (e) => e.type === 'pattern_load_failed' && e.slug === recoverSlug
    );
    expect(recoverFailure, 'recovery slug must not log a failure').toBeFalsy();
  });
});
