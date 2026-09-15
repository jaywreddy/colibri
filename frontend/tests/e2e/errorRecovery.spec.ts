/**
 * The two ways a regenerate can fail to produce a manifest, and what the UI
 * owes the user in each.
 *
 * 1. The BACKEND rejects it (forced 500). regen() ran, so the failure is a
 *    network/server one: it must reach window.__log and the header banner, the
 *    app must keep self-healing on its retry heartbeat, and the next spec
 *    change must recover with no page reload.
 * 2. The SPEC is invalid (validateBox fails). regen() never POSTs at all
 *    (App.tsx logs box_regen_skipped_invalid and returns), so this is a
 *    different code path from 1 — the one that used to be untested. The
 *    always-visible validation strip must name the offending geometry and no
 *    request may reach /boxes/generate: the held manifest still describes the
 *    last VALID design, so a silent regen of the invalid one would put a box
 *    on screen that the cut list below it does not describe.
 *
 * (The third assertion these tests used to carry — that the "Export fab
 * bundle" button refuses while the spec is invalid — went with the button:
 * the fab archive is a CLI step now, run against a named box id.)
 *
 * Test budgets: every inner wait must fit INSIDE the test timeout, or
 * Playwright kills the test before expectLogEvent can dump the log buffer and
 * the failure becomes unreadable (the recovery test used to allow 90 s inside
 * the config's 60 s default). Each test therefore raises its own timeout above
 * the sum of its waits.
 */
import { test, expect } from './fixtures';
import { clearLog, expectLogEvent, waitForStudio } from './helpers';

/** Longest single regen wait used here (cold six-plate compose on this host). */
const REGEN_WAIT_MS = 120_000;

test.describe('error recovery', () => {
  test('500 on /boxes/generate surfaces in the log and the app recovers', async ({
    page,
  }) => {
    // Boot + failure round trip + the post-unroute recovery regen, which is
    // the one that can legitimately take minutes on a cold data/ cache.
    test.setTimeout(REGEN_WAIT_MS + 90_000);
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
    await expectLogEvent(page, 'box_regen_done', undefined, { timeout: REGEN_WAIT_MS });
  });

  test('an invalid box width pauses regen and names the fault', async ({
    page,
  }) => {
    // Boot regen + the restore regen at the end, both bounded by REGEN_WAIT_MS.
    test.setTimeout(2 * REGEN_WAIT_MS + 60_000);
    await page.goto('/');
    await waitForStudio(page);
    // A valid boot spec: no strip.
    await expectLogEvent(page, 'box_regen_done', undefined, { timeout: REGEN_WAIT_MS });
    await expect(page.getByTestId('validation-errors')).toHaveCount(0);

    // Count everything that reaches the generate endpoint from here on. The
    // validation gate's whole job is that this stays at zero while invalid.
    let generatePosts = 0;
    await page.route('**/boxes/generate', async (route) => {
      generatePosts += 1;
      await route.continue();
    });
    await clearLog(page);

    // Drive the REAL width slider (not the store) to its 10 mm minimum. That
    // is what makes this a UI test: SliderRow's handler is the mm -> um
    // boundary (`Math.round(mm * 1000)`), the only place in the app where a
    // display unit becomes a spec unit, and nothing exercised it before.
    //
    // Why 10 mm is invalid, from src/assembly.ts with the defaults:
    //   hinge run L = coverage * W = 0.8 * 10000 = 8000 um
    //   segment     = (L - (n-1) * SEGMENT_GAP) / n = (8000 - 4*400) / 5 = 1280 um
    //   1280 um <= tube OD 2400 um  -> uncuttable segments
    // (The apertures survive: the narrowest plate side is 10000 um against a
    // 2 * 3425 um keep-out, leaving 3150 um, just over MIN_APERTURE_UM.)
    const width = page.getByTestId('box-width');
    await width.fill('10');
    expect(
      await page.evaluate(
        () => (window as any).__studio.store.getState().boxSpec.width_um
      ),
      'the mm slider must land as micrometers in the spec'
    ).toBe(10000);
    await expect(page.getByTestId('cut-bottom')).toContainText('10.0');

    await expectLogEvent(page, 'box_regen_skipped_invalid', undefined, { timeout: 30_000 });
    const strip = page.getByTestId('validation-errors');
    await expect(strip).toBeVisible();
    await expect(strip).toContainText('INVALID SPEC');
    // The actual geometry, not a generic "invalid" — 1.28 mm segments under a
    // 2.40 mm tube. If this message ever stops rendering, the user gets a
    // silent no-regen with the previous design still exportable.
    await expect(strip).toContainText('Hinge tube segments come out 1.28 mm');
    await expect(strip).toContainText('2.40 mm');
    // Header echo of errors[0].
    await expect(page.getByTestId('regen-error')).toBeVisible();
    await expect(page.getByTestId('regen-error')).toContainText('spec error');

    // The gate is client-side: nothing was sent for the backend to reject.
    expect(
      generatePosts,
      'an invalid spec must never reach POST /boxes/generate'
    ).toBe(0);

    // Restore validity through the same control: the strip clears and the
    // regen resumes.
    await width.fill('50');
    await expect(page.getByTestId('validation-errors')).toHaveCount(0);
    await expectLogEvent(page, 'box_regen_done', undefined, { timeout: REGEN_WAIT_MS });
    expect(generatePosts).toBeGreaterThanOrEqual(1);
    await page.unroute('**/boxes/generate');
  });
});
