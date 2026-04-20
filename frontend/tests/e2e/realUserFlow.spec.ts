/**
 * End-to-end user flow driven by real DOM events.
 *
 * Context: the pre-existing suite mostly exercised state via `page.evaluate`,
 * which bypassed the Gallery click handler where the selection race lived.
 * This spec clicks real buttons, slides a real slider, switches real radios
 * — the stuff a user actually does — and cross-checks both the rendered
 * canvas AND the structured log ring buffer.
 */
import { test, expect } from './fixtures';
import {
  clickCardAndWait,
  clearLog,
  expectCanvasNotBlank,
  expectLogEvent,
  readLog,
  sampleMany,
  waitForTexturesBound,
  waitForThree,
} from './helpers';

test.describe('real user flow', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
    await waitForThree(page);
    await waitForTexturesBound(page);
  });

  test('clicking through three different patterns updates canvas each time', async ({ page }) => {
    // Enumerate the first three cards from the live DOM rather than hardcoding
    // slugs — the catalog can grow without this spec rotting.
    const slugs = await page.locator('button[data-slug]').evaluateAll((els) =>
      els.slice(0, 3).map((el) => el.getAttribute('data-slug')!).filter(Boolean)
    );
    expect(slugs.length).toBe(3);

    for (const slug of slugs) {
      await clearLog(page);
      await clickCardAndWait(page, slug);
      await expectCanvasNotBlank(page);

      const log = await readLog(page);
      const selected = log.find((e) => e.type === 'pattern_selected' && e.slug === slug);
      expect(selected, `pattern_selected for ${slug}`).toBeTruthy();
      expect(selected!.committed).toBe(true);

      const bound = log.find((e) => e.type === 'texture_bound' && e.slug === slug);
      expect(bound, `texture_bound for ${slug}`).toBeTruthy();
    }
  });

  test('dragging a parameter slider triggers regen and changes the canvas', async ({ page }) => {
    // Scope to the Parameters panel so we don't accidentally grab the
    // IlluminationPanel's light-az/el sliders (which happen to come first
    // in document order for some layouts).
    const slider = page
      .getByTestId('parameter-panel')
      .locator('input[type="range"]')
      .first();
    await expect(slider).toBeVisible();

    const before = await sampleMany(page, [
      [100, 100],
      [200, 150],
      [300, 200],
    ]);

    await clearLog(page);

    // `fill` fires a React-compatible change event on range inputs.
    const max = await slider.getAttribute('max');
    const min = await slider.getAttribute('min');
    const step = await slider.getAttribute('step');
    // Pick a value near the middle that is clearly different from the default.
    const target = (() => {
      const hi = parseFloat(max ?? '1');
      const lo = parseFloat(min ?? '0');
      const s = parseFloat(step ?? '0.01');
      const mid = lo + (hi - lo) * 0.75;
      return (Math.round(mid / s) * s).toString();
    })();
    await slider.fill(target);

    await expectLogEvent(page, 'param_changed');
    // The debounced regen completes within a couple of seconds on the default
    // pattern — give it 15s to be safe on cold caches.
    await expectLogEvent(page, 'param_regen_done', undefined, { timeout: 15_000 });

    // After regen a new texture_bound should fire.
    await expectLogEvent(page, 'texture_bound', undefined, { timeout: 15_000 });

    // Canvas must actually change. We don't know the direction, just that
    // the pixel signature is no longer the stale one.
    const after = await sampleMany(page, [
      [100, 100],
      [200, 150],
      [300, 200],
    ]);
    const changed = before.some((v, i) => Math.abs(v - after[i]) > 2);
    expect(
      changed,
      `canvas did not change after slider drag; before=${JSON.stringify(
        before
      )}, after=${JSON.stringify(after)}`
    ).toBe(true);
  });

  test('switching illumination and laser color emits the expected log events', async ({ page }) => {
    await clearLog(page);

    await page.locator('button[data-mode="laser"]').click();
    await expectLogEvent(
      page,
      'illumination_changed',
      (e) => e.to === 'laser'
    );

    // Laser color picker appears only in laser mode — pick blue.
    await page.locator('button[data-color="blue"]').click();
    await expectLogEvent(
      page,
      'laser_color_changed',
      (e) => e.to === 'blue'
    );

    await expectCanvasNotBlank(page);
  });
});
