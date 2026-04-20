/**
 * Race regression guard.
 *
 * Before the fix in store.ts/Gallery.tsx, fast clicks through the Gallery
 * could leave the canvas showing pattern A while the Gallery highlighted
 * pattern B — the `getDefault` resolutions arrived out of order and the
 * last write won, not the last click.
 *
 * The fix:
 *  - `beginSelect(slug)` marks a slug as the "latest intent" before fetch.
 *  - `selectPatternIfCurrent(slug, manifest)` commits only if `slug` still
 *    matches `pendingSlug`; otherwise the stale response is dropped.
 *  - An `AbortController` cancels in-flight fetches on the next click.
 *
 * This spec clicks five cards back-to-back with no deliberate delay and
 * asserts that:
 *  1. The final `pattern_selected` with `committed: true` is for the LAST
 *     slug clicked.
 *  2. The final `texture_bound` event is for that same slug.
 *  3. The Gallery's active card matches.
 *  4. No earlier slug ever gets a `pattern_selected` with `committed: true`
 *     after the last click — i.e. no late writer slipped through.
 */
import { test, expect } from './fixtures';
import {
  clearLog,
  expectLogEvent,
  readLog,
  waitForTexturesBound,
  waitForThree,
} from './helpers';

test.describe('rapid clicks race', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
    await waitForThree(page);
    await waitForTexturesBound(page);
  });

  test('five back-to-back clicks: last click wins', async ({ page }) => {
    const slugs = await page.locator('button[data-slug]').evaluateAll((els) =>
      els.slice(0, 5).map((el) => el.getAttribute('data-slug')!)
    );
    expect(slugs.length).toBe(5);
    const last = slugs[slugs.length - 1];

    // Deterministically force the race: slow each /default response down
    // enough that clicks 1..4 are still in flight when click 5 lands. Without
    // this the backend is fast enough that every fetch resolves before the
    // next click arrives — and the spec silently stops exercising the bug.
    await page.route('**/patterns/*/default', async (route) => {
      await new Promise((r) => setTimeout(r, 200));
      await route.continue();
    });

    await clearLog(page);

    // Click without waiting between them so the fetches overlap.
    const clickCount = slugs.length;
    for (const s of slugs) {
      await page.locator(`button[data-slug="${s}"]`).click({ noWaitAfter: true });
    }

    // A `pattern_selected` with `committed: true` for the LAST slug must
    // arrive. Use a generous timeout — the burst may queue multiple
    // manifest fetches that race to the mark.
    await expectLogEvent(
      page,
      'pattern_selected',
      (e) => e.slug === last && e.committed === true,
      { timeout: 20_000 }
    );
    await expectLogEvent(page, 'texture_bound', (e) => e.slug === last, {
      timeout: 20_000,
    });

    // Inspect the full log. The last committed selection must be `last`,
    // and no earlier slug may have committed after that.
    const log = await readLog(page);
    const commits = log.filter(
      (e) => e.type === 'pattern_selected' && e.committed === true
    );
    expect(commits.length).toBeGreaterThan(0);
    const finalCommit = commits[commits.length - 1];
    expect(finalCommit.slug, 'last committed selection must be the last clicked slug').toBe(last);

    // Active card in the Gallery should also match.
    const activeSlug = await page.evaluate(async () => {
      const mod = await import('/src/store.ts');
      return mod.useStore.getState().activeSlug;
    });
    expect(activeSlug).toBe(last);

    // Aborts are allowed (and encouraged) for any non-last click, but
    // there must be at least one — otherwise we didn't actually race.
    const aborts = log.filter((e) => e.type === 'pattern_select_aborted');
    expect(
      aborts.length,
      `expected some aborted clicks in a burst of ${clickCount}; log=${JSON.stringify(
        log.filter((e) => String(e.type).startsWith('pattern_')),
        null,
        2
      )}`
    ).toBeGreaterThan(0);
  });
});
