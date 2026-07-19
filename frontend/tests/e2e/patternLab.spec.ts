/**
 * Pattern Lab (@lab) — the 2D dual-layer preview panel. Pure canvas-2D, no
 * WebGL involvement: assert the panel opens from the header, the composite
 * paints non-blank, and tilt / illumination changes actually recomposite.
 * Pattern regeneration round-trips are exercised by the backend suite, so
 * here we only assert the Regenerate button is present and enabled.
 */
import type { Page } from '@playwright/test';
import { test, expect } from './fixtures';
import { waitForStudio } from './helpers';

/** Sparse pixel signature of the lab canvas (stride-sampled RGB values). */
async function labSignature(page: Page): Promise<number[]> {
  return await page.evaluate(() => {
    const cv = document.querySelector(
      '[data-testid="lab-canvas"]'
    ) as HTMLCanvasElement | null;
    if (!cv) return [];
    const ctx = cv.getContext('2d');
    if (!ctx) return [];
    const d = ctx.getImageData(0, 0, cv.width, cv.height).data;
    const out: number[] = [];
    for (let i = 0; i < d.length; i += 4 * 173) out.push(d[i], d[i + 1], d[i + 2]);
    return out;
  });
}

/** Count of signature entries that moved by more than a small epsilon. */
function diffCount(a: number[], b: number[]): number {
  let n = 0;
  const len = Math.min(a.length, b.length);
  for (let i = 0; i < len; i++) if (Math.abs(a[i] - b[i]) > 6) n++;
  return n;
}

/** Open the lab and wait until the 2D composite actually painted. */
async function openLab(page: Page): Promise<void> {
  await page.getByTestId('lab-toggle').click();
  await expect(page.getByTestId('pattern-lab')).toBeVisible();
  // Manifest + two mask PNGs must land first (a cold pattern materializes
  // ~2 s server-side, then caches), so allow a generous window.
  await page.waitForFunction(
    () => {
      const cv = document.querySelector(
        '[data-testid="lab-canvas"]'
      ) as HTMLCanvasElement | null;
      if (!cv || cv.width === 0) return false;
      const ctx = cv.getContext('2d');
      if (!ctx) return false;
      const d = ctx.getImageData(0, 0, cv.width, cv.height).data;
      for (let i = 0; i < d.length; i += 4 * 173) {
        if (d[i] + d[i + 1] + d[i + 2] > 24) return true;
      }
      return false;
    },
    null,
    { timeout: 30_000 }
  );
}

test.describe('@lab Pattern Lab 2D preview', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
    await waitForStudio(page);
    await openLab(page);
  });

  test('@lab panel opens from the header and paints a non-blank composite', async ({
    page,
  }) => {
    await expect(page.getByTestId('lab-canvas')).toBeVisible();
    await expect(page.getByTestId('lab-pattern')).toBeVisible();
    await expect(page.getByTestId('lab-tilt-x')).toBeVisible();
    await expect(page.getByTestId('lab-tilt-y')).toBeVisible();
    await expect(page.getByTestId('lab-illum')).toBeVisible();
    // openLab already proved non-blank pixels; double-check the signature.
    const sig = await labSignature(page);
    expect(Math.max(...sig)).toBeGreaterThan(8);
    // Proof capture for the visual-harness output dir (same convention as
    // the @effects suite: PNGs under test-results/visual/latest/).
    const path = await import('path');
    const { promises: fs } = await import('fs');
    const { fileURLToPath } = await import('url');
    const here = path.dirname(fileURLToPath(import.meta.url));
    const outDir = path.resolve(
      process.env.OPTICS_VISUAL_OUT_DIR ??
        path.join(here, '..', '..', 'test-results', 'visual', 'latest'),
      'lab'
    );
    await fs.mkdir(outDir, { recursive: true });
    await page
      .getByTestId('pattern-lab')
      .screenshot({ path: path.join(outDir, 'pattern-lab-panel.png') });
  });

  test('@lab tilt-x slider recomposites and reports a nonzero um shift', async ({
    page,
  }) => {
    const before = await labSignature(page);
    await expect(page.getByTestId('lab-shift-readout')).toContainText('Δx 0.00 um');

    await page.getByTestId('lab-tilt-x').fill('24');
    // Recomposite is a synchronous effect — poll for the pixel change.
    await expect
      .poll(async () => diffCount(await labSignature(page), before), {
        timeout: 10_000,
      })
      .toBeGreaterThan(5);
    // 24 deg through 500 um fused silica -> ~145 um of back-mask slide.
    await expect(page.getByTestId('lab-shift-readout')).not.toContainText(
      'Δx 0.00 um'
    );
    await expect(page.getByTestId('lab-shift-readout')).toContainText('um');
  });

  test('@lab switching illumination recomposites the canvas', async ({ page }) => {
    const before = await labSignature(page);
    await page.getByTestId('lab-illum').selectOption('laser');
    await expect
      .poll(async () => diffCount(await labSignature(page), before), {
        timeout: 10_000,
      })
      .toBeGreaterThan(10);

    const laser = await labSignature(page);
    await page.getByTestId('lab-illum').selectOption('backlight');
    await expect
      .poll(async () => diffCount(await labSignature(page), laser), {
        timeout: 10_000,
      })
      .toBeGreaterThan(10);
  });

  test('@lab zone quick-sets appear for period-carrying patterns and land on the switch peak', async ({
    page,
  }) => {
    // The default face pattern (wayuu) carries no slit/carrier period in
    // recipe_data, so no zone UI is shown.
    await expect(page.getByTestId('lab-zone-pos')).toHaveCount(0);

    // A barrier pattern advertises slit_period_um (default p = 40 um) — the
    // zone readout and first-zone quick-sets must appear. Cold generation of
    // the default variant can take a couple of seconds server-side.
    await page.getByTestId('lab-pattern').selectOption('colibri-globe-lenticular');
    await expect(page.getByTestId('lab-zone-pos')).toBeVisible({ timeout: 30_000 });

    // Slit barriers switch at ±p/4 of back-mask shift (audited: the ±p/2
    // point aliases back to a 50/50 blend), so the quick-set lands there
    // (inverse Snell: tilt = asin(n·sin(atan((p/4)/t))) ≈ 1.67° at p=40).
    await page.getByTestId('lab-zone-pos').click();
    await expect(page.getByTestId('lab-zone-readout')).toContainText('+0.25 × period');
    await page.getByTestId('lab-zone-neg').click();
    await expect(page.getByTestId('lab-zone-readout')).toContainText('-0.25 × period');
    await page.getByTestId('lab-zone-zero').click();
    await expect(page.getByTestId('lab-zone-readout')).toContainText('0.00 × period');

    // Proof capture with the zone UI visible (companion to the panel shot).
    const path = await import('path');
    const { promises: fs } = await import('fs');
    const { fileURLToPath } = await import('url');
    const here = path.dirname(fileURLToPath(import.meta.url));
    const outDir = path.resolve(
      process.env.OPTICS_VISUAL_OUT_DIR ??
        path.join(here, '..', '..', 'test-results', 'visual', 'latest'),
      'lab'
    );
    await fs.mkdir(outDir, { recursive: true });
    await page.getByTestId('lab-zone-pos').click();
    await page
      .getByTestId('pattern-lab')
      .screenshot({ path: path.join(outDir, 'pattern-lab-zone.png') });
  });

  test('@lab regenerate button exists and is enabled once masks load', async ({
    page,
  }) => {
    // Skip the slow POST /patterns/generate round-trip — the backend suite
    // covers it. Presence + enabled is the UI contract here.
    await expect(page.getByTestId('lab-regenerate')).toBeVisible();
    await expect(page.getByTestId('lab-regenerate')).toBeEnabled();
  });
});
