/**
 * Render-recipe regression spec.
 *
 * Post-strip catalog has three recipes:
 *  - `stereo_lenticular` (uRecipe == 0) — colibri-globe-lenticular swaps the
 *    visible image as the camera tilts past the slit-switch threshold.
 *  - `moire_interactive` (uRecipe == 1) — wayuu, emerald-facet, and
 *    colibri-globe-moire all sample front × back with Snell-shifted parallax;
 *    orbit shifts the fringes.
 *  - `phase_shift_overlay` (uRecipe == 2) — colibri-globe-phase biases which
 *    carrier phase the eye samples by the sign of the projected view angle.
 *
 * We prove the recipe is actually selected and that orbiting the camera
 * actually changes the pixel output — the optics quantitatives belong to
 * the backend sanity tests.
 */
import { test, expect } from './fixtures';
import {
  clickCardAndWait,
  clearLog,
  expectCanvasNotBlank,
  expectLogEvent,
  readUniform,
  sampleMany,
  waitForTexturesBound,
  waitForThree,
} from './helpers';

async function orbitCamera(page: import('@playwright/test').Page, dAz: number): Promise<void> {
  await page.evaluate((delta: number) => {
    const t = (window as any).__three;
    const controls = t.controls;
    const target = controls.target;
    const cam = t.camera;
    const dx = cam.position.x - target.x;
    const dz = cam.position.z - target.z;
    const r = Math.hypot(dx, dz);
    const a = Math.atan2(dx, dz) + delta;
    cam.position.x = target.x + r * Math.sin(a);
    cam.position.z = target.z + r * Math.cos(a);
    cam.lookAt(target);
    controls.update();
  }, dAz);
}

test.describe('render recipes', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
    await waitForThree(page);
    await waitForTexturesBound(page);
  });

  test('wayuu binds moire_interactive and canvas shifts when the camera orbits', async ({
    page,
  }) => {
    await clearLog(page);
    await clickCardAndWait(page, 'wayuu-kanasu-moire');

    const bound = await expectLogEvent(
      page,
      'recipe_bound',
      (e) => e.slug === 'wayuu-kanasu-moire'
    );
    expect(bound.recipe).toBe('moire_interactive');
    expect(await readUniform<number>(page, 'uRecipe')).toBe(1);

    await expectCanvasNotBlank(page);

    const samplePts: [number, number][] = [
      [150, 180],
      [220, 210],
      [290, 230],
      [360, 250],
      [430, 270],
    ];
    const before = await sampleMany(page, samplePts);
    await orbitCamera(page, 0.25);
    await page.waitForTimeout(100);
    const after = await sampleMany(page, samplePts);

    const deltas = before.map((v, i) => Math.abs(v - after[i]));
    const shifted = deltas.filter((d) => d > 2).length;
    expect(
      shifted,
      `moire_interactive should shift fringes with parallax; ` +
        `before=${JSON.stringify(before)}, after=${JSON.stringify(after)}`
    ).toBeGreaterThan(0);
  });

  test('colibri-globe-lenticular binds stereo_lenticular and tilting swaps the image', async ({
    page,
  }) => {
    await clearLog(page);
    await clickCardAndWait(page, 'colibri-globe-lenticular');

    const bound = await expectLogEvent(
      page,
      'recipe_bound',
      (e) => e.slug === 'colibri-globe-lenticular'
    );
    expect(bound.recipe).toBe('stereo_lenticular');
    expect(await readUniform<number>(page, 'uRecipe')).toBe(0);

    await expectLogEvent(
      page,
      'stereo_views_bound',
      (e) => e.slug === 'colibri-globe-lenticular'
    );

    await expectCanvasNotBlank(page);

    const samplePts: [number, number][] = [
      [180, 200],
      [260, 220],
      [340, 240],
      [420, 260],
      [500, 280],
    ];
    await orbitCamera(page, -0.6);
    await page.waitForTimeout(120);
    const leftTilt = await sampleMany(page, samplePts);
    await orbitCamera(page, 1.2);
    await page.waitForTimeout(120);
    const rightTilt = await sampleMany(page, samplePts);

    const lrDeltas = leftTilt.map((v, i) => Math.abs(v - rightTilt[i]));
    const swapped = lrDeltas.filter((d) => d > 2).length;
    expect(
      swapped,
      `stereo_lenticular should swap images on opposite tilts; ` +
        `left=${JSON.stringify(leftTilt)}, right=${JSON.stringify(rightTilt)}`
    ).toBeGreaterThan(0);
  });

  test('colibri-globe-moire binds moire_interactive and parallax walks the beat', async ({
    page,
  }) => {
    await clearLog(page);
    await clickCardAndWait(page, 'colibri-globe-moire');

    const bound = await expectLogEvent(
      page,
      'recipe_bound',
      (e) => e.slug === 'colibri-globe-moire'
    );
    expect(bound.recipe).toBe('moire_interactive');
    expect(await readUniform<number>(page, 'uRecipe')).toBe(1);

    await expectCanvasNotBlank(page);

    const samplePts: [number, number][] = [
      [160, 200],
      [240, 220],
      [320, 240],
      [400, 260],
      [480, 280],
    ];
    const before = await sampleMany(page, samplePts);
    await orbitCamera(page, 0.35);
    await page.waitForTimeout(120);
    const after = await sampleMany(page, samplePts);

    const deltas = before.map((v, i) => Math.abs(v - after[i]));
    const shifted = deltas.filter((d) => d > 2).length;
    expect(
      shifted,
      `dual-grating moire should shift beat fringes with tilt; ` +
        `before=${JSON.stringify(before)}, after=${JSON.stringify(after)}`
    ).toBeGreaterThan(0);
  });

  test('colibri-globe-phase binds phase_shift_overlay and tilt biases reveal', async ({
    page,
  }) => {
    await clearLog(page);
    await clickCardAndWait(page, 'colibri-globe-phase');

    const bound = await expectLogEvent(
      page,
      'recipe_bound',
      (e) => e.slug === 'colibri-globe-phase'
    );
    expect(bound.recipe).toBe('phase_shift_overlay');
    expect(await readUniform<number>(page, 'uRecipe')).toBe(2);

    const carrier = await readUniform<number>(page, 'uCarrierPeriodUm');
    expect(carrier, 'carrier period should come from manifest').toBeGreaterThan(0);

    await expectCanvasNotBlank(page);

    const samplePts: [number, number][] = [
      [180, 200],
      [260, 220],
      [340, 240],
      [420, 260],
      [500, 280],
    ];
    await orbitCamera(page, -0.5);
    await page.waitForTimeout(120);
    const leftTilt = await sampleMany(page, samplePts);
    await orbitCamera(page, 1.0);
    await page.waitForTimeout(120);
    const rightTilt = await sampleMany(page, samplePts);

    const lrDeltas = leftTilt.map((v, i) => Math.abs(v - rightTilt[i]));
    const shifted = lrDeltas.filter((d) => d > 2).length;
    expect(
      shifted,
      `phase_shift_overlay should bias the visible image with tilt; ` +
        `left=${JSON.stringify(leftTilt)}, right=${JSON.stringify(rightTilt)}`
    ).toBeGreaterThan(0);
  });
});
