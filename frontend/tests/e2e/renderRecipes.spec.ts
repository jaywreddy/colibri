/**
 * Render-recipe regression spec.
 *
 * Historically every pattern rendered as a flat gold mask on a plane —
 * "cafetero-iridescence" showed stripes, not a rainbow; "wayuu-kanasu-moire"
 * showed a dense weave, not moving moiré fringes. Phases B+C wire three
 * recipes into the shader:
 *
 *  - `iridescent_grating` (uRecipe == 0) — analytical diffraction, cafetero
 *    should now emit visibly different hues as the camera orbits the plate.
 *  - `stereo_lenticular` (uRecipe == 1) — front slit barrier + two scene
 *    layers. Sombrero and caravel should swap scenes as the camera tilts
 *    past the slit-switch angle.
 *  - `moire_interactive` (uRecipe == 2) — front × back sampled with a
 *    Snell-parallax shift; wayuu / compass / emerald should show fringes
 *    whose pixel signature shifts when the view angle changes.
 *
 * We don't try to validate the optics quantitatively here — that's for the
 * backend sanity tests. We just prove the recipe is actually selected and
 * that orbiting the camera changes the pixel output (i.e. the view angle
 * really participates in the shader now).
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
  // Drive OrbitControls directly via the debug hook. We avoid mouse events
  // here because Playwright's drag inside the WebGL canvas is flaky under
  // Chromium's headless compositing — direct controls.setAzimuthalAngle is
  // deterministic.
  await page.evaluate((delta: number) => {
    const t = (window as any).__three;
    const controls = t.controls;
    const curr = controls.getAzimuthalAngle();
    // three's OrbitControls doesn't expose a setter — rotate the camera
    // around the target ourselves, then update.
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
    void curr; // silence unused
  }, dAz);
}

test.describe('render recipes', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
    await waitForThree(page);
    await waitForTexturesBound(page);
  });

  test('cafetero binds iridescent_grating recipe and looks different at different camera angles', async ({
    page,
  }) => {
    await clearLog(page);
    await clickCardAndWait(page, 'cafetero-iridescence');

    const bound = await expectLogEvent(
      page,
      'recipe_bound',
      (e) => e.slug === 'cafetero-iridescence'
    );
    expect(bound.recipe).toBe('iridescent_grating');
    expect(await readUniform<number>(page, 'uRecipe')).toBe(0);

    const period = await readUniform<number>(page, 'uGratingPeriodUm');
    expect(period, 'grating period should come from manifest').toBeGreaterThan(0);

    await expectCanvasNotBlank(page);

    const samplePts: [number, number][] = [
      [120, 120],
      [240, 160],
      [360, 200],
      [480, 240],
    ];
    const before = await sampleMany(page, samplePts);
    await orbitCamera(page, 0.35); // ≈20° azimuth
    await page.waitForTimeout(100);
    const after = await sampleMany(page, samplePts);

    const deltas = before.map((v, i) => Math.abs(v - after[i]));
    const maxDelta = Math.max(...deltas);
    expect(
      maxDelta,
      `iridescent_grating should change the plate's appearance when the camera orbits; before=${JSON.stringify(
        before
      )}, after=${JSON.stringify(after)}`
    ).toBeGreaterThan(3);
  });

  test('sombrero binds stereo_lenticular recipe and tilting swaps the scene', async ({
    page,
  }) => {
    await clearLog(page);
    await clickCardAndWait(page, 'sombrero-vueltiao-parallax');

    const bound = await expectLogEvent(
      page,
      'recipe_bound',
      (e) => e.slug === 'sombrero-vueltiao-parallax'
    );
    expect(bound.recipe).toBe('stereo_lenticular');
    expect(await readUniform<number>(page, 'uRecipe')).toBe(1);

    // The generator emits view_a.png + view_b.png; PlateScene fetches them
    // and logs stereo_views_bound once both are live.
    await expectLogEvent(
      page,
      'stereo_views_bound',
      (e) => e.slug === 'sombrero-vueltiao-parallax'
    );

    await expectCanvasNotBlank(page);

    const samplePts: [number, number][] = [
      [180, 160],
      [260, 190],
      [340, 220],
      [420, 250],
      [500, 280],
    ];
    // Orbit far past the slit-switch threshold (smoothstep ends ~|proj|=0.15,
    // which is a tangent-space projection, so 0.6 rad azimuth is well past).
    const before = await sampleMany(page, samplePts);
    await orbitCamera(page, -0.6);
    await page.waitForTimeout(120);
    const leftTilt = await sampleMany(page, samplePts);
    await orbitCamera(page, 1.2); // swing through normal to the other side
    await page.waitForTimeout(120);
    const rightTilt = await sampleMany(page, samplePts);

    const lrDeltas = leftTilt.map((v, i) => Math.abs(v - rightTilt[i]));
    const swapped = lrDeltas.filter((d) => d > 2).length;
    expect(
      swapped,
      `stereo_lenticular should swap scenes between opposite tilts; ` +
        `before=${JSON.stringify(before)}, left=${JSON.stringify(leftTilt)}, right=${JSON.stringify(rightTilt)}`
    ).toBeGreaterThan(0);
  });

  test('caravel binds stereo_lenticular recipe and tilting reveals the ship', async ({
    page,
  }) => {
    await clearLog(page);
    await clickCardAndWait(page, 'caravel-latent');

    const bound = await expectLogEvent(
      page,
      'recipe_bound',
      (e) => e.slug === 'caravel-latent'
    );
    expect(bound.recipe).toBe('stereo_lenticular');
    expect(await readUniform<number>(page, 'uRecipe')).toBe(1);

    await expectLogEvent(
      page,
      'stereo_views_bound',
      (e) => e.slug === 'caravel-latent'
    );

    await expectCanvasNotBlank(page);

    const samplePts: [number, number][] = [
      [140, 200],
      [230, 220],
      [320, 240],
      [410, 260],
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
      `caravel should reveal the ship on opposite tilt; ` +
        `left=${JSON.stringify(leftTilt)}, right=${JSON.stringify(rightTilt)}`
    ).toBeGreaterThan(0);
  });

  test('wayuu binds moire_interactive recipe and canvas shifts when the camera orbits', async ({
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
    expect(await readUniform<number>(page, 'uRecipe')).toBe(2);

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
      `moire_interactive should produce parallax-driven fringe motion across multiple samples; ` +
        `before=${JSON.stringify(before)}, after=${JSON.stringify(after)}`
    ).toBeGreaterThan(0);
  });
});
