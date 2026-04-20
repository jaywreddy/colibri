/**
 * Orbit the camera and drag the light sliders. These are the only ways a
 * user can change the view of a selected pattern without re-rendering the
 * texture — so the failure mode they catch is "clicks work, but orbit
 * silently dies after a state update" (classic refs-stale bug).
 */
import { test, expect } from './fixtures';
import {
  clearLog,
  expectLogEvent,
  readLog,
  sampleMany,
  waitForTexturesBound,
  waitForThree,
} from './helpers';

test.describe('camera and lighting interaction', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
    await waitForThree(page);
    await waitForTexturesBound(page);
  });

  test('mouse drag on canvas orbits the camera and emits tilt_changed', async ({ page }) => {
    // OrbitControls live on the Three.js canvas. Drag from roughly the
    // center of the viewport.
    const canvas = page.locator('canvas').first();
    const box = await canvas.boundingBox();
    expect(box).toBeTruthy();
    const cx = box!.x + box!.width / 2;
    const cy = box!.y + box!.height / 2;

    await clearLog(page);

    // A modest horizontal drag — enough to register an azimuth change.
    await page.mouse.move(cx, cy);
    await page.mouse.down();
    for (let i = 1; i <= 10; i++) {
      await page.mouse.move(cx + i * 10, cy);
      // Tiny wait so OrbitControls' damping loop picks up each frame.
      await page.waitForTimeout(16);
    }
    await page.mouse.up();

    await expectLogEvent(page, 'tilt_changed', undefined, { timeout: 5_000 });

    const events = (await readLog(page)).filter((e) => e.type === 'tilt_changed');
    expect(events.length, 'expected multiple tilt_changed events during drag').toBeGreaterThan(1);
  });

  test('dragging light sliders updates uniforms and shifts sampled pixels', async ({ page }) => {
    const az = page.getByTestId('light-az');
    const el = page.getByTestId('light-el');
    await expect(az).toBeVisible();
    await expect(el).toBeVisible();

    const before = await sampleMany(page, [
      [120, 120],
      [240, 180],
      [360, 260],
    ]);

    await clearLog(page);
    await az.fill('-120');
    await el.fill('25');

    await expectLogEvent(page, 'light_moved', (e) => e.az === -120);
    await expectLogEvent(page, 'light_moved', (e) => e.el === 25);

    // Give the render loop a frame to apply the new uniform values before
    // sampling.
    await page.waitForTimeout(100);
    const after = await sampleMany(page, [
      [120, 120],
      [240, 180],
      [360, 260],
    ]);
    const maxDelta = Math.max(...before.map((v, i) => Math.abs(v - after[i])));
    expect(
      maxDelta,
      `moving the light from default to (-120,25) should change shading; before=${JSON.stringify(
        before
      )}, after=${JSON.stringify(after)}`
    ).toBeGreaterThan(2);
  });
});
