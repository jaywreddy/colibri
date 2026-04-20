import { type Page, type TestInfo, expect } from '@playwright/test';

/**
 * Shape of a single entry in the frontend log ring buffer.
 * Mirrors `LogEvent` in frontend/src/logger.ts.
 */
export type LogEvent = {
  t: number;
  type: string;
  [key: string]: unknown;
};

/**
 * Wait for the Three.js global debug hook to be installed by PlateScene
 * AND for the WebGL context to be live. In headless Chromium the context
 * briefly drops (webglcontextlost / webglcontextrestored) shortly after
 * creation — sampling pixels during that window returns zeros. We arm a
 * one-shot webglcontextrestored listener and wait for a stretch of
 * uninterrupted good state before returning.
 */
export async function waitForThree(page: Page): Promise<void> {
  await page.waitForFunction(() => !!(window as any).__three?.material, null, {
    timeout: 20_000,
  });
  await page.evaluate(
    () =>
      new Promise<void>((resolve) => {
        const t = (window as any).__three;
        const canvas = t.renderer.domElement as HTMLCanvasElement;
        const gl = t.renderer.getContext() as WebGLRenderingContext;
        const onRestored = () => resolve();
        canvas.addEventListener('webglcontextrestored', onRestored, { once: true });
        // If the context hasn't been lost by the time we've given it a
        // settling window, assume it won't be lost during the test.
        setTimeout(() => {
          if (!gl.isContextLost()) resolve();
        }, 2500);
      })
  );
}

/** Wait for the first pattern's textures to be bound (front/back !== blank 1x1). */
export async function waitForTexturesBound(page: Page): Promise<void> {
  await page.waitForFunction(
    () => {
      const t = (window as any).__three;
      if (!t?.material) return false;
      const front = t.material.uniforms.uFront.value;
      return !!front && (front.image?.width ?? 1) > 1;
    },
    null,
    { timeout: 30_000 }
  );
}

/** Read a uniform by name from the active ShaderMaterial. */
export async function readUniform<T = unknown>(page: Page, name: string): Promise<T> {
  return (await page.evaluate((n) => {
    const t = (window as any).__three;
    return t.material.uniforms[n].value;
  }, name)) as T;
}

/**
 * Sample a pixel from the Three.js canvas. Returns [r, g, b, a] in 0..255.
 *
 * The WebGL default framebuffer is cleared/swapped after each render in the
 * RAF loop, so readPixels from an unrelated task returns zeros. We render
 * + readPixels atomically inside a single evaluate to catch the back buffer
 * between write and swap.
 */
export async function sampleCanvasPixel(
  page: Page,
  x: number,
  y: number
): Promise<[number, number, number, number]> {
  return (await page.evaluate(
    ([px, py]) => {
      const t = (window as any).__three;
      t.renderer.render(t.scene, t.camera);
      const gl = t.renderer.getContext() as WebGLRenderingContext;
      const buf = new Uint8Array(4);
      gl.readPixels(px, py, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, buf);
      return [buf[0], buf[1], buf[2], buf[3]] as [number, number, number, number];
    },
    [x, y]
  )) as [number, number, number, number];
}

/** Sample many pixels in a single round-trip. Avoids per-pixel evaluate cost. */
export async function sampleMany(
  page: Page,
  pts: [number, number][]
): Promise<number[]> {
  return (await page.evaluate((coords) => {
    const t = (window as any).__three;
    t.renderer.render(t.scene, t.camera);
    const gl = t.renderer.getContext() as WebGLRenderingContext;
    const buf = new Uint8Array(4);
    const out: number[] = [];
    for (const [x, y] of coords) {
      gl.readPixels(x, y, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, buf);
      out.push((buf[0] + buf[1] + buf[2]) / 3);
    }
    return out;
  }, pts)) as number[];
}

/** Brightness of a pixel sampled from the canvas. */
export async function sampleBrightness(page: Page, x: number, y: number): Promise<number> {
  const [r, g, b] = await sampleCanvasPixel(page, x, y);
  return (r + g + b) / 3;
}

/** Assert the canvas is not entirely black/dark. */
export async function expectCanvasNotBlank(page: Page): Promise<void> {
  const samples = await sampleMany(page, [
    [80, 80],
    [160, 120],
    [240, 200],
    [320, 240],
    [400, 280],
  ]);
  const maxB = Math.max(...samples);
  expect(
    maxB,
    `canvas appears entirely dark, samples=${JSON.stringify(samples)}`
  ).toBeGreaterThan(10);
}

/** Snapshot the frontend log ring buffer. Returns [] before the app boots. */
export async function readLog(page: Page): Promise<LogEvent[]> {
  return (await page.evaluate(() => (window as unknown as { __log?: LogEvent[] }).__log ?? [])) as LogEvent[];
}

/** Clear the frontend log ring buffer — handy between assertions in one spec. */
export async function clearLog(page: Page): Promise<void> {
  await page.evaluate(() => {
    (window as unknown as { __log?: unknown[] }).__log = [];
  });
}

/**
 * Poll the log buffer for an event matching `type` (and optionally a
 * predicate on the whole event). On timeout, fails with the full buffer
 * dumped to the assertion message so failures are self-diagnosing.
 */
export async function expectLogEvent(
  page: Page,
  type: string,
  match?: (ev: LogEvent) => boolean,
  opts: { timeout?: number } = {}
): Promise<LogEvent> {
  const timeout = opts.timeout ?? 15_000;
  const deadline = Date.now() + timeout;
  let lastBuf: LogEvent[] = [];
  while (Date.now() < deadline) {
    lastBuf = await readLog(page);
    const hit = lastBuf.find((e) => e.type === type && (!match || match(e)));
    if (hit) return hit;
    await page.waitForTimeout(50);
  }
  throw new Error(
    `expectLogEvent: no event of type '${type}' within ${timeout}ms. ` +
      `Recent log (last 30):\n${JSON.stringify(lastBuf.slice(-30), null, 2)}`
  );
}

/**
 * Canonical "click card → wait for render" helper. Clicks the Gallery
 * button with the given slug, waits for a `pattern_selected` event with
 * `committed: true` for that slug, then waits for `texture_bound` to fire
 * for the same slug. This is the minimum guarantee that the user's click
 * actually reached the GPU.
 */
export async function clickCardAndWait(page: Page, slug: string): Promise<void> {
  await page.locator(`button[data-slug="${slug}"]`).click();
  await expectLogEvent(
    page,
    'pattern_selected',
    (e) => e.slug === slug && e.committed !== false
  );
  await expectLogEvent(page, 'texture_bound', (e) => e.slug === slug);
}

/**
 * Attach the frontend log (and any backend-log hint) to a failing test.
 * Call from `test.afterEach` in fixtures.ts. Safe to call on pass too —
 * bails out when the test didn't fail.
 */
export async function dumpLogOnFailure(page: Page, testInfo: TestInfo): Promise<void> {
  if (testInfo.status === testInfo.expectedStatus) return;
  try {
    const buf = await readLog(page);
    await testInfo.attach('frontend-log.json', {
      body: JSON.stringify(buf, null, 2),
      contentType: 'application/json',
    });
  } catch {
    // Page may already be closed by the time afterEach fires; nothing to do.
  }
}
