import { type Page, expect } from '@playwright/test';

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
