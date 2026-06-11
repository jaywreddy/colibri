import { type Page, type TestInfo, expect } from '@playwright/test';
import { promises as fs } from 'fs';
import path from 'path';

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
 * Wait for the Ring Box Studio debug hook (window.__studio, installed by
 * BoxScene) AND for the WebGL context to be live. In headless Chromium the
 * context briefly drops shortly after creation — sampling pixels during that
 * window returns zeros. We arm a one-shot webglcontextrestored listener and
 * wait for a stretch of uninterrupted good state before returning.
 */
export async function waitForStudio(page: Page): Promise<void> {
  await page.waitForFunction(() => !!(window as any).__studio?.renderer, null, {
    timeout: 20_000,
  });
  await page.evaluate(
    () =>
      new Promise<void>((resolve) => {
        const s = (window as any).__studio;
        const canvas = s.renderer.domElement as HTMLCanvasElement;
        const gl = s.renderer.getContext() as WebGLRenderingContext;
        const onRestored = () => resolve();
        canvas.addEventListener('webglcontextrestored', onRestored, { once: true });
        setTimeout(() => {
          if (!gl.isContextLost()) resolve();
        }, 2500);
      })
  );
}

/** Wait for the box manifest's pattern textures to bind (front !== blank 1x1). */
export async function waitForBoxTextures(page: Page): Promise<void> {
  await page.waitForFunction(
    () => {
      const s = (window as any).__studio;
      if (!s?.faces) return false;
      const front = s.faces.front?.shader?.uniforms?.uFront?.value;
      return !!front && (front.image?.width ?? 1) > 1;
    },
    null,
    { timeout: 60_000 }
  );
}

/** Read a uniform by name from one face's pattern ShaderMaterial. */
export async function readFaceUniform<T = unknown>(
  page: Page,
  faceId: string,
  name: string
): Promise<T> {
  return (await page.evaluate(
    ([fid, n]) => {
      const s = (window as any).__studio;
      return s.faces[fid].shader.uniforms[n].value;
    },
    [faceId, name]
  )) as T;
}

/**
 * Sample a pixel from the Three.js canvas. Returns [r, g, b, a] in 0..255.
 * Render + readPixels are done atomically inside a single evaluate to catch
 * the back buffer between write and swap.
 */
export async function sampleCanvasPixel(
  page: Page,
  x: number,
  y: number
): Promise<[number, number, number, number]> {
  return (await page.evaluate(
    ([px, py]) => {
      const s = (window as any).__studio;
      s.renderer.render(s.scene, s.camera);
      const gl = s.renderer.getContext() as WebGLRenderingContext;
      const buf = new Uint8Array(4);
      gl.readPixels(px, py, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, buf);
      return [buf[0], buf[1], buf[2], buf[3]] as [number, number, number, number];
    },
    [x, y]
  )) as [number, number, number, number];
}

/** Sample many pixels in a single round-trip. Returns mean luminance per point. */
export async function sampleMany(page: Page, pts: [number, number][]): Promise<number[]> {
  return (await page.evaluate((coords) => {
    const s = (window as any).__studio;
    s.renderer.render(s.scene, s.camera);
    const gl = s.renderer.getContext() as WebGLRenderingContext;
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
  return (await page.evaluate(
    () => (window as unknown as { __log?: LogEvent[] }).__log ?? []
  )) as LogEvent[];
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
 * Programmatic camera placement via the studio handles. Values in degrees;
 * elevation measured from +Y (0 = overhead).
 */
export async function setCameraAzEl(page: Page, azDeg: number, elDeg: number): Promise<void> {
  await page.evaluate(
    ([az, el]) => {
      const s = (window as any).__studio;
      if (!s?.controls || !s?.camera) return;
      const camera = s.camera;
      const controls = s.controls;
      const target = controls.target ?? { x: 0, y: 0, z: 0 };
      const azRad = (az * Math.PI) / 180;
      const phi = ((90 - el) * Math.PI) / 180;
      const dx = camera.position.x - target.x;
      const dy = camera.position.y - target.y;
      const dz = camera.position.z - target.z;
      const radius = Math.sqrt(dx * dx + dy * dy + dz * dz) || 2.5;
      const sinPhi = Math.sin(phi);
      camera.position.set(
        target.x + radius * sinPhi * Math.sin(azRad),
        target.y + radius * Math.cos(phi),
        target.z + radius * sinPhi * Math.cos(azRad)
      );
      camera.lookAt(target.x, target.y, target.z);
      if (typeof controls.update === 'function') controls.update();
    },
    [azDeg, elDeg]
  );
}

export type SceneCapture = {
  imagePath: string;
  metaPath: string;
  state: Record<string, unknown>;
};

/**
 * Capture a visual scene: the Three.js canvas (PNG) + a JSON metadata
 * sidecar describing the studio state (box dims, lid angle, layout).
 */
export async function captureScene(
  page: Page,
  outDir: string,
  name: string
): Promise<SceneCapture> {
  await fs.mkdir(outDir, { recursive: true });
  const safe = name.replace(/[^a-z0-9._-]/gi, '_');

  const canvasDataUrl = (await page.evaluate(() => {
    const s = (window as any).__studio;
    s.renderer.render(s.scene, s.camera);
    const canvas = s.renderer.domElement as HTMLCanvasElement;
    return canvas.toDataURL('image/png');
  })) as string;
  const imagePath = path.join(outDir, `${safe}.png`);
  const b64 = canvasDataUrl.split(',', 2)[1] ?? '';
  await fs.writeFile(imagePath, Buffer.from(b64, 'base64'));

  const state = (await page.evaluate(() => {
    const s = (window as any).__studio;
    const st = s?.store?.getState?.() ?? {};
    const spec = st.boxSpec ?? {};
    const buf = (window as any).__log ?? [];
    return {
      dims_um: { width: spec.width_um, depth: spec.depth_um, height: spec.height_um },
      foilFinish: spec.foil?.finish ?? null,
      hingeSegments: spec.hinge?.segments ?? null,
      lidTargetDeg: st.lidTargetDeg ?? null,
      lidCurrentDeg: s?.getLidDeg?.() ?? null,
      lidPivotRotX: s?.lidPivot?.rotation?.x ?? null,
      layout: st.layout ?? null,
      illumination: st.illumination ?? null,
      manifestId: st.boxManifest?.id ?? null,
      contentHash: st.boxManifest?.content_hash ?? null,
      logTail: Array.isArray(buf) ? buf.slice(-30) : [],
      timestamp: new Date().toISOString(),
    };
  })) as Record<string, unknown>;

  const metaPath = path.join(outDir, `${safe}.meta.json`);
  await fs.writeFile(
    metaPath,
    JSON.stringify({ name: safe, ...state, image: path.basename(imagePath) }, null, 2)
  );

  return { imagePath, metaPath, state };
}

/**
 * Attach the frontend log to a failing test. Call from `test.afterEach` in
 * fixtures.ts. Safe to call on pass too — bails out when the test passed.
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
