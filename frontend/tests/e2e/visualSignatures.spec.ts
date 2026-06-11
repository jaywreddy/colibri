/**
 * Visual-signature spec — image-based verification harness for the Ring Box
 * Studio. For every scene in `visualCatalog.ts` we:
 *
 *   1. Apply the scene setup (lid angle, layout, illumination, camera) via
 *      window.__studio + the store.
 *   2. Run the cheap native checks declared on the scene (canvas not blank,
 *      lid pivot rotation bounds, layout state).
 *   3. Capture the canvas + a metadata sidecar via `captureScene(...)`.
 *   4. Enrich the sidecar with the native-check results so the downstream
 *      vision verifier can skip failing scenes without paying tokens.
 *
 * Capture is always attempted, even when native checks fail — the PNGs are
 * needed to diagnose what went wrong. A scene flagged `required: true`
 * hard-fails the test on a native regression.
 *
 * Output directory comes from `OPTICS_VISUAL_OUT_DIR` (absolute path) with a
 * fallback under `test-results/visual/latest`.
 */
import { test, expect } from './fixtures';
import { promises as fs } from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import {
  captureScene,
  expectCanvasNotBlank,
  sampleMany,
  setCameraAzEl,
  waitForBoxTextures,
  waitForStudio,
  type SceneCapture,
} from './helpers';
import { BOX_SCENES, type BoxVisualScene } from './visualCatalog';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const DEFAULT_OUT_DIR = path.resolve(__dirname, '..', '..', 'test-results', 'visual', 'latest');
const OUT_DIR = process.env.OPTICS_VISUAL_OUT_DIR
  ? path.resolve(process.env.OPTICS_VISUAL_OUT_DIR)
  : DEFAULT_OUT_DIR;

type NativeCheckResult = {
  passed: boolean;
  failures: string[];
  measurements: Record<string, unknown>;
};

async function applyScene(page: import('@playwright/test').Page, scene: BoxVisualScene) {
  const s = scene.setup;
  await page.evaluate(
    ([lidDeg, layout, illumination]) => {
      const st = (window as any).__studio.store.getState();
      if (layout) st.setLayout(layout);
      if (typeof lidDeg === 'number') st.setLidTargetDeg(lidDeg);
      if (illumination) st.setIllumination(illumination);
    },
    [s.lidDeg ?? null, s.layout ?? null, s.illumination ?? null] as const
  );
  if (s.cameraAzEl) {
    await setCameraAzEl(page, s.cameraAzEl[0], s.cameraAzEl[1]);
  }
  await page.waitForTimeout(s.settleMs ?? 800);
}

async function runNativeChecks(
  page: import('@playwright/test').Page,
  scene: BoxVisualScene
): Promise<NativeCheckResult> {
  const n = scene.expect.native ?? {};
  const failures: string[] = [];
  const measurements: Record<string, unknown> = {};

  if (n.canvasNotBlank) {
    const samples = await sampleMany(page, [
      [80, 80],
      [160, 120],
      [240, 200],
      [320, 240],
      [400, 280],
    ]);
    measurements.canvasSamples = samples;
    if (Math.max(...samples) <= 10) {
      failures.push(`canvas appears blank (max sample = ${Math.max(...samples).toFixed(1)})`);
    }
  }

  if (n.lidRotationAtLeastDeg !== undefined || n.lidRotationAtMostDeg !== undefined) {
    const rotDeg = await page.evaluate(() => {
      const p = (window as any).__studio.lidPivot;
      return p ? Math.abs((p.rotation.x * 180) / Math.PI) : null;
    });
    measurements.lidRotationDeg = rotDeg;
    if (rotDeg === null) {
      failures.push('lidPivot missing (flat layout or scene not built)');
    } else {
      if (n.lidRotationAtLeastDeg !== undefined && rotDeg < n.lidRotationAtLeastDeg) {
        failures.push(`lid rotation ${rotDeg.toFixed(1)}deg < ${n.lidRotationAtLeastDeg}deg`);
      }
      if (n.lidRotationAtMostDeg !== undefined && rotDeg > n.lidRotationAtMostDeg) {
        failures.push(`lid rotation ${rotDeg.toFixed(1)}deg > ${n.lidRotationAtMostDeg}deg`);
      }
    }
  }

  if (n.layoutEquals) {
    const layout = await page.evaluate(
      () => (window as any).__studio.store.getState().layout
    );
    measurements.layout = layout;
    if (layout !== n.layoutEquals) {
      failures.push(`layout '${layout}' !== '${n.layoutEquals}'`);
    }
  }

  return { passed: failures.length === 0, failures, measurements };
}

async function writeEnrichedMeta(
  capture: SceneCapture,
  scene: BoxVisualScene,
  native: NativeCheckResult
) {
  const raw = await fs.readFile(capture.metaPath, 'utf8');
  const meta = JSON.parse(raw);
  meta.scene = {
    name: scene.name,
    required: !!scene.required,
    claim: scene.expect.claim,
    signature: scene.expect.signature,
    failModes: scene.expect.failModes,
  };
  meta.native_checks = native;
  await fs.writeFile(capture.metaPath, JSON.stringify(meta, null, 2));
}

test.describe('@visual box signature capture', () => {
  test.beforeAll(async () => {
    await fs.mkdir(OUT_DIR, { recursive: true });
  });

  test('@visual ring-box scenes (closed, open-100, flat)', async ({ page }, testInfo) => {
    test.setTimeout(180_000);
    await page.goto('/');
    await waitForStudio(page);
    await waitForBoxTextures(page);
    await expectCanvasNotBlank(page);

    const outDir = path.join(OUT_DIR, 'ring-box');
    await fs.mkdir(outDir, { recursive: true });

    const softFailures: { scene: string; failures: string[] }[] = [];

    for (const scene of BOX_SCENES) {
      await applyScene(page, scene);
      const native = await runNativeChecks(page, scene);
      const cap = await captureScene(page, outDir, scene.name);
      await writeEnrichedMeta(cap, scene, native);

      if (!native.passed) {
        if (scene.required) {
          expect(
            native.passed,
            `[${scene.name}] required scene failed native checks:\n  - ${native.failures.join('\n  - ')}`
          ).toBe(true);
        } else {
          softFailures.push({ scene: scene.name, failures: native.failures });
        }
      }
    }

    if (softFailures.length > 0) {
      testInfo.annotations.push({
        type: 'visual-soft-fail',
        description: JSON.stringify(softFailures, null, 2),
      });
    }
  });
});
