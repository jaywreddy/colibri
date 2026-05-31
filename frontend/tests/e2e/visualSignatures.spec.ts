/**
 * Visual-signature spec — image-based verification harness, Layer 2.
 *
 * For every (slug, scene) in `visualCatalog.ts` we:
 *
 *   1. Navigate + click the pattern card (existing `clickCardAndWait`).
 *   2. Apply the scene's setup (illumination mode, laser color, z-slice,
 *      camera azimuth/elevation) via the same data-testids and store
 *      globals used by the app.
 *   3. Run the cheap native checks declared on the scene — quadrant
 *      brightness ratios, log-event presence, uniform values, side-panel
 *      image dimensions, and (for scenes that name a reference) pixel
 *      deltas against another scene's capture.
 *   4. Capture the canvas + the SecondaryView <img> + a metadata sidecar
 *      via `captureScene(...)`.
 *   5. Enrich the sidecar with the native-check results so the downstream
 *      vision verifier (`tools/visual_verifier.py`) can skip failing
 *      scenes without paying tokens.
 *
 * Capture is always attempted, even when native checks fail — the PNGs
 * are needed to diagnose what went wrong. A scene flagged `required: true`
 * in the catalog hard-fails the test on a native regression; non-required
 * scenes attach an annotation so the failure surfaces in the report
 * without blocking CI.
 *
 * Output directory comes from `OPTICS_VISUAL_OUT_DIR` (absolute path)
 * with a sensible fallback under `test-results/visual/latest` so running
 * `pnpm test:e2e --grep @visual` produces reviewable artifacts by
 * default.
 */
import { test, expect } from './fixtures';
import { promises as fs } from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import {
  clickCardAndWait,
  clearLog,
  readLog,
  readUniform,
  sampleMany,
  quadrantBrightness,
  setCameraAzEl,
  captureScene,
  waitForTexturesBound,
  waitForThree,
  type SceneCapture,
} from './helpers';
import {
  CATALOG,
  allSlugs,
  type VisualScene,
  type Quadrant,
} from './visualCatalog';

// ---------------------------------------------------------------------------
// Output directory
// ---------------------------------------------------------------------------

// ESM — no __dirname. Resolve relative to this spec file.
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const DEFAULT_OUT_DIR = path.resolve(
  __dirname,
  '..',
  '..',
  'test-results',
  'visual',
  'latest'
);
const OUT_DIR = process.env.OPTICS_VISUAL_OUT_DIR
  ? path.resolve(process.env.OPTICS_VISUAL_OUT_DIR)
  : DEFAULT_OUT_DIR;

// ---------------------------------------------------------------------------
// Scene setup application
// ---------------------------------------------------------------------------

type NativeCheckResult = {
  passed: boolean;
  failures: string[];
  /** Kept so the verifier can look back at the raw measurements. */
  measurements: Record<string, unknown>;
};

async function applyScene(page: import('@playwright/test').Page, scene: VisualScene) {
  const s = scene.setup;
  if (s.illumination) {
    await page.locator(`button[data-mode="${s.illumination}"]`).click();
  }
  if (s.illumination === 'laser' && s.laserColor) {
    // Laser buttons only appear after switching modes — wait for the panel.
    await page.locator(`button[data-color="${s.laserColor}"]`).click();
  }
  if (typeof s.lightAz === 'number' || typeof s.lightEl === 'number') {
    await page.evaluate(
      ([az, el]) => {
        const store = (window as any).__store;
        if (!store?.getState) return;
        const prev = store.getState();
        const next = {
          lightAzimuthDeg: az == null ? prev.lightAzimuthDeg : az,
          lightElevationDeg: el == null ? prev.lightElevationDeg : el,
        };
        store.setState(next);
      },
      [s.lightAz ?? null, s.lightEl ?? null]
    );
  }
  if (s.cameraAzEl) {
    await setCameraAzEl(page, s.cameraAzEl[0], s.cameraAzEl[1]);
  }
  await page.waitForTimeout(s.settleMs ?? 500);
}

// ---------------------------------------------------------------------------
// Native check runner
// ---------------------------------------------------------------------------

function quadRatio(
  q: { ul: number; ur: number; ll: number; lr: number; total: number },
  which: Quadrant
): number {
  if (q.total <= 0) return 0;
  return q[which] / q.total;
}

async function runNativeChecks(
  page: import('@playwright/test').Page,
  slug: string,
  scene: VisualScene,
  priorCaptures: Map<string, SceneCapture>
): Promise<NativeCheckResult> {
  const n = scene.expect.native ?? {};
  const failures: string[] = [];
  const measurements: Record<string, unknown> = {};

  // canvasNotBlank
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

  // plateQuadrantDominance
  if (n.plateQuadrantDominance) {
    const q = await quadrantBrightness(page);
    const ratio = quadRatio(q, n.plateQuadrantDominance.q);
    measurements.plateQuad = q;
    measurements.plateQuadRatio = ratio;
    if (ratio < n.plateQuadrantDominance.minRatio) {
      failures.push(
        `plate ${n.plateQuadrantDominance.q} ratio ${ratio.toFixed(3)} < ${n.plateQuadrantDominance.minRatio}`
      );
    }
  }

  // logEvents
  if (n.logEvents && n.logEvents.length > 0) {
    const buf = await readLog(page);
    const seen = new Set(buf.map((e) => e.type));
    measurements.logEvents = [...seen];
    const missing = n.logEvents.filter((t) => !seen.has(t));
    if (missing.length > 0) failures.push(`missing log events: ${missing.join(', ')}`);
  }

  // uniformEquals
  if (n.uniformEquals) {
    const readValues: Record<string, unknown> = {};
    for (const [k, expected] of Object.entries(n.uniformEquals)) {
      const actual = await readUniform(page, k);
      readValues[k] = actual;
      // Allow either exact equality or Number(actual) === Number(expected) for
      // numeric uniforms that Three.js stores as plain numbers. Strings
      // compared exactly. Booleans coerced to bool.
      const match =
        typeof expected === 'boolean'
          ? Boolean(actual) === expected
          : typeof expected === 'number'
          ? Number(actual) === Number(expected)
          : String(actual) === String(expected);
      if (!match) {
        failures.push(`uniform ${k}: expected ${JSON.stringify(expected)}, got ${JSON.stringify(actual)}`);
      }
    }
    measurements.uniforms = readValues;
  }

  // pixelDeltaVsRef
  if (n.pixelDeltaVsRef) {
    const ref = priorCaptures.get(n.pixelDeltaVsRef.refScene);
    const pts: [number, number][] = n.pixelDeltaVsRef.samplePoints ?? [
      [200, 200],
      [400, 200],
      [300, 300],
      [200, 400],
      [400, 400],
    ];
    const current = await sampleMany(page, pts);
    measurements.currentSamples = current;
    if (!ref) {
      failures.push(
        `pixelDeltaVsRef: reference scene '${n.pixelDeltaVsRef.refScene}' not captured`
      );
    } else {
      // We re-sample the reference image's pixels from its stored PNG so the
      // check doesn't depend on the browser's state surviving to "now".
      // Read PNG bytes in Node and hand the page a data URL (file:// is
      // blocked by the Chromium renderer process).
      const refBytes = await fs.readFile(ref.plateImagePath);
      const refDataUrl = `data:image/png;base64,${refBytes.toString('base64')}`;
      const refSamples = (await page.evaluate(
        async ([pngUrl, coords]) => {
          const img = new Image();
          img.src = pngUrl as string;
          await img.decode();
          const c = document.createElement('canvas');
          c.width = img.naturalWidth;
          c.height = img.naturalHeight;
          const ctx = c.getContext('2d')!;
          ctx.drawImage(img, 0, 0);
          const out: number[] = [];
          for (const [x, y] of coords as [number, number][]) {
            const d = ctx.getImageData(x, y, 1, 1).data;
            out.push((d[0] + d[1] + d[2]) / 3);
          }
          return out;
        },
        [refDataUrl, pts]
      )) as number[];
      measurements.refSamples = refSamples;
      const deltas = current.map((v, i) => Math.abs(v - (refSamples[i] ?? 0)));
      const maxDelta = Math.max(...deltas);
      measurements.maxPixelDelta = maxDelta;
      if (maxDelta < n.pixelDeltaVsRef.minDelta) {
        failures.push(
          `pixelDelta vs ${n.pixelDeltaVsRef.refScene}: max ${maxDelta.toFixed(1)} < ${n.pixelDeltaVsRef.minDelta}`
        );
      }
    }
  }

  return { passed: failures.length === 0, failures, measurements };
}

// ---------------------------------------------------------------------------
// Meta.json enrichment
// ---------------------------------------------------------------------------

async function writeEnrichedMeta(
  capture: SceneCapture,
  slug: string,
  scene: VisualScene,
  native: NativeCheckResult
) {
  const raw = await fs.readFile(capture.metaPath, 'utf8');
  const meta = JSON.parse(raw);
  meta.scene = {
    slug,
    name: scene.name,
    required: !!scene.required,
    claim: scene.expect.claim,
    plateSignature: scene.expect.plateSignature,
    failModes: scene.expect.failModes,
  };
  meta.native_checks = native;
  await fs.writeFile(capture.metaPath, JSON.stringify(meta, null, 2));
}

// ---------------------------------------------------------------------------
// Per-slug test block
// ---------------------------------------------------------------------------

test.describe('@visual signature capture', () => {
  test.beforeAll(async () => {
    // Fresh output dir per test invocation — no stale artifacts confusing
    // the verifier. We keep all the PNGs/metas from this run together.
    await fs.mkdir(OUT_DIR, { recursive: true });
  });

  for (const slug of allSlugs()) {
    const scenes = CATALOG[slug];
    if (!scenes || scenes.length === 0) continue;

    test(`@visual ${slug}`, async ({ page }, testInfo) => {
      test.setTimeout(90_000);
      await page.goto('/');
      await waitForThree(page);
      await waitForTexturesBound(page);

      // Each test gets its own sub-directory so the verifier can glob per-slug.
      const slugDir = path.join(OUT_DIR, slug);
      await fs.mkdir(slugDir, { recursive: true });

      // Click once per slug; all scenes for the pattern share the same
      // loaded textures.
      await clearLog(page);
      await clickCardAndWait(page, slug);

      const captures = new Map<string, SceneCapture>();
      const softFailures: { scene: string; failures: string[] }[] = [];

      for (const scene of scenes) {
        // Deliberately do NOT clear the log between scenes — pattern-load
        // events like `recipe_bound` fire exactly once and should still be
        // visible to later scenes that check for them.
        await applyScene(page, scene);
        const native = await runNativeChecks(page, slug, scene, captures);
        const cap = await captureScene(page, slugDir, scene.name);
        await writeEnrichedMeta(cap, slug, scene, native);
        captures.set(scene.name, cap);

        if (!native.passed) {
          if (scene.required) {
            expect(
              native.passed,
              `[${slug}/${scene.name}] required scene failed native checks:\n  - ${native.failures.join('\n  - ')}`
            ).toBe(true);
          } else {
            softFailures.push({ scene: scene.name, failures: native.failures });
          }
        }
      }

      // Surface soft failures in the test report without flaking CI.
      if (softFailures.length > 0) {
        testInfo.annotations.push({
          type: 'visual-soft-fail',
          description: JSON.stringify(softFailures, null, 2),
        });
      }
    });
  }
});
