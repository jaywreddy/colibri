/**
 * Visual-signature spec — image-based verification harness for the Ring Box
 * Studio. For every scene in `visualCatalog.ts` we:
 *
 *   1. Apply the scene setup (lid angle, layout, illumination, camera) via
 *      window.__studio + the store.
 *   2. Run the cheap native checks declared on the scene (canvas not blank,
 *      lid pivot rotation bounds, layout state) PLUS the scene-graph metalwork
 *      census, which runs for every scene that declares a layout.
 *   3. Capture the canvas + a metadata sidecar via `captureScene(...)`.
 *   4. Enrich the sidecar with the native-check results so the downstream
 *      vision verifier can skip failing scenes without paying tokens.
 *
 * The census is what makes the catalog's metalwork CLAIM ('copper-foil strips
 * along every plate border, solder beads on the bottom and corner seams, and a
 * brass tube-and-rod hinge') a native, deterministic assertion instead of a
 * vision-only one: `tools/visual_verifier.py` needs ANTHROPIC_API_KEY and is NOT
 * part of `just test-e2e`, so before this the whole default gate stayed green
 * with a box that had no metalwork at all. Every count below is DERIVED from
 * src/assembly.ts against the live spec — no magic numbers to drift.
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
import {
  cutList,
  hingeLayout,
  overlapUm,
  platePlacements,
  seamSegments,
} from '../../src/assembly';
import type { BoxSpec } from '../../src/api';

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

/**
 * Every `userData.kind` BoxScene stamps (see BoxScene.tsx::SceneMeshKind). The
 * list is duplicated here on purpose: it is the TEST side of that contract, so a
 * rename in the scene must fail here loudly rather than silently stop counting.
 */
const CENSUS_KINDS = [
  'plate-outer',
  'plate-inner',
  'foil-strip',
  'seam-bead',
  'seam-corner',
  'tin-rim',
  'hinge-tube',
  'hinge-rod',
] as const;
type CensusKind = (typeof CENSUS_KINDS)[number];
type Census = Record<CensusKind, number>;

type SceneCensus = {
  counts: Census;
  /** hinge-tube meshes that hang off lidPivot (they must swing with the lid). */
  lidPivotTubes: number;
  hasLidPivot: boolean;
};

/** Count tagged meshes in the live scene graph (one round-trip). */
async function readSceneCensus(page: import('@playwright/test').Page): Promise<SceneCensus> {
  return (await page.evaluate((kinds) => {
    const s = (window as any).__studio;
    const counts: Record<string, number> = {};
    for (const k of kinds) counts[k] = 0;
    s.scene.traverse((o: any) => {
      const k = o?.userData?.kind;
      if (typeof k === 'string' && k in counts) counts[k] += 1;
    });
    let lidPivotTubes = 0;
    const pivot = s.lidPivot;
    if (pivot) {
      pivot.traverse((o: any) => {
        if (o?.userData?.kind === 'hinge-tube') lidPivotTubes += 1;
      });
    }
    return { counts, lidPivotTubes, hasLidPivot: !!pivot };
  }, CENSUS_KINDS as unknown as string[])) as unknown as SceneCensus;
}

/**
 * What the scene graph MUST contain for a given spec + layout, derived from
 * src/assembly.ts (the same module BoxScene builds from) rather than pinned
 * numbers:
 *
 *   - plate-outer / plate-inner: one of each per plate — 6 (platePlacements in
 *     the assembled layout, cutList in the 2x3 fab grid).
 *   - foil-strip: BoxScene frames each plate on BOTH borders (outer + inner) via
 *     addFoilFrame / addFoilFrameRealistic, whose four strips are skipped when
 *     degenerate; mirrored exactly below (mm units, `ov > 1e-4`, left/right
 *     strips need `h - 2*ov > 0`). Default ring box: 2 frames x 4 strips x 6
 *     plates = 48.
 *   - seam-bead: one per seamSegments() entry (4 bottom + 4 vertical corners).
 *   - seam-corner: one blob per DISTINCT seam endpoint, deduped on the same
 *     mm/2-decimal key BoxScene uses — 8 for a box (4 bottom + 4 top corners).
 *   - tin-rim: the two fixed literal arrays in BoxScene (4 wall top rims + 4 lid
 *     edge faces) = 8; they carry no skip guard.
 *   - hinge-tube: spec.hinge.segments (default 5); hinge-rod: exactly 1.
 *
 * Flat layout is the fab-inspection grid: plates + foil only, and the catalog
 * lists 'Seam beads or hinge visible in flat mode' as a FAIL mode — so every
 * seam/hinge/tin count must be 0 and there must be no lid pivot at all.
 */
function expectedCensus(spec: BoxSpec, layout: 'assembled' | 'flat'): Census {
  const plates = layout === 'assembled' ? platePlacements(spec).length : cutList(spec).length;
  const overlapMm = overlapUm(spec) / 1000;
  let foil = 0;
  for (const c of cutList(spec)) {
    const w = c.width_um / 1000;
    const h = c.height_um / 1000;
    const ov = Math.min(overlapMm, Math.min(w, h) / 2);
    if (ov <= 1e-4) continue;
    const perFrame = (w > 0 ? 2 : 0) + (h - 2 * ov > 0 ? 2 : 0);
    foil += 2 * perFrame; // outer border frame + inner border frame
  }
  const base: Census = {
    'plate-outer': plates,
    'plate-inner': plates,
    'foil-strip': foil,
    'seam-bead': 0,
    'seam-corner': 0,
    'tin-rim': 0,
    'hinge-tube': 0,
    'hinge-rod': 0,
  };
  if (layout === 'flat') return base;

  const seams = seamSegments(spec);
  const corners = new Set<string>();
  for (const s of seams) {
    for (const p of [s.start_um, s.end_um]) {
      corners.add(p.map((v) => (v / 1000).toFixed(2)).join(','));
    }
  }
  return {
    ...base,
    'seam-bead': seams.length,
    'seam-corner': corners.size,
    'tin-rim': 8,
    'hinge-tube': hingeLayout(spec).segments.length,
    'hinge-rod': 1,
  };
}

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

  // --- metalwork census -----------------------------------------------------
  // Runs for every scene that declares a layout (all of them). Guards the
  // catalog's own metalwork claim + fail modes natively: 'flat planes with no
  // metalwork' in the assembled scenes, 'seam beads or hinge visible in flat
  // mode' in the fab grid.
  const censusLayout = n.layoutEquals ?? scene.setup.layout ?? null;
  if (censusLayout) {
    const spec = (await page.evaluate(
      () => (window as any).__studio.store.getState().boxSpec
    )) as BoxSpec;
    const want = expectedCensus(spec, censusLayout);
    const got = await readSceneCensus(page);
    measurements.census = got.counts;
    measurements.censusExpected = want;
    measurements.lidPivotTubes = got.lidPivotTubes;
    for (const kind of CENSUS_KINDS) {
      if (got.counts[kind] !== want[kind]) {
        failures.push(`census '${kind}': ${got.counts[kind]} !== ${want[kind]} expected`);
      }
    }
    if (censusLayout === 'assembled') {
      // The hinge is only real if the lid-owned knuckles actually swing with the
      // lid — a tube parented to the body group would look right in a still and
      // shear through the lid the moment it opens.
      const wantLidTubes = hingeLayout(spec).segments.filter((s) => s.owner === 'lid').length;
      if (!got.hasLidPivot) {
        failures.push('lidPivot missing in assembled layout (hinge cannot open)');
      } else if (got.lidPivotTubes !== wantLidTubes) {
        failures.push(
          `hinge tubes under lidPivot: ${got.lidPivotTubes} !== ${wantLidTubes} expected`
        );
      }
    } else if (got.hasLidPivot) {
      failures.push('lidPivot present in flat layout (hinge must be hidden in the fab grid)');
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
