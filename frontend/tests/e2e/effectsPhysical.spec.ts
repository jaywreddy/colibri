/**
 * Physical-honesty effects suite (@effects) — verifies every view-dependent
 * effect the 3D renderer claims, using pixel metrics on the live WebGL
 * buffer. See effectsCatalog.ts for the four honesty axioms and per-scenario
 * claims; every scenario here also dumps a PNG frame sequence + meta.json
 * (with the measured metrics) so `tools/visual_verifier.py` can grade the
 * sequences semantically and humans can eyeball a failure.
 *
 * These tests are the anti-cheat gate the project was missing: a shader that
 * fakes moire procedurally, scrolls a texture over time, or ignores the
 * substrate physics fails here even if a static screenshot looks right.
 *
 * Three of the four axioms used to be enforced only as RELATIVE pixel deltas,
 * which any view-keyed procedural shader also satisfies. The quantitative gates
 * that close that hole:
 *   - GEOMETRIC (absolute): the inner plane's gap must EQUAL the manifest's
 *     paraxial T/n, not merely be nonzero (moire-parallax-physics).
 *   - TEXTURE-DRIVEN: swapping only the bound mask at a fixed camera and gap
 *     must change the plate pixels, and every plane's bound image must be the
 *     PNG its own face manifest declares (carrier-reveal-tilt +
 *     moire-fringe-flow).
 *   - ALL SIX FACES run the two-plane foliage_moire recipe on both planes with
 *     real masks, and no face was refused (moire-fringe-flow).
 *
 * Run: `just test-effects` (Playwright, single worker — heavy process).
 */
import { test, expect } from './fixtures';
import { promises as fs } from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import { readLog, setCameraAzEl, waitForBoxTextures, waitForStudio } from './helpers';
import {
  allFaceRenderState,
  assignFacePattern,
  captureGray,
  diffFrames,
  dumpFramePng,
  facePlateROI,
  faceRecipeId,
  faceSubstrate,
  lidRotationDeg,
  scaleBackPlaneGap,
  zoomForMicroPatterns,
  settle,
  waitForAllFaceMasks,
  waitForStableFrame,
  FACE_IDS,
  GAP_COLLAPSE_FACTOR,
  type DiffMetrics,
  type GrayFrame,
  type Roi,
} from './effectsHelpers';
import { EFFECT_SCENARIOS, TEST_PATTERNS } from './effectsCatalog';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const DEFAULT_OUT = path.resolve(__dirname, '..', '..', 'test-results', 'visual', 'latest');
const OUT_ROOT = path.join(
  process.env.OPTICS_VISUAL_OUT_DIR ? path.resolve(process.env.OPTICS_VISUAL_OUT_DIR) : DEFAULT_OUT,
  'effects'
);

type Page = import('@playwright/test').Page;

/** Write the verifier-compatible sidecar for one scenario. */
async function writeMeta(
  scenarioKey: keyof typeof EFFECT_SCENARIOS,
  outDir: string,
  frames: string[],
  metrics: Record<string, unknown>,
  failures: string[]
): Promise<void> {
  const sc = EFFECT_SCENARIOS[scenarioKey];
  const meta = {
    name: sc.name,
    kind: 'effect-sequence',
    // First frame doubles as the legacy single-image field so older
    // verifier builds still pick the scene up.
    image: frames.length > 0 ? path.basename(frames[0]) : undefined,
    frames: frames.map((f) => path.basename(f)),
    scene: {
      name: sc.name,
      required: true,
      claim: sc.claim,
      signature: sc.signature,
      failModes: sc.failModes,
    },
    native_checks: {
      passed: failures.length === 0,
      failures,
      measurements: metrics,
    },
    timestamp: new Date().toISOString(),
  };
  await fs.writeFile(path.join(outDir, `${sc.name}.meta.json`), JSON.stringify(meta, null, 2));
}

/** Round DiffMetrics for readable meta.json / log output. */
const rd = (m: DiffMetrics) => ({
  mad: Number(m.mad.toFixed(3)),
  changedFrac: Number(m.changedFrac.toFixed(4)),
  corr: Number(m.corr.toFixed(4)),
});

async function gotoStudio(page: Page): Promise<void> {
  await page.goto('/');
  await waitForStudio(page);
  await waitForBoxTextures(page);
  // Deterministic baseline: assembled, closed, ambient, turntable off.
  await page.evaluate(() => {
    const st = (window as any).__studio.store.getState();
    st.setLayout('assembled');
    st.setLidTargetDeg(0);
    st.setIllumination('ambient');
    st.setAutoRotate(false);
  });
  // Quiescence by measurement, not by wall clock (see waitForStableFrame).
  await waitForStableFrame(page, 900);
}

/** Head-on view of the front face, slight elevation. */
async function faceFrontOn(page: Page, azDeg = 0, elDeg = 4): Promise<void> {
  await setCameraAzEl(page, azDeg, elDeg);
  await waitForStableFrame(page, 400);
}

/**
 * The front plate's ROI, computed fresh at the current camera. Fails the
 * test loudly if the plate is off-screen (harness bug, not a render bug).
 */
async function frontROI(page: Page): Promise<Roi> {
  const roi = await facePlateROI(page, 'front');
  expect(roi, 'front plate ROI could not be projected (face off-screen?)').not.toBeNull();
  return roi!;
}

/**
 * Sweep the camera azimuth and capture the front-plate ROI at each stop.
 * The ROI is FIXED from the center frame so all captures are comparable —
 * over +/-10 deg the projected plate moves only slightly and the inset
 * absorbs it.
 */
async function azimuthSweep(
  page: Page,
  azimuths: number[],
  elDeg: number,
  roi: Roi
): Promise<GrayFrame[]> {
  const frames: GrayFrame[] = [];
  for (const az of azimuths) {
    await setCameraAzEl(page, az, elDeg);
    await waitForStableFrame(page, 300);
    frames.push(await captureGray(page, roi));
  }
  return frames;
}

test.describe('@effects physical honesty of renderer effects', () => {
  /**
   * Pre-warm the two non-default pattern variants this suite assigns
   * (TEST_PATTERNS.stereo, TEST_PATTERNS.reveal), one at a time, BEFORE any
   * page exists. Both assignments used to pay a cold pattern generate inside a
   * test, hidden behind assignFacePattern's 90 s wait, and nothing else in the
   * run reused it. Materializing the default variant here (GET
   * /{slug}/default, the same variant the specs assign) leaves the in-test wait
   * covering only the plate compose + texture bind.
   *
   * Strictly sequential and strictly before the first `goto`, so this never
   * runs beside a box regen (CLAUDE.md's single-heavy-compute rule). beforeAll
   * can only use worker-scoped fixtures, hence a hand-built request context
   * against the project's own baseURL (vite proxies /patterns to the backend).
   */
  test.beforeAll(async ({ playwright }, testInfo) => {
    test.setTimeout(300_000); // two cold pattern generates, sequential
    const baseURL = (testInfo.project.use.baseURL as string | undefined) ?? '';
    const api = await playwright.request.newContext(baseURL ? { baseURL } : {});
    try {
      for (const slug of [TEST_PATTERNS.stereo, TEST_PATTERNS.reveal]) {
        const t0 = Date.now();
        const r = await api.get(`/patterns/${slug}/default`, { timeout: 140_000 });
        // A pre-warm failure is not a test failure: the in-test assignment
        // still generates on demand (slower). Log it so a slow run is
        // explainable instead of mysterious.
        console.log(
          `[effects] pre-warm ${slug}: HTTP ${r.status()} in ${Date.now() - t0} ms`
        );
      }
    } finally {
      await api.dispose();
    }
  });

  test.beforeEach(async ({ page }) => {
    await gotoStudio(page);
  });

  test('@effects time-invariance: static camera => static frame', async ({ page }) => {
    const outDir = path.join(OUT_ROOT, 'time-invariance');
    await fs.mkdir(outDir, { recursive: true });
    await faceFrontOn(page, 20, 25);
    await waitForStableFrame(page, 800); // damping fully decayed (measured)

    const f0 = await captureGray(page, null);
    const p0 = await dumpFramePng(page, outDir, 'frame-0');
    await settle(page, 400);
    const f1 = await captureGray(page, null);
    await settle(page, 400);
    const f2 = await captureGray(page, null);
    const p2 = await dumpFramePng(page, outDir, 'frame-2');

    const d01 = diffFrames(f0, f1);
    const d02 = diffFrames(f0, f2);
    const metrics = { d01: rd(d01), d02: rd(d02) };
    console.log('[effects] time-invariance', JSON.stringify(metrics));

    const failures: string[] = [];
    if (d01.changedFrac > 0.005 || d02.changedFrac > 0.005) {
      failures.push(
        `static-camera frames differ (changedFrac ${d01.changedFrac.toFixed(4)}, ${d02.changedFrac.toFixed(4)}) — time-animated shader cheat?`
      );
    }
    await writeMeta('time-invariance', outDir, [p0, p2], metrics, failures);
    expect(failures, failures.join('; ')).toHaveLength(0);
  });

  test('@effects moire fringes flow under camera orbit', async ({ page }) => {
    const outDir = path.join(OUT_ROOT, 'moire-fringe-flow');
    await fs.mkdir(outDir, { recursive: true });
    // Post-merge every composed plate runs the two-plane foliage_moire
    // recipe (id 3): front foliage carrier on the outer plane, uniform back
    // carrier on the REAL inner plane at the paraxial T/n gap.
    //
    // Asserted for ALL SIX faces, on BOTH plane materials — the front face
    // alone used to stand in for the box, so five walls could have run the
    // banned single-plane path (or stayed on the 1x1 blank) unnoticed. And
    // because uRecipe's material-creation default IS foliage_moire, the id on
    // its own proves nothing: each face must also carry the masks its OWN
    // manifest declares (outer plane <- files.front_png, inner <-
    // files.back_png), which is the suite's native texture-driven gate, and
    // BoxScene's recipe refusal (face_recipe_unsupported -> planes hidden)
    // must not have fired for any face.
    await waitForAllFaceMasks(page);
    const faceStates = await allFaceRenderState(page);
    const bindFailures: string[] = [];
    expect(faceStates).toHaveLength(FACE_IDS.length);
    for (const st of faceStates) {
      if (st.manifestRecipe !== 'foliage_moire') {
        bindFailures.push(`${st.face}: manifest render_recipe ${st.manifestRecipe}`);
      }
      if (st.recipe !== 3 || st.recipeBack !== 3) {
        bindFailures.push(`${st.face}: uRecipe ${st.recipe}/${st.recipeBack} (want 3/3)`);
      }
      if (!st.visible || !st.visibleBack) {
        bindFailures.push(
          `${st.face}: plane hidden (outer ${st.visible}, inner ${st.visibleBack})`
        );
      }
      if (st.maskW <= 1 || st.maskBackW <= 1) {
        bindFailures.push(
          `${st.face}: placeholder mask still bound (${st.maskW}px / ${st.maskBackW}px)`
        );
      }
      if (!st.maskMatchesManifest || !st.maskBackMatchesManifest) {
        bindFailures.push(
          `${st.face}: bound mask URLs are not the manifest's front/back PNGs ` +
            `(outer ${st.maskMatchesManifest}, inner ${st.maskBackMatchesManifest})`
        );
      }
      const tUm = st.thicknessUm;
      const nSub = st.n;
      if (tUm === null || nSub === null) {
        bindFailures.push(`${st.face}: manifest carries no substrate`);
      } else if (Math.abs(st.uThicknessUm - tUm) > 1e-6 || Math.abs(st.uN - nSub) > 1e-9) {
        bindFailures.push(
          `${st.face}: substrate not bound (uThicknessUm ${st.uThicknessUm} vs ` +
            `${st.thicknessUm}, uN ${st.uN} vs ${st.n})`
        );
      }
    }
    const unsupported = (await readLog(page)).filter((e) => e.type === 'face_recipe_unsupported');
    if (unsupported.length > 0) {
      bindFailures.push(
        `face_recipe_unsupported logged: ${JSON.stringify(unsupported.map((e) => e.face))}`
      );
    }
    expect(bindFailures, bindFailures.join('; ')).toHaveLength(0);

    const EL = 4;
    const AZ = [-10, -5, 0, 5, 10];
    await faceFrontOn(page, 0, EL);
    const roi = await frontROI(page);

    const frames = await azimuthSweep(page, AZ, EL, roi);
    const pngs: string[] = [];
    for (let i = 0; i < AZ.length; i++) {
      await setCameraAzEl(page, AZ[i], EL);
      await settle(page, 200);
      pngs.push(await dumpFramePng(page, outDir, `az${String(i).padStart(2, '0')}_${AZ[i]}`));
    }

    const consecutive = frames.slice(1).map((f, i) => diffFrames(frames[i], f));
    const endToEnd = diffFrames(frames[0], frames[frames.length - 1]);
    const nearPair = diffFrames(frames[0], frames[1]);
    const metrics = {
      consecutive: consecutive.map(rd),
      endToEnd: rd(endToEnd),
      faces: faceStates.map((s) => ({
        face: s.face,
        recipe: s.recipe,
        recipeBack: s.recipeBack,
        maskPx: [s.maskW, s.maskBackW],
        maskFromManifest: s.maskMatchesManifest && s.maskBackMatchesManifest,
      })),
    };
    console.log('[effects] moire-fringe-flow', JSON.stringify(metrics));

    const failures: string[] = [];
    // (a) fringes actually move between every 5-degree stop
    consecutive.forEach((d, i) => {
      if (d.changedFrac < 0.02) {
        failures.push(
          `fringes frozen between az ${AZ[i]} and ${AZ[i + 1]} (changedFrac ${d.changedFrac.toFixed(4)})`
        );
      }
    });
    // (b) flow is progressive: the 20-degree pair decorrelates at least as
    // much as the 5-degree pair (allowing beat-pattern periodicity slack).
    if (endToEnd.mad < nearPair.mad * 0.8) {
      failures.push(
        `no progressive flow: end-to-end mad ${endToEnd.mad.toFixed(2)} < near-pair mad ${nearPair.mad.toFixed(2)}`
      );
    }
    await writeMeta('moire-fringe-flow', outDir, pngs, metrics, failures);
    expect(failures, failures.join('; ')).toHaveLength(0);
  });

  test('@effects moire parallax obeys substrate physics (geometric gap)', async ({ page }) => {
    // The merged renderer derives parallax from GEOMETRY: the back gold
    // layer lives on a real inner plane at the paraxial air gap T/n below
    // the outer plane, and fringe motion emerges from perspective across
    // that gap (the legacy uThicknessUm uniform is dead on the foliage
    // path — poking it proves nothing). So the substrate test manipulates
    // the actual gap at a FIXED oblique camera: the gap must EQUAL the
    // manifest's T/n (the absolute contract, below); collapsing it toward
    // registration must move the fringes; a partial collapse must move them
    // LESS; and re-rendering at the same gap must be pixel-identical
    // (determinism control, run at both the design and the collapsed gap).
    const outDir = path.join(OUT_ROOT, 'moire-parallax-physics');
    await fs.mkdir(outDir, { recursive: true });

    await faceFrontOn(page, 10, 6); // oblique: real in-plane view component
    // The shader no longer magnifies the centerpiece pitch (honesty fix), so
    // the 60 um comb is sub-pixel at the default view and its fringe response
    // to the gap averages away. Zoom (angle-preserving) until it resolves.
    const unzoom = await zoomForMicroPatterns(page);
    await waitForStableFrame(page, 600);
    const roi = await frontROI(page);
    const capture = async () => await captureGray(page, roi);

    // Baseline at the design gap (also stashes the design gap for restore).
    const designGapMm = await scaleBackPlaneGap(page, 'front', 1);
    const fNatural = await capture();
    const pngNat = await dumpFramePng(page, outDir, 'oblique-design-gap');

    // --- the ONE quantitative renderer contract ------------------------------
    // Every other honesty check in this suite is RELATIVE (collapsing the gap
    // changes the frame; a partial collapse changes it less; the same gap is
    // deterministic) and passes for ANY monotonic gap — including the
    // pre-paraxial `outerZ - T` bug or a hardcoded constant. The absolute value
    // is what puts every switch/reveal/scanimation crossing at its true tilt
    // angle, so pin it: the inner plane sits exactly T/n below the outer plane
    // (BoxScene: `inner.position.z = outerZ - T / nGlass`, scene units are mm,
    // T = mm(thickness_um)), with n read from THIS face's manifest substrate —
    // the same numbers bound into uThicknessUm/uN. Mirror BoxScene's n<=1 guard
    // so a degenerate manifest is compared against the same fallback the
    // renderer used, and fail if the manifest is missing entirely (the store
    // must hold a box manifest by now — waitForBoxTextures ran in beforeEach).
    const sub = await faceSubstrate(page, 'front');
    expect(sub, 'box manifest has no substrate for the front face').not.toBeNull();
    const nEff = sub!.n > 1.0 ? sub!.n : 1.46;
    const expectedGapMm = sub!.thicknessUm / nEff / 1000;

    // Determinism control: same gap, recapture — must be identical.
    const fAgain = await capture();
    const dControl = diffFrames(fNatural, fAgain);

    // Gap -> near registration: the layers register and all cross-layer
    // parallax collapses. GAP_COLLAPSE_FACTOR is a small NON-ZERO floor on
    // purpose — at exactly 0 the two opaque plane meshes become coplanar AND
    // the inner plane leaves the glass slab, so the delta would be dominated by
    // depth fighting and backdrop change rather than by layer registration
    // (see the constant's docstring for the arithmetic). The PNG keeps its
    // historical `gap0` name so the frame sequences stay comparable.
    await scaleBackPlaneGap(page, 'front', GAP_COLLAPSE_FACTOR);
    const fZeroGap = await capture();
    const pngZero = await dumpFramePng(page, outDir, 'oblique-gap0');
    // Determinism control AT THE COLLAPSED STATE too: the collapse frame is the
    // one every anti-cheat threshold below leans on, so any nondeterminism
    // introduced there (MSAA/depth fighting near-registration) must be caught
    // rather than assumed away from the design-gap control.
    const fZeroAgain = await capture();
    const dControlZero = diffFrames(fZeroGap, fZeroAgain);

    // Gap -> 80%: response must be smaller than the full collapse. The probe
    // sits at 80% (20% displacement), NOT deeper, because the pixel response
    // saturates once the fringe shift exceeds its correlation length: the
    // measured curve on the linear-light renderer is mad 2.7 @ 0.9, 4.8 @ 0.8,
    // 5.9 @ 0.7, 6.6 @ 0.6, then a flat shoulder to 7.2 @ 0.12. The old 0.6
    // probe sat ON that shoulder, passing the <0.9x gate by 0.012 mad on the
    // blurrier pre-linear pipeline and failing on any contrast improvement.
    // At 0.8 the ratio is ~0.66 with the gate unchanged — and the anti-cheat
    // is STRONGER: a binary fake (any nonzero collapse -> same frame) still
    // reads ~1.0 here and fails.
    await scaleBackPlaneGap(page, 'front', 0.8);
    const fPartial = await capture();
    await scaleBackPlaneGap(page, 'front', 1); // restore design gap
    await unzoom();

    const dCollapse = diffFrames(fNatural, fZeroGap);
    const dPartial = diffFrames(fNatural, fPartial);

    const metrics = {
      gapCollapse: rd(dCollapse),
      partialCollapse: rd(dPartial),
      sameGapControl: rd(dControl),
      collapsedGapControl: rd(dControlZero),
      collapseFactor: GAP_COLLAPSE_FACTOR,
      designGapMm: Number(designGapMm.toFixed(6)),
      expectedGapMm: Number(expectedGapMm.toFixed(6)),
      substrate: { thickness_um: sub!.thicknessUm, n: sub!.n },
    };
    console.log('[effects] moire-parallax-physics', JSON.stringify(metrics));

    const failures: string[] = [];
    // (0) the absolute gap IS the paraxial air gap T/n. 1e-4 mm = 0.1 um.
    if (!(Math.abs(designGapMm - expectedGapMm) < 1e-4)) {
      failures.push(
        `inner-plane gap ${designGapMm.toFixed(6)} mm != T/n ` +
          `${expectedGapMm.toFixed(6)} mm (T=${sub!.thicknessUm} um, n=${nEff}) — ` +
          `every switch/reveal tilt angle is wrong by that ratio`
      );
    }
    if (dCollapse.changedFrac < 0.03) {
      failures.push(
        `fringes ignore the two-plane gap (collapse changedFrac ${dCollapse.changedFrac.toFixed(4)}) — parallax not geometric?`
      );
    }
    if (!(dPartial.mad < dCollapse.mad * 0.9 && dPartial.mad > 0.02)) {
      failures.push(
        `response does not scale with the gap (80%-gap mad ${dPartial.mad.toFixed(2)} vs full collapse ${dCollapse.mad.toFixed(2)})`
      );
    }
    if (dControl.mad > 0.5) {
      failures.push(
        `same-gap recapture differs (mad ${dControl.mad.toFixed(2)}) — nondeterministic rendering`
      );
    }
    if (dControlZero.mad > 0.5) {
      failures.push(
        `collapsed-gap recapture differs (mad ${dControlZero.mad.toFixed(2)}) — ` +
          `nondeterministic rendering at the state the collapse metric is measured from`
      );
    }
    await writeMeta(
      'moire-parallax-physics',
      outDir,
      [pngNat, pngZero],
      metrics,
      failures
    );
    expect(failures, failures.join('; ')).toHaveLength(0);
  });

  test('@effects stereo lenticular flips views across the slit axis', async ({ page }) => {
    test.setTimeout(150_000);
    const outDir = path.join(OUT_ROOT, 'stereo-lenticular-flip');
    await fs.mkdir(outDir, { recursive: true });

    // Post-merge the plate always binds as foliage_moire (id 3); the
    // barrier switch runs procedurally in its centerpiece region, driven by
    // the pattern's recipe_data (slit period/axis) and the two-plane gap.
    await assignFacePattern(page, 'front', TEST_PATTERNS.stereo, 'foliage_moire');
    await settle(page, 400);
    expect(await faceRecipeId(page, 'front')).toBe(3);

    // Texture-driven axiom under foliage_moire: the plate binds the REAL
    // front/back litho masks per plane (stereo view textures only bind for
    // the legacy recipe-0 path, so stereo_views is informational now).
    const stereoViews = await page.evaluate(() => {
      const buf = ((window as unknown as { __log?: any[] }).__log ?? []) as any[];
      const ev = buf
        .filter((e) => e.type === 'face_texture_bound' && e.face === 'front')
        .pop();
      return ev?.stereo_views ?? false;
    });

    // Tilt across the slit axis. Under foliage_moire the recipe-0
    // uSlitOrientation uniform is never bound for composed plates (it sits at
    // its material-creation default), so read the axis from the manifest the
    // pattern actually shipped — the same source plates.py drives the
    // composed-plate barrier from — mirroring how the carrier-reveal test
    // reads its period. Tangent +X maps to camera azimuth for the front face.
    const axisResp = await page.request.get(`/patterns/${TEST_PATTERNS.stereo}/default`);
    expect(axisResp.ok(), `GET /patterns/${TEST_PATTERNS.stereo}/default failed`).toBeTruthy();
    const axisRd = ((await axisResp.json()) as { recipe_data?: Record<string, unknown> })
      .recipe_data ?? {};
    const slitAxisDeg =
      typeof axisRd.slit_axis_deg === 'number'
        ? (axisRd.slit_axis_deg as number)
        : typeof axisRd.switch_axis_deg === 'number'
          ? (axisRd.switch_axis_deg as number)
          : 0;
    const slitRad = (slitAxisDeg * Math.PI) / 180;
    const alongAzimuth = Math.abs(Math.cos(slitRad)) >= 0.5;
    const TILT = 14;
    const view = async (t: number) => {
      if (alongAzimuth) await setCameraAzEl(page, t, 4);
      else await setCameraAzEl(page, 0, 4 + t);
      await waitForStableFrame(page, 350);
    };

    await view(0);
    const roi = await frontROI(page);
    await view(-TILT);
    const fA = await captureGray(page, roi);
    const pA = await dumpFramePng(page, outDir, 'tilt-neg');
    await view(TILT);
    const fB = await captureGray(page, roi);
    const pB = await dumpFramePng(page, outDir, 'tilt-pos');
    await view(0);
    const fC = await captureGray(page, roi);
    const pC = await dumpFramePng(page, outDir, 'head-on');

    const dAB = diffFrames(fA, fB);
    const dAC = diffFrames(fA, fC);
    const dBC = diffFrames(fB, fC);
    const metrics = { AB: rd(dAB), AC: rd(dAC), BC: rd(dBC), stereoViews, slitRad };
    console.log('[effects] stereo-lenticular-flip', JSON.stringify(metrics));

    const failures: string[] = [];
    if (dAB.changedFrac < 0.06) {
      failures.push(`opposite tilts render the same (AB changedFrac ${dAB.changedFrac.toFixed(4)})`);
    }
    // NOTE deliberately no "head-on is intermediate" pixel check: with the
    // audited p/4 switch half-width, a close perspective camera splits the
    // head-on plate into left/right A|B viewing zones (real barrier
    // behavior), so head-on is a spatial mix, not a uniform blend, and both
    // mad- and corr-based intermediacy assertions are invalid. First-zone
    // switch quality is quantified headlessly in
    // backend/tests/test_pattern_types.py::switch_metrics instead; the
    // head-on capture stays in the sequence for the vision grader.
    await writeMeta('stereo-lenticular-flip', outDir, [pA, pC, pB], metrics, failures);
    expect(failures, failures.join('; ')).toHaveLength(0);
  });

  test('@effects carrier reveal appears at the half-period tilt and is substrate-driven', async ({
    page,
  }) => {
    // This slot used to exercise phase_shift_overlay (recipe 2). That recipe
    // was retired: its two-image front/back phase split can never switch
    // under honest parallax (the front layer does not move with tilt), and
    // its on-screen flip was an explicit view-sign bias. The honest
    // replacement is the T5 carrier reveal — figure halftone AND carrier in
    // FRONT, uniform image-free carrier in BACK — rendered by plain
    // moire_interactive mask sampling. Same four-axiom structure as before:
    // view-dependent (contrast appears with tilt), time-invariant (covered
    // by the time-invariance scenario), texture-driven (real masks bound),
    // parameterized (the reveal tilt is DERIVED from the manifest's carrier
    // period + substrate via theta(p/2) = asin(n*sin(atan(p/(2t))))).
    test.setTimeout(150_000);
    const outDir = path.join(OUT_ROOT, 'carrier-reveal-tilt');
    await fs.mkdir(outDir, { recursive: true });

    // --- TEXTURE-DRIVEN AXIOM (axiom 3), natively asserted -------------------
    // Every other metric in this suite is a delta under a CAMERA or GEOMETRY
    // change, which any procedural view-keyed shader also produces. This one
    // holds camera and geometry fixed and changes only the BOUND MASK: capture
    // the head-on ROI with the default front pattern (globe-duo-phase), assign
    // a visually unrelated slug (the monogram carrier reveal), and capture the
    // SAME ROI at the SAME camera. Nothing but the mask content differs, so a
    // shader drawing procedural fringes instead of sampling the litho masks
    // renders the two identically and fails here.
    const slugBefore = await page.evaluate(
      () => (window as any).__studio.store.getState().boxSpec.faces.front.pattern_slug as string
    );
    expect(
      slugBefore,
      'texture-driven check needs the front face to start on a DIFFERENT slug'
    ).not.toBe(TEST_PATTERNS.reveal);
    await faceFrontOn(page, 0, 4); // exactly the camera view(0) uses below
    // Computed once and reused for every capture in this test: the plate
    // geometry does not move when only the pattern slug changes, and identical
    // ROIs are what makes the frames diffable.
    const roi = await frontROI(page);
    const fSlugBefore = await captureGray(page, roi);
    const pSlugBefore = await dumpFramePng(page, outDir, 'mask-before-assign');

    // Post-merge the plate binds as foliage_moire (id 3): the carrier-reveal
    // masks ride the two real planes (front mask on the outer plane, back
    // anti-phase carrier on the inner plane at the T/n gap).
    await assignFacePattern(page, 'front', TEST_PATTERNS.reveal, 'foliage_moire');
    await settle(page, 400);
    expect(await faceRecipeId(page, 'front')).toBe(3);
    // Structural half of the same axiom: the newly bound masks are the PNGs
    // this face's manifest declares, on both planes, at real resolution.
    const frontState = (await allFaceRenderState(page)).find((s) => s.face === 'front');
    expect(frontState, 'no render state for the front face').toBeTruthy();
    expect(
      frontState!.maskW > 1 && frontState!.maskBackW > 1,
      `front planes still on the placeholder mask (${frontState!.maskW}/${frontState!.maskBackW} px)`
    ).toBeTruthy();
    expect(
      frontState!.maskMatchesManifest && frontState!.maskBackMatchesManifest,
      'front planes are not bound to the manifest front/back PNGs'
    ).toBeTruthy();

    // First-zone calibration: the reveal completes when the Snell-refracted
    // back shift equals HALF the carrier period, and zones repeat every full
    // period of shift — a hardcoded ±14° (84 um ≈ 1-4 periods) can land
    // right back at registration where the figure vanishes. So read p and
    // the substrate from the manifest the pattern actually shipped.
    const mResp = await page.request.get(`/patterns/${TEST_PATTERNS.reveal}/default`);
    expect(mResp.ok(), `GET /patterns/${TEST_PATTERNS.reveal}/default failed`).toBeTruthy();
    const manifest = (await mResp.json()) as {
      recipe_data?: Record<string, unknown>;
      substrate?: { thickness_um?: number; n?: number };
    };
    const rdata = manifest.recipe_data ?? {};
    const periodUm =
      typeof rdata.carrier_period_um === 'number' ? (rdata.carrier_period_um as number) : 40;
    const tUm = manifest.substrate?.thickness_um ?? 500;
    const nSub = manifest.substrate?.n ?? 1.46;
    const tiltDeg =
      (Math.asin(Math.min(1, nSub * Math.sin(Math.atan(periodUm / (2 * tUm))))) * 180) /
      Math.PI; // 3.35 deg at p=40, t=500, n=1.46

    // Carrier stripes are vertical (switch axis +x) in every carrier/barrier
    // generator in this repo, so the reveal tilt is a camera-azimuth move.
    const view = async (t: number) => {
      await setCameraAzEl(page, t, 4);
      await waitForStableFrame(page, 350);
    };

    await view(0);
    const f0 = await captureGray(page, roi);
    const p0 = await dumpFramePng(page, outDir, 'head-on');
    // Same camera, same gap, same ROI as fSlugBefore — only the mask changed.
    const dMaskSwap = diffFrames(fSlugBefore, f0);
    await view(-tiltDeg);
    const fNeg = await captureGray(page, roi);
    const pNeg = await dumpFramePng(page, outDir, 'tilt-neg');
    await view(tiltDeg);
    const fPos = await captureGray(page, roi);
    const pPos = await dumpFramePng(page, outDir, 'tilt-pos');

    const d0P = diffFrames(f0, fPos);
    const d0N = diffFrames(f0, fNeg);
    const dPN = diffFrames(fPos, fNeg);
    const metrics = {
      headOnVsPos: rd(d0P),
      headOnVsNeg: rd(d0N),
      posVsNeg: rd(dPN),
      maskSwap: rd(dMaskSwap),
      maskSwapSlugs: [slugBefore, TEST_PATTERNS.reveal],
      // Kept out of `frames` on purpose: it shows a DIFFERENT pattern, so the
      // vision grader must not read it as part of the tilt sequence.
      maskSwapFrame: path.basename(pSlugBefore),
      periodUm,
      tiltDeg: Number(tiltDeg.toFixed(3)),
    };
    console.log('[effects] carrier-reveal-tilt', JSON.stringify(metrics));

    const failures: string[] = [];
    // (0) Texture-driven: swapping ONLY the bound mask (camera + gap fixed)
    // must change the plate pixels. Thresholds sit far above the suite's
    // same-state noise floor (the parallax scenario pins identical-state
    // recapture at mad <= 0.5) and far below what two unrelated centerpieces
    // produce, so this fails on mask-independent rendering without being a
    // sensitivity knob.
    if (dMaskSwap.changedFrac < 0.03 || dMaskSwap.mad < 1.5) {
      failures.push(
        `plate pixels do not follow the bound mask: swapping ${slugBefore} -> ` +
          `${TEST_PATTERNS.reveal} at a fixed camera/gap moved almost nothing ` +
          `(changedFrac ${dMaskSwap.changedFrac.toFixed(4)}, mad ${dMaskSwap.mad.toFixed(2)}) — ` +
          `procedural, mask-independent shading?`
      );
    }
    // (a) The reveal happens: at ±theta(p/2) the figure contrast appears, so
    // both tilted frames must differ substantially from head-on.
    if (d0P.changedFrac < 0.03 || d0N.changedFrac < 0.03) {
      failures.push(
        `no contrast change at the half-period tilt ±${tiltDeg.toFixed(2)}° ` +
          `(changedFrac +${d0P.changedFrac.toFixed(4)} / -${d0N.changedFrac.toFixed(4)}) — parallax not applied?`
      );
    }
    // (b) Substrate anti-cheat: with the camera parked at +theta(p/2),
    // collapsing the geometric two-plane gap registers the layers, so the
    // revealed figure must snap back toward registration. A view-sign bias
    // (the retired recipe-2 cheat) would ignore the gap and sail through
    // unchanged. GAP_COLLAPSE_FACTOR (not 0) keeps the two plane meshes from
    // becoming coplanar — see its docstring.
    // Zoomed pair for the collapse metric: at the default view the true-pitch
    // carrier is sub-pixel (the shader no longer magnifies it), so the gap
    // response averages away. Zoom preserves the parked tilt angle; the
    // comparison frames are BOTH taken zoomed so they stay comparable.
    const unzoomReveal = await zoomForMicroPatterns(page);
    await waitForStableFrame(page, 400);
    const roiZoom = await frontROI(page);
    const fPosZoom = await captureGray(page, roiZoom);
    await scaleBackPlaneGap(page, 'front', GAP_COLLAPSE_FACTOR);
    await settle(page, 150);
    const fZeroGap = await captureGray(page, roiZoom);
    await scaleBackPlaneGap(page, 'front', 1);
    await unzoomReveal();
    const dGap = diffFrames(fPosZoom, fZeroGap);
    (metrics as Record<string, unknown>).gapCollapse = rd(dGap);
    console.log('[effects] carrier-reveal gapCollapse', JSON.stringify(rd(dGap)));
    if (dGap.changedFrac < 0.03) {
      failures.push(
        `reveal ignores the two-plane gap (collapse changedFrac ${dGap.changedFrac.toFixed(4)}) — not parallax-driven`
      );
    }
    await writeMeta('carrier-reveal-tilt', outDir, [pNeg, p0, pPos], metrics, failures);
    expect(failures, failures.join('; ')).toHaveLength(0);
  });

  test('@effects lid transition is a monotonic hinge rotation and round-trips', async ({
    page,
  }) => {
    test.setTimeout(120_000);
    const outDir = path.join(OUT_ROOT, 'lid-transition');
    await fs.mkdir(outDir, { recursive: true });
    await faceFrontOn(page, 30, 30);
    await waitForStableFrame(page, 800);

    const closed0 = await captureGray(page, null);
    const pngs: string[] = [await dumpFramePng(page, outDir, 'closed-initial')];

    // Open to 110 and sample the damped animation.
    await page.evaluate(() => (window as any).__studio.setLid(110));
    const rotSamples: number[] = [];
    const motionDiffs: DiffMetrics[] = [];
    let prev: GrayFrame | null = null;
    for (let i = 0; i < 6; i++) {
      await page.waitForTimeout(130);
      const rot = await lidRotationDeg(page);
      expect(rot, 'lidPivot missing during animation').not.toBeNull();
      rotSamples.push(rot!);
      const f = await captureGray(page, null);
      if (prev) motionDiffs.push(diffFrames(prev, f));
      prev = f;
      if (i === 2) pngs.push(await dumpFramePng(page, outDir, 'opening-mid'));
    }
    // Settle fully open.
    await page.waitForFunction(
      () => Math.abs((window as any).__studio.getLidDeg() - 110) < 0.5,
      null,
      { timeout: 10_000 }
    );
    await settle(page, 500);
    const openRot = await lidRotationDeg(page);
    pngs.push(await dumpFramePng(page, outDir, 'open-110'));

    // Close and verify the exact initial state returns.
    await page.evaluate(() => (window as any).__studio.setLid(0));
    await page.waitForFunction(() => (window as any).__studio.getLidDeg() < 0.05, null, {
      timeout: 10_000,
    });
    // The round-trip bound (changedFrac <= 0.02) needs the hinge damping fully
    // decayed, so wait for measured quiescence rather than a flat 600 ms.
    await waitForStableFrame(page, 600);
    const closed1 = await captureGray(page, null);
    pngs.push(await dumpFramePng(page, outDir, 'closed-final'));
    const roundTrip = diffFrames(closed0, closed1);

    const metrics = {
      rotSamples: rotSamples.map((r) => Number(r.toFixed(1))),
      openRot: Number((openRot ?? -1).toFixed(1)),
      motionDiffs: motionDiffs.map(rd),
      roundTrip: rd(roundTrip),
    };
    console.log('[effects] lid-transition', JSON.stringify(metrics));

    const failures: string[] = [];
    for (let i = 1; i < rotSamples.length; i++) {
      if (rotSamples[i] < rotSamples[i - 1] - 0.5) {
        failures.push(`lid rotation reversed mid-open: ${rotSamples.join(' -> ')}`);
        break;
      }
    }
    if (!(openRot !== null && Math.abs(openRot - 110) < 2)) {
      failures.push(`lid settled at ${openRot} deg, expected ~110`);
    }
    if (!motionDiffs.some((d) => d.changedFrac > 0.01)) {
      failures.push('no visible motion during the lid animation');
    }
    if (roundTrip.changedFrac > 0.02) {
      failures.push(
        `re-closed frame differs from initial closed frame (changedFrac ${roundTrip.changedFrac.toFixed(4)}) — state leak`
      );
    }
    await writeMeta('lid-transition', outDir, pngs, metrics, failures);
    expect(failures, failures.join('; ')).toHaveLength(0);
  });

  test('@effects illumination modes are distinct and laser is laser-colored', async ({
    page,
  }) => {
    const outDir = path.join(OUT_ROOT, 'illumination-modes');
    await fs.mkdir(outDir, { recursive: true });
    await faceFrontOn(page, 0, 4);
    const roi = await frontROI(page);

    const setIllum = async (mode: string, color?: string) => {
      await page.evaluate(
        ([m, c]) => {
          const st = (window as any).__studio.store.getState();
          st.setIllumination(m);
          if (c) st.setLaserColor(c);
        },
        [mode, color ?? null] as const
      );
      await settle(page, 300);
    };

    await setIllum('ambient');
    const fAmbient = await captureGray(page, roi);
    const pngs = [await dumpFramePng(page, outDir, 'ambient')];
    await setIllum('laser', 'green');
    const fLaserG = await captureGray(page, roi);
    pngs.push(await dumpFramePng(page, outDir, 'laser-green'));
    await setIllum('laser', 'red');
    const fLaserR = await captureGray(page, roi);
    pngs.push(await dumpFramePng(page, outDir, 'laser-red'));
    await setIllum('backlight');
    const fBack = await captureGray(page, roi);
    pngs.push(await dumpFramePng(page, outDir, 'backlight'));
    await setIllum('ambient');

    const dAL = diffFrames(fAmbient, fLaserG);
    const dAB = diffFrames(fAmbient, fBack);
    const dLB = diffFrames(fLaserG, fBack);
    const metrics = {
      ambientVsLaser: rd(dAL),
      ambientVsBacklight: rd(dAB),
      laserVsBacklight: rd(dLB),
      laserGreenMeanRgb: fLaserG.meanRgb.map((v) => Number(v.toFixed(1))),
      laserRedMeanRgb: fLaserR.meanRgb.map((v) => Number(v.toFixed(1))),
    };
    console.log('[effects] illumination-modes', JSON.stringify(metrics));

    const failures: string[] = [];
    // Distinctness is a compound signal: changedFrac alone sits at ~0.049 for
    // ambient-vs-backlight now that the shader renders true (non-magnified)
    // pitches — the modes still differ clearly (mad ~5), the per-pixel deltas
    // are just spread thinner. Either a broad change OR a strong mean delta
    // proves the modes are distinct; identical renders fail both.
    const same = (d: { changedFrac: number; mad: number }) =>
      d.changedFrac < 0.03 && d.mad < 2.0;
    if (same(dAL)) failures.push('ambient and laser render the same');
    if (same(dAB)) failures.push('ambient and backlight render the same');
    if (same(dLB)) failures.push('laser and backlight render the same');
    if (!(fLaserG.meanRgb[1] > fLaserG.meanRgb[0])) {
      failures.push(
        `green laser is not green-dominant (rgb ${fLaserG.meanRgb.map((v) => v.toFixed(1)).join(',')})`
      );
    }
    if (!(fLaserR.meanRgb[0] > fLaserR.meanRgb[1])) {
      failures.push(
        `red laser is not red-dominant (rgb ${fLaserR.meanRgb.map((v) => v.toFixed(1)).join(',')})`
      );
    }
    await writeMeta('illumination-modes', outDir, pngs, metrics, failures);
    expect(failures, failures.join('; ')).toHaveLength(0);
  });

  test('@effects turntable flows when on, freezes when off', async ({ page }) => {
    const outDir = path.join(OUT_ROOT, 'turntable-flow');
    await fs.mkdir(outDir, { recursive: true });
    await faceFrontOn(page, 20, 25);
    await waitForStableFrame(page, 800);

    await page.evaluate(() => (window as any).__studio.store.getState().setAutoRotate(true));
    await settle(page, 300);
    const f0 = await captureGray(page, null);
    const p0 = await dumpFramePng(page, outDir, 'rotating-0');
    await settle(page, 700);
    const f1 = await captureGray(page, null);
    const p1 = await dumpFramePng(page, outDir, 'rotating-1');
    const dOn = diffFrames(f0, f1);

    await page.evaluate(() => (window as any).__studio.store.getState().setAutoRotate(false));
    // The 'stopped' bound (changedFrac <= 0.005) is the flakiest in the suite:
    // OrbitControls' damping decay is frame-rate dependent, so a flat 900 ms on
    // a loaded host could leave residual sub-pixel drift. Wait for MEASURED
    // quiescence instead — it returns as soon as two consecutive frame pairs are
    // quiet (typically well under the old 900 ms) and only spends the larger cap
    // when the scene genuinely has not stopped yet.
    const decay = await waitForStableFrame(page, 1600);
    const g0 = await captureGray(page, null);
    await settle(page, 500);
    const g1 = await captureGray(page, null);
    const dOff = diffFrames(g0, g1);

    const metrics = {
      rotating: rd(dOn),
      stopped: rd(dOff),
      decay: { stable: decay.stable, waitedMs: decay.waitedMs },
    };
    console.log('[effects] turntable-flow', JSON.stringify(metrics));

    const failures: string[] = [];
    if (dOn.changedFrac < 0.01) failures.push('canvas static with auto-rotate enabled');
    if (dOff.changedFrac > 0.005) failures.push('motion continues after auto-rotate disabled');
    await writeMeta('turntable-flow', outDir, [p0, p1], metrics, failures);
    expect(failures, failures.join('; ')).toHaveLength(0);
  });
});
