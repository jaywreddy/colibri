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
 * Run: `just test-effects` (Playwright, single worker — heavy process).
 */
import { test, expect } from './fixtures';
import { promises as fs } from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import { setCameraAzEl, waitForBoxTextures, waitForStudio } from './helpers';
import {
  assignFacePattern,
  captureGray,
  diffFrames,
  dumpFramePng,
  facePlateROI,
  faceRecipeId,
  lidRotationDeg,
  scaleBackPlaneGap,
  settle,
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
  await settle(page, 900);
}

/** Head-on view of the front face, slight elevation. */
async function faceFrontOn(page: Page, azDeg = 0, elDeg = 4): Promise<void> {
  await setCameraAzEl(page, azDeg, elDeg);
  await settle(page, 350);
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
    await settle(page, 250);
    frames.push(await captureGray(page, roi));
  }
  return frames;
}

test.describe('@effects physical honesty of renderer effects', () => {
  test.beforeEach(async ({ page }) => {
    await gotoStudio(page);
  });

  test('@effects time-invariance: static camera => static frame', async ({ page }) => {
    const outDir = path.join(OUT_ROOT, 'time-invariance');
    await fs.mkdir(outDir, { recursive: true });
    await faceFrontOn(page, 20, 25);
    await settle(page, 800); // damping fully decayed

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
    expect(await faceRecipeId(page, 'front')).toBe(3);

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
    // the actual gap at a FIXED oblique camera: collapsing it to zero must
    // register the layers and move the fringes; a partial collapse must
    // move them LESS; and re-rendering at the same gap must be
    // pixel-identical (determinism control).
    const outDir = path.join(OUT_ROOT, 'moire-parallax-physics');
    await fs.mkdir(outDir, { recursive: true });

    await faceFrontOn(page, 10, 6); // oblique: real in-plane view component
    await settle(page, 600);
    const roi = await frontROI(page);
    const capture = async () => await captureGray(page, roi);

    // Baseline at the design gap (also stashes the design gap for restore).
    const designGapMm = await scaleBackPlaneGap(page, 'front', 1);
    const fNatural = await capture();
    const pngNat = await dumpFramePng(page, outDir, 'oblique-design-gap');

    // Determinism control: same gap, recapture — must be identical.
    const fAgain = await capture();
    const dControl = diffFrames(fNatural, fAgain);

    // Gap -> 0: layers register, all cross-layer parallax collapses.
    await scaleBackPlaneGap(page, 'front', 0);
    const fZeroGap = await capture();
    const pngZero = await dumpFramePng(page, outDir, 'oblique-gap0');

    // Gap -> 60%: response must be smaller than the full collapse.
    await scaleBackPlaneGap(page, 'front', 0.6);
    const fPartial = await capture();
    await scaleBackPlaneGap(page, 'front', 1); // restore design gap

    const dCollapse = diffFrames(fNatural, fZeroGap);
    const dPartial = diffFrames(fNatural, fPartial);

    const metrics = {
      gapCollapse: rd(dCollapse),
      partialCollapse: rd(dPartial),
      sameGapControl: rd(dControl),
      designGapMm: Number(designGapMm.toFixed(4)),
    };
    console.log('[effects] moire-parallax-physics', JSON.stringify(metrics));

    const failures: string[] = [];
    if (dCollapse.changedFrac < 0.03) {
      failures.push(
        `fringes ignore the two-plane gap (collapse changedFrac ${dCollapse.changedFrac.toFixed(4)}) — parallax not geometric?`
      );
    }
    if (!(dPartial.mad < dCollapse.mad * 0.9 && dPartial.mad > 0.02)) {
      failures.push(
        `response does not scale with the gap (60% collapse mad ${dPartial.mad.toFixed(2)} vs full ${dCollapse.mad.toFixed(2)})`
      );
    }
    if (dControl.mad > 0.5) {
      failures.push(
        `same-gap recapture differs (mad ${dControl.mad.toFixed(2)}) — nondeterministic rendering`
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
      await settle(page, 300);
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

    // Post-merge the plate binds as foliage_moire (id 3): the carrier-reveal
    // masks ride the two real planes (front mask on the outer plane, back
    // anti-phase carrier on the inner plane at the T/n gap).
    await assignFacePattern(page, 'front', TEST_PATTERNS.reveal, 'foliage_moire');
    await settle(page, 400);
    expect(await faceRecipeId(page, 'front')).toBe(3);

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
      await settle(page, 300);
    };

    await view(0);
    const roi = await frontROI(page);
    const f0 = await captureGray(page, roi);
    const p0 = await dumpFramePng(page, outDir, 'head-on');
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
      periodUm,
      tiltDeg: Number(tiltDeg.toFixed(3)),
    };
    console.log('[effects] carrier-reveal-tilt', JSON.stringify(metrics));

    const failures: string[] = [];
    // (a) The reveal happens: at ±theta(p/2) the figure contrast appears, so
    // both tilted frames must differ substantially from head-on.
    if (d0P.changedFrac < 0.03 || d0N.changedFrac < 0.03) {
      failures.push(
        `no contrast change at the half-period tilt ±${tiltDeg.toFixed(2)}° ` +
          `(changedFrac +${d0P.changedFrac.toFixed(4)} / -${d0N.changedFrac.toFixed(4)}) — parallax not applied?`
      );
    }
    // (b) Substrate anti-cheat: with the camera parked at +theta(p/2),
    // collapsing the geometric two-plane gap to zero registers the layers,
    // so the revealed figure must snap back toward registration. A
    // view-sign bias (the retired recipe-2 cheat) would ignore the gap and
    // sail through unchanged.
    await scaleBackPlaneGap(page, 'front', 0);
    await settle(page, 150);
    const fZeroGap = await captureGray(page, roi);
    await scaleBackPlaneGap(page, 'front', 1);
    const dGap = diffFrames(fPos, fZeroGap);
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
    await settle(page, 800);

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
    await settle(page, 600);
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
    if (dAL.changedFrac < 0.05) failures.push('ambient and laser render the same');
    if (dAB.changedFrac < 0.05) failures.push('ambient and backlight render the same');
    if (dLB.changedFrac < 0.05) failures.push('laser and backlight render the same');
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
    await settle(page, 800);

    await page.evaluate(() => (window as any).__studio.store.getState().setAutoRotate(true));
    await settle(page, 300);
    const f0 = await captureGray(page, null);
    const p0 = await dumpFramePng(page, outDir, 'rotating-0');
    await settle(page, 700);
    const f1 = await captureGray(page, null);
    const p1 = await dumpFramePng(page, outDir, 'rotating-1');
    const dOn = diffFrames(f0, f1);

    await page.evaluate(() => (window as any).__studio.store.getState().setAutoRotate(false));
    await settle(page, 900); // let damping decay
    const g0 = await captureGray(page, null);
    await settle(page, 500);
    const g1 = await captureGray(page, null);
    const dOff = diffFrames(g0, g1);

    const metrics = { rotating: rd(dOn), stopped: rd(dOff) };
    console.log('[effects] turntable-flow', JSON.stringify(metrics));

    const failures: string[] = [];
    if (dOn.changedFrac < 0.01) failures.push('canvas static with auto-rotate enabled');
    if (dOff.changedFrac > 0.005) failures.push('motion continues after auto-rotate disabled');
    await writeMeta('turntable-flow', outDir, [p0, p1], metrics, failures);
    expect(failures, failures.join('; ')).toHaveLength(0);
  });
});
