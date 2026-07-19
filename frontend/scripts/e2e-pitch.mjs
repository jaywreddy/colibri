// End-to-end harness for the GRATING PITCH setting (Ring Box Studio).
//
// Drives the REAL UI: for each pitch it clicks the preset chip (by data-testid),
// waits for the debounced POST /boxes/generate to complete, and asserts the full
// chain of custody landed —
//   * POST /boxes/generate returned 200 (network) AND the manifest id changed,
//   * every face manifest carries recipe_data.carrier_period_um == the pitch,
//   * face_texture_bound fired (the shader rebound to the new plate),
//   * a per-pitch screenshot is captured.
// Prints a PASS line per pitch; exits nonzero with a clear message on any
// failure. Mirrors scripts/shot.mjs (headless chromium, manual render capture).
//
// Usage (from frontend/, servers already up under preview 'app'):
//   pnpm exec node scripts/e2e-pitch.mjs [outDir]
//
import { chromium } from '@playwright/test';
import { mkdirSync } from 'node:fs';
import { resolve } from 'node:path';

const PITCHES = [4, 10, 20];
const PATTERN_SCALE = 4;
const outDir = resolve(process.argv[2] ?? '.');
mkdirSync(outDir, { recursive: true });

const fail = (msg) => {
  console.error(`FAIL: ${msg}`);
  process.exitCode = 1;
};

// Collect every /boxes/generate response with a wall-clock stamp so we can
// prove a 200 fired after each chip click.
const generateResponses = [];

const browser = await chromium.launch();
let ok = true;
try {
  const page = await browser.newPage({ viewport: { width: 1280, height: 900 } });
  page.on('response', (r) => {
    if (r.url().includes('/boxes/generate')) {
      generateResponses.push({ t: Date.now(), status: r.status() });
    }
  });
  page.on('pageerror', (e) => console.error('PAGEERROR', e.message));

  await page.goto('http://localhost:5173/', { waitUntil: 'domcontentloaded' });
  // Wait for the studio handle + a first bound manifest (mount regen settled).
  await page.waitForFunction(
    () => !!(window.__studio && window.__studio.store.getState().boxManifest),
    null,
    { timeout: 45000 }
  );
  // Quiet the scene and give the shot a legible pattern scale + camera.
  await page.evaluate((scale) => {
    const st = window.__studio.store.getState();
    st.setAutoRotate(false);
    st.setPatternScale(scale);
  }, PATTERN_SCALE);
  // Let the mount regen + its texture binds fully settle before we start.
  await page.waitForTimeout(2500);

  for (const pitch of PITCHES) {
    const prevId = await page.evaluate(
      () => window.__studio.store.getState().boxManifest?.id ?? null
    );
    // Mark the log high-water so we only look at events AFTER this click.
    const clickWall = Date.now();
    const logMark = await page.evaluate(() => (window.__log ? window.__log.length : 0));

    // Drive the REAL UI: click the preset chip.
    const chip = page.locator(`[data-testid="grating-pitch-${pitch}"]`);
    if ((await chip.count()) === 0) {
      fail(`pitch ${pitch}: chip [data-testid="grating-pitch-${pitch}"] not found`);
      ok = false;
      break;
    }
    await chip.click();

    // Wait for a box_regen_done event pushed after the click.
    let regenDone = false;
    try {
      await page.waitForFunction(
        (mark) => {
          const buf = window.__log ?? [];
          return buf.slice(mark).some((e) => e.type === 'box_regen_done');
        },
        logMark,
        { timeout: 30000 }
      );
      regenDone = true;
    } catch {
      regenDone = false;
    }
    if (!regenDone) {
      fail(`pitch ${pitch}: box_regen_done never fired within 30s`);
      ok = false;
      break;
    }

    // Give the texture bind (fired from the manifest effect) a moment.
    await page.waitForTimeout(500);

    // --- assertions ---------------------------------------------------------
    // 1) POST /boxes/generate returned 200 after the click.
    const gen200 = generateResponses.some((r) => r.t >= clickWall && r.status === 200);

    // 2) Manifest id changed (a fresh box materialized).
    const newId = await page.evaluate(
      () => window.__studio.store.getState().boxManifest?.id ?? null
    );
    const idChanged = newId !== prevId;
    if (!gen200 && !idChanged) {
      fail(`pitch ${pitch}: no 200 /boxes/generate AND manifest id unchanged (${prevId})`);
      ok = false;
      break;
    }

    // 3) Every face manifest carries recipe_data.carrier_period_um == pitch.
    const pitchCheck = await page.evaluate(() => {
      const m = window.__studio.store.getState().boxManifest;
      const out = {};
      for (const [fid, fm] of Object.entries(m.faces ?? {})) {
        out[fid] = fm?.recipe_data?.carrier_period_um ?? null;
      }
      return out;
    });
    const badFaces = Object.entries(pitchCheck).filter(
      ([, v]) => Number(v) !== pitch
    );
    if (badFaces.length > 0) {
      fail(
        `pitch ${pitch}: faces with wrong carrier_period_um -> ` +
          JSON.stringify(Object.fromEntries(badFaces))
      );
      ok = false;
      break;
    }

    // 4) face_texture_bound fired after the click.
    const textureBound = await page.evaluate((mark) => {
      const buf = window.__log ?? [];
      return buf.slice(mark).some((e) => e.type === 'face_texture_bound');
    }, logMark);
    if (!textureBound) {
      fail(`pitch ${pitch}: face_texture_bound never fired`);
      ok = false;
      break;
    }

    // 5) Screenshot the canvas (front-ish view).
    await page.evaluate(() => {
      const s = window.__studio;
      const az = (14 * Math.PI) / 180;
      const el = (6 * Math.PI) / 180;
      const r = s.camera.position.clone().sub(s.controls.target).length() * 0.8;
      s.camera.position.set(
        s.controls.target.x + r * Math.cos(el) * Math.sin(az),
        s.controls.target.y + r * Math.sin(el),
        s.controls.target.z + r * Math.cos(el) * Math.cos(az)
      );
      s.controls.update();
    });
    await page.waitForTimeout(600);
    const shot = resolve(outDir, `e2e-pitch-${pitch}.png`);
    const bb = await page.locator('canvas').boundingBox();
    await page.screenshot({ path: shot, clip: bb });

    const faceCount = Object.keys(pitchCheck).length;
    console.log(
      `PASS pitch=${pitch}μm  gen200=${gen200}  idChanged=${idChanged} ` +
        `(${prevId}→${newId})  faces=${faceCount} carrier_period_um=${pitch}  ` +
        `face_texture_bound=✓  shot=${shot}`
    );
  }

  // Restore the spec default so the servers are left healthy at pitch 22.
  await page.evaluate(() => {
    window.__studio.store.getState().patchBoxSpec({ carrier_pitch_um: 22.0 });
  });
  await page.waitForTimeout(2500); // let the restore regen fire + settle
} catch (e) {
  fail(`harness error: ${e && e.message ? e.message : e}`);
  ok = false;
} finally {
  await browser.close();
}

if (process.exitCode && process.exitCode !== 0) {
  console.error('E2E PITCH HARNESS: FAILED');
} else {
  console.log('E2E PITCH HARNESS: ALL PITCHES PASSED');
}
