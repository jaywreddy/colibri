// Headless screenshot of the running Ring Box Studio preview (localhost:5173).
// The MCP preview_screenshot tool waits for render-idle, which never happens on
// a continuously-animating WebGL canvas — this script captures immediately.
//
// Usage (from frontend/):
//   pnpm exec node scripts/shot.mjs out.png '{"finish":"gold","lid":40,"azimuthDeg":25,"elevationDeg":18,"zoom":1.0}'
//
// Patch keys (all optional):
//   finish       'bright'|'copper'|'gold'|'rose'|'patina'|'gunmetal'
//   lid          lid opening in degrees (0..~110)
//   layout       'assembled' | 'flat'
//   illumination 'ambient' | 'laser' | 'backlight'
//   azimuthDeg   camera azimuth around the box
//   elevationDeg camera elevation
//   zoom         camera distance multiplier (1 = current)
//   boxId        load a specific saved box (demo boxes) instead of the default
import { chromium } from '@playwright/test';

const out = process.argv[2] ?? 'shot.png';
const patch = JSON.parse(process.argv[3] ?? '{}');

const browser = await chromium.launch();
try {
  const page = await browser.newPage({ viewport: { width: 1280, height: 900 } });
  await page.goto('http://localhost:5173/', { waitUntil: 'domcontentloaded' });
  // Wait for the studio handle + a bound manifest (textures requested).
  await page.waitForFunction(
    () => !!(window.__studio && window.__studio.store.getState().boxManifest),
    null,
    { timeout: 45000 }
  );

  // Optional: load a specific saved box (demo boxes) instead of the default.
  // The app fires a debounced default-box regen on mount whose generateBox()
  // promise resolves LATE and rebinds the FRONT face to the DEFAULT plate even
  // though the store manifest is our demo box (an async texture-bind race). To
  // win it deterministically we: (1) let all mount-regens fully settle, (2) swap
  // in the demo box, then (3) POLL until the front face's uBack GPU texture
  // actually points at the demo plate hash. If a late regen clobbers it during
  // the poll we re-assert and keep polling — so the shot never races the bind.
  if (patch.boxId) {
    await page.waitForTimeout(2500); // let mount-regen + its texture binds finish
    const wantHash = await page.evaluate(async (id) => {
      const r = await fetch('/boxes/' + id);
      const m = await r.json();
      return m.faces.front.files.back_png.split('/plates/')[1].split('/')[0];
    }, patch.boxId);
    let bound = false;
    for (let i = 0; i < 30 && !bound; i++) {
      await page.evaluate(async (id) => {
        const s = window.__studio;
        s.store.getState().setAutoRotate(false);
        if (s.store.getState().boxManifest?.id !== id) {
          const r = await fetch('/boxes/' + id);
          if (r.ok) s.store.getState().setBoxManifest(await r.json());
        }
      }, patch.boxId);
      await page.waitForTimeout(400);
      bound = await page.evaluate((want) => {
        const t = window.__studio.faces.front.shader.uniforms.uBack.value;
        return !!(t && t.image && t.image.currentSrc && t.image.currentSrc.includes('/plates/' + want + '/'));
      }, wantHash);
    }
    if (!bound) console.error('WARNING: front face never bound to demo plate ' + wantHash);
  }

  await page.evaluate(async (p) => {
    const s = window.__studio;
    const st = s.store.getState();
    st.setAutoRotate(false);
    if (p.finish) st.patchFoil({ finish: p.finish });
    if (p.patternScale !== undefined) st.setPatternScale(p.patternScale);
    if (p.layout) st.setLayout(p.layout);
    if (p.illumination) st.setIllumination(p.illumination);
    if (p.lid !== undefined) st.setLidTargetDeg(p.lid);
    if (p.azimuthDeg !== undefined || p.elevationDeg !== undefined || p.zoom !== undefined) {
      const az = ((p.azimuthDeg ?? 20) * Math.PI) / 180;
      const el = ((p.elevationDeg ?? 18) * Math.PI) / 180;
      const cam = s.camera;
      const tgt = s.controls.target;
      const r = cam.position.clone().sub(tgt).length() * (p.zoom ?? 1);
      cam.position.set(
        tgt.x + r * Math.cos(el) * Math.sin(az),
        tgt.y + r * Math.sin(el),
        tgt.z + r * Math.cos(el) * Math.cos(az)
      );
      s.controls.update();
    }
  }, patch);
  // Let textures finish loading + lid/damping animation settle.
  await page.waitForTimeout(3000);
  // Final guard for demo boxes: a mount-regen texture bind may have resolved in
  // the settle window and clobbered the front face back to the default plate.
  // Re-assert + re-poll one last time so the shot is on the demo plate.
  if (patch.boxId) {
    const wantHash = await page.evaluate(async (id) => {
      const r = await fetch('/boxes/' + id);
      const m = await r.json();
      return m.faces.front.files.back_png.split('/plates/')[1].split('/')[0];
    }, patch.boxId);
    let bound = false;
    for (let i = 0; i < 20 && !bound; i++) {
      await page.evaluate(async (id) => {
        const s = window.__studio;
        if (s.store.getState().boxManifest?.id !== id) {
          const r = await fetch('/boxes/' + id);
          if (r.ok) s.store.getState().setBoxManifest(await r.json());
        }
      }, patch.boxId);
      await page.waitForTimeout(350);
      bound = await page.evaluate((want) => {
        const t = window.__studio.faces.front.shader.uniforms.uBack.value;
        return !!(t && t.image && t.image.currentSrc && t.image.currentSrc.includes('/plates/' + want + '/'));
      }, wantHash);
    }
    if (!bound) console.error('WARNING: front never bound to demo plate ' + wantHash);
    else console.error('OK: front bound to demo plate ' + wantHash);
    await page.waitForTimeout(600); // let the confirmed texture paint one frame
  }
  // Clip-screenshot the canvas region without Playwright's element-stability
  // wait: progressive thumbnail loads in the side panels keep reflowing the
  // canvas bbox by a pixel or two, so locator.screenshot() times out even
  // though the scene renders fine.
  const bb = await page.locator('canvas').boundingBox();
  await page.screenshot({ path: out, clip: bb });
  console.log(`saved ${out}`);
} finally {
  await browser.close();
}
