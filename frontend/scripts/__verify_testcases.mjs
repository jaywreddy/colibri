// Verification test cases for round 8 — ablation / thickness / physics / param flip.
// Run from frontend/: pnpm exec node <this file> <scratchpadDir>
import { chromium } from '@playwright/test';

const S = process.argv[2];
const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 1280, height: 900 } });

async function fresh() {
  await page.goto('http://localhost:5173/', { waitUntil: 'domcontentloaded' });
  await page.waitForFunction(
    () => !!(window.__studio && window.__studio.store.getState().boxManifest),
    null, { timeout: 45000 }
  );
  // wait for face textures to bind + regen settle
  await page.waitForTimeout(4000);
}

async function cam(az, el, zoom) {
  await page.evaluate(([a, e, z]) => {
    const s = window.__studio;
    const st = s.store.getState();
    st.setAutoRotate(false);
    st.patchFoil({ finish: 'bright' });
    st.setLidTargetDeg(0);
    const azr = (a * Math.PI) / 180, elr = (e * Math.PI) / 180;
    const tgt = s.controls.target;
    const r = s.camera.position.clone().sub(tgt).length() * z;
    s.camera.position.set(
      tgt.x + r * Math.cos(elr) * Math.sin(azr),
      tgt.y + r * Math.sin(elr),
      tgt.z + r * Math.cos(elr) * Math.cos(azr)
    );
    s.controls.update();
  }, [az, el, zoom]);
  await page.waitForTimeout(1200);
}

async function shot(name) {
  await page.screenshot({ path: `${S}/${name}.png`, animations: 'disabled' });
  console.log('shot', name);
}

function blankBack(faceId) {
  return page.evaluate((fid) => {
    const s = window.__studio;
    const u = s.faces[fid].shader.uniforms;
    const cv = document.createElement('canvas');
    cv.width = 2; cv.height = 2;
    const c2 = cv.getContext('2d');
    c2.fillStyle = 'black'; c2.fillRect(0, 0, 2, 2);
    const T = u.uBack.value.constructor; // THREE.Texture
    const bt = new T(cv);
    bt.needsUpdate = true;
    u.uBack.value = bt;
    return 'blanked ' + fid;
  }, faceId);
}

// ---- 3. Thickness=0 (kills parallax -> fringes freeze wrt tilt) -------------
await fresh();
await page.evaluate(() => {
  window.__studio.faces.front.shader.uniforms.uThicknessUm.value = 0.0;
});
await cam(6, 5, 0.8);
await shot('testcase_t0_az6');
await cam(14, 5, 0.8);
await shot('testcase_t0_az14');

// ---- 4. Physics prediction + parameter flip ---------------------------------
await fresh();
await cam(0, 0, 0.55);
const meta = await page.evaluate(() => {
  const s = window.__studio;
  let mesh = null;
  s.scene.traverse((o) => {
    if (o.isMesh && o.material === s.faces.front.shader) mesh = o;
  });
  if (!mesh) return { err: 'front mesh not found' };
  mesh.geometry.computeBoundingBox();
  const bb = mesh.geometry.boundingBox;
  const V3 = s.camera.position.constructor;
  const canvas = s.renderer.domElement;
  const proj = (x, y, z) => {
    const v = new V3(x, y, z).applyMatrix4(mesh.matrixWorld).project(s.camera);
    return [ (v.x + 1) / 2 * canvas.clientWidth, (1 - v.y) / 2 * canvas.clientHeight ];
  };
  // corners of the local bbox front surface
  const zf = bb.max.z;
  const c00 = proj(bb.min.x, bb.min.y, zf);
  const c10 = proj(bb.max.x, bb.min.y, zf);
  const c01 = proj(bb.min.x, bb.max.y, zf);
  const c11 = proj(bb.max.x, bb.max.y, zf);
  const u = s.faces.front.shader.uniforms;
  // zero the per-bucket angle spread so every leaf beats at dTheta = 3 deg
  u.uFrameAngleSpan.value = 0.0;
  return {
    canvas: [canvas.width, canvas.height, canvas.clientWidth, canvas.clientHeight],
    dpr: window.devicePixelRatio,
    corners: { c00, c10, c01, c11 },
    extentUm: u.uExtentUm.value.toArray(),
    slitPeriod: u.uSlitPeriodUm.value,
    carrierPeriod: u.uCarrierPeriodUm.value,
    slitAngleDeg: u.uSlitAngle.value * 180 / Math.PI,
    carrierAngleDeg: u.uCarrierAngle.value * 180 / Math.PI,
    duty: u.uGratingDuty.value,
    thickness: u.uThicknessUm.value,
  };
});
console.log('META ' + JSON.stringify(meta));
await page.waitForTimeout(400);
await shot('testcase_physics_span0_carrier70');
await page.evaluate(() => {
  window.__studio.faces.front.shader.uniforms.uCarrierPeriodUm.value = 84.0;
});
await page.waitForTimeout(400);
await shot('testcase_flip_carrier84');
await page.evaluate(() => {
  window.__studio.faces.front.shader.uniforms.uCarrierPeriodUm.value = 56.0;
});
await page.waitForTimeout(400);
await shot('testcase_flip_carrier56');

await browser.close();
console.log('done');
