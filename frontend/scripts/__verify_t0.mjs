import { chromium } from '@playwright/test';
const S = process.argv[2];
const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 1280, height: 900 } });
await page.goto('http://localhost:5173/', { waitUntil: 'domcontentloaded' });
await page.waitForFunction(
  () => !!(window.__studio && window.__studio.store.getState().boxManifest),
  null, { timeout: 45000 }
);
await page.waitForTimeout(4000);
// baseline radius, camera helper with ABSOLUTE radius factor
await page.evaluate(() => {
  const s = window.__studio;
  s.store.getState().setAutoRotate(false);
  s.store.getState().setLidTargetDeg(0);
  window.__r0 = s.camera.position.clone().sub(s.controls.target).length();
  window.__cam = (a, e, z) => {
    const azr = (a * Math.PI) / 180, elr = (e * Math.PI) / 180;
    const tgt = s.controls.target;
    const r = window.__r0 * z;
    s.camera.position.set(
      tgt.x + r * Math.cos(elr) * Math.sin(azr),
      tgt.y + r * Math.sin(elr),
      tgt.z + r * Math.cos(elr) * Math.cos(azr)
    );
    s.controls.update();
  };
});
async function shotAt(az, el, z, thick, name) {
  await page.evaluate(([a, e, zz, t]) => {
    window.__studio.faces.front.shader.uniforms.uThicknessUm.value = t;
    window.__cam(a, e, zz);
  }, [az, el, z, thick]);
  await page.waitForTimeout(900);
  await page.screenshot({ path: `${S}/${name}.png`, animations: 'disabled' });
  console.log('shot', name);
}
await shotAt(6, 5, 0.8, 500, 'testcase_t500_az6');
await shotAt(14, 5, 0.8, 500, 'testcase_t500_az14');
await shotAt(6, 5, 0.8, 0, 'testcase_t0b_az6');
await shotAt(14, 5, 0.8, 0, 'testcase_t0b_az14');
await browser.close();
console.log('done');
