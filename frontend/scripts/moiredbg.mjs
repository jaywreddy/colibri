import { chromium } from '@playwright/test';
const qs = process.argv[2] ?? 'p=240&az=10&el=5';
const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 1280, height: 900 } });
page.on('console', (m) => console.log('CONSOLE', m.type(), m.text()));
page.on('pageerror', (e) => console.log('PAGEERROR', e.message));
await page.goto('http://localhost:5173/moiretest.html?' + qs, { waitUntil: 'domcontentloaded' });
await page.waitForFunction(() => window.__ready === true, null, { timeout: 20000 }).catch(() => console.log('no __ready'));
await page.waitForTimeout(1000);
const info = await page.evaluate(() => {
  const c = document.querySelector('canvas');
  const gl = c.getContext('webgl2') || c.getContext('webgl');
  // sample center pixel
  const px = new Uint8Array(4);
  if (gl) gl.readPixels(640, 450, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, px);
  return { ppp: window.__ppp, w: c.width, h: c.height, center: Array.from(px), glLost: gl ? gl.isContextLost() : 'nogl' };
});
console.log('INFO', JSON.stringify(info));
await browser.close();
