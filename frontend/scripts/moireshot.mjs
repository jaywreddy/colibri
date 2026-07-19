// Headless screenshot of the two-plane moire sub-case rig (moiretest.html).
// Usage (from frontend/):
//   pnpm exec node scripts/moireshot.mjs out.png 'p=130&az=8&aa=1'
import { chromium } from '@playwright/test';

const out = process.argv[2] ?? 'moire.png';
const qs = process.argv[3] ?? '';

const browser = await chromium.launch();
try {
  const page = await browser.newPage({ viewport: { width: 1280, height: 900 } });
  await page.goto('http://localhost:5173/moiretest.html?' + qs, { waitUntil: 'domcontentloaded' });
  await page.waitForFunction(() => window.__ready === true, null, { timeout: 20000 });
  const ppp = await page.evaluate(() => window.__ppp);
  await page.waitForTimeout(3000);
  await page.locator('canvas').screenshot({ path: out });
  console.log(`saved ${out}  ppp=${ppp.toFixed(2)}`);
} finally {
  await browser.close();
}
