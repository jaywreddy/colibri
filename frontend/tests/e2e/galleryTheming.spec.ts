/**
 * Gallery groups patterns by theme and badges each card with its tier.
 * Without these, new patterns silently disappear from the UI.
 */
import { test, expect } from '@playwright/test';

test('gallery renders Colombia and Global Travel section headers', async ({ page }) => {
  await page.goto('/');
  await page.waitForResponse((r) => r.url().endsWith('/patterns') && r.status() === 200);

  const colombia = page.locator('h3[data-theme="Colombia"]');
  const global = page.locator('h3[data-theme="Global Travel"]');
  await expect(colombia).toBeVisible();
  await expect(global).toBeVisible();
  await expect(colombia).toHaveText(/colombia/i);
  await expect(global).toHaveText(/global travel/i);
});

test('gallery shows at least one card of each tier', async ({ page }) => {
  await page.goto('/');
  await page.waitForResponse((r) => r.url().endsWith('/patterns') && r.status() === 200);

  await expect(page.locator('button[data-tier="1"]').first()).toBeVisible();
  await expect(page.locator('button[data-tier="2"]').first()).toBeVisible();
  await expect(page.locator('button[data-tier="3"]').first()).toBeVisible();
});

test('clicking a Colombia card loads a Colombia-themed pattern', async ({ page }) => {
  await page.goto('/');
  await page.waitForResponse((r) => r.url().endsWith('/patterns') && r.status() === 200);

  const firstColombia = page.locator('button[data-theme="Colombia"]').first();
  await expect(firstColombia).toBeVisible();
  const slug = await firstColombia.getAttribute('data-slug');
  expect(slug).toBeTruthy();

  // Register the listener BEFORE the click so the /default fetch is caught.
  const respP = page.waitForResponse(
    (r) => r.url().includes(`/patterns/${slug}/default`) && r.status() === 200,
    { timeout: 30_000 }
  );
  await firstColombia.click();
  const resp = await respP;
  const manifest = await resp.json();
  expect(manifest.slug).toBe(slug);

  const catalog = await page.evaluate(async () => {
    const mod = await import('/src/store.ts');
    return mod.useStore.getState().catalog;
  });
  const entry = catalog.find((c: { slug: string }) => c.slug === slug);
  expect(entry?.theme).toBe('Colombia');
});
