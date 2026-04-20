/**
 * Shared Playwright fixture that auto-attaches the frontend log ring
 * buffer to any failing test. Specs opt in by importing `test` and
 * `expect` from this module instead of `@playwright/test` directly:
 *
 *     import { test, expect } from './fixtures';
 *
 * Specs that don't need this (the pre-existing specs) can keep importing
 * from '@playwright/test' as before.
 */
import { test as base, expect } from '@playwright/test';
import { dumpLogOnFailure } from './helpers';

export const test = base.extend<{ autoLogDump: void }>({
  autoLogDump: [
    async ({ page }, use, testInfo) => {
      await use();
      await dumpLogOnFailure(page, testInfo);
    },
    { auto: true },
  ],
});

export { expect };
