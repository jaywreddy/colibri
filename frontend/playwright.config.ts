import { defineConfig } from '@playwright/test';

const FRONTEND_PORT = 5173;
const BACKEND_PORT = 8765;

export default defineConfig({
  testDir: './tests/e2e',
  timeout: 60_000,
  expect: { timeout: 10_000 },
  fullyParallel: false,
  workers: 1, // WebGL context is singleton per page; parallel contexts flake
  retries: 0,
  reporter: process.env.CI ? [['dot'], ['html', { open: 'never' }]] : 'list',
  use: {
    baseURL: `http://localhost:${FRONTEND_PORT}`,
    trace: 'retain-on-failure',
    video: 'retain-on-failure',
    screenshot: 'only-on-failure',
    // Chromium WebGL stability in headless: ANGLE over D3D11 keeps the
    // context alive instead of cycling lost/restored on startup.
    launchOptions: {
      args: [
        '--use-gl=angle',
        '--use-angle=d3d11',
        '--enable-webgl',
        '--ignore-gpu-blocklist',
      ],
    },
  },
  webServer: [
    {
      // NOTE: no --reload flag. A past regression was two uvicorn processes
      // racing each other because --reload was left on.
      //
      // reuseExistingServer is ALWAYS true (incl. CI) so Playwright never
      // spawns a competing uvicorn chain against a manually-started backend.
      // When the port is free it still starts one; when taken it just uses
      // the existing listener. The previous `!process.env.CI` guard caused
      // orphan uv.exe→uvicorn.exe→python.exe chains on Windows dev machines.
      command: `uv run uvicorn app.main:app --host 127.0.0.1 --port ${BACKEND_PORT}`,
      cwd: '../backend',
      url: `http://127.0.0.1:${BACKEND_PORT}/patterns`,
      reuseExistingServer: true,
      timeout: 120_000,
      stdout: 'ignore',
      stderr: 'pipe',
    },
    {
      command: 'pnpm dev',
      url: `http://localhost:${FRONTEND_PORT}`,
      reuseExistingServer: true,
      timeout: 60_000,
      stdout: 'ignore',
      stderr: 'pipe',
    },
  ],
});
