import { defineConfig } from '@playwright/test';

export default defineConfig({
  testDir: './tests/visual',
  outputDir: './tests/visual/test-results',
  snapshotPathTemplate: '{testDir}/__screenshots__/{testFilePath}/{arg}{ext}',
  timeout: 30_000,
  expect: {
    toHaveScreenshot: {
      // Allow small pixel differences across environments
      maxDiffPixelRatio: 0.01,
    },
  },
  use: {
    baseURL: 'http://localhost:3000',
    screenshot: 'only-on-failure',
  },
  projects: [
    {
      name: 'chromium',
      use: { browserName: 'chromium', viewport: { width: 1440, height: 900 } },
    },
  ],
  // Dev server must be started separately (e.g., via run-dev-server.sh)
  // In CI, the workflow handles this before running tests
  webServer: {
    command: 'echo "Expecting dev server to be running on localhost:3000"',
    url: 'http://localhost:3000',
    reuseExistingServer: true,
    timeout: 5_000,
  },
});
