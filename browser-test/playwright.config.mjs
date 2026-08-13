import { defineConfig } from "@playwright/test";
import { webGPU } from "../webgpu-playwright.mjs";

export default defineConfig({
  testDir: "tests",
  globalSetup: "./scripts/build-examples.mjs",
  fullyParallel: false,
  workers: 1,
  timeout: 60_000,
  expect: { timeout: 10_000 },
  reporter: "line",
  use: {
    ...webGPU,
    baseURL: "http://127.0.0.1:4173",
  },
  webServer: {
    command: "node server.mjs",
    url: "http://127.0.0.1:4173/health",
    reuseExistingServer: false,
    timeout: 30_000,
  },
});
