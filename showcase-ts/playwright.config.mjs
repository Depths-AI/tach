import { defineConfig } from "@playwright/test";
import { webGPU } from "../webgpu-playwright.mjs";

export default defineConfig({
  testDir: "tests",
  workers: 1,
  timeout: 120_000,
  expect: { timeout: 15_000 },
  metadata: { showcaseProfile: "quick" },
  reporter: [
    ["list"],
    ["./markdown-reporter.mjs"],
  ],
  use: {
    ...webGPU,
    baseURL: "http://127.0.0.1:4174",
  },
  webServer: {
    command: "vite preview --host 127.0.0.1 --port 4174 --strictPort",
    url: "http://127.0.0.1:4174",
    reuseExistingServer: false,
    timeout: 30_000,
  },
});
