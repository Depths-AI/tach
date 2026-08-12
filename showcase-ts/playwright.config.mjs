import { defineConfig } from "@playwright/test";
import { webGPU } from "../webgpu-playwright.mjs";

export default defineConfig({
  testDir: "benchmarks",
  workers: 1,
  timeout: 600_000,
  reporter: "line",
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
