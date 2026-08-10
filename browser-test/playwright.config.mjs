import { defineConfig } from "@playwright/test";

// Prefer the machine's normal WebGPU adapter. The unsafe flags make Chromium's
// bundled SwiftShader fallback available when no physical adapter exists; they
// do not force it when hardware is usable.
const launchArgs = [
  "--enable-gpu",
  "--enable-unsafe-webgpu",
  "--enable-unsafe-swiftshader",
];

if (process.platform === "linux") {
  launchArgs.push(
    "--use-angle=vulkan",
    "--enable-features=Vulkan",
    "--disable-vulkan-surface",
  );
}

export default defineConfig({
  testDir: "tests",
  fullyParallel: false,
  workers: 1,
  timeout: 60_000,
  expect: { timeout: 10_000 },
  reporter: [
    ["list"],
    ["html", { open: "never" }],
    ["./markdown-reporter.mjs", { outputFile: "test-report.md" }],
  ],
  use: {
    baseURL: "http://127.0.0.1:4173",
    channel: "chromium",
    headless: true,
    trace: "retain-on-failure",
    launchOptions: { args: launchArgs },
  },
  webServer: {
    command: "node server.mjs",
    url: "http://127.0.0.1:4173/health",
    reuseExistingServer: false,
    timeout: 30_000,
  },
});
