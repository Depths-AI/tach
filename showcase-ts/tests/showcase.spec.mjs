import { expect, test } from "@playwright/test";
import { writeFile } from "node:fs/promises";

test("the showcase runs five repeated GPU workloads", async ({ page }, testInfo) => {
  const pageErrors = [];
  page.on("pageerror", (error) => pageErrors.push(error.message));
  const profile = process.env.TACH_SHOWCASE_FULL === "1" ? "full" : testInfo.config.metadata.showcaseProfile;
  const full = profile === "full";
  const gpuOnly = profile === "gpu-only";

  await page.goto(gpuOnly ? "/?gpu-only=1" : full ? "/" : "/?quick=1");
  const results = await page.evaluate(() => globalThis.__tachShowcaseReady);
  testInfo.annotations.push(
    { type: "Benchmark profile", description: profile },
    { type: "WebGPU adapter", description: await page.locator("#adapter").innerText() },
  );
  await testInfo.attach("benchmark-results", {
    body: JSON.stringify(results),
    contentType: "application/json",
  });

  expect(results.map((result) => result.id)).toEqual(["particles", "fractal", "matrix", "options", "render"]);
  const expectedDispatches = full || gpuOnly ? [1, 4, 4, 8, 3] : [1, 2, 2, 2, 1];
  for (const [index, result] of results.entries()) {
    expect(result.samples).toBe(full || gpuOnly ? 5 : 3);
    expect(result.gpuSamplesMs).toHaveLength(result.samples);
    expect(result.gpuSamplesMs.every((sample) => sample > 0)).toBe(true);
    expect(result.dispatches).toBe(expectedDispatches[index]);
    expect(result.gpuMs).toBeGreaterThan(0);
    if (gpuOnly) {
      expect(result.cpuMs).toBeNull();
      expect(result.speedup).toBeNull();
      expect(result.cpuRate).toBeNull();
      expect(result.correct).toBeNull();
    } else {
      expect(result.cpuMs).toBeGreaterThan(0);
      expect(result.cpuSamplesMs).toHaveLength(result.samples);
      expect(result.speedup).toBeGreaterThan(0);
      expect(result.cpuRate).toBeGreaterThan(0);
      expect(result.correct, result.check).toBe(true);
    }
  }
  await expect(page.locator("#benchmarks .benchmark")).toHaveCount(5);
  await expect(page.locator(gpuOnly ? ".check.gpu" : ".check.pass")).toHaveCount(5);
  const preview = page.locator("#render-preview");
  await expect(preview).toHaveJSProperty("width", 1920);
  await expect(preview).toHaveJSProperty("height", 1080);
  const screenshot = testInfo.outputPath("procedural-scene.png");
  const png = await preview.evaluate((canvas) => canvas.toDataURL("image/png"));
  await writeFile(screenshot, png.slice(png.indexOf(",") + 1), "base64");
  await testInfo.attach("procedural-scene", { path: screenshot, contentType: "image/png" });
  await expect(page.locator("#status")).toContainText(`${gpuOnly ? "Measured" : "Verified"} 5 benchmarks`);
  expect(pageErrors).toEqual([]);
  if (full) console.table(results.map(({ name, gpuMs, gpuSamplesMs, cpuMs, speedup, gpuRate, rateUnit }) => ({
    workload: name,
    gpuMs: gpuMs.toFixed(2),
    samples: gpuSamplesMs.map((sample) => sample.toFixed(2)).join(" / "),
    cpuMs: cpuMs.toFixed(2),
    speedup: `${speedup.toFixed(2)}x`,
    throughput: `${gpuRate.toFixed(2)} ${rateUnit}`,
  })));
  if (gpuOnly) console.table(results.map(({ name, gpuMs, gpuSamplesMs, gpuRate, rateUnit }) => ({
    workload: name,
    gpuMs: gpuMs.toFixed(2),
    samples: gpuSamplesMs.map((sample) => sample.toFixed(2)).join(" / "),
    throughput: `${gpuRate.toFixed(2)} ${rateUnit}`,
  })));
});
