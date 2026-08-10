import { expect, test } from "@playwright/test";

test("the showcase verifies five repeated GPU and TypeScript workloads", async ({ page }, testInfo) => {
  const pageErrors = [];
  page.on("pageerror", (error) => pageErrors.push(error.message));
  const full = process.env.TACH_SHOWCASE_FULL === "1" || testInfo.config.metadata.showcaseProfile === "full";

  await page.goto(full ? "/" : "/?quick=1");
  const results = await page.evaluate(() => globalThis.__tachShowcaseReady);
  testInfo.annotations.push(
    { type: "Benchmark profile", description: full ? "full" : "quick" },
    { type: "WebGPU adapter", description: await page.locator("#adapter").innerText() },
  );
  await testInfo.attach("benchmark-results", {
    body: JSON.stringify(results),
    contentType: "application/json",
  });

  expect(results.map((result) => result.id)).toEqual(["particles", "fractal", "matrix", "options", "render"]);
  const expectedDispatches = full ? [128, 4, 4, 8, 3] : [2, 2, 2, 2, 1];
  for (const [index, result] of results.entries()) {
    expect(result.samples).toBe(full ? 5 : 3);
    expect(result.dispatches).toBe(expectedDispatches[index]);
    expect(result.gpuMs).toBeGreaterThan(0);
    expect(result.cpuMs).toBeGreaterThan(0);
    expect(result.speedup).toBeGreaterThan(0);
    expect(result.correct, result.check).toBe(true);
  }
  await expect(page.locator("#benchmarks .benchmark")).toHaveCount(5);
  await expect(page.locator(".check.pass")).toHaveCount(5);
  await expect(page.locator("#render-preview")).toHaveJSProperty("width", 1920);
  await expect(page.locator("#render-preview")).toHaveJSProperty("height", 1080);
  await expect(page.locator("#status")).toContainText("Verified 5 benchmarks");
  expect(pageErrors).toEqual([]);
  if (full) console.table(results.map(({ name, gpuMs, cpuMs, speedup, gpuRate, rateUnit }) => ({
    workload: name,
    gpuMs: gpuMs.toFixed(2),
    cpuMs: cpuMs.toFixed(2),
    speedup: `${speedup.toFixed(2)}x`,
    throughput: `${gpuRate.toFixed(2)} ${rateUnit}`,
  })));
});
