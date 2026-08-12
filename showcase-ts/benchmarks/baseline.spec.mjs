import { expect, test } from "@playwright/test";
import { mkdir, writeFile } from "node:fs/promises";
import { join } from "node:path";

test("GPU baselines", async ({ page }, testInfo) => {
  const profile = testInfo.config.metadata.profile;
  await page.goto(`/?profile=${profile}`);
  const result = await page.evaluate(() => globalThis.__tachShowcaseReady);
  const report = { generatedAt: new Date().toISOString(), adapter: await page.locator("#adapter").innerText(), ...result };
  await mkdir("reports", { recursive: true });
  await writeFile(join("reports", `${profile}.json`), `${JSON.stringify(report, null, 2)}\n`);

  expect(result.profile).toBe(profile);
  expect(result.results.map(({ id }) => id)).toEqual(["particles", "fractal", "matrices", "options", "render"]);
  for (const benchmark of result.results) {
    expect(benchmark.correct, benchmark.check).toBe(true);
    expect(benchmark.dispatches).toBe(2);
    expect(benchmark.gpuSamplesMs).toHaveLength(benchmark.samples);
    expect(benchmark.gpuMs).toBeGreaterThan(0);
    if (profile === "gpu-large") expect(benchmark.cpuMs).toBeNull();
    else {
      expect(benchmark.cpuMs).toBeGreaterThan(0);
      expect(benchmark.gpuVsCpu).toBeGreaterThan(0);
    }
  }
  console.table(result.results.map((benchmark) => ({
    workload: benchmark.name,
    gpuMs: benchmark.gpuMs.toFixed(2),
    cpuMs: benchmark.cpuMs?.toFixed(2) ?? "—",
  })));
});
