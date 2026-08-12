import { expect, test } from "@playwright/test";
import { mkdir, writeFile } from "node:fs/promises";
import { join } from "node:path";

test("fused and unfused GPU graphs", async ({ page }, testInfo) => {
  const profile = testInfo.config.metadata.profile;
  await page.goto(`/?profile=${profile}`);
  const result = await page.evaluate(() => globalThis.__tachShowcaseReady);
  const adapter = await page.locator("#adapter").innerText();
  const report = {
    generatedAt: new Date().toISOString(),
    adapter,
    ...result,
  };
  await mkdir("reports", { recursive: true });
  await writeFile(join("reports", `${profile}.json`), `${JSON.stringify(report, null, 2)}\n`);

  expect(result.profile).toBe(profile);
  expect(result.results.map(({ id }) => id)).toEqual(["particles", "fractal", "matrices", "options", "render"]);
  for (const benchmark of result.results) {
    expect(benchmark.correct, benchmark.check).toBe(true);
    expect(benchmark.fusedDispatches).toBe(1);
    expect(benchmark.unfusedDispatches).toBe(2);
    expect(benchmark.fusedSamplesMs).toHaveLength(benchmark.samples);
    expect(benchmark.unfusedSamplesMs).toHaveLength(benchmark.samples);
    expect(benchmark.fusedMs).toBeGreaterThan(0);
    expect(benchmark.unfusedMs).toBeGreaterThan(0);
    if (profile === "gpu-large") {
      expect(benchmark.cpuMs).toBeNull();
    } else {
      expect(benchmark.cpuMs).toBeGreaterThan(0);
      expect(benchmark.gpuVsCpu).toBeGreaterThan(0);
    }
  }
  console.table(result.results.map((benchmark) => ({
    workload: benchmark.name,
    fusedMs: benchmark.fusedMs.toFixed(2),
    unfusedMs: benchmark.unfusedMs.toFixed(2),
    fusion: `${benchmark.fusionSpeedup.toFixed(2)}x`,
    cpuMs: benchmark.cpuMs?.toFixed(2) ?? "—",
  })));
});
