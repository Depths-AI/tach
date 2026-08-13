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
  for (const id of ["procedural", "mesh"]) {
    const data = await page.locator(`#preview-${id}`).evaluate((canvas) => canvas.toDataURL("image/png"));
    await writeFile(join("reports", `${profile}-${id}.png`), Buffer.from(data.slice(data.indexOf(",") + 1), "base64"));
  }

  expect(result.profile).toBe(profile);
  expect(result.results.map(({ id }) => id)).toEqual(["particles", "fractal", "matrices", "options", "procedural", "mesh"]);
  for (const benchmark of result.results) {
    expect(benchmark.correct, benchmark.check).toBe(true);
    expect(benchmark.dispatches).toBe(benchmark.id === "mesh" ? 4 : 2);
    expect(benchmark.gpuSamplesMs).toHaveLength(benchmark.samples);
    expect(benchmark.gpuMs).toBeGreaterThan(0);
    if (profile === "gpu-cpu-medium" && !["procedural", "mesh"].includes(benchmark.id)) {
      expect(benchmark.cpuMs).toBeGreaterThan(0);
      expect(benchmark.gpuVsCpu).toBeGreaterThan(0);
    } else expect(benchmark.cpuMs).toBeNull();
    if (["procedural", "mesh"].includes(benchmark.id)) expect(benchmark.framesPerSecond).toBeGreaterThan(0);
    else expect(benchmark.framesPerSecond).toBeNull();
  }
  console.table(result.results.map((benchmark) => ({
    workload: benchmark.name,
    gpuMs: benchmark.gpuMs.toFixed(2),
    fps: benchmark.framesPerSecond?.toFixed(1) ?? "—",
    cpuMs: benchmark.cpuMs?.toFixed(2) ?? "—",
  })));
});
