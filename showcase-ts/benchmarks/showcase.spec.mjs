import { expect, test } from "@playwright/test";
import { mkdir, writeFile } from "node:fs/promises";
import { join } from "node:path";

const cell = (value) => String(value).replaceAll("|", "\\|");
const number = (value) => typeof value === "number" ? Number.isInteger(value) ? value.toLocaleString("en-US") : value.toFixed(4) : String(value);

function markdown(report) {
  const lines = [
    "# Tach large GPU showcase",
    "",
    `Generated: ${report.generatedAt}`,
    "",
    `Adapter: ${report.adapter}`,
    "",
    `Timing: ${report.timing}`,
    "",
    "## Results",
    "",
    "| Category | Workload | Median | Raw samples (ms) | Throughput | FPS | Validation |",
    "|---|---|---:|---|---:|---:|---|",
  ];
  for (const result of report.results) lines.push(`| ${result.category} | ${cell(result.name)} | ${result.gpuMs.toFixed(3)} ms | ${result.gpuSamplesMs.map((sample) => sample.toFixed(3)).join(", ")} | ${result.throughput.toFixed(3)} ${result.throughputUnit} | ${result.framesPerSecond?.toFixed(1) ?? "-"} | ${cell(result.check)} |`);
  lines.push("", "## Workloads", "");
  for (const result of report.results) {
    lines.push(`### ${result.name}`, "", result.problem, "", `Dispatches per timed sample: ${result.dispatches}`, "", "| Detail | Value |", "|---|---:|");
    for (const [key, value] of Object.entries(result.details)) lines.push(`| ${cell(key)} | ${cell(number(value))} |`);
    lines.push("");
  }
  lines.push("## Contract", "", "Each workload receives one untimed warmup followed by five timed samples in one persistent Tach session. Every sample measures command submission through GPU completion. Allocation, initial upload, readback, PNG encoding, report generation, and validation are excluded.", "");
  return lines.join("\n");
}

test("large GPU workloads", async ({ page }) => {
  await page.goto("/");
  const result = await page.evaluate(() => globalThis.__tachShowcaseReady);
  const report = { generatedAt: new Date().toISOString(), adapter: await page.locator("#adapter").innerText(), ...result };
  await mkdir("reports", { recursive: true });
  await Promise.all([
    writeFile(join("reports", "gpu.json"), `${JSON.stringify(report, null, 2)}\n`),
    writeFile(join("reports", "gpu.md"), markdown(report)),
    ...["procedural", "mesh"].map(async (id) => {
      const data = await page.locator(`#${id}`).evaluate((canvas) => canvas.toDataURL("image/png"));
      await writeFile(join("reports", `${id}.png`), Buffer.from(data.slice(data.indexOf(",") + 1), "base64"));
    }),
  ]);

  expect(result.results.map(({ id }) => id)).toEqual(["procedural", "mesh", "matrix", "monte-carlo", "particles", "wave"]);
  for (const benchmark of result.results) {
    expect(benchmark.correct, benchmark.check).toBe(true);
    expect(benchmark.gpuSamplesMs).toHaveLength(5);
    expect(benchmark.gpuMs).toBeGreaterThan(0);
  }
  console.table(result.results.map((benchmark) => ({
    category: benchmark.category,
    workload: benchmark.name,
    medianMs: benchmark.gpuMs.toFixed(2),
    throughput: `${benchmark.throughput.toFixed(2)} ${benchmark.throughputUnit}`,
    fps: benchmark.framesPerSecond?.toFixed(1) ?? "-",
  })));
});
