import { tach } from "@depths/tach";
import { runBenchmarks } from "./benchmarks.ts";

const run = tach(async (gpu) => {
  const result = await runBenchmarks(gpu);
  await fetch("/result", {
    method: "POST",
    body: JSON.stringify({ adapter: gpu.adapter, report: result.report }),
  });
  for (const [name, frame] of Object.entries(result.frames)) {
    await fetch(`/frame/${name}`, {
      method: "POST",
      body: frame.slice().buffer,
    });
  }
  return { adapter: gpu.adapter, workloads: result.report.results.length };
}, { powerPreference: "high-performance" });

Object.assign(globalThis, { __tachShowcase: run });
