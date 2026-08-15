/// <reference lib="dom" />

import { tach } from "@depths/tach";
import { runBenchmarks, stressPresentations } from "./benchmarks.ts";

const run = tach(async (gpu) => {
  const canvas = document.createElement("canvas");
  canvas.width = 1920;
  canvas.height = 1080;
  document.body.append(canvas);
  await stressPresentations(gpu, canvas);
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
