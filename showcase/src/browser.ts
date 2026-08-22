/// <reference lib="dom" />

import { tach } from "@depths/tach";
import { runBenchmarks } from "./benchmarks.ts";

const canvases = Object.fromEntries(
  ["procedural", "mesh"].map((name) => {
    const canvas = document.createElement("canvas");
    canvas.width = 1920;
    canvas.height = 1080;
    document.body.append(canvas);
    return [name, canvas];
  }),
) as Record<"procedural" | "mesh", HTMLCanvasElement>;

const png = (canvas: HTMLCanvasElement): Promise<Blob> =>
  new Promise((resolve, reject) =>
    canvas.toBlob(
      (blob) => blob ? resolve(blob) : reject(new Error("PNG capture failed")),
      "image/png",
    )
  );

const run = tach(async (gpu) => {
  const report = await runBenchmarks(gpu, canvases);
  await fetch("/result", {
    method: "POST",
    body: JSON.stringify({ adapter: gpu.adapter, report }),
  });
  for (const [name, canvas] of Object.entries(canvases)) {
    await fetch(`/frame/${name}`, { method: "POST", body: await png(canvas) });
  }
  return { adapter: gpu.adapter, workloads: report.results.length };
}, { powerPreference: "high-performance" });

Object.assign(globalThis, { __tachShowcase: run });
