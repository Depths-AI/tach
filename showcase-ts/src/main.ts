import { tach } from "@depths/tach";
import { runBenchmarks } from "./benchmarks.js";

const ready = tach({ adapter: { powerPreference: "high-performance" } }).then(async (gpu) => {
  document.querySelector("#adapter")!.textContent = [gpu.adapter.info.description, gpu.adapter.info.vendor, gpu.adapter.info.architecture].filter(Boolean).join(" / ");
  try { return await runBenchmarks(gpu); } finally { gpu.close(); }
});

Object.assign(globalThis, { __tachShowcaseReady: ready });
