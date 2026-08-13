import { tach } from "@depths/tach";
import { runBenchmarks, type BenchmarkProfile } from "./benchmarks.js";

const ready = tach({ adapter: { powerPreference: "high-performance" } }).then(async (gpu) => {
  const profile: BenchmarkProfile = new URLSearchParams(location.search).get("profile") === "gpu-large" ? "gpu-large" : "gpu-cpu-medium";
  document.querySelector("#adapter")!.textContent = [gpu.adapter.info.description, gpu.adapter.info.vendor, gpu.adapter.info.architecture].filter(Boolean).join(" · ");
  try { return await runBenchmarks(gpu, profile, () => {}); } finally { gpu.close(); }
});
Object.assign(globalThis, { __tachShowcaseReady: ready });
