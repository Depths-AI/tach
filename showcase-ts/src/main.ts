import { tach } from "@depths/tach";
import { runBenchmarks, type BenchmarkProfile, type BenchmarkReport, type BenchmarkResult, type RenderedFrame } from "./benchmarks.js";
import "./style.css";

function required<T extends Element>(selector: string): T {
  const element = document.querySelector<T>(selector);
  if (element === null) throw new Error(`showcase is missing ${selector}`);
  return element;
}

const status = required<HTMLParagraphElement>("#status");
const adapter = required<HTMLSpanElement>("#adapter");
const grid = required<HTMLElement>("#benchmarks");
const summary = required<HTMLElement>("#summary");
const output = required<HTMLPreElement>("#output");
const canvas = required<HTMLCanvasElement>("#render-preview");

function milliseconds(value: number): string {
  return value < 1 ? `${(value * 1000).toFixed(0)} µs` : `${value.toFixed(value < 10 ? 2 : 1)} ms`;
}

function card(result: BenchmarkResult): HTMLElement {
  const element = document.createElement("article");
  element.className = "benchmark";
  element.dataset.benchmark = result.id;
  element.innerHTML = `
    <div class="card-heading">
      <div><p class="kind">2 GPU stages → 1 dispatch</p><h2>${result.name}</h2></div>
      <span class="check ${result.correct ? "pass" : "fail"}">${result.correct ? "verified" : "mismatch"}</span>
    </div>
    <p class="problem">${result.problem}; median of ${result.samples} alternating samples</p>
    <div class="metrics">
      <div><span>Fused GPU</span><strong>${milliseconds(result.fusedMs)}</strong></div>
      <div><span>Unfused GPU</span><strong>${milliseconds(result.unfusedMs)}</strong></div>
      <div class="speedup"><span>Fusion speedup</span><strong>${result.fusionSpeedup.toFixed(2)}×</strong></div>
      <div><span>CPU</span><strong>${result.cpuMs === null ? "not run" : milliseconds(result.cpuMs)}</strong></div>
    </div>
    <p class="verification">${result.logicalStages} logical stage executions; ${result.millionElementsPerSecond.toFixed(1)} million fused elements/s; ${result.check}</p>`;
  return element;
}

function paint(frame: RenderedFrame): void {
  const context = canvas.getContext("2d");
  if (context === null) throw new Error("2D canvas is unavailable");
  const bytes = new Uint8ClampedArray(frame.pixels.slice().buffer as ArrayBuffer);
  canvas.width = frame.width;
  canvas.height = frame.height;
  context.putImageData(new ImageData(bytes, frame.width, frame.height), 0, 0);
}

async function main(): Promise<BenchmarkReport> {
  const requested = new URLSearchParams(location.search).get("profile");
  const profile: BenchmarkProfile = requested === "gpu-large" ? "gpu-large" : "gpu-cpu-medium";
  const gpu = await tach({ adapter: { powerPreference: "high-performance" } });
  const info = gpu.adapter.info;
  adapter.textContent = [info.description, info.vendor, info.architecture].filter(Boolean).join(" · ") || "WebGPU adapter";
  try {
    const run = await runBenchmarks(gpu, profile, (name, index) => {
      status.textContent = `Running ${index + 1} of 5: ${name}…`;
    });
    paint(run.frame);
    grid.replaceChildren(...run.report.results.map(card));
    const geometricMean = Math.exp(run.report.results.reduce((sum, result) => sum + Math.log(result.fusionSpeedup), 0) / run.report.results.length);
    const verified = run.report.results.every((result) => result.correct);
    summary.innerHTML = `<strong>${geometricMean.toFixed(2)}×</strong><span>geometric-mean fused-GPU advantage across five A/B workloads</span>`;
    summary.classList.toggle("failed", !verified);
    status.textContent = `${verified ? "Verified" : "Failed"} ${profile} on ${adapter.textContent}.`;
    output.textContent = JSON.stringify(run.report, null, 2);
    return run.report;
  } finally {
    gpu.close();
  }
}

globalThis.__tachShowcaseReady = main().catch((error: unknown) => {
  status.textContent = "Benchmark failed.";
  output.textContent = error instanceof Error ? error.stack ?? error.message : String(error);
  throw error;
});
