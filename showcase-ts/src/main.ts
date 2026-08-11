import { tach } from "@depths/tach";
import { runBenchmarks, type BenchmarkResult, type RenderedFrame } from "./benchmarks.js";
import "./style.css";

function required<T extends Element>(selector: string): T {
  const element = document.querySelector<T>(selector);
  if (!element) throw new Error(`showcase is missing ${selector}`);
  return element;
}

const status = required<HTMLParagraphElement>("#status");
const adapter = required<HTMLSpanElement>("#adapter");
const grid = required<HTMLElement>("#benchmarks");
const summary = required<HTMLElement>("#summary");
const output = required<HTMLPreElement>("#output");
const canvas = required<HTMLCanvasElement>("#render-preview");

function milliseconds(value: number): string {
  return value < 1 ? `${(value * 1000).toFixed(0)} μs` : `${value.toFixed(value < 10 ? 2 : 1)} ms`;
}

function rate(value: number): string {
  if (value >= 100) return value.toFixed(0);
  if (value >= 10) return value.toFixed(1);
  return value.toFixed(2);
}

function card(result: BenchmarkResult): HTMLElement {
  const compared = result.cpuMs !== null && result.speedup !== null && result.cpuRate !== null;
  const element = document.createElement("article");
  element.className = "benchmark";
  element.dataset.benchmark = result.id;
  element.innerHTML = `
    <div class="card-heading">
      <div><p class="kind">${result.dispatches} dispatches per timed batch</p><h2>${result.name}</h2></div>
      <span class="check ${result.correct === null ? "gpu" : result.correct ? "pass" : "fail"}">${result.correct === null ? "GPU only" : result.correct ? "verified" : "mismatch"}</span>
    </div>
    <p class="description">${result.description}</p>
    <p class="problem">${result.problem}; median of ${result.samples} samples</p>
    <div class="metrics">
      <div><span>WebGPU batch</span><strong>${milliseconds(result.gpuMs)}</strong></div>
      <div><span>Pure TypeScript</span><strong>${compared ? milliseconds(result.cpuMs!) : "not run"}</strong></div>
      <div class="speedup"><span>Acceleration</span><strong>${compared ? `${result.speedup!.toFixed(2)}×` : "—"}</strong></div>
    </div>
    <div class="throughput">
      <span>WebGPU ${rate(result.gpuRate)}</span>
      <i style="--ratio:${compared ? Math.min(1, result.gpuRate / Math.max(result.gpuRate, result.cpuRate!)) : 1}"></i>
      <span>${compared ? `TypeScript ${rate(result.cpuRate!)} ${result.rateUnit}` : result.rateUnit}</span>
    </div>
    <p class="verification">${result.check}</p>`;
  return element;
}

function paint(frame: RenderedFrame): void {
  const context = canvas.getContext("2d");
  if (!context) throw new Error("the browser did not provide a 2D canvas context");
  const rgba = new Uint8ClampedArray(frame.pixels.length * 4);
  for (let index = 0; index < frame.pixels.length; index++) {
    const packed = frame.pixels[index]!;
    const offset = index * 4;
    rgba[offset] = packed & 255;
    rgba[offset + 1] = (packed >>> 8) & 255;
    rgba[offset + 2] = (packed >>> 16) & 255;
    rgba[offset + 3] = packed >>> 24;
  }
  canvas.width = frame.width;
  canvas.height = frame.height;
  context.putImageData(new ImageData(rgba, frame.width, frame.height), 0, 0);
}

async function main(): Promise<readonly BenchmarkResult[]> {
  const gpu = await tach({ adapter: { powerPreference: "high-performance" } });
  const info = gpu.adapter.info;
  adapter.textContent = [info.description, info.vendor, info.architecture].filter(Boolean).join(" · ") || "WebGPU adapter";
  const search = new URLSearchParams(location.search);
  const fast = search.has("quick");
  const compareCPU = !search.has("gpu-only");

  try {
    const run = await runBenchmarks(gpu, fast, compareCPU, (name, index, total) => {
      status.textContent = `Running ${index + 1} of ${total}: ${name}…`;
    });
    const results = run.results;
    paint(run.frame);
    grid.replaceChildren(...results.map(card));
    const verified = results.every((item) => item.correct === true);
    summary.innerHTML = compareCPU
      ? `<strong>${Math.exp(results.reduce((sum, item) => sum + Math.log(item.speedup!), 0) / results.length).toFixed(2)}×</strong><span>geometric-mean acceleration across five workloads</span>`
      : "<strong>GPU</strong><span>five full-size WebGPU workloads; CPU comparison skipped</span>";
    summary.classList.toggle("failed", compareCPU && !verified);
    status.textContent = `${compareCPU ? verified ? "Verified" : "Completed" : "Measured"} 5 benchmarks on ${adapter.textContent}.`;
    output.textContent = JSON.stringify(results, null, 2);
    return results;
  } finally {
    gpu.close();
  }
}

const ready = main().catch((error: unknown) => {
  status.textContent = "The benchmark suite could not run.";
  output.textContent = error instanceof Error ? error.stack ?? error.message : String(error);
  throw error;
});

globalThis.__tachShowcaseReady = ready;
