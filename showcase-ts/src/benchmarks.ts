import type { ComputeBuffer, ComputeCommand, Tach } from "@depths/tach";
import {
  fractal,
  fractalColor,
  fractalOrbit,
  matrices,
  matrixActivate,
  matrixProduct,
  optionPrice,
  optionRisk,
  options,
  particleCommit,
  particlePredict,
  particles,
  render,
  renderColor,
  renderField,
  type CountParams,
  type ImageParams,
  type ParticleParams,
} from "../build/benchmarks.js";

type NumericArray = Float32Array | Uint32Array;
export type BenchmarkProfile = "gpu-large" | "gpu-cpu-medium";

export interface BenchmarkResult {
  readonly id: string;
  readonly name: string;
  readonly problem: string;
  readonly samples: number;
  readonly logicalStages: number;
  readonly fusedDispatches: 1;
  readonly unfusedDispatches: 2;
  readonly fusedMs: number;
  readonly unfusedMs: number;
  readonly fusionSpeedup: number;
  readonly fusedSamplesMs: readonly number[];
  readonly unfusedSamplesMs: readonly number[];
  readonly cpuMs: number | null;
  readonly gpuVsCpu: number | null;
  readonly correct: boolean;
  readonly check: string;
  readonly millionElementsPerSecond: number;
}

export interface BenchmarkReport {
  readonly profile: BenchmarkProfile;
  readonly results: readonly BenchmarkResult[];
}

export interface RenderedFrame {
  readonly width: number;
  readonly height: number;
  readonly pixels: Uint32Array;
}

interface Profile {
  readonly name: BenchmarkProfile;
  readonly samples: number;
  readonly cpu: boolean;
  readonly particleCount: number;
  readonly particleIterations: number;
  readonly fractalSize: number;
  readonly fractalIterations: number;
  readonly matrixCells: number;
  readonly matrixIterations: number;
  readonly optionCount: number;
  readonly optionIterations: number;
  readonly renderSize: number;
  readonly renderIterations: number;
}

const profiles: Record<BenchmarkProfile, Profile> = {
  "gpu-large": {
    name: "gpu-large",
    samples: 5,
    cpu: false,
	particleCount: 16_000_000,
    particleIterations: 16,
    fractalSize: 3072,
    fractalIterations: 4,
    matrixCells: 1 << 23,
    matrixIterations: 8,
    optionCount: 1 << 22,
    optionIterations: 4,
    renderSize: 3072,
    renderIterations: 16,
  },
  "gpu-cpu-medium": {
    name: "gpu-cpu-medium",
    samples: 3,
    cpu: true,
    particleCount: 1 << 20,
    particleIterations: 4,
    fractalSize: 768,
    fractalIterations: 1,
    matrixCells: 1 << 19,
    matrixIterations: 2,
    optionCount: 1 << 18,
    optionIterations: 1,
    renderSize: 1024,
    renderIterations: 4,
  },
};

interface Workload {
  readonly id: string;
  readonly name: string;
  readonly problem: string;
  readonly elements: number;
  readonly iterations: number;
  readonly fused: ComputeCommand;
  readonly unfused: readonly [ComputeCommand, ComputeCommand];
  readonly fusedOutput: ComputeBuffer<NumericArray>;
  readonly unfusedOutput: ComputeBuffer<NumericArray>;
  readonly buffers: readonly ComputeBuffer<unknown>[];
  readonly cpu?: () => NumericArray;
  readonly cpuTolerance: number;
}

interface CompletedWorkload {
  readonly result: BenchmarkResult;
  readonly output: NumericArray;
}

function median(values: readonly number[]): number {
  const sorted = [...values].sort((a, b) => a - b);
  return sorted[Math.floor(sorted.length / 2)]!;
}

async function complete(gpu: Tach, first: ComputeCommand, second?: ComputeCommand): Promise<void> {
  await (second === undefined ? gpu.submit(first) : gpu.submit(first, second));
  await gpu.idle();
}

async function timed(work: () => Promise<void>): Promise<number> {
  const start = performance.now();
  await work();
  return performance.now() - start;
}

async function measureGPU(
  samples: number,
  fused: () => Promise<void>,
  unfused: () => Promise<void>,
): Promise<{ fused: number[]; unfused: number[] }> {
  await fused();
  await unfused();
  const result = { fused: [] as number[], unfused: [] as number[] };
  for (let sample = 0; sample < samples; sample++) {
    if (sample % 2 === 0) {
      result.fused.push(await timed(fused));
      result.unfused.push(await timed(unfused));
    } else {
      result.unfused.push(await timed(unfused));
      result.fused.push(await timed(fused));
    }
  }
  return result;
}

function measureCPU(samples: number, work: () => NumericArray): number {
  work();
  const times: number[] = [];
  for (let sample = 0; sample < samples; sample++) {
    const start = performance.now();
    work();
    times.push(performance.now() - start);
  }
  return median(times);
}

function maxDifference(left: NumericArray, right: NumericArray): number {
  let maximum = 0;
  if (left instanceof Uint32Array && right instanceof Uint32Array) {
    for (let i = 0; i < left.length; i++) {
      for (let shift = 0; shift < 32; shift += 8) {
        maximum = Math.max(maximum, Math.abs(((left[i]! >>> shift) & 255) - ((right[i]! >>> shift) & 255)));
      }
    }
    return maximum;
  }
  for (let i = 0; i < left.length; i++) maximum = Math.max(maximum, Math.abs(left[i]! - right[i]!));
  return maximum;
}

async function execute(gpu: Tach, profile: Profile, workload: Workload): Promise<CompletedWorkload> {
  try {
    const samples = await measureGPU(
      profile.samples,
      () => complete(gpu, workload.fused),
      () => complete(gpu, workload.unfused[0], workload.unfused[1]),
    );
    const fusedMs = median(samples.fused);
    const unfusedMs = median(samples.unfused);
    const fusedOutput = await workload.fusedOutput.read();
    const unfusedOutput = await workload.unfusedOutput.read();
    const gpuError = maxDifference(fusedOutput, unfusedOutput);
    let cpuMs: number | null = null;
    let cpuError: number | null = null;
    if (profile.cpu && workload.cpu !== undefined) {
      cpuMs = measureCPU(profile.samples, workload.cpu);
      cpuError = maxDifference(fusedOutput, workload.cpu());
    }
    const correct = gpuError === 0 && (cpuError === null || cpuError <= workload.cpuTolerance);
    return {
      output: fusedOutput,
      result: {
        id: workload.id,
        name: workload.name,
        problem: workload.problem,
        samples: profile.samples,
        logicalStages: workload.iterations * 2,
        fusedDispatches: 1,
        unfusedDispatches: 2,
        fusedMs,
        unfusedMs,
        fusionSpeedup: unfusedMs / fusedMs,
        fusedSamplesMs: samples.fused,
        unfusedSamplesMs: samples.unfused,
        cpuMs,
        gpuVsCpu: cpuMs === null ? null : cpuMs / fusedMs,
        correct,
        check: `fused/unfused max error ${gpuError}${cpuError === null ? "" : `; CPU max error ${cpuError}`}`,
        millionElementsPerSecond: workload.elements * workload.iterations / (fusedMs / 1000) / 1e6,
      },
    };
  } finally {
    for (const buffer of workload.buffers) buffer.destroy();
  }
}

function particleWorkload(gpu: Tach, profile: Profile): Workload {
  const count = profile.particleCount;
  const positions = new Float32Array(count);
  const velocities = new Float32Array(count);
  for (let i = 0; i < count; i++) {
    positions[i] = (i % 1024) * 0.001;
    velocities[i] = ((i * 17) % 257 - 128) * 0.0001;
  }
  const gpuPositions = gpu.buffer(positions);
  const gpuVelocities = gpu.buffer(velocities);
  const predicted = gpu.buffer(new Float32Array(count));
  const fusedOutput = gpu.buffer(new Float32Array(count));
  const unfusedOutput = gpu.buffer(new Float32Array(count));
  const params: ParticleParams = { count, dt: 0.001 };
  const cpuOutput = new Float32Array(count);
  const cpu = (): Float32Array => {
    for (let iteration = 0; iteration < profile.particleIterations; iteration++) {
      for (let i = 0; i < count; i++) cpuOutput[i] = (positions[i]! + velocities[i]! * params.dt) * 0.99991 + 0.00003;
    }
    return cpuOutput;
  };
  return {
    id: "particles",
    name: "Particle prediction",
    problem: `${count.toLocaleString()} components × ${profile.particleIterations} iterations`,
    elements: count,
    iterations: profile.particleIterations,
    fused: particles(gpuPositions, gpuVelocities, fusedOutput, params, { repeat: profile.particleIterations }),
    unfused: [
      particlePredict(gpuPositions, gpuVelocities, predicted, params, { repeat: profile.particleIterations }),
      particleCommit(predicted, unfusedOutput, { repeat: profile.particleIterations }),
    ],
    fusedOutput,
    unfusedOutput,
    buffers: [gpuPositions, gpuVelocities, predicted, fusedOutput, unfusedOutput],
    cpu,
    cpuTolerance: 1e-5,
  };
}

function fractalCPU(params: ImageParams, iterations: number): Uint32Array {
  const output = new Uint32Array(params.width * params.height);
	const f32 = Math.fround;
  for (let repeat = 0; repeat < iterations; repeat++) {
    for (let y = 0; y < params.height; y++) {
      for (let x = 0; x < params.width; x++) {
		const cx = f32(f32(f32(x - f32(params.width * 0.5)) * params.scale) - 0.6);
		const cy = f32(f32(y - f32(params.height * 0.5)) * params.scale);
        let zx = 0;
        let zy = 0;
		for (let step = 0; step < 16; step++) [zx, zy] = [f32(f32(f32(zx * zx) - f32(zy * zy)) + cx), f32(f32(f32(2 * zx) * zy) + cy)];
		const magnitude = f32(f32(zx * zx) + f32(zy * zy));
		const raw = f32(Math.log2(f32(magnitude + 1)) * 24);
		const value = Number.isNaN(raw) ? 0 : Math.trunc(Math.min(255, Math.max(0, raw)));
        output[y * params.width + x] = (value | ((value * 3) << 8) | ((255 - value) << 16) | (255 << 24)) >>> 0;
      }
    }
  }
  return output;
}

function fractalWorkload(gpu: Tach, profile: Profile): Workload {
  const size = profile.fractalSize;
  const count = size * size;
  const orbit = gpu.buffer(new Float32Array(count));
  const fusedOutput = gpu.buffer(new Uint32Array(count));
  const unfusedOutput = gpu.buffer(new Uint32Array(count));
  const params: ImageParams = { width: size, height: size, scale: 3.2 / size, time: 0 };
  const launch = { size: count, repeat: profile.fractalIterations };
  return {
    id: "fractal",
    name: "Fixed-depth fractal",
    problem: `${size} × ${size} pixels × 16 orbit steps × ${profile.fractalIterations} iterations`,
    elements: count,
    iterations: profile.fractalIterations,
    fused: fractal(fusedOutput, params, { repeat: profile.fractalIterations }),
    unfused: [fractalOrbit(orbit, params, launch), fractalColor(orbit, unfusedOutput, launch)],
    fusedOutput,
    unfusedOutput,
    buffers: [orbit, fusedOutput, unfusedOutput],
    cpu: () => fractalCPU(params, profile.fractalIterations),
    cpuTolerance: 32,
  };
}

function matrixWorkload(gpu: Tach, profile: Profile): Workload {
  const count = profile.matrixCells;
  const left = new Float32Array(count);
  const right = new Float32Array(count);
  for (let i = 0; i < count; i++) {
    left[i] = ((i * 13) % 31 - 15) / 16;
    right[i] = ((i * 7) % 29 - 14) / 15;
  }
  const gpuLeft = gpu.buffer(left);
  const gpuRight = gpu.buffer(right);
  const product = gpu.buffer(new Float32Array(count));
  const fusedOutput = gpu.buffer(new Float32Array(count));
  const unfusedOutput = gpu.buffer(new Float32Array(count));
  const params: CountParams = { count };
  const cpuOutput = new Float32Array(count);
  const cpu = (): Float32Array => {
    for (let repeat = 0; repeat < profile.matrixIterations; repeat++) {
      for (let i = 0; i < count; i++) {
        const base = Math.trunc(i / 16) * 16;
        const cell = i % 16;
        const row = Math.trunc(cell / 4);
        const column = cell % 4;
        const value = left[base + row * 4]! * right[base + column]! +
          left[base + row * 4 + 1]! * right[base + 4 + column]! +
          left[base + row * 4 + 2]! * right[base + 8 + column]! +
          left[base + row * 4 + 3]! * right[base + 12 + column]!;
        cpuOutput[i] = value / (1 + Math.abs(value));
      }
    }
    return cpuOutput;
  };
  return {
    id: "matrices",
    name: "Batched 4 × 4 matrices",
    problem: `${(count / 16).toLocaleString()} products × ${profile.matrixIterations} iterations`,
    elements: count,
    iterations: profile.matrixIterations,
    fused: matrices(gpuLeft, gpuRight, fusedOutput, params, { repeat: profile.matrixIterations }),
    unfused: [
      matrixProduct(gpuLeft, gpuRight, product, { repeat: profile.matrixIterations }),
      matrixActivate(product, unfusedOutput, { repeat: profile.matrixIterations }),
    ],
    fusedOutput,
    unfusedOutput,
    buffers: [gpuLeft, gpuRight, product, fusedOutput, unfusedOutput],
    cpu,
    cpuTolerance: 1e-5,
  };
}

function normalCDF(x: number): number {
  const a = Math.abs(x);
  const t = 1 / (1 + 0.2316419 * a);
  const p = t * (0.319381530 + t * (-0.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429))));
  const y = 1 - 0.398942280 * Math.exp(-0.5 * a * a) * p;
  return x < 0 ? 1 - y : y;
}

function optionWorkload(gpu: Tach, profile: Profile): Workload {
  const count = profile.optionCount;
  const prices = gpu.buffer(new Float32Array(count));
  const fusedOutput = gpu.buffer(new Float32Array(count));
  const unfusedOutput = gpu.buffer(new Float32Array(count));
  const params: CountParams = { count };
  const cpuOutput = new Float32Array(count);
  const cpu = (): Float32Array => {
    for (let repeat = 0; repeat < profile.optionIterations; repeat++) {
      for (let i = 0; i < count; i++) {
        const spot = 80 + (i % 400) * 0.1;
        const strike = 90 + (i % 200) * 0.1;
        const years = 0.25 + (i % 365) / 365;
        const volatility = 0.15 + (i % 100) * 0.001;
        const root = volatility * Math.sqrt(years);
        const d1 = (Math.log(spot / strike) + (0.03 + 0.5 * volatility * volatility) * years) / root;
        const price = spot * normalCDF(d1) - strike * Math.exp(-0.03 * years) * normalCDF(d1 - root);
        cpuOutput[i] = price + Math.sqrt(Math.abs(price)) * 0.01;
      }
    }
    return cpuOutput;
  };
  return {
    id: "options",
    name: "Black–Scholes pricing",
    problem: `${count.toLocaleString()} options × ${profile.optionIterations} iterations`,
    elements: count,
    iterations: profile.optionIterations,
    fused: options(fusedOutput, params, { repeat: profile.optionIterations }),
    unfused: [
      optionPrice(prices, params, { repeat: profile.optionIterations }),
      optionRisk(prices, unfusedOutput, { repeat: profile.optionIterations }),
    ],
    fusedOutput,
    unfusedOutput,
    buffers: [prices, fusedOutput, unfusedOutput],
    cpu,
    cpuTolerance: 0.002,
  };
}

function renderCPU(params: ImageParams, iterations: number): Uint32Array {
  const output = new Uint32Array(params.width * params.height);
  for (let repeat = 0; repeat < iterations; repeat++) {
    for (let y = 0; y < params.height; y++) {
      const v = (y + 0.5) / params.height - 0.5;
      for (let x = 0; x < params.width; x++) {
        const u = (x + 0.5) / params.width - 0.5;
        const wave = Math.sin(u * 31 + params.time) * Math.cos(v * 23 - params.time);
        const field = Math.exp(-Math.sqrt(u * u + v * v) * 4) + wave * 0.35;
        const value = Math.min(1, Math.max(0, field * 0.5 + 0.5));
        const red = Math.trunc(value * 255);
        const green = Math.trunc(Math.sqrt(value) * 255);
        const blue = Math.trunc((1 - value) * 255);
        output[y * params.width + x] = (red | (green << 8) | (blue << 16) | (255 << 24)) >>> 0;
      }
    }
  }
  return output;
}

function renderWorkload(gpu: Tach, profile: Profile): Workload {
  const size = profile.renderSize;
  const count = size * size;
  const field = gpu.buffer(new Float32Array(count));
  const fusedOutput = gpu.buffer(new Uint32Array(count));
  const unfusedOutput = gpu.buffer(new Uint32Array(count));
  const params: ImageParams = { width: size, height: size, scale: 0, time: 1.7 };
  const launch = { size: count, repeat: profile.renderIterations };
  return {
    id: "render",
    name: "Procedural rendering",
    problem: `${size} × ${size} pixels × ${profile.renderIterations} iterations`,
    elements: count,
    iterations: profile.renderIterations,
    fused: render(fusedOutput, params, { repeat: profile.renderIterations }),
    unfused: [renderField(field, params, launch), renderColor(field, unfusedOutput, launch)],
    fusedOutput,
    unfusedOutput,
    buffers: [field, fusedOutput, unfusedOutput],
    cpu: () => renderCPU(params, profile.renderIterations),
    cpuTolerance: 2,
  };
}

export async function runBenchmarks(
  gpu: Tach,
  profileName: BenchmarkProfile,
  progress: (name: string, index: number) => void,
): Promise<{ report: BenchmarkReport; frame: RenderedFrame }> {
  const profile = profiles[profileName];
  const factories = [particleWorkload, fractalWorkload, matrixWorkload, optionWorkload, renderWorkload] as const;
  const results: BenchmarkResult[] = [];
  let frame: RenderedFrame | undefined;
  for (let index = 0; index < factories.length; index++) {
    const workload = factories[index]!(gpu, profile);
    progress(workload.name, index);
    const completed = await execute(gpu, profile, workload);
    results.push(completed.result);
    if (workload.id === "render") frame = { width: profile.renderSize, height: profile.renderSize, pixels: completed.output as Uint32Array };
  }
  if (frame === undefined) throw new Error("render benchmark did not produce a frame");
  return { report: { profile: profile.name, results }, frame };
}
