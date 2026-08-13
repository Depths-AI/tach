import type { ComputeBuffer, ComputeCommand, Tach } from "@depths/tach";
import {
  fractalColor,
  fractalOrbit,
  matrixActivate,
  matrixProduct,
  optionPrice,
  optionRisk,
  particleCommit,
  particlePredict,
  meshFrame,
  proceduralFrame,
  type CountParams,
  type ImageParams,
  type MeshParams,
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
  readonly dispatches: number;
  readonly gpuMs: number;
  readonly gpuSamplesMs: readonly number[];
  readonly cpuMs: number | null;
  readonly gpuVsCpu: number | null;
  readonly correct: boolean;
  readonly check: string;
  readonly millionElementsPerSecond: number;
  readonly framesPerSecond: number | null;
}

export interface BenchmarkReport {
  readonly profile: BenchmarkProfile;
  readonly results: readonly BenchmarkResult[];
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
  readonly visualWidth: number;
  readonly visualHeight: number;
  readonly meshColumns: number;
  readonly meshRows: number;
  readonly meshLayers: number;
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
    visualWidth: 1920,
    visualHeight: 1080,
    meshColumns: 256,
    meshRows: 144,
    meshLayers: 4,
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
    visualWidth: 960,
    visualHeight: 540,
    meshColumns: 160,
    meshRows: 90,
    meshLayers: 2,
  },
};

interface Workload {
  readonly id: string;
  readonly name: string;
  readonly problem: string;
  readonly elements: number;
  readonly iterations: number;
  readonly stages?: number;
  readonly dispatches?: number;
  readonly commands: readonly [ComputeCommand, ...ComputeCommand[]];
  readonly output: ComputeBuffer<NumericArray>;
  readonly buffers: readonly ComputeBuffer<unknown>[];
  readonly cpu?: () => NumericArray;
  readonly validate?: (output: NumericArray) => string | null;
  readonly visual?: "procedural" | "mesh";
  readonly cpuTolerance: number;
}

function median(values: readonly number[]): number {
  const sorted = [...values].sort((a, b) => a - b);
  return sorted[Math.floor(sorted.length / 2)]!;
}

async function complete(gpu: Tach, commands: readonly [ComputeCommand, ...ComputeCommand[]]): Promise<void> {
  await gpu.submit(...commands);
  await gpu.idle();
}

async function timed(work: () => Promise<void>): Promise<number> {
  const start = performance.now();
  await work();
  return performance.now() - start;
}

async function measureGPU(samples: number, work: () => Promise<void>): Promise<number[]> {
  await work();
  const result: number[] = [];
  for (let sample = 0; sample < samples; sample++) result.push(await timed(work));
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

function renderVisual(id: "procedural" | "mesh", width: number, height: number, pixels: Uint32Array): void {
  const bytes = new Uint8ClampedArray(pixels.length * 4);
  for (let i = 0; i < pixels.length; i++) {
    const pixel = pixels[i]!;
    const offset = i * 4;
    bytes[offset] = pixel & 255;
    bytes[offset + 1] = (pixel >>> 8) & 255;
    bytes[offset + 2] = (pixel >>> 16) & 255;
    bytes[offset + 3] = pixel >>> 24;
  }
  const canvas = document.createElement("canvas");
  canvas.id = `preview-${id}`;
  canvas.width = width;
  canvas.height = height;
  const context = canvas.getContext("2d");
  if (context === null) throw new Error("2D canvas is unavailable for visual capture");
  context.putImageData(new ImageData(bytes, width, height), 0, 0);
  document.body.append(canvas);
}

async function execute(gpu: Tach, profile: Profile, workload: Workload): Promise<BenchmarkResult> {
  try {
    const samples = await measureGPU(profile.samples, () => complete(gpu, workload.commands));
    const gpuMs = median(samples);
    const output = await workload.output.read();
    let cpuMs: number | null = null;
    let cpuError: number | null = null;
    if (profile.cpu && workload.cpu !== undefined) {
      cpuMs = measureCPU(profile.samples, workload.cpu);
      cpuError = maxDifference(output, workload.cpu());
    }
    const validation = cpuError === null ? workload.validate?.(output) ?? null : null;
    const correct = cpuError === null ? validation === null : cpuError <= workload.cpuTolerance;
    if (workload.visual !== undefined && output instanceof Uint32Array) renderVisual(workload.visual, profile.visualWidth, profile.visualHeight, output);
    return {
        id: workload.id,
        name: workload.name,
        problem: workload.problem,
        samples: profile.samples,
        logicalStages: workload.iterations * (workload.stages ?? 2),
        dispatches: workload.dispatches ?? 2,
        gpuMs,
        gpuSamplesMs: samples,
        cpuMs,
        gpuVsCpu: cpuMs === null ? null : cpuMs / gpuMs,
        correct,
        check: cpuError === null ? validation ?? (workload.validate === undefined ? "completed" : "GPU frame validated") : `CPU max error ${cpuError}`,
        millionElementsPerSecond: workload.elements * workload.iterations / (gpuMs / 1000) / 1e6,
        framesPerSecond: workload.visual === undefined ? null : 1000 / gpuMs,
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
  const output = gpu.buffer(new Float32Array(count));
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
    commands: [
      particlePredict(gpuPositions, gpuVelocities, predicted, params, { repeat: profile.particleIterations }),
      particleCommit(predicted, output, { repeat: profile.particleIterations }),
    ],
    output,
    buffers: [gpuPositions, gpuVelocities, predicted, output],
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
  const output = gpu.buffer(new Uint32Array(count));
  const params: ImageParams = { width: size, height: size, scale: 3.2 / size, time: 0 };
  const launch = { size: count, repeat: profile.fractalIterations };
  return {
    id: "fractal",
    name: "Fixed-depth fractal",
    problem: `${size} × ${size} pixels × 16 orbit steps × ${profile.fractalIterations} iterations`,
    elements: count,
    iterations: profile.fractalIterations,
    commands: [fractalOrbit(orbit, params, launch), fractalColor(orbit, output, launch)],
    output,
    buffers: [orbit, output],
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
  const output = gpu.buffer(new Float32Array(count));
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
    commands: [
      matrixProduct(gpuLeft, gpuRight, product, { repeat: profile.matrixIterations }),
      matrixActivate(product, output, { repeat: profile.matrixIterations }),
    ],
    output,
    buffers: [gpuLeft, gpuRight, product, output],
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
  const output = gpu.buffer(new Float32Array(count));
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
    commands: [
      optionPrice(prices, params, { repeat: profile.optionIterations }),
      optionRisk(prices, output, { repeat: profile.optionIterations }),
    ],
    output,
    buffers: [prices, output],
    cpu,
    cpuTolerance: 0.002,
  };
}

function validateVisual(output: NumericArray): string | null {
  if (!(output instanceof Uint32Array)) return "visual output is not packed RGBA8";
  const samples = Math.min(output.length, 4096);
  const stride = Math.max(1, Math.trunc(output.length / samples));
  const first = output[0]!;
  let different = 0;
  for (let i = 0; i < output.length; i += stride) {
    const pixel = output[i]!;
    if ((pixel >>> 24) !== 255) return `pixel ${i} is not opaque`;
    if (pixel !== first) different++;
  }
  return different < samples / 16 ? "visual output lacks scene variation" : null;
}

function proceduralWorkload(gpu: Tach, profile: Profile): Workload {
  const count = profile.visualWidth * profile.visualHeight;
  const output = gpu.buffer(new Uint32Array(count));
  const params: ImageParams = { width: profile.visualWidth, height: profile.visualHeight, scale: 0, time: 1.7 };
  return {
    id: "procedural",
    name: "Procedural scene",
    problem: `${profile.visualWidth} × ${profile.visualHeight} pixels, up to 64 signed-distance steps`,
    elements: count,
    iterations: 1,
    commands: [proceduralFrame(output, params)],
    output,
    buffers: [output],
    validate: validateVisual,
    visual: "procedural",
    cpuTolerance: 0,
  };
}

function createMesh(columns: number, rows: number, layers: number): { vertices: Float32Array; indices: Uint32Array } {
  const layerVertices = (columns + 1) * (rows + 1);
  const vertices = new Float32Array(layerVertices * layers * 4);
  const indices = new Uint32Array(columns * rows * layers * 6);
  let vertex = 0;
  for (let layer = 0; layer < layers; layer++) {
    for (let y = 0; y <= rows; y++) {
      const ny = 1 - y / rows * 2;
      for (let x = 0; x <= columns; x++) {
        const nx = x / columns * 2 - 1;
        const depth = 2.1 + layer * 0.52 + Math.sin(nx * 6.1 + layer) * Math.cos(ny * 4.7 - layer) * 0.14;
        vertices[vertex++] = nx * depth * 0.92;
        vertices[vertex++] = ny * depth * 0.92;
        vertices[vertex++] = depth;
        vertices[vertex++] = layer;
      }
    }
  }
  let index = 0;
  for (let layer = 0; layer < layers; layer++) {
    const base = layer * layerVertices;
    for (let y = 0; y < rows; y++) {
      for (let x = 0; x < columns; x++) {
        const a = base + y * (columns + 1) + x;
        const b = a + 1;
        const c = a + columns + 1;
        const d = c + 1;
        indices[index++] = a;
        indices[index++] = b;
        indices[index++] = c;
        indices[index++] = b;
        indices[index++] = d;
        indices[index++] = c;
      }
    }
  }
  return { vertices, indices };
}

function meshWorkload(gpu: Tach, profile: Profile): Workload {
  const count = profile.visualWidth * profile.visualHeight;
  const mesh = createMesh(profile.meshColumns, profile.meshRows, profile.meshLayers);
  const triangles = mesh.indices.length / 3;
  if (triangles >= 0x000f_ffff) throw new RangeError("mesh benchmark exceeds packed triangle identifier capacity");
  const vertices = gpu.buffer(mesh.vertices);
  const indices = gpu.buffer(mesh.indices);
  const visibility = gpu.buffer(new Uint32Array(count));
  const output = gpu.buffer(new Uint32Array(count));
  const params: MeshParams = { width: profile.visualWidth, height: profile.visualHeight, time: 1.7 };
  return {
    id: "mesh",
    name: "Triangle-mesh software rendering",
    problem: `${profile.visualWidth} × ${profile.visualHeight} pixels, ${triangles.toLocaleString()} triangles, ${profile.meshLayers} depth layers`,
    elements: triangles,
    iterations: 1,
    stages: 4,
    dispatches: 4,
    commands: [meshFrame(vertices, indices, visibility, output, params)],
    output,
    buffers: [vertices, indices, visibility, output],
    validate: validateVisual,
    visual: "mesh",
    cpuTolerance: 0,
  };
}

export async function runBenchmarks(
  gpu: Tach,
  profileName: BenchmarkProfile,
  progress: (name: string, index: number) => void,
): Promise<BenchmarkReport> {
  const profile = profiles[profileName];
  const factories = [particleWorkload, fractalWorkload, matrixWorkload, optionWorkload, proceduralWorkload, meshWorkload] as const;
  const results: BenchmarkResult[] = [];
  for (let index = 0; index < factories.length; index++) {
    const workload = factories[index]!(gpu, profile);
    progress(workload.name, index);
    results.push(await execute(gpu, profile, workload));
  }
  return { profile: profile.name, results };
}
