import type { Tach } from "@depths/tach";
import {
  integrateParticles,
  mandelbrot,
  multiplyMatrices,
  priceOptions,
  proceduralScene,
  type FractalParams,
  type RenderParams,
} from "../build/benchmarks.js";

export interface BenchmarkResult {
  readonly id: string;
  readonly name: string;
  readonly description: string;
  readonly problem: string;
  readonly samples: number;
  readonly dispatches: number;
  readonly gpuMs: number;
  readonly cpuMs: number;
  readonly speedup: number;
  readonly gpuRate: number;
  readonly cpuRate: number;
  readonly rateUnit: string;
  readonly correct: boolean;
  readonly check: string;
}

export interface RenderedFrame {
  readonly width: number;
  readonly height: number;
  readonly pixels: Uint32Array;
}

export interface BenchmarkRun {
  readonly results: readonly BenchmarkResult[];
  readonly frame: RenderedFrame;
}

interface Profile {
  readonly samples: number;
  readonly particleValues: number;
  readonly particleDispatches: number;
  readonly fractalSize: number;
  readonly fractalIterations: number;
  readonly fractalDispatches: number;
  readonly matrixSize: number;
  readonly matrixDispatches: number;
  readonly optionCount: number;
  readonly optionDispatches: number;
  readonly renderDispatches: number;
}

const full: Profile = {
  samples: 5,
  particleValues: 1 << 20,
  particleDispatches: 128,
  fractalSize: 768,
  fractalIterations: 192,
  fractalDispatches: 4,
  matrixSize: 256,
  matrixDispatches: 4,
  optionCount: 1 << 20,
  optionDispatches: 8,
  renderDispatches: 3,
};

const quick: Profile = {
  samples: 3,
  particleValues: 1 << 18,
  particleDispatches: 2,
  fractalSize: 128,
  fractalIterations: 64,
  fractalDispatches: 2,
  matrixSize: 96,
  matrixDispatches: 2,
  optionCount: 1 << 16,
  optionDispatches: 2,
  renderDispatches: 1,
};

type Progress = (name: string, index: number, total: number) => void;

function median(values: readonly number[]): number {
  const sorted = [...values].sort((left, right) => left - right);
  return sorted[Math.floor(sorted.length / 2)] ?? Number.NaN;
}

function nextFrame(): Promise<void> {
  return new Promise((resolve) => requestAnimationFrame(() => resolve()));
}

async function measureGPU(samples: number, work: () => Promise<void>): Promise<number[]> {
  const times: number[] = [];
  for (let sample = 0; sample < samples; sample++) {
    await nextFrame();
    const start = performance.now();
    await work();
    times.push(performance.now() - start);
  }
  return times;
}

async function measureCPU(samples: number, work: () => void): Promise<number[]> {
  const times: number[] = [];
  for (let sample = 0; sample < samples; sample++) {
    await nextFrame();
    const start = performance.now();
    work();
    times.push(performance.now() - start);
  }
  return times;
}

function result(
  identity: Pick<BenchmarkResult, "id" | "name" | "description" | "problem" | "dispatches" | "rateUnit">,
  samples: number,
  work: number,
  rateScale: number,
  gpuTimes: readonly number[],
  cpuTimes: readonly number[],
  correct: boolean,
  check: string,
): BenchmarkResult {
  const gpuMs = median(gpuTimes);
  const cpuMs = median(cpuTimes);
  return {
    ...identity,
    samples,
    gpuMs,
    cpuMs,
    speedup: cpuMs / gpuMs,
    gpuRate: work / (gpuMs / 1000) / rateScale,
    cpuRate: work / (cpuMs / 1000) / rateScale,
    correct,
    check,
  };
}

async function particles(gpu: Tach, profile: Profile): Promise<BenchmarkResult> {
  const count = profile.particleValues;
  const positions = new Float32Array(count);
  const velocities = new Float32Array(count);
  for (let i = 0; i < count; i++) {
    positions[i] = (i % 1024) * 0.001;
    velocities[i] = ((i * 17) % 257 - 128) * 0.0001;
  }
  const gpuPositions = gpu.buffer(positions);
  const gpuVelocities = gpu.buffer(velocities);
  const params = { dt: 0.001, count };
  await integrateParticles(gpuPositions, gpuVelocities, params, { size: count });
  const gpuTimes = await measureGPU(profile.samples, () => integrateParticles(
    gpuPositions,
    gpuVelocities,
    params,
    { size: count, dispatches: profile.particleDispatches },
  ));
  const actual = await gpuPositions.read();

  const expected = positions.slice();
  const integrateCPU = (dispatches: number): void => {
    for (let dispatch = 0; dispatch < dispatches; dispatch++) {
      for (let i = 0; i < count; i++) expected[i] = expected[i]! + velocities[i]! * params.dt;
    }
  };
  integrateCPU(1);
  const cpuTimes = await measureCPU(profile.samples, () => integrateCPU(profile.particleDispatches));
  let maxError = 0;
  for (let i = 0; i < count; i++) maxError = Math.max(maxError, Math.abs(actual[i]! - expected[i]!));
  gpuPositions.destroy();
  gpuVelocities.destroy();

  return result({
    id: "particles",
    name: "Particle integration",
    description: "Persistent position and velocity buffers; each batch advances every scalar component repeatedly.",
    problem: `${(count / 4).toLocaleString()} particles × ${profile.particleDispatches} steps`,
    dispatches: profile.particleDispatches,
    rateUnit: "million component updates/s",
  }, profile.samples, count * profile.particleDispatches, 1e6, gpuTimes, cpuTimes, maxError < 0.002,
  `maximum absolute error ${maxError.toExponential(2)}`);
}

function mandelbrotCPU(output: Uint32Array, params: FractalParams, dispatches: number): void {
  const f32 = Math.fround;
  for (let dispatch = 0; dispatch < dispatches; dispatch++) {
    for (let y = 0; y < params.height; y++) {
      const cy = f32(f32(f32(y - params.height * 0.5) * params.scale) + params.centerY);
      for (let x = 0; x < params.width; x++) {
        const cx = f32(f32(f32(x - params.width * 0.5) * params.scale) + params.centerX);
        let zx = 0;
        let zy = 0;
        let iteration = 0;
        while (iteration < params.maxIterations && f32(f32(zx * zx) + f32(zy * zy)) <= 4) {
          const nextX = f32(f32(f32(zx * zx) - f32(zy * zy)) + cx);
          zy = f32(f32(f32(2 * zx) * zy) + cy);
          zx = nextX;
          iteration++;
        }
        output[y * params.width + x] = iteration;
      }
    }
  }
}

async function fractal(gpu: Tach, profile: Profile): Promise<BenchmarkResult> {
  const width = profile.fractalSize;
  const height = profile.fractalSize;
  const pixels = width * height;
  const params: FractalParams = {
    width,
    height,
    maxIterations: profile.fractalIterations,
    scale: 3.2 / width,
    centerX: -0.6,
    centerY: 0,
  };
  const output = gpu.buffer(new Uint32Array(pixels));
  await mandelbrot(output, params, { size: [width, height] });
  const gpuTimes = await measureGPU(profile.samples, () => mandelbrot(
    output,
    params,
    { size: [width, height], dispatches: profile.fractalDispatches },
  ));
  const actual = await output.read();

  const expected = new Uint32Array(pixels);
  mandelbrotCPU(expected, params, 1);
  const cpuTimes = await measureCPU(profile.samples, () => mandelbrotCPU(
    expected,
    params,
    profile.fractalDispatches,
  ));
  let close = 0;
  let maxDifference = 0;
  for (let i = 0; i < pixels; i++) {
    const difference = Math.abs(actual[i]! - expected[i]!);
    if (difference <= 1) close++;
    maxDifference = Math.max(maxDifference, difference);
  }
  output.destroy();
  const agreement = close / pixels;

  return result({
    id: "fractal",
    name: "Mandelbrot escape",
    description: "A divergent, data-dependent loop computes every pixel independently with f32 arithmetic.",
    problem: `${width} × ${height}, limit ${params.maxIterations} × ${profile.fractalDispatches} renders`,
    dispatches: profile.fractalDispatches,
    rateUnit: "million pixel-dispatches/s",
  }, profile.samples, pixels * profile.fractalDispatches, 1e6, gpuTimes, cpuTimes, agreement >= 0.995,
  `${(agreement * 100).toFixed(3)}% within one iteration; maximum difference ${maxDifference}`);
}

function multiplyCPU(
  left: Float32Array,
  right: Float32Array,
  output: Float32Array,
  size: number,
  dispatches: number,
): void {
  const f32 = Math.fround;
  for (let dispatch = 0; dispatch < dispatches; dispatch++) {
    for (let row = 0; row < size; row++) {
      for (let column = 0; column < size; column++) {
        let sum = 0;
        for (let k = 0; k < size; k++) {
          sum = f32(sum + f32(left[row * size + k]! * right[k * size + column]!));
        }
        output[row * size + column] = sum;
      }
    }
  }
}

async function matrices(gpu: Tach, profile: Profile): Promise<BenchmarkResult> {
  const size = profile.matrixSize;
  const cells = size * size;
  const left = new Float32Array(cells);
  const right = new Float32Array(cells);
  for (let i = 0; i < cells; i++) {
    left[i] = ((i * 13) % 31 - 15) / 16;
    right[i] = ((i * 7) % 29 - 14) / 15;
  }
  const gpuLeft = gpu.buffer(left);
  const gpuRight = gpu.buffer(right);
  const gpuOutput = gpu.buffer(new Float32Array(cells));
  await multiplyMatrices(gpuLeft, gpuRight, gpuOutput, { size }, { size: [size, size] });
  const gpuTimes = await measureGPU(profile.samples, () => multiplyMatrices(
    gpuLeft,
    gpuRight,
    gpuOutput,
    { size },
    { size: [size, size], dispatches: profile.matrixDispatches },
  ));
  const actual = await gpuOutput.read();

  const expected = new Float32Array(cells);
  multiplyCPU(left, right, expected, size, 1);
  const cpuTimes = await measureCPU(profile.samples, () => multiplyCPU(
    left,
    right,
    expected,
    size,
    profile.matrixDispatches,
  ));
  let maxError = 0;
  for (let i = 0; i < cells; i++) maxError = Math.max(maxError, Math.abs(actual[i]! - expected[i]!));
  gpuLeft.destroy();
  gpuRight.destroy();
  gpuOutput.destroy();

  return result({
    id: "matrix",
    name: "Tiled matrix multiply",
    description: "16 × 16 tiles reuse two Workgroup arrays and synchronize twice per tile.",
    problem: `${size} × ${size} matrices × ${profile.matrixDispatches} products`,
    dispatches: profile.matrixDispatches,
    rateUnit: "GFLOP/s",
  }, profile.samples, 2 * size * size * size * profile.matrixDispatches, 1e9,
  gpuTimes, cpuTimes, maxError < 0.01, `maximum absolute error ${maxError.toExponential(2)}`);
}

function normalCDF(x: number): number {
  const magnitude = Math.abs(x);
  const t = 1 / (1 + 0.2316419 * magnitude);
  const polynomial = t * (0.319381530 + t * (-0.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429))));
  const positive = 1 - 0.398942280 * Math.exp(-0.5 * magnitude * magnitude) * polynomial;
  return x < 0 ? 1 - positive : positive;
}

function priceOptionsCPU(output: Float32Array, dispatches: number): void {
  for (let dispatch = 0; dispatch < dispatches; dispatch++) {
    for (let i = 0; i < output.length; i++) {
      const spot = 80 + (i % 400) * 0.1;
      const strike = 90 + (i % 200) * 0.1;
      const years = 0.25 + (i % 365) / 365;
      const volatility = 0.15 + (i % 100) * 0.001;
      const rate = 0.03;
      const sigmaRootTime = volatility * Math.sqrt(years);
      const d1 = (Math.log(spot / strike) + (rate + 0.5 * volatility * volatility) * years) / sigmaRootTime;
      const d2 = d1 - sigmaRootTime;
      output[i] = spot * normalCDF(d1) - strike * Math.exp(-rate * years) * normalCDF(d2);
    }
  }
}

async function options(gpu: Tach, profile: Profile): Promise<BenchmarkResult> {
  const count = profile.optionCount;
  const output = gpu.buffer(new Float32Array(count));
  await priceOptions(output, { count }, { size: count });
  const gpuTimes = await measureGPU(profile.samples, () => priceOptions(
    output,
    { count },
    { size: count, dispatches: profile.optionDispatches },
  ));
  const actual = await output.read();

  const expected = new Float32Array(count);
  priceOptionsCPU(expected, 1);
  const cpuTimes = await measureCPU(profile.samples, () => priceOptionsCPU(expected, profile.optionDispatches));
  let maxError = 0;
  for (let i = 0; i < count; i++) maxError = Math.max(maxError, Math.abs(actual[i]! - expected[i]!));
  output.destroy();

  return result({
    id: "options",
    name: "Black–Scholes pricing",
    description: "Independent option prices stress log, exp, sqrt, branching, and a generated helper function.",
    problem: `${count.toLocaleString()} options × ${profile.optionDispatches} valuations`,
    dispatches: profile.optionDispatches,
    rateUnit: "million option valuations/s",
  }, profile.samples, count * profile.optionDispatches, 1e6, gpuTimes, cpuTimes, maxError < 0.002,
  `maximum absolute error ${maxError.toExponential(2)}`);
}

function saturate(value: number): number {
  return Math.min(1, Math.max(0, value));
}

function smoothRange(edge0: number, edge1: number, value: number): number {
  const t = saturate((value - edge0) / (edge1 - edge0));
  return t * t * (3 - 2 * t);
}

function blend(base: number, layer: number, amount: number): number {
  return base + (layer - base) * saturate(amount);
}

function hashPixel(x: number, y: number): number {
  let value = (Math.imul(x, 374_761_393) + Math.imul(y, 668_265_263)) >>> 0;
  value = Math.imul(value ^ (value >>> 13), 1_274_126_177) >>> 0;
  value = (value ^ (value >>> 16)) >>> 0;
  return (value & 65_535) / 65_535;
}

function renderSceneCPU(output: Uint32Array, params: RenderParams, dispatches: number): void {
  const { width, height, time } = params;
  const aspect = width / height;
  for (let dispatch = 0; dispatch < dispatches; dispatch++) {
    for (let y = 0; y < height; y++) {
      const v = (y + 0.5) / height;
      const py = 0.5 - v;
      for (let x = 0; x < width; x++) {
        const u = (x + 0.5) / width;
        const px = (u - 0.5) * aspect;
        let red = 0.015 + 0.18 * v;
        let green = 0.025 + 0.06 * v;
        let blue = 0.11 + 0.22 * (1 - v);

        const cloudWave = 0.5 + 0.5 * Math.sin(px * 5 + Math.sin(py * 9 + time) * 1.7);
        const auroraHeight = 0.14 + 0.055 * Math.sin(px * 3.2 + time * 0.7) +
          0.025 * Math.sin(px * 11 - time);
        const aurora = Math.exp(-Math.abs(py - auroraHeight) * 20) * (0.35 + 0.65 * cloudWave);
        red += aurora * 0.08;
        green += aurora * 0.42;
        blue += aurora * 0.48;

        const sunX = px - 0.33;
        const sunY = py - 0.12;
        const sunDistance = Math.sqrt(sunX * sunX + sunY * sunY);
        const halo = Math.exp(-sunDistance * 7);
        const disk = smoothRange(0.145, 0.125, sunDistance);
        const rings = Math.exp(-Math.abs(sunDistance - 0.205) * 95) +
          0.55 * Math.exp(-Math.abs(sunDistance - 0.255) * 120);
        red += halo * 0.38 + rings * 0.55;
        green += halo * 0.08 + rings * 0.12;
        blue += halo * 0.16 + rings * 0.68;
        red = blend(red, 1, disk * 0.96);
        green = blend(green, 0.44 + 0.3 * (sunY / 0.145 + 0.5), disk * 0.96);
        blue = blend(blue, 0.16, disk * 0.96);

        const cellSize = 12;
        const cellX = Math.trunc(x / cellSize);
        const cellY = Math.trunc(y / cellSize);
        const seed = hashPixel(cellX, cellY);
        if (seed > 0.91 && py > -0.06) {
          const centerX = 2 + hashPixel(cellX + 17, cellY + 31) * 8;
          const centerY = 2 + hashPixel(cellX + 47, cellY + 11) * 8;
          const starX = x % cellSize - centerX;
          const starY = y % cellSize - centerY;
          const core = Math.exp(-(starX * starX + starY * starY) * 1.7);
          const rays = 0.22 * Math.exp(-Math.abs(starX) * 4 - Math.abs(starY) * 0.45) +
            0.22 * Math.exp(-Math.abs(starY) * 4 - Math.abs(starX) * 0.45);
          const sparkle = (core + rays) * (0.45 + seed * 0.8);
          red += sparkle;
          green += sparkle * 0.88;
          blue += sparkle * 1.15;
        }

        const distantHeight = -0.075 + 0.045 * Math.sin(px * 6.5 + 0.4) +
          0.022 * Math.sin(px * 16 - 1.3);
        const distant = smoothRange(distantHeight + 0.018, distantHeight - 0.012, py);
        red = blend(red, 0.075, distant);
        green = blend(green, 0.045, distant);
        blue = blend(blue, 0.16, distant);
        const ridge = Math.exp(-Math.abs(py - distantHeight) * 150);
        red += ridge * 0.24;
        blue += ridge * 0.38;

        const nearHeight = -0.19 + 0.065 * Math.sin(px * 4.3 - 0.8) +
          0.038 * Math.sin(px * 12.5 + 1.7);
        const near = smoothRange(nearHeight + 0.014, nearHeight - 0.018, py);
        red = blend(red, 0.018, near);
        green = blend(green, 0.025, near);
        blue = blend(blue, 0.055, near);

        const ground = smoothRange(-0.20, -0.235, py);
        const perspective = 0.57 + py;
        const verticalGrid = Math.exp(-Math.abs(Math.sin(px * 14 / perspective)) * 22);
        const horizontalGrid = Math.exp(-Math.abs(Math.sin(1.35 / perspective + time * 0.35)) * 24);
        const grid = saturate(verticalGrid + horizontalGrid);
        const reflection = Math.exp(-Math.abs(px - 0.33) * 7) *
          (0.45 + 0.55 * Math.sin(py * 210 + Math.sin(px * 19) * 2));
        red = blend(red, 0.018 + grid * 0.55 + reflection * 0.32, ground);
        green = blend(green, 0.035 + grid * 0.12 + reflection * 0.06, ground);
        blue = blend(blue, 0.09 + grid * 0.72 + reflection * 0.42, ground);

        const vignette = saturate(1 - (px * px * 0.5 + (v - 0.5) * (v - 0.5)) * 1.35);
        const grain = 0.97 + hashPixel(x + 101, y + 79) * 0.06;
        const scanline = 0.975 + 0.025 * Math.sin(y * 1.65);
        const exposure = (0.62 + 0.38 * vignette) * grain * scanline;
        red *= exposure;
        green *= exposure;
        blue *= exposure;
        const alpha = 0.94 + 0.06 * vignette;

        const red8 = Math.trunc(saturate(red) * 255 + 0.5);
        const green8 = Math.trunc(saturate(green) * 255 + 0.5);
        const blue8 = Math.trunc(saturate(blue) * 255 + 0.5);
        const alpha8 = Math.trunc(saturate(alpha) * 255 + 0.5);
        output[y * width + x] = (red8 | (green8 << 8) | (blue8 << 16) | (alpha8 << 24)) >>> 0;
      }
    }
  }
}

async function rendering(
  gpu: Tach,
  profile: Profile,
): Promise<{ readonly result: BenchmarkResult; readonly frame: RenderedFrame }> {
  const width = 1920;
  const height = 1080;
  const pixels = width * height;
  const params: RenderParams = { width, height, time: 1.7 };
  const output = gpu.buffer(new Uint32Array(pixels));
  await proceduralScene(output, params, { size: [width, height] });
  const gpuTimes = await measureGPU(profile.samples, () => proceduralScene(
    output,
    params,
    { size: [width, height], dispatches: profile.renderDispatches },
  ));
  const actual = await output.read();

  const expected = new Uint32Array(pixels);
  renderSceneCPU(expected, params, 1);
  const cpuTimes = await measureCPU(profile.samples, () => renderSceneCPU(
    expected,
    params,
    profile.renderDispatches,
  ));
  let totalDifference = 0;
  let maximumDifference = 0;
  let closePixels = 0;
  for (let i = 0; i < pixels; i++) {
    const gpuPixel = actual[i]!;
    const cpuPixel = expected[i]!;
    let pixelDifference = 0;
    for (let shift = 0; shift < 24; shift += 8) {
      const difference = Math.abs(((gpuPixel >>> shift) & 255) - ((cpuPixel >>> shift) & 255));
      totalDifference += difference;
      pixelDifference = Math.max(pixelDifference, difference);
      maximumDifference = Math.max(maximumDifference, difference);
    }
    if (pixelDifference <= 8) closePixels++;
  }
  const meanDifference = totalDifference / (pixels * 3);
  const agreement = closePixels / pixels;
  output.destroy();

  return {
    result: result({
      id: "render",
      name: "Procedural RGBA composition",
      description: "A full-HD scene composes gradients, aurora, stars, rings, mountains, a perspective grid, reflection, grain, and RGBA packing per pixel.",
      problem: `${width} × ${height} RGBA × ${profile.renderDispatches} frames`,
      dispatches: profile.renderDispatches,
      rateUnit: "million RGBA pixels/s",
    }, profile.samples, pixels * profile.renderDispatches, 1e6, gpuTimes, cpuTimes,
    meanDifference <= 2 && agreement >= 0.98,
    `mean RGB error ${meanDifference.toFixed(3)}; ${(agreement * 100).toFixed(3)}% of pixels within 8; maximum ${maximumDifference}`),
    frame: { width, height, pixels: actual },
  };
}

export async function runBenchmarks(
  gpu: Tach,
  fast: boolean,
  progress: Progress,
): Promise<BenchmarkRun> {
  const profile = fast ? quick : full;
  const workloads = [particles, fractal, matrices, options] as const;
  const results: BenchmarkResult[] = [];
  const total = workloads.length + 1;
  for (let index = 0; index < workloads.length; index++) {
    const workload = workloads[index]!;
    progress(["Particle integration", "Mandelbrot escape", "Tiled matrix multiply", "Black–Scholes pricing"][index]!, index, total);
    results.push(await workload(gpu, profile));
  }
  progress("Procedural RGBA composition", workloads.length, total);
  const rendered = await rendering(gpu, profile);
  results.push(rendered.result);
  return { results, frame: rendered.frame };
}
