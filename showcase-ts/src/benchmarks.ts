import type { ComputeBuffer, ComputeCommand, Tach } from "@depths/tach";
import { denseMatrixProduct } from "../build/matrix.js";
import { meshWorld, type MeshParams } from "../build/mesh.js";
import { monteCarloPaths, type MonteCarloParams } from "../build/monte-carlo.js";
import { particleDynamics, type ParticleParams } from "../build/particles.js";
import { proceduralWorld, type ProceduralParams } from "../build/procedural.js";
import { waveSimulation, type WaveParams } from "../build/wave.js";

const samples = 5;
const frameWidth = 1920;
const frameHeight = 1080;
type Detail = string | number | boolean;

export interface BenchmarkResult {
  readonly id: string;
  readonly category: "rendering" | "mathematics" | "physics";
  readonly name: string;
  readonly problem: string;
  readonly dispatches: number;
  readonly gpuSamplesMs: readonly number[];
  readonly gpuMs: number;
  readonly throughput: number;
  readonly throughputUnit: string;
  readonly framesPerSecond: number | null;
  readonly correct: boolean;
  readonly check: string;
  readonly details: Readonly<Record<string, Detail>>;
}

export interface BenchmarkReport {
  readonly samples: number;
  readonly timing: string;
  readonly results: readonly BenchmarkResult[];
}

interface Readback {
  readonly correct: boolean;
  readonly check: string;
  readonly details?: Readonly<Record<string, Detail>>;
  readonly frame?: Uint32Array;
  readonly units?: number;
}

interface Workload {
  readonly id: string;
  readonly category: BenchmarkResult["category"];
  readonly name: string;
  readonly problem: string;
  readonly dispatches: number;
  readonly command: ComputeCommand;
  readonly buffers: readonly ComputeBuffer<unknown>[];
  readonly units: number;
  readonly divisor: number;
  readonly throughputUnit: string;
  readonly details: Readonly<Record<string, Detail>>;
  readonly frame?: "procedural" | "mesh";
  readonly readback: () => Promise<Readback>;
}

function median(values: readonly number[]): number {
  return [...values].sort((a, b) => a - b)[Math.floor(values.length / 2)]!;
}

async function complete(gpu: Tach, command: ComputeCommand): Promise<void> {
  await gpu.submit(command);
  await gpu.idle();
}

async function timed(gpu: Tach, command: ComputeCommand): Promise<number> {
  const start = performance.now();
  await complete(gpu, command);
  return performance.now() - start;
}

function renderFrame(id: "procedural" | "mesh", pixels: Uint32Array): void {
  const canvas = document.querySelector<HTMLCanvasElement>(`#${id}`)!;
  canvas.width = frameWidth;
  canvas.height = frameHeight;
  const bytes = new Uint8ClampedArray(pixels.byteLength);
  bytes.set(new Uint8Array(pixels.buffer, pixels.byteOffset, pixels.byteLength));
  canvas.getContext("2d")!.putImageData(new ImageData(bytes, frameWidth, frameHeight), 0, 0);
}

function validateFrame(pixels: Uint32Array): Readback {
  const stride = Math.max(1, Math.trunc(pixels.length / 8192));
  const colors = new Set<number>();
  for (let i = 0; i < pixels.length; i += stride) {
    const pixel = pixels[i]!;
    if ((pixel >>> 24) !== 255) return { correct: false, check: `pixel ${i} is not opaque` };
    colors.add(pixel);
  }
  return {
    correct: colors.size >= 64,
    check: colors.size >= 64 ? "GPU frame is opaque and spatially varied" : "frame lacks spatial variation",
    details: { sampledColors: colors.size },
    frame: pixels,
  };
}

function proceduralWorkload(gpu: Tach): Workload {
  const count = frameWidth * frameHeight;
  const pixels = gpu.buffer(new Uint32Array(count));
  const params: ProceduralParams = { width: frameWidth, height: frameHeight, time: 1.7 };
  return {
    id: "procedural",
    category: "rendering",
    name: "Procedural world",
    problem: "1920 x 1080 analytic world with traversal, shadows, ambient occlusion, lighting, fog, and post-processing",
    dispatches: 3,
    command: proceduralWorld(pixels, params),
    buffers: [pixels],
    units: count,
    divisor: 1e6,
    throughputUnit: "million pixels/s",
    details: { width: frameWidth, height: frameHeight, traceSteps: 72, shadowSteps: 8, ambientOcclusionSamples: 3 },
    frame: "procedural",
    readback: async () => validateFrame(await pixels.read()),
  };
}

function terrainHeight(x: number, z: number): number {
  return -1.05 + Math.sin(x * 0.42) * 0.22 + Math.cos(z * 0.31) * 0.16;
}

function noise(index: number, salt: number): number {
  let value = Math.imul(index ^ salt, 0x45d9f3b);
  value = Math.imul(value ^ (value >>> 16), 0x45d9f3b);
  return ((value ^ (value >>> 16)) >>> 0) / 0x1_0000_0000;
}

type Point = readonly [number, number, number];
type Surface = (u: number, v: number) => Point;
interface Scene { readonly vertices: number[]; readonly normals: number[]; readonly indices: number[] }

function normalize3(x: number, y: number, z: number): Point {
  const length = Math.hypot(x, y, z);
  return length > 1e-8 ? [x / length, y / length, z / length] : [0, 1, 0];
}

function appendSurface(scene: Scene, columns: number, rows: number, wrapU: boolean, wrapV: boolean, material: number, roughness: number, point: Surface): void {
  const columnVertices = wrapU ? columns : columns + 1;
  const rowVertices = wrapV ? rows : rows + 1;
  const base = scene.vertices.length / 4;
  const parameter = (value: number, wrap: boolean): number => wrap ? ((value % 1) + 1) % 1 : Math.max(0, Math.min(1, value));
  const sample = (u: number, v: number): Point => point(parameter(u, wrapU), parameter(v, wrapV));
  const epsilon = 0.001;
  for (let row = 0; row < rowVertices; row++) for (let column = 0; column < columnVertices; column++) {
    const u = column / columns;
    const v = row / rows;
    const position = sample(u, v);
    const beforeU = sample(u - epsilon, v); const afterU = sample(u + epsilon, v);
    const beforeV = sample(u, v - epsilon); const afterV = sample(u, v + epsilon);
    const ux = afterU[0] - beforeU[0]; const uy = afterU[1] - beforeU[1]; const uz = afterU[2] - beforeU[2];
    const vx = afterV[0] - beforeV[0]; const vy = afterV[1] - beforeV[1]; const vz = afterV[2] - beforeV[2];
    const normal = normalize3(vy * uz - vz * uy, vz * ux - vx * uz, vx * uy - vy * ux);
    scene.vertices.push(position[0], position[1], position[2], material);
    scene.normals.push(normal[0], normal[1], normal[2], roughness);
  }
  for (let row = 0; row < rows; row++) for (let column = 0; column < columns; column++) {
    const nextColumn = (column + 1) % columnVertices;
    const nextRow = (row + 1) % rowVertices;
    const a = base + row * columnVertices + column;
    const b = base + row * columnVertices + nextColumn;
    const c = base + nextRow * columnVertices + column;
    const d = base + nextRow * columnVertices + nextColumn;
    scene.indices.push(a, c, b, b, c, d);
  }
}

function knotPoint(u: number, v: number, centerX: number, centerY: number, centerZ: number, scale: number, phase: number): Point {
  const curve = (angle: number): Point => {
    const radius = 2 + 0.68 * Math.cos(3 * angle + phase);
    return [radius * Math.cos(2 * angle), 0.68 * Math.sin(3 * angle + phase), radius * Math.sin(2 * angle)];
  };
  const angle = Math.PI * 2 * u;
  const center = curve(angle);
  const before = curve(angle - 0.001);
  const after = curve(angle + 0.001);
  const tangent = normalize3(after[0] - before[0], after[1] - before[1], after[2] - before[2]);
  const side = normalize3(-tangent[2], 0, tangent[0]);
  const up = normalize3(side[1] * tangent[2] - side[2] * tangent[1], side[2] * tangent[0] - side[0] * tangent[2], side[0] * tangent[1] - side[1] * tangent[0]);
  const tube = 0.26 + 0.035 * Math.sin(angle * 5 + phase);
  const ring = Math.PI * 2 * v;
  return [
    centerX + scale * (center[0] + tube * (Math.cos(ring) * up[0] + Math.sin(ring) * side[0])),
    centerY + scale * (center[1] + tube * (Math.cos(ring) * up[1] + Math.sin(ring) * side[1])),
    centerZ + scale * (center[2] + tube * (Math.cos(ring) * up[2] + Math.sin(ring) * side[2])),
  ];
}

function ribbonPoint(u: number, v: number, centerX: number, centerY: number, centerZ: number, scale: number, twists: number): Point {
  const angle = Math.PI * 2 * u;
  const across = (v - 0.5) * 1.25;
  const twist = angle * twists;
  const radius = 1.75 + across * Math.cos(twist);
  return [centerX + scale * radius * Math.cos(angle), centerY + scale * (across * Math.sin(twist) + 0.18 * Math.sin(angle * 3)), centerZ + scale * radius * Math.sin(angle)];
}

function signedPower(value: number, exponent: number): number {
  return Math.sign(value) * Math.abs(value) ** exponent;
}

function superquadricPoint(u: number, v: number, centerX: number, centerY: number, centerZ: number, radiusX: number, radiusY: number, radiusZ: number, latitudePower: number, longitudePower: number, phase: number): Point {
  const longitude = Math.PI * 2 * u;
  const latitude = Math.PI * (v - 0.5);
  const latitudeCosine = signedPower(Math.cos(latitude), latitudePower);
  const ripple = 1 + 0.10 * Math.sin(longitude * 5 + phase) * Math.cos(latitude * 3);
  return [
    centerX + radiusX * latitudeCosine * signedPower(Math.cos(longitude), longitudePower) * ripple,
    centerY + radiusY * signedPower(Math.sin(latitude), latitudePower) * ripple,
    centerZ + radiusZ * latitudeCosine * signedPower(Math.sin(longitude), longitudePower) * ripple,
  ];
}

function createScene(): { vertices: Float32Array; normals: Float32Array; indices: Uint32Array; elements: number } {
  const scene: Scene = { vertices: [], normals: [], indices: [] };
  appendSurface(scene, 256, 144, false, false, 0, 0.92, (u, v) => {
    const x = (u * 2 - 1) * 19;
    const z = 3 + v * 40;
    return [x, terrainHeight(x, z), z];
  });

  appendSurface(scene, 192, 32, true, true, 1, 0.24, (u, v) => knotPoint(u, v, 0, 3.2, 11.0, 1.12, 0.3));
  appendSurface(scene, 160, 20, true, false, 2, 0.34, (u, v) => ribbonPoint(u, v, -5.1, 2.7, 11.8, 1.05, 3));
  appendSurface(scene, 128, 64, true, false, 3, 0.38, (u, v) => superquadricPoint(u, v, 5.2, 2.35, 12.2, 1.55, 2.25, 1.50, 0.38, 0.56, 0.7));
  appendSurface(scene, 128, 24, true, true, 5, 0.18, (u, v) => knotPoint(u, v, 0.3, 2.4, 18.0, 0.84, 2.1));

  const elements = 240;
  for (let i = 0; i < elements; i++) {
    const column = i % 12;
    const row = Math.trunc(i / 12);
    const x = (column - 5.5) * 2.85 + (noise(i, 17) - 0.5) * 0.65;
    const z = 16 + row * 1.35 + (noise(i, 31) - 0.5) * 0.55;
    const scale = 0.45 + noise(i, 47) * 0.55;
    const y = terrainHeight(x, z) + scale * 1.2;
    const material = 3 + i % 3;
    if (i % 3 === 0) appendSurface(scene, 32, 16, true, false, material, 0.38 + noise(i, 59) * 0.40, (u, v) => superquadricPoint(u, v, x, y, z, scale * 0.82, scale * 1.25, scale * 0.82, 0.35 + noise(i, 71) * 0.65, 0.35 + noise(i, 83) * 0.65, noise(i, 97) * 6));
    else if (i % 3 === 1) appendSurface(scene, 40, 10, true, true, material, 0.25 + noise(i, 109) * 0.45, (u, v) => knotPoint(u, v, x, y, z, scale * 0.30, noise(i, 127) * 6));
    else appendSurface(scene, 40, 8, true, false, material, 0.30 + noise(i, 149) * 0.45, (u, v) => ribbonPoint(u, v, x, y, z, scale * 0.48, 2 + i % 4));
  }
  return { vertices: new Float32Array(scene.vertices), normals: new Float32Array(scene.normals), indices: new Uint32Array(scene.indices), elements: elements + 4 };
}

function meshWorkload(gpu: Tach): Workload {
  const scene = createScene();
  const triangles = scene.indices.length / 3;
  if (triangles >= 0x000f_ffff) throw new RangeError("mesh exceeds packed triangle identity capacity");
  const pixelCount = frameWidth * frameHeight;
  const vertices = gpu.buffer(scene.vertices);
  const normals = gpu.buffer(scene.normals);
  const indices = gpu.buffer(scene.indices);
  const visibility = gpu.buffer(new Uint32Array(pixelCount));
  const coverage = gpu.buffer(new Uint32Array(triangles));
  const pixels = gpu.buffer(new Uint32Array(pixelCount));
  const params: MeshParams = { width: frameWidth, height: frameHeight, time: 1.7 };
  return {
    id: "mesh",
    category: "rendering",
    name: "Arbitrary mesh world",
    problem: `1920 x 1080 terrain and ${scene.elements.toLocaleString()} torus-knot, twisted-ribbon, superquadric, and organic elements (${triangles.toLocaleString()} triangles)`,
    dispatches: 5,
    command: meshWorld(vertices, normals, indices, visibility, coverage, pixels, params),
    buffers: [vertices, normals, indices, visibility, coverage, pixels],
    units: triangles,
    divisor: 1e6,
    throughputUnit: "million candidate fragments/s",
    details: { width: frameWidth, height: frameHeight, vertices: scene.vertices.length / 4, triangles, complexElements: scene.elements, meshFamilies: 4, smoothVertexNormals: true, perspectiveCorrectAttributes: true },
    frame: "mesh",
    readback: async () => {
      const [frame, fragments, owners] = await Promise.all([pixels.read(), coverage.read(), visibility.read()]);
      let candidates = 0;
      for (const count of fragments) candidates += count;
      let visiblePixels = 0;
      for (const owner of owners) if (owner !== 0xffff_ffff) visiblePixels++;
      const validation = validateFrame(frame);
      return {
        ...validation,
        units: candidates,
        details: {
          ...validation.details,
          candidateFragments: candidates,
          candidatesPerOutputPixel: candidates / pixelCount,
          candidatesPerVisiblePixel: candidates / visiblePixels,
          visiblePixels,
          visibleFramePercent: visiblePixels / pixelCount * 100,
        },
      };
    },
  };
}

function matrixValue(row: number, column: number, salt: number): number {
  return (((Math.imul(row + salt, 17) + Math.imul(column + salt, 13)) % 31) - 15) / 31;
}

function matrixWorkload(gpu: Tach): Workload {
  const size = 2048;
  const cells = size * size;
  const leftValues = new Float32Array(cells);
  const rightValues = new Float32Array(cells);
  for (let row = 0; row < size; row++) for (let column = 0; column < size; column++) {
    const index = row * size + column;
    leftValues[index] = matrixValue(row, column, 1);
    rightValues[index] = matrixValue(row, column, 7);
  }
  const left = gpu.buffer(leftValues);
  const right = gpu.buffer(rightValues);
  const output = gpu.buffer(new Float32Array(cells));
  const operations = 2 * size * size * size;
  return {
    id: "matrix",
    category: "mathematics",
    name: "Dense matrix algebra",
    problem: "2048 x 2048 tiled dense matrix multiplication",
    dispatches: 1,
    command: denseMatrixProduct(left, right, output, size),
    buffers: [left, right, output],
    units: operations,
    divisor: 1e9,
    throughputUnit: "GFLOP/s",
    details: { size, outputCells: cells, floatingPointOperations: operations, tile: "16 x 16" },
    readback: async () => {
      const result = await output.read();
      let maximumError = 0;
      for (const [row, column] of [[0, 0], [1, 31], [127, 509], [511, 1023], [1024, 257], [2047, 2047]]) {
        let expected = 0;
        for (let k = 0; k < size; k++) expected += Math.fround(matrixValue(row!, k, 1)) * Math.fround(matrixValue(k, column!, 7));
        maximumError = Math.max(maximumError, Math.abs(result[row! * size + column!]! - expected));
      }
      return { correct: maximumError < 0.01, check: `six reference cells; max error ${maximumError}`, details: { maximumSampleError: maximumError } };
    },
  };
}

function monteCarloWorkload(gpu: Tach): Workload {
  const paths = 1 << 20;
  const steps = 64;
  const rate = 0.03;
  const volatility = 0.2;
  const dt = 1 / steps;
  const payoffs = gpu.buffer(new Float32Array(paths));
  const params: MonteCarloParams = {
    steps,
    drift: (rate - volatility * volatility * 0.5) * dt,
    diffusion: volatility * Math.sqrt(dt),
    discount: Math.exp(-rate),
    strike: 100,
  };
  return {
    id: "monte-carlo",
    category: "mathematics",
    name: "Monte Carlo integration",
    problem: `${paths.toLocaleString()} geometric Brownian paths x ${steps} Gaussian time steps`,
    dispatches: 1,
    command: monteCarloPaths(payoffs, params),
    buffers: [payoffs],
    units: paths * steps,
    divisor: 1e6,
    throughputUnit: "million path-steps/s",
    details: { paths, steps, gaussianDraws: paths * steps, model: "discounted European call" },
    readback: async () => {
      const result = await payoffs.read();
      let sum = 0;
      let positive = 0;
      for (const payoff of result) {
        if (!Number.isFinite(payoff) || payoff < 0) return { correct: false, check: "non-finite or negative payoff" };
        sum += payoff;
        if (payoff > 0) positive++;
      }
      const mean = sum / result.length;
      const positivePercent = positive / result.length * 100;
      const correct = mean > 8 && mean < 11 && positivePercent > 50 && positivePercent < 54;
      return { correct, check: correct ? "payoff distribution matches model bounds" : "payoff distribution escaped model bounds", details: { meanPayoff: mean, positivePathPercent: positivePercent } };
    },
  };
}

function particleWorkload(gpu: Tach): Workload {
  const count = 1 << 21;
  const substeps = 64;
  const positionsData = new Float32Array(count * 4);
  const velocitiesData = new Float32Array(count * 4);
  for (let i = 0; i < count; i++) {
    const offset = i * 4;
    const x = (noise(i, 307) * 2 - 1) * 8;
    const y = (noise(i, 331) * 2 - 1) * 8;
    const z = (noise(i, 353) * 2 - 1) * 8;
    positionsData[offset] = x; positionsData[offset + 1] = y; positionsData[offset + 2] = z; positionsData[offset + 3] = 1;
    velocitiesData[offset] = -y * 0.025; velocitiesData[offset + 1] = x * 0.025;
    velocitiesData[offset + 2] = (noise(i, 379) * 2 - 1) * 0.08; velocitiesData[offset + 3] = noise(i, 401);
  }
  const positions = gpu.buffer(positionsData);
  const velocities = gpu.buffer(velocitiesData);
  const params: ParticleParams = { dt: 0.006, time: 1.7, softening: 0.18 };
  return {
    id: "particles",
    category: "physics",
    name: "Multi-attractor particles",
    problem: `${count.toLocaleString()} particles x ${substeps} integration steps x 4 attractors`,
    dispatches: 1,
    command: particleDynamics(positions, velocities, params, { repeat: substeps }),
    buffers: [positions, velocities],
    units: count * substeps * 4,
    divisor: 1e9,
    throughputUnit: "billion force interactions/s",
    details: { particles: count, substeps, attractors: 4, stateBytes: positionsData.byteLength + velocitiesData.byteLength },
    readback: async () => {
      const result = await positions.read();
      const stride = Math.trunc(count / 8192) * 4;
      let radiusSum = 0;
      let maximumRadius = 0;
      let checked = 0;
      for (let offset = 0; offset < result.length; offset += stride) {
        const x = result[offset]!; const y = result[offset + 1]!; const z = result[offset + 2]!;
        if (!Number.isFinite(x + y + z)) return { correct: false, check: "non-finite particle state" };
        const radius = Math.sqrt(x * x + y * y + z * z);
        radiusSum += radius;
        maximumRadius = Math.max(maximumRadius, radius);
        checked++;
      }
      const correct = maximumRadius <= Math.sqrt(3) * 12.001 && radiusSum > 0;
      return { correct, check: correct ? "sampled particle state is finite and bounded" : "particle escaped simulation bounds", details: { sampledParticles: checked, meanRadius: radiusSum / checked, maximumRadius } };
    },
  };
}

function waveWorkload(gpu: Tach): Workload {
  const width = 2048;
  const height = 2048;
  const cells = width * height;
  const pairs = 32;
  const firstData = new Float32Array(cells * 2);
  for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) {
    const nx = x / width * 2 - 1;
    const ny = y / height * 2 - 1;
    const firstRadius = (nx + 0.28) * (nx + 0.28) + (ny - 0.12) * (ny - 0.12);
    const secondRadius = (nx - 0.34) * (nx - 0.34) + (ny + 0.22) * (ny + 0.22);
    firstData[(y * width + x) * 2] = Math.exp(-firstRadius * 180) - Math.exp(-secondRadius * 240) * 0.72;
  }
  const first = gpu.buffer(firstData);
  const second = gpu.buffer(firstData.slice());
  const params: WaveParams = { width, height, dt: 0.12, stiffness: 2.8, damping: 0.9985 };
  return {
    id: "wave",
    category: "physics",
    name: "Wave-field simulation",
    problem: `${width} x ${height} height-velocity field x ${pairs * 2} stencil steps`,
    dispatches: pairs * 2,
    command: waveSimulation(first, second, params, { repeat: pairs }),
    buffers: [first, second],
    units: cells * pairs * 2,
    divisor: 1e9,
    throughputUnit: "billion cell updates/s",
    details: { width, height, cells, steps: pairs * 2, neighborReadsPerStep: 4, stateBytes: firstData.byteLength * 2 },
    readback: async () => {
      const result = await first.read();
      const stride = Math.trunc(cells / 8192) * 2;
      let maximumAmplitude = 0;
      let energy = 0;
      let checked = 0;
      for (let offset = 0; offset < result.length; offset += stride) {
        const heightValue = result[offset]!;
        const velocity = result[offset + 1]!;
        if (!Number.isFinite(heightValue + velocity)) return { correct: false, check: "non-finite wave state" };
        maximumAmplitude = Math.max(maximumAmplitude, Math.abs(heightValue));
        energy += heightValue * heightValue + velocity * velocity;
        checked++;
      }
      const correct = maximumAmplitude > 0.001 && maximumAmplitude < 2 && energy > 0;
      return { correct, check: correct ? "sampled wave remains finite, active, and stable" : "wave stability bounds failed", details: { sampledCells: checked, maximumAmplitude, sampledEnergy: energy } };
    },
  };
}

async function execute(gpu: Tach, workload: Workload): Promise<BenchmarkResult> {
  try {
    await complete(gpu, workload.command);
    const gpuSamplesMs: number[] = [];
    for (let sample = 0; sample < samples; sample++) gpuSamplesMs.push(await timed(gpu, workload.command));
    const gpuMs = median(gpuSamplesMs);
    const readback = await workload.readback();
    if (workload.frame && readback.frame) renderFrame(workload.frame, readback.frame);
    const units = readback.units ?? workload.units;
    return {
      id: workload.id,
      category: workload.category,
      name: workload.name,
      problem: workload.problem,
      dispatches: workload.dispatches,
      gpuSamplesMs,
      gpuMs,
      throughput: units / (gpuMs / 1000) / workload.divisor,
      throughputUnit: workload.throughputUnit,
      framesPerSecond: workload.frame ? 1000 / gpuMs : null,
      correct: readback.correct,
      check: readback.check,
      details: { ...workload.details, ...readback.details },
    };
  } finally {
    for (const buffer of workload.buffers) buffer.destroy();
  }
}

export async function runBenchmarks(gpu: Tach): Promise<BenchmarkReport> {
  const factories = [proceduralWorkload, meshWorkload, matrixWorkload, monteCarloWorkload, particleWorkload, waveWorkload] as const;
  const results: BenchmarkResult[] = [];
  for (const factory of factories) results.push(await execute(gpu, factory(gpu)));
  return {
    samples,
    timing: "warm pipeline; gpu.submit(command) through gpu.idle(); readback and validation excluded",
    results,
  };
}
