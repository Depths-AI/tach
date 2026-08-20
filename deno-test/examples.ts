import { type Tach, tach } from "@depths/tach";
import * as programs from "#kernels";

function equal(actual: unknown, expected: unknown, label: string): void {
  if (JSON.stringify(actual) !== JSON.stringify(expected)) {
    throw new Error(
      `${label}: ${JSON.stringify(actual)} != ${JSON.stringify(expected)}`,
    );
  }
}

function expectedBitwise(): number {
  const value = 0xff00;
  let mixed =
    ((((value << (40 & 31)) >>> 0) | (value >>> (36 & 31))) ^ ~value) >>> 0;
  mixed = ((mixed & 0xffff) << (33 & 31)) >>> 0;
  return (mixed | ((-64 >> (35 & 31)) >>> 0)) >>> 0;
}

function expectedMath(index: number): number[] {
  const a = [index + 1, 2, 3],
    length = Math.hypot(...a),
    normalized = a.map((value) => value / length),
    cross = [-3, 0, a[0]!],
    distance = Math.hypot(
      a[0]! - cross[0]!,
      a[1]! - cross[1]!,
      a[2]! - cross[2]!,
    ),
    wave = Math.sin(length) + Math.cos(length) + Math.tan(0.25),
    shaped = Math.sqrt(Math.abs(wave)) + 1 / Math.sqrt(length + 1);
  return [
    ...normalized,
    shaped + 2 ** Math.log2(length + 1) + Math.exp(Math.log(length + 1)) +
    Math.floor((length + 1) ** 2) + Math.ceil(distance) + Math.trunc(shaped) +
    Math.max(1, Math.min(Math.min(index, 1024), Math.max(index, 1))),
  ];
}

function expectedFloat16Math(index: number): number[] {
  const x = index + 1,
    value = [x, 2, 3],
    length = Math.hypot(...value),
    unit = value.map((lane) => lane / length),
    distance = Math.hypot(...value.map((lane, i) => lane - unit[i]!)),
    crossX = value[1]! * unit[2]! - value[2]! * unit[1]!,
    geometry = value.reduce((sum, lane, i) => sum + lane * unit[i]!, 0) +
      length + distance + crossX,
    wave = Math.sin(x) + Math.cos(x) + Math.tan(0.25),
    shaped = Math.sqrt(Math.abs(wave)) + 1 / Math.sqrt(x + 1),
    exponential = 2 ** Math.log2(x + 1) + Math.exp(Math.log(x + 1)),
    rounded = Math.floor((x + 1) ** 2) + Math.ceil(geometry) +
      Math.trunc(shaped);
  return [...unit, shaped + exponential + rounded];
}

function expectedContextualMath(index: number, scale: number): number[] {
  const value = [index + 1, 2, 3], length = Math.hypot(...value);
  const wave = Math.sin(1);
  return [
    value[0]! / length * scale + 1 + wave,
    value[1]! / length * scale + 2 + wave,
    value[2]! / length * scale + 3 + wave,
    wave + 2 ** 3 + 32,
  ];
}

async function verifyLanguage(gpu: Tach): Promise<void> {
  const counters = gpu.buffer({ total: 0 }),
    accumulation = programs.accumulate(counters);
  await gpu.prepare(accumulation);
  await gpu.submit(accumulation);
  equal(await counters.read(), { total: 64 }, "atomics");

  const bits = gpu.buffer(Array<number>(8).fill(0));
  await gpu.submit(programs.bitwise(bits));
  equal(await bits.read(), Array<number>(8).fill(expectedBitwise()), "bitwise");

  const source = Array.from({ length: 64 }, (_, index) => index + 1),
    transformed = gpu.buffer(source);
  await gpu.submit(
    programs.transform(transformed, { scale: 2, count: 64, enabled: true }),
  );
  equal(
    await transformed.read(),
    source.map((value) => value > 50 ? 100 : value * 2 + 1),
    "control flow",
  );

  const scan = gpu.buffer([0, 0, 0, 0]),
    scanInput = gpu.buffer([1, 2, 3, 4, 0, 6, 7, 8, 600, 10, 11, 12]);
  await gpu.submit(programs.selectiveScan(scan, scanInput, {
    stride: 4,
    count: 12,
    scale: 2,
    threshold: 1000,
  }));
  equal(
    await scan.read(),
    [1202, 36, 42, 48],
    "break, continue, and float32 fma",
  );

  const affine = gpu.buffer(new Float16Array([1, 2, 3, 4, 5, 6, 7, 8]));
  await gpu.submit(programs.affineFloat16(affine, [2, 3, 4, 5], [1, 1, 1, 1]));
  equal(
    Array.from(await affine.read()),
    [3, 7, 13, 21, 11, 19, 29, 41],
    "float16 vector fma",
  );

  const contextual = gpu.buffer(
    Array.from({ length: 4 }, () => [0, 0, 0, 0] as const),
  );
  await gpu.submit(programs.contextualMath(contextual, 2));
  (await contextual.read()).flat().forEach((value, index) => {
    const expected = expectedContextualMath(
      Math.floor(index / 4),
      2,
    )[index % 4]!;
    if (Math.abs(value - expected) > 0.0001) {
      throw new Error(
        `contextual inference[${index}]: ${value} != ${expected}`,
      );
    }
  });

  const lanes = Array<number>(256).fill(0);
  lanes.splice(0, 4, 1, 2, 3, 4);
  const reduced = gpu.buffer(lanes);
  await gpu.submit(programs.reduceLanes(reduced));
  const expectedLanes = Array<number>(256).fill(0);
  expectedLanes[0] = 10;
  equal(await reduced.read(), expectedLanes, "workgroup reduction");

  const math = gpu.buffer(
    Array.from({ length: 4 }, () => [0, 0, 0, 0] as const),
  );
  await gpu.submit(programs.math(math));
  const actual = (await math.read()).flat(),
    expected = Array.from({ length: 4 }, (_, index) => expectedMath(index))
      .flat();
  actual.forEach((value, index) => {
    if (Math.abs(value - expected[index]!) > 0.0005) {
      throw new Error(`math[${index}]: ${value} != ${expected[index]}`);
    }
  });

  const particles = gpu.buffer([
    { position: [1, 2, 3, 4] as const, velocity: [2, 4, 6, 8] as const },
    { position: [-1, -2, -3, -4] as const, velocity: [1, 2, 3, 4] as const },
  ]);
  await gpu.submit(programs.integrate(particles, { dt: 0.5, count: 2 }));
  equal(await particles.read(), [
    { position: [2, 4, 6, 8], velocity: [2, 4, 6, 8] },
    { position: [-0.5, -1, -1.5, -2], velocity: [1, 2, 3, 4] },
  ], "imported struct orchestration");

  const values = gpu.buffer(new Float32Array([1, 2, 3, 4]));
  await gpu.submit(programs.scale(values, 2), programs.scale(values, 3));
  await gpu.submit(programs.scale(values, 4));
  equal(Array.from(await values.read()), [24, 48, 72, 96], "parameter arena");

  const halves = gpu.buffer(new Float16Array([1, 2, 3, 4]));
  await gpu.submit(programs.scaleFloat16(halves, 0.5));
  equal(Array.from(await halves.read()), [0.5, 1, 1.5, 2], "float16");
  const singleHalf = gpu.buffer(new Float16Array([2]));
  await gpu.submit(programs.scaleFloat16(singleHalf, 0.5));
  equal(Array.from(await singleHalf.read()), [1], "odd-sized float16 buffer");
  await gpu.submit(programs.halveFloat16(singleHalf));
  equal(Array.from(await singleHalf.read()), [0.5], "float16 plan constant");

  const halfInput = gpu.buffer({
    offset: 0,
    values: new Float16Array([4, 8, 12, 16]),
  });
  const halfMath = gpu.buffer(new Float16Array(16));
  await gpu.submit(programs.float16Math(halfInput, halfMath));
  Array.from(await halfMath.read()).forEach((value, index) => {
    const expected = expectedFloat16Math(Math.floor(index / 4))[index % 4]!;
    if (Math.abs(value - expected) > (index % 4 === 3 ? 1 : 0.01)) {
      throw new Error(`float16 math[${index}]: ${value} != ${expected}`);
    }
  });
}

const width = 128, height = 72;
const reusableView = programs.gradient({ width, height, bias: 0 });

async function verifyViews(gpu: Tach): Promise<void> {
  await gpu.prepare(reusableView);
  await gpu.submit(reusableView);
  await gpu.submit(programs.swatch());
  const swatch = gpu.buffer(new Float32Array(16));
  await gpu.submit(programs.swatchInto(swatch));
  equal(Array.from(await swatch.read()), [
    0,
    0,
    0,
    1,
    1,
    0,
    0,
    1,
    0,
    1,
    0,
    1,
    1,
    1,
    0,
    1,
  ], "external swatch source");
  const pixels = gpu.buffer(new Float32Array(width * height * 4));
  const [first, ...rest] = Array.from(
    { length: 32 },
    (_, frame) =>
      frame % 2 === 0
        ? programs.gradient({ width, height, bias: frame / 64 })
        : programs.gradientInto(pixels, {
          width,
          height,
          bias: frame / 64,
        }),
  );
  await gpu.submit(first!, ...rest);
  await gpu.submit(
    programs.gradientInto(pixels, { width, height, bias: 0.25 }),
  );
  const written = await pixels.read();
  if (written[0] !== 0.25 || written[3] !== 1) {
    throw new Error("external view source was not written");
  }
}

const first = await tach(async (gpu) => {
  equal(Object.keys(programs).sort(), [
    "accumulate",
    "affineFloat16",
    "bitwise",
    "contextualMath",
    "float16Math",
    "gradient",
    "gradientInto",
    "halveFloat16",
    "integrate",
    "math",
    "reduceLanes",
    "scale",
    "scaleFloat16",
    "selectiveScan",
    "swatch",
    "swatchInto",
    "transform",
  ], "public programs");
  await verifyLanguage(gpu);
  await verifyViews(gpu);
  return gpu.adapter.name;
});
const second = await tach(async (gpu) => {
  await verifyViews(gpu);
  return gpu.adapter.name;
});
equal(second, first, "owner-neutral scalar view across sessions");
console.log(`Vulkan execution: ${first}; 17 programs; 72 projected frames`);
