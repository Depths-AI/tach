/// <reference lib="dom" />

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
}

const width = 128, height = 72;
const reusableView = programs.gradient({ width, height, bias: 0 });

const run = tach(async (gpu) => {
  equal(Object.keys(programs).sort(), [
    "accumulate",
    "bitwise",
    "gradient",
    "gradientInto",
    "integrate",
    "math",
    "reduceLanes",
    "scale",
    "transform",
  ], "public programs");
  await verifyLanguage(gpu);

  await gpu.prepare(reusableView);
  await gpu.submit(reusableView);
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  document.body.append(canvas);
  const pixels = gpu.buffer(new Float32Array(width * height * 4));
  await gpu.present(
    canvas,
    programs.gradientInto(pixels, { width, height, bias: 0.25 }),
  );
  const written = await pixels.read();
  if (written[0] !== 0.25 || written[3] !== 1) {
    throw new Error("external view source was not written");
  }
  const views = Array.from(
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
  await Promise.all(views.map((view) => gpu.present(canvas, view)));
  const image = await new Promise<Blob>((resolve, reject) =>
    canvas.toBlob(
      (blob) => blob ? resolve(blob) : reject(new Error("PNG capture failed")),
      "image/png",
    )
  );
  if (image.size < 100) throw new Error("presented frame is empty");
  return {
    adapter: gpu.adapter,
    programs: Object.keys(programs).length,
    presentedFrames: views.length + 1,
    pngBytes: image.size,
  };
});

Object.assign(globalThis, { __tachTest: run });
