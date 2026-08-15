import { tach } from "@depths/tach";
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
    cross = [-3, 0, a[0]!];
  const distance = Math.hypot(
    a[0]! - cross[0]!,
    a[1]! - cross[1]!,
    a[2]! - cross[2]!,
  );
  const wave = Math.sin(length) + Math.cos(length) + Math.tan(0.25),
    shaped = Math.sqrt(Math.abs(wave)) + 1 / Math.sqrt(length + 1);
  const final = shaped + 2 ** Math.log2(length + 1) +
    Math.exp(Math.log(length + 1)) + Math.floor((length + 1) ** 2) +
    Math.ceil(distance) + Math.trunc(shaped) +
    Math.max(1, Math.min(Math.min(index, 1024), Math.max(index, 1)));
  return [...normalized, final];
}

const control = Array.from({ length: 64 }, (_, index) => index + 1),
  lanes = Array<number>(256).fill(0);
lanes.splice(0, 4, 1, 2, 3, 4);

function execute(): Promise<string> {
  return tach(async (gpu) => {
    const counters = gpu.buffer({ total: 0 });
    await gpu.submit(programs.accumulate(counters));
    equal(await counters.read(), { total: 64 }, "atomics");

    const bits = gpu.buffer(Array<number>(8).fill(0));
    await gpu.submit(programs.bitwise(bits));
    equal(
      await bits.read(),
      Array<number>(8).fill(expectedBitwise()),
      "bitwise",
    );

    const transformed = gpu.buffer(control);
    await gpu.submit(
      programs.transform(transformed, { scale: 2, count: 64, enabled: true }),
    );
    equal(
      await transformed.read(),
      control.map((value) => value > 50 ? 100 : value * 2 + 1),
      "control flow",
    );

    const reduced = gpu.buffer(lanes);
    await gpu.submit(programs.reduceLanes(reduced));
    const expectedLanes = Array<number>(256).fill(0);
    expectedLanes[0] = 10;
    equal(await reduced.read(), expectedLanes, "for loop");

    const math = gpu.buffer(
      Array.from({ length: 4 }, () => [0, 0, 0, 0] as const),
    );
    await gpu.submit(programs.math(math));
    const actualMath = (await math.read()).flat(),
      wantedMath = Array.from({ length: 4 }, (_, index) => expectedMath(index))
        .flat();
    actualMath.forEach((value, index) => {
      if (Math.abs(value - wantedMath[index]!) > 0.0005) {
        throw new Error(`math[${index}]: ${value} != ${wantedMath[index]}`);
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
    ], "particles");

    const values = gpu.buffer(new Float32Array([1, 2, 3, 4]));
    await gpu.submit(programs.scale(values, 2), programs.scale(values, 3));
    await gpu.submit(programs.scale(values, 4));
    equal(
      Array.from(await values.read()),
      [24, 48, 72, 96],
      "batched parameters",
    );

    return gpu.adapter.name;
  });
}

const adapter = await execute();
equal(await execute(), adapter, "sequential session adapter");
console.log(`Vulkan execution: ${adapter}`);
