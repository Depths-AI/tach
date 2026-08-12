import { expect, test } from "@playwright/test";

function expectedBitwise() {
  const value = 0xff00;
  const left = (value << (40 & 31)) >>> 0;
  const logical = value >>> (36 & 31);
  const arithmetic = (-64 >> (35 & 31)) >>> 0;
  let mixed = ((left | logical) ^ ~value) >>> 0;
  mixed = (mixed & 0xffff) >>> 0;
  mixed = (mixed << (33 & 31)) >>> 0;
  return (mixed | arithmetic) >>> 0;
}

function expectedMath(index) {
  const a = [index + 1, 2, 3];
  const length = Math.hypot(...a);
  const normalized = a.map((value) => value / length);
  const cross = [-3, 0, a[0]];
  const distance = Math.hypot(a[0] - cross[0], a[1] - cross[1], a[2] - cross[2]);
  const wave = Math.sin(length) + Math.cos(length) + Math.tan(0.25);
  const shaped = Math.sqrt(Math.abs(wave)) + 1 / Math.sqrt(length + 1);
  const expo = 2 ** Math.log2(length + 1) + Math.exp(Math.log(length + 1));
  const powered = (length + 1) ** 2;
  const rounded = Math.floor(powered) + Math.ceil(distance) + Math.trunc(shaped);
  const bounded = Math.max(1, Math.min(Math.min(index, 1024), Math.max(index, 1)));
  return [...normalized, shaped + expo + rounded + bounded];
}

const controlInput = Array.from({ length: 64 }, (_, index) => index + 1);
const forInput = Array(256).fill(0);
forInput.splice(0, 4, 1, 2, 3, 4);

const cases = [
  {
    name: "atomics",
		program: "accumulate",
    resources: { counters: { total: 0 } },
    readParam: "counters",
    assert(value) { expect(value.total).toBe(64); },
  },
  {
    name: "bitwise",
		program: "bitwise",
    resources: { out: Array(8).fill(0) },
    readParam: "out",
    assert(value) { expect(value).toEqual(Array(8).fill(expectedBitwise())); },
  },
  {
    name: "control",
		program: "transform",
    resources: { data: controlInput, params: { scale: 2, count: 64, enabled: true } },
    readParam: "data",
    assert(value) {
      const expected = controlInput.map((item) => item > 50 ? 100 : item * 2 + 1);
      expect(value).toEqual(expected);
    },
  },
  {
    name: "for",
		program: "reduceLanes",
    resources: { data: forInput },
    readParam: "data",
    assert(value) {
      const expected = Array(256).fill(0);
      expected[0] = 10;
      expect(value).toEqual(expected);
    },
  },
	{
		name: "fusion",
		program: "transform",
		resources: { input: [1, 2, 3, 4], output: [0, 0, 0, 0], factor: 2, bias: 1 },
		readParam: "output",
		assert(value) { expect(value).toEqual([3, 5, 7, 9]); },
	},
	{
		name: "fusion",
		program: "neighbor",
		resources: { values: [1, 2, 3, 4] },
		readParam: "values",
		assert(value) { expect(value).toEqual([5, 7, 9, 5]); },
	},
  {
    name: "math",
		program: "math",
    resources: { out: Array.from({ length: 4 }, () => [0, 0, 0, 0]) },
    readParam: "out",
    assert(value) {
      const actual = value.flat();
      const expected = Array.from({ length: 4 }, (_, index) => expectedMath(index)).flat();
      expect(actual).toHaveLength(expected.length);
      for (let index = 0; index < actual.length; index++) {
        expect(actual[index], `math component ${index}`).toBeCloseTo(expected[index], 3);
      }
    },
  },
  {
    name: "particles",
		program: "integrate",
    resources: {
      particles: [
        { position: [1, 2, 3, 4], velocity: [2, 4, 6, 8] },
        { position: [-1, -2, -3, -4], velocity: [1, 2, 3, 4] },
      ],
      params: { dt: 0.5, count: 2 },
    },
    readParam: "particles",
    assert(value) {
      expect(value).toEqual([
        { position: [2, 4, 6, 8], velocity: [2, 4, 6, 8] },
        { position: [-0.5, -1, -1.5, -2], velocity: [1, 2, 3, 4] },
      ]);
    },
  },
  {
    name: "scalars",
		program: "scale",
    resources: { data: [1, 2, 3, 4], factor: 2.5 },
    readParam: "data",
    assert(value) { expect(value).toEqual([2.5, 5, 7.5, 10]); },
  },
];

async function executeCase(testCase) {
  if (!("gpu" in navigator)) return { available: false, reason: "navigator.gpu is unavailable" };
  const { TachError, tach } = await import("@depths/tach");
  const uncapturedErrors = [];
  try {
    const executed = await tach(async (gpu) => {
      const onUncaptured = (event) => uncapturedErrors.push(event.error.message);
      gpu.device.addEventListener("uncapturederror", onUncaptured);
      try {
        const generated = await import(`/build/${testCase.name}.js`);
        const metadata = await (await fetch(`/build/${testCase.name}.tach.json`)).json();
        const wgsl = await (await fetch(`/build/${testCase.name}.wgsl`)).text();
        const diagnosticModule = gpu.device.createShaderModule({
          label: `Tach ${testCase.name} diagnostics`,
          code: wgsl,
        });
        const compilation = await diagnosticModule.getCompilationInfo();
        const compilationErrors = compilation.messages
          .filter((message) => message.type === "error")
          .map((message) => `${message.lineNum}:${message.linePos} ${message.message}`);

				const program = metadata.programs.find((item) => item.name === testCase.program);
				if (!program) throw new Error(`program ${testCase.program} is missing`);
        const args = [];
        const computeBuffers = new Map();
				for (const parameter of program.parameters) {
          const value = testCase.resources[parameter.name];
          if (parameter.kind === "buffer") {
            const computeBuffer = gpu.buffer(value);
            computeBuffers.set(parameter.name, computeBuffer);
            args.push(computeBuffer);
          } else {
            args.push(value);
          }
        }

        const output = computeBuffers.get(testCase.readParam);
        if (!output) throw new Error(`readback resource ${testCase.readParam} is missing`);
				await gpu.submit(generated[testCase.program](...args));
        const value = await output.read();

        const info = gpu.adapter.info ?? {};
        const identityLooksSoftware = /swiftshader|software|llvmpipe|lavapipe|softpipe|warp|basic render/i.test(
          [info.description, info.vendor, info.architecture, info.device].filter(Boolean).join(" "),
        );
        const software = info.isFallbackAdapter === true || identityLooksSoftware;
        return {
          adapter: {
            architecture: info.architecture ?? "",
            description: info.description ?? "",
            device: info.device ?? "",
            isFallbackAdapter: info.isFallbackAdapter ?? null,
            mode: software ? "software-emulated" : "hardware-accelerated",
            vendor: info.vendor ?? "",
          },
          value,
          compilationErrors,
					plan: (() => {
						const index = metadata.programs.indexOf(program);
						const plan = metadata.targets.web.programs[index];
						return {
							dispatches: plan.steps.filter((step) => step.kind === "dispatch").length,
							transients: plan.transients.length,
						};
					})(),
        };
      } finally {
        gpu.device.removeEventListener("uncapturederror", onUncaptured);
      }
    });
    return { available: true, ...executed, uncapturedErrors };
  } catch (error) {
    if (error instanceof TachError && (error.code === "webgpu-unavailable" || error.code === "adapter-unavailable")) {
      return { available: false, reason: error.message };
    }
    return {
      available: true,
      error: error instanceof TachError ? `[${error.code}] ${error.message}` : String(error),
      uncapturedErrors,
    };
  }
}

test.describe.configure({ mode: "serial" });
for (const testCase of cases) {
	test(`${testCase.name}/${testCase.program} executes and returns expected typed data`, async ({ page }, testInfo) => {
    await page.goto("/");
    const payload = { ...testCase };
    delete payload.assert;
    const result = await page.evaluate(executeCase, payload);

    if (!result.available) {
      throw new Error(`The unified harness could not obtain a hardware or software WebGPU adapter: ${result.reason}`);
    }

    expect(result.error).toBeUndefined();
    expect(result.compilationErrors).toEqual([]);
    expect(result.uncapturedErrors).toEqual([]);
    const adapterLabel = [result.adapter.description, result.adapter.vendor, result.adapter.architecture]
      .filter(Boolean)
      .join(" / ") || "WebGPU adapter";
    testInfo.annotations.push({ type: "WebGPU mode", description: result.adapter.mode });
    testInfo.annotations.push({ type: "WebGPU adapter", description: adapterLabel });
    if (testCase.name === "atomics") {
      console.log(`WebGPU execution: ${result.adapter.mode} (${adapterLabel})`);
    }
    testCase.assert(result.value);
		if (testCase.name === "fusion" && testCase.program === "transform") {
			expect(result.plan).toEqual({ dispatches: 1, transients: 0 });
		}
		if (testCase.name === "fusion" && testCase.program === "neighbor") {
			expect(result.plan.dispatches).toBe(2);
		}
  });
}

test("parameter commands remain distinct within a submission and across frames", async ({ page }) => {
  await page.goto("/");
  const result = await page.evaluate(async () => {
    const { tach } = await import("@depths/tach");
    const { scale } = await import("/build/scalars.js");
    return tach(async (gpu) => {
      const data = gpu.buffer(new Float32Array([1, 2, 3, 4]));
      await gpu.submit(scale(data, 2), scale(data, 3));
      await gpu.submit(scale(data, 4));
      return data.read();
    });
  });

  expect(Array.from(result)).toEqual([24, 48, 72, 96]);
});
