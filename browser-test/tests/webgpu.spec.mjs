import { expect, test } from "@playwright/test";

function decode(bytes, kind) {
  const data = Uint8Array.from(bytes);
  const view = new DataView(data.buffer, data.byteOffset, data.byteLength);
  const values = [];
  for (let offset = 0; offset < data.byteLength; offset += 4) {
    values.push(kind === "f32" ? view.getFloat32(offset, true) : view.getUint32(offset, true));
  }
  return values;
}

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
    kernel: "accumulate",
    workgroups: 1,
    resources: { counters: { total: 0 } },
    readParam: "counters",
    assert(bytes) { expect(decode(bytes, "u32")[0]).toBe(64); },
  },
  {
    name: "bitwise",
    kernel: "bitwise",
    workgroups: 1,
    resources: { out: Array(8).fill(0) },
    readParam: "out",
    assert(bytes) { expect(decode(bytes, "u32")).toEqual(Array(8).fill(expectedBitwise())); },
  },
  {
    name: "control",
    kernel: "transform",
    workgroups: 1,
    resources: { data: controlInput, params: { scale: 2, count: 64 } },
    readParam: "data",
    assert(bytes) {
      const expected = controlInput.map((value) => value > 50 ? 100 : value * 2 + 1);
      expect(decode(bytes, "f32")).toEqual(expected);
    },
  },
  {
    name: "for",
    kernel: "reduceLanes",
    workgroups: 1,
    resources: { data: forInput },
    readParam: "data",
    assert(bytes) {
      const expected = Array(256).fill(0);
      expected[0] = 10;
      expect(decode(bytes, "u32")).toEqual(expected);
    },
  },
  {
    name: "math",
    kernel: "math",
    workgroups: 1,
    resources: { out: Array.from({ length: 4 }, () => [0, 0, 0, 0]) },
    readParam: "out",
    assert(bytes) {
      const actual = decode(bytes, "f32");
      const expected = Array.from({ length: 4 }, (_, index) => expectedMath(index)).flat();
      expect(actual).toHaveLength(expected.length);
      for (let index = 0; index < actual.length; index++) {
        expect(actual[index], `math component ${index}`).toBeCloseTo(expected[index], 3);
      }
    },
  },
  {
    name: "particles",
    kernel: "integrate",
    workgroups: 1,
    resources: {
      particles: [
        { position: [1, 2, 3, 4], velocity: [2, 4, 6, 8] },
        { position: [-1, -2, -3, -4], velocity: [1, 2, 3, 4] },
      ],
      params: { dt: 0.5, count: 2 },
    },
    readParam: "particles",
    assert(bytes) {
      expect(decode(bytes, "f32")).toEqual([
        2, 4, 6, 8, 2, 4, 6, 8,
        -0.5, -1, -1.5, -2, 1, 2, 3, 4,
      ]);
    },
  },
  {
    name: "scalars",
    kernel: "scale",
    workgroups: 1,
    resources: { data: [1, 2, 3, 4], factor: 2.5 },
    readParam: "data",
    assert(bytes) { expect(decode(bytes, "f32")).toEqual([2.5, 5, 7.5, 10]); },
  },
];

async function executeCase(testCase) {
  if (!("gpu" in navigator)) return { available: false, reason: "navigator.gpu is unavailable" };
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) return { available: false, reason: "navigator.gpu.requestAdapter() returned null" };

  let device;
  const uncapturedErrors = [];
  try {
    device = await adapter.requestDevice();
    device.addEventListener("uncapturederror", (event) => uncapturedErrors.push(event.error.message));
    device.pushErrorScope("validation");

    const generated = await import(`/build/${testCase.name}.js`);
    const program = generated.createTachProgram(device);
    const compilation = await program.shaderModule.getCompilationInfo();
    const compilationErrors = compilation.messages
      .filter((message) => message.type === "error")
      .map((message) => `${message.lineNum}:${message.linePos} ${message.message}`);

    const kernelInfo = generated.metadata.kernels.find((kernel) => kernel.name === testCase.kernel);
    if (!kernelInfo) throw new Error(`kernel ${testCase.kernel} is missing`);
    const resources = {};
    const sizes = {};
    const buffers = [];
    for (const parameter of kernelInfo.resources) {
      const resource = generated.metadata.resources[parameter.resource];
      const suffix = `${resource.name}_g${resource.group}_b${resource.binding}`;
      const value = testCase.resources[parameter.param];
      const bytes = generated[`pack_${suffix}`](value);
      sizes[parameter.param] = Math.max(4, (bytes.byteLength + 3) & ~3);
      const buffer = generated[`create_${suffix}`](device, value);
      resources[parameter.param] = buffer;
      buffers.push(buffer);
    }

    const output = resources[testCase.readParam];
    const outputSize = sizes[testCase.readParam];
    if (!output || !outputSize) throw new Error(`readback resource ${testCase.readParam} is missing`);
    const readback = device.createBuffer({
      label: `Tach ${testCase.name} readback`,
      size: outputSize,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    const encoder = device.createCommandEncoder({ label: `Tach ${testCase.name} commands` });
    program.kernels[testCase.kernel].dispatch(encoder, resources, testCase.workgroups);
    encoder.copyBufferToBuffer(output, 0, readback, 0, outputSize);
    device.queue.submit([encoder.finish()]);

    const validationError = await device.popErrorScope();
    await device.queue.onSubmittedWorkDone();
    await readback.mapAsync(GPUMapMode.READ);
    const bytes = Array.from(new Uint8Array(readback.getMappedRange()).slice());
    readback.unmap();
    readback.destroy();
    for (const buffer of buffers) buffer.destroy();

    const info = adapter.info ?? {};
    const identityLooksSoftware = /swiftshader|software|llvmpipe|warp/i.test(
      [info.description, info.vendor, info.architecture, info.device].filter(Boolean).join(" "),
    );
    const software = info.isFallbackAdapter === true || identityLooksSoftware;
    return {
      available: true,
      adapter: {
        architecture: info.architecture ?? "",
        description: info.description ?? "",
        device: info.device ?? "",
        isFallbackAdapter: info.isFallbackAdapter ?? null,
        mode: software ? "software-emulated" : "hardware-accelerated",
        vendor: info.vendor ?? "",
      },
      bytes,
      compilationErrors,
      uncapturedErrors,
      validationError: validationError?.message ?? null,
    };
  } catch (error) {
    return { available: true, error: error instanceof Error ? error.stack : String(error), uncapturedErrors };
  } finally {
    device?.destroy();
  }
}

test.describe.configure({ mode: "serial" });
for (const testCase of cases) {
  test(`${testCase.name} executes and returns expected data`, async ({ page }, testInfo) => {
    await page.goto("/");
    const payload = { ...testCase };
    delete payload.assert;
    const result = await page.evaluate(executeCase, payload);

    if (!result.available) {
      throw new Error(`The unified harness could not obtain a hardware or software WebGPU adapter: ${result.reason}`);
    }

    expect(result.error).toBeUndefined();
    expect(result.compilationErrors).toEqual([]);
    expect(result.validationError).toBeNull();
    expect(result.uncapturedErrors).toEqual([]);
    const adapterLabel = [result.adapter.description, result.adapter.vendor, result.adapter.architecture]
      .filter(Boolean)
      .join(" / ") || "WebGPU adapter";
    testInfo.annotations.push({ type: "WebGPU mode", description: result.adapter.mode });
    testInfo.annotations.push({ type: "WebGPU adapter", description: adapterLabel });
    if (testCase.name === "atomics") console.log(`WebGPU execution: ${result.adapter.mode} (${adapterLabel})`);
    testCase.assert(result.bytes);
  });
}
