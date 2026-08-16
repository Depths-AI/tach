import { assert } from "./assert.js";

const test = Deno.test;

import * as tachAPI from "../src/index.ts";
import { defineModule as defineRuntimeModule } from "../src/internal.ts";
import { createTach } from "../src/runtime.ts";
import { openWeb } from "../src/web.ts";

function tach(workOrOptions = {}, options = {}) {
  const configured = typeof workOrOptions === "function"
    ? options
    : workOrOptions;
  const run = createTach((nativeOptions) =>
    openWeb(nativeOptions, configured.gpu)
  );
  return typeof workOrOptions === "function" ? run(workOrOptions) : run();
}

const compressedShader = new URL(
  "data:application/gzip;base64,H4sIAAAAAAAEAFMAAEXPbOkBAAAA",
);

function defineModule({ shader = compressedShader, resources, kernels }) {
  const kernel = kernels[0];
  const publicParameters = kernel.parameters.map((parameter) =>
    parameter.resource === undefined
      ? { name: parameter.name, kind: "value", type: "uint32" }
      : {
        name: parameter.name,
        kind: "buffer",
        type: "uint32[]",
        resource: parameter.resource,
      }
  );
  const external = resources.map((resource) => ({
    ...resource,
    type: "uint32[]",
  }));
  const bindings = resources.map((resource, binding) => ({
    group: 0,
    binding,
    access: resource.access,
    type: "uint32[]",
    minimumByteSize: resource.minimumByteSize,
    kind: "buffer",
  }));
  const block = kernel.parameterBlock && {
    group: 0,
    binding: kernel.parameterBlock.binding,
    byteSize: kernel.parameterBlock.byteSize,
    fields: kernel.parameterBlock.fields.map((field) => ({
      type: field.layout.kind === "f32"
        ? "float32"
        : field.layout.kind === "bool"
        ? "bool"
        : "uint32",
      byteOffset: field.offset,
      layout: field.layout,
    })),
  };
  const values = kernel.parameterBlock?.fields.map((field) => ({
    kind: "parameter",
    parameter: field.parameter,
    path: field.path,
  })) ?? [];
  const target = {
    kernels: [{
      entryPoint: kernel.entryPoint,
      workgroupSize: kernel.workgroupSize,
      bindings,
      parameterBlock: block,
    }],
    programs: [{
      program: 0,
      transients: [],
      steps: [{
        kind: "dispatch",
        kernel: 0,
        domain: Array.from(
          { length: kernel.dimensions },
          (_, axis) => ({ op: "launchAxis", axis }),
        ),
        resources: resources.map((_, binding) => ({
          kind: "external",
          binding,
          resource: binding,
        })),
        parameters: values,
      }],
      repeat: "program",
    }],
  };
  return defineRuntimeModule({
    shaders: {
      web: shader,
      spirv: new URL("file:///unused.spv"),
    },
    schema: 2,
    types: [],
    programs: [{
      name: kernel.name,
      parameters: publicParameters,
      resources: external,
      launch: {
        dimensions: kernel.dimensions,
        inferFromResource: kernel.dimensions === 1 ? 0 : undefined,
      },
    }],
    targets: { web: target, spirv: target },
  });
}

function fakeWebGPU({
  failCopy = false,
  failCopyUndefined = false,
  failReadbackCleanup = false,
  failWork = false,
  scopeErrors = [],
  shaderError,
} = {}) {
  const buffers = [];
  const calls = {
    bindGroups: 0,
    buffers: 0,
    buffersDestroyed: 0,
    deviceDestroyed: 0,
    dispatches: 0,
    dynamicOffsets: [],
    passes: 0,
    pipelines: 0,
    scopesPopped: 0,
    scopesPushed: 0,
    shaders: 0,
    shaderSources: [],
    submitted: 0,
    textures: 0,
    texturesDestroyed: 0,
    textureViews: 0,
    canvasConfigured: 0,
    currentTextures: 0,
    workDone: 0,
    writes: 0,
  };
  const device = {
    limits: { minUniformBufferOffsetAlignment: 256 },
    lost: new Promise(() => {}),
    queue: {
      submit() {
        calls.submitted++;
      },
      writeBuffer(buffer, bufferOffset, data, dataOffset = 0, size) {
        calls.writes++;
        const source = ArrayBuffer.isView(data)
          ? new Uint8Array(data.buffer, data.byteOffset, data.byteLength)
          : new Uint8Array(data);
        const length = size ?? source.byteLength - dataOffset;
        buffer.storage.set(
          source.subarray(dataOffset, dataOffset + length),
          bufferOffset,
        );
      },
      onSubmittedWorkDone() {
        calls.workDone++;
        return failWork ? Promise.reject(undefined) : Promise.resolve();
      },
    },
    addEventListener() {},
    removeEventListener() {},
    pushErrorScope() {
      calls.scopesPushed++;
    },
    popErrorScope() {
      calls.scopesPopped++;
      return Promise.resolve(scopeErrors.shift() ?? null);
    },
    destroy() {
      calls.deviceDestroyed++;
    },
    createShaderModule(descriptor) {
      if (shaderError) throw shaderError;
      calls.shaders++;
      calls.shaderSources.push(descriptor.code);
      return { descriptor };
    },
    createBindGroupLayout(descriptor) {
      return { descriptor };
    },
    createPipelineLayout(descriptor) {
      return { descriptor };
    },
    createComputePipelineAsync(descriptor) {
      calls.pipelines++;
      return Promise.resolve({ descriptor });
    },
    createBindGroup(descriptor) {
      calls.bindGroups++;
      return { descriptor };
    },
    createBuffer(descriptor) {
      calls.buffers++;
      const storage = new Uint8Array(descriptor.size);
      const buffer = {
        descriptor,
        size: descriptor.size,
        storage,
        destroy() {
          calls.buffersDestroyed++;
          if (failReadbackCleanup && descriptor.label === "Tach readback") {
            throw new Error("cleanup failed");
          }
        },
        getMappedRange() {
          return storage.buffer;
        },
        async mapAsync() {},
        unmap() {},
      };
      buffers.push(buffer);
      return buffer;
    },
    createTexture(descriptor) {
      calls.textures++;
      return {
        descriptor,
        createView() {
          calls.textureViews++;
          return {};
        },
        destroy() {
          calls.texturesDestroyed++;
        },
      };
    },
    createCommandEncoder() {
      const pass = {
        setPipeline() {},
        setBindGroup(_index, _group, offsets = []) {
          calls.dynamicOffsets.push([...offsets]);
        },
        dispatchWorkgroups() {
          calls.dispatches++;
        },
        end() {},
      };
      return {
        beginComputePass() {
          calls.passes++;
          return pass;
        },
        copyBufferToBuffer(
          source,
          sourceOffset,
          destination,
          destinationOffset,
          size,
        ) {
          if (failCopyUndefined) throw undefined;
          if (failCopy) throw new Error("copy failed");
          destination.storage.set(
            source.storage.subarray(sourceOffset, sourceOffset + size),
            destinationOffset,
          );
        },
        finish() {
          return {};
        },
      };
    },
  };
  const adapter = {
    info: { description: "test adapter" },
    requestDevice() {
      return Promise.resolve(device);
    },
  };
  return {
    buffers,
    calls,
    canvas(width, height, context = true) {
      const texture = device.createTexture({ size: [width, height] });
      return {
        width,
        height,
        getContext(name) {
          if (name !== "webgpu" || !context) return null;
          return {
            configure(descriptor) {
              calls.canvasConfigured++;
              this.descriptor = descriptor;
            },
            getCurrentTexture() {
              calls.currentTextures++;
              return texture;
            },
          };
        },
      };
    },
    gpu: {
      requestAdapter() {
        return Promise.resolve(adapter);
      },
    },
  };
}

const scalarBuffer = {
  name: "data",
  group: 0,
  binding: 0,
  kind: "storage",
  access: "read_write",
  minimumByteSize: 4,
  runtime: true,
  layout: {
    kind: "runtime",
    stride: 4,
    runtime: true,
    elem: { kind: "u32", size: 4 },
  },
};

const clear = defineModule({
  resources: [scalarBuffer],
  kernels: [{
    name: "clear",
    entryPoint: "clear",
    dimensions: 1,
    workgroupSize: [1, 1, 1],
    parameters: [{ name: "data", resource: 0 }],
  }],
});

const plane = defineModule({
  resources: [scalarBuffer],
  kernels: [{
    name: "plane",
    entryPoint: "plane",
    dimensions: 2,
    workgroupSize: [8, 8, 1],
    parameters: [{ name: "data", resource: 0 }],
  }],
});

const fill = defineModule({
  resources: [{
    name: "data",
    group: 0,
    binding: 0,
    kind: "storage",
    access: "read_write",
    minimumByteSize: 4,
    runtime: true,
    layout: {
      kind: "runtime",
      stride: 4,
      runtime: true,
      elem: { kind: "u32", size: 4 },
    },
  }],
  kernels: [{
    name: "fill",
    entryPoint: "fill",
    dimensions: 1,
    workgroupSize: [1, 1, 1],
    parameters: [{ name: "data", resource: 0 }, { name: "value" }],
    parameterBlock: {
      group: 0,
      binding: 1,
      byteSize: 16,
      fields: [{
        parameter: 1,
        path: [],
        offset: 0,
        layout: { kind: "u32", size: 4 },
      }],
    },
  }],
});

const configure = defineModule({
  resources: [scalarBuffer],
  kernels: [{
    name: "configure",
    entryPoint: "configure",
    dimensions: 1,
    workgroupSize: [1, 1, 1],
    parameters: [{ name: "data", resource: 0 }, { name: "settings" }],
    parameterBlock: {
      group: 0,
      binding: 1,
      byteSize: 16,
      fields: [
        {
          parameter: 1,
          path: ["enabled"],
          offset: 0,
          layout: { kind: "bool", size: 4 },
        },
        {
          parameter: 1,
          path: ["scale"],
          offset: 4,
          layout: { kind: "f32", size: 4 },
        },
      ],
    },
  }],
});

const vectors = defineModule({
  resources: [{
    name: "values",
    group: 0,
    binding: 0,
    kind: "storage",
    access: "read_write",
    minimumByteSize: 16,
    runtime: true,
    layout: {
      kind: "runtime",
      stride: 16,
      runtime: true,
      elem: {
        kind: "vector",
        count: 4,
        size: 16,
        elem: { kind: "f32", size: 4 },
      },
    },
  }],
  kernels: [{
    name: "vectors",
    entryPoint: "vectors",
    dimensions: 1,
    workgroupSize: [1, 1, 1],
    parameters: [{ name: "values", resource: 0 }],
  }],
});

const combine = defineModule({
  resources: [0, 1].map((binding) => ({
    name: `data${binding}`,
    group: 0,
    binding,
    kind: "storage",
    access: binding === 0 ? "read" : "read_write",
    minimumByteSize: 4,
    runtime: true,
    layout: {
      kind: "runtime",
      stride: 4,
      runtime: true,
      elem: { kind: "u32", size: 4 },
    },
  })),
  kernels: [{
    name: "combine",
    entryPoint: "combine",
    dimensions: 1,
    workgroupSize: [1, 1, 1],
    parameters: [
      { name: "input", resource: 0 },
      { name: "output", resource: 1 },
    ],
  }],
});

const graph = defineRuntimeModule({
  shaders: {
    web: compressedShader,
    spirv: new URL("file:///unused.spv"),
  },
  schema: 2,
  types: [],
  programs: [{
    name: "graph",
    parameters: [{
      name: "data",
      kind: "buffer",
      type: "uint32[]",
      resource: 0,
    }],
    resources: [{ ...scalarBuffer, type: "uint32[]" }],
  }],
  targets: {
    web: {
      kernels: [0, 1].map((index) => ({
        entryPoint: `_tach_k${index}`,
        workgroupSize: [1, 1, 1],
        bindings: [0, 1].map((binding) => ({
          group: 0,
          binding,
          access: "read_write",
          type: "uint32[]",
          minimumByteSize: 4,
        })),
      })),
      programs: [{
        program: 0,
        transients: [{
          type: "uint32[]",
          stride: 4,
          alignment: 4,
          minimumByteSize: 4,
          length: { op: "resourceLength", resource: 0 },
          color: 0,
          firstStep: 0,
          lastStep: 1,
        }],
        steps: [{
          kind: "dispatch",
          kernel: 0,
          domain: [{ op: "resourceLength", resource: 0 }],
          resources: [
            { kind: "external", binding: 0, resource: 0 },
            { kind: "transient", binding: 1, resource: 0 },
          ],
        }, {
          kind: "dispatch",
          kernel: 1,
          domain: [{ op: "resourceLength", resource: 0 }],
          resources: [
            { kind: "transient", binding: 0, resource: 0 },
            { kind: "external", binding: 1, resource: 0 },
          ],
        }],
        repeat: "program",
      }],
    },
    spirv: {
      kernels: [0, 1].map((index) => ({
        entryPoint: `_tach_k${index}`,
        workgroupSize: [1, 1, 1],
        bindings: [0, 1].map((binding) => ({
          group: 0,
          binding,
          access: "read_write",
          type: "uint32[]",
          minimumByteSize: 4,
        })),
      })),
      programs: [{
        program: 0,
        transients: [{
          type: "uint32[]",
          stride: 4,
          alignment: 4,
          minimumByteSize: 4,
          length: { op: "resourceLength", resource: 0 },
          color: 0,
          firstStep: 0,
          lastStep: 1,
        }],
        steps: [],
        repeat: "program",
      }],
    },
  },
});

const u32 = { kind: "u32", size: 4 };
const dimension = (parameter) => ({ op: "parameter", parameter });
function viewTarget(texture) {
  const width = dimension(0),
    height = dimension(1),
    pixels = { op: "mul", left: width, right: height };
  return {
    kernels: [{
      entryPoint: "_tach_k0",
      workgroupSize: [64, 1, 1],
      bindings: [{
        group: 0,
        binding: 0,
        access: "read_write",
        type: texture ? "rgba8unorm" : "uint32[]",
        minimumByteSize: 4,
        kind: texture ? "texture" : "buffer",
      }],
      parameterBlock: {
        group: 0,
        binding: 1,
        byteSize: 16,
        fields: [0, 4].map((byteOffset) => ({
          type: "uint32",
          byteOffset,
          layout: u32,
        })),
      },
    }],
    programs: [{
      program: 0,
      transients: [],
      steps: [],
      repeat: "program",
      view: {
        format: "srgb8",
        step: {
          kind: "dispatch",
          kernel: 0,
          domain: [pixels],
          resources: [],
          parameters: [
            { kind: "parameter", parameter: 0 },
            { kind: "parameter", parameter: 1 },
          ],
        },
        width,
        height,
        outputColor: 0,
        output: 0,
        fused: true,
      },
    }],
  };
}
const image = defineRuntimeModule({
  shaders: { web: compressedShader, spirv: new URL("file:///unused.spv") },
  schema: 2,
  types: [],
  programs: [{
    name: "image",
    parameters: ["width", "height"].map((name) => ({
      name,
      kind: "value",
      type: "uint32",
    })),
    resources: [],
    view: true,
  }],
  targets: { web: viewTarget(true), spirv: viewTarget(false) },
});

test("the public runtime has one entry point and one error type", () => {
  assert.deepEqual(Object.keys(tachAPI).sort(), ["TachError", "tach"]);
});

test("tach reports unavailable adapters as a typed failure", async () => {
  await assert.rejects(
    tach({
      gpu: {
        requestAdapter() {
          return Promise.resolve(null);
        },
      },
    }),
    (error) => {
      assert.equal(error instanceof tachAPI.TachError, true);
      assert.equal(error.code, "adapter-unavailable");
      assert.equal(error.message, "WebGPU did not provide an adapter");
      assert.equal(error.operation, "requestAdapter");
      return true;
    },
  );
});

test("tach owns the device and every buffer through success", async () => {
  const fake = fakeWebGPU();
  let buffer;
  const result = await tach(async (gpu) => {
    buffer = gpu.buffer({ values: [1, 2] });
    const first = await buffer.read();
    first.values[0] = 99;
    assert.deepEqual(await buffer.read(), { values: [1, 2] });
    buffer.write({ values: [3, 4] });
    return buffer.read();
  }, { gpu: fake.gpu });

  assert.deepEqual(result, { values: [3, 4] });
  assert.equal(fake.calls.deviceDestroyed, 1);
  assert.equal(fake.calls.workDone, 1);
  await assert.rejects(buffer.read(), /Tach session is closed/);
  buffer.destroy();
});

test("tach preserves callback failures and still closes", async () => {
  const fake = fakeWebGPU();
  await assert.rejects(
    tach(() => {
      throw new Error("application exploded");
    }, { gpu: fake.gpu }),
    (error) => {
      assert.equal(error.code, "user");
      assert.equal(error.message, "application exploded");
      assert.equal(error.operation, "tach");
      return true;
    },
  );
  assert.equal(fake.calls.deviceDestroyed, 1);
});

test("tach normalizes non-stringifiable failures", async () => {
  const fake = fakeWebGPU();
  const cause = Object.create(null);
  await assert.rejects(
    tach(() => {
      throw cause;
    }, { gpu: fake.gpu }),
    (error) => {
      assert.equal(error.code, "user");
      assert.equal(error.message, "Unknown error");
      assert.equal(error.cause, cause);
      return true;
    },
  );
  assert.equal(fake.calls.deviceDestroyed, 1);
});

test("kernel validation failures retain their typed GPU code", async () => {
  class GPUValidationError extends Error {}
  const fake = fakeWebGPU({
    scopeErrors: [null, null, null, new GPUValidationError("invalid binding")],
  });

  await assert.rejects(
    tach(async (gpu) => {
      await gpu.submit(clear.command(0, [gpu.buffer([1])]));
    }, { gpu: fake.gpu }),
    (error) => {
      assert.equal(error.code, "gpu-validation");
      assert.equal(error.message, "invalid binding");
      assert.equal(error.operation, "submit");
      return true;
    },
  );
  assert.equal(fake.calls.scopesPushed, 6);
  assert.equal(fake.calls.scopesPopped, 6);
  assert.equal(fake.calls.buffersDestroyed, 1);
  assert.equal(fake.calls.deviceDestroyed, 1);
});

test("synchronous shader failures retain their kernel error", async () => {
  const fake = fakeWebGPU({ shaderError: new Error("shader failed") });
  await assert.rejects(
    tach(async (gpu) => {
      await gpu.submit(clear.command(0, [gpu.buffer([1])]));
    }, { gpu: fake.gpu }),
    (error) => {
      assert.equal(error.code, "kernel");
      assert.equal(error.message, "shader failed");
      assert.equal(error.operation, "clear");
      return true;
    },
  );
  assert.equal(fake.calls.deviceDestroyed, 1);
});

test("undefined GPU work failures cannot be mistaken for success", async () => {
  const fake = fakeWebGPU({ failWork: true });
  const gpu = await tach({ gpu: fake.gpu });
  try {
    await gpu.submit(clear.command(0, [gpu.buffer([1])]));
    await assert.rejects(gpu.idle(), (failure) => {
      assert.equal(failure.code, "device-lost");
      assert.equal(failure.message, "undefined");
      return true;
    });
  } finally {
    gpu.close();
  }
});

test("storage encoding failures retain their buffer error", async () => {
  const fake = fakeWebGPU();
  await assert.rejects(
    tach(async (gpu) => {
      await gpu.submit(clear.command(0, [gpu.buffer(["bad"])]));
    }, { gpu: fake.gpu }),
    (error) => {
      assert.equal(error.code, "buffer");
      assert.match(error.message, /must be a number/u);
      assert.equal(error.operation, "data");
      return true;
    },
  );
  assert.equal(fake.calls.deviceDestroyed, 1);
});

test("one submission batches dispatches without waiting for the queue", async () => {
  const fake = fakeWebGPU();
  const gpu = await tach({ gpu: fake.gpu });
  try {
    const data = gpu.buffer(new Uint32Array([1, 2, 3]));
    await gpu.submit(clear.command(0, [data], { size: 3, repeat: 4 }));
    assert.equal(fake.calls.submitted, 1);
    assert.equal(fake.calls.workDone, 0);
    await gpu.idle();
    const result = await data.read();
    assert.equal(result instanceof Uint32Array, true);
    assert.deepEqual(result, new Uint32Array([1, 2, 3]));
    assert.equal(fake.calls.dispatches, 4);
    assert.equal(fake.calls.submitted, 2);
    assert.equal(fake.calls.workDone, 1);
  } finally {
    gpu.close();
  }
});

test("packed vector arrays stay flat typed arrays across upload and readback", async () => {
  const fake = fakeWebGPU();
  const gpu = await tach({ gpu: fake.gpu });
  try {
    const values = new Float32Array([1, 2, 3, 4, 5, 6, 7, 8]);
    const buffer = gpu.buffer(values);
    await gpu.submit(vectors.command(0, [buffer], { size: 2 }));
    assert.deepEqual(await buffer.read(), values);

    const partial = gpu.buffer(new Float32Array([1, 2, 3]));
    await assert.rejects(
      gpu.submit(vectors.command(0, [partial])),
      /complete 4-component elements/u,
    );
  } finally {
    gpu.close();
  }
});

test("kernel buffer parameters cannot alias the same compute buffer", async () => {
  const gpu = await tach({ gpu: fakeWebGPU().gpu });
  try {
    const data = gpu.buffer(new Uint32Array([1]));
    assert.throws(
      () => combine.command(0, [data, data]),
      /buffer parameters require distinct compute buffers/u,
    );
  } finally {
    gpu.close();
  }
});

test("one pass carries multiple kernels and reuses resident parameters and bind groups", async () => {
  const fake = fakeWebGPU();
  const gpu = await tach({ gpu: fake.gpu });
  try {
    const data = gpu.buffer(new Uint32Array([0]));
    await gpu.submit(
      fill.command(0, [data, 1]),
      fill.command(0, [data, 2]),
    );
    assert.equal(fake.calls.submitted, 1);
    assert.equal(fake.calls.passes, 1);
    assert.equal(fake.calls.dispatches, 2);
    assert.equal(fake.calls.buffers, 2);
    assert.equal(fake.calls.writes, 1);
    assert.equal(fake.calls.bindGroups, 1);
    assert.deepEqual(fake.calls.dynamicOffsets, [[0], [256]]);

    await gpu.submit(fill.command(0, [data, 3]));
    assert.equal(fake.calls.submitted, 2);
    assert.equal(fake.calls.passes, 2);
    assert.equal(fake.calls.dispatches, 3);
    assert.equal(fake.calls.buffers, 2);
    assert.equal(fake.calls.writes, 2);
    assert.equal(fake.calls.bindGroups, 1);
    assert.deepEqual(fake.calls.dynamicOffsets, [[0], [256], [0]]);
    assert.equal(fake.calls.shaders, 1);
    assert.equal(fake.calls.pipelines, 1);
    assert.equal(fake.calls.workDone, 0);
    await gpu.idle();
    assert.equal(fake.calls.workDone, 1);
  } finally {
    gpu.close();
  }
});

test("prepare compiles without execution and submit reuses the pipeline", async () => {
  const fake = fakeWebGPU(), gpu = await tach({ gpu: fake.gpu });
  try {
    const command = fill.command(0, [gpu.buffer(new Uint32Array([0])), 1]);
    await gpu.prepare(command);
    assert.equal(fake.calls.shaders, 1);
    assert.deepEqual(fake.calls.shaderSources, [" "]);
    assert.equal(fake.calls.pipelines, 1);
    assert.equal(fake.calls.submitted, 0);
    assert.equal(fake.calls.dispatches, 0);
    await gpu.submit(command);
    assert.equal(fake.calls.shaders, 1);
    assert.equal(fake.calls.pipelines, 1);
    assert.equal(fake.calls.submitted, 1);
    assert.equal(fake.calls.dispatches, 1);
  } finally {
    gpu.close();
  }
});

test("decompressed WGSL remains cached across sessions for seven days", async () => {
  const url = new URL("https://tach.test/kernel.wgsl.gz?v=test"),
    module = defineModule({
      shader: url,
      resources: [scalarBuffer],
      kernels: [{
        name: "cached",
        entryPoint: "cached",
        dimensions: 1,
        workgroupSize: [1, 1, 1],
        parameters: [{ name: "data", resource: 0 }],
      }],
    }),
    responses = new Map();
  const key = (request) =>
    request instanceof Request ? request.url : String(request);
  const cache = {
    keys() {
      return Promise.resolve(
        [...responses.keys()].map((url) => new Request(url)),
      );
    },
    match(request) {
      return Promise.resolve(responses.get(key(request))?.clone());
    },
    put(request, response) {
      responses.set(key(request), response.clone());
      return Promise.resolve();
    },
    delete(request) {
      return Promise.resolve(responses.delete(key(request)));
    },
  };
  const originalFetch = globalThis.fetch,
    originalCaches = Object.getOwnPropertyDescriptor(globalThis, "caches");
  let fetches = 0;
  globalThis.fetch = () => {
    fetches++;
    return Promise.resolve(
      new Response(
        Uint8Array.from(
          atob("H4sIAAAAAAAEAFMAAEXPbOkBAAAA"),
          (character) => character.charCodeAt(0),
        ),
      ),
    );
  };
  Object.defineProperty(globalThis, "caches", {
    configurable: true,
    value: { open: () => Promise.resolve(cache) },
  });
  const prepare = async () => {
    const fake = fakeWebGPU(), gpu = await tach({ gpu: fake.gpu });
    try {
      await gpu.prepare(
        module.command(0, [gpu.buffer(new Uint32Array([0]))]),
      );
      assert.deepEqual(fake.calls.shaderSources, [" "]);
    } finally {
      gpu.close();
    }
  };
  try {
    await prepare();
    await prepare();
    assert.equal(fetches, 1);
    responses.set(
      url.href,
      new Response("expired", { headers: { "x-tach-expires": "0" } }),
    );
    await prepare();
    assert.equal(fetches, 2);
  } finally {
    globalThis.fetch = originalFetch;
    if (originalCaches) {
      Object.defineProperty(globalThis, "caches", originalCaches);
    } else delete globalThis.caches;
  }
});

test("one command records a repeated multi-step plan with reusable growing scratch", async () => {
  const fake = fakeWebGPU();
  const gpu = await tach({ gpu: fake.gpu });
  try {
    await gpu.submit(
      graph.command(0, [gpu.buffer(new Uint32Array(1))], { repeat: 3 }),
    );
    assert.equal(fake.calls.shaders, 1);
    assert.equal(fake.calls.pipelines, 2);
    assert.equal(fake.calls.dispatches, 6);
    assert.equal(fake.calls.passes, 1);
    assert.equal(fake.calls.submitted, 1);
    assert.equal(
      fake.buffers.filter((buffer) =>
        buffer.descriptor.label === "Tach scratch 0"
      ).length,
      1,
    );

    await gpu.submit(graph.command(0, [gpu.buffer(new Uint32Array(2048))]));
    assert.equal(
      fake.buffers.filter((buffer) =>
        buffer.descriptor.label === "Tach scratch 0"
      ).length,
      2,
    );
    assert.equal(fake.calls.buffersDestroyed, 0);
    await gpu.idle();
    assert.equal(fake.calls.buffersDestroyed, 1);
  } finally {
    gpu.close();
  }
});

test("parameter blocks pack nested values and bools into compiler-planned offsets", async () => {
  const fake = fakeWebGPU();
  const gpu = await tach({ gpu: fake.gpu });
  try {
    const data = gpu.buffer(new Uint32Array([0]));
    await gpu.submit(
      configure.command(0, [data, { enabled: true, scale: 2.5 }]),
    );
    const arena = fake.buffers.find((buffer) =>
      buffer.descriptor.label === "Tach parameter arena"
    );
    assert.ok(arena);
    const view = new DataView(arena.storage.buffer);
    assert.equal(view.getUint32(0, true), 1);
    assert.equal(view.getFloat32(4, true), 2.5);
    await gpu.submit(
      configure.command(0, [data, { enabled: false, scale: 1.5 }]),
    );
    assert.equal(view.getUint32(0, true), 0);
    assert.equal(view.getFloat32(4, true), 1.5);
    assert.throws(
      () => configure.command(0, [data, { enabled: 1, scale: 2.5 }]),
      /configure\.settings\.enabled must be a boolean/u,
    );
  } finally {
    gpu.close();
  }
});

test("repeat count must be a positive integer", async () => {
  const fake = fakeWebGPU();
  const gpu = await tach({ gpu: fake.gpu });
  try {
    assert.throws(
      () => clear.command(0, [gpu.buffer(new Uint32Array([1]))], { repeat: 0 }),
      (failure) => {
        assert.equal(failure.code, "kernel");
        assert.match(failure.message, /positive integer/u);
        return true;
      },
    );
  } finally {
    gpu.close();
  }
});

test("launch size must match the kernel's logical rank", async () => {
  const gpu = await tach({ gpu: fakeWebGPU().gpu });
  try {
    const data = gpu.buffer(new Uint32Array([1]));
    assert.throws(
      () => clear.command(0, [data], { size: [1, 1] }),
      /exact 1D launch size/u,
    );
  } finally {
    gpu.close();
  }
});

test("multidimensional dispatch does not infer a shape from flat storage", async () => {
  const fake = fakeWebGPU();
  const gpu = await tach({ gpu: fake.gpu });
  try {
    const data = gpu.buffer(new Uint32Array(64));
    await gpu.submit(plane.command(0, [data]));
    await gpu.submit(plane.command(0, [data], { size: [16, 8] }));
    assert.throws(
      () => plane.command(0, [data], { size: 64 }),
      /exact 2D launch size/u,
    );
    assert.equal(fake.calls.dispatches, 2);
  } finally {
    gpu.close();
  }
});

test("awaiting a command without submitting it fails loudly", async () => {
  const fake = fakeWebGPU();
  const gpu = await tach({ gpu: fake.gpu });
  try {
    const command = clear.command(0, [gpu.buffer([1])]);
    await assert.rejects(async () => await command, (failure) => {
      assert.equal(failure.code, "kernel");
      assert.match(failure.message, /Tach\.submit/u);
      return true;
    });
    assert.equal(fake.calls.submitted, 0);
  } finally {
    gpu.close();
  }
});

test("command ownership and resident-buffer lifetime are enforced", async () => {
  const first = await tach({ gpu: fakeWebGPU().gpu });
  const second = await tach({ gpu: fakeWebGPU().gpu });
  try {
    const data = first.buffer([1]);
    const command = clear.command(0, [data]);
    assert.throws(() => second.submit(command), /different Tach session/u);

    data.destroy();
    await assert.rejects(first.submit(command), (failure) => {
      assert.equal(failure.code, "lifecycle");
      assert.match(failure.message, /destroyed/u);
      return true;
    });
  } finally {
    first.close();
    second.close();
  }
});

test("scalar view recipes are owner-neutral and offscreen textures are reused", async () => {
  const recipe = image.command(0, [32, 16]),
    firstFake = fakeWebGPU(),
    secondFake = fakeWebGPU(),
    first = await tach({ gpu: firstFake.gpu }),
    second = await tach({ gpu: secondFake.gpu });
  try {
    assert.throws(
      () => image.command(0, [32, 16], { repeat: 2 }),
      /view commands cannot repeat/u,
    );
    await first.prepare(recipe);
    await first.submit(recipe);
    await second.submit(recipe);
    assert.equal(firstFake.calls.dispatches, 1);
    assert.equal(secondFake.calls.dispatches, 1);
    assert.equal(firstFake.calls.textures, 1);
    await first.submit(image.command(0, [32, 16]));
    assert.equal(firstFake.calls.textures, 1);
    await first.submit(image.command(0, [16, 16]));
    assert.equal(firstFake.calls.textures, 2);
    assert.equal(firstFake.calls.texturesDestroyed, 0);
    await first.idle();
    assert.equal(firstFake.calls.texturesDestroyed, 1);
  } finally {
    first.close();
    second.close();
  }
});

test("present serializes completed frames into one configured canvas", async () => {
  const fake = fakeWebGPU(),
    gpu = await tach({ gpu: fake.gpu }),
    canvas = fake.canvas(32, 16),
    view = image.command(0, [32, 16]);
  try {
    await Promise.all(
      Array.from({ length: 8 }, () => gpu.present(canvas, view)),
    );
    assert.equal(fake.calls.canvasConfigured, 1);
    assert.equal(fake.calls.currentTextures, 8);
    assert.equal(fake.calls.dispatches, 8);
    assert.equal(fake.calls.submitted, 8);
    assert.equal(fake.calls.workDone, 8);
    assert.throws(
      () => gpu.present(canvas, clear.command(0, [gpu.buffer([0])])),
      /requires a generated Tach view/u,
    );
  } finally {
    gpu.close();
  }
});

test("present validates view extents and WebGPU canvas capability", async () => {
  for (
    const [view, canvas, message] of [
      [image.command(0, [32, 16]), fakeWebGPU().canvas(16, 16), "dimensions"],
      [
        image.command(0, [32, 16]),
        fakeWebGPU().canvas(32, 16, false),
        "context",
      ],
      [image.command(0, [0, 16]), fakeWebGPU().canvas(0, 16), "overflow"],
    ]
  ) {
    const gpu = await tach({ gpu: fakeWebGPU().gpu });
    try {
      await assert.rejects(gpu.present(canvas, view), new RegExp(message, "u"));
    } finally {
      gpu.close();
    }
  }
});

test("readback setup failures destroy temporary buffers", async () => {
  const fake = fakeWebGPU({ failCopy: true });

  await assert.rejects(
    tach(async (gpu) => {
      const data = gpu.buffer([1]);
      await gpu.submit(clear.command(0, [data]));
      return data.read();
    }, { gpu: fake.gpu }),
    (error) => {
      assert.equal(error.code, "buffer");
      return true;
    },
  );
  assert.equal(fake.calls.buffersDestroyed, 2);
  assert.equal(fake.calls.deviceDestroyed, 1);
});

test("readback cleanup failures are not discarded", async () => {
  const fake = fakeWebGPU({ failReadbackCleanup: true });
  const gpu = await tach({ gpu: fake.gpu });
  try {
    const data = gpu.buffer([1]);
    await gpu.submit(clear.command(0, [data]));
    await assert.rejects(data.read(), /cleanup failed/u);
  } finally {
    gpu.close();
  }
});

test("undefined readback setup failures cannot be mistaken for success", async () => {
  const fake = fakeWebGPU({ failCopyUndefined: true });
  const gpu = await tach({ gpu: fake.gpu });
  try {
    const data = gpu.buffer([1]);
    await gpu.submit(clear.command(0, [data]));
    await assert.rejects(data.read(), (failure) => {
      assert.equal(failure.code, "buffer");
      assert.equal(failure.message, "undefined");
      return true;
    });
  } finally {
    gpu.close();
  }
  assert.equal(fake.calls.buffersDestroyed, 2);
});
