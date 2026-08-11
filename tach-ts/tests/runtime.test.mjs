import assert from "node:assert/strict";
import test from "node:test";

import { openTach, tach } from "../dist/index.js";
import { defineModule } from "../dist/internal.js";

function fakeWebGPU({
  failCopy = false,
  failCopyUndefined = false,
  failWork = false,
  scopeErrors = [],
  shaderError,
} = {}) {
  const calls = {
    bindGroups: 0,
    buffers: 0,
    buffersDestroyed: 0,
    deviceDestroyed: 0,
    dispatches: 0,
    passes: 0,
    pipelines: 0,
    scopesPopped: 0,
    scopesPushed: 0,
    shaders: 0,
    submitted: 0,
    workDone: 0,
    writes: 0,
  };
  const device = {
    limits: { minUniformBufferOffsetAlignment: 256 },
    lost: new Promise(() => {}),
    queue: {
      submit() { calls.submitted++; },
      writeBuffer(buffer, bufferOffset, data, dataOffset = 0, size) {
        calls.writes++;
        const source = ArrayBuffer.isView(data)
          ? new Uint8Array(data.buffer, data.byteOffset, data.byteLength)
          : new Uint8Array(data);
        const length = size ?? source.byteLength - dataOffset;
        buffer.storage.set(source.subarray(dataOffset, dataOffset + length), bufferOffset);
      },
      async onSubmittedWorkDone() {
        calls.workDone++;
        if (failWork) throw undefined;
      },
    },
    addEventListener() {},
    removeEventListener() {},
    pushErrorScope() { calls.scopesPushed++; },
    async popErrorScope() {
      calls.scopesPopped++;
      return scopeErrors.shift() ?? null;
    },
    destroy() { calls.deviceDestroyed++; },
    createShaderModule(descriptor) {
      if (shaderError) throw shaderError;
      calls.shaders++;
      return { descriptor };
    },
    createBindGroupLayout(descriptor) { return { descriptor }; },
    createPipelineLayout(descriptor) { return { descriptor }; },
    async createComputePipelineAsync(descriptor) {
      calls.pipelines++;
      return { descriptor };
    },
    createBindGroup(descriptor) {
      calls.bindGroups++;
      return { descriptor };
    },
    createBuffer(descriptor) {
      calls.buffers++;
      const storage = new Uint8Array(descriptor.size);
      return {
        size: descriptor.size,
        storage,
        destroy() { calls.buffersDestroyed++; },
        getMappedRange() { return storage.buffer; },
        async mapAsync() {},
        unmap() {},
      };
    },
    createCommandEncoder() {
      const pass = {
        setPipeline() {},
        setBindGroup() {},
        dispatchWorkgroups() { calls.dispatches++; },
        end() {},
      };
      return {
        beginComputePass() { calls.passes++; return pass; },
        copyBufferToBuffer(source, sourceOffset, destination, destinationOffset, size) {
          if (failCopyUndefined) throw undefined;
          if (failCopy) throw new Error("copy failed");
          destination.storage.set(
            source.storage.subarray(sourceOffset, sourceOffset + size),
            destinationOffset,
          );
        },
        finish() { return {}; },
      };
    },
  };
  const adapter = {
    info: { description: "test adapter" },
    async requestDevice() { return device; },
  };
  return {
    calls,
    gpu: { async requestAdapter() { return adapter; } },
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
  source: "",
  resources: [scalarBuffer],
  kernels: [{
    name: "clear",
    entryPoint: "clear",
    dimensions: 1,
    workgroupSize: [1, 1, 1],
    resources: [{ name: "data", resource: 0 }],
  }],
});

const plane = defineModule({
  source: "",
  resources: [scalarBuffer],
  kernels: [{
    name: "plane",
    entryPoint: "plane",
    dimensions: 2,
    workgroupSize: [8, 8, 1],
    resources: [{ name: "data", resource: 0 }],
  }],
});

const fill = defineModule({
  source: "",
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
  }, {
    name: "value",
    group: 0,
    binding: 1,
    kind: "uniform",
    access: "read",
    byteSize: 16,
    minimumByteSize: 16,
    runtime: false,
    layout: { kind: "u32", size: 4 },
  }],
  kernels: [{
    name: "fill",
    entryPoint: "fill",
    dimensions: 1,
    workgroupSize: [1, 1, 1],
    resources: [{ name: "data", resource: 0 }, { name: "value", resource: 1 }],
  }],
});

const vectors = defineModule({
  source: "",
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
    resources: [{ name: "values", resource: 0 }],
  }],
});

const combine = defineModule({
  source: "",
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
    resources: [
      { name: "input", resource: 0 },
      { name: "output", resource: 1 },
    ],
  }],
});

test("openTach reports unavailable adapters as data", async () => {
  const result = await openTach({ gpu: { async requestAdapter() { return null; } } });
  assert.deepEqual(result, {
    ok: false,
    error: {
      code: "adapter-unavailable",
      message: "WebGPU did not provide an adapter",
      operation: "requestAdapter",
    },
  });
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

  assert.deepEqual(result, { ok: true, value: { values: [3, 4] } });
  assert.equal(fake.calls.deviceDestroyed, 1);
  assert.equal(fake.calls.workDone, 1);
  await assert.rejects(buffer.read(), /Tach session is closed/);
  buffer.destroy();
});

test("tach converts callback failures to data and still closes", async () => {
  const fake = fakeWebGPU();
  const result = await tach(() => {
    throw new Error("application exploded");
  }, { gpu: fake.gpu });

  assert.equal(result.ok, false);
  assert.equal(result.error.code, "user");
  assert.equal(result.error.message, "application exploded");
  assert.equal(result.error.operation, "tach");
  assert.equal(fake.calls.deviceDestroyed, 1);
});

test("tach normalizes non-stringifiable failures", async () => {
  const fake = fakeWebGPU();
  const cause = Object.create(null);
  const result = await tach(() => { throw cause; }, { gpu: fake.gpu });

  assert.equal(result.ok, false);
  assert.equal(result.error.code, "user");
  assert.equal(result.error.message, "Unknown error");
  assert.equal(result.error.cause, cause);
  assert.equal(fake.calls.deviceDestroyed, 1);
});

test("kernel validation failures retain their typed GPU code", async () => {
  class GPUValidationError extends Error {}
  const fake = fakeWebGPU({
    scopeErrors: [null, null, null, new GPUValidationError("invalid binding")],
  });

  const result = await tach(async (gpu) => {
    await gpu.submit(clear.dispatch(0, [gpu.buffer([1])]));
  }, { gpu: fake.gpu });

  assert.equal(result.ok, false);
  assert.equal(result.error.code, "gpu-validation");
  assert.equal(result.error.message, "invalid binding");
  assert.equal(result.error.operation, "submit");
  assert.equal(fake.calls.scopesPushed, 6);
  assert.equal(fake.calls.scopesPopped, 6);
  assert.equal(fake.calls.buffersDestroyed, 1);
  assert.equal(fake.calls.deviceDestroyed, 1);
});

test("synchronous shader failures retain their kernel error", async () => {
  const fake = fakeWebGPU({ shaderError: new Error("shader failed") });
  const result = await tach(async (gpu) => {
    await gpu.submit(clear.dispatch(0, [gpu.buffer([1])]));
  }, { gpu: fake.gpu });

  assert.equal(result.ok, false);
  assert.equal(result.error.code, "kernel");
  assert.equal(result.error.message, "shader failed");
  assert.equal(result.error.operation, "clear");
  assert.equal(fake.calls.deviceDestroyed, 1);
});

test("undefined GPU work failures cannot be mistaken for success", async () => {
  const fake = fakeWebGPU({ failWork: true });
  const opened = await openTach({ gpu: fake.gpu });
  assert.equal(opened.ok, true);
  try {
    await opened.value.submit(clear.dispatch(0, [opened.value.buffer([1])]));
    await assert.rejects(opened.value.idle(), (failure) => {
      assert.equal(failure.data.code, "device-lost");
      assert.equal(failure.data.message, "undefined");
      return true;
    });
  } finally {
    opened.value.close();
  }
});

test("storage encoding failures retain their buffer error", async () => {
  const fake = fakeWebGPU();
  const result = await tach(async (gpu) => {
    await gpu.submit(clear.dispatch(0, [gpu.buffer(["bad"])]));
  }, { gpu: fake.gpu });

  assert.equal(result.ok, false);
  assert.equal(result.error.code, "buffer");
  assert.match(result.error.message, /must be a number/u);
  assert.equal(result.error.operation, "data");
  assert.equal(fake.calls.deviceDestroyed, 1);
});

test("one submission batches dispatches without waiting for the queue", async () => {
  const fake = fakeWebGPU();
  const opened = await openTach({ gpu: fake.gpu });
  assert.equal(opened.ok, true);
  try {
    const data = opened.value.buffer(new Uint32Array([1, 2, 3]));
    await opened.value.submit(clear.dispatch(0, [data], { size: 3, dispatches: 4 }));
    assert.equal(fake.calls.submitted, 1);
    assert.equal(fake.calls.workDone, 0);
    await opened.value.idle();
    const result = await data.read();
    assert.equal(result instanceof Uint32Array, true);
    assert.deepEqual(result, new Uint32Array([1, 2, 3]));
    assert.equal(fake.calls.dispatches, 4);
    assert.equal(fake.calls.submitted, 2);
    assert.equal(fake.calls.workDone, 1);
  } finally {
    opened.value.close();
  }
});

test("packed vector arrays stay flat typed arrays across upload and readback", async () => {
  const fake = fakeWebGPU();
  const opened = await openTach({ gpu: fake.gpu });
  assert.equal(opened.ok, true);
  try {
    const values = new Float32Array([1, 2, 3, 4, 5, 6, 7, 8]);
    const buffer = opened.value.buffer(values);
    await opened.value.submit(vectors.dispatch(0, [buffer], { size: 2 }));
    assert.deepEqual(await buffer.read(), values);

    const partial = opened.value.buffer(new Float32Array([1, 2, 3]));
    await assert.rejects(
      opened.value.submit(vectors.dispatch(0, [partial])),
      /complete 4-component elements/u,
    );
  } finally {
    opened.value.close();
  }
});

test("kernel resource parameters cannot alias the same compute buffer", async () => {
  const opened = await openTach({ gpu: fakeWebGPU().gpu });
  assert.equal(opened.ok, true);
  try {
    const data = opened.value.buffer(new Uint32Array([1]));
    assert.throws(
      () => combine.dispatch(0, [data, data]),
      /resource parameters require distinct compute buffers/u,
    );
  } finally {
    opened.value.close();
  }
});

test("one pass carries multiple kernels and reuses resident uniforms and bind groups", async () => {
  const fake = fakeWebGPU();
  const opened = await openTach({ gpu: fake.gpu });
  assert.equal(opened.ok, true);
  try {
    const data = opened.value.buffer(new Uint32Array([0]));
    await opened.value.submit(
      fill.dispatch(0, [data, 1]),
      fill.dispatch(0, [data, 2]),
    );
    assert.equal(fake.calls.submitted, 1);
    assert.equal(fake.calls.passes, 1);
    assert.equal(fake.calls.dispatches, 2);
    assert.equal(fake.calls.buffers, 2);
    assert.equal(fake.calls.writes, 1);
    assert.equal(fake.calls.bindGroups, 2);

    await opened.value.submit(fill.dispatch(0, [data, 3]));
    assert.equal(fake.calls.submitted, 2);
    assert.equal(fake.calls.passes, 2);
    assert.equal(fake.calls.dispatches, 3);
    assert.equal(fake.calls.buffers, 2);
    assert.equal(fake.calls.writes, 2);
    assert.equal(fake.calls.bindGroups, 2);
    assert.equal(fake.calls.shaders, 1);
    assert.equal(fake.calls.pipelines, 1);
    assert.equal(fake.calls.workDone, 0);
    await opened.value.idle();
    assert.equal(fake.calls.workDone, 1);
  } finally {
    opened.value.close();
  }
});

test("dispatch count must be a positive integer", async () => {
  const fake = fakeWebGPU();
  const opened = await openTach({ gpu: fake.gpu });
  assert.equal(opened.ok, true);
  try {
    assert.throws(
      () => clear.dispatch(0, [opened.value.buffer(new Uint32Array([1]))], { dispatches: 0 }),
      (failure) => {
        assert.equal(failure.data.code, "kernel");
        assert.match(failure.data.message, /positive integer/u);
        return true;
      },
    );
  } finally {
    opened.value.close();
  }
});

test("dispatch size must match the kernel's logical rank", async () => {
  const opened = await openTach({ gpu: fakeWebGPU().gpu });
  assert.equal(opened.ok, true);
  try {
    const data = opened.value.buffer(new Uint32Array([1]));
    await assert.rejects(
      opened.value.submit(clear.dispatch(0, [data], { size: [1, 1] })),
      /exact 1D dispatch size/u,
    );
  } finally {
    opened.value.close();
  }
});

test("multidimensional dispatch does not infer a shape from flat storage", async () => {
  const fake = fakeWebGPU();
  const opened = await openTach({ gpu: fake.gpu });
  assert.equal(opened.ok, true);
  try {
    const data = opened.value.buffer(new Uint32Array(64));
    await opened.value.submit(plane.dispatch(0, [data]));
    await opened.value.submit(plane.dispatch(0, [data], { size: [16, 8] }));
    await assert.rejects(
      opened.value.submit(plane.dispatch(0, [data], { size: 64 })),
      /exact 2D dispatch size/u,
    );
    assert.equal(fake.calls.dispatches, 2);
  } finally {
    opened.value.close();
  }
});

test("awaiting a dispatch without submitting it fails loudly", async () => {
  const fake = fakeWebGPU();
  const opened = await openTach({ gpu: fake.gpu });
  assert.equal(opened.ok, true);
  try {
    const dispatch = clear.dispatch(0, [opened.value.buffer([1])]);
    await assert.rejects(async () => await dispatch, (failure) => {
      assert.equal(failure.data.code, "kernel");
      assert.match(failure.data.message, /Tach\.submit/u);
      return true;
    });
    assert.equal(fake.calls.submitted, 0);
  } finally {
    opened.value.close();
  }
});

test("dispatch ownership and resident-buffer lifetime are enforced", async () => {
  const first = await openTach({ gpu: fakeWebGPU().gpu });
  const second = await openTach({ gpu: fakeWebGPU().gpu });
  assert.equal(first.ok, true);
  assert.equal(second.ok, true);
  try {
    const data = first.value.buffer([1]);
    const dispatch = clear.dispatch(0, [data]);
    assert.throws(() => second.value.submit(dispatch), /different Tach session/u);

    data.destroy();
    await assert.rejects(first.value.submit(dispatch), (failure) => {
      assert.equal(failure.data.code, "lifecycle");
      assert.match(failure.data.message, /destroyed/u);
      return true;
    });
  } finally {
    first.value.close();
    second.value.close();
  }
});

test("readback setup failures destroy temporary buffers", async () => {
  const fake = fakeWebGPU({ failCopy: true });

  const result = await tach(async (gpu) => {
    const data = gpu.buffer([1]);
    await gpu.submit(clear.dispatch(0, [data]));
    return data.read();
  }, { gpu: fake.gpu });

  assert.equal(result.ok, false);
  assert.equal(result.error.code, "buffer");
  assert.equal(fake.calls.buffersDestroyed, 2);
  assert.equal(fake.calls.deviceDestroyed, 1);
});

test("undefined readback setup failures cannot be mistaken for success", async () => {
  const fake = fakeWebGPU({ failCopyUndefined: true });
  const opened = await openTach({ gpu: fake.gpu });
  assert.equal(opened.ok, true);
  try {
    const data = opened.value.buffer([1]);
    await opened.value.submit(clear.dispatch(0, [data]));
    await assert.rejects(data.read(), (failure) => {
      assert.equal(failure.data.code, "buffer");
      assert.equal(failure.data.message, "undefined");
      return true;
    });
  } finally {
    opened.value.close();
  }
  assert.equal(fake.calls.buffersDestroyed, 2);
});
