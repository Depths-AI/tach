import { expect, test } from "@playwright/test";

const exampleNames = ["atomics", "bitwise", "control", "for", "math", "particles", "scalars"];
const values = {
  atomics: { counters: { total: 7 } },
  bitwise: { out: [1, 2, 3] },
  control: { data: [1, 2, 3], params: { scale: 2, count: 3 } },
  for: { data: [1, 2, 3, 4] },
  math: { out: [[1, 2, 3, 4]] },
  particles: {
    particles: [{ position: [1, 2, 3, 4], velocity: [5, 6, 7, 8] }],
    params: { dt: 0.5, count: 1 },
  },
  scalars: { data: [1, 2, 3], factor: 3 },
};

test("inspection UI loads all compiled examples", async ({ page }) => {
  const pageErrors = [];
  page.on("pageerror", (error) => pageErrors.push(error.message));
  await page.goto("/");
  await page.evaluate(() => globalThis.__tachHarnessReady);

  await expect(page).toHaveTitle("Tach Browser Harness");
  await expect(page.locator("#examples tr")).toHaveCount(exampleNames.length);
  await expect(page.locator("#build-status")).toContainText(`generated ${exampleNames.length} modules`);
  await expect(page.locator("#gpu-status")).not.toHaveText("probing…");
  expect(await page.locator("#examples td:first-child").allTextContents()).toEqual(exampleNames);
  expect(pageErrors).toEqual([]);
});

test("generated modules expose stable metadata and deterministic ABI packers", async ({ page }) => {
  await page.goto("/");
  const result = await page.evaluate(async (inputs) => {
    const manifest = await (await fetch("/build/manifest.json")).json();
    const modules = [];

    for (const entry of manifest.examples) {
      const generated = await import(entry.module);
      const packed = [];
      for (const resource of generated.metadata.resources) {
        const helper = `pack_${resource.name}_g${resource.group}_b${resource.binding}`;
        const pack = generated[helper];
        if (typeof pack !== "function") throw new Error(`${entry.name} is missing ${helper}`);
        const value = inputs[entry.name][resource.name];
        const first = pack(value);
        const second = pack(value);
        const expectedSize = resource.runtime
          ? (resource.runtimeOffset ?? 0) + resource.runtimeStride * value.length
          : resource.bindingSize;
        packed.push({
          resource: resource.name,
          size: first.byteLength,
          expectedSize,
          deterministic: Array.from(first).every((byte, index) => byte === second[index]),
        });
      }
      modules.push({
        name: entry.name,
        format: generated.metadata.format,
        layout: generated.metadata.abi.layout,
        kernels: generated.metadata.kernels.length,
        resources: generated.metadata.resources.length,
        hasComputeWGSL: generated.wgsl.includes("@compute"),
        packed,
      });
    }

    const scalars = await import("/build/scalars.js");
    const factorBytes = scalars.pack_factor_g0_b1(3);
    const atomics = await import("/build/atomics.js");
    const counterBytes = atomics.pack_counters_g0_b0({ total: 7 });
    const particles = await import("/build/particles.js");
    const particleBytes = particles.pack_particles_g0_b0(inputs.particles.particles);
    return {
      modules,
      encoded: {
        factor: new DataView(factorBytes.buffer, factorBytes.byteOffset, factorBytes.byteLength).getFloat32(0, true),
        counter: new DataView(counterBytes.buffer, counterBytes.byteOffset, counterBytes.byteLength).getUint32(0, true),
        particlePositionX: new DataView(particleBytes.buffer, particleBytes.byteOffset, particleBytes.byteLength).getFloat32(0, true),
        particleVelocityX: new DataView(particleBytes.buffer, particleBytes.byteOffset, particleBytes.byteLength).getFloat32(16, true),
      },
    };
  }, values);

  expect(result.modules.map((module) => module.name)).toEqual(exampleNames);
  for (const module of result.modules) {
    expect(module.format).toBe("tach.module.v1");
    expect(module.layout).toBe("tach-portable-v1");
    expect(module.kernels).toBe(1);
    expect(module.resources).toBeGreaterThan(0);
    expect(module.hasComputeWGSL).toBe(true);
    for (const resource of module.packed) {
      expect(resource.size, `${module.name}.${resource.resource} size`).toBe(resource.expectedSize);
      expect(resource.deterministic, `${module.name}.${resource.resource} determinism`).toBe(true);
    }
  }
  expect(result.encoded).toEqual({ factor: 3, counter: 7, particlePositionX: 1, particleVelocityX: 5 });
});

test("generated programs bind and dispatch through a mock WebGPU interface", async ({ page }) => {
  await page.goto("/");
  const result = await page.evaluate(async (inputs) => {
    if (!("GPUShaderStage" in globalThis)) globalThis.GPUShaderStage = { COMPUTE: 4 };
    if (!("GPUBufferUsage" in globalThis)) {
      globalThis.GPUBufferUsage = { COPY_DST: 8, COPY_SRC: 4, STORAGE: 128, UNIFORM: 64 };
    }

    const manifest = await (await fetch("/build/manifest.json")).json();
    const summaries = [];
    for (const entry of manifest.examples) {
      const calls = { bindGroups: 0, buffers: 0, dispatches: [], pipelines: 0, shaders: 0, writes: 0 };
      const device = {
        queue: {
          writeBuffer() { calls.writes++; },
        },
        createShaderModule(descriptor) {
          calls.shaders++;
          return { descriptor };
        },
        createBindGroupLayout(descriptor) { return { descriptor }; },
        createPipelineLayout(descriptor) { return { descriptor }; },
        createComputePipeline(descriptor) {
          calls.pipelines++;
          return { descriptor };
        },
        createBindGroup(descriptor) {
          calls.bindGroups++;
          return { descriptor };
        },
        createBuffer(descriptor) {
          calls.buffers++;
          const storage = new ArrayBuffer(descriptor.size);
          return {
            descriptor,
            getMappedRange() { return storage; },
            unmap() {},
          };
        },
      };
      const pass = {
        setPipeline() {},
        setBindGroup() {},
        dispatchWorkgroups(x, y, z) { calls.dispatches.push([x, y, z]); },
        end() {},
      };
      const encoder = { beginComputePass() { return pass; } };

      const generated = await import(entry.module);
      const buffers = [];
      for (const resource of generated.metadata.resources) {
        const create = generated[`create_${resource.name}_g${resource.group}_b${resource.binding}`];
        buffers[resource.index] = create(device, inputs[entry.name][resource.name]);
      }
      const program = generated.createTachProgram(device);
      const kernelInfo = generated.metadata.kernels[0];
      const resources = {};
      for (const parameter of kernelInfo.resources) resources[parameter.param] = buffers[parameter.resource];
      program.kernels[kernelInfo.name].dispatch(encoder, resources, [1, 1, 1]);

      const firstResource = generated.metadata.resources[0];
      const write = generated[`write_${firstResource.name}_g${firstResource.group}_b${firstResource.binding}`];
      write(device, buffers[firstResource.index], inputs[entry.name][firstResource.name]);
      summaries.push({ name: entry.name, calls });
    }
    return summaries;
  }, values);

  expect(result.map((summary) => summary.name)).toEqual(exampleNames);
  for (const { name, calls } of result) {
    expect(calls.shaders, `${name} shader modules`).toBe(1);
    expect(calls.pipelines, `${name} pipelines`).toBe(1);
    expect(calls.buffers, `${name} buffers`).toBeGreaterThan(0);
    expect(calls.bindGroups, `${name} bind groups`).toBeGreaterThan(0);
    expect(calls.dispatches, `${name} dispatches`).toEqual([[1, 1, 1]]);
    expect(calls.writes, `${name} queue writes`).toBe(1);
  }
});
