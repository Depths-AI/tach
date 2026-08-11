import { expect, test } from "@playwright/test";

const exampleNames = ["atomics", "bitwise", "control", "for", "math", "particles", "scalars"];
const values = {
  atomics: { counters: { total: 7 } },
  bitwise: { out: [1, 2, 3] },
  control: { data: [1, 2, 3], params: { scale: 2, count: 3, enabled: true } },
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

test("metadata is plain and uses source kernel names", async ({ page }) => {
  await page.goto("/");
  const modules = await page.evaluate(async () => {
    const manifest = await (await fetch("/build/manifest.json")).json();
    const result = [];
    for (const entry of manifest.examples) {
      const metadata = await (await fetch(entry.metadata)).json();
      result.push({
        name: entry.name,
        topLevelKeys: Object.keys(metadata).sort(),
        entries: metadata.kernels.map((kernel) => [kernel.name, kernel.entryPoint]),
        resources: metadata.resources.length,
      });
    }
    return result;
  });

  expect(modules.map((module) => module.name)).toEqual(exampleNames);
  for (const module of modules) {
    expect(module.topLevelKeys).toEqual(["kernels", "resources", "types"]);
    expect(module.entries.length).toBe(1);
    expect(module.entries[0][1]).toBe(module.entries[0][0]);
    expect(module.resources).toBeGreaterThan(0);
  }
});

test("@depths/tach owns sessions and generated modules expose only direct commands", async ({ page }) => {
  await page.goto("/");
  const result = await page.evaluate(async (inputs) => {
    const { tach } = await import("@depths/tach");
    const manifest = await (await fetch("/build/manifest.json")).json();
    const summaries = [];
    for (const entry of manifest.examples) {
      const calls = {
        bindGroups: 0,
        buffers: 0,
        destroyed: 0,
        dispatches: [],
        deviceDestroyed: 0,
        passes: 0,
        pipelines: 0,
        scopesPopped: 0,
        scopesPushed: 0,
        shaders: 0,
        submits: 0,
        workDone: 0,
        writes: 0,
      };
      const listeners = new Map();
      const device = {
        limits: { minUniformBufferOffsetAlignment: 256 },
        lost: new Promise(() => {}),
        queue: {
          writeBuffer() { calls.writes++; },
          submit() { calls.submits++; },
          async onSubmittedWorkDone() { calls.workDone++; },
        },
        addEventListener(type, listener) { listeners.set(type, listener); },
        removeEventListener(type) { listeners.delete(type); },
        pushErrorScope() { calls.scopesPushed++; },
        async popErrorScope() { calls.scopesPopped++; return null; },
        destroy() { calls.deviceDestroyed++; },
        createShaderModule(descriptor) {
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
          const storage = new ArrayBuffer(descriptor.size);
          return {
            size: descriptor.size,
            descriptor,
            destroy() { calls.destroyed++; },
            getMappedRange() { return storage; },
            unmap() {},
          };
        },
        createCommandEncoder() {
          const pass = {
            setPipeline() {},
            setBindGroup() {},
            dispatchWorkgroups(x, y, z) { calls.dispatches.push([x, y, z]); },
            end() {},
          };
          return {
            beginComputePass() { calls.passes++; return pass; },
            finish() { return {}; },
          };
        },
      };
      const gpuApi = {
        async requestAdapter() {
          return {
            info: { description: "Tach interface-test adapter" },
            async requestDevice() { return device; },
          };
        },
      };

      const generated = await import(entry.module);
      const metadata = await (await fetch(entry.metadata)).json();
      const kernel = metadata.kernels[0];
      const executed = await tach(async (gpu) => {
        const args = [];
        const computeBuffers = [];
        let initialRead;

        for (const parameter of kernel.parameters) {
          const value = inputs[entry.name][parameter.name];
          if (parameter.kind === "buffer") {
            const computeBuffer = gpu.buffer(value);
            initialRead ??= await computeBuffer.read();
            computeBuffers.push(computeBuffer);
            args.push(computeBuffer);
          } else {
            args.push(value);
          }
        }

        const lazyBufferCount = calls.buffers;
        await gpu.submit(
          generated[kernel.name](...args),
          generated[kernel.name](...args, {
            size: kernel.workgroupSize[0] + 1,
            repeat: 2,
          }),
        );
        const firstBuffer = kernel.parameters.find((parameter) => parameter.kind === "buffer");
        computeBuffers[0].write(inputs[entry.name][firstBuffer.name]);
        return {
          name: entry.name,
          exports: Object.keys(generated),
          kernelName: kernel.name,
          storageCount: computeBuffers.length,
          hasParameterBlock: kernel.parameterBlock !== undefined,
          lazyBufferCount,
          initialRead,
        };
      }, { gpu: gpuApi });
      summaries.push({ ...executed, calls });
    }
    return summaries;
  }, values);

  expect(result.map((summary) => summary.name)).toEqual(exampleNames);
  for (const summary of result) {
    expect(summary.exports).toEqual([summary.kernelName]);
    expect(summary.lazyBufferCount).toBe(0);
    expect(summary.initialRead).toBeDefined();
    expect(summary.calls.shaders).toBe(1);
    expect(summary.calls.pipelines).toBe(1);
    expect(summary.calls.buffers).toBe(summary.storageCount + (summary.hasParameterBlock ? 1 : 0));
    expect(summary.calls.bindGroups).toBe(1);
    expect(summary.calls.dispatches).toEqual([[1, 1, 1], [2, 1, 1], [2, 1, 1]]);
    expect(summary.calls.passes).toBe(1);
    expect(summary.calls.submits).toBe(1);
    expect(summary.calls.workDone).toBe(1);
    expect(summary.calls.writes).toBe(1 + (summary.hasParameterBlock ? 1 : 0));
    expect(summary.calls.destroyed).toBe(summary.storageCount + (summary.hasParameterBlock ? 1 : 0));
    expect(summary.calls.scopesPushed).toBe(6);
    expect(summary.calls.scopesPopped).toBe(6);
    expect(summary.calls.deviceDestroyed).toBe(1);
  }
});
