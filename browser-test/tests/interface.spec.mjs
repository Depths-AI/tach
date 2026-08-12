import { expect, test } from "@playwright/test";

const exampleNames = ["atomics", "bitwise", "control", "for", "fusion", "math", "particles", "scalars"];

test("inspection UI loads every schema-1 example", async ({ page }) => {
  const pageErrors = [];
  page.on("pageerror", (error) => pageErrors.push(error.message));
  await page.goto("/");
  await page.evaluate(() => globalThis.__tachHarnessReady);
  await expect(page.locator("#examples tr")).toHaveCount(exampleNames.length);
  expect(await page.locator("#examples td:first-child").allTextContents()).toEqual(exampleNames);
  expect(pageErrors).toEqual([]);
});

test("metadata separates public programs from private physical kernels", async ({ page }) => {
  await page.goto("/");
  const modules = await page.evaluate(async () => {
    const manifest = await (await fetch("/build/manifest.json")).json();
    return Promise.all(manifest.examples.map(async (entry) => {
      const metadata = await (await fetch(entry.metadata)).json();
      return {
        name: entry.name,
        schema: metadata.schema,
        keys: Object.keys(metadata).sort(),
        programs: metadata.programs.map((program) => program.name),
        entries: metadata.targets.web.kernels.map((kernel) => kernel.entryPoint),
        plans: metadata.targets.web.programs,
      };
    }));
  });
  for (const module of modules) {
    expect(module.schema).toBe(1);
    expect(module.keys).toEqual(["programs", "schema", "targets", "types"]);
    expect(module.entries.every((entry, index) => entry === `_tach_k${index}`)).toBe(true);
    expect(module.programs.every((name) => !module.entries.includes(name))).toBe(true);
    expect(module.plans).toHaveLength(module.programs.length);
  }
  const fusion = modules.find((module) => module.name === "fusion");
  expect(fusion.programs).toEqual(["transform", "neighbor"]);
  expect(fusion.plans[0].steps.filter((step) => step.kind === "dispatch")).toHaveLength(1);
  expect(fusion.plans[0].transients).toHaveLength(0);
  expect(fusion.plans[1].steps.filter((step) => step.kind === "dispatch")).toHaveLength(2);
});

test("one generated command owns fused or multi-dispatch execution", async ({ page }) => {
  await page.goto("/");
  const calls = await page.evaluate(async () => {
    const observed = { bindGroups: 0, destroyed: 0, dispatches: 0, passes: 0, pipelines: 0, submits: 0 };
    const device = {
      limits: { minUniformBufferOffsetAlignment: 256 },
      lost: new Promise(() => {}),
      queue: { writeBuffer() {}, submit() { observed.submits++; }, async onSubmittedWorkDone() {} },
      addEventListener() {}, removeEventListener() {}, pushErrorScope() {}, async popErrorScope() { return null; },
      destroy() {}, createShaderModule(descriptor) { return { descriptor }; },
      createBindGroupLayout(descriptor) { return { descriptor }; }, createPipelineLayout(descriptor) { return { descriptor }; },
      async createComputePipelineAsync(descriptor) { observed.pipelines++; return { descriptor }; },
      createBindGroup(descriptor) { observed.bindGroups++; return { descriptor }; },
      createBuffer(descriptor) {
        const storage = new ArrayBuffer(descriptor.size);
        return { descriptor, size: descriptor.size, destroy() { observed.destroyed++; }, getMappedRange() { return storage; }, unmap() {} };
      },
      createCommandEncoder() {
        const pass = { setPipeline() {}, setBindGroup() {}, dispatchWorkgroups() { observed.dispatches++; }, end() {} };
        return { beginComputePass() { observed.passes++; return pass; }, finish() { return {}; } };
      },
    };
    const { tach } = await import("@depths/tach");
    const { transform, neighbor } = await import("/build/fusion.js");
    await tach(async (gpu) => {
      const input = gpu.buffer(new Float32Array([1, 2, 3, 4]));
      const output = gpu.buffer(new Float32Array(4));
      const values = gpu.buffer(new Float32Array([1, 2, 3, 4]));
      await gpu.submit(transform(input, output, 2, 1, { repeat: 2 }), neighbor(values, { repeat: 2 }));
    }, { gpu: { async requestAdapter() { return { info: {}, async requestDevice() { return device; } }; } } });
    return observed;
  });
  expect(calls.pipelines).toBe(3);
  expect(calls.dispatches).toBe(5);
  expect(calls.passes).toBe(1);
  expect(calls.submits).toBe(1);
});
