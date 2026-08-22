import type {
  Driver,
  DriverBuffer,
  KernelDefinition,
  ModuleDefinition,
  PreparedCommand,
  PreparedStep,
  PreparedView,
} from "./driver.ts";
import { normalizeError, TachError, type TachErrorCode } from "./api.ts";
import type { PresentationCanvas, TachOptions } from "./api.ts";

const usage = {
  mapRead: 0x0001,
  copySrc: 0x0004,
  copyDst: 0x0008,
  uniform: 0x0040,
  storage: 0x0080,
} as const;
const shaderStageCompute = 0x0004;
const textureUsage = { storage: 0x08, render: 0x10 } as const;
const noFailure = Symbol();

interface CompiledKernel {
  readonly layout: GPUBindGroupLayout;
  readonly pipeline: GPUComputePipeline;
}
interface ModuleCache {
  module?: Promise<GPUShaderModule>;
  readonly pipelines: Map<number, Promise<CompiledKernel>>;
}

const shaderCacheName = "tach-wgsl-v1";
const shaderCacheLifetime = 7 * 24 * 60 * 60 * 1000;

async function shaderSource(url: URL): Promise<string> {
  let cache: Cache | undefined;
  if (
    (url.protocol === "http:" || url.protocol === "https:") &&
    typeof caches !== "undefined"
  ) {
    try {
      cache = await caches.open(shaderCacheName);
      const now = Date.now();
      // DECISION: O(n) expiry keeps one standards-only store; shard it if an
      // origin ever accumulates thousands of generated Tach projects.
      await Promise.all((await cache.keys()).map(async (request) => {
        const stored = await cache!.match(request),
          expires = Number(stored?.headers.get("x-tach-expires"));
        if (!Number.isSafeInteger(expires) || expires <= now) {
          await cache!.delete(request);
        }
      }));
      const cached = await cache.match(url.href);
      if (cached) {
        const expires = Number(cached.headers.get("x-tach-expires"));
        if (Number.isSafeInteger(expires) && expires > now) {
          return await cached.text();
        }
      }
    } catch { /* Persistent caching must not gate GPU execution. */ }
  }
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`could not load WGSL (${response.status})`);
  }
  if (!response.body) throw new Error("WGSL response has no body");
  const source = await new Response(
    response.body.pipeThrough(new DecompressionStream("gzip")),
  ).text();
  if (cache) {
    void cache.put(
      url.href,
      new Response(source, {
        headers: {
          "content-type": "text/plain",
          "x-tach-expires": String(Date.now() + shaderCacheLifetime),
        },
      }),
    ).catch(() => {});
  }
  return source;
}

function gpuBuffer(buffer: DriverBuffer | undefined): GPUBuffer {
  if (!buffer || typeof buffer === "number") {
    throw new TypeError("invalid WebGPU buffer");
  }
  return buffer as GPUBuffer;
}
function align(value: number, alignment: number): number {
  return Math.ceil(value / alignment) * alignment;
}

class WebDriver implements Driver {
  readonly adapter;
  readonly #modules = new WeakMap<ModuleDefinition, ModuleCache>();
  readonly #bindGroups = new Map<string, GPUBindGroup>();
  readonly #objectIDs = new WeakMap<object, number>();
  readonly #scratch = new Map<
    number,
    { buffer: GPUBuffer; capacity: number }
  >();
  readonly #retired: GPUBuffer[] = [];
  readonly #retiredTextures: GPUTexture[] = [];
  readonly #uncaptured: GPUError[] = [];
  #nextObjectID = 1;
  #parameters?: GPUBuffer;
  #parameterCapacity = 0;
  #offscreen?: { texture: GPUTexture; width: number; height: number };
  readonly #canvases = new WeakMap<object, GPUCanvasContext>();
  #closed = false;
  #lost?: GPUDeviceLostInfo;

  constructor(readonly gpuAdapter: GPUAdapter, readonly device: GPUDevice) {
    const info = gpuAdapter.info;
    this.adapter = {
      backend: "webgpu" as const,
      name: info.description || info.device || "WebGPU adapter",
      ...(info.vendor ? { vendor: info.vendor } : {}),
      ...(info.architecture ? { architecture: info.architecture } : {}),
      type: "unknown" as const,
    };
    device.addEventListener("uncapturederror", this.#onUncaptured);
    void device.lost.then((lost) => {
      if (!this.#closed) this.#lost = lost;
    });
  }

  readonly #onUncaptured = (event: GPUUncapturedErrorEvent): void => {
    event.preventDefault();
    this.#uncaptured.push(event.error);
  };

  #healthy(operation: string): void {
    if (this.#closed) {
      throw new TachError("lifecycle", "WebGPU driver is closed", {
        operation,
      });
    }
    if (this.#lost) {
      throw new TachError(
        "device-lost",
        this.#lost.message || `GPU device was lost (${this.#lost.reason})`,
        { operation, cause: this.#lost },
      );
    }
    const error = this.#uncaptured.shift();
    if (error) throw normalizeError(error, "gpu-internal", operation);
  }

  async #capture<T>(
    operation: string,
    fallback: TachErrorCode,
    issue: () => T | Promise<T>,
  ): Promise<T> {
    this.#healthy(operation);
    this.device.pushErrorScope("internal");
    this.device.pushErrorScope("out-of-memory");
    this.device.pushErrorScope("validation");
    let pending: T | Promise<T> | undefined, failure: unknown = noFailure;
    try {
      pending = issue();
    } catch (cause) {
      failure = cause;
    }
    const scopes = [
      this.device.popErrorScope(),
      this.device.popErrorScope(),
      this.device.popErrorScope(),
    ];
    let value: T | undefined;
    if (failure === noFailure) {
      try {
        value = await pending;
      } catch (cause) {
        failure = cause;
      }
    }
    let errors: readonly (GPUError | null)[] = [];
    try {
      errors = await Promise.all(scopes);
    } catch (cause) {
      if (failure === noFailure) failure = cause;
    }
    if (failure !== noFailure) {
      throw normalizeError(failure, fallback, operation);
    }
    const error = errors.find((candidate): candidate is GPUError =>
      candidate !== null
    );
    if (error) throw normalizeError(error, fallback, operation);
    this.#healthy(operation);
    return value as T;
  }

  createBuffer(label: string, bytes: Uint8Array): DriverBuffer {
    this.#healthy("buffer");
    const buffer = this.device.createBuffer({
      label,
      size: Math.max(4, align(bytes.byteLength, 4)),
      usage: usage.storage | usage.copySrc | usage.copyDst,
      mappedAtCreation: true,
    });
    try {
      new Uint8Array(buffer.getMappedRange()).set(bytes);
      buffer.unmap();
      return buffer;
    } catch (cause) {
      buffer.destroy();
      throw cause;
    }
  }

  writeBuffer(buffer: DriverBuffer, bytes: Uint8Array): void {
    this.#healthy("buffer.write");
    const upload = bytes.byteLength % 4 === 0
      ? bytes
      : new Uint8Array(align(bytes.byteLength, 4));
    if (upload !== bytes) upload.set(bytes);
    this.device.queue.writeBuffer(
      gpuBuffer(buffer),
      0,
      upload.buffer,
      upload.byteOffset,
      upload.byteLength,
    );
  }

  async readBuffer(
    buffer: DriverBuffer,
    byteLength: number,
  ): Promise<Uint8Array> {
    const readback = this.device.createBuffer({
      label: "Tach readback",
      size: Math.max(4, align(byteLength, 4)),
      usage: usage.copyDst | usage.mapRead,
    });
    let mapped = false;
    try {
      return await this.#capture("buffer.read", "buffer", async () => {
        const encoder = this.device.createCommandEncoder({
          label: "Tach readback commands",
        });
        encoder.copyBufferToBuffer(
          gpuBuffer(buffer),
          0,
          readback,
          0,
          align(byteLength, 4),
        );
        this.device.queue.submit([encoder.finish()]);
        await readback.mapAsync(1);
        mapped = true;
        return new Uint8Array(readback.getMappedRange()).slice(0, byteLength);
      });
    } finally {
      if (mapped) readback.unmap();
      readback.destroy();
    }
  }

  destroyBuffer(buffer: DriverBuffer): void {
    gpuBuffer(buffer).destroy();
    this.#bindGroups.clear();
  }

  async #compile(
    command: PreparedCommand,
    index: number,
  ): Promise<CompiledKernel> {
    let cache = this.#modules.get(command.module);
    if (!cache) {
      cache = { pipelines: new Map() };
      this.#modules.set(command.module, cache);
    }
    let pending = cache.pipelines.get(index);
    if (!pending) {
      const kernel = command.target.kernels[index];
      if (!kernel) throw new TypeError(`invalid WebGPU kernel ${index}`);
      for (const feature of command.target.features ?? []) {
        if (!this.device.features.has(feature as GPUFeatureName)) {
          throw new TachError(
            "device-request-failed",
            `This command requires Tach float16, but the GPU does not support ${feature}; use float32 or a Float16-capable adapter`,
            { operation: kernel.entryPoint },
          );
        }
      }
      pending = this.#capture(kernel.entryPoint, "kernel", async () => {
        cache!.module ??= shaderSource(command.shader).then(async (code) => {
          const module = this.device.createShaderModule({
            label: "Tach shader module",
            code,
          });
          const errors = (await module.getCompilationInfo()).messages.filter(
            (message) => message.type === "error",
          );
          if (errors.length) {
            throw new Error(
              errors.map((message) => message.message).join("\n"),
            );
          }
          return module;
        });
        return this.#compilePipeline(await cache!.module, kernel);
      });
      cache.pipelines.set(index, pending);
    }
    try {
      return await pending;
    } catch (cause) {
      cache.pipelines.delete(index);
      throw cause;
    }
  }

  async #compilePipeline(
    module: GPUShaderModule,
    kernel: KernelDefinition,
  ): Promise<CompiledKernel> {
    const entries: GPUBindGroupLayoutEntry[] = kernel.bindings.map((binding) =>
      binding.kind === "texture"
        ? {
          binding: binding.binding,
          visibility: shaderStageCompute,
          storageTexture: {
            access: "write-only",
            format: "rgba8unorm",
            viewDimension: "2d",
          },
        }
        : {
          binding: binding.binding,
          visibility: shaderStageCompute,
          buffer: {
            type: binding.access === "read_write"
              ? "storage"
              : "read-only-storage",
            minBindingSize: binding.minimumByteSize,
          },
        }
    );
    if (kernel.parameterBlock) {
      entries.push({
        binding: kernel.parameterBlock.binding,
        visibility: shaderStageCompute,
        buffer: {
          type: "uniform",
          minBindingSize: kernel.parameterBlock.byteSize,
          hasDynamicOffset: true,
        },
      });
    }
    entries.sort((left, right) => left.binding - right.binding);
    const layout = this.device.createBindGroupLayout({
      label: `Tach ${kernel.entryPoint} group 0`,
      entries,
    });
    const pipelineLayout = this.device.createPipelineLayout({
      label: `Tach ${kernel.entryPoint} layout`,
      bindGroupLayouts: [layout],
    });
    const pipeline = await this.device.createComputePipelineAsync({
      label: `Tach ${kernel.entryPoint}`,
      layout: pipelineLayout,
      compute: { module, entryPoint: kernel.entryPoint },
    });
    return { layout, pipeline };
  }

  async #compileCommands(
    commands: readonly PreparedCommand[],
  ): Promise<Map<PreparedCommand, Map<number, CompiledKernel>>> {
    const compiled = new Map<PreparedCommand, Map<number, CompiledKernel>>();
    await Promise.all(commands.map(async (command) => {
      const kernels = new Map<number, CompiledKernel>();
      compiled.set(command, kernels);
      const indices = new Set(
        command.steps.flatMap((step) =>
          step.kind === "dispatch" ? [step.kernel] : []
        ),
      );
      if (command.view) indices.add(command.view.kernel);
      await Promise.all(
        [...indices].map(async (index) =>
          kernels.set(index, await this.#compile(command, index))
        ),
      );
    }));
    return compiled;
  }

  async prepare(commands: readonly PreparedCommand[]): Promise<void> {
    await this.#compileCommands(commands);
  }

  async submit(commands: readonly PreparedCommand[]): Promise<void> {
    const compiled = await this.#compileCommands(commands);
    await this.#capture("submit", "kernel", () => {
      const scratch = this.#resolveScratch(commands),
        parameters = this.#writeParameters(commands);
      const encoder = this.device.createCommandEncoder({
          label: "Tach submission",
        }),
        pass = encoder.beginComputePass({ label: "Tach compute pass" });
      for (const command of commands) {
        this.#dispatchCommand(
          pass,
          command,
          compiled.get(command)!,
          scratch,
          parameters,
        );
        if (command.view) {
          this.#dispatchView(
            pass,
            command,
            command.view,
            compiled.get(command)!,
            scratch,
            parameters.get(command.view),
            this.#offscreenTexture(command.view),
          );
        }
      }
      pass.end();
      this.device.queue.submit([encoder.finish()]);
    });
  }

  async present(
    canvas: PresentationCanvas,
    command: PreparedCommand,
  ): Promise<void> {
    const view = command.view;
    if (!view) {
      throw new TachError("kernel", "command has no view", {
        operation: "present",
      });
    }
    if (canvas.width !== view.width || canvas.height !== view.height) {
      throw new TachError(
        "kernel",
        "canvas dimensions must equal the view extent",
        { operation: "present" },
      );
    }
    const compiled = await this.#compileCommands([command]);
    await this.#capture("present", "kernel", async () => {
      const scratch = this.#resolveScratch([command]),
        parameters = this.#writeParameters([command]),
        encoder = this.device.createCommandEncoder({
          label: "Tach presentation",
        }),
        pass = encoder.beginComputePass({ label: "Tach presentation pass" });
      this.#dispatchCommand(
        pass,
        command,
        compiled.get(command)!,
        scratch,
        parameters,
      );
      this.#dispatchView(
        pass,
        command,
        view,
        compiled.get(command)!,
        scratch,
        parameters.get(view),
        this.#canvasTexture(canvas),
      );
      pass.end();
      this.device.queue.submit([encoder.finish()]);
      await this.device.queue.onSubmittedWorkDone();
    });
  }

  #offscreenTexture(view: PreparedView): GPUTexture {
    if (
      this.#offscreen?.width === view.width &&
      this.#offscreen.height === view.height
    ) {
      return this.#offscreen.texture;
    }
    if (this.#offscreen) this.#retiredTextures.push(this.#offscreen.texture);
    const texture = this.device.createTexture({
      label: "Tach offscreen view",
      size: [view.width, view.height],
      format: "rgba8unorm",
      usage: textureUsage.storage,
    });
    this.#offscreen = { texture, width: view.width, height: view.height };
    return texture;
  }

  #canvasTexture(canvas: PresentationCanvas): GPUTexture {
    let context = this.#canvases.get(canvas);
    if (!context) {
      context = canvas.getContext("webgpu") as GPUCanvasContext | null ??
        undefined;
      if (!context) {
        throw new TachError(
          "webgpu-unavailable",
          "canvas has no WebGPU context",
          { operation: "present" },
        );
      }
      context.configure({
        device: this.device,
        format: "rgba8unorm",
        usage: textureUsage.storage | textureUsage.render,
        alphaMode: "opaque",
      });
      this.#canvases.set(canvas, context);
    }
    return context.getCurrentTexture();
  }

  #dispatchView(
    pass: GPUComputePassEncoder,
    command: PreparedCommand,
    view: PreparedView,
    compiled: Map<number, CompiledKernel>,
    scratch: ReadonlyMap<number, GPUBuffer>,
    parameter: GPUBufferBinding | undefined,
    target: GPUTexture,
  ): void {
    const kernel = command.target.kernels[view.kernel]!,
      pipeline = compiled.get(view.kernel)!;
    if (kernel.parameterBlock && !parameter) {
      throw new TypeError("view parameters are missing");
    }
    const entries: GPUBindGroupEntry[] = view.resources.map((resource) => ({
      binding: resource.binding,
      resource: {
        buffer: resource.buffer === undefined
          ? scratch.get(resource.scratch!)!
          : gpuBuffer(resource.buffer),
        size: align(resource.byteSize, 4),
      },
    }));
    entries.push({ binding: view.output, resource: target.createView() });
    const dynamic: number[] = [];
    if (kernel.parameterBlock && parameter) {
      entries.push({
        binding: kernel.parameterBlock.binding,
        resource: {
          buffer: parameter.buffer,
          size: kernel.parameterBlock.byteSize,
        },
      });
      dynamic.push(parameter.offset ?? 0);
    }
    entries.sort((left, right) => left.binding - right.binding);
    const group = this.device.createBindGroup({
      label: "Tach view",
      layout: pipeline.layout,
      entries,
    });
    pass.setPipeline(pipeline.pipeline);
    pass.setBindGroup(0, group, dynamic);
    pass.dispatchWorkgroups(...view.groups);
  }

  #dispatchCommand(
    pass: GPUComputePassEncoder,
    command: PreparedCommand,
    compiled: Map<number, CompiledKernel>,
    scratch: ReadonlyMap<number, GPUBuffer>,
    parameters: ReadonlyMap<object, GPUBufferBinding>,
  ): void {
    for (let repeat = 0; repeat < command.repeat; repeat++) {
      for (const step of command.steps) {
        if (step.kind === "dispatch") {
          this.#dispatch(
            pass,
            command,
            step,
            compiled,
            scratch,
            parameters.get(step),
          );
        }
      }
    }
  }

  #dispatch(
    pass: GPUComputePassEncoder,
    command: PreparedCommand,
    step: Extract<PreparedStep, { readonly kind: "dispatch" }>,
    compiled: Map<number, CompiledKernel>,
    scratch: ReadonlyMap<number, GPUBuffer>,
    parameter?: GPUBufferBinding,
  ): void {
    const kernel = command.target.kernels[step.kernel]!,
      pipeline = compiled.get(step.kernel)!;
    const entries = step.resources.map((resource) => ({
      binding: resource.binding,
      resource: {
        buffer: resource.buffer === undefined
          ? scratch.get(resource.scratch!)!
          : gpuBuffer(resource.buffer),
        size: align(resource.byteSize, 4),
      },
    }));
    const dynamic: number[] = [];
    if (kernel.parameterBlock && parameter) {
      entries.push({
        binding: kernel.parameterBlock.binding,
        resource: {
          buffer: parameter.buffer,
          size: kernel.parameterBlock.byteSize,
        },
      });
      dynamic.push(parameter.offset ?? 0);
    }
    entries.sort((left, right) => left.binding - right.binding);
    pass.setPipeline(pipeline.pipeline);
    pass.setBindGroup(
      0,
      this.#bindGroup(kernel.entryPoint, pipeline.layout, entries),
      dynamic,
    );
    pass.dispatchWorkgroups(...step.groups);
  }

  #bindGroup(
    label: string,
    layout: GPUBindGroupLayout,
    entries: readonly GPUBindGroupEntry[],
  ): GPUBindGroup {
    const key = [
      this.#objectID(layout),
      ...entries.map((entry) => {
        const resource = entry.resource as GPUBufferBinding;
        return `${entry.binding}:${this.#objectID(resource.buffer)}:${
          resource.offset ?? 0
        }:${resource.size ?? resource.buffer.size}`;
      }),
    ].join("|");
    let group = this.#bindGroups.get(key);
    if (!group) {
      group = this.device.createBindGroup({
        label: `Tach ${label} group 0`,
        layout,
        entries: [...entries],
      });
      this.#bindGroups.set(key, group);
    }
    return group;
  }

  #objectID(value: object): number {
    let id = this.#objectIDs.get(value);
    if (id === undefined) {
      id = this.#nextObjectID++;
      this.#objectIDs.set(value, id);
    }
    return id;
  }

  #resolveScratch(
    commands: readonly PreparedCommand[],
  ): ReadonlyMap<number, GPUBuffer> {
    const required = new Map<number, number>();
    for (const command of commands) {
      for (const [color, bytes] of command.scratch) {
        required.set(color, Math.max(required.get(color) ?? 0, bytes));
      }
    }
    const resolved = new Map<number, GPUBuffer>();
    for (const [color, bytes] of required) {
      let allocation = this.#scratch.get(color);
      if (!allocation || allocation.capacity < bytes) {
        let capacity = Math.max(4096, allocation?.capacity ?? 0);
        while (capacity < bytes) capacity *= 2;
        const buffer = this.device.createBuffer({
          label: `Tach scratch ${color}`,
          size: align(capacity, 4),
          usage: usage.storage,
        });
        if (allocation) this.#retired.push(allocation.buffer);
        allocation = { buffer, capacity };
        this.#scratch.set(color, allocation);
        this.#bindGroups.clear();
      }
      resolved.set(color, allocation.buffer);
    }
    return resolved;
  }

  #writeParameters(
    commands: readonly PreparedCommand[],
  ): ReadonlyMap<object, GPUBufferBinding> {
    const steps: { readonly parameters: Uint8Array }[] = commands.flatMap((
      command,
    ) => [
      ...command.steps.filter((
        step,
      ): step is Extract<PreparedStep, { readonly kind: "dispatch" }> & {
        readonly parameters: Uint8Array;
      } => step.kind === "dispatch" && step.parameters !== undefined),
      ...(command.view?.parameters
        ? [
          command.view as PreparedView & {
            readonly parameters: Uint8Array;
          },
        ]
        : []),
    ]);
    if (steps.length === 0) return new Map();
    const alignment = this.device.limits.minUniformBufferOffsetAlignment,
      offsets: number[] = [];
    let byteLength = 0;
    for (const step of steps) {
      byteLength = align(byteLength, alignment);
      offsets.push(byteLength);
      byteLength += step.parameters!.byteLength;
    }
    const buffer = this.#parameterBuffer(byteLength),
      upload = new Uint8Array(byteLength),
      bindings = new Map<object, GPUBufferBinding>();
    steps.forEach((step, index) => {
      const parameters = step.parameters!, offset = offsets[index]!;
      upload.set(parameters, offset);
      bindings.set(step, { buffer, offset, size: parameters.byteLength });
    });
    this.device.queue.writeBuffer(
      buffer,
      0,
      upload.buffer,
      upload.byteOffset,
      upload.byteLength,
    );
    return bindings;
  }

  #parameterBuffer(bytes: number): GPUBuffer {
    if (this.#parameters && this.#parameterCapacity >= bytes) {
      return this.#parameters;
    }
    let capacity = Math.max(4096, this.#parameterCapacity);
    while (capacity < bytes) capacity *= 2;
    const buffer = this.device.createBuffer({
      label: "Tach parameter arena",
      size: align(capacity, 4),
      usage: usage.copyDst | usage.uniform,
    });
    if (this.#parameters) this.#retired.push(this.#parameters);
    this.#parameters = buffer;
    this.#parameterCapacity = capacity;
    this.#bindGroups.clear();
    return buffer;
  }

  async idle(): Promise<void> {
    this.#healthy("idle");
    await this.device.queue.onSubmittedWorkDone();
    for (const buffer of this.#retired.splice(0)) buffer.destroy();
    for (const texture of this.#retiredTextures.splice(0)) texture.destroy();
    this.#healthy("idle");
  }

  close(): void {
    if (this.#closed) return;
    this.#parameters?.destroy();
    this.#offscreen?.texture.destroy();
    for (const allocation of this.#scratch.values()) {
      allocation.buffer.destroy();
    }
    for (const buffer of this.#retired) buffer.destroy();
    for (const texture of this.#retiredTextures) texture.destroy();
    this.#scratch.clear();
    this.#retired.length = 0;
    this.#retiredTextures.length = 0;
    this.#bindGroups.clear();
    this.device.removeEventListener("uncapturederror", this.#onUncaptured);
    this.#closed = true;
    this.device.destroy();
  }
}

export async function openWeb(
  options: TachOptions,
  gpu = typeof navigator === "undefined" ? undefined : navigator.gpu,
): Promise<Driver> {
  if (!gpu) {
    throw new TachError(
      "webgpu-unavailable",
      "WebGPU is unavailable in this environment",
      { operation: "tach" },
    );
  }
  let adapter: GPUAdapter | null;
  try {
    adapter = await gpu.requestAdapter(
      options.powerPreference
        ? { powerPreference: options.powerPreference }
        : undefined,
    );
  } catch (cause) {
    throw normalizeError(cause, "adapter-unavailable", "requestAdapter");
  }
  if (!adapter) {
    throw new TachError(
      "adapter-unavailable",
      "WebGPU did not provide an adapter",
      { operation: "requestAdapter" },
    );
  }
  try {
    const requiredFeatures: GPUFeatureName[] = [];
    if (adapter.features.has("shader-f16")) requiredFeatures.push("shader-f16");
    return new WebDriver(
      adapter,
      await adapter.requestDevice({ requiredFeatures }),
    );
  } catch (cause) {
    throw normalizeError(cause, "device-request-failed", "requestDevice");
  }
}
