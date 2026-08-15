import type {
  Driver,
  DriverBuffer,
  KernelDefinition,
  ModuleDefinition,
  PreparedCommand,
  PreparedStep,
} from "./driver.ts";
import { normalizeError, TachError, type TachErrorCode } from "./api.ts";
import type { TachOptions } from "./api.ts";

const usage = {
  mapRead: 0x0001,
  copySrc: 0x0004,
  copyDst: 0x0008,
  uniform: 0x0040,
  storage: 0x0080,
} as const;
const shaderStageCompute = 0x0004;
const noFailure = Symbol();

interface CompiledKernel {
  readonly layout: GPUBindGroupLayout;
  readonly pipeline: GPUComputePipeline;
}
interface ModuleCache {
  source?: Promise<string>;
  module?: GPUShaderModule;
  readonly pipelines: Map<number, Promise<CompiledKernel>>;
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
  readonly #uncaptured: GPUError[] = [];
  #nextObjectID = 1;
  #parameters?: GPUBuffer;
  #parameterCapacity = 0;
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
    this.device.queue.writeBuffer(
      gpuBuffer(buffer),
      0,
      bytes.buffer,
      bytes.byteOffset,
      bytes.byteLength,
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
          byteLength,
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
      pending = this.#capture(kernel.entryPoint, "kernel", async () => {
        cache!.source ??= fetch(command.shader).then((response) => {
          if (!response.ok) {
            throw new Error(`could not load WGSL (${response.status})`);
          }
          return response.text();
        });
        cache!.module ??= this.device.createShaderModule({
          label: "Tach shader module",
          code: await cache!.source,
        });
        return this.#compilePipeline(cache!.module!, kernel);
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
    const entries: GPUBindGroupLayoutEntry[] = kernel.bindings.map((
      binding,
    ) => ({
      binding: binding.binding,
      visibility: shaderStageCompute,
      buffer: {
        type: binding.access === "read_write" ? "storage" : "read-only-storage",
        minBindingSize: binding.minimumByteSize,
      },
    }));
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

  async submit(commands: readonly PreparedCommand[]): Promise<void> {
    const compiled = new Map<PreparedCommand, Map<number, CompiledKernel>>();
    await Promise.all(commands.map(async (command) => {
      const kernels = new Map<number, CompiledKernel>();
      compiled.set(command, kernels);
      const indices = new Set(
        command.steps.flatMap((step) =>
          step.kind === "dispatch" ? [step.kernel] : []
        ),
      );
      await Promise.all(
        [...indices].map(async (index) =>
          kernels.set(index, await this.#compile(command, index))
        ),
      );
    }));
    await this.#capture("submit", "kernel", () => {
      const scratch = this.#resolveScratch(commands),
        parameters = this.#writeParameters(commands);
      const encoder = this.device.createCommandEncoder({
          label: "Tach submission",
        }),
        pass = encoder.beginComputePass({ label: "Tach compute pass" });
      for (const command of commands) {
        const record = () => {
          for (const step of command.steps) {
            if (step.kind === "dispatch") {
              this.#dispatch(
                pass,
                command,
                step,
                compiled.get(command)!,
                scratch,
                parameters.get(step),
              );
            }
          }
        };
        for (let repeat = 0; repeat < command.repeat; repeat++) record();
      }
      pass.end();
      this.device.queue.submit([encoder.finish()]);
    });
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
        size: resource.byteSize,
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
  ): ReadonlyMap<PreparedStep, GPUBufferBinding> {
    const steps: Extract<PreparedStep, { readonly kind: "dispatch" }>[] =
      commands.flatMap((command) =>
        command.steps.filter((
          step,
        ): step is Extract<PreparedStep, { readonly kind: "dispatch" }> =>
          step.kind === "dispatch" && step.parameters !== undefined
        )
      );
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
      bindings = new Map<PreparedStep, GPUBufferBinding>();
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
    this.#healthy("idle");
  }

  close(): void {
    if (this.#closed) return;
    this.#parameters?.destroy();
    for (const allocation of this.#scratch.values()) {
      allocation.buffer.destroy();
    }
    for (const buffer of this.#retired) buffer.destroy();
    this.#scratch.clear();
    this.#retired.length = 0;
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
    return new WebDriver(adapter, await adapter.requestDevice());
  } catch (cause) {
    throw normalizeError(cause, "device-request-failed", "requestDevice");
  }
}
