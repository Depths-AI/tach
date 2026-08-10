import {
  createComputeDispatch,
  getBufferState,
  type BufferBindGroupEntry,
  type BufferCodec,
  type BufferState,
  type ComputeBuffer,
  type ComputeDispatch,
  type DispatchOptions,
  type DispatchSize,
  type PreparedDispatch,
  type RuntimeOwner,
} from "./runtime.js";
import { normalizeError, TachFailure, tachError } from "./result.js";

const bufferUsage = {
  copySrc: 0x0004,
  copyDst: 0x0008,
  storage: 0x0080,
} as const;

const shaderStage = {
  compute: 0x0004,
} as const;

interface HostLayoutField {
  readonly name: string;
  readonly offset: number;
  readonly type: HostLayout;
}

interface HostLayout {
  readonly kind: "i32" | "u32" | "f32" | "vector" | "array" | "runtime" | "struct";
  readonly size?: number;
  readonly stride?: number;
  readonly count?: number;
  readonly runtime?: boolean;
  readonly elem?: HostLayout;
  readonly fields?: readonly HostLayoutField[];
}

interface ResourceDefinition {
  readonly name: string;
  readonly group: number;
  readonly binding: number;
  readonly kind: "uniform" | "storage";
  readonly access: "read" | "read_write";
  readonly byteSize?: number;
  readonly minimumByteSize: number;
  readonly runtime: boolean;
  readonly layout: HostLayout;
}

interface KernelResourceDefinition {
  readonly name: string;
  readonly resource: number;
}

interface KernelDefinition {
  readonly name: string;
  readonly entryPoint: string;
  readonly workgroupSize: readonly [number, number, number];
  readonly resources: readonly KernelResourceDefinition[];
}

export interface ModuleDefinition {
  readonly source: string;
  readonly resources: readonly ResourceDefinition[];
  readonly kernels: readonly KernelDefinition[];
}

interface CompiledKernel {
  readonly bindGroupLayouts: readonly GPUBindGroupLayout[];
  readonly pipeline: GPUComputePipeline;
}

interface DeviceCache {
  module?: GPUShaderModule;
  readonly pipelines: Map<number, Promise<CompiledKernel>>;
}

export interface DefinedModule {
  dispatch(
    kernel: number,
    values: readonly unknown[],
    options?: DispatchOptions,
  ): ComputeDispatch;
}

function sequence(value: unknown, path: string): ArrayLike<unknown> {
  const length = value === null || value === undefined
    ? undefined
    : (value as Partial<ArrayLike<unknown>>).length;
  if (typeof value === "string" || !Number.isSafeInteger(length) || (length ?? -1) < 0) {
    throw new TypeError(`${path} must be an array or typed array`);
  }
  return value as ArrayLike<unknown>;
}

function number(value: unknown, path: string): number {
  if (typeof value !== "number") throw new TypeError(`${path} must be a number`);
  return value;
}

function required<T>(value: T | undefined, description: string): T {
  if (value === undefined) throw new TypeError(`invalid generated layout: missing ${description}`);
  return value;
}

function writeValue(
  view: DataView,
  offset: number,
  type: HostLayout,
  value: unknown,
  path: string,
): void {
  switch (type.kind) {
    case "i32": {
      const scalar = number(value, path);
      if (!Number.isInteger(scalar) || scalar < -2_147_483_648 || scalar > 2_147_483_647) {
        throw new RangeError(`${path} must be a signed 32-bit integer`);
      }
      view.setInt32(offset, scalar, true);
      return;
    }
    case "u32": {
      const scalar = number(value, path);
      if (!Number.isInteger(scalar) || scalar < 0 || scalar > 4_294_967_295) {
        throw new RangeError(`${path} must be an unsigned 32-bit integer`);
      }
      view.setUint32(offset, scalar, true);
      return;
    }
    case "f32":
      view.setFloat32(offset, number(value, path), true);
      return;
    case "vector": {
      const values = sequence(value, path);
      const count = required(type.count, `${path} vector count`);
      const element = required(type.elem, `${path} vector element`);
      if (values.length !== count) throw new RangeError(`${path} must contain ${count} components`);
      for (let index = 0; index < count; index++) {
        writeValue(view, offset + index * 4, element, values[index], `${path}[${index}]`);
      }
      return;
    }
    case "array": {
      const values = sequence(value, path);
      const count = required(type.count, `${path} array count`);
      const stride = required(type.stride, `${path} array stride`);
      const element = required(type.elem, `${path} array element`);
      if (values.length !== count) throw new RangeError(`${path} must contain ${count} elements`);
      for (let index = 0; index < count; index++) {
        writeValue(view, offset + index * stride, element, values[index], `${path}[${index}]`);
      }
      return;
    }
    case "runtime": {
      const values = sequence(value, path);
      const stride = required(type.stride, `${path} runtime stride`);
      const element = required(type.elem, `${path} runtime element`);
      for (let index = 0; index < values.length; index++) {
        writeValue(view, offset + index * stride, element, values[index], `${path}[${index}]`);
      }
      return;
    }
    case "struct": {
      if (value === null || typeof value !== "object") {
        throw new TypeError(`${path} must be an object`);
      }
      const record = value as Record<string, unknown>;
      for (const field of type.fields ?? []) {
        if (!(field.name in record)) throw new TypeError(`${path} is missing field ${field.name}`);
        writeValue(
          view,
          offset + field.offset,
          field.type,
          record[field.name],
          `${path}.${field.name}`,
        );
      }
      return;
    }
  }
}

function readValue(
  view: DataView,
  offset: number,
  type: HostLayout,
  available: number,
): unknown {
  switch (type.kind) {
    case "i32":
      return view.getInt32(offset, true);
    case "u32":
      return view.getUint32(offset, true);
    case "f32":
      return view.getFloat32(offset, true);
    case "vector": {
      const count = required(type.count, "vector count");
      const element = required(type.elem, "vector element");
      return Array.from({ length: count }, (_, index) =>
        readValue(view, offset + index * 4, element, 4));
    }
    case "array": {
      const count = required(type.count, "array count");
      const stride = required(type.stride, "array stride");
      const element = required(type.elem, "array element");
      return Array.from({ length: count }, (_, index) =>
        readValue(view, offset + index * stride, element, stride));
    }
    case "runtime": {
      const stride = required(type.stride, "runtime stride");
      const element = required(type.elem, "runtime element");
      const count = available / stride;
      return Array.from({ length: count }, (_, index) =>
        readValue(view, offset + index * stride, element, stride));
    }
    case "struct": {
      const value: Record<string, unknown> = {};
      for (const field of type.fields ?? []) {
        const fieldBytes = field.type.runtime
          ? available - field.offset
          : required(field.type.size, `${field.name} size`);
        value[field.name] = readValue(view, offset + field.offset, field.type, fieldBytes);
      }
      return value;
    }
  }
}

function logicalByteLength(type: HostLayout, value: unknown, path: string): number {
  if (!type.runtime) return required(type.size, `${path} size`);
  if (type.kind === "runtime") {
    return sequence(value, path).length * required(type.stride, `${path} stride`);
  }
  const fields = type.fields ?? [];
  const tail = fields.at(-1);
  if (!tail || value === null || typeof value !== "object") {
    throw new TypeError(`${path} must contain a runtime array tail`);
  }
  const record = value as Record<string, unknown>;
  return required(type.size, `${path} prefix size`) +
    sequence(record[tail.name], `${path}.${tail.name}`).length *
      required(tail.type.stride, `${path}.${tail.name} stride`);
}

function bytesOf(source: ArrayBuffer | ArrayBufferView): Uint8Array {
  if (ArrayBuffer.isView(source)) {
    return new Uint8Array(source.buffer, source.byteOffset, source.byteLength);
  }
  return new Uint8Array(source);
}

const littleEndian = new Uint8Array(new Uint32Array([1]).buffer)[0] === 1;
const scalarArrayConstructors = {
  f32: Float32Array,
  i32: Int32Array,
  u32: Uint32Array,
} as const;

function scalarArrayConstructor(type: HostLayout): typeof Float32Array | typeof Int32Array | typeof Uint32Array | undefined {
  if (type.kind !== "runtime") return undefined;
  const kind = type.elem?.kind;
  return kind === "f32" || kind === "i32" || kind === "u32"
    ? scalarArrayConstructors[kind]
    : undefined;
}

function packedScalarArray(type: HostLayout, value: unknown, path: string): Uint8Array | undefined {
  const constructor = scalarArrayConstructor(type);
  if (!constructor || !ArrayBuffer.isView(value)) return undefined;
  if (!(value instanceof constructor)) {
    throw new TypeError(`${path} must use ${constructor.name} when passed as a typed array`);
  }
  return littleEndian ? bytesOf(value) : undefined;
}

function unpackedScalarArray(
  type: HostLayout,
  source: ArrayBuffer | ArrayBufferView,
  representation: unknown,
): Float32Array | Int32Array | Uint32Array | undefined {
  const constructor = scalarArrayConstructor(type);
  if (!littleEndian || !constructor || !(representation instanceof constructor)) return undefined;
  return new constructor(bytesOf(source).slice().buffer);
}

function byteLength(resource: ResourceDefinition, value: unknown): number {
  return resource.runtime
    ? logicalByteLength(resource.layout, value, resource.name)
    : required(resource.byteSize, `${resource.name} byte size`);
}

function pack(resource: ResourceDefinition, value: unknown): Uint8Array {
  const size = byteLength(resource, value);
  if (!Number.isSafeInteger(size)) {
    throw new RangeError(`${resource.name} byte size is outside JavaScript's safe integer range`);
  }
  if (size < resource.minimumByteSize) {
    throw new RangeError(`${resource.name} requires at least ${resource.minimumByteSize} bytes`);
  }
  const direct = packedScalarArray(resource.layout, value, resource.name);
  if (direct) return direct;
  const bytes = new Uint8Array(size);
  writeValue(new DataView(bytes.buffer), 0, resource.layout, value, resource.name);
  return bytes;
}

function unpack(
  resource: ResourceDefinition,
  source: ArrayBuffer | ArrayBufferView,
  representation: unknown,
): unknown {
  const bytes = bytesOf(source);
  if (bytes.byteLength < resource.minimumByteSize) {
    throw new RangeError(`${resource.name} requires at least ${resource.minimumByteSize} bytes`);
  }
  if (!resource.runtime && bytes.byteLength < required(resource.byteSize, "resource byte size")) {
    throw new RangeError(`${resource.name} requires ${resource.byteSize} bytes`);
  }
  if (resource.runtime) {
    const base = resource.layout.kind === "runtime"
      ? 0
      : required(resource.layout.size, `${resource.name} prefix size`);
    const fields = resource.layout.fields ?? [];
    const stride = resource.layout.kind === "runtime"
      ? required(resource.layout.stride, `${resource.name} stride`)
      : required(fields.at(-1)?.type.stride, `${resource.name} tail stride`);
    if ((bytes.byteLength - base) % stride !== 0) {
      throw new RangeError(`${resource.name} has a partial runtime element`);
    }
  }
  const direct = unpackedScalarArray(resource.layout, bytes, representation);
  if (direct) return direct;
  return readValue(
    new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength),
    0,
    resource.layout,
    bytes.byteLength,
  );
}

function layoutKey(resource: ResourceDefinition): string {
  return JSON.stringify([
    resource.byteSize ?? 0,
    resource.minimumByteSize,
    resource.runtime,
    resource.layout,
  ]);
}

function materialize<T>(state: BufferState<T>, resource: ResourceDefinition): GPUBuffer {
  try {
    state.owner.assertHealthy(resource.name);
    if (state.destroyed) {
      throw new TachFailure(tachError(
        "lifecycle",
        "compute buffer has been destroyed",
        { operation: resource.name },
      ));
    }
    const key = layoutKey(resource);
    if (state.gpu) {
      if (state.codec?.key !== key) {
        throw new TypeError(`${resource.name} has a different layout from this compute buffer`);
      }
      return state.gpu;
    }

    const codec: BufferCodec<T> = {
      key,
      pack: (value) => pack(resource, value),
      unpack: (source) => unpack(resource, source, state.value) as T,
    };
    const bytes = codec.pack(state.value);
    const gpu = upload(
      state.owner.device,
      `Tach ${resource.name}`,
      bufferUsage.storage | bufferUsage.copyDst | bufferUsage.copySrc,
      bytes,
    );
    state.byteLength = bytes.byteLength;
    state.codec = codec;
    state.gpu = gpu;
    return gpu;
  } catch (cause) {
    throw new TachFailure(normalizeError(cause, "buffer", resource.name));
  }
}

function runtimeLength(resource: ResourceDefinition, state: BufferState<unknown>): number | undefined {
  if (!resource.runtime) return undefined;
  if (resource.layout.kind === "runtime") {
    return state.byteLength / required(resource.layout.stride, `${resource.name} stride`);
  }
  const tail = resource.layout.fields?.at(-1)?.type;
  return (state.byteLength - required(resource.layout.size, `${resource.name} prefix size`)) /
    required(tail?.stride, `${resource.name} tail stride`);
}

function workgroups(
  invocations: DispatchSize,
  workgroupSize: readonly [number, number, number],
): readonly [number, number, number] {
  const size = typeof invocations === "number" ? [invocations, 1, 1] : invocations;
  if (!Array.isArray(size) || size.length < 1 || size.length > 3) {
    throw new TypeError("size must be a positive integer or [x, y?, z?]");
  }
  const dimensions = [size[0] ?? Number.NaN, size[1] ?? 1, size[2] ?? 1] as const;
  for (const value of dimensions) {
    if (!Number.isSafeInteger(value) || value <= 0) {
      throw new RangeError("size dimensions must be positive integers");
    }
  }
  return [
    Math.ceil(dimensions[0] / workgroupSize[0]),
    Math.ceil(dimensions[1] / workgroupSize[1]),
    Math.ceil(dimensions[2] / workgroupSize[2]),
  ];
}

function dispatchOptions(value: DispatchOptions | undefined): {
  readonly size: DispatchSize | undefined;
  readonly dispatches: number;
} {
  if (value === undefined) return { size: undefined, dispatches: 1 };
  if (value === null || typeof value !== "object" || Array.isArray(value)) {
    throw new TypeError("dispatch options must be an object");
  }
  const dispatches = value.dispatches ?? 1;
  if (!Number.isSafeInteger(dispatches) || dispatches <= 0) {
    throw new RangeError("dispatches must be a positive integer");
  }
  return { size: value.size, dispatches };
}

async function compileKernel(
  device: GPUDevice,
  shaderModule: GPUShaderModule,
  resources: readonly ResourceDefinition[],
  info: KernelDefinition,
): Promise<CompiledKernel> {
  const grouped = new Map<number, GPUBindGroupLayoutEntry[]>();
  let maxGroup = -1;
  for (const parameter of info.resources) {
    const resource = required(resources[parameter.resource], parameter.name);
    maxGroup = Math.max(maxGroup, resource.group);
    let entries = grouped.get(resource.group);
    if (!entries) grouped.set(resource.group, entries = []);
    entries.push({
      binding: resource.binding,
      visibility: shaderStage.compute,
      buffer: {
        type: resource.kind === "uniform"
          ? "uniform"
          : resource.access === "read_write" ? "storage" : "read-only-storage",
        minBindingSize: resource.minimumByteSize,
      },
    });
  }

  const bindGroupLayouts: GPUBindGroupLayout[] = [];
  for (let group = 0; group <= maxGroup; group++) {
    const entries = grouped.get(group) ?? [];
    entries.sort((left, right) => left.binding - right.binding);
    bindGroupLayouts.push(device.createBindGroupLayout({
      label: `Tach ${info.name} group ${group}`,
      entries,
    }));
  }
  const layout = device.createPipelineLayout({
    label: `Tach ${info.name} layout`,
    bindGroupLayouts,
  });
  const pipeline = await device.createComputePipelineAsync({
    label: `Tach ${info.name}`,
    layout,
    compute: { module: shaderModule, entryPoint: info.entryPoint },
  });
  return { bindGroupLayouts, pipeline };
}

export function defineModule(definition: ModuleDefinition): DefinedModule {
  const deviceCache = new WeakMap<GPUDevice, DeviceCache>();

  async function compiled(
    owner: RuntimeOwner,
    kernelIndex: number,
    info: KernelDefinition,
  ): Promise<CompiledKernel> {
    let state = deviceCache.get(owner.device);
    if (!state) {
      state = { pipelines: new Map() };
      deviceCache.set(owner.device, state);
    }
    let pipeline = state.pipelines.get(kernelIndex);
    if (!pipeline) {
      pipeline = owner.capture(info.name, "kernel", () => {
        state.module ??= owner.device.createShaderModule({
          label: "Tach shader module",
          code: definition.source,
        });
        const pending = compileKernel(owner.device, state.module, definition.resources, info);
        return { finish: () => pending };
      });
      state.pipelines.set(kernelIndex, pipeline);
    }
    try {
      return await pipeline;
    } catch (cause) {
      state.pipelines.delete(kernelIndex);
      throw new TachFailure(normalizeError(cause, "kernel", info.name));
    }
  }

  function dispatch(
    kernelIndex: number,
    values: readonly unknown[],
    options?: DispatchOptions,
  ): ComputeDispatch {
    const info = definition.kernels[kernelIndex];
    if (!info) {
      throw new TachFailure(tachError("kernel", `unknown Tach kernel ${kernelIndex}`, {
        operation: "kernel",
      }));
    }
    if (values.length !== info.resources.length) {
      throw new TachFailure(tachError(
        "kernel",
        `${info.name} received the wrong number of parameters`,
        { operation: info.name },
      ));
    }

    let owner: RuntimeOwner | undefined;
    const storage = new Map<number, BufferState<unknown>>();
    for (let index = 0; index < info.resources.length; index++) {
      const parameter = required(info.resources[index], `${info.name} parameter ${index}`);
      const resource = required(definition.resources[parameter.resource], parameter.name);
      if (resource.kind !== "storage") continue;
      const state = getBufferState(
        values[index] as ComputeBuffer<unknown>,
        `${info.name}.${parameter.name}`,
      );
      if (owner && state.owner !== owner) {
        throw new TachFailure(tachError(
          "buffer",
          `${info.name} compute buffers belong to different Tach sessions`,
          { operation: info.name },
        ));
      }
      owner = state.owner;
      storage.set(index, state);
    }
    if (!owner) {
      throw new TachFailure(tachError(
        "kernel",
        `${info.name} has no compute buffer`,
        { operation: info.name },
      ));
    }
    owner.assertHealthy(info.name);
    let configured: ReturnType<typeof dispatchOptions>;
    const uniforms: Uint8Array[] = [];
    try {
      configured = dispatchOptions(options);
      for (let index = 0; index < info.resources.length; index++) {
        const parameter = required(info.resources[index], `${info.name} parameter ${index}`);
        const resource = required(definition.resources[parameter.resource], parameter.name);
        if (resource.kind === "uniform") uniforms.push(pack(resource, values[index]));
      }
    } catch (cause) {
      throw new TachFailure(normalizeError(cause, "kernel", info.name));
    }

    return createComputeDispatch({
      owner,
      async prepare(): Promise<PreparedDispatch> {
        const kernel = await compiled(owner, kernelIndex, info);
        return {
          uniforms,
          encode(pass, uniformBindings) {
            owner.assertHealthy(info.name);
            const entriesByGroup = new Map<number, BufferBindGroupEntry[]>();
            let inferredSize: number | undefined;
            let uniformIndex = 0;
            for (let index = 0; index < info.resources.length; index++) {
              const parameter = required(
                info.resources[index],
                `${info.name} parameter ${index}`,
              );
              const resource = required(definition.resources[parameter.resource], parameter.name);
              let binding: GPUBufferBinding;
              if (resource.kind === "storage") {
                const state = required(storage.get(index), parameter.name);
                binding = { buffer: materialize(state, resource) };
                inferredSize ??= runtimeLength(resource, state);
              } else {
                binding = required(uniformBindings[uniformIndex++], parameter.name);
              }
              let entries = entriesByGroup.get(resource.group);
              if (!entries) entriesByGroup.set(resource.group, entries = []);
              entries.push({ binding: resource.binding, resource: binding });
            }

            pass.setPipeline(kernel.pipeline);
            for (let group = 0; group < kernel.bindGroupLayouts.length; group++) {
              const entries = entriesByGroup.get(group) ?? [];
              entries.sort((left, right) => left.binding - right.binding);
              pass.setBindGroup(group, owner.bindGroup(
                `Tach ${info.name} group ${group}`,
                required(kernel.bindGroupLayouts[group], `${info.name} group ${group}`),
                entries,
              ));
            }
            const groups = workgroups(
              configured.size ?? inferredSize ?? info.workgroupSize,
              info.workgroupSize,
            );
            for (let index = 0; index < configured.dispatches; index++) {
              pass.dispatchWorkgroups(groups[0], groups[1], groups[2]);
            }
          },
        };
      },
    });
  }

  return Object.freeze({ dispatch });
}

function align4(value: number): number {
  return Math.ceil(value / 4) * 4;
}

function upload(
  device: GPUDevice,
  label: string,
  usage: GPUBufferUsageFlags,
  bytes: Uint8Array,
): GPUBuffer {
  const gpu = device.createBuffer({
    label,
    size: Math.max(4, align4(bytes.byteLength)),
    usage,
    mappedAtCreation: true,
  });
  try {
    new Uint8Array(gpu.getMappedRange()).set(bytes);
    gpu.unmap();
    return gpu;
  } catch (cause) {
    gpu.destroy();
    throw cause;
  }
}
