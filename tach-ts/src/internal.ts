import type {
  HostLayout,
  ModuleDefinition,
  ParameterBlockDefinition,
  PreparedBarrierResource,
  PreparedCommand,
  PreparedResource,
  PreparedStep,
  PublicProgramDefinition,
  ResourceDefinition,
  ResourceSource,
  ShapeExpression,
  ValueSource,
} from "./driver.ts";
import type {
  CommandOptions,
  ComputeBuffer,
  ComputeCommand,
  LaunchOptions,
  LaunchSize,
} from "./api.ts";
import { normalizeError, TachError } from "./api.ts";
import {
  type BufferCodec,
  type BufferState,
  createComputeCommand,
  getBufferState,
  type RuntimeOwner,
} from "./runtime.ts";

export type { ModuleDefinition } from "./driver.ts";
export interface DefinedModule {
  command(
    program: number,
    values: readonly unknown[],
    options?: LaunchOptions | CommandOptions,
  ): ComputeCommand;
}

function required<T>(value: T | undefined, description: string): T {
  if (value === undefined) {
    throw new TypeError(`invalid generated layout: missing ${description}`);
  }
  return value;
}

function sequence(value: unknown, path: string): ArrayLike<unknown> {
  const length = value === null || value === undefined
    ? undefined
    : (value as Partial<ArrayLike<unknown>>).length;
  if (
    typeof value === "string" || !Number.isSafeInteger(length) ||
    (length ?? -1) < 0
  ) throw new TypeError(`${path} must be an array or typed array`);
  return value as ArrayLike<unknown>;
}

function number(value: unknown, path: string): number {
  if (typeof value !== "number") {
    throw new TypeError(`${path} must be a number`);
  }
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
    case "bool":
      if (typeof value !== "boolean") {
        throw new TypeError(`${path} must be a boolean`);
      }
      view.setUint32(offset, value ? 1 : 0, true);
      return;
    case "i32": {
      const scalar = number(value, path);
      if (
        !Number.isInteger(scalar) || scalar < -0x8000_0000 ||
        scalar > 0x7fff_ffff
      ) throw new RangeError(`${path} must be a signed 32-bit integer`);
      view.setInt32(offset, scalar, true);
      return;
    }
    case "u32": {
      const scalar = number(value, path);
      if (!Number.isInteger(scalar) || scalar < 0 || scalar > 0xffff_ffff) {
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
      if (values.length !== count) {
        throw new RangeError(`${path} must contain ${count} components`);
      }
      for (let index = 0; index < count; index++) {
        writeValue(
          view,
          offset + index * 4,
          element,
          values[index],
          `${path}[${index}]`,
        );
      }
      return;
    }
    case "array": {
      const values = sequence(value, path);
      const count = required(type.count, `${path} array count`);
      const stride = required(type.stride, `${path} array stride`);
      const element = required(type.elem, `${path} array element`);
      if (values.length !== count) {
        throw new RangeError(`${path} must contain ${count} elements`);
      }
      for (let index = 0; index < count; index++) {
        writeValue(
          view,
          offset + index * stride,
          element,
          values[index],
          `${path}[${index}]`,
        );
      }
      return;
    }
    case "runtime": {
      const values = sequence(value, path);
      const stride = required(type.stride, `${path} runtime stride`);
      const element = required(type.elem, `${path} runtime element`);
      for (let index = 0; index < values.length; index++) {
        writeValue(
          view,
          offset + index * stride,
          element,
          values[index],
          `${path}[${index}]`,
        );
      }
      return;
    }
    case "struct": {
      if (value === null || typeof value !== "object") {
        throw new TypeError(`${path} must be an object`);
      }
      const record = value as Record<string, unknown>;
      for (const field of type.fields ?? []) {
        if (!(field.name in record)) {
          throw new TypeError(`${path} is missing field ${field.name}`);
        }
        writeValue(
          view,
          offset + field.offset,
          field.type,
          record[field.name],
          `${path}.${field.name}`,
        );
      }
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
    case "bool":
      return view.getUint32(offset, true) !== 0;
    case "i32":
      return view.getInt32(offset, true);
    case "u32":
      return view.getUint32(offset, true);
    case "f32":
      return view.getFloat32(offset, true);
    case "vector": {
      const count = required(type.count, "vector count"),
        element = required(type.elem, "vector element");
      return Array.from(
        { length: count },
        (_, index) => readValue(view, offset + index * 4, element, 4),
      );
    }
    case "array": {
      const count = required(type.count, "array count"),
        stride = required(type.stride, "array stride"),
        element = required(type.elem, "array element");
      return Array.from(
        { length: count },
        (_, index) => readValue(view, offset + index * stride, element, stride),
      );
    }
    case "runtime": {
      const stride = required(type.stride, "runtime stride"),
        element = required(type.elem, "runtime element");
      return Array.from(
        { length: available / stride },
        (_, index) => readValue(view, offset + index * stride, element, stride),
      );
    }
    case "struct": {
      const value: Record<string, unknown> = {};
      for (const field of type.fields ?? []) {
        value[field.name] = readValue(
          view,
          offset + field.offset,
          field.type,
          field.type.runtime
            ? available - field.offset
            : required(field.type.size, `${field.name} size`),
        );
      }
      return value;
    }
  }
}

function bytesOf(source: ArrayBuffer | ArrayBufferView): Uint8Array {
  return ArrayBuffer.isView(source)
    ? new Uint8Array(source.buffer, source.byteOffset, source.byteLength)
    : new Uint8Array(source);
}

const littleEndian = new Uint8Array(new Uint32Array([1]).buffer)[0] === 1;
const scalarArrays = {
  f32: Float32Array,
  i32: Int32Array,
  u32: Uint32Array,
} as const;

function typedArrayConstructor(
  type: HostLayout,
): typeof Float32Array | typeof Int32Array | typeof Uint32Array | undefined {
  if (type.kind !== "runtime") return undefined;
  let element = type.elem;
  if (element?.kind === "vector") {
    if (element.count === undefined || type.stride !== element.count * 4) {
      return undefined;
    }
    element = element.elem;
  }
  return element?.kind === "f32" || element?.kind === "i32" ||
      element?.kind === "u32"
    ? scalarArrays[element.kind]
    : undefined;
}

function logicalByteLength(
  type: HostLayout,
  value: unknown,
  path: string,
): number {
  if (!type.runtime) return required(type.size, `${path} size`);
  if (type.kind === "runtime") {
    const values = sequence(value, path),
      width = type.elem?.kind === "vector"
        ? required(type.elem.count, "vector count")
        : 1;
    if (ArrayBuffer.isView(value) && values.length % width !== 0) {
      throw new RangeError(
        `${path} must contain complete ${width}-component elements`,
      );
    }
    return (ArrayBuffer.isView(value) ? values.length / width : values.length) *
      required(type.stride, `${path} stride`);
  }
  const tail = type.fields?.at(-1);
  if (!tail || value === null || typeof value !== "object") {
    throw new TypeError(`${path} must contain a runtime array tail`);
  }
  return required(type.size, `${path} prefix size`) +
    sequence(
        (value as Record<string, unknown>)[tail.name],
        `${path}.${tail.name}`,
      ).length * required(tail.type.stride, `${path}.${tail.name} stride`);
}

function pack(resource: ResourceDefinition, value: unknown): Uint8Array {
  const size = resource.runtime
    ? logicalByteLength(resource.layout, value, resource.name)
    : required(resource.byteSize, `${resource.name} byte size`);
  if (!Number.isSafeInteger(size) || size < resource.minimumByteSize) {
    throw new RangeError(
      `${resource.name} requires at least ${resource.minimumByteSize} bytes`,
    );
  }
  const constructor = typedArrayConstructor(resource.layout);
  if (littleEndian && constructor && ArrayBuffer.isView(value)) {
    if (!(value instanceof constructor)) {
      throw new TypeError(
        `${resource.name} must use ${constructor.name} when passed as a typed array`,
      );
    }
    return bytesOf(value).slice();
  }
  const bytes = new Uint8Array(size);
  writeValue(
    new DataView(bytes.buffer),
    0,
    resource.layout,
    value,
    resource.name,
  );
  return bytes;
}

function unpack(
  resource: ResourceDefinition,
  source: ArrayBuffer | ArrayBufferView,
  representation: unknown,
): unknown {
  const bytes = bytesOf(source);
  if (bytes.byteLength < resource.minimumByteSize) {
    throw new RangeError(
      `${resource.name} requires at least ${resource.minimumByteSize} bytes`,
    );
  }
  const constructor = typedArrayConstructor(resource.layout);
  if (littleEndian && constructor && representation instanceof constructor) {
    return new constructor(bytes.slice().buffer);
  }
  return readValue(
    new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength),
    0,
    resource.layout,
    bytes.byteLength,
  );
}

function materialize<T>(state: BufferState<T>, resource: ResourceDefinition) {
  try {
    state.owner.assertHealthy(resource.name);
    if (state.destroyed) {
      throw new TachError("lifecycle", "compute buffer has been destroyed", {
        operation: resource.name,
      });
    }
    const key = JSON.stringify([
      resource.byteSize ?? 0,
      resource.minimumByteSize,
      resource.runtime,
      resource.layout,
    ]);
    if (state.driverBuffer !== undefined) {
      if (state.codec?.key !== key) {
        throw new TypeError(
          `${resource.name} has a different layout from this compute buffer`,
        );
      }
      return state.driverBuffer;
    }
    const codec: BufferCodec<T> = {
      key,
      pack: (value) => pack(resource, value),
      unpack: (source) => unpack(resource, source, state.value) as T,
    };
    const bytes = codec.pack(state.value);
    state.byteLength = bytes.byteLength;
    state.codec = codec;
    state.driverBuffer = state.owner.driver.createBuffer(
      `Tach ${resource.name}`,
      bytes,
    );
    return state.driverBuffer;
  } catch (cause) {
    throw normalizeError(cause, "buffer", resource.name);
  }
}

function runtimeLength(
  resource: ResourceDefinition,
  state: BufferState<unknown>,
  path: readonly string[] = [],
): number | undefined {
  if (!resource.runtime) return undefined;
  let bytes = state.byteLength ||
      logicalByteLength(resource.layout, state.value, resource.name),
    layout = resource.layout;
  for (const name of path) {
    const field = layout.fields?.find((candidate) => candidate.name === name);
    if (!field) {
      throw new TypeError(`${resource.name} has no layout field ${name}`);
    }
    bytes -= field.offset;
    layout = field.type;
  }
  if (layout.kind === "runtime") {
    return bytes / required(layout.stride, `${resource.name} stride`);
  }
  return (bytes - required(layout.size, `${resource.name} prefix size`)) /
    required(
      layout.fields?.at(-1)?.type.stride,
      `${resource.name} tail stride`,
    );
}

function checkedU32(value: unknown, path: string): number {
  if (
    typeof value !== "number" || !Number.isInteger(value) || value < 0 ||
    value > 0xffff_ffff
  ) throw new RangeError(`${path} must be uint32`);
  return value;
}
function pathValue(
  value: unknown,
  path: readonly string[] | undefined,
  name: string,
): unknown {
  let current = value;
  for (const field of path ?? []) {
    if (
      current === null || typeof current !== "object" || !(field in current)
    ) throw new TypeError(`${name} is missing field ${field}`);
    current = (current as Record<string, unknown>)[field];
  }
  return current;
}
function shapeValue(
  expression: ShapeExpression,
  values: readonly unknown[],
  definitions: readonly ResourceDefinition[],
  resources: readonly BufferState<unknown>[],
  launch: readonly number[],
): number {
  switch (expression.op) {
    case "constant":
      return checkedU32(expression.value, "shape constant");
    case "parameter":
      return checkedU32(
        pathValue(
          values[required(expression.parameter, "shape parameter")],
          expression.path,
          "shape parameter",
        ),
        "shape parameter",
      );
    case "resourceLength": {
      const index = required(expression.resource, "shape resource");
      return checkedU32(
        runtimeLength(
          required(definitions[index], "shape resource"),
          required(resources[index], "shape resource"),
          expression.path,
        ),
        "resource length",
      );
    }
    case "launchAxis":
      return checkedU32(launch[expression.axis ?? 0], "launch axis");
  }
  const left = BigInt(
    shapeValue(
      required(expression.left, "shape left"),
      values,
      definitions,
      resources,
      launch,
    ),
  );
  const right = BigInt(
    shapeValue(
      required(expression.right, "shape right"),
      values,
      definitions,
      resources,
      launch,
    ),
  );
  let result: bigint;
  switch (expression.op) {
    case "add":
      result = left + right;
      break;
    case "sub":
      result = left - right;
      break;
    case "mul":
      result = left * right;
      break;
    case "div":
      if (right === 0n) throw new RangeError("shape division by zero");
      result = left / right;
      break;
    case "rem":
      if (right === 0n) throw new RangeError("shape remainder by zero");
      result = left % right;
      break;
    case "min":
      result = left < right ? left : right;
      break;
    case "max":
      result = left > right ? left : right;
      break;
    case "ceilDiv":
      if (right === 0n) {
        throw new RangeError("ceilDiv denominator must be positive");
      }
      result = (left + right - 1n) / right;
      break;
  }
  return checkedU32(Number(result), "shape result");
}

function valueOf(
  source: ValueSource,
  program: PublicProgramDefinition,
  values: readonly unknown[],
  resources: readonly BufferState<unknown>[],
  launch: readonly number[],
  repeat: number,
): unknown {
  switch (source.kind) {
    case "parameter":
      return pathValue(
        values[required(source.parameter, "value parameter")],
        source.path,
        "value parameter",
      );
    case "bool":
    case "i32":
    case "u32":
      return source.value;
    case "f32Bits":
      return new Float32Array(
        new Uint32Array([checkedU32(source.value, "f32 bits")]).buffer,
      )[0];
    case "shape":
      return shapeValue(
        required(source.expression, "shape expression"),
        values,
        program.resources,
        resources,
        launch,
      );
    case "repeat":
      return repeat;
  }
}

function packBlock(
  program: PublicProgramDefinition,
  block: ParameterBlockDefinition,
  sources: readonly ValueSource[],
  values: readonly unknown[],
  resources: readonly BufferState<unknown>[],
  launch: readonly number[],
  repeat: number,
): Uint8Array {
  if (sources.length !== block.fields.length) {
    throw new TypeError("parameter source count mismatch");
  }
  const bytes = new Uint8Array(block.byteSize),
    view = new DataView(bytes.buffer);
  for (let index = 0; index < block.fields.length; index++) {
    const field = required(block.fields[index], "parameter field"),
      source = required(sources[index], "parameter source");
    writeValue(
      view,
      field.byteOffset,
      field.layout,
      valueOf(source, program, values, resources, launch, repeat),
      `${program.name} parameter ${index}`,
    );
  }
  return bytes;
}

function launchOptions(
  value: LaunchOptions | undefined,
): { readonly size?: LaunchSize; readonly repeat: number } {
  if (value === undefined) return { repeat: 1 };
  if (value === null || typeof value !== "object" || Array.isArray(value)) {
    throw new TypeError("launch options must be an object");
  }
  const repeat = value.repeat ?? 1;
  if (!Number.isInteger(repeat) || repeat <= 0 || repeat > 0xffff_ffff) {
    throw new RangeError("repeat must be a positive integer within uint32");
  }
  return value.size === undefined ? { repeat } : { size: value.size, repeat };
}

function workgroups(
  invocations: readonly number[],
  workgroup: readonly [number, number, number],
): readonly [number, number, number] {
  if (invocations.length < 1 || invocations.length > 3) {
    throw new TypeError("dispatch domain must have 1..3 dimensions");
  }
  const size = [
    invocations[0],
    invocations[1] ?? 1,
    invocations[2] ?? 1,
  ] as const;
  for (const value of size) {
    if (!Number.isSafeInteger(value) || value! <= 0) {
      throw new RangeError("size dimensions must be positive integers");
    }
  }
  return [
    Math.ceil(size[0]! / workgroup[0]),
    Math.ceil(size[1]! / workgroup[1]),
    Math.ceil(size[2]! / workgroup[2]),
  ];
}

function barrierResource(
  source: ResourceSource,
  storage: readonly BufferState<unknown>[],
  transients: readonly { readonly color: number }[],
): PreparedBarrierResource {
  return source.kind === "external"
    ? {
      buffer: required(
        required(storage[source.resource], "barrier resource").driverBuffer,
        "materialized barrier resource",
      ),
    }
    : {
      scratch: required(transients[source.resource], "barrier transient").color,
    };
}

export function defineModule(definition: ModuleDefinition): DefinedModule {
  if (
    definition.schema !== 1 ||
    definition.targets.web.programs.length !== definition.programs.length ||
    definition.targets.spirv.programs.length !== definition.programs.length
  ) throw new TypeError("invalid generated Tach schema");
  function command(
    programIndex: number,
    values: readonly unknown[],
    options?: LaunchOptions | CommandOptions,
  ): ComputeCommand {
    const info = required(
      definition.programs[programIndex],
      `program ${programIndex}`,
    );
    if (values.length !== info.parameters.length) {
      throw new TachError(
        "kernel",
        `${info.name} received the wrong number of parameters`,
        { operation: info.name },
      );
    }
    let owner: RuntimeOwner | undefined;
    const storage: BufferState<unknown>[] = [],
      seen = new Map<BufferState<unknown>, string>();
    for (let index = 0; index < info.parameters.length; index++) {
      const parameter = required(
        info.parameters[index],
        `${info.name} parameter ${index}`,
      );
      if (parameter.kind !== "buffer") continue;
      const state = getBufferState(
        values[index] as ComputeBuffer<unknown>,
        `${info.name}.${parameter.name}`,
      );
      if (owner && state.owner !== owner) {
        throw new TachError(
          "buffer",
          `${info.name} compute buffers belong to different Tach sessions`,
          { operation: info.name },
        );
      }
      const previous = seen.get(state);
      if (previous) {
        throw new TachError(
          "buffer",
          `${info.name}.${parameter.name} aliases ${info.name}.${previous}; buffer parameters require distinct compute buffers`,
          { operation: info.name },
        );
      }
      owner = state.owner;
      seen.set(state, parameter.name);
      storage[required(parameter.resource, parameter.name)] = state;
    }
    if (!owner) {
      throw new TachError("kernel", `${info.name} has no compute buffer`, {
        operation: info.name,
      });
    }
    owner.assertHealthy(info.name);
    const host = owner.driver.adapter.backend === "webgpu" ? "web" : "spirv";
    const target = definition.targets[host],
      plan = required(
        target.programs[programIndex],
        `program plan ${programIndex}`,
      );
    let configured: ReturnType<typeof launchOptions>;
    try {
      configured = launchOptions(options as LaunchOptions | undefined);
    } catch (cause) {
      throw normalizeError(cause, "kernel", info.name);
    }
    let launch: number[] = [];
    if (info.launch) {
      if (configured.size !== undefined) {
        launch = typeof configured.size === "number"
          ? [configured.size]
          : [...configured.size];
        if (launch.length !== info.launch.dimensions) {
          throw new TachError(
            "kernel",
            `kernel requires an exact ${info.launch.dimensions}D launch size`,
            { operation: info.name },
          );
        }
      } else if (info.launch.inferFromResource === undefined) {
        const step = plan.steps.find((candidate) =>
            candidate.kind === "dispatch"
          ),
          kernel = required(
            target.kernels[required(step?.kernel, "launch kernel")],
            "launch kernel",
          );
        launch = kernel.workgroupSize.slice(0, info.launch.dimensions);
      }
    }
    for (const step of plan.steps) {
      if (step.kind !== "dispatch") continue;
      const block = target.kernels[required(step.kernel, "validation kernel")]
        ?.parameterBlock;
      if (!block) continue;
      const bytes = new Uint8Array(block.byteSize),
        view = new DataView(bytes.buffer),
        sources = step.parameters ?? [];
      block.fields.forEach((field, index) => {
        const source = required(sources[index], "validation parameter source");
        if (source.kind !== "shape") {
          const parameter = source.kind === "parameter"
            ? required(
              info.parameters[required(source.parameter, "value parameter")],
              "parameter",
            ).name + (source.path ?? []).map((name) => `.${name}`).join("")
            : `parameter ${index}`;
          writeValue(
            view,
            field.byteOffset,
            field.layout,
            valueOf(source, info, values, storage, launch, configured.repeat),
            `${info.name}.${parameter}`,
          );
        }
      });
    }
    return createComputeCommand({
      owner,
      prepare(): PreparedCommand {
        for (let index = 0; index < storage.length; index++) {
          if (storage[index] && info.resources[index]) {
            materialize(storage[index]!, info.resources[index]!);
          }
        }
        if (info.launch && launch.length === 0) {
          const index = required(
            info.launch.inferFromResource,
            "launch resource",
          );
          launch = [
            required(
              runtimeLength(
                required(info.resources[index], "launch resource"),
                required(storage[index], "launch buffer"),
              ),
              "launch size",
            ),
          ];
        }
        const transientBytes = plan.transients.map((transient) => {
          const bytes = shapeValue(
            transient.length,
            values,
            info.resources,
            storage,
            launch,
          ) * transient.stride;
          if (
            !Number.isSafeInteger(bytes) || bytes <= 0 || bytes > 0xffff_ffff
          ) throw new RangeError("transient byte size overflow");
          return bytes;
        });
        const scratch = new Map<number, number>();
        plan.transients.forEach((transient, index) =>
          scratch.set(
            transient.color,
            Math.max(
              scratch.get(transient.color) ?? 0,
              required(transientBytes[index], "transient bytes"),
            ),
          )
        );
        const steps: PreparedStep[] = plan.steps.map((step) => {
          if (step.kind === "barrier") {
            return {
              kind: "barrier",
              resources: step.resources.map((source) =>
                barrierResource(source, storage, plan.transients)
              ),
            };
          }
          const kernelIndex = required(step.kernel, "dispatch kernel"),
            kernel = required(target.kernels[kernelIndex], "kernel");
          const resources: PreparedResource[] = step.resources.map((source) =>
            source.kind === "external"
              ? {
                binding: required(source.binding, "resource binding"),
                buffer: required(
                  required(storage[source.resource], "external resource")
                    .driverBuffer,
                  "materialized external resource",
                ),
                byteSize:
                  required(storage[source.resource], "external resource")
                    .byteLength,
              }
              : {
                binding: required(source.binding, "resource binding"),
                scratch:
                  required(plan.transients[source.resource], "transient").color,
                byteSize: required(
                  transientBytes[source.resource],
                  "transient byte size",
                ),
              }
          );
          const parameters = kernel.parameterBlock
            ? packBlock(
              info,
              kernel.parameterBlock,
              step.parameters ?? [],
              values,
              storage,
              launch,
              configured.repeat,
            )
            : undefined;
          const domain = (step.domain ?? []).map((axis) =>
            shapeValue(axis, values, info.resources, storage, launch)
          );
          return parameters
            ? {
              kind: "dispatch",
              kernel: kernelIndex,
              groups: workgroups(domain, kernel.workgroupSize),
              resources,
              parameters,
            }
            : {
              kind: "dispatch",
              kernel: kernelIndex,
              groups: workgroups(domain, kernel.workgroupSize),
              resources,
            };
        });
        const repeatBarrier = configured.repeat > 1 && plan.repeatBarrier
          ? plan.repeatBarrier.resources.map((source) =>
            barrierResource(source, storage, plan.transients)
          )
          : undefined;
        return repeatBarrier
          ? {
            module: definition,
            shader: definition.shaders[host],
            target,
            steps,
            repeat: plan.repeat === "program" ? configured.repeat : 1,
            repeatBarrier,
            scratch,
          }
          : {
            module: definition,
            shader: definition.shaders[host],
            target,
            steps,
            repeat: plan.repeat === "program" ? configured.repeat : 1,
            scratch,
          };
      },
    });
  }
  return Object.freeze({ command });
}
