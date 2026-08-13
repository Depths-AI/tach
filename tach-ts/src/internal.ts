import {
  createComputeCommand,
  getBufferState,
  type BufferBindGroupEntry,
  type BufferCodec,
  type BufferState,
  type ComputeBuffer,
  type ComputeCommand,
	type CommandOptions,
  type LaunchOptions,
  type LaunchSize,
  type PreparedCommand,
  type RuntimeOwner,
} from "./runtime.js";
import { normalizeError, TachError } from "./error.js";

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
  readonly kind: "bool" | "i32" | "u32" | "f32" | "vector" | "array" | "runtime" | "struct";
  readonly size?: number;
  readonly stride?: number;
  readonly count?: number;
  readonly runtime?: boolean;
  readonly elem?: HostLayout;
  readonly fields?: readonly HostLayoutField[];
}

interface ResourceDefinition {
  readonly name: string;
	readonly type: string;
  readonly byteSize?: number;
  readonly minimumByteSize: number;
  readonly runtime: boolean;
  readonly layout: HostLayout;
}

interface PublicParameterDefinition {
  readonly name: string;
	readonly kind: "buffer" | "value";
	readonly type: string;
	readonly resource?: number;
}

interface ParameterFieldDefinition {
	readonly type: string;
	readonly byteOffset: number;
  readonly layout: HostLayout;
}

interface ParameterBlockDefinition {
  readonly group: number;
  readonly binding: number;
  readonly byteSize: number;
  readonly fields: readonly ParameterFieldDefinition[];
}

interface KernelDefinition {
  readonly entryPoint: string;
  readonly workgroupSize: readonly [number, number, number];
	readonly bindings: readonly { readonly group: 0; readonly binding: number; readonly access: "read" | "read_write"; readonly type: string; readonly minimumByteSize: number }[];
  readonly parameterBlock?: ParameterBlockDefinition;
}

interface ShapeExpression { readonly op: "constant"|"parameter"|"resourceLength"|"launchAxis"|"add"|"sub"|"mul"|"div"|"rem"|"min"|"max"|"ceilDiv"; readonly value?: number; readonly parameter?: number; readonly resource?: number; readonly path?: readonly string[]; readonly axis?: 0|1|2; readonly left?: ShapeExpression; readonly right?: ShapeExpression }
interface ValueSource { readonly kind: "parameter"|"bool"|"i32"|"u32"|"f32Bits"|"shape"|"repeat"; readonly parameter?: number; readonly path?: readonly string[]; readonly value?: number|boolean; readonly expression?: ShapeExpression }
interface StepDefinition { readonly kind: "dispatch"|"barrier"; readonly kernel?: number; readonly domain?: readonly ShapeExpression[]; readonly resources: readonly { readonly binding?: number; readonly kind: "external"|"transient"; readonly resource: number }[]; readonly parameters?: readonly ValueSource[] }
interface TransientDefinition { readonly type: string; readonly stride: number; readonly alignment: number; readonly minimumByteSize: number; readonly length: ShapeExpression; readonly color: number; readonly firstStep: number; readonly lastStep: number }
interface ProgramPlanDefinition { readonly program: number; readonly transients: readonly TransientDefinition[]; readonly steps: readonly StepDefinition[]; readonly repeat: "program"|"invocation-loop" }
interface PublicProgramDefinition { readonly name: string; readonly parameters: readonly PublicParameterDefinition[]; readonly resources: readonly ResourceDefinition[]; readonly launch?: { readonly dimensions: 1|2|3; readonly inferFromResource?: number } }

export interface ModuleDefinition {
	readonly shader: string;
	readonly schema: 1;
	readonly types: readonly unknown[];
	readonly programs: readonly PublicProgramDefinition[];
	readonly target: { readonly kernels: readonly KernelDefinition[]; readonly programs: readonly ProgramPlanDefinition[] };
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
  command(
		program: number,
    values: readonly unknown[],
		options?: LaunchOptions | CommandOptions,
  ): ComputeCommand;
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
    case "bool":
      if (typeof value !== "boolean") throw new TypeError(`${path} must be a boolean`);
      view.setUint32(offset, value ? 1 : 0, true);
      return;
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
    case "bool":
      return view.getUint32(offset, true) !== 0;
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
    const constructor = typedArrayConstructor(type);
    if (constructor && ArrayBuffer.isView(value)) {
      if (!(value instanceof constructor)) {
        throw new TypeError(`${path} must use ${constructor.name} when passed as a typed array`);
      }
      const width = runtimeElementWidth(type);
      if (value.length % width !== 0) {
        throw new RangeError(`${path} must contain complete ${width}-component elements`);
      }
      return value.length / width * required(type.stride, `${path} stride`);
    }
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

function typedArrayConstructor(type: HostLayout): typeof Float32Array | typeof Int32Array | typeof Uint32Array | undefined {
  if (type.kind !== "runtime") return undefined;
  let element = type.elem;
  if (element?.kind === "vector") {
    const count = element.count;
    if (count === undefined || type.stride !== count * 4) return undefined;
    element = element.elem;
  }
  const kind = element?.kind;
  return kind === "f32" || kind === "i32" || kind === "u32"
    ? scalarArrayConstructors[kind]
    : undefined;
}

function runtimeElementWidth(type: HostLayout): number {
  return type.elem?.kind === "vector" ? required(type.elem.count, "vector count") : 1;
}

function packedTypedArray(type: HostLayout, value: unknown, path: string): Uint8Array | undefined {
  const constructor = typedArrayConstructor(type);
  if (!constructor || !ArrayBuffer.isView(value)) return undefined;
  if (!(value instanceof constructor)) {
    throw new TypeError(`${path} must use ${constructor.name} when passed as a typed array`);
  }
  return littleEndian ? bytesOf(value) : undefined;
}

function unpackedTypedArray(
  type: HostLayout,
  source: ArrayBuffer | ArrayBufferView,
  representation: unknown,
): Float32Array | Int32Array | Uint32Array | undefined {
  const constructor = typedArrayConstructor(type);
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
  const direct = packedTypedArray(resource.layout, value, resource.name);
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
  const direct = unpackedTypedArray(resource.layout, bytes, representation);
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
      throw new TachError(
        "lifecycle",
        "compute buffer has been destroyed",
        { operation: resource.name },
      );
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
    throw normalizeError(cause, "buffer", resource.name);
  }
}

function runtimeLength(
  resource: ResourceDefinition,
  state: BufferState<unknown>,
  path: readonly string[] = [],
): number | undefined {
  if (!resource.runtime) return undefined;
	let bytes = state.byteLength || logicalByteLength(resource.layout, state.value, resource.name);
  let layout = resource.layout;
  for (const name of path) {
    const field = layout.fields?.find((candidate) => candidate.name === name);
    if (!field) throw new TypeError(`${resource.name} has no layout field ${name}`);
    bytes -= field.offset;
    layout = field.type;
  }
  if (layout.kind === "runtime") {
		return bytes / required(layout.stride, `${resource.name} stride`);
  }
  const tail = layout.fields?.at(-1)?.type;
	return (bytes - required(layout.size, `${resource.name} prefix size`)) /
    required(tail?.stride, `${resource.name} tail stride`);
}

function workgroups(
  invocations: LaunchSize,
  workgroupSize: readonly [number, number, number],
  rank: 1 | 2 | 3,
): readonly [number, number, number] {
  if (typeof invocations === "number" ? rank !== 1 : !Array.isArray(invocations) || invocations.length !== rank) {
    throw new TypeError(`kernel requires an exact ${rank}D launch size`);
  }
  const size = typeof invocations === "number" ? [invocations, 1, 1] : invocations;
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

function launchOptions(value: LaunchOptions | undefined): {
  readonly size: LaunchSize | undefined;
  readonly repeat: number;
} {
  if (value === undefined) return { size: undefined, repeat: 1 };
  if (value === null || typeof value !== "object" || Array.isArray(value)) {
    throw new TypeError("launch options must be an object");
  }
  const repeat = value.repeat ?? 1;
  if (!Number.isInteger(repeat) || repeat <= 0 || repeat > 0xffff_ffff) {
		throw new RangeError("repeat must be a positive integer within uint32");
  }
  return { size: value.size, repeat };
}

async function compileKernel(device: GPUDevice, shaderModule: GPUShaderModule, info: KernelDefinition): Promise<CompiledKernel> {
  const entries: GPUBindGroupLayoutEntry[] = info.bindings.map((binding) => ({ binding: binding.binding, visibility: shaderStage.compute, buffer: { type: binding.access === "read_write" ? "storage" : "read-only-storage", minBindingSize: binding.minimumByteSize } }));
  if (info.parameterBlock) entries.push({ binding: info.parameterBlock.binding, visibility: shaderStage.compute, buffer: { type: "uniform", minBindingSize: info.parameterBlock.byteSize, hasDynamicOffset: true } });
  entries.sort((left, right) => left.binding - right.binding);
  const bindGroupLayouts = [device.createBindGroupLayout({ label: `Tach ${info.entryPoint} group 0`, entries })];
  const layout = device.createPipelineLayout({ label: `Tach ${info.entryPoint} layout`, bindGroupLayouts });
  const pipeline = await device.createComputePipelineAsync({ label: `Tach ${info.entryPoint}`, layout, compute: { module: shaderModule, entryPoint: info.entryPoint } });
  return { bindGroupLayouts, pipeline };
}

function checkedU32(value: unknown, path: string): number { if (typeof value !== "number" || !Number.isInteger(value) || value < 0 || value > 0xffff_ffff) throw new RangeError(`${path} must be uint32`); return value; }
function pathValue(value: unknown, path: readonly string[] | undefined, name: string): unknown { let current=value; for(const field of path??[]){if(current===null||typeof current!=="object"||!(field in current))throw new TypeError(`${name} is missing field ${field}`);current=(current as Record<string,unknown>)[field]};return current }
function shapeValue(expression: ShapeExpression, values: readonly unknown[], resourceDefinitions: readonly ResourceDefinition[], resources: readonly BufferState<unknown>[], launch: readonly number[]): number {
  switch(expression.op){
    case "constant": return checkedU32(expression.value, "shape constant");
    case "parameter": return checkedU32(pathValue(values[required(expression.parameter,"shape parameter")],expression.path,"shape parameter"),"shape parameter");
    case "resourceLength": { const index=required(expression.resource,"shape resource"); const state=required(resources[index],"shape resource"); return checkedU32(runtimeLength(required(resourceDefinitions[index],"shape resource"),state,expression.path),"resource length"); }
    case "launchAxis": return checkedU32(launch[expression.axis??0],"launch axis");
  }
  const left=shapeValue(required(expression.left,"shape left"),values,resourceDefinitions,resources,launch), right=shapeValue(required(expression.right,"shape right"),values,resourceDefinitions,resources,launch);
  if(expression.op==="min")return Math.min(left,right);if(expression.op==="max")return Math.max(left,right);
  const a=BigInt(left),b=BigInt(right);let result:bigint;switch(expression.op){case"add":result=a+b;break;case"sub":result=a-b;break;case"mul":result=a*b;break;case"div":if(b===0n)throw new RangeError("shape division by zero");result=a/b;break;case"rem":if(b===0n)throw new RangeError("shape remainder by zero");result=a%b;break;case"ceilDiv":if(b===0n)throw new RangeError("ceilDiv denominator must be positive");result=(a+b-1n)/b;break;default:throw new TypeError("invalid shape operation")};return checkedU32(Number(result),"shape result");
}
function valueOf(source: ValueSource, program:PublicProgramDefinition, values: readonly unknown[], resources: readonly BufferState<unknown>[], launch: readonly number[], repeat:number): unknown { switch(source.kind){case"parameter":return pathValue(values[required(source.parameter,"value parameter")],source.path,"value parameter");case"bool":return source.value;case"i32":case"u32":return source.value;case"f32Bits":{const bits=checkedU32(source.value,"f32 bits");return new Float32Array(new Uint32Array([bits]).buffer)[0]};case"shape":return shapeValue(required(source.expression,"shape expression"),values,program.resources,resources,launch);case"repeat":return repeat} }
function packBlock(program:PublicProgramDefinition,block: ParameterBlockDefinition, sources: readonly ValueSource[], values: readonly unknown[], resources: readonly BufferState<unknown>[], launch: readonly number[], repeat:number): Uint8Array { if(sources.length!==block.fields.length)throw new TypeError("parameter source count mismatch");const bytes=new Uint8Array(block.byteSize),view=new DataView(bytes.buffer);for(let i=0;i<block.fields.length;i++){const field=required(block.fields[i],"parameter field"),source=required(sources[i],"parameter source");let path=`${program.name} parameter ${i}`;if(source.kind==="parameter"){path=`${program.name}.${required(program.parameters[required(source.parameter,"parameter")],"parameter").name}`;for(const name of source.path??[])path+=`.${name}`}writeValue(view,field.byteOffset,field.layout,valueOf(source,program,values,resources,launch,repeat),path)}return bytes }

export function defineModule(definition: ModuleDefinition): DefinedModule {
  if(definition.schema!==1||definition.target.programs.length!==definition.programs.length)throw new TypeError("invalid generated Tach schema");
  const deviceCache=new WeakMap<GPUDevice,DeviceCache>();
  async function compiled(owner:RuntimeOwner,index:number):Promise<CompiledKernel>{const info=required(definition.target.kernels[index],`kernel ${index}`);let state=deviceCache.get(owner.device);if(!state){state={pipelines:new Map()};deviceCache.set(owner.device,state)}let pending=state.pipelines.get(index);if(!pending){pending=owner.capture(info.entryPoint,"kernel",()=>{state!.module??=owner.device.createShaderModule({label:"Tach shader module",code:definition.shader});return{finish:()=>compileKernel(owner.device,state!.module!,info)}});state.pipelines.set(index,pending)}try{return await pending}catch(cause){state.pipelines.delete(index);throw normalizeError(cause,"kernel",info.entryPoint)}}
  function command(programIndex:number,values:readonly unknown[],options?:LaunchOptions|CommandOptions):ComputeCommand{
    const info=required(definition.programs[programIndex],`program ${programIndex}`),plan=required(definition.target.programs[programIndex],`program plan ${programIndex}`);if(values.length!==info.parameters.length)throw new TachError("kernel",`${info.name} received the wrong number of parameters`,{operation:info.name});
    let owner:RuntimeOwner|undefined;const storage:BufferState<unknown>[]=[];const seen=new Map<BufferState<unknown>,string>();
    for(let i=0;i<info.parameters.length;i++){const parameter=required(info.parameters[i],`${info.name} parameter ${i}`);if(parameter.kind!=="buffer")continue;const resourceIndex=required(parameter.resource,parameter.name),state=getBufferState(values[i] as ComputeBuffer<unknown>,`${info.name}.${parameter.name}`);required(info.resources[resourceIndex],parameter.name);if(owner&&state.owner!==owner)throw new TachError("buffer",`${info.name} compute buffers belong to different Tach sessions`,{operation:info.name});const previous=seen.get(state);if(previous)throw new TachError("buffer",`${info.name}.${parameter.name} aliases ${info.name}.${previous}; buffer parameters require distinct compute buffers`,{operation:info.name});owner=state.owner;seen.set(state,parameter.name);storage[resourceIndex]=state}
    if(!owner)throw new TachError("kernel",`${info.name} has no compute buffer`,{operation:info.name});owner.assertHealthy(info.name);let configured:ReturnType<typeof launchOptions>;try{configured=launchOptions(options as LaunchOptions|undefined)}catch(cause){throw normalizeError(cause,"kernel",info.name)}let launch:number[]=[];const launchKernel=info.launch?required(definition.target.kernels[required(plan.steps.find(s=>s.kind==="dispatch")?.kernel,"launch kernel")],"launch kernel"):undefined;if(info.launch){if(configured.size!==undefined&&(typeof configured.size==="number"?info.launch.dimensions!==1:!Array.isArray(configured.size)||configured.size.length!==info.launch.dimensions))throw new TachError("kernel",`kernel requires an exact ${info.launch.dimensions}D launch size`,{operation:info.name});if(configured.size!==undefined){launch=typeof configured.size==="number"?[configured.size]:[...configured.size]}else if(info.launch.inferFromResource===undefined){const size=defaultLaunch(info.launch.dimensions,launchKernel!.workgroupSize);launch=typeof size==="number"?[size]:[...size]}}
		for(const step of plan.steps){if(step.kind!=="dispatch"||(step.parameters??[]).some(source=>source.kind==="shape"||source.kind==="repeat"))continue;const kernel=required(definition.target.kernels[required(step.kernel,"dispatch kernel")],"kernel");if(kernel.parameterBlock)try{packBlock(info,kernel.parameterBlock,step.parameters??[],values,storage,launch,configured.repeat)}catch(cause){throw normalizeError(cause,"kernel",info.name)}}let transientBytes:number[]=[];
    return createComputeCommand({owner,async prepare():Promise<PreparedCommand>{for(let i=0;i<storage.length;i++){const state=storage[i],resource=info.resources[i];if(state&&resource)materialize(state,resource)}if(info.launch&&launch.length===0){const index=required(info.launch.inferFromResource,"launch resource");const size=required(runtimeLength(required(info.resources[index],"launch resource"),required(storage[index],"launch buffer")),"launch size");launch=[size]}transientBytes=plan.transients.map(t=>{const length=shapeValue(t.length,values,info.resources,storage,launch);if(length===0)throw new RangeError("transient length must be positive");const bytes=length*t.stride;if(!Number.isSafeInteger(bytes)||bytes>0xffff_ffff)throw new RangeError("transient byte size overflow");return bytes});const compiledKernels=new Map<number,CompiledKernel>();for(const step of plan.steps)if(step.kind==="dispatch"&&!compiledKernels.has(required(step.kernel,"dispatch kernel")))compiledKernels.set(step.kernel!,await compiled(owner!,step.kernel!));const parameters:Uint8Array[]=[];for(const step of plan.steps){if(step.kind!=="dispatch")continue;const kernel=required(definition.target.kernels[required(step.kernel,"dispatch kernel")],"kernel");if(kernel.parameterBlock)parameters.push(packBlock(info,kernel.parameterBlock,step.parameters??[],values,storage,launch,configured.repeat))}return{parameters,scratch:plan.transients.map((t,i)=>({color:t.color,byteSize:required(transientBytes[i],"transient bytes")})),encode(pass,parameterBindings,scratch){const record=()=>{let parameterIndex=0;for(const step of plan.steps){if(step.kind!=="dispatch")continue;const kernelIndex=required(step.kernel,"dispatch kernel"),kernel=required(definition.target.kernels[kernelIndex],"kernel"),compiledKernel=required(compiledKernels.get(kernelIndex),"compiled kernel");const entries:BufferBindGroupEntry[]=[];for(const source of step.resources){const binding=required(source.binding,"resource binding");if(source.kind==="external"){const state=required(storage[source.resource],"external resource"),resource=required(info.resources[source.resource],"external resource");entries.push({binding,resource:{buffer:materialize(state,resource)}})}else{const transient=required(plan.transients[source.resource],"transient");entries.push({binding,resource:{buffer:required(scratch.get(transient.color),"scratch buffer"),size:required(transientBytes[source.resource],"transient byte size")}})}}let dynamic:number[]=[];if(kernel.parameterBlock){const parameter=required(parameterBindings[parameterIndex++],"parameter binding");entries.push({binding:kernel.parameterBlock.binding,resource:{buffer:parameter.buffer,size:kernel.parameterBlock.byteSize}});dynamic=[parameter.offset??0]}entries.sort((a,b)=>a.binding-b.binding);pass.setPipeline(compiledKernel.pipeline);pass.setBindGroup(0,owner!.bindGroup(`Tach ${kernel.entryPoint} group 0`,compiledKernel.bindGroupLayouts[0]!,entries),dynamic);const domain=(step.domain??[]).map(axis=>shapeValue(axis,values,info.resources,storage,launch));const groups=workgroups(domain.length===1?domain[0]!:domain as [number,number,number],kernel.workgroupSize,domain.length as 1|2|3);pass.dispatchWorkgroups(...groups)}};if(plan.repeat==="program")for(let i=0;i<configured.repeat;i++)record();else record()}}}})}
  return Object.freeze({command});
}

function defaultLaunch(dimensions:1|2|3,workgroup:readonly[number,number,number]):LaunchSize{return dimensions===1?workgroup[0]:dimensions===2?[workgroup[0],workgroup[1]]:workgroup}

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
