# Tach program, memory, and runtime ABI

Tach owns one external contract across a complete source project, generated
JavaScript/TypeScript, WebGPU/WGSL, Vulkan/SPIR-V, and reflection metadata.
This document defines names, resource identity, bytes, physical dispatch
plans, launch geometry, lifetime, and synchronization.

Application code normally consumes generated modules and `@depths/tach`; it
does not implement this ABI. The Tach-owned WebGPU/Vulkan drivers and compiler
contributors need the complete contract.

See [the language guide](language.md) for source semantics and
[the IR guide](ir.md) for logical programs, kernel templates, and executable
plans.

## 1. Three connected boundaries

The ABI separates three identities that a one-function example can make look
the same:

```text
public program       host-callable generated function
logical stage        indexed portable GPU operation
physical kernel      target entry created for one dispatch
```

For baseline shorthand:

```tach
export function scale[i](values: buffer<float32[]>, factor: float32) {
  if (i < values.length) {
    values[i] *= factor;
  }
}
```

`scale` is both the source stage name and public program name, but its emitted
shader entry is private, currently `_tach_k0`. A multi-stage public program has
several private physical entries. Host code must execute the generated plan;
it must not infer a shader entry from the public name.

The contract has six parts:

1. **Program ABI:** public parameters, resources, and optional launch input.
2. **Plan ABI:** ordered dispatches, target kernels, transients, and barriers.
3. **Binding ABI:** storage and parameter-block bindings for each physical
   kernel.
4. **Memory ABI:** canonical host-visible bytes.
5. **Runtime ABI:** commands, ownership, synchronization, and errors.
6. **Documentation ABI:** target-neutral descriptions used for JSDoc and
   generated Markdown.

## 2. Public names and generated signatures

Every source-named struct type and every exported program becomes a
JavaScript/TypeScript export in the project's single package entry point.
Their names and public parameter names must be portable ASCII identifiers and
must avoid reserved JavaScript/TypeScript and generated names. Struct names
also exclude `Float32Array`, `Int32Array`, `Uint32Array`, and `ReadonlyArray`,
which retain their host-collection meanings in generated signatures. Runtime
API names remain available because all runtime types use compiler-private
`$...` aliases. Private helpers, stages, physical entries, wrappers, and fields
may be mangled.

An exported indexed shorthand receives `LaunchOptions`:

```ts
import type {
  ComputeBuffer as $ComputeBuffer,
  ComputeCommand as $ComputeCommand,
  LaunchOptions as $LaunchOptions,
} from "@depths/tach";

export function scale(
  values: $ComputeBuffer<Float32Array | readonly number[]>,
  factor: number,
  $launch?: $LaunchOptions<number>,
): $ComputeCommand;
```

An explicit public program derives every dispatch domain in source and
receives only `CommandOptions`:

```ts
import type {
  CommandOptions as $CommandOptions,
  ComputeBuffer as $ComputeBuffer,
  ComputeCommand as $ComputeCommand,
} from "@depths/tach";

export function transform(
  input: $ComputeBuffer<Float32Array | readonly number[]>,
  output: $ComputeBuffer<Float32Array | readonly number[]>,
  count: number,
  factor: number,
  bias: number,
  $options?: $CommandOptions,
): $ComputeCommand;
```

The final options object is compiler-generated and is not a Tach source
parameter. Both forms return an opaque command; neither executes immediately.

## 3. Public resources and non-aliasing

Each public `buffer<T>` parameter creates one external resource in that
program. Public parameters retain source order and map buffer parameters to
resource-table indices. Plain parameters remain typed host values.

Different public buffer parameters are distinct, non-aliasing objects. The
shared runtime rejects one `ComputeBuffer` passed in more than one buffer
position of a command before either driver sees it. In-place
work uses one buffer parameter.

Within an explicit program, a stage buffer argument names either an external
resource or a compiler-managed transient. One resource cannot fill two buffer
formals of the same dispatch. Resource-version checks prove that a transient
read has been defined by an earlier complete write.

Access is inferred from all stages that touch an external resource:

| Logical access | Meaning |
|---|---|
| `read` | no stage writes it |
| `write` | stages define it without reading its previous value |
| `readWrite` | stages both read and write |
| `atomic` | any stage performs atomic access |

This program-level access appears in documentation. Physical shader bindings
use the read/read-write requirement of their particular stage.

## 4. Canonical host layout

`src/layout` computes one checked, target-independent layout. Logical types do
not acquire fake padding fields; offsets and padding exist only at the host
boundary.

### Scalars and vectors

| Tach type | Size | Alignment |
|---|---:|---:|
| `int32`, `uint32`, `float32` | 4 | 4 |
| `atomic<int32>`, `atomic<uint32>` | 4 | 4 |
| two-lane numeric vector | 8 | 8 |
| three-lane numeric vector | 12 | 16 |
| four-lane numeric vector | 16 | 16 |

`bool` has no direct storage-buffer representation. In a physical parameter
block, each logical bool leaf becomes one `uint32` word containing `0` or `1`.

### Structs

Struct alignment is at least 16 bytes. A field begins at its required aligned
offset. Nested structs reserve a 16-byte-aligned extent. A fixed struct's final
size is rounded to its alignment.

```tach
type Particle = {
  position: float32x3,
  mass: float32,
  velocity: float32x3,
};

export function preserve[i](particles: buffer<Particle[]>) {
  if (i < particles.length) {
    particles[i] = particles[i];
  }
}
```

`Particle` has:

```text
offset  0..11  position
offset 12..15  mass
offset 16..27  velocity
offset 28..31  padding
alignment 16, size 32
```

Source struct-literal order is irrelevant, but declaration field order is part
of the byte ABI.

### Arrays and runtime tails

For a fixed or runtime array element with size `S` and alignment `A`:

```text
stride = roundUp(A, S)
```

A fixed array has `stride * count` bytes. Fixed arrays currently exist only in
workgroup memory.

A direct runtime array `T[]` has no fixed size. A struct may contain one
runtime array as its final field; the struct layout records a fixed prefix,
tail offset, and tail stride. A materialized runtime resource must contain at
least one complete element:

```text
minimumByteSize = runtimeOffset + runtimeStride
actualByteSize  = runtimeOffset + elementCount * runtimeStride
```

Partial and zero-element runtime resources are rejected.

### Fixed resource wrapper size

A public fixed-size buffer is allocated as:

```text
roundUp(16, logicalSize)
```

This matches the compiler-generated storage wrapper. The public metadata
records that padded `byteSize`. Runtime arrays retain their natural stride so
padding cannot become a phantom element.

All size arithmetic is checked against the 32-bit ABI limit.

## 5. Physical kernels and bindings

Target lowering clones one logical indexed stage for every surviving program
dispatch. The current policy is intentionally one physical kernel per
dispatch, in target-plan order.

For each physical kernel:

- the entry point is `_tach_kN` in target-plan order;
- storage bindings are dense from binding `0` in source-formal order;
- WebGPU uses group `0` and Vulkan uses descriptor set `0`;
- each binding records read/read-write access, logical type, and minimum size;
- the selected workgroup is part of the target plan; and
- a parameter block, when present, takes the next binding.

Binding numbers therefore restart for each physical kernel's pipeline layout.
They are not module-global public-resource IDs.

## 6. One value block per physical kernel

After target-neutral Kernel IR optimization and backend parameter pruning,
`src/abi` flattens the remaining plain stage parameters into one private
struct. It walks parameters in order and struct fields in declaration order.
Numeric leaves retain their type; bool leaves become `uint32`.

The canonical layout determines every field offset and rounds the struct to
its 16-byte alignment. A stage with no remaining values has no parameter
block. A block is limited to 16 KiB, the portable floor shared by the target
profiles.

The same plan drives WGSL, SPIR-V, embedded metadata, TypeScript packing, and
the Vulkan runtime. Kernel IR continues to contain logical parameters; no
binding, physical bool, or padding member leaks into it.

## 7. Program plans and shape evaluation

Each public program receives one plan per target. A plan contains:

```text
program index
physical kernels
ordered dispatch/barrier steps
external/transient resource sources for each step
value sources for each parameter-block leaf
transient lifetimes and allocation colors
repeat mode and optional repeat barrier
```

Dispatch domains are trees of checked shape operations. Leaves can reference a
public value/path, an external runtime-array length, a baseline launch axis, or
a constant. Operators are `add`, `sub`, `mul`, `div`, `rem`, `min`, `max`, and
`ceilDiv`.

The host evaluates shapes with `uint32` range checks. Resource lengths come
from the materialized byte length, runtime-tail offset, and stride. A dispatch
axis or transient length must be positive.

Parameter-block value sources can reference public values/paths, literal bool,
`int32`, `uint32`, or exact `float32` bits, a shape expression, or the command
repeat count.

## 8. Transient allocation and synchronization

Each transient records element type, stride, alignment, minimum size, length
shape, first/last use, and an allocation color. Non-overlapping lifetimes can
share a color. At preparation time the runtime evaluates byte sizes and gives
each color one buffer large enough for the largest active requirement.

The WebGPU session retains scratch buffers by color and grows them
geometrically. Replaced buffers are retired until the next `idle()` so queued
work cannot observe destroyed storage.

WebGPU records all dispatches of submitted commands into one compute pass in
program order. The SPIR-V plan inserts explicit compute-to-compute barriers
between adjacent steps when an earlier stage writes a resource touched by the
next stage. It also records a barrier across repeated program iterations when
required.

## 9. Repeat semantics

Both generated option types include `repeat?: number`. The value must be a
positive integer within `uint32`; omission means `1`.

For an ordinary multi-dispatch program, repeat executes the complete ordered
plan repeatedly:

```text
stage A -> stage B -> stage A -> stage B -> ...
```

For a one-dispatch program, target lowering may internalize repeat as an
invocation-local loop only when the stage:

- contains no source loop;
- has no atomic, shared-memory, or barrier effects; and
- touches buffers only at the exact current 1D coordinate.

Under those proofs, each invocation can perform all repetitions while its
values remain in registers, preserving observable behavior and removing
repeat dispatches. Otherwise the host plan repeats the dispatch literally.

The metadata field `repeat` records `"invocation-loop"` or `"program"`; native
runtimes must obey it.

## 10. Metadata schema 1

Schema-1 execution metadata is embedded in the singular generated JavaScript
module and consumed by either host driver. It is not a build sidecar or public
package export. The complete internal shape is:

```ts
interface Metadata {
  schema: 1;
  types: Array<{
    name: string;
    fields: Array<{ name: string; type: string }>;
  }>;
  programs: PublicProgram[];
  targets: { web: TargetPlan; spirv: TargetPlan };
}

interface PublicProgram {
  name: string;
  parameters: Array<{
    name: string;
    kind: "buffer" | "value";
    type: string;
    resource?: number;
  }>;
  resources: Array<{
    name: string;
    type: string;
    byteSize?: number;
    alignment: number;
    runtime: boolean;
    runtimeOffset?: number;
    runtimeStride?: number;
    minimumByteSize: number;
    layout: HostLayout;
  }>;
  launch?: {
    dimensions: 1 | 2 | 3;
    inferFromResource?: number;
  };
}

interface TargetPlan {
  vulkan?: "1.3";
  spirv?: "1.6";
  features?: Array<"synchronization2" | "shaderZeroInitializeWorkgroupMemory">;
  kernels: Array<{
    entryPoint: string;
    workgroupSize: [number, number, number];
    bindings: Array<{
      group: 0;
      binding: number;
      access: "read" | "read_write";
      type: string;
      minimumByteSize: number;
    }>;
    parameterBlock?: {
      group: 0;
      binding: number;
      byteSize: number;
      fields: Array<{
        type: string;
        byteOffset: number;
        layout: HostLayout;
      }>;
    };
  }>;
  programs: Array<{
    program: number;
    transients: Transient[];
    steps: Step[];
    repeatBarrier?: Step;
    repeat: "program" | "invocation-loop";
  }>;
}

interface Transient {
  type: string;
  stride: number;
  alignment: number;
  minimumByteSize: number;
  length: Shape;
  color: number;
  firstStep: number;
  lastStep: number;
}

interface Step {
  kind: "dispatch" | "barrier";
  kernel: number;
  domain?: Shape[];
  resources: Array<{
    binding: number;
    kind: "external" | "transient";
    resource: number;
  }>;
  parameters?: ValueSource[];
}

interface Shape {
  op: "constant" | "parameter" | "resourceLength" | "launchAxis"
    | "add" | "sub" | "mul" | "div" | "rem"
    | "min" | "max" | "ceilDiv";
  value?: number;
  parameter: number;
  resource: number;
  path?: string[];
  axis: number;
  left?: Shape;
  right?: Shape;
}

interface ValueSource {
  kind: "parameter" | "bool" | "i32" | "u32" | "f32Bits"
    | "shape" | "repeat";
  parameter: number;
  path?: string[];
  value?: boolean | number;
  expression?: Shape;
}

interface HostLayout {
  kind: "bool" | "i32" | "u32" | "f32" | "vector"
    | "array" | "runtime" | "struct";
  size?: number;
  stride?: number;
  count?: number;
  runtime?: boolean;
  elem?: HostLayout;
  fields?: Array<{ name: string; offset: number; type: HostLayout }>;
}
```

Both targets are mandatory and contain parallel `kernels` and `programs`
arrays. Only `targets.spirv` carries the exact Vulkan/SPIR-V feature profile:

```text
targets.web.kernels[]       physical entry, workgroup, bindings, value block
targets.web.programs[]      steps, shapes, transients, repeat mode
targets.spirv.kernels[]     target-specific physical entries
targets.spirv.programs[]    target-specific barriers and plan
```

Array indices are cross-references. Every program plan records its public
program index; a dispatch step records a physical-kernel index; resource
sources identify an external or transient table plus its index. Schema
validation rejects dangling indices, mismatched target/program counts,
invalid bindings, layouts, steps, shapes, and value sources.

Zero-valued optional numeric fields may be absent in JSON. Consumers must use
the schema and discriminator fields rather than property presence as a general
truth test.

Schema 1 is compiler/runtime internal while Tach is pre-1.0. Rebuild all
artifacts together rather than treating it as a stable third-party wire
format.

### Target-neutral project description: schema 2

The private compiler engine writes one schema-2 JSON value to the TypeScript
CLI over stdout. This is a compiler/tooling protocol, not a generated file. It
contains no JavaScript or TypeScript syntax:

```ts
interface ProjectDescription {
  schema: 2;
  name: string;       // Tach project identity
  version: string;
  package: string;    // generated npm package identity
  title: string;
  summary: string;
  modules: Array<{
    name: string;
    kernels: Array<{
      name: string;
      identity: string; // canonical module/kernel
      title?: string;
      summary?: string;
      types: DocumentedType[];
      functions: DocumentedFunction[];
    }>;
  }>;
}

interface DocumentedFunction {
  name: string;
  role: "helper" | "stage" | "kernel" | "program";
  exported: boolean;
  summary?: string;
  coordinates: Array<{ name: string; description?: string }>;
  parameters: Array<{
    name: string;
    type: TypeRef;
    buffer: boolean;
    access?: "read" | "write" | "readWrite" | "atomic";
    description?: string;
  }>;
  returns?: { type: TypeRef; description?: string };
}

interface DocumentedType {
  name: string;
  summary?: string;
  fields: Array<{
    name: string;
    type: TypeRef;
    description?: string;
  }>;
}

interface TypeRef {
  tach: string;
  kind:
    | "void" | "bool" | "i32" | "u32" | "f32"
    | "vector" | "struct" | "atomic"
    | "fixedArray" | "runtimeArray";
  name?: string;
  elem?: TypeRef;
  count?: number;
  lanes?: number;
}
```

`TypeRef` carries the Tach spelling plus a target-neutral structural kind,
element/count/lane information, and a struct name where applicable. Optional
numeric fields are omitted when zero; consumers branch on `kind`, not on
incidental property presence. Types and functions remain owned by their
canonical kernel. Arrays are already ordered module, kernel, declaration,
field, coordinate, and parameter sequences; renderers must preserve that
compiler-owned order. The TypeScript renderer uses this description to
generate Markdown and validated package-name imports; there is no dependency
from Go source to TypeScript presentation.

## 11. WGSL representation

WGSL contains all physical kernels for the WebGPU plan. Each has its own
group-0 storage variables, wrappers, optional uniform block, selected builtin
inputs, and private entry name.

Fixed resources use an aligned wrapper; runtime tails preserve natural
stride. Parameter blocks use the `uniform` address space and reconstruct
logical values at entry. Storage access is `read` or `read_write` from the
stage proof. Only coordinate inputs still used after backend optimization are
emitted.

The runtime builds layouts from metadata and never parses WGSL.

## 12. SPIR-V representation

SPIR-V 1.6 uses Logical addressing and the Shader capability under Tach's
Vulkan 1.3 profile. Host-visible storage and uniform aggregates carry
descriptor, member-offset, and
array-stride decorations. Logical SSA/helper values and Workgroup memory use
undecorated logical aggregate types.

Aggregate loads, stores, and parameter reconstruction cross the logical/
physical boundary field by field. Padding never becomes a logical member.

Tach guarantees zero-initialized shared memory. Every Workgroup variable has
an `OpConstantNull` initializer and the host requires
`shaderZeroInitializeWorkgroupMemory`; no synthetic store loop or barrier is
inserted. The host also requires Synchronization2 for plan barriers.

## 13. Host values and materialization

`gpu.buffer(value)` stores a structured clone without choosing a physical
layout. The first submitted use:

1. selects the compiler-emitted resource layout;
2. validates and packs the host value;
3. requires the exact fixed size or a complete valid runtime tail;
4. creates and uploads the selected driver's storage buffer; and
5. fixes that handle's codec and byte length.

Before materialization, `write(value)` may change the future size. Afterward,
it must pack to the same byte length. A buffer cannot later be interpreted by
a different layout; create a new handle.

Packing checks integer ranges, vector/array counts, struct fields, offsets,
and strides. Multi-byte values are little-endian. On little-endian hosts,
matching scalar and tightly packed vector typed arrays can cross without
element-wise packing. Three-lane vector arrays use structured tuples because
their element stride includes padding.

```ts
interface ComputeBuffer<T> {
  write(value: T): void;
  read(): Promise<T>;
  destroy(): void;
}
```

Reading an unmaterialized handle returns a clone. Reading a materialized handle
waits for earlier submissions, copies through the driver's readback path,
decodes, releases temporary transfer state, and returns a clone. `destroy()` is
idempotent; later use is a lifecycle error.

## 14. Commands and submission

A generated program call validates buffer handles, ownership, non-aliasing,
options, and every parameter block that can be evaluated immediately. The
opaque command holds its public arguments until preparation; do not mutate a
plain object/array argument between command construction and submission.

Accidentally awaiting a command throws a targeted error. Execution occurs only
through:

```ts
await gpu.submit(first, second);
```

All commands must belong to one session. `submit` prepares host-neutral plans
in argument order and passes one batch to the selected driver. WebGPU records
one compute pass/command buffer; Vulkan records a pooled native command buffer
with explicit barriers. Its promise resolves after preparation/submission, not
after GPU completion.

For indexed shorthand, explicit launch sizes must be positive safe integers
of exact rank. Workgroup counts are:

```text
groups[axis] = ceil(logicalSize[axis] / workgroupSize[axis])
```

When size is omitted, a 1D program uses the first runtime resource length when
available; otherwise it uses exactly one workgroup. Public explicit programs
have no host launch size because their shapes are in source.

## 15. Sessions, caches, and completion

Scoped ownership:

```ts
const result = await tach(async (gpu) => {
  const data = gpu.buffer(initial);
  await gpu.submit(scale(data, 2));
  return data.read();
});
```

The callback form opens a session, runs user code, waits for queued work,
returns the callback value, and closes every owned resource.

Persistent ownership:

```ts
const gpu = await tach();
try {
  await gpu.submit(step(state));
  await gpu.idle();
} finally {
  gpu.close();
}
```

The caller owns synchronization and closure. Both forms share one engine and
the same ownership rules.

The shared session caches ownership and submission state. WebGPU caches shader
modules and physical pipelines per device, bind groups by layout/buffer range,
one aligned parameter arena, and scratch allocations by transient color;
dynamic offsets select parameter blocks and replaced buffers retire until
`idle()`. Vulkan caches SPIR-V modules, lazy physical pipelines/layouts,
device-local external and scratch buffers, and reusable submission records
containing descriptor pool, command buffer, fence, and mapped parameter arena.

Completion boundaries are:

- `gpu.idle()` for all earlier session work;
- `buffer.read()` for earlier work plus readback; and
- exit from `tach(callback)`.

`close()` is synchronous and idempotent teardown. Call `idle()` first when
successful GPU completion must be observed before teardown.

## 16. Errors

Public asynchronous APIs follow ordinary promise semantics and normalize
failures to `TachError`:

```ts
class TachError extends Error {
  readonly code: TachErrorCode;
  readonly operation: string | undefined;
}
```

Codes distinguish WebGPU and Vulkan availability, adapter/device acquisition
and loss, Vulkan profile/native failures, GPU validation/out-of-memory/internal
errors, compiler installation/execution,
buffer and program failures, lifecycle misuse, and user callback failures.
Original causes are retained. Error scopes, device loss, and uncaptured errors
surface at submission or synchronization boundaries rather than disappearing.

## 17. Tach-owned Vulkan 1.3 obligations

The Deno driver and native library execute the target plan rather than merely
loading one entry point:

1. decode and validate schema-1 metadata;
2. select the public program and matching SPIR-V program plan;
3. allocate every external resource using its canonical layout and size;
4. evaluate transient lengths and allocate each color's maximum required size;
5. create pipelines for referenced physical kernels;
6. build each kernel's set-0 storage and optional uniform descriptors;
7. pack every parameter-block leaf from its recorded value source and layout;
8. evaluate dispatch domains and divide by recorded workgroup dimensions;
9. record steps in order, including plan barriers;
10. honor `program` versus `invocation-loop` repeat mode;
11. bind distinct memory for distinct public buffer parameters; and
12. synchronize work before upload and after execution as required by the
    embedding application.

The native boundary rejects incompatible metadata before module creation,
requires Vulkan API 1.3 plus Synchronization2 and
`shaderZeroInitializeWorkgroupMemory`, validates buffer sizes and dispatch
limits, and reports driver failures through the coarse Tach FFI wire. The Deno
correctness harness additionally runs Khronos
`spirv-val --target-env vulkan1.3` before hardware execution.

## 18. Project package and artifact compatibility

`tach.json` plus every discovered module/kernel source is the source of truth.
One successful build atomically installs one complete dual-host inventory:

```text
build/
  package.json
  index.js
  index.d.ts
  kernel.wgsl
  kernel.spv
  README.md
  docs/<module>.md
```

The `package.json` uses the project version and JavaScript package name, exposes
only `index.js`/`index.d.ts`, and declares the exact installed `@depths/tach`
version because generated JavaScript imports `@depths/tach/internal`. It does
not re-export the runtime. A consumer installs both packages and imports
`tach` separately.

The one `index.js` embeds both executable plans and sibling URLs for WGSL and
SPIR-V. Browser and Deno consumers import the same facade; runtime host
selection chooses the matching plan and shader. `tach build --verbose` adds
only compiler diagnostics. Docs-only generation preserves every compiled
entry and changes only `README.md` and `docs/`.

Public names, parameter order, logical types, host bytes, plan semantics,
runtime ownership, documentation, and package identity form one synchronized
result. Do not hand-edit generated files or combine outputs from different
compiler versions or project builds.
