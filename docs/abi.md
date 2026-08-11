# Tach resource, host, and runtime ABI

Tach owns one external contract across source kernels, WGSL/WebGPU,
SPIR-V/Vulkan, generated JavaScript/TypeScript, and reflection metadata. This
document defines resource identity, names, bytes, launches, host values,
lifetime, synchronization, and errors.

The source language is described in [the language reference](language.md).
Core IR and optimization are described in [the IR reference](ir.md).

## 1. What the ABI covers

The ABI has four connected parts:

1. **Resource ABI:** which kernel parameter maps to which shader resource.
2. **Memory ABI:** how a logical Tach value occupies host-visible bytes.
3. **Launch ABI:** how logical extents become workgroup counts.
4. **Runtime ABI:** how generated commands, buffers, sessions, synchronization,
   and failures behave in TypeScript.

All four are compiler-owned. Tach source contains no descriptor coordinates,
padding fields, byte offsets, target address spaces, or WebGPU lifecycle code.

## 2. Public names

An exported kernel keeps its source name everywhere:

```text
Tach kernel       integrate
WGSL entry point  integrate
SPIR-V entry      integrate
metadata name     integrate
JavaScript export integrate
TypeScript export integrate
```

Generated host-visible type, kernel, and parameter names must be portable
ASCII JavaScript/TypeScript identifiers and must not conflict with reserved
generated names or language keywords. Semantic source identifiers may use
Unicode where no generated public identifier is required.

Only compiler-private shader symbols are mangled. Applications must not depend
on private names.

## 3. Resource identity and bindings

Every compute parameter becomes one module resource in deterministic
declaration traversal order. A compute function records the positional mapping
from its source parameters to those module resources.

For module resource index `N`, both current targets use:

```text
WGSL       @group(0) @binding(N)
SPIR-V     DescriptorSet 0, Binding N
metadata   group: 0, binding: N
```

The index is global to the compiled module, not restarted for each kernel. A
kernel's metadata lists only the resources that kernel receives. Source cannot
override these coordinates.

Each resource records:

```text
name
kind          uniform | buffer
logical type
inferred access
module index
physical host layout
```

### Kinds and access

`uniform<T>` becomes a read-only uniform binding. It must be fixed-size and
cannot contain atomics.

`buffer<T>` becomes a storage binding. Tach infers read-only versus mutable
access from stores and atomic operations. A type containing an atomic is
physically `read_write` even if source only calls `atomicLoad`, because the
target storage class must admit atomic access.

The generated WebGPU bind-group layout uses `uniform`, `read-only-storage`, or
`storage` from that proven result. SPIR-V variables and decorations describe
the same resource contract.

### Non-aliasing

Different resource parameters of one kernel are distinct, non-aliasing memory
objects. This is a language and ABI rule, not an optimizer guess.

The TypeScript runtime rejects one `ComputeBuffer` passed to two parameters of
the same generated command. Native Vulkan callers must enforce the same rule.
An in-place algorithm should declare one buffer parameter and read/write
through that parameter.

## 4. Canonical host layout

Tach logical types never acquire artificial padding fields. `src/layout`
computes a separate physical layout with 32-bit checked size arithmetic.

### Scalars and vectors

| Tach type | Byte size | Byte alignment |
|---|---:|---:|
| `i32` | 4 | 4 |
| `u32` | 4 | 4 |
| `f32` | 4 | 4 |
| `atomic<i32>` | 4 | 4 |
| `atomic<u32>` | 4 | 4 |
| `i32x2`, `u32x2`, `f32x2` | 8 | 8 |
| `i32x3`, `u32x3`, `f32x3` | 12 | 16 |
| `i32x4`, `u32x4`, `f32x4` | 16 | 16 |

`bool` is a control/value type and has no host ABI representation.

### Structs

Struct alignment is at least 16 bytes. Each field begins at the next multiple
of its required alignment. A nested struct is placed and reserved at a
16-byte-aligned extent. A fixed-size struct's total size is rounded up to its
alignment.

For example:

```tach
type Particle = {
  position: f32x3,
  mass: f32,
  velocity: f32x3,
};
```

has this layout:

```text
offset  0..11   position
offset 12..15   mass
offset 16..27   velocity
offset 28..31   trailing padding

alignment 16
byte size 32
```

Field order is therefore semantically visible to the byte ABI even though a
source struct literal may list fields in any order.

### Fixed arrays

For an element with size `S` and alignment `A`:

```text
array stride = roundUp(A, S)
array size   = stride * element count
array align  = A
```

Fixed arrays are part of the layout engine and are currently admitted to
workgroup memory, not host resource parameters.

### Runtime arrays and tails

A direct runtime array `T[]` has no fixed byte size. Its alignment is the
element alignment and its stride is `roundUp(elementAlignment, elementSize)`.

A struct may contain one runtime array as its final field. The struct's
`byteSize` is then the fixed prefix ending at the runtime-tail offset; actual
resource size is:

```text
runtimeOffset + elementCount * runtimeStride
```

For example:

```tach
type Samples = {
  count: u32,
  values: f32[],
};
```

has a 16-byte alignment, a four-byte fixed prefix, runtime offset `4`, and
runtime stride `4`. A resource must contain at least one complete runtime
element, so its minimum binding size is `8` bytes.

## 5. Resource binding size

Fixed-size host resources are physically wrapped so the binding occupies:

```text
roundUp(16, logical layout size)
```

Consequently `uniform<f32>` has logical size/alignment `4/4` but a physical
`byteSize` and `minimumByteSize` of `16`. The wrapper does not change the
logical value type used by helpers or optimization.

Runtime resources are not rounded to a fixed 16-byte wrapper. Their minimum is
one element:

```text
minimumByteSize = runtimeOffset + runtimeStride
```

Actual materialized buffers can contain more whole elements. Partial elements
and zero-element runtime resources are rejected by the managed runtime.

## 6. Physical representation in WGSL

The WGSL backend emits group `0` bindings at each module resource index.
Fixed-size bindings use compiler-private wrappers with a 16-byte-aligned first
field so WebGPU observes the Tach binding size. Runtime arrays keep the natural
Tach stride.

Storage access is emitted as read-only or read-write from semantic inference.
Entry-point parameter inputs are limited to the coordinates the lowered kernel
actually uses.

The runtime uses compiler metadata for `minBindingSize` and packing. It never
parses WGSL to infer a layout.

## 7. Physical representation in SPIR-V

The SPIR-V backend decorates every host-visible resource variable with
descriptor set `0` and its module binding index. Descriptor-reachable structs
and arrays receive the exact Tach member offsets and array strides.

There are two intentionally separate representations:

```text
logical representation    SSA values, helper values, Workgroup memory
physical representation   Uniform and StorageBuffer memory
```

Logical aggregates are undecorated. Host-visible aggregates carry ABI
decorations. Aggregate loads and stores cross the boundary field by field, so
padding bytes never appear as logical members.

Tach's binary validator checks both directions: descriptor-reachable physical
types must carry the expected layout, while logical and Workgroup-reachable
types must not be contaminated by host-layout decorations.

## 8. Reflection metadata

Every build writes `<module>.tach.json`. Its exact top-level shape is:

```ts
interface Metadata {
  types: Array<{
    name: string;
    byteSize: number;
    alignment: number;
    runtime: boolean;
    fields: Array<{
      name: string;
      type: string;
      byteOffset: number;
    }>;
  }>;
  resources: Array<{
    name: string;
    group: number;
    binding: number;
    kind: "uniform" | "storage";
    access: "read" | "read_write";
    type: string;
    byteSize?: number;
    alignment: number;
    runtime: boolean;
    runtimeOffset?: number;
    runtimeStride?: number;
    minimumByteSize: number;
  }>;
  kernels: Array<{
    name: string;
    entryPoint: string;
    dimensions: number;
    workgroupSize: [number, number, number];
    resources: Array<{
      name: string;
      resource: number;
    }>;
  }>;
}
```

Zero-valued optional numeric fields are omitted by JSON encoding. For the
`scale` kernel in the project README, metadata is:

```json
{
  "types": [],
  "resources": [
    {
      "name": "data",
      "group": 0,
      "binding": 0,
      "kind": "storage",
      "access": "read_write",
      "type": "f32[]",
      "alignment": 4,
      "runtime": true,
      "runtimeStride": 4,
      "minimumByteSize": 4
    },
    {
      "name": "factor",
      "group": 0,
      "binding": 1,
      "kind": "uniform",
      "access": "read",
      "type": "f32",
      "byteSize": 16,
      "alignment": 4,
      "runtime": false,
      "minimumByteSize": 16
    }
  ],
  "kernels": [
    {
      "name": "scale",
      "entryPoint": "scale",
      "dimensions": 1,
      "workgroupSize": [256, 1, 1],
      "resources": [
        { "name": "data", "resource": 0 },
        { "name": "factor", "resource": 1 }
      ]
    }
  ]
}
```

Metadata is deliberately plain and unversioned while Tach is early. Generated
JavaScript, declarations, metadata, and shaders must be rebuilt together from
the same `.tach` source; mixing artifacts from different compiler builds is
not supported.

## 9. Generated TypeScript contract

Named Tach structs become readonly interfaces. Scalar values become `number`;
vectors become readonly tuples. A kernel becomes a positional function in
source parameter order:

```ts
import type {
  ComputeBuffer,
  ComputeDispatch,
  DispatchOptions,
} from "@depths/tach";

export function scale(
  data: ComputeBuffer<Float32Array | readonly number[]>,
  factor: number,
  $dispatch?: DispatchOptions<number>,
): ComputeDispatch;
```

The optional final `$dispatch` parameter is generated, not part of Tach source.
Its `size` type is tied to kernel rank:

```text
1D  number
2D  readonly [x, y]
3D  readonly [x, y, z]
```

Runtime arrays of scalar values may use their matching `Float32Array`,
`Int32Array`, or `Uint32Array`. Runtime arrays of tightly packed two- and
four-lane vectors may use a flat matching typed array or nested tuples.
Three-lane vectors use nested tuples because their 16-byte element stride
contains padding.

Generated JavaScript exports the source-named command constructors and embeds
the WGSL plus private resource/layout descriptors. It imports execution only
from `@depths/tach/internal`; packing and reflection plumbing are not public
application APIs.

## 10. Host values and buffer materialization

`gpu.buffer(value)` creates a session-owned `ComputeBuffer<T>` and stores a
structured clone of the initial value. It does not immediately know which
compiled resource layout will consume that value.

The first submitted command that uses the buffer:

1. selects the compiler-emitted resource codec;
2. validates and packs the host value;
3. requires at least the resource's minimum binding size;
4. creates and uploads the WebGPU buffer; and
5. fixes that buffer's layout and byte length.

Before materialization, `write(value)` replaces the cloned host value and may
change its eventual size. After materialization, writes must pack to the exact
same byte length. A buffer cannot later be used as a differently laid-out
resource; create another `ComputeBuffer` instead.

Packing validates numeric shape and range. Integers must fit `i32`/`u32`;
vectors and fixed arrays must have the exact count; structs must provide every
declared field; runtime storage must end on a complete element. Multi-byte
numbers use little-endian byte order.

On little-endian hosts, correctly typed scalar and tightly packed vector
arrays can cross the boundary without element-wise packing. Readback preserves
that typed-array representation. Other values use compiler-generated
`DataView` codecs and return cloned object/array structures.

`ComputeBuffer` exposes only:

```ts
write(value: T): void;
read(): Promise<T>;
destroy(): void;
```

Reading an unmaterialized buffer returns a clone of its host value. Reading a
materialized buffer waits for prior submissions, copies to a temporary map-read
buffer, decodes the bytes, destroys the temporary, and returns a clone.
`destroy()` is idempotent; subsequent use is a lifecycle error.

## 11. Commands and launch geometry

A generated kernel call validates its arguments and returns an opaque
`ComputeDispatch`. It does not submit. Accidentally awaiting the command throws
a targeted error; the only execution boundary is:

```ts
await gpu.submit(commandA, commandB);
```

Uniform arguments are packed and snapshotted when the command is constructed.
Buffer arguments remain live resident handles. All command buffers must belong
to the same Tach session.

`submit` records its commands in order into one compute pass and one command
buffer, then performs one queue submission. `dispatches` repeats that command's
`dispatchWorkgroups` call inside the same pass:

```ts
step(state, params, { size: count, dispatches: 100 })
```

Every explicit size component must be a positive safe integer and the value's
rank must exactly match the kernel. Workgroup counts are:

```text
groups[axis] = ceil(logicalSize[axis] / workgroupSize[axis])
```

The final workgroup may therefore execute coordinates outside the logical
extent. Kernel code owns bounds guards.

When `size` is omitted:

- a 1D kernel infers it from the first runtime-sized storage resource;
- otherwise the logical extent defaults to exactly one workgroup.

Two- and three-dimensional problem extents cannot be inferred from a flat
buffer and should normally be explicit.

## 12. Sessions and ownership

The public runtime exposes one session behavior through two lifetimes.

### Scoped lifetime

```ts
const result = await tach(async (gpu) => {
  const data = gpu.buffer(initial);
  await gpu.submit(scale(data, 2));
  return data.read();
});
```

`tach(work, options?)` opens a session, calls `work`, waits for submitted GPU
work, converts any failure to `Result`, and closes the session. Every buffer
created by that session is destroyed on exit. A returned `ComputeBuffer` is
therefore not usable after the scope.

### Persistent lifetime

```ts
const opened = await openTach();
if (!opened.ok) throw new Error(opened.error.message);

const gpu = opened.value;
try {
  // Reuse resident buffers and cached pipelines across many submissions.
  await gpu.idle();
} finally {
  gpu.close();
}
```

`openTach(options?)` returns the same session without automatic shutdown. The
caller owns synchronization and `close()`. This form supports frame loops and
long iterative jobs without reacquiring an adapter/device or recreating
resident state.

Both forms use the same `ComputeBuffer`, commands, caches, submission ordering,
and error model. There is no batch-only and frame-only execution split.

## 13. Submission, synchronization, and caches

`submit()` awaits asynchronous pipeline preparation and queue submission, but
does not wait for GPU completion. Session submissions are chained so calls
issued from interleaved promises retain invocation order.

The explicit completion boundaries are:

- `gpu.idle()`;
- `buffer.read()` for all prior session submissions; and
- successful or failed exit from `tach(...)`.

Generated modules cache a shader module and compute pipelines per WebGPU
device. Sessions cache bind groups by layout and exact buffer range. Uniforms
share an aligned session buffer sized against
`minUniformBufferOffsetAlignment`; it grows geometrically and is reused under
WebGPU queue ordering. Destroying a buffer or growing the uniform arena clears
affected bind-group cache entries.

`gpu.close()` is idempotent. It destroys all owned GPU buffers, the uniform
arena, bind-group state, event handling, and the WebGPU device.

## 14. Error contract

`openTach()` returns `Promise<Result<Tach>>`. `tach(...)` returns
`Promise<Result<T>>`:

```ts
type Result<T, E = TachError> =
  | { readonly ok: true; readonly value: T }
  | { readonly ok: false; readonly error: E };
```

`TachError` contains a stable category, message, optional operation, and
optional cause. Current categories cover WebGPU availability, adapter/device
acquisition and loss, GPU validation/out-of-memory/internal errors, buffers,
kernels, lifecycle, user callback failures, and compiler delivery/execution.

Operations on an open session throw `TachFailure` carrying that structured
error. An enclosing `tach(...)` catches and normalizes it. Asynchronous WebGPU
errors are retained by the session and surface at a later submission or
synchronization boundary rather than being silently discarded.

## 15. Native Vulkan caller obligations

The TypeScript runtime enforces the WebGPU side automatically. A native
consumer of `.spv` and `.tach.json` must perform the equivalent work:

1. create descriptor bindings at set `0` and the recorded module binding
   numbers used by the selected kernel;
2. allocate at least `minimumByteSize`, with every runtime tail ending on its
   recorded stride;
3. pack scalars, vectors, fields, padding, and runtime tails according to the
   Tach layout;
4. provide uniform versus storage usage and read-only/read-write access
   compatible with metadata;
5. bind distinct memory objects for distinct resource parameters;
6. dispatch `ceil(logicalExtent / workgroupSize)` groups on every axis; and
7. insert Vulkan synchronization appropriate for work before and after the
   Tach pipeline.

Workgroup-memory initialization and intra-kernel barriers are already encoded
in the SPIR-V module. Bounds outside a rounded-up logical extent remain the
kernel author's responsibility.

## 16. Compatibility rule

The `.tach` file is the source of truth. The seven generated artifacts form
one compiler result and must travel together. Public kernel names, resource
order, byte layout, launch rank, and runtime lifetime are ABI, but the metadata
schema is intentionally not versioned yet. Recompile rather than hand-editing
or combining generated output.
