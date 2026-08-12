# Tach buffer, parameter, host, and runtime ABI

Tach owns one external contract across source kernels, WGSL/WebGPU,
SPIR-V/Vulkan, generated JavaScript/TypeScript, and reflection metadata. This
document defines buffer and parameter identity, names, bytes, launches, host values,
lifetime, synchronization, and errors.

The source language is described in [the language reference](language.md).
Core IR and optimization are described in [the IR reference](ir.md).

Normal TypeScript applications do not implement this contract: generated
modules and `@depths/tach` do it. Read the early sections when debugging data
shape or launch size. Read the complete document when building another host
runtime or consuming SPIR-V directly.

## 1. What the ABI covers

The ABI has five connected parts:

1. **Buffer ABI:** which `buffer<T>` parameter maps to which storage binding.
2. **Parameter ABI:** how all plain values become one internal block per kernel.
3. **Memory ABI:** how a logical Tach value occupies host-visible bytes.
4. **Launch ABI:** how logical extents become workgroup counts.
5. **Runtime ABI:** how generated commands, buffers, sessions, synchronization,
   and failures behave in TypeScript.

All five are compiler-owned. Tach source contains no descriptor coordinates,
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

## 3. Buffers, values, and bindings

Every `buffer<T>` parameter becomes one module buffer in deterministic
declaration traversal order. Plain parameters remain logical values in Core IR.
A kernel records its original positional parameter list and maps each entry to
either a module buffer or a value ID.

For module buffer index `N`, both current targets use:

```text
WGSL       @group(0) @binding(N)
SPIR-V     DescriptorSet 0, Binding N
metadata   group: 0, binding: N
```

The index is global to the compiled module, not restarted for each kernel.
Source cannot override these coordinates.

Each module buffer records:

```text
name
logical type
inferred access
module index
physical host layout
```

`buffer<T>` becomes a storage binding. Tach infers read-only versus mutable
access from stores and atomic operations. A type containing an atomic is
physically `read_write` even if source only calls `atomicLoad`, because the
target storage class must admit atomic access.

The generated WebGPU bind-group layout uses `read-only-storage` or `storage`
from that proven result. SPIR-V variables and decorations describe the same
buffer contract.

### Non-aliasing

Different buffer parameters of one kernel are distinct, non-aliasing memory
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
| `int32` | 4 | 4 |
| `uint32` | 4 | 4 |
| `float32` | 4 | 4 |
| `atomic<int32>` | 4 | 4 |
| `atomic<uint32>` | 4 | 4 |
| `int32x2`, `uint32x2`, `float32x2` | 8 | 8 |
| `int32x3`, `uint32x3`, `float32x3` | 12 | 16 |
| `int32x4`, `uint32x4`, `float32x4` | 16 | 16 |

`bool` has no direct storage-buffer representation. It is valid in ordinary
kernel values; the parameter planner represents each bool leaf as a private
`uint32` field containing `0` or `1`.

### Structs

Struct alignment is at least 16 bytes. Each field begins at the next multiple
of its required alignment. A nested struct is placed and reserved at a
16-byte-aligned extent. A fixed-size struct's total size is rounded up to its
alignment.

For example:

```tach
type Particle = {
  position: float32x3,
  mass: float32,
  velocity: float32x3,
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
shared memory, not host parameters.

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
  count: uint32,
  values: float32[],
};
```

has a 16-byte alignment, a four-byte fixed prefix, runtime offset `4`, and
runtime stride `4`. A resource must contain at least one complete runtime
element, so its minimum binding size is `8` bytes.

## 5. Buffer binding size

Fixed-size buffers are physically wrapped so the binding occupies:

```text
roundUp(16, logical layout size)
```

Consequently `buffer<float32>` has logical size/alignment `4/4` but a physical
`byteSize` and `minimumByteSize` of `16`. The wrapper does not change the
logical value type used by loads, helpers, or optimization.

Runtime resources are not rounded to a fixed 16-byte wrapper. Their minimum is
one element:

```text
minimumByteSize = runtimeOffset + runtimeStride
```

Actual materialized buffers can contain more whole elements. Partial elements
and zero-element runtime buffers are rejected by the managed runtime.

## 6. One parameter block per kernel

After Core IR optimization, `src/abi` plans the physical representation of all
plain parameters of each kernel. It walks value parameters in source order and
struct fields in declaration order, flattening every numeric, vector, or bool
leaf into one private struct. Numeric leaves keep their type; bool leaves use a
physical `uint32`. The normal layout engine supplies the exact offsets and
rounds the completed struct to 16-byte alignment.

For a kernel with module buffers and parameter blocks already assigned, the
next block receives the next group/set `0` binding. A kernel with no plain
values has no block. A block may contain at most 16 KiB, the limit guaranteed
by Vulkan core and WebGPU compatibility mode.

The planner is shared. WGSL, SPIR-V, public metadata, generated JavaScript, the
WebGPU packer, and the Vulkan harness all consume its offsets. Core IR still
contains the original logical values; no physical field, padding byte, binding,
or bool encoding enters Tach semantics.

## 7. Physical representation in WGSL

The WGSL backend emits group `0` storage bindings at each module buffer index.
Fixed-size buffers use compiler-private wrappers with a 16-byte-aligned first
field so WebGPU observes the Tach binding size. Runtime arrays keep the natural
Tach stride. A kernel parameter block is emitted as a compiler-private block in
WGSL's `uniform` address space, then reconstructed into logical values at entry.

Storage access is emitted as read-only or read-write from semantic inference.
Entry-point parameter inputs are limited to the coordinates the lowered kernel
actually uses.

The runtime uses compiler metadata for `minBindingSize` and packing. It never
parses WGSL to infer a layout.

## 8. Physical representation in SPIR-V

The SPIR-V backend decorates every host-visible buffer and parameter-block
variable with descriptor set `0` and its planned binding. Descriptor-reachable
structs and arrays receive the exact Tach member offsets and array strides.

There are two intentionally separate representations:

```text
logical representation    SSA values, helper values, Workgroup memory
physical representation   parameter-block and StorageBuffer memory
```

Logical aggregates are undecorated. Host-visible aggregates carry ABI
decorations. Aggregate loads and stores cross the boundary field by field, so
padding bytes never appear as logical members.

Tach's binary validator checks both directions: descriptor-reachable physical
types must carry the expected layout, while logical and Workgroup-reachable
types must not be contaminated by host-layout decorations.

## 9. Generated-module metadata

Web builds embed the execution plan inside generated JavaScript. Its exact
top-level shape is:

```ts
interface Metadata {
  types: Array<{
    name: string;
    fields: Array<{
      name: string;
      type: string;
    }>;
  }>;
  resources: Array<{
    name: string;
    group: number;
    binding: number;
    kind: "storage";
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
    parameters: Array<{
      name: string;
      kind: "buffer" | "value";
      type: string;
      resource?: number;
    }>;
    parameterBlock?: {
      group: number;
      binding: number;
      byteSize: number;
      fields: Array<{
        parameter: number;
        path: string[];
        type: string;
        byteOffset: number;
      }>;
    };
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
      "name": "values",
      "group": 0,
      "binding": 0,
      "kind": "storage",
      "access": "read_write",
      "type": "float32[]",
      "alignment": 4,
      "runtime": true,
      "runtimeStride": 4,
      "minimumByteSize": 4
    }
  ],
  "kernels": [
    {
      "name": "scale",
      "entryPoint": "scale",
      "dimensions": 1,
      "workgroupSize": [256, 1, 1],
      "parameters": [
        { "name": "values", "kind": "buffer", "type": "float32[]", "resource": 0 },
        { "name": "factor", "kind": "value", "type": "float32" }
      ],
      "parameterBlock": {
        "group": 0,
        "binding": 1,
        "byteSize": 16,
        "fields": [
          { "parameter": 1, "path": [], "type": "float32", "byteOffset": 0 }
        ]
      }
    }
  ]
}
```

Metadata is deliberately plain and unversioned while Tach is early. Generated
JavaScript, declarations, metadata, and shaders must be rebuilt together from
the same `.tach` source; mixing artifacts from different compiler builds is
not supported.

## 10. Generated TypeScript contract

Named Tach structs become exported readonly object type aliases. Numeric values
become `number`, bool becomes TypeScript `boolean`, and vectors become readonly
tuples. A kernel becomes a positional function in source parameter order:

```ts
import type {
  ComputeBuffer,
  ComputeCommand,
  LaunchOptions,
} from "@depths/tach";

export function scale(
  values: ComputeBuffer<Float32Array | readonly number[]>,
  factor: number,
  $launch?: LaunchOptions<number>,
): ComputeCommand;
```

The optional final `$launch` parameter is generated, not part of Tach source.
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

## 11. Host values and buffer materialization

`gpu.buffer(value)` creates a session-owned `ComputeBuffer<T>` and stores a
structured clone of the initial value. It does not immediately know which
compiled buffer layout will consume that value.

The first submitted command that uses the buffer:

1. selects the compiler-emitted buffer codec;
2. validates and packs the host value;
3. requires at least the buffer's minimum binding size;
4. creates and uploads the WebGPU buffer; and
5. fixes that buffer's layout and byte length.

Before materialization, `write(value)` replaces the cloned host value and may
change its eventual size. After materialization, writes must pack to the exact
same byte length. A buffer cannot later be used as a differently laid-out
buffer; create another `ComputeBuffer` instead.

Packing validates numeric shape and range. Integers must fit `int32`/`uint32`;
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

## 12. Commands and launch geometry

A generated kernel call validates its arguments and returns an opaque
`ComputeCommand`. It does not submit. Accidentally awaiting the command throws
a targeted error; the only execution boundary is:

```ts
await gpu.submit(commandA, commandB);
```

Plain arguments are packed together and snapshotted when the command is
constructed. Buffer arguments remain live resident handles. All command
buffers must belong to the same Tach session.

`submit` records its commands in order into one compute pass and one command
buffer, then performs one queue submission. `repeat` repeats that command's
`dispatchWorkgroups` call inside the same pass:

```ts
step(state, params, { size: count, repeat: 100 })
```

Every explicit size component must be a positive safe integer and the value's
rank must exactly match the kernel. Workgroup counts are:

```text
groups[axis] = ceil(logicalSize[axis] / workgroupSize[axis])
```

The final workgroup may therefore execute coordinates outside the logical
extent. Kernel code owns bounds guards.

When `size` is omitted:

- a 1D kernel infers it from the first runtime-sized buffer;
- otherwise the logical extent defaults to exactly one workgroup.

Two- and three-dimensional problem extents cannot be inferred from a flat
buffer and should normally be explicit.

## 13. Sessions and ownership

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
work, returns the callback value, and closes the session. Failures reject with
`TachError`. Every buffer created by that session is destroyed on exit. A
returned `ComputeBuffer` is therefore not usable after the scope.

### Persistent lifetime

```ts
const gpu = await tach();
try {
  // Reuse resident buffers and cached pipelines across many submissions.
  await gpu.idle();
} finally {
  gpu.close();
}
```

`tach(options?)` returns the same session without automatic shutdown. The
caller owns synchronization and `close()`. This form supports frame loops and
long iterative jobs without reacquiring an adapter/device or recreating
resident state.

Both forms use the same `ComputeBuffer`, commands, caches, submission ordering,
and error model. There is no batch-only and frame-only execution split.

## 14. Submission, synchronization, and caches

`submit()` awaits asynchronous pipeline preparation and queue submission, but
does not wait for GPU completion. Session submissions are chained so calls
issued from interleaved promises retain invocation order.

The explicit completion boundaries are:

- `gpu.idle()`;
- `buffer.read()` for all prior session submissions; and
- successful or failed exit from `tach(...)`.

Generated modules cache a shader module and compute pipelines per WebGPU
device. Sessions cache bind groups by layout and storage-buffer range. Parameter
blocks share an aligned session buffer sized against WebGPU's
`minUniformBufferOffsetAlignment`; dynamic offsets select each command's
snapshot, so commands with the same storage buffers reuse one bind group. The
arena grows geometrically and is reused under queue ordering. Destroying a
buffer or growing the parameter arena clears affected bind-group cache entries.

`gpu.close()` is idempotent. It destroys all owned GPU buffers, the parameter
arena, bind-group state, event handling, and the WebGPU device.

## 15. Error contract

Both `tach(...)` overloads and all session operations use ordinary TypeScript
success and failure semantics:

```ts
tach(options?): Promise<Tach>;
tach<T>(work: (gpu: Tach) => T | Promise<T>, options?): Promise<T>;
```

Success returns or resolves to the value. Failure throws or rejects with
`TachError`, which extends `Error` and carries a stable `code`, an optional
operation, and the original cause. Current categories cover WebGPU
availability, adapter/device acquisition and loss, GPU
validation/out-of-memory/internal errors, buffers, kernels, lifecycle, user
callback failures, and compiler delivery/execution. Asynchronous WebGPU errors
are retained by the session and surface at a later submission or
synchronization boundary rather than being silently discarded.

## 16. Native Vulkan caller obligations

The TypeScript runtime enforces the WebGPU side automatically. A native
consumer of the compiler API's SPIR-V and execution plan must perform the equivalent work:

1. create descriptor bindings at set `0` and the recorded module binding
   numbers used by the selected kernel;
2. allocate at least `minimumByteSize`, with every runtime tail ending on its
   recorded stride;
3. pack scalars, vectors, fields, padding, and runtime tails according to the
   Tach layout;
4. create storage-buffer descriptors for buffer parameters and, when present,
   one uniform-buffer descriptor for the compiler-owned parameter block;
5. build that block from each field's parameter index, path, type, and byte
   offset, encoding bool as `0` or `1` in a `uint32` slot;
6. bind distinct memory objects for distinct buffer parameters;
7. dispatch `ceil(logicalExtent / workgroupSize)` groups on every axis; and
8. insert Vulkan synchronization appropriate for work before and after the
   Tach pipeline.

Workgroup-memory initialization and intra-kernel barriers are already encoded
in the SPIR-V module. Bounds outside a rounded-up logical extent remain the
kernel author's responsibility.

## 17. Compatibility rule

The `.tach` file is the source of truth. Files from one selected build form one
compiler result and must travel together; the `all` target is the seven-file
diagnostic superset. Public kernel names, parameter order, byte layout, launch
rank, and runtime lifetime are ABI, but the metadata schema is intentionally
not versioned yet. Recompile rather than hand-editing or combining output.
