# `@depths/tach`

`@depths/tach` delivers the native Tach compiler and the managed WebGPU
runtime used by generated modules. Applications work with typed public
programs, GPU-resident buffers, opaque commands, and explicit submission—never
shader strings, layouts, bind groups, or descriptor numbers.

## Five-minute start

Compilation requires Node.js 22 or newer. Execution requires a browser or
injected environment with WebGPU.

```sh
npm install @depths/tach
```

Save `kernels/scale.tach`:

```tach
export function scale[i](values: buffer<float32[]>, factor: float32) {
  if (i < values.length) {
    values[i] *= factor;
  }
}
```

Compile and validate it:

```sh
npx tach check kernels/scale.tach
npx tach build kernels/scale.tach
```

Use the generated public program as a typed command constructor:

```ts
import { tach } from "@depths/tach";
import { scale } from "./build/scale.js";

const result = await tach(async (gpu) => {
  const values = gpu.buffer(new Float32Array([1, 2, 3, 4]));
  await gpu.submit(scale(values, 2));
  return values.read();
});

console.log(result); // Float32Array [2, 4, 6, 8]
```

The generated declaration is equivalent to:

```ts
function scale(
  values: ComputeBuffer<Float32Array | readonly number[]>,
  factor: number,
  $launch?: LaunchOptions<number>,
): ComputeCommand;
```

Source parameter order becomes host parameter order. The final options
parameter is compiler-generated.

## Mental model

There are three application-facing operations:

```text
gpu.buffer(hostValue)       -> ComputeBuffer
generatedProgram(...)      -> ComputeCommand
gpu.submit(commands...)    -> ordered WebGPU work
```

A generated call does not execute. `submit` is the execution boundary; a
buffer `read()` or `gpu.idle()` is a completion boundary.

One public program may contain one stage or an ordered multi-stage plan. The
host call is identical either way:

```tach
function first[i](input: buffer<float32[]>, scratch: buffer<float32[]>) {
  scratch[i] = input[i] * 2;
}

function second[i](scratch: buffer<float32[]>, output: buffer<float32[]>) {
  output[i] = scratch[i] + 1;
}

export function transform(
  input: buffer<float32[]>,
  output: buffer<float32[]>,
  count: uint32,
) {
  const scratch = transient<float32>(count);
  run first(input, scratch) over count;
  run second(scratch, output) over count;
}
```

```ts
await gpu.submit(transform(input, output, count));
```

The generated module embeds the physical entries, dispatch order, scratch
requirements, shapes, bindings, and parameter sources. Distinct source stages
remain distinct dispatches; the runtime does not infer or perform fusion.

## Choose a lifetime

Both `tach` overloads open the same session. The difference is ownership.

### Scoped job

```ts
const result = await tach(async (gpu) => {
  const data = gpu.buffer(initial);
  await gpu.submit(transform(data, params));
  return data.read();
});
```

`tach(work, options?)`:

1. requests an adapter and device;
2. runs `work`;
3. waits for all queued GPU work;
4. returns the callback value or rejects with `TachError`; and
5. closes the session and destroys owned resources.

Return host data, not a `ComputeBuffer`; the latter closes with its session.

### Persistent session

Use `tach(options?)` for a frame loop, simulation, solver, or service:

```ts
import { tach } from "@depths/tach";
import { step } from "./build/simulation.js";

const gpu = await tach({
  adapter: { powerPreference: "high-performance" },
});
const state = gpu.buffer(initialState);

try {
  for (let frame = 0; frame < 1_000; frame++) {
    await gpu.submit(step(state, { dt: 1 / 60 }));
  }
  await gpu.idle();
} finally {
  gpu.close();
}
```

The adapter, device, resident buffers, shader modules, pipelines, bind groups,
parameter arena, and transient scratch remain reusable. `close()` is immediate
and idempotent; await `idle()` first when graceful completion matters. Device
loss requires a new session and recreated application state.

## Buffers

```ts
interface ComputeBuffer<T> {
  write(value: T): void;
  read(): Promise<T>;
  destroy(): void;
}
```

`gpu.buffer(value)` stores a structured clone but does not allocate GPU memory
or choose a layout. The first submitted use supplies the compiler codec,
validates and packs the value, creates the WebGPU buffer, and fixes its byte
length and layout.

Before materialization, `write(value)` may change the future size. Afterward,
it writes the resident GPU buffer and must preserve byte length. Create a new
handle to resize or use a different Tach layout.

`read()` waits for earlier session submissions, copies materialized storage to
a temporary map-read buffer, decodes it, and returns a clone. Reading a never-
submitted handle returns a clone of its host value without creating GPU
storage.

`destroy()` is idempotent. A destroyed buffer, a closed-session buffer, or a
buffer passed to another session is a lifecycle error.

Different public buffer parameters of one command require different
`ComputeBuffer` objects. Use one parameter for in-place work.

## Commands and submission

A generated call validates handles, ownership, non-aliasing, options, and
immediately evaluable parameter blocks, then returns an opaque command:

```ts
const command = scale(values, 2);
await gpu.submit(command);
```

Buffer arguments remain live handles. Plain primitive arguments are naturally
stable; object/array value arguments are retained until command preparation.
Do not mutate them between construction and submission.

Accidentally writing `await scale(...)` throws a targeted error instead of
silently doing nothing. `submit` accepts only generated commands owned by that
session.

```ts
await gpu.submit(
  scale(values, 2),
  scale(values, 0.5),
);
```

Commands are prepared concurrently, then recorded in argument order into one
compute pass and command buffer and submitted once. Concurrent JavaScript
calls to `submit` are serialized through the session, preserving call order.

The returned promise covers preparation and queue submission, not device
completion. Wait only when the CPU needs the result:

- `await gpu.idle()` waits for all earlier session work;
- `await buffer.read()` waits and performs readback; and
- `tach(callback)` waits before closing.

## Launch and command options

Every command supports repeat:

```ts
interface CommandOptions {
  readonly repeat?: number;
}
```

`repeat` is a positive integer within `uint32`, defaulting to `1`. A multi-stage
program repeats its complete plan. A safe one-dispatch 1D program may be
compiled as one invocation-local loop instead of repeated dispatches when all
accesses are independent at the exact current coordinate and there are no
loops or synchronization effects. The generated plan records which behavior
the runtime must use.

Only exported indexed shorthand accepts a logical size:

```ts
type LaunchSize =
  | number
  | readonly [x: number, y: number]
  | readonly [x: number, y: number, z: number];

interface LaunchOptions<Size extends LaunchSize = LaunchSize>
  extends CommandOptions {
  readonly size?: Size;
}
```

| Tach coordinates | TypeScript `size` |
|---|---|
| `[i]` | `number` |
| `[x, y]` | `readonly [number, number]` |
| `[x, y, z]` | `readonly [number, number, number]` |

Every component is a positive safe integer and rank must match exactly. Tach
rounds workgroups up; source guards edge coordinates.

A 1D shorthand can omit `size` when its first runtime-sized public resource
provides a length. Otherwise omission means one workgroup. Explicit public
programs derive all domains in Tach source and therefore accept
`CommandOptions`, not `LaunchOptions`.

## Host data shapes

Generated declarations encode host representation:

| Tach value | TypeScript value |
|---|---|
| `int32`, `uint32`, `float32` | `number` |
| `bool` | `boolean` |
| storage atomic | `number` |
| numeric vector | readonly numeric tuple |
| named struct | generated readonly object type |
| scalar runtime array | matching typed array or readonly array |
| two-/four-lane runtime vector array | flat typed array or tuple array |
| three-lane runtime vector array | tuple array |

Three-lane vectors have a padded stride, so a flat typed array would describe
the wrong element spacing. Scalar, two-lane, and four-lane runtime arrays are
tightly packed. On little-endian hosts, matching `Float32Array`, `Int32Array`,
and `Uint32Array` values can cross without per-element packing, and readback
preserves that representation.

Generated codecs validate integer ranges, lane/array counts, struct fields,
runtime-tail completeness, offsets, and strides. Applications never supply
padding fields or call a public packer.

## Residency and performance

First use may upload storage, create a shader module, compile every referenced
pipeline, create bind groups, allocate transient scratch, and grow the
parameter arena. Warm execution reuses them.

Generated modules cache shader modules and physical pipelines per device. A
session caches resident buffers, bind groups, a shared aligned uniform arena,
and scratch buffers by compiler allocation color. Dynamic uniform offsets let
commands with identical storage bindings reuse bind groups even when values
differ.

For a hot loop:

1. use one persistent session;
2. create long-lived buffers once;
3. warm the complete program before measuring;
4. submit related commands together; and
5. avoid `read()` and `idle()` until the CPU needs completion.

Scratch and parameter arenas grow geometrically. Replaced GPU buffers remain
retired until `idle()` so queued work cannot observe premature destruction.
The bind-group cache currently retains live combinations for the session; a
bounded policy should be added only if real high-churn workloads justify it.

## Public API

```ts
interface TachOptions {
  readonly gpu?: GPU;
  readonly adapter?: GPURequestAdapterOptions;
  readonly device?: GPUDeviceDescriptor;
}

interface Tach {
  readonly adapter: GPUAdapter;
  readonly device: GPUDevice;
  buffer<T>(value: T): ComputeBuffer<T>;
  submit(first: ComputeCommand, ...rest: readonly ComputeCommand[]): Promise<void>;
  idle(): Promise<void>;
  close(): void;
}
```

`gpu` overrides `navigator.gpu`, primarily for controlled environments and
tests. Adapter and device objects pass to the corresponding WebGPU requests.
`ComputeBuffer` intentionally hides its `GPUBuffer`; layout and lifetime are
runtime-owned.

The package root exports only `tach`, `TachError`, and their public types.

## Failures

Asynchronous APIs return values on success and reject with `TachError` on
failure:

```ts
type TachErrorCode =
  | "webgpu-unavailable"
  | "adapter-unavailable"
  | "device-request-failed"
  | "device-lost"
  | "gpu-validation"
  | "gpu-out-of-memory"
  | "gpu-internal"
  | "buffer"
  | "kernel"
  | "lifecycle"
  | "user"
  | "compiler-platform"
  | "compiler-install"
  | "compiler-execution";

class TachError extends Error {
  readonly code: TachErrorCode;
  readonly operation: string | undefined;
}
```

Codes cover WebGPU availability, adapter/device acquisition and loss, GPU
validation/out-of-memory/internal errors, buffers, programs, lifecycle misuse,
user callbacks, and compiler delivery/execution. Original causes are retained.

Use `try`/`finally` around persistent sessions. Scoped callback exceptions are
normalized with code `"user"`. Error scopes, uncaptured errors, and device
loss are retained and surface at submission or synchronization boundaries.

## Compiler command

The package exposes the native compiler as `tach`:

```text
tach build [--target web|spirv|all] FILE.tach
tach check [--target web|spirv|all] FILE.tach
tach docs FILE.tach
tach ir FILE.tach
tach wgsl FILE.tach
tach spirv-dis FILE.tach
tach version
```

`build` and `check` default to `web`.

| Target | Written artifacts |
|---|---|
| `web` | `.wgsl`, `.js`, `.d.ts` |
| `spirv` | `.spv` |
| `all` | those four plus `.tir`, `.spvasm` |

Rebuilding the same module removes stale Tach siblings outside the new target.
`ir`, `wgsl`, and `spirv-dis` run the complete cross-target compilation before
printing diagnostics.

`tach docs` writes Markdown to standard output. Structured source docs also
become `.d.ts` JSDoc. The Markdown TypeScript sample is rendered by this
package from a target-neutral compiler description; TypeScript syntax does not
leak into the Go front end.

Published packages support Linux, macOS, and Windows on x64 and arm64. Native
compiler resolution is:

1. the explicit `TACH_BIN` path;
2. an installed package compiler;
3. `dist/tach` or `dist/tach.exe` in a development checkout; then
4. a release asset matching package version, OS, and architecture.

Release downloads use SHA-256 from `checksums.txt`, three bounded attempts,
and atomic placement. An invalid `TACH_BIN` is an error, not a fallback.

## Node compiler API

Node-only tools may import `@depths/tach/compiler`:

```ts
import { build, compilerPath, runCompiler } from "@depths/tach/compiler";

const compiler = await compilerPath();
const checked = await runCompiler(["check", "kernels/scale.tach"]);
const built = await build("kernels/scale.tach", { cwd: process.cwd() });
const spirv = await build("kernels/scale.tach", {
  cwd: process.cwd(),
  target: "spirv",
});
```

`compilerPath()` resolves to the selected executable. `runCompiler()` and
`build()` resolve with compiler path, captured stdout, and captured stderr;
nonzero exit rejects with `TachError`. Options can set `cwd`, overlay `env`,
and select `target: "web" | "spirv" | "all"` for `build`. Do not import this
entry into a browser bundle.

## Generated and internal boundaries

Generated `.js` imports `defineModule` from `@depths/tach/internal` and embeds
WGSL plus schema-1 public and target-plan data. That subpath is implementation
API for code generated by the matching compiler, not application API.

Generated files are one validated set. Do not edit them or mix compiler
versions.

## Repository development

From the repository root:

```sh
npm ci
npm run compiler
npm run check --workspace=@depths/tach
npm test --workspace=@depths/tach
```

Unit tests cover compiler discovery, checksum/install failures, host packing,
typed arrays, ownership, non-aliasing, multi-step plans, transient scratch,
shape evaluation, repeat modes, batching, size inference, caches, error
scopes, synchronization, and cleanup with a controlled WebGPU implementation.
`browser-test` and `showcase-ts` exercise real Chromium WebGPU.

Further reading:

- [Project overview](../README.md)
- [Language](../docs/language.md)
- [Architecture](../docs/architecture.md)
- [ABI](../docs/abi.md)

`@depths/tach` is licensed under `AGPL-3.0-only`.
