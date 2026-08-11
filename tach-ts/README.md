# `@depths/tach`

`@depths/tach` contains the Tach compiler command and the browser/WebGPU
runtime used by generated kernels. Application code gets typed kernel
functions, GPU-resident buffers, explicit submission, and two clear lifetime
choices. It does not manage shader strings or bind groups.

## Five-minute start

Installation and compilation need Node.js 22 or newer. Runtime execution needs
a browser or injected environment with WebGPU.

```sh
npm install @depths/tach
```

Save `kernels/scale.tach`:

```tach
export function scale[i](
  values: buffer<float32[]>,
  factor: float32,
) {
  if (i < values.length) {
    values[i] *= factor;
  }
}
```

Compile it:

```sh
npx tach check kernels/scale.tach
npx tach build kernels/scale.tach
```

Use the generated function like a typed command constructor:

```ts
import { tach } from "@depths/tach";
import { scale } from "./build/scale.js";

const result = await tach(async (gpu) => {
  const values = gpu.buffer(new Float32Array([1, 2, 3, 4]));

  await gpu.submit(scale(values, 2));
  return values.read();
});

if (!result.ok) {
  throw new Error(`${result.error.code}: ${result.error.message}`);
}

console.log(result.value); // Float32Array [2, 4, 6, 8]
```

The generated declaration is equivalent to:

```ts
function scale(
  values: ComputeBuffer<Float32Array | readonly number[]>,
  factor: number,
  $dispatch?: DispatchOptions<number>,
): ComputeDispatch;
```

Source parameter order becomes host parameter order. The final dispatch object
is generated automatically.

## The whole mental model

There are only three application-facing concepts:

```text
gpu.buffer(hostValue)     -> ComputeBuffer
generatedKernel(...)     -> ComputeDispatch
gpu.submit(dispatches)   -> ordered WebGPU work
```

A kernel call does not execute. It builds an opaque command. `submit` is the
execution boundary. A read or `idle()` is the completion boundary.

This separation lets one submission contain several kernels:

```ts
await gpu.submit(
  scale(values, 2),
  scale(values, 0.5),
);
```

They are recorded in argument order into one compute pass and one queue
submission.

## Choose a lifetime

Both APIs create the same `Tach` session. The difference is who closes it.

### A scoped job: `tach(...)`

Use `tach` when one callback owns the whole job:

```ts
const result = await tach(async (gpu) => {
  const data = gpu.buffer(initial);
  await gpu.submit(transform(data, params));
  return data.read();
});
```

It:

1. requests an adapter and device;
2. runs the callback;
3. waits for queued work;
4. returns success or failure as a `Result`; and
5. closes the session and destroys its buffers.

Return host data from the callback. A returned `ComputeBuffer` is already
closed with its session.

### Resident state: `openTach()`

Use `openTach` for a frame loop, simulation, solver, or service:

```ts
import { openTach } from "@depths/tach";
import { step } from "./build/simulation.js";

const opened = await openTach({
  adapter: { powerPreference: "high-performance" },
});
if (!opened.ok) throw new Error(opened.error.message);

const gpu = opened.value;
const state = gpu.buffer(initialState);

try {
  for (let frame = 0; frame < 1_000; frame++) {
    await gpu.submit(step(state, { dt: 1 / 60 }));
    // No readback or idle wait in the hot path.
  }

  await gpu.idle();
} finally {
  gpu.close();
}
```

The adapter, device, shader modules, pipelines, bind groups, parameter arena, and
buffers stay resident. `close()` is immediate and idempotent. If graceful GPU
completion matters, await `idle()` first. A lost device requires a new session
and recreated application state.

There is no separate batch engine and frame engine. Both lifetimes use the
same buffers, commands, caches, ordering, and errors.

## Buffers

```ts
interface ComputeBuffer<T> {
  write(value: T): void;
  read(): Promise<T>;
  destroy(): void;
}
```

`gpu.buffer(value)` stores a structured clone. The first submitted kernel use
supplies the compiler-generated layout, packs the value, creates the WebGPU
buffer, and fixes its byte length.

Before that first use, `write(value)` may change the eventual size. After it,
`write` updates the resident GPU buffer and must preserve byte length. Create a
new buffer to resize.

`read()` waits for earlier session submissions, copies through a temporary
readback buffer, decodes the compiler-owned layout, and returns a clone.
Reading a never-submitted buffer simply returns a clone of its host value.

`destroy()` is idempotent. A destroyed buffer, a closed-session buffer, and a
buffer passed to another session are lifecycle errors.

Different buffer parameters of one command must receive different
`ComputeBuffer` objects. If a kernel updates data in place, give it one buffer
parameter.

## Commands, submission, and synchronization

A generated kernel call validates its arguments, snapshots plain values, and
returns a `ComputeDispatch`:

```ts
const command = scale(values, 2);
await gpu.submit(command);
```

Buffer arguments remain live handles. Plain arguments are frozen at command
construction. Accidentally writing `await scale(...)` throws a targeted error
instead of silently doing nothing.

`submit(first, ...rest)` waits for asynchronous pipeline preparation and host
queue submission. It does not wait for GPU completion. Concurrent JavaScript
callers still retain `submit` call order through the session.

Wait only where the CPU actually needs a result:

- `await gpu.idle()` waits for all earlier session work;
- `await buffer.read()` waits and reads back; and
- `tach(...)` waits before returning and closing.

## Logical size and repeated dispatches

Generated kernels accept an optional final object:

```ts
interface DispatchOptions<Size extends DispatchSize = DispatchSize> {
  readonly size?: Size;
  readonly dispatches?: number;
}
```

`size` matches the coordinate rank:

| Tach declaration | TypeScript `size` |
|---|---|
| `kernel[i]` | `number` |
| `kernel[x, y]` | `readonly [number, number]` |
| `kernel[x, y, z]` | `readonly [number, number, number]` |

For example:

```ts
await gpu.submit(render(pixels, params, { size: [1920, 1080] }));
```

Each component must be a positive safe integer. Tach divides by the compiled
workgroup size and rounds up, so the kernel must guard its edge coordinates.

A one-dimensional kernel can omit `size` when its first runtime-sized storage
buffer gives the obvious length. Otherwise omission means exactly one
workgroup. Two- and three-dimensional shapes normally need an explicit size.

`dispatches` repeats one prepared command inside the same compute pass:

```ts
await gpu.submit(step(state, params, {
  size: particleCount,
  dispatches: 100,
}));
```

Every repetition uses the same buffers, parameter snapshot, and launch size. Use
separate commands when values differ. Repetition is ordered dispatch, not
compiler kernel fusion.

## Host data shapes

Generated declarations perform the mapping for you:

| Tach value | TypeScript value |
|---|---|
| `int32`, `uint32`, `float32` | `number` |
| `bool` | `boolean` |
| storage atomic | `number` |
| numeric vector | readonly numeric tuple |
| named struct | generated readonly object type |
| scalar runtime array | matching typed array or readonly number array |
| two-/four-lane runtime vector array | flat typed array or tuple array |
| three-lane runtime vector array | tuple array |

Three-lane GPU vectors have padded element stride, so they use tuple arrays.
Scalar, two-lane, and four-lane runtime arrays are tightly packed. On a
little-endian host, matching `Float32Array`, `Int32Array`, and `Uint32Array`
values cross that boundary without element-by-element packing, and readback
preserves their representation.

For object/array values, compiler-emitted codecs validate integer ranges,
vector lane counts, fields, runtime element completeness, offsets, and strides.
Application code never writes a padding field or calls a public packer.

## Performance rules that matter

The first use is cold. It may allocate and upload a buffer, create a shader
module, compile a pipeline, create a bind group, and grow the parameter arena.

After warm-up, generated modules cache shader modules and pipelines per device;
the session keeps buffers resident, reuses stable bind groups, and shares an
aligned compiler-owned parameter buffer.

For a hot loop:

1. open one persistent session;
2. create long-lived buffers once;
3. warm up before measuring;
4. batch commands at real dependency boundaries; and
5. avoid `read()` and `idle()` inside the loop unless the CPU needs completion.

The bind-group cache currently retains live combinations for the session. If a
profile later proves that high-churn applications need bounded eviction, that
policy belongs in the runtime rather than application code.

## Options and the public session

Both lifetime APIs accept:

```ts
interface TachOptions {
  readonly gpu?: GPU;
  readonly adapter?: GPURequestAdapterOptions;
  readonly device?: GPUDeviceDescriptor;
}
```

`gpu` overrides `navigator.gpu`, mainly for controlled environments and tests.
The other objects pass through to `requestAdapter` and `requestDevice`.

The opened session is:

```ts
interface Tach {
  readonly adapter: GPUAdapter;
  readonly device: GPUDevice;
  buffer<T>(value: T): ComputeBuffer<T>;
  submit(first: ComputeDispatch, ...rest: readonly ComputeDispatch[]): Promise<void>;
  idle(): Promise<void>;
  close(): void;
}
```

`ComputeBuffer` intentionally hides its `GPUBuffer`; Tach owns its layout and
lifetime.

## Results and failures

Opening and scoped work use error-as-data:

```ts
type Result<T, E = TachError> =
  | { readonly ok: true; readonly value: T }
  | { readonly ok: false; readonly error: E };

interface TachError {
  readonly code: TachErrorCode;
  readonly message: string;
  readonly operation?: string;
  readonly cause?: unknown;
}
```

Categories distinguish WebGPU availability, adapter/device acquisition and
loss, GPU validation/out-of-memory/internal errors, buffers, kernels,
lifecycle misuse, callback failures, and compiler setup/execution.

Operations on a persistent session throw or reject on failure, so use
`try`/`finally`. Inside `tach(...)`, failures and callback exceptions become
the returned `Result`.

The runtime uses WebGPU error scopes and retains asynchronous errors. Device
loss, uncaptured errors, and scoped errors surface at a later submission or
synchronization boundary instead of disappearing.

## Compiler command

The package exposes the native compiler as `tach`:

```text
tach build FILE.tach
tach check FILE.tach
tach ir FILE.tach
tach wgsl FILE.tach
tach spirv-dis FILE.tach
tach version
```

`build` writes `.tir`, `.wgsl`, `.spv`, `.spvasm`, `.js`, `.d.ts`, and
`.tach.json` to `build/`. Rebuild them together from `.tach` source.

Published packages support Linux, macOS, and Windows on x64 and arm64. The
installer selects the release asset for the package version and verifies its
SHA-256 checksum. Compiler resolution checks, in order:

1. `TACH_BIN`;
2. an already installed package compiler;
3. `dist/tach` or `dist/tach.exe` in a containing development checkout; and
4. the verified release asset.

An invalid `TACH_BIN` is an error; it does not silently fall through.

## Node compiler API

Node-only tools may import `@depths/tach/compiler`:

```ts
import { build, compilerPath, runCompiler } from "@depths/tach/compiler";

const compiler = await compilerPath();
if (!compiler.ok) throw new Error(compiler.error.message);

const checked = await runCompiler(["check", "kernels/scale.tach"]);
if (!checked.ok) throw new Error(checked.error.message);

const built = await build("kernels/scale.tach", { cwd: process.cwd() });
if (!built.ok) throw new Error(built.error.message);
```

These APIs return `Result` values. They capture output and the resolved binary.
`cwd` selects the child directory and `env` overlays environment variables.
Do not import this entry point into a browser bundle.

## Generated code boundary

Generated `.js` imports `defineModule` from `@depths/tach/internal` and embeds
WGSL plus compiler-owned descriptors. That subpath is for generated code, not
applications. Application imports belong at `@depths/tach`.

The generated JavaScript, declarations, metadata, WGSL, SPIR-V, and IR are one
validated compiler result. They are not designed to be edited or mixed across
compiler versions.

## Repository development

From the Tach repository root:

```sh
npm ci
npm run compiler
npm run check --workspace=@depths/tach
npm test --workspace=@depths/tach
```

The workspace tests cover compiler discovery, packing, typed arrays,
ownership, non-aliasing, batching, size inference, caching, error scopes,
synchronization, and cleanup with a controlled WebGPU implementation. The
`browser-test` and `showcase-ts` workspaces exercise real Chromium WebGPU.

Further reading:

- [Project overview](../README.md)
- [Language guide](../docs/language.md)
- [Architecture](../docs/architecture.md)
- [Resource and runtime ABI](../docs/abi.md)

`@depths/tach` is licensed under `AGPL-3.0-only`.
