# `@depths/tach`

`@depths/tach` is Tach's browser/WebGPU runtime and native compiler
distribution. It gives an application three things:

- the `tach` and `openTach` execution APIs;
- typed host bindings generated directly from `.tach` kernels; and
- the native `tach` compiler as an npm executable and Node API.

Tach kernels remain provider-neutral. The generated module embeds WGSL and the
compiler-owned resource/layout contract; this package owns adapter, device,
buffer, pipeline, submission, readback, and error behavior.

## Requirements and installation

The package requires Node.js 22 or newer for installation and compilation. The
runtime executes in a browser or injected environment that provides WebGPU.

```sh
npm install @depths/tach
npx tach version
```

Published packages support Linux, macOS, and Windows on x64 and arm64. During
installation, Tach selects the matching native compiler, downloads it from the
package version's GitHub release, and verifies it against that release's
SHA-256 manifest.

## First kernel, end to end

Save this as `kernels/scale.tach`:

```tach
export compute scale[i](
  data: buffer<f32[]>,
  factor: uniform<f32>,
) {
  if (i < data.length) {
    data[i] *= factor;
  }
}
```

Compile it from the application root:

```sh
npx tach check kernels/scale.tach
npx tach build kernels/scale.tach
```

`check` runs the complete compiler and validation pipeline without writing
files. `build` writes these synchronized artifacts to `build/`:

```text
scale.tir
scale.wgsl
scale.spv
scale.spvasm
scale.js
scale.d.ts
scale.tach.json
```

Import the generated JavaScript module as ordinary application code:

```ts
import { tach } from "@depths/tach";
import { scale } from "./build/scale.js";

const initial = new Float32Array([1, 2, 3, 4]);

const result = await tach(async (gpu) => {
  const data = gpu.buffer(initial);
  await gpu.submit(scale(data, 2));
  return data.read();
});

if (!result.ok) {
  throw new Error(`${result.error.code}: ${result.error.message}`);
}

console.log(result.value); // Float32Array [2, 4, 6, 8]
```

The generated `scale` signature is equivalent to:

```ts
function scale(
  data: ComputeBuffer<Float32Array | readonly number[]>,
  factor: number,
  $dispatch?: DispatchOptions<number>,
): ComputeDispatch;
```

There is no shader-module factory or resource map in application code. Source
parameters remain positional, named structs become TypeScript interfaces, and
exported kernel names remain exported function names.

## The execution model

Tach's host model has three explicit objects and one explicit boundary:

```text
host value -> ComputeBuffer
kernel call -> ComputeDispatch
ComputeDispatch -> gpu.submit(...) -> WebGPU queue
```

### Buffers are session-owned handles

`gpu.buffer(value)` creates a `ComputeBuffer<T>` owned by that `Tach` session:

```ts
interface ComputeBuffer<T> {
  write(value: T): void;
  read(): Promise<T>;
  destroy(): void;
}
```

The runtime takes a structured clone, so later mutation of the original host
object does not mutate the buffer. Before the buffer reaches a submitted
kernel, it remains a host value. Its first submitted use supplies the
compiler-generated codec and layout, packs the bytes, creates the WebGPU
buffer, and fixes its physical byte length.

After materialization, `write(value)` uploads new bytes without replacing the
GPU buffer. The packed length must remain identical; resize by creating a new
`ComputeBuffer`. `read()` waits for prior session submissions, copies through a
temporary readback buffer, decodes the value, and returns another clone.
`destroy()` is idempotent, and any later use is a lifecycle error.

### Kernel calls create commands

A generated kernel call validates its parameters, snapshots its uniform bytes,
and returns an opaque `ComputeDispatch`:

```ts
const command = scale(data, 2);
```

It does not submit or wait. Awaiting a command directly fails loudly with an
instruction to pass it to `Tach.submit`; a forgotten command cannot silently
become a no-op.

Buffer arguments are live handles, whereas uniform arguments are snapshots at
command construction. Different buffer parameters of one command must be
different `ComputeBuffer` objects. To mutate in place, use one buffer parameter
in the Tach kernel.

### `submit` is the execution boundary

```ts
await gpu.submit(commandA, commandB, commandC);
```

One call prepares every command, encodes them in argument order into one
compute pass and one command buffer, and performs one WebGPU queue submission.
Commands must belong to the session receiving them.

Awaiting `submit()` waits for asynchronous shader/pipeline preparation and
host-side queue submission. It does **not** wait for the GPU queue to become
idle. Calls are chained internally so submissions made by interleaved promises
retain call order.

Use a synchronization boundary only when the CPU needs completion:

- `await gpu.idle()` waits for all prior session work;
- `await buffer.read()` orders and waits for a readback; and
- `tach(...)` waits automatically before closing its scope.

## Two lifetimes, one session implementation

The choice between `tach` and `openTach` is about ownership, not execution
semantics. Both forms use the same buffers, commands, queue ordering, caches,
errors, and cleanup rules.

### Scoped work with `tach`

```ts
function tach<T>(
  work: (gpu: Tach) => T | Promise<T>,
  options?: TachOptions,
): Promise<Result<T>>;
```

`tach(...)` acquires an adapter and device, invokes the callback, waits for
queued work, converts success or failure to a discriminated `Result`, and
closes the session in every path. It is the compact form for a complete batch:

```ts
const result = await tach(async (gpu) => {
  const data = gpu.buffer(initial);

  await gpu.submit(
    scale(data, 2),
    scale(data, 2),
  );

  return data.read();
});
```

Every buffer created inside the callback becomes unusable when the scope ends.
Return host data, not a `ComputeBuffer`, when the caller needs the result.

### Resident work with `openTach`

```ts
function openTach(options?: TachOptions): Promise<Result<Tach>>;
```

`openTach()` returns a caller-owned session. Adapter, device, resident buffers,
compiled shader modules, pipelines, stable bind groups, and the uniform upload
arena survive across submissions:

```ts
import { openTach } from "@depths/tach";
import { scale } from "./build/scale.js";

const opened = await openTach({
  adapter: { powerPreference: "high-performance" },
});
if (!opened.ok) throw new Error(opened.error.message);

const gpu = opened.value;
const data = gpu.buffer(initial);

try {
  for (let frame = 0; frame < 1_000; frame++) {
    await gpu.submit(scale(data, 1.0001));
    // submit does not insert a queue-idle stall.
  }

  const final = await data.read(); // The read is the completion boundary.
  console.log(final);
} finally {
  gpu.close();
}
```

This is the form for animation, simulation, iterative solvers, and services
that should not reacquire a device or rebuild resident state per iteration.
If no final readback is needed, call `await gpu.idle()` before graceful
shutdown. `close()` itself is immediate and idempotent: it destroys every
owned buffer, the uniform arena, cached binding state, and the WebGPU device.

A lost device invalidates resident resources. Recovery opens a new session and
recreates application state.

## Logical size and repeated dispatch

The optional final `DispatchOptions` object controls logical extent and command
repetition:

```ts
interface DispatchOptions<Size extends DispatchSize = DispatchSize> {
  readonly size?: Size;
  readonly dispatches?: number;
}
```

`size` must match the kernel's declared coordinate rank:

| Tach kernel | Generated `size` type |
|---|---|
| `kernel[i]` | `number` |
| `kernel[x, y]` | `readonly [number, number]` |
| `kernel[x, y, z]` | `readonly [number, number, number]` |

Each component must be a positive safe integer. The runtime divides each
logical component by the compiler-recorded workgroup component and rounds up.
The kernel must guard coordinates in the partially filled edge workgroup.

For a one-dimensional kernel, omitted `size` is inferred from the first
runtime-sized storage buffer. If that is unavailable, the default is exactly
one workgroup. Two- and three-dimensional shapes are not inferred from flat
storage; their omitted default is also one workgroup.

Use an explicit extent when it is clearer or when the buffer contains more
data than one invocation consumes:

```ts
await gpu.submit(scale(data, 2, { size: elementCount }));
```

`dispatches` must be a positive safe integer. It repeats the same prepared
command—with the same buffers, uniform snapshot, and workgroup counts—inside
the containing compute pass:

```ts
await gpu.submit(step(state, params, {
  size: particleCount,
  dispatches: 128,
}));
```

This amortizes command construction and queue submission for iterative kernels.
Use separate commands when an iteration needs different uniform values.

## Host data mapping

Generated declarations map Tach types directly:

| Tach | TypeScript host shape |
|---|---|
| `i32`, `u32`, `f32` | `number` |
| `atomic<i32>`, `atomic<u32>` in storage | `number` |
| numeric vectors such as `f32x4` | readonly numeric tuple |
| named struct | generated readonly interface |
| scalar runtime array | matching typed array or readonly number array |
| two-/four-lane runtime vector array | flat typed array or tuple array |
| three-lane runtime vector array | tuple array |

Three-lane vectors use tuple arrays because their GPU element stride contains
padding. Scalar, two-lane, and four-lane runtime arrays are tightly packed; on
little-endian hosts the matching `Float32Array`, `Int32Array`, or `Uint32Array`
can cross the boundary without element-wise packing, and `read()` preserves
that representation.

Ordinary arrays and objects are packed with compiler-provided byte offsets and
strides. The runtime checks integer ranges, vector lane counts, required
struct fields, complete runtime elements, and minimum resource size. Runtime
arrays must contain at least one element under the current ABI.

Packing is deliberately private. Application code works with generated types
and `ComputeBuffer`; it does not reproduce shader layout or call a public
packer.

## Performance model

The first submitted use pays unavoidable cold costs:

- buffer layout selection, packing, allocation, and upload;
- shader-module creation; and
- asynchronous compute-pipeline creation.

After warm-up:

- each generated module caches its shader module and pipelines per device;
- each session keeps buffers resident;
- stable bind groups are cached by layout and buffer ranges;
- uniforms share one aligned session buffer that grows as needed; and
- several commands or repeated dispatches can share one pass and submission.

For a hot loop, open one persistent session, create long-lived buffers once,
avoid readback/`idle()` in the hot path, and batch commands at the actual
application dependency boundary. Measure steady-state work separately from
first-use compilation and allocation.

The bind-group cache currently retains live binding combinations for the
session lifetime. Applications with continuously changing combinations should
destroy obsolete buffers; a bounded eviction policy is a future upgrade if
profiles justify it.

## WebGPU selection options

Both lifetime APIs accept:

```ts
interface TachOptions {
  readonly gpu?: GPU;
  readonly adapter?: GPURequestAdapterOptions;
  readonly device?: GPUDeviceDescriptor;
}
```

- `gpu` overrides `navigator.gpu`, which is useful for controlled environments
  and tests;
- `adapter` is forwarded to `requestAdapter`; and
- `device` is forwarded to `requestDevice`.

The resulting `Tach` exposes its selected `adapter` and `device`, plus:

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

`ComputeBuffer` intentionally does not expose its underlying `GPUBuffer`; the
compiler/runtime retains ownership of its layout and lifetime.

## Results and failures

Adapter/device acquisition and scoped execution use error-as-data:

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

Current error categories distinguish WebGPU availability, adapter and device
acquisition, device loss, GPU validation/out-of-memory/internal failures,
buffer encoding/readback, kernel preparation/submission, lifecycle misuse,
user callback failures, and native compiler setup/execution.

Methods on a caller-owned persistent session throw or reject on operational
failure, so use `try`/`finally` and always close. Inside `tach(...)`, those
failures and callback exceptions are normalized into the returned `Result`.

The session wraps synchronous WebGPU calls in validation, out-of-memory, and
internal error scopes. Errors reported asynchronously are retained and surface
at a later submission or explicit synchronization boundary. Uncaptured errors
and device loss also invalidate later operations instead of being ignored.

## Generated-module boundary

Generated `.js` files import `defineModule` from `@depths/tach/internal` and
embed compiler-owned WGSL, resource descriptors, layouts, and kernel records.
That subpath exists for generated code; applications should not import it.

The public application entry point is `@depths/tach`. Generated modules export
only source-named kernels, while their `.d.ts` files import the public
`ComputeBuffer`, `ComputeDispatch`, and `DispatchOptions` types.

Rebuild `.js`, `.d.ts`, `.tach.json`, WGSL, SPIR-V, and IR together. They form
one validated compiler result and are not designed to be mixed across compiler
versions or edited by hand.

## Native compiler CLI

npm exposes the package compiler as `tach`:

```text
tach build FILE.tach
tach check FILE.tach
tach ir FILE.tach
tach wgsl FILE.tach
tach spirv-dis FILE.tach
tach version
```

A package script avoids a global installation:

```json
{
  "scripts": {
    "kernels": "tach build kernels/scale.tach"
  }
}
```

Compiler resolution uses the first available source:

1. executable path from `TACH_BIN`;
2. a compiler already installed into the package;
3. `dist/tach` or `dist/tach.exe` from a containing Tach development checkout;
4. the verified native asset for the installed package version.

`TACH_BIN` may be absolute or relative to the current working directory. An
invalid override is returned as a `compiler-install` error; it does not fall
through silently.

## Node compiler API

The Node-only `@depths/tach/compiler` entry point exposes compiler discovery and
execution without requiring application code to manage child processes:

```ts
import {
  build,
  compilerPath,
  runCompiler,
} from "@depths/tach/compiler";

const resolved = await compilerPath();
if (!resolved.ok) throw new Error(resolved.error.message);

const checked = await runCompiler(["check", "kernels/scale.tach"]);
if (!checked.ok) throw new Error(checked.error.message);

const built = await build("kernels/scale.tach", { cwd: process.cwd() });
if (built.ok) {
  console.log(built.value.path);
  console.log(built.value.stdout);
}
```

`runCompiler(args, options?)` captures `stdout`, `stderr`, and the resolved
executable path. `options.cwd` selects the child working directory;
`options.env` overlays environment variables. Nonzero exits, signals, spawn
failures, and setup failures return `Result` errors rather than rejecting the
API promise.

Do not import this Node entry point into browser bundles.

## Development in the Tach repository

From the repository root:

```sh
npm ci
npm run compiler
npm run check --workspace=@depths/tach
npm test --workspace=@depths/tach
```

The repository package version is `0.0.0`; its installer therefore uses the
locally built `dist/tach` compiler and warns, rather than attempting a release
download, when that binary is missing.

The workspace tests cover compiler resolution/execution and the runtime's
ownership, cloning, packing, typed-array preservation, batching, inference,
non-aliasing, cache reuse, error scopes, synchronization, and cleanup with a
controlled WebGPU implementation. Real-browser execution is covered by the
repository's `browser-test` and `showcase-ts` workspaces.

## Further reference

- [Tach project overview](https://github.com/Depths-AI/tach)
- [Language reference](https://github.com/Depths-AI/tach/blob/main/docs/language.md)
- [Compiler and runtime architecture](https://github.com/Depths-AI/tach/blob/main/docs/architecture.md)
- [Resource, host, and runtime ABI](https://github.com/Depths-AI/tach/blob/main/docs/abi.md)

`@depths/tach` is licensed under `AGPL-3.0-only`.
