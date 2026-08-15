# `@depths/tach`

`@depths/tach` is Tach's complete Deno-first npm delivery: the public GPU
runtime, project command, native compiler resolver, AI-agent instructions, and
Tach-owned Vulkan 1.3 host. Generated projects use one JavaScript and
TypeScript interface in browsers and Deno. Applications never select a
backend, load shader text, or construct WebGPU/Vulkan resources themselves.

```ts
import { tach } from "@depths/tach";
import { scale } from "./build/index.js";

const result = await tach(async (gpu) => {
  const values = gpu.buffer(new Float32Array([1, 2, 3, 4]));
  await gpu.submit(scale(values, 2));
  return values.read();
});
```

In a browser this code uses WebGPU and `build/kernel.wgsl`. In Deno it uses
Tach's native Vulkan runtime and `build/kernel.spv`. `tach`, `scale`, buffer
shapes, command construction, synchronization, errors, and declarations are
identical.

## Install and build

Install the npm package in a Deno application:

```sh
npm install @depths/tach
```

Create `tach.json` at the Tach project root:

```json
{
  "name": "scaling",
  "version": "0.1.0",
  "javascript": {
    "package": "@example/scaling"
  },
  "docs": {
    "title": "Scaling kernels",
    "summary": "Typed GPU scaling for the application."
  }
}
```

A Tach project is not an npm project. The manifest declares Tach identity,
one version, the npm identity of its generated JavaScript facade, and project
documentation. The compiler discovers source from immediate module
directories; the manifest never lists modules or files.

Save this as `kernels/scale.tach`:

```tach
export function scale[i](values: buffer<float32[]>, factor: float32) {
  if (i < values.length) {
    values[i] *= factor;
  }
}
```

Commands may run anywhere under the project because Tach finds the nearest
ancestor `tach.json`:

```sh
npx tach check
npx tach build
```

`check` validates the whole project, WGSL path, SPIR-V path, package facade,
and generated documentation without writing. `build` atomically replaces
`build/` with one complete package:

```text
build/
  package.json
  index.js
  index.d.ts
  kernel.wgsl
  kernel.spv
  README.md
  docs/
    kernels.md
```

The package manifest declares `@depths/tach` because generated `index.js`
uses its private runtime protocol. Application code still imports `tach`
directly from `@depths/tach` and public commands from the generated package.
The generated package never re-exports the runtime.

## Singular host model

The public API has three core operations:

```text
gpu.buffer(hostValue)    -> ComputeBuffer
generatedProgram(...)   -> ComputeCommand
gpu.submit(commands...) -> ordered GPU work
```

A generated function constructs an opaque command; it does not execute.
`submit()` prepares commands and queues work in call order. `read()`, `idle()`,
and scoped-session exit are completion boundaries.

Backend selection is an implementation fact:

- a browser global selects WebGPU, requests an adapter/device, and fetches the
  generated WGSL beside `index.js`;
- a Deno global selects the packaged Vulkan 1.3 FFI library and reads the
  generated SPIR-V module beside `index.js`;
- all other environments fail with a structured availability error.

There is one physical `index.js`, one `index.d.ts`, one `tach` export, and one
generated command ABI. There are no browser/server entry points or conditional
package exports.

## Programs and commands

An exported indexed function is syntax sugar for one public dispatch:

```tach
export function scale[i](values: buffer<float32[]>, factor: float32) {
  if (i < values.length) {
    values[i] *= factor;
  }
}
```

An exported orchestration function can expose a multi-stage plan through the
same command type:

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

Generated metadata carries stages, dispatch order, barriers, transients,
shape expressions, bindings, layouts, and parameter sources. Both drivers
consume that metadata rather than independently interpreting Tach semantics.

A command retains its buffer handles and plain arguments until submission.
Do not mutate object/array value arguments between construction and
submission. Commands belong to the session that materializes them and cannot
cross sessions. Different buffer parameters of one public command require
different `ComputeBuffer` handles; express intentional in-place access with a
single parameter.

Accidentally awaiting a generated command throws a targeted error:

```ts
await scale(values, 2); // error: pass commands to gpu.submit()
```

Batch related commands to preserve order with one submission:

```ts
await gpu.submit(
  scale(values, 2),
  scale(values, 0.5),
);
```

Concurrent calls to `submit()` are serialized by the session. A resolved
submission promise means preparation and queue submission completed, not that
the GPU is idle.

## Session lifetime

### Scoped work

```ts
const output = await tach(async (gpu) => {
  const input = gpu.buffer(initial);
  const result = gpu.buffer(new Float32Array(initial.length));
  await gpu.submit(transform(input, result, initial.length));
  return result.read();
}, { powerPreference: "high-performance" });
```

`tach(work, options?)` opens a session, invokes `work`, waits for all submitted
GPU work, closes every owned resource, and returns the callback result. A
callback failure becomes a `TachError` with code `"user"` while preserving its
cause. Return decoded host data, not a session-owned buffer.

### Persistent work

```ts
const gpu = await tach({ powerPreference: "high-performance" });
const state = gpu.buffer(initialState);

try {
  for (let frame = 0; frame < 1_000; frame++) {
    await gpu.submit(step(state, 1 / 60));
  }
  await gpu.idle();
} finally {
  gpu.close();
}
```

Use a persistent session for a frame loop, simulation, solver, or service.
Pipelines, modules, buffers, descriptor/bind groups, submission resources,
parameter storage, and transient scratch remain reusable. `close()` is
idempotent. Await `idle()` first when the caller requires successful
completion before teardown.

## Buffers

```ts
interface ComputeBuffer<T> {
  write(value: T): void;
  read(): Promise<T>;
  destroy(): void;
}
```

`gpu.buffer(value)` structured-clones the initial host value. Its first command
use provides the compiler-generated codec, fixes the byte representation, and
materializes backend storage. Before materialization, `write()` may replace the
value with another size. After materialization, writes must preserve exact byte
length; allocate a new handle to resize.

`write()` validates and packs immediately. WebGPU writes the resident storage
buffer through its queue. Vulkan transfers through Tach-owned staging memory
to device-local storage.

`read()` first waits for earlier submissions, then transfers, decodes, and
returns a clone. Reading an unused handle returns a clone of its host value and
does not allocate GPU memory. Readback is deliberately explicit and should
stay outside hot loops.

`destroy()` is idempotent. Use after destroy, after session close, or from a
different session raises a lifecycle error. Session close destroys all
remaining handles.

### Host data shapes

| Tach value | JavaScript/TypeScript value |
|---|---|
| `int32`, `uint32`, `float32` | `number` |
| `bool` | `boolean` |
| storage atomic | `number` |
| numeric vector | readonly numeric tuple |
| named struct | generated readonly object |
| scalar runtime array | matching typed array or readonly array |
| two-/four-lane runtime vector array | flat typed array or tuple array |
| three-lane runtime vector array | tuple array |

Three-lane vectors have padded storage stride, so flat typed arrays would
describe the wrong layout. Matching scalar, two-lane, and four-lane typed
arrays can use their native little-endian representation. Generated codecs
validate numeric ranges, vector lanes, array counts, struct fields, runtime
tails, offsets, and strides; callers never provide ABI padding.

## Launch and repeat options

Every command supports repeat:

```ts
interface CommandOptions {
  readonly repeat?: number;
}
```

`repeat` is a positive `uint32` integer and defaults to one. It repeats the
complete public program. The compiler may represent a provably independent
one-stage repetition as an invocation-local loop; generated target plans make
that decision once for both hosts.

Only exported indexed shorthand accepts host launch size:

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

| Tach coordinates | `size` value |
|---|---|
| `[i]` | `number` |
| `[x, y]` | `readonly [number, number]` |
| `[x, y, z]` | `readonly [number, number, number]` |

Components are positive safe integers and rank must match. A 1D shorthand can
infer size from its first runtime-sized public resource. Otherwise omission
means one workgroup. Tach rounds dispatches up to full workgroups; kernels must
guard edge coordinates. Explicit orchestration programs derive domains in
Tach source and accept `CommandOptions`, not `LaunchOptions`.

## Adapter information and options

```ts
interface TachOptions {
  readonly powerPreference?: "low-power" | "high-performance";
}

interface TachAdapterInfo {
  readonly backend: "webgpu" | "vulkan";
  readonly name: string;
  readonly vendor?: string;
  readonly architecture?: string;
  readonly type?: "integrated" | "discrete" | "virtual" | "cpu" | "unknown";
}

interface Tach {
  readonly adapter: TachAdapterInfo;
  buffer<T>(value: T): ComputeBuffer<T>;
  submit(first: ComputeCommand, ...rest: readonly ComputeCommand[]): Promise<void>;
  idle(): Promise<void>;
  close(): void;
}
```

`powerPreference` is a preference, not a guarantee. The browser passes it to
WebGPU adapter selection. Vulkan ranks suitable physical devices using the
same intent. The returned adapter object is deliberately backend-neutral and
contains no raw `GPUDevice`, Vulkan handle, or provider-specific escape hatch.

## Residency and measurement

First use can upload storage, read shader bytes, create physical pipelines,
allocate parameter storage, and grow transient scratch. Warm work reuses those
objects.

WebGPU caches shader modules and physical pipelines per generated module,
bind groups by exact layout/buffer tuple, one aligned uniform arena, and
scratch by compiler allocation color. Vulkan caches the loaded SPIR-V module,
pipeline/layout per used physical stage, resident device-local buffers,
descriptor/command/fence submission objects, one mapped parameter arena, and
scratch allocations. Submission objects return to the session pool after
their fence completes.

For meaningful timings:

1. keep one persistent session;
2. allocate resident buffers once;
3. warm every measured program;
4. batch naturally related commands;
5. time `submit()` through `idle()`; and
6. read and validate after timing.

## Failures

```ts
type TachErrorCode =
  | "webgpu-unavailable"
  | "adapter-unavailable"
  | "device-request-failed"
  | "device-lost"
  | "gpu-validation"
  | "gpu-out-of-memory"
  | "gpu-internal"
  | "vulkan-unavailable"
  | "vulkan-profile"
  | "native"
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

WebGPU error scopes, uncaptured errors, device loss, Vulkan loader/profile
failures, native validation, buffer codecs, lifecycle misuse, callback
failures, and compiler delivery all cross the public boundary as `TachError`.
Original causes are retained when available. Deferred submission failures
surface again at later submission or synchronization boundaries.

## Project command

The executable published by this npm package is a Deno script. It is the only
public compiler command surface:

```text
tach build [--verbose]
tach check
tach docs
tach fmt
tach instructions [--details <section>...]
tach version
tach help
tach --help
tach -h
```

Project commands find the nearest `tach.json` and never accept a source-file
argument.

- `build` validates and writes the fixed complete package. `--verbose` adds
  diagnostics under `build/diagnostics/` without changing executable output.
- `check` executes discovery, both DAG checks, recovery parsing, semantics,
  IR verification/optimization, both target plans, WGSL, SPIR-V 1.6, package
  bindings, and documentation rendering entirely in memory.
- `docs` runs the cheaper documentation path and transactionally refreshes
  only generated README/module Markdown while preserving compiled artifacts.
- `fmt` validates and transactionally formats every `.tach` file in the
  project. One bad file prevents all writes.
- `instructions` needs no project or compiler. It prints the compact AI-agent
  guide; `--details 20 21 22` retrieves only those numbered reference chunks.

An ordinary build contains exactly:

```text
README.md
docs/<module>.md
index.d.ts
index.js
kernel.spv
kernel.wgsl
package.json
```

`--verbose` additionally contains `diagnostics/{flow.ir,kernel.ir,
kernel.spvasm,project.json,runtime.json,spirv.kernel.ir,spirv.plan.json,
web.kernel.ir,web.plan.json}`. Generated trees are compiler-owned atomic sets;
never edit a file or mix versions.

## Deno compiler API

Deno build tools can import the same operations:

```ts
import {
  build,
  check,
  compilerPath,
  docs,
  format,
} from "@depths/tach/compiler";

const cwd = Deno.cwd();
const compiler = await compilerPath();
const checked = await check({ cwd });
const built = await build({ cwd, verbose: true });
await docs({ cwd });
await format({ cwd });
```

`build`, `check`, and `docs` return the canonical root and checked project
description. `CompilerRunOptions` accepts a starting `cwd` and optional
environment overlay. `compilerPath()` resolves and version-checks the private
Go engine. The API operates only on complete projects and does not expose the
private native protocol. Do not include `@depths/tach/compiler` in a browser
bundle.

Compiler resolution uses:

1. explicit `TACH_BIN`;
2. a package-local compiler;
3. the repository `dist/tach[.exe]` during development; or
4. the exact release asset for package version, OS, and architecture.

Release downloads are SHA-256 checked, retried a bounded number of times, and
placed atomically. A bad `TACH_BIN` is an error rather than a silent fallback.

## Package boundaries

The package root exports `tach`, `TachError`, and public types. The
`@depths/tach/compiler` subpath is the Deno project API. Generated `index.js`
alone imports `@depths/tach/internal`; applications must not.

Generated declarations use compiler-private `$...` aliases for runtime types,
so user Tach names cannot collide. Generated JavaScript embeds schema-1
runtime metadata and sibling URLs for both shader artifacts. The runtime is
the sole interpreter of that metadata, keeping compiler, WebGPU, and Vulkan
execution in one protocol.

## Deno permissions and supported hosts

The public project command needs file, environment, subprocess, and download
permissions. Deno Vulkan execution needs `--allow-ffi --allow-read` and access
to the packaged native library plus local generated SPIR-V. Browser execution
uses normal browser WebGPU security and module fetching.

The native Vulkan host currently ships for x86-64 Windows and Linux and
requires a Vulkan 1.3 loader/device supporting Synchronization2 and
`shaderZeroInitializeWorkgroupMemory`. Compiler binaries are published for
x64 and arm64 Windows and Linux, plus Apple-silicon macOS.

## Repository validation

From the Tach repository root:

```sh
npm ci --ignore-scripts
npm run check
npm test
```

The package unit suite checks compiler discovery/install/atomic output,
generated TypeScript, host packing, typed arrays, ownership, non-aliasing,
multi-stage plans, transient scratch, shapes, repeats, batching, caches,
errors, synchronization, and cleanup. `browser-test` and `deno-test` execute
the same canonical example project on WebGPU and Vulkan. `showcase-ts` executes
the same six large workloads through both hosts and reports exact observations.

Further reading:

- [Project overview](../README.md)
- [Language](../docs/language.md)
- [Architecture](../docs/architecture.md)
- [ABI](../docs/abi.md)

`@depths/tach` is licensed under `AGPL-3.0-only`.
