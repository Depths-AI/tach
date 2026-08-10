# `@depths/tach`

The first-class TypeScript runtime and native compiler distribution for Tach.

```sh
npm install @depths/tach
```

```ts
import { tach } from "@depths/tach";
import { integrate, type Particle } from "./build/particles.js";

const result = await tach(async (gpu) => {
  const particles = gpu.buffer(initial);
  await gpu.submit(integrate(particles, { dt: 0.5, count: initial.length }));
  return particles.read();
});

if (!result.ok) {
  console.error(result.error.code, result.error.message);
} else {
  console.log(result.value);
}
```

`tach(...)` acquires a WebGPU adapter and device, owns every buffer created by
the scope, waits for submitted GPU work, and releases buffers and the device in
all success and failure paths. Failures leave the scope as a discriminated
`Result<T, TachError>` rather than an unhandled exception.

Generated modules import their private executor from `@depths/tach/internal`
and expose only the TypeScript interfaces and same-named functions declared by
the Tach source. Applications never import the internal entry point directly.

Generated kernel functions construct `ComputeDispatch` commands. They do not
silently submit or wait. `gpu.submit(command, ...)` is the single execution
boundary: every supplied command is encoded in source order into one compute
pass and one queue submission. Awaiting `submit()` waits for pipeline
preparation and host-side submission, not for the GPU queue to become idle.
Awaiting a generated command directly fails with an instruction to submit it,
so a missing dispatch cannot silently become a no-op.
Call `gpu.idle()` only when the CPU actually needs a completion boundary.
`ComputeBuffer.read()` orders its copy after pending submissions and waits for
that readback; the enclosing `tach(...)` scope also waits before cleanup.
Asynchronous WebGPU validation failures are retained by the session and surface
from the next submission, readback, or `idle()` boundary.

Use `openTach()` for a manually managed long-lived session:

```ts
import { openTach } from "@depths/tach";

const opened = await openTach({ adapter: { powerPreference: "high-performance" } });
if (!opened.ok) throw new Error(opened.error.message);

const gpu = opened.value;
const particles = gpu.buffer(initial);

async function frame(dt: number): Promise<void> {
  // Both simulation steps share one compute pass and queue submission.
  await gpu.submit(
    integrate(particles, { dt, count: initial.length }),
    integrate(particles, { dt, count: initial.length }),
  );
}

await frame(1 / 60);
// Continue calling frame() from requestAnimationFrame without gpu.idle().

await gpu.idle();
gpu.close();
```

The adapter, device, storage buffers, shader modules, pipelines, bind-group
layouts, stable bind groups, and uniform upload arena survive across calls.
Queue ordering makes consecutive awaited submissions safe without a per-frame
queue stall. Await `idle()` before graceful shutdown; `close()` immediately
destroys the owned device and buffers. A lost device invalidates its resident
resources, so recovery opens a new session and recreates application state.

Generated kernels infer their logical invocation count from the first
runtime-sized storage buffer. The optional final `DispatchOptions` object can
override that size and batch repeated dispatches:

```ts
await gpu.submit(integrate(
  particles,
  params,
  { size: particleCount, dispatches: 128 },
));
await gpu.idle();
```

All repetitions are encoded in one compute pass and submitted once; the
explicit `idle()` above is needed only because this example wants completed
timing/results immediately. Scalar runtime storage arrays accept
`Float32Array`, `Uint32Array`, or `Int32Array` as appropriate; their bytes are
uploaded directly and the same typed-array representation is preserved by
`read()`.

Installing the package also installs the matching native `tach` compiler for
Linux, macOS, or Windows on x64 or arm64. npm exposes it to package scripts:

```json
{
  "scripts": {
    "kernel": "tach build kernels/particles.tach"
  }
}
```

The package verifies the compiler against the SHA-256 manifest published with
the corresponding GitHub release. `TACH_BIN` selects an explicit compiler for
local development or testing.

Node programs can also drive compilation without spawning their own shell:

```ts
import { build } from "@depths/tach/compiler";

const result = await build("kernels/particles.tach");
if (!result.ok) console.error(result.error.code, result.error.message);
```

`@depths/tach` is licensed under `AGPL-3.0-only`.
