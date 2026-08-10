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
  await integrate(particles, { dt: 0.5, count: initial.length });
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

Use `openTach()` when an application needs a manually managed long-lived
session. Call `close()` when it is no longer needed.

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
