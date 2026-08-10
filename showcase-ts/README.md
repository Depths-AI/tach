# Tach TypeScript showcase

This private Vite workspace consumes `@depths/tach` and a generated Tach module
from strict, ordinary TypeScript. The public binding surface is deliberately
direct:

```ts
import { tach } from "@depths/tach";
import { integrate, type Particle } from "../build/particles.js";

const result = await tach(async (gpu) => {
  const particles = gpu.buffer(initialParticles);
  await integrate(particles, { dt: 0.5, count: initialParticles.length });
  return particles.read();
});
```

`Particle` and `Params` are generated TypeScript interfaces matching the Tach
structs. `integrate` is the exported Tach kernel as a TypeScript function.
Storage arguments use the single persistent `ComputeBuffer<T>` abstraction;
uniform arguments remain plain TypeScript values. The `tach(...)` scope owns
adapter, device, buffers, queued work, and cleanup, then resolves to a
discriminated success or error value.

The invocation count is inferred from the first runtime-sized storage buffer.
Pass an optional final number or `[x, y, z]` only when a kernel needs a
different logical invocation size.

## Run

Install the root workspace, build the native development compiler, then run the
showcase:

```sh
npm ci
npm run compiler
npm run check --workspace=@tach/showcase-ts
npm run dev --workspace=@tach/showcase-ts
```

`npm run build` performs the same kernel generation and strict type-check before
creating a production bundle. `npm test --workspace=@tach/showcase-ts` launches
the page under headless Chromium and verifies its computed particle data. Set
`npm test` serves and exercises the production bundle. Set `TACH_BIN` to use
an explicit Tach executable instead of `dist/tach`.
