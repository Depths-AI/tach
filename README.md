# Tach

Tach is a small, typed language for GPU compute. It is designed to feel
familiar to a TypeScript developer while keeping shader-provider details out of
application code.

You write this:

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

Tach turns it into validated WGSL for WebGPU, validated SPIR-V for Vulkan,
typed JavaScript/TypeScript bindings, reflection metadata, and a readable
target-neutral IR. You never write a WGSL invocation builtin, a Vulkan
descriptor number, or a padding field in Tach source.

## Run your first kernel

The npm package needs Node.js 22 or newer. Install it in your application:

```sh
npm install @depths/tach
```

Save the example above as `scale.tach`, then compile it:

```sh
npx tach check scale.tach
npx tach build scale.tach
```

`check` validates without writing files. `build` writes synchronized artifacts
to `build/`, including `scale.js` and `scale.d.ts`.

Call the generated function from TypeScript:

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

That example contains the whole everyday model:

1. `gpu.buffer(...)` creates GPU-resident state owned by one Tach session.
2. `scale(values, 2)` creates a command; it does not execute yet.
3. `gpu.submit(...)` records and submits one or more commands.
4. `values.read()` waits for prior work and returns host data.

## Reading Tach source

A kernel is an exported function with one to three logical coordinates:

```tach
export function line[i](out: buffer<uint32[]>) {
  if (i < out.length) {
    out[i] = i;
  }
}
```

The brackets are the one intentional extension to an ordinary TypeScript-like
function declaration. `i` is an immutable `uint32` supplied to each invocation.
It is not a host argument and it is not a provider-specific invocation object.

Two-dimensional work is equally direct:

```tach
export function image[x, y](pixels: buffer<float32x4[]>) {
  const width = 1920;
  const pixel = y * width + x;
  if (pixel < pixels.length) {
    pixels[pixel] = float32x4(float32(x) / 1920.0, float32(y) / 1080.0, 0.5, 1.0);
  }
}
```

The host supplies a matching logical size:

```ts
await gpu.submit(image(pixels, { size: [1920, 1080] }));
```

Kernel parameters describe data movement:

- `buffer<T>` is storage that a kernel may read or write. Tach infers access.
- a plain type such as `float32` or `Params` is an immutable value copied into
  the command;
- `T[]` is a runtime-sized array and exposes `.length` inside the kernel.

That is the complete distinction: buffers are resident memory; ordinary typed
parameters are values. Shader storage classes never appear in Tach source.

Regular helper functions use the same spelling TypeScript does:

```tach
type Particle = {
  position: float32x4,
  velocity: float32x4,
};

function advance(particle: Particle, dt: float32): Particle {
  return {
    position: particle.position + particle.velocity * dt,
    velocity: particle.velocity,
  };
}

export function step[i](particles: buffer<Particle[]>) {
  if (i < particles.length) {
    particles[i] = advance(particles[i], 0.016);
  }
}
```

Tach uses TypeScript-shaped `type`, `function`, `export`, `const`, `let`, `if`,
`else`, `while`, `for`, ternaries, object literals, property access, indexing,
and return annotations. GPU value types stay explicit: `bool`, `int32`,
`uint32`, `float32`, and vectors such as `float32x4`. A plain `number` would hide
representation decisions that buffers and GPU arithmetic need to agree on.

Numeric literals have no shader suffixes. Write `0`, not `0u`; context infers
the concrete type. Use `uint32(value)`, `int32(value)`, or `float32(value)` for
an explicit conversion when needed.

The complete source contract is in the [language guide](docs/language.md).

## Workgroups are optional

Tach chooses portable defaults from kernel rank:

| Coordinates | Default workgroup |
|---:|---:|
| `[i]` | `256 x 1 x 1` |
| `[x, y]` | `16 x 16 x 1` |
| `[x, y, z]` | `8 x 8 x 4` |

Most kernels should stop there. State a workgroup only when the algorithm
depends on its shape, usually because it uses shared memory:

```tach
@workgroup(64)
export function blockTotals[i](out: buffer<uint32[]>) {
  let partial: shared<uint32[64]>;
  const lane = i % 64;

  partial[lane] = i;
  workgroupBarrier();

  if (lane == 0 && i < out.length) {
    out[i] = partial[0];
  }
}
```

`shared<T>`, barriers, and atomics express real parallel-memory semantics, so
they remain explicit. Mapping coordinate arithmetic to efficient target inputs
is the compiler's job.

## Batch jobs and long-running loops

`tach(...)` owns a complete scoped job. It opens a device, runs the callback,
waits for submitted work, returns the callback value, and closes everything:

```ts
const result = await tach(async (gpu) => {
  const state = gpu.buffer(initialState);
  await gpu.submit(step(state));
  return state.read();
});
```

Calling `tach()` without a callback gives the caller the same session without
automatic shutdown. Use it for animation, simulation, services, or any loop that should keep the
device, pipelines, bind groups, and buffers resident:

```ts
import { tach } from "@depths/tach";

const gpu = await tach();
const state = gpu.buffer(initialState);

try {
  for (let frame = 0; frame < 1_000; frame++) {
    await gpu.submit(step(state));
  }
  await gpu.idle();
} finally {
  gpu.close();
}
```

`submit()` waits for preparation and queue submission, not GPU completion.
`idle()`, buffer readback, and the end of `tach(...)` are completion
boundaries. Both lifetime forms use the same execution engine and ownership
rules. The [TypeScript runtime guide](tach-ts/README.md) covers the full API.

## Compiler output

`build` and `check` use the same target selection. Both default to `web`:

| Target | `build` writes | `check` validates |
|---|---|---|
| `web` | `.wgsl`, `.js`, `.d.ts` | WGSL and generated bindings |
| `spirv` | `.spv` | SPIR-V |
| `all` | every artifact below | both backends plus diagnostics |

The complete `all` artifact set is:

| File | What it is for |
|---|---|
| `.tir` | readable, verified Tach Core IR diagnostic |
| `.wgsl` | WebGPU compute shader |
| `.spv` | SPIR-V 1.3 compute module |
| `.spvasm` | diagnostic disassembly of the emitted SPIR-V bytes |
| `.js` | generated command constructors |
| `.d.ts` | generated TypeScript object types and function signatures |
| `.tach.json` | diagnostic reflection for buffers, values, layouts, entry points, and launches |

The useful CLI commands are:

```text
tach build FILE.tach
tach build --target spirv FILE.tach
tach build --target all FILE.tach
tach check [--target web|spirv|all] FILE.tach
tach ir FILE.tach
tach wgsl FILE.tach
tach spirv-dis FILE.tach
tach version
```

The `.tach` file is the source of truth. A later build of the same module
removes stale Tach artifacts outside the selected target, so `build/` reflects
that invocation exactly. Do not edit or mix generated artifacts by hand.

## What the compiler owns

Compilation is deliberately layered:

```text
Tach source
  -> parse and type-check
  -> verified target-neutral Core IR
  -> target-neutral optimization
  -> shared parameter-block planning
       |                    |
       v                    v
     WGSL IR             SPIR-V IR
  -> target pass       -> target pass
  -> emit + validate   -> emit + validate
       \____________________/
                 |
                 v
        bindings + metadata
        + contract validation
```

The first optimization stage reasons only about Tach semantics. Each backend
then performs representation-specific work privately. Provider terms may
appear in generated WGSL or SPIR-V, never in Tach source or Core IR.

The compiler owns the host boundary. Storage buffers use one canonical layout;
all plain values of a kernel are flattened into one compiler-planned parameter
block. The same offsets drive WGSL, SPIR-V, metadata, the WebGPU runtime, and
the native harness. Host code never parses a shader to rediscover an interface.

For deeper detail:

- [Language guide](docs/language.md): write kernels, from first principles to
  the exact grammar.
- [Architecture guide](docs/architecture.md): how source, IR, optimization,
  backends, bindings, and runtime fit together.
- [Core IR guide](docs/ir.md): values, places, structured control, verification,
  and both optimization levels.
- [ABI guide](docs/abi.md): buffer identity, byte layout, launch geometry,
  host values, sessions, and native caller obligations.

## Develop this repository

Compiler development needs Go 1.23 or newer. Runtime and browser work needs
Node.js 22 or newer.

```sh
npm ci
npm run compiler
npm run check
go test -count=1 ./src/...
go vet ./...
```

Install Chromium once, then exercise generated WGSL in a real WebGPU
implementation:

```sh
npm run install:browser --workspace=@tach/browser-test
npm run install:browser --workspace=@tach/showcase-ts
npm test
```

Exercise generated SPIR-V through Vulkan:

```sh
npm run test:spirv
```

That native harness additionally needs CGO, a C compiler, a Vulkan loader and
driver, and Khronos `spirv-val`.

The focused harness guides are:

- [browser-test](browser-test/README.md): correctness through generated
  TypeScript bindings and Chromium WebGPU;
- [spirv-test](spirv-test/README.md): correctness through SPIR-V and Vulkan;
- [showcase-ts](showcase-ts/README.md): sustained compute and procedural
  rendering workloads with reports and screenshots.

## Repository map

```text
main.go          CLI
src/lexer        tokenization
src/parser       source grammar
src/ast          source-shaped syntax tree
src/types        logical types
src/sema         checking and Core IR lowering
src/ir           Core IR and verification
src/opt          target-neutral optimization
src/backend      shared backend analysis
src/layout       canonical buffer and physical field layout
src/abi          names and shared parameter-block planning
src/wgsl         WGSL lowering, optimization, emission, validation
src/spirv        SPIR-V lowering, optimization, emission, validation
src/bindings     metadata and JS/TypeScript generation
src/compiler     end-to-end orchestration
tach-ts          npm compiler delivery and WebGPU runtime
examples         executable language corpus
browser-test     browser correctness harness
spirv-test       native Vulkan correctness harness
showcase-ts      performance and rendering showcase
```

## Releases and license

Maintainers can build native targets, pack the npm package, and create checksums
with:

```sh
./release.sh v0.1.0
```

`./release.sh v0.1.0 --publish` additionally publishes the release and package;
it requires authenticated GitHub and npm CLIs plus a clean committed tree.

Tach is licensed under the [GNU Affero General Public License version 3](LICENSE)
only (`AGPL-3.0-only`).
