# Tach

Tach is a small, strictly typed language and compiler for general-purpose GPU
compute. Its source looks familiar to TypeScript and C programmers, but it does
not expose WGSL builtins, SPIR-V instructions, descriptor coordinates, or GPU
provider quirks. A kernel names the logical coordinates it needs; Tach owns the
mapping, optimization, validation, memory layout, and host bindings.

```tach
export compute scale[i](
  values: buffer<f32[]>,
  factor: uniform<f32>,
) {
  if (i < values.length) {
    values[i] *= factor;
  }
}
```

One source module produces all of these artifacts:

| Artifact | Purpose |
|---|---|
| `.tir` | readable, verified Tach Core IR |
| `.wgsl` | WebGPU compute shader |
| `.spv` | SPIR-V 1.3 compute module |
| `.spvasm` | Tach's disassembly of the emitted SPIR-V bytes |
| `.js` | generated WebGPU command constructors |
| `.d.ts` | generated TypeScript interfaces and function signatures |
| `.tach.json` | reflection, resource, launch, and host-layout metadata |

The compiler validates Core IR before and after optimization, lowers each
backend through its own private representation, validates emitted WGSL and
SPIR-V, and checks the generated binding contract before reporting success.

## Quick start

Tach's npm package requires Node.js 22 or newer. Install it in the application
that will compile and run kernels:

```sh
npm install @depths/tach
npx tach version
```

The package selects a native compiler for Linux, macOS, or Windows on amd64 or
arm64, downloads the binary from the matching GitHub release, and verifies its
SHA-256 checksum. A global installation is optional:

```sh
npm install --global @depths/tach
```

Save the first example as `scale.tach`, then compile it:

```sh
npx tach check scale.tach
npx tach build scale.tach
```

`check` runs the entire compilation and validation pipeline without writing
artifacts. `build` writes the seven artifacts to `build/`, using the source
filename as the base name.

The generated module mirrors the source kernel instead of exposing shader
plumbing:

```ts
import { tach } from "@depths/tach";
import { scale } from "./build/scale.js";

const initial = new Float32Array([1, 2, 3, 4]);
const result = await tach(async (gpu) => {
  const values = gpu.buffer(initial);

  await gpu.submit(scale(values, 2));
  return values.read();
});

if (!result.ok) {
  throw new Error(`${result.error.code}: ${result.error.message}`);
}

console.log(result.value); // Float32Array [2, 4, 6, 8]
```

The generated `scale` function returns an opaque compute command. It does not
submit work by itself. `gpu.submit(...)` can accept one or many commands and
records them into one compute pass and one queue submission. A buffer read,
`gpu.idle()`, or the end of `tach(...)` waits for submitted GPU work.

For a one-dimensional kernel, Tach infers the logical invocation count from
the first runtime-sized buffer when `size` is omitted. Launch dimensions can
also be explicit:

```ts
await gpu.submit(scale(values, 2, { size: initial.length }));
```

The generated TypeScript type requires a scalar size for a one-dimensional
kernel, a two-element tuple for a two-dimensional kernel, and a three-element
tuple for a three-dimensional kernel.

## Two host lifetimes, one runtime

`tach(...)` is the scoped form. It acquires an adapter and device, runs the
callback, waits for queued work, converts failures to a `Result`, and destroys
the session and every owned GPU buffer on exit. It is ideal for a complete
batch of work:

```ts
const result = await tach(async (gpu) => {
  const state = gpu.buffer(initialState);
  await gpu.submit(step(state, params, { dispatches: 100 }));
  return state.read();
});
```

`openTach()` exposes the same session as a caller-owned, persistent object. It
is suitable for frame loops and iterative applications because buffers stay
resident, shader modules and pipelines are cached per device, stable bind
groups are reused, and uniforms share a persistent upload arena:

```ts
import { openTach } from "@depths/tach";
import { step } from "./build/simulation.js";

const opened = await openTach();
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

These are ownership choices, not separate execution engines. Both forms use
the same `ComputeBuffer`, generated command, submission, synchronization,
error, cache, and cleanup rules.

## The source model

An exported kernel declares one to three immutable logical coordinates:

```tach
export compute line[i](out: buffer<u32[]>) { /* i: u32 */ }

export compute image[x, y](out: buffer<f32x4[]>) {
  const pixel = y * 1920 + x;
  // ...
}

export compute volume[x, y, z](out: buffer<f32[]>) { /* ... */ }
```

The coordinate names are ordinary Tach `u32` values. Source code never asks
for a target invocation object. The compiler selects these portable default
workgroups:

| Kernel rank | Default workgroup |
|---:|---:|
| 1D | `256 × 1 × 1` |
| 2D | `16 × 16 × 1` |
| 3D | `8 × 8 × 4` |

Most kernels should use the default. Algorithms that depend on a particular
tile shape or shared-memory protocol can state it explicitly:

```tach
@workgroup(16, 16)
export compute tiled[x, y](out: buffer<f32[]>) {
  workgroup tile: f32[256];
  const localX = x % 16;
  const localY = y % 16;
  const local = localY * 16 + localX;
  // ...
  workgroupBarrier();
}
```

That arithmetic remains target-neutral Tach. Backend optimization recognizes
local-coordinate and row-major local-index forms and maps them to the most
appropriate target inputs without changing source or Core IR semantics.

The current portable language includes:

- `bool`, `i32`, `u32`, and `f32` scalars;
- two-, three-, and four-lane signed, unsigned, and floating vectors such as
  `f32x4`;
- named structs, contextual object-shaped literals, vector construction, lane
  access, swizzles, and indexing;
- suffix-free context-typed numeric literals and explicit scalar conversions;
- immutable `const`, rebindable `let`, and pure value helper functions;
- structured `if`, `else if`, `else`, ternary expressions, `while`, and
  counted `for` loops;
- read-only uniforms and buffers whose read/write access is inferred;
- runtime buffer arrays with `.length` and fixed workgroup arrays;
- integer atomics, zero-initialized workgroup memory, and workgroup/buffer
  barriers;
- arithmetic, comparisons, short-circuit boolean logic, bitwise operations,
  modulo-32 shifts, compound assignment, and `++`/`--`;
- portable scalar and vector math intrinsics.

See [the language reference](docs/language.md) for the exact syntax and type
rules.

## Compiler commands

The CLI accepts exactly one source file for every compilation command:

```text
tach build FILE.tach
tach check FILE.tach
tach ir FILE.tach
tach wgsl FILE.tach
tach spirv-dis FILE.tach
tach version
```

Use the inspection commands to understand what the compiler proved and
emitted:

```sh
npx tach ir scale.tach
npx tach wgsl scale.tach
npx tach spirv-dis scale.tach
```

The `.tir` text and `ir` command are diagnostic views of Core IR, not a second
accepted input language. Rebuild generated artifacts together from `.tach`
source; do not edit generated JS, declarations, metadata, WGSL, or SPIR-V by
hand.

## How compilation works

```text
Tach source
  -> lexer and parser
  -> type checking + semantic lowering
  -> verified structured Core IR
  -> target-independent optimization
  -> verified optimized Core IR
       |                         |
       v                         v
     WGSL lowering             SPIR-V lowering
     + target pass             + target pass
       |                         |
       v                         v
     WGSL emission             SPIR-V 1.3 emission
     + validation              + binary validation
                                 + disassembly
       \_________________________/
                    |
                    v
       JS / TypeScript / metadata generation
       + generated-contract validation
```

Core IR separates immutable SSA values from typed addressable places. Source
local mutation becomes structured region results and loop-carried values;
actual memory effects remain explicit loads, stores, atomics, and barriers.
This keeps optimization semantic and target-neutral while allowing WGSL to
remain structured and SPIR-V to receive explicit blocks, branches, and
`OpPhi` nodes.

The compiler owns one host ABI for both targets. The same layout calculation
drives WGSL wrappers, SPIR-V offsets and strides, reflection metadata, minimum
binding sizes, and TypeScript runtime packing. Host code never parses a shader
to rediscover its interface.

For the full design, read:

- [Compiler architecture](docs/architecture.md)
- [Tach language reference](docs/language.md)
- [Core IR reference](docs/ir.md)
- [Resource, host, and runtime ABI](docs/abi.md)

## Building this repository

Compiler development requires Go 1.23 or newer. Runtime and browser development
requires Node.js 22 or newer.

```sh
npm ci
npm run compiler
npm run check
```

`npm run compiler` builds the native CLI to `dist/`. `npm run check` rebuilds
it and performs every workspace's static checks.

Run compiler unit tests without the native Vulkan harness:

```sh
go test -count=1 ./src/...
go vet ./...
```

Install Chromium once, then run the WebGPU runtime, browser examples, and
showcase suites:

```sh
npm run install:browser --workspace=@tach/browser-test
npm run install:browser --workspace=@tach/showcase-ts
npm test
```

Run the SPIR-V backend through a native Vulkan compute pipeline:

```sh
npm run test:spirv
```

That harness additionally requires CGO, a C compiler, a Vulkan 1.1 loader and
compute-capable driver, and Khronos `spirv-val`. It prefers hardware and can run
against a CPU Vulkan implementation such as Mesa Lavapipe. The complete Go
suite includes this harness:

```sh
go test -count=1 ./...
```

The private integration workspaces document their focused workflows:

- [browser-test](browser-test/README.md) executes the seven examples through
  generated WGSL and TypeScript bindings in Chromium;
- [spirv-test](spirv-test/README.md) executes the same examples through native
  SPIR-V and Vulkan;
- [showcase-ts](showcase-ts/README.md) compares five sustained CPU/GPU
  workloads, including a procedural 1920×1080 RGBA image, and writes a
  human-readable benchmark report.

## Repository map

```text
main.go                 native CLI
src/lexer               source tokenization
src/parser              syntax tree construction
src/ast                 source AST
src/types               semantic type model
src/sema                checking and Core IR lowering
src/ir                  Core IR, verification, uniformity, use accounting
src/opt                 target-independent optimization
src/backend             shared target-lowering analysis
src/layout              compiler-owned host layout
src/wgsl                WGSL lowering, emission, and validation
src/spirv               SPIR-V lowering, emission, validation, disassembly
src/bindings            metadata and JS/TypeScript generation
src/compiler            end-to-end compilation orchestration
tach-ts                 @depths/tach compiler delivery and WebGPU runtime
examples                executable language examples
browser-test            browser/WebGPU integration harness
spirv-test              native Vulkan integration harness
showcase-ts             TypeScript performance and rendering showcase
```

## Core invariants

1. Tach syntax and Core IR are target-neutral.
2. Logical kernel coordinates are explicit; provider-specific invocation
   objects are not part of the language.
3. Values and addressable places are different IR concepts.
4. Control flow remains structured until a backend requires a CFG.
5. Portable semantics are fixed before backend emission.
6. Target-independent and backend-specific optimization are separate stages.
7. Resource identity, byte layout, entry names, and reflection belong to the
   compiler.
8. A successful compilation has validated every generated artifact it owns.

## Releases

Maintainers can test, cross-compile the six supported native targets, pack the
npm package, and create a checksum manifest locally:

```sh
./release.sh v0.1.0
```

Adding `--publish` uploads the native binaries to GitHub Releases and publishes
the wrapper to npm. It requires authenticated GitHub and npm CLIs plus a clean,
committed worktree:

```sh
./release.sh v0.1.0 --publish
```

## License

Tach is licensed under the [GNU Affero General Public License version 3](LICENSE)
only (`AGPL-3.0-only`).
