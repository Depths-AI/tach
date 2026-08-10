# Tach

Tach is a lean, strictly typed GPGPU language and compiler written in Go. Its source surface is deliberately familiar to TypeScript/C-family programmers while its semantics are GPU-native.

A Tach module is compiled once into:

- structured SSA-ish Tach IR (`.tir`)
- WGSL (`.wgsl`)
- SPIR-V 1.3 binary (`.spv`)
- Tach's own SPIR-V disassembly (`.spvasm`)
- WebGPU JavaScript bindings (`.js`)
- TypeScript declarations for those bindings (`.d.ts`)
- compiler-owned reflection/ABI metadata (`.tach.json`)

Tach owns every stage of that pipeline, including validation of its IR, generated WGSL subset, SPIR-V binary, and generated binding contract. Compilation does not delegate correctness to an external shader compiler or validator.

## Install

Install the TypeScript runtime and compiler as one project dependency (Node.js
22 or newer):

```sh
npm install @depths/tach
npx tach version
```

The package detects Linux, macOS, or Windows and x64 or arm64, fetches the
matching native compiler from the package version's GitHub Release, and
verifies it against that release's SHA-256 manifest. npm exposes `tach` to
project scripts and through `npx` on every platform. A global install is also
available when a machine-wide command is preferable:

```sh
npm install --global @depths/tach
```

## Build

Tach requires Go 1.23 or newer.

```sh
go build -o dist/ .
```

Run the full test suite:

```sh
go test ./...
go vet ./...
```

The repository is one npm workspace containing the published runtime plus its
private browser and showcase consumers:

```sh
npm ci
npm test
```

`npm run check` performs the compiler build and strict TypeScript checks without
launching the browser suites.

## Release

Releases are built entirely on the local development machine. To run the Go,
package, showcase, and browser tests; cross-compile all six native targets; pack
`@depths/tach`; and write the checksum manifest under `dist/VERSION/`:

```sh
./release.sh v0.1.0
```

To perform the same local build, upload the raw native binaries to GitHub
Releases, and publish the wrapper to npm:

```sh
./release.sh v0.1.0 --publish
```

Publishing requires authenticated GitHub and npm CLIs plus a clean, committed
worktree. GitHub stores the finished native binaries; it does not compile or
test them.

## Browser testing

`browser-test/` is a private workspace that compiles every example through
`@depths/tach`, loads the generated direct kernel functions in headless
Chromium, checks their browser/ABI interfaces, and executes every kernel.

```sh
npm ci
npm run install:browser --workspace=@tach/browser-test
npm test --workspace=@tach/browser-test
```

The same `npm test` command runs the full suite on every machine. Chromium
prefers a physical adapter and falls back to CPU-backed SwiftShader when needed;
the harness reports `hardware-accelerated` or `software-emulated` with the
adapter identity. See [`browser-test/README.md`](browser-test/README.md) for the
complete workflow. Every run also writes `browser-test/test-report.md` for
direct inspection on headless hosts.

## First kernel

```tach
// particles.tach

type Params = {
  dt: f32,
  count: u32,
};

type Particle = {
  position: vec4f,
  velocity: vec4f,
};

fn integrateParticle(p: Particle, dt: f32): Particle {
  return {
    position: p.position + p.velocity * dt,
    velocity: p.velocity,
  };
}

@workgroupSize(256)
export compute integrate(
  @group(0) @binding(0) particles: storage<Particle[], read_write>,
  @group(0) @binding(1) params: uniform<Params>,
) {
  const i = globalId.x;
  if (i >= params.count) {
    return;
  }
  particles[i] = integrateParticle(particles[i], params.dt);
}
```

Validate the entire compilation pipeline:

```sh
npx tach check particles.tach
```

Build all artifacts:

```sh
npx tach build particles.tach
```

Artifacts are written to `build/` using the source filename as their base name.

The generated JavaScript/TypeScript interface mirrors the Tach source:

```ts
import { tach } from "@depths/tach";
import { integrate, type Particle } from "./build/particles.js";

const initial: readonly Particle[] = [
  {
    position: [1, 2, 3, 1],
    velocity: [2, 4, 6, 0],
  },
];

const result = await tach(async (gpu) => {
  const particles = gpu.buffer(initial);
  await integrate(particles, { dt: 0.5, count: initial.length });
  return particles.read();
});

if (result.ok) {
  console.log(result.value);
} else {
  console.error(result.error.code, result.error.message);
}
```

Tach structs become ordinary TypeScript interfaces and exported kernels become
same-named positional functions. Storage parameters use the single persistent
`ComputeBuffer<T>` abstraction created by the scope; uniform parameters are
plain values. `tach(...)` owns adapter, device, buffers, queued work, and cleanup
and returns failures as a discriminated result. Invocation count is inferred
from the first runtime-sized storage buffer, with an optional final number or
`[x, y, z]` available for kernels that need an explicit size.
See [`showcase-ts/`](showcase-ts/) for a standalone strict-TypeScript app using
this interface end to end.

Inspect individual compiler stages:

```sh
npx tach ir particles.tach
npx tach wgsl particles.tach
npx tach spirv-dis particles.tach
```

## Language shape

Tach's current portable core includes:

- `bool`, `i32`, `u32`, `f32`
- `vec2/3/4` integer and floating vectors
- named structs and object-shaped struct construction
- immutable `const` and rebindable `let`
- pure helper functions
- structured `if` / `else if` / `else`, `while`, counted `for`, and ternary expressions
- storage and uniform resources
- fixed workgroup arrays
- `atomic<i32>` / `atomic<u32>` and integer atomic operations
- workgroup and storage barriers
- compute builtins (`globalId`, `localId`, `localIndex`, `workgroupId`, `numWorkgroups`)
- runtime storage array `.length`
- arithmetic, comparison, boolean, bitwise, shifts, conversions, vector construction/access
- portable float math and vector math intrinsics

The compiler lowers source-level mutable locals into SSA values. Addressability exists only where the program actually needs a memory location: resources and workgroup memory become explicit typed **places** in Core IR.

## Architecture

```text
Tach source
    │
    ▼
lexer + parser
    │
    ▼
typed semantic lowering
    │
    ▼
Structured Core IR
(values + places + regions + resource ABI)
    │
    ▼
IR optimization + verification
    │
    ├─────────────────────────────┐
    ▼                             ▼
WGSL backend                  SPIR-V backend
    │                             │
    ▼                             ▼
Tach WGSL validator           Tach binary validator
                                  │
                                  ▼
                             Tach disassembler
    │                             │
    └──────────────┬──────────────┘
                   ▼
          JS/TS binding generator
                   │
                   ▼
          Tach binding validation
```

The IR is intentionally target-neutral. It contains semantic operations rather than WGSL syntax or SPIR-V opcodes. Structured control remains structured through optimization; the WGSL backend emits structured statements, while the SPIR-V backend materializes merge blocks, loop headers, branches, and `OpPhi` nodes.

See:

- [`docs/architecture.md`](docs/architecture.md) — compiler architecture and invariants
- [`docs/language.md`](docs/language.md) — source language reference
- [`docs/ir.md`](docs/ir.md) — Core IR model and lowering rules
- [`docs/abi.md`](docs/abi.md) — resource, memory layout, reflection, and binding ABI

## Repository layout

```text
main.go             CLI entry point
package.json        private npm workspace root
tach-ts/            published @depths/tach runtime + compiler delivery
src/lexer/          source tokenizer
src/parser/         source parser
src/ast/            syntax tree
src/types/          semantic type model
src/sema/           type checking + Core IR lowering
src/ir/             structured SSA-ish IR + verification + uniformity
src/opt/            target-neutral optimization passes
src/layout/         compiler-owned host ABI layout
src/wgsl/           WGSL emitter + Tach WGSL validator
src/spirv/          SPIR-V emitter + decoder + validator + disassembler
src/bindings/       WebGPU JS/TS + metadata generation/validation
src/compiler/       end-to-end compilation pipeline
examples/           executable Tach examples
browser-test/       private headless-browser and WebGPU test workspace
showcase-ts/        private strict-TypeScript consumption showcase
```

## Design invariants

1. **Values and places are separate.** Normal expressions produce immutable SSA values. Loads/stores/atomics operate on explicit addressable places.
2. **Control flow stays structured.** `if` and loops are region constructs in Core IR; branches and phis are a SPIR-V backend concern.
3. **The ABI belongs to Tach.** Resource bindings, physical buffer layout, entry-point naming, and reflection are computed before backend emission.
4. **One semantic program reaches both targets.** Portable semantics such as shift behavior and barrier/atomic meaning are fixed in Tach rather than left to backend interpretation.
5. **Generated artifacts are checked before success.** `Compile` returns only after every owned verification layer succeeds.

## License

Tach is licensed under the [GNU Affero General Public License version 3](LICENSE)
only (`AGPL-3.0-only`).
