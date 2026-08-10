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

Release binaries are published through GitHub Releases for Linux, macOS, and Windows on amd64 and arm64.

Unix (Linux or macOS):

```sh
curl --proto '=https' --tlsv1.2 -fsSL https://raw.githubusercontent.com/Depths-AI/tach/master/install.sh | sh
```

This installs `tach` to `~/.local/bin` by default. Set `TACH_INSTALL_DIR` on the `sh` process to choose another directory.

Windows PowerShell:

```powershell
irm https://raw.githubusercontent.com/Depths-AI/tach/master/install.ps1 | iex
```

This installs `tach.exe` to `%LOCALAPPDATA%\Tach\bin` and adds that directory to the user `PATH`. Both installers verify the downloaded release archive against the published SHA-256 checksum manifest.

## Build

Tach requires Go 1.23 or newer.

```sh
go build -o bin/tach .
```

Run the full test suite:

```sh
go test ./...
go vet ./...
```

## Release

Releases are built entirely on the local development machine. To run the tests,
cross-compile all six supported OS/architecture targets, package them, and write
their checksum manifest under `dist/VERSION/`:

```sh
./release.sh v0.1.0
```

To perform the same local build and then upload the completed archives to GitHub
Releases:

```sh
./release.sh v0.1.0 --publish
```

Publishing requires an authenticated GitHub CLI and a clean, committed worktree.
GitHub stores the finished archives; it does not compile or test them.

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
bin/tach check particles.tach
```

Build all artifacts:

```sh
bin/tach build particles.tach
```

Artifacts are written to `build/` using the source filename as their base name.

Inspect individual compiler stages:

```sh
bin/tach ir particles.tach
bin/tach wgsl particles.tach
bin/tach spirv-dis particles.tach
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
```

## Design invariants

1. **Values and places are separate.** Normal expressions produce immutable SSA values. Loads/stores/atomics operate on explicit addressable places.
2. **Control flow stays structured.** `if` and loops are region constructs in Core IR; branches and phis are a SPIR-V backend concern.
3. **The ABI belongs to Tach.** Resource bindings, physical buffer layout, entry-point naming, and reflection are computed before backend emission.
4. **One semantic program reaches both targets.** Portable semantics such as shift behavior and barrier/atomic meaning are fixed in Tach rather than left to backend interpretation.
5. **Generated artifacts are checked before success.** `Compile` returns only after every owned verification layer succeeds.
