# Pine

Pine is a lean, strictly typed GPGPU language and compiler written in Go. Its source surface is deliberately familiar to TypeScript/C-family programmers while its semantics are GPU-native.

A Pine module is compiled once into:

- structured SSA-ish Pine IR (`.pir`)
- WGSL (`.wgsl`)
- SPIR-V 1.3 binary (`.spv`)
- Pine's own SPIR-V disassembly (`.spvasm`)
- WebGPU JavaScript bindings (`.js`)
- TypeScript declarations for those bindings (`.d.ts`)
- compiler-owned reflection/ABI metadata (`.pine.json`)

Pine owns every stage of that pipeline, including validation of its IR, generated WGSL subset, SPIR-V binary, and generated binding contract. Compilation does not delegate correctness to an external shader compiler or validator.

## Build

Pine requires Go 1.23 or newer.

```sh
go build -o bin/pine ./cmd/pine
```

Run the full test suite:

```sh
go test ./...
go vet ./...
```

## First kernel

```pine
// particles.pine

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
bin/pine check particles.pine
```

Build all artifacts:

```sh
bin/pine build -o build particles.pine
```

Inspect individual compiler stages:

```sh
bin/pine ir particles.pine
bin/pine wgsl particles.pine
bin/pine spirv-dis particles.pine
```

## Language shape

Pine's current portable core includes:

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
Pine source
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
Pine WGSL validator           Pine binary validator
                                  │
                                  ▼
                             Pine disassembler
    │                             │
    └──────────────┬──────────────┘
                   ▼
          JS/TS binding generator
                   │
                   ▼
          Pine binding validation
```

The IR is intentionally target-neutral. It contains semantic operations rather than WGSL syntax or SPIR-V opcodes. Structured control remains structured through optimization; the WGSL backend emits structured statements, while the SPIR-V backend materializes merge blocks, loop headers, branches, and `OpPhi` nodes.

See:

- [`docs/architecture.md`](docs/architecture.md) — compiler architecture and invariants
- [`docs/language.md`](docs/language.md) — source language reference
- [`docs/ir.md`](docs/ir.md) — Core IR model and lowering rules
- [`docs/abi.md`](docs/abi.md) — resource, memory layout, reflection, and binding ABI

## Repository layout

```text
cmd/pine/            CLI
internal/lexer/      source tokenizer
internal/parser/     source parser
internal/ast/        syntax tree
internal/types/      semantic type model
internal/sema/       type checking + Core IR lowering
internal/ir/         structured SSA-ish IR + verification + uniformity
internal/opt/        target-neutral optimization passes
internal/layout/     compiler-owned host ABI layout
internal/wgsl/       WGSL emitter + Pine WGSL validator
internal/spirv/      SPIR-V emitter + decoder + validator + disassembler
internal/bindings/   WebGPU JS/TS + metadata generation/validation
internal/compiler/   end-to-end compilation pipeline
examples/            executable Pine examples
```

## Design invariants

1. **Values and places are separate.** Normal expressions produce immutable SSA values. Loads/stores/atomics operate on explicit addressable places.
2. **Control flow stays structured.** `if` and loops are region constructs in Core IR; branches and phis are a SPIR-V backend concern.
3. **The ABI belongs to Pine.** Resource bindings, physical buffer layout, entry-point naming, and reflection are computed before backend emission.
4. **One semantic program reaches both targets.** Portable semantics such as shift behavior and barrier/atomic meaning are fixed in Pine rather than left to backend interpretation.
5. **Generated artifacts are checked before success.** `Compile` returns only after every owned verification layer succeeds.
