# Tach compiler architecture

## 1. Objective

Tach is a GPGPU language whose source syntax optimizes for composition and low cognitive load, while the compiler core optimizes for exact GPU semantics and shallow backends.

The primary targets are:

- WGSL for WebGPU
- SPIR-V for native Vulkan compute
- generated JavaScript/TypeScript host bindings for WebGPU

The architectural center is a structured SSA-ish Core IR with an explicit resource ABI.

## 2. Compiler pipeline

```text
source
  │
  ├─ lexer
  ├─ parser
  │    └─ AST
  │
  ├─ semantic analysis
  │    ├─ declaration/type resolution
  │    ├─ strict expression typing
  │    ├─ source literal canonicalization
  │    ├─ resource/binding resolution
  │    ├─ lexical/local flow analysis
  │    └─ lowering
  │
  └─ Structured Core IR
       │
       ├─ optimizer
       ├─ verifier
       ├─ uniformity analysis
       │
       ├─ WGSL emitter
       │    └─ Tach WGSL-subset validator
       │
       ├─ SPIR-V emitter
       │    ├─ Tach binary decoder/validator
       │    └─ Tach disassembler
       │
       └─ binding/reflection generator
            └─ Tach generated-contract validator
```

`src/compiler.Compile` is the single orchestration point. A successful `Result` means every stage above has completed and the generated artifacts have passed Tach's validators.

## 3. Why structured SSA-ish IR

GPU compute wants both SSA values and structured control.

Tach therefore keeps two separate concepts:

```text
Value<T>
Place<Space, T, Access>
```

A value is immutable and identified by an SSA ID. A place is an addressable memory path rooted in a resource or workgroup variable.

Example:

```text
%4 = const u32 1
%5 = place.index @particles, %4
%6 = load %5
%7 = extract %6, .velocity
store %5, %8
```

This avoids pretending every source variable is memory. A source `let` is a rebindable lexical name; mutation of that name becomes new SSA values and structured region yields. Real memory mutation remains explicit.

## 4. Structured regions

Core IR represents `if` and loops directly.

A value-producing conditional conceptually looks like:

```text
%r = if %cond -> i32 {
  then {
    yield %a
  }
  else {
    yield %b
  }
}
```

The WGSL backend realizes this with a temporary mutable local and structured `if`. The SPIR-V backend realizes it with selection merge blocks and `OpPhi`.

Loops carry values explicitly:

```text
%result = loop (%i = %zero, %acc = %init) -> f32 {
  condition {
    %keepGoing = ...
    yield %keepGoing
  }
  body {
    ...
    continue %nextI, %nextAcc
  }
}
```

The SPIR-V backend turns loop carriers into header `OpPhi` nodes. The WGSL backend turns them into compiler-generated mutable locals around `loop`.

This architecture means neither backend reconstructs structured control from a generic CFG.

## 5. Source mutation and SSA

Tach deliberately permits familiar local code:

```tach
let sum = 0.0;
let i = 0u;
while (i < n) {
  sum += values[i];
  i++;
}
```

Semantic lowering identifies locals assigned by a structured region and materializes them as region results / loop-carried values.

This is the meaning of "SSA-ish": Core IR values are SSA, while explicit memory locations remain mutable by design.

## 6. Places and memory

A place is compiler-semantic, not a general machine pointer. Current place operations cover:

- resource root
- workgroup root
- struct field projection
- array/runtime-array indexing
- vector-lane indexing
- load
- store
- atomics

There is no pointer arithmetic or pointer/integer casting in the portable core.

Access mode is known at the root, so the verifier rejects illegal writes and illegal atomic access before either backend runs.

## 7. Type system boundary

Core IR uses concrete types:

```text
bool
i32
u32
f32
vec2<T>
vec3<T>
vec4<T>
struct
atomic<i32/u32>
fixed array
runtime array
```

Source literals are canonicalized during semantic analysis. Backend emission never needs to guess whether a numeric literal means integer or floating point.

Logical value types are separate from physical host layouts. A struct remains a logical struct throughout optimization; the layout engine separately computes byte offsets, alignment, array stride, and runtime-array tail information for host-visible memory.

## 8. Resource model

Compute parameters are semantic resources, not ordinary function parameters.

```tach
export compute step(
  state: storage<State[], read_write>,
  params: uniform<Params>,
) { ... }
```

Semantic lowering assigns each resource:

- kind (`storage` or `uniform`)
- access (`read` or `read_write`)
- group/set
- binding
- logical type
- physical layout

Explicit `@group` / `@binding` annotations pin the ABI. Omitted locations are assigned deterministically from the module-global binding space.

The same group/binding pair becomes:

```text
Tach group   -> WGSL @group       -> Vulkan DescriptorSet
Tach binding -> WGSL @binding     -> SPIR-V Binding
```

## 9. ABI ownership

The compiler computes host layout once in `src/layout`.

That one result drives:

- WGSL resource wrappers/layout-sensitive declarations
- SPIR-V member offsets and array strides
- generated JS DataView packers
- minimum binding sizes
- reflection metadata

Host bindings never reverse-engineer generated WGSL.

Kernel entry-point names are also ABI-owned and shared by both backends (`__tach_k_<mangled-name>`).

## 10. Portable semantics belong to Core IR

Where target semantics differ, Tach defines one behavior before emission.

A concrete example is 32-bit shifts. Tach normalizes the shift count to its low five bits in Core IR. Both WGSL and SPIR-V then receive the same already-defined operation.

Likewise, atomics and barriers have Tach-level meaning. The WGSL backend selects the corresponding builtin; the SPIR-V backend selects scopes and memory-semantics operands from that Tach operation.

## 11. Uniformity analysis

Core IR computes a conservative per-value uniformity lattice:

```text
Uniform
Varying
Unknown
```

The analysis propagates through structured regions and loop-carried values. Barriers are validated against control-flow uniformity at IR level, before target lowering.

Representative sources:

- constants: uniform
- `workgroupId`, `numWorkgroups`: uniform within a workgroup
- local/global invocation IDs: varying
- uniform-resource load from a uniform address: uniform
- storage/workgroup loads: varying
- atomics: varying

## 12. Optimizer boundary

Optimization is target-neutral and happens after semantic lowering but before backend emission.

The current pass manager runs recursive dead SSA value elimination to a fixed point across nested structured regions. It only removes instructions classified as side-effect-free; memory operations, calls, atomics, barriers, and structured effects remain conservative.

All optimization passes are required to preserve a verifiable Core IR. `opt.Run` verifies both before and after running the pass set.

## 13. WGSL backend

WGSL emission is intentionally shallow:

- values -> expressions/compiler locals
- places -> WGSL reference-producing access expressions
- loads/stores -> WGSL value use/assignment
- structured if -> WGSL `if`
- loop region -> WGSL `loop`
- resources -> module globals with group/binding attributes
- builtins -> compute entry-point builtin parameters
- atomics/math -> WGSL builtins

The Tach WGSL validator reparses the exact WGSL subset emitted by Tach and rejects malformed output shape/syntax. Semantic correctness has already been established in Core IR; this layer protects backend serialization.

## 14. SPIR-V backend

The SPIR-V emitter owns binary word encoding and ID allocation. It materializes the lower-level CFG representation that Core IR deliberately avoids exposing:

- structured selections and merge blocks
- loop headers / merge / continue blocks
- branches
- `OpPhi`
- pointer/storage-class types
- access chains
- explicit load/store
- decorations and interface variables
- atomic scope/memory-semantics operands
- GLSL.std.450 extended math instructions

The Tach SPIR-V validator decodes the emitted bytes independently and validates module structure, IDs, types, decorations, resource layout, function structure, CFG predecessor sets, dominance/SSA use, phi edges, structured merges, memory operations, atomics, barriers, and supported extended instructions.

The disassembler consumes binary bytes as well; it is not an emitter-side debug dump.

## 15. Generated host bindings

The binding generator consumes Core IR + ABI metadata and emits:

- embedded WGSL
- reflection metadata
- typed struct/resource packers
- WebGPU buffer creation/write helpers
- bind-group layout construction
- compute pipeline creation
- dispatch encoding
- TypeScript declarations

The generated contract validator checks metadata/version invariants, exports, declaration correspondence, bindings, and generated source shape before compilation succeeds.

## 16. Durability rule

New source-language conveniences should lower into existing Core IR concepts whenever possible. `for`, ternary expressions, `else if`, and vector-lane writes already follow this rule.

New target capabilities should add semantic operations/types/capabilities to Core IR, followed by independent lowering in each backend. Backend encodings stay out of the source type system.
