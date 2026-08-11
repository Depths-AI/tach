# Tach compiler and runtime architecture

Tach is one portable compute language with one semantic IR, one host ABI, two
shader backends, and one WebGPU runtime. The architecture keeps those concerns
separate so source remains small while correctness and target detail stay in
the compiler.

This document explains the complete system and the ownership boundary of each
stage. For exact source rules, read [the language reference](language.md). For
the internal representation, read [the Core IR reference](ir.md). For byte and
host contracts, read [the ABI reference](abi.md).

You do not need this document to write a kernel. Its shortest useful model is:

```text
front end owns meaning -> Core IR owns portable semantics
                       -> each backend owns target representation
                       -> layout and bindings own the host boundary
                       -> the runtime owns WebGPU lifetime and submission
```

When a behavior appears in two targets or two host paths, move upward to the
first shared owner rather than patching both consumers.

## 1. Design goals

Tach is organized around eight invariants:

1. Source syntax describes an algorithm, not a shader provider.
2. Logical kernel coordinates are explicit ordinary values.
3. Semantic values and addressable memory are different concepts.
4. Structured control stays structured until a backend needs lower-level CFG.
5. Portable behavior is fixed before target emission.
6. General optimization and backend representation optimization are distinct.
7. Buffer identity, parameter packing, host layout, and generated bindings are
   compiler-owned.
8. Compilation succeeds only when every emitted artifact passes its owning
   validator.

These invariants are practical. They let a simple `for` loop remain simple
source while the compiler carries its state in SSA, promotes repeated buffer
traffic when safe, and emits either structured WGSL or explicit SPIR-V phi
nodes.

## 2. End-to-end pipeline

```text
.tach source
    |
    v
lexer -> parser -> AST
                    |
                    v
           semantic checking
           + Core IR lowering
                    |
                    v
              Core IR verify
                    |
                    v
       target-independent optimization
       + post-optimization verification
                    |
                    v
       shared parameter-block planning
             /                    \
            v                      v
       WGSL lowering          SPIR-V lowering
       + target pass          + target pass
       + emission             + binary emission
       + validation           + binary validation
                                   |
                                   v
                              disassembly
             \                    /
              v                  v
          JS + TypeScript + metadata generation
          + generated-contract validation
```

`src/compiler.CompileTarget` is the orchestration point. It always runs the
shared front end and optimizer, then enters only the requested backend:
`web`, `spirv`, or both for `all`. `Compile` is the complete `all` entry point
used by internal cross-target tests. `WriteDirectory` writes exactly the
selected validated set and removes stale Tach siblings for that module.

No generated stage reads another artifact to rediscover semantics. In
particular, bindings do not parse WGSL, and the SPIR-V disassembler decodes the
emitted binary rather than printing emitter-side state.

## 3. Front end

### Lexer and parser

`src/lexer` turns source into tokens with spans. It owns comments, numeric
spelling, identifiers, punctuation, keywords, and operators.

`src/parser` builds `src/ast` declarations, types, statements, expressions,
attributes, and source spans. It establishes grammatical structure but does
not decide overloads, buffer access, host layout, or target representation.

The AST is intentionally source-shaped. Constructs such as `for`, ternary
expressions, compound assignment, and object-shaped struct literals still
exist there because diagnostics should refer to what the author wrote.

### Semantic analysis and lowering

`src/sema` is the language authority. It performs declaration collection,
type resolution, cycle detection, literal inference, expression checking,
overload selection, lexical-flow analysis, call-cycle rejection, kernel
parameter validation, buffer-access inference, and Core IR construction.

Semantic analysis lowers while it checks because the decisions are coupled:

- a literal enters IR only after it has one concrete type;
- a buffer projection becomes a typed place;
- a local rebinding becomes a new value;
- branch rebindings become `If` results;
- loop rebindings become loop parameters and results;
- short-circuit operators become structured regions; and
- buffer access is inferred from actual memory effects.

The result is fully typed. Backends never repeat source overload resolution or
guess what an unadorned `0` meant.

## 4. The semantic center: Core IR

Core IR in `src/ir` separates:

```text
Value<T>                 immutable computed data
Place<T>                 typed path to GPU memory
Block + terminator       structured execution region
Buffer resource          external kernel memory contract
```

Ordinary locals and plain kernel parameters lower to values. Buffers and shared
variables lower to places. Loads, stores, atomics, and barriers remain explicit
effects.

`If` and `Loop` own their child regions and their merged/carried values. This
is high enough for source-level reasoning and low enough to map directly to
both targets:

- WGSL receives structured statements and generated private locals;
- SPIR-V receives blocks, merge instructions, branches, and `OpPhi`.

The IR has logical Tach types only. Physical padding and target pointer types
belong to later layers.

## 5. Verification and uniformity

`src/ir/verify.go` is not an optional debug checker. It is the executable
boundary between semantic analysis, optimization, and code generation. It
checks definition uniqueness, structured availability, exact types, place
paths, access, calls, returns, region yields, loop carriers, intrinsics,
atomics, barriers, runtime arrays, and shared-memory declarations.

`src/ir/uniformity.go` derives whether each value is uniform or varying within
a workgroup. It follows value dependencies, nested control, helper calls, and
loop-carried fixed points. Barrier verification then rejects collective
operations under varying control.

Uniformity is deliberately a property of proven IR, not a source annotation
and not a backend-specific qualifier.

## 6. Two optimization levels

Tach makes the optimization boundary explicit:

```text
source semantics
    |
    v
Core IR passes                 shared by every target
    |
    v
backend-lowered program        private target representation
    |
    v
backend passes                 target input/encoding choices
```

### Core IR optimization

`src/opt` currently applies:

1. common value/place elimination;
2. loop-invariant code motion;
3. another commoning pass;
4. conservative loop buffer-value promotion;
5. final commoning; and
6. dead pure definition removal to a fixed point.

The pass manager verifies before and after the sequence. Its rules are based
on Tach types, effects, structured dominance, buffer access, and the
non-aliasing buffer contract. It has no WGSL or SPIR-V cases.

Loop promotion illustrates the intended standard. It may keep a repeated
buffer update in a loop-carried SSA value, but only when there is one clear
load/store path and no synchronization, early exit, atomic, or competing
touch of that buffer. It preserves zero-trip behavior by loading lazily and
writing back only if the loop ran.

### Backend optimization

Each backend first creates a private lowered program. The shared analysis in
`src/backend` recognizes exact local-coordinate arithmetic derived from the
kernel's named logical indices and workgroup dimensions. It can replace:

- `coordinate % workgroupDimension` with a target local coordinate; and
- the matching row-major combination with a target local linear coordinate.

It also removes target inputs made unused by those replacements. The source
and Core IR remain provider-neutral; only the backend representation knows
which target builtin implements the result.

This is the current backend pass, not a ceiling. Future target-specific
instruction selection or legalization belongs at the same layer rather than
in source syntax.

## 7. One ABI and layout engine

`src/layout` computes Tach's canonical buffer layout and the physical layouts
used by ABI fields. It owns scalar and vector size/alignment, struct field
offsets, nested-struct extent, fixed-array stride, runtime-tail offset and
stride, and overflow checking.

That one calculation drives:

- WGSL buffer wrappers;
- SPIR-V member offsets and array strides;
- reflection metadata;
- WebGPU minimum binding sizes; and
- generated runtime codecs.

`src/abi` owns external names and the shared immutable-parameter plan. Exported
kernel names preserve their source spelling in WGSL, SPIR-V, metadata,
JavaScript, and TypeScript. Only private compiler symbols are mangled.

Every `buffer<T>` parameter becomes one deterministic module buffer. Plain
parameters remain ordinary Core IR values. After Core optimization, the ABI
planner recursively flattens those values into one fixed-size block per kernel,
turning logical `bool` leaves into physical `uint32` fields. Both shader
backends and host binding generation consume that exact plan. Buffer bindings
come first; parameter blocks receive subsequent group/set `0` bindings. Source
has no binding annotation and host code never allocates bindings itself.

## 8. WGSL backend

`src/wgsl` performs four jobs:

1. verify Core IR;
2. lower logical coordinates and run the backend coordinate pass;
3. emit WGSL declarations and structured code; and
4. validate the exact emitted WGSL subset.

The emitter maps values to expressions or compiler locals, places to access
expressions, structured regions to WGSL control, buffers to group/binding
globals, and Tach intrinsics/effects to their WGSL operations.

Fixed-size buffers use compiler-generated wrappers so their binding size
follows the Tach ABI. Runtime arrays retain their natural element stride. Each
kernel with plain values gets one private uniform-address-space block, and the
entry reconstructs logical values before executing Core instructions.
Generated private names are isolated from the public entry point.

The WGSL validator reparses what Tach emits. It protects serialization shape
and syntax; semantic guarantees already come from typed, verified Core IR.
Browser integration tests provide the independent WebGPU implementation check.

## 9. SPIR-V backend

`src/spirv` emits SPIR-V 1.3 for the Logical addressing model with the Shader
capability and GLSL.std.450 where required. It owns:

- word encoding and result-ID allocation;
- logical and physical type lowering;
- entry-point interface variables and decorations;
- structured selection and loop CFG;
- dominance-correct phi construction;
- buffer and parameter-block variables, access chains, loads, and stores;
- atomics, scopes, memory semantics, and barriers; and
- extended math instructions.

Host-visible parameter and StorageBuffer aggregates use decorated physical ABI
types. SSA values and Workgroup memory use undecorated logical types. Buffer
aggregate loads/stores and parameter reconstruction cross this boundary field
by field, so padding and physical bool words never become logical values.

Tach promises zeroed workgroup memory. Because native SPIR-V cannot assume the
same initialization behavior as WGSL, the backend emits a prologue in which
the first local invocation stores zero values, then the workgroup synchronizes
before source code.

The in-tree validator independently decodes emitted bytes and checks header,
section order, IDs, types, capabilities, layouts, decorations, function CFG,
predecessors, dominance, phi edges, structured merges, memory operations,
atomics, barriers, intrinsics, and the exact used input interface. The native
harness additionally runs Khronos `spirv-val` before Vulkan execution.

## 10. Binding generation

`src/bindings` consumes the optimized module, emitted WGSL, and canonical
layout. It emits three synchronized artifacts:

- an ES module embedding WGSL and private module descriptors;
- TypeScript type aliases and positional kernel signatures; and
- plain JSON reflection metadata.

Named Tach structs become readonly TypeScript object aliases. Each exported
kernel becomes a same-named function returning an opaque `ComputeCommand`.
Buffer parameters use `ComputeBuffer<T>`; plain parameters stay ordinary typed
values.

The generated JavaScript contains no adapter, device, buffer, cache, or queue
lifecycle. It imports the private executor from `@depths/tach/internal` and
describes the compiled module. `ValidateGenerated` checks the correspondence
among JavaScript exports, declarations, metadata, parameter mappings, and
package imports before compilation succeeds.

## 11. WebGPU runtime

The runtime in `tach-ts/src` has one session implementation and two ownership
forms:

```text
tach(callback)       scoped ownership; wait and close on callback exit
tach(options?)       caller ownership; explicit idle and close
```

Both return or expose the same `Tach` session contract:

- acquire one adapter and device;
- create session-owned `ComputeBuffer` handles;
- construct opaque generated commands;
- submit one or more commands in order;
- synchronize through `idle()` or buffer readback; and
- destroy resources at the ownership boundary.

### Command construction and submission

A generated kernel call validates argument shape, buffer ownership, and
buffer non-aliasing, snapshots its plain arguments into at most one parameter
block, and returns a command. A thenable trap reports accidental
`await kernel(...)`; commands must go through `gpu.submit(...)`.

`submit(first, ...rest)` prepares all commands, records them in order into one
compute pass and command buffer, and performs one queue submission. Its promise
resolves after recording/submission, not after the GPU becomes idle. Submission
calls are serialized through a session tail so JavaScript concurrency cannot
reorder them.

The optional `repeat` count repeats a command in the same pass. One-
dimensional launch size can be inferred from the first runtime-sized storage
buffer; otherwise the default is exactly one workgroup. Explicit sizes must
match the kernel rank and contain positive safe integers.

Batching commands into one pass removes host submission boundaries but does not
fuse shader bodies. The runtime preserves every dispatch boundary because it
cannot prove cross-invocation memory independence. True fusion belongs in the
compiler after Core IR dependence analysis, never in a host option that guesses.

### Residency and caching

`gpu.buffer(value)` initially keeps a structured clone on the host. The first
submitted command supplies its compiler codec and layout, packs it, and creates
the physical GPU buffer. From then on, `write` must keep the same byte length.

Generated modules cache one shader module and each compute pipeline per device.
Sessions cache bind groups by layout and resident buffer ranges. Parameter
blocks use a session-owned aligned upload buffer that grows geometrically and
is reused under WebGPU queue ordering. Dynamic offsets let different parameter
snapshots reuse one bind group for the same resident buffers. Persistent
sessions therefore retain the device, buffers, pipelines, layouts, and stable
binding sets across frames.

### Synchronization and errors

`submit()` intentionally does not stall for completion. `idle()`, a materialized
buffer's `read()`, and the end of `tach(...)` are synchronization boundaries.

WebGPU error scopes, uncaptured errors, device loss, packing failures,
lifecycle errors, and application exceptions are normalized to `TachError`.
Both `tach(...)` overloads and live-session methods return values on success and
throw or reject with `TachError` on failure.

`close()` is idempotent. It destroys every live session buffer, the parameter
arena, caches, and the WebGPU device. Buffer `destroy()` is also idempotent and
invalidates cached bind groups that referred to it.

## 12. Validation boundaries

Each validator answers a different question:

| Boundary | Question |
|---|---|
| parser | Is the source grammatical? |
| semantic analysis | Is it a valid, typed Tach program? |
| Core IR verifier | Is lowered meaning internally sound? |
| optimizer post-verify | Did optimization preserve that contract? |
| WGSL validator | Did Tach serialize its WGSL subset correctly? |
| SPIR-V validator | Is the binary structurally, semantically, and ABI valid? |
| generated validator | Do JS, declarations, metadata, buffers, and values agree? |
| browser harness | Does real WebGPU compile and execute the artifacts? |
| native harness | Does `spirv-val` and Vulkan execute the SPIR-V correctly? |

Internal validation catches compiler faults close to their owner. External
harnesses catch incorrect assumptions about real implementations.

## 13. Package responsibilities

```text
main.go          CLI parsing and artifact writing
src/source       source positions and spans
src/lexer        tokens
src/ast          source-shaped syntax tree
src/parser       grammar
src/types        logical type model and domain predicates
src/sema         language semantics and Core IR lowering
src/ir           IR model, dump, verification, uniformity, use counts
src/opt          target-independent optimization
src/backend      shared backend-lowered coordinate analysis
src/layout       canonical buffer and physical field layout
src/abi          external names, private mangling, parameter-block planning
src/wgsl         WGSL lowering, target pass, emission, validation
src/spirv        SPIR-V lowering, emission, decoding, validation, disassembly
src/bindings     metadata and JavaScript/TypeScript generation
src/compiler     end-to-end orchestration
tach-ts/src      compiler delivery, WebGPU execution, errors, public exports
```

The dependency direction should follow that list toward orchestration. A
front-end package does not import a backend to learn semantics; the runtime
does not reverse-engineer generated shaders.

## 14. Test architecture

Tests are layered to match ownership:

- lexer, parser, semantic, IR, optimizer, layout, binding, and backend unit
  tests cover local contracts and rejection cases;
- mutation tests corrupt emitted SPIR-V and require the validator to reject it;
- compiler tests assert cross-artifact results;
- compiler tests compile every `tach` fence in these six maintained guides;
- `browser-test` compiles and runs the example corpus through generated
  TypeScript/WebGPU bindings;
- `spirv-test` compiles the same corpus, validates it externally, creates a
  native Vulkan pipeline, and compares readback; and
- `showcase-ts` sustains diverse workloads long enough to measure the runtime
  and compiler rather than one-off adapter or dispatch overhead.

Any change to a shared semantic path should be tested at the lowest owning
layer and then through both executable backends when it affects emission or
runtime behavior.

## 15. How a feature should enter Tach

Before adding syntax, locate the first layer that actually needs to change:

1. If an existing expression or library pattern already expresses the
   operation, no language feature is needed.
2. If it is only source convenience, lower it into existing Core IR.
3. If it adds portable semantics, add the smallest logical type/instruction,
   verifier rule, effect rule, and tests, then implement both backends.
4. If it is merely a target representation improvement, keep it in backend
   lowering/optimization and do not expose it to source or Core IR.
5. If it changes bytes, bindings, launch, names, or lifetime, update the one
   ABI owner and every consumer together.

This keeps the user-facing language coherent while leaving optimization and
hardware adaptation where they belong: inside the compiler.
