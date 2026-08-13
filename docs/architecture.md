# Tach compiler and runtime architecture

Tach has two target-independent IRs, two target executable plans, two shader
emitters, one canonical host layout, and one managed WebGPU runtime. The split
keeps baseline syntax small while giving explicit multi-dispatch programs a
real representation instead of hiding orchestration in generated glue.

For exact source rules, read [the language guide](language.md). For internal
data models, read [the IR guide](ir.md). For bytes and host execution, read
[the ABI guide](abi.md).

The shortest accurate model is:

```text
Kernel IR owns per-invocation portable meaning
Flow IR owns public programs and dispatch dependencies
target plans own physical kernels, bindings, scratch, and barriers
emitters own WGSL/SPIR-V representation
bindings and runtimes own the host boundary
```

## 1. Design invariants

Tach is organized around these rules:

1. Source describes portable algorithms and programs, never provider objects.
2. Logical coordinates are explicit values with source-chosen names.
3. Immutable values and addressable GPU places are different IR concepts.
4. Structured control remains structured until a backend needs a CFG.
5. Public resources and their versions are explicit across dispatches.
6. General Kernel IR optimization is separate from target planning.
7. Physical kernels are target-owned clones, not public API identities.
8. Layout, parameter packing, metadata, and runtimes share one ABI plan.
9. Every emitted artifact is validated at its owning boundary.
10. Dispatch boundaries are preserved unless a separately proved
    optimization changes them; inter-stage fusion is not currently present.

These rules let `export function scale[i]` stay a one-screen beginner feature
while the compiler internally represents its public command, logical stage,
private shader entry, launch source, and host resources separately.

## 2. End-to-end pipeline

```text
.tach source
    |
    v
lexer -> parser -> source-shaped AST
                    |
                    v
              semantic checking
               /            \
              v              v
       Kernel IR          Flow IR
  helpers + stages    programs + resources
       verify          + versions + shapes
              \              /
               v            v
          target-neutral Kernel IR optimization
                         |
              +----------+----------+
              |                     |
              v                     v
       Web target planning    SPIR-V target planning
       + physical kernels     + physical kernels
       + transients           + transients/barriers
       + coordinate pass      + coordinate pass
              |                     |
              v                     v
         WGSL emission         SPIR-V emission
         + validation          + binary validation
                                    |
                                    v
                               disassembly
              \                     /
               v                   v
          JS + TypeScript + metadata
          + generated-contract validation
```

`src/compiler.CompileTarget` orchestrates the pipeline. The front end and
target-neutral optimizer always run. `web` builds only the Web executable and
bindings, `spirv` builds only SPIR-V plus metadata, and `all` builds both plans
and the diagnostic dump/disassembly.

No consumer reparses another artifact to recover meaning. Bindings consume IR
and executable plans, not WGSL. SPIR-V disassembly decodes emitted bytes, not
emitter state.

## 3. Front end

### Source, lexer, parser, and AST

`src/source` owns positions, spans, and source errors. `src/lexer` owns Unicode
identifiers, strings used by `@docs`, suffix-free numbers, line comments,
punctuation, and operators.

`src/parser` builds `src/ast`. The AST retains source roles:

- module and declaration attributes;
- types and fields;
- helpers, indexed stages, and exported functions;
- ordinary statements and structured control;
- `run` domains and `transient<T>(length)` expressions; and
- exact source spans.

The parser decides grammar only. It does not resolve types, decide whether an
export is baseline sugar or explicit orchestration, infer resource access, or
assign target representation.

### Semantic analysis

`src/sema` is the language authority. Its order matters:

1. normalize and validate structured documentation;
2. collect type names, resolve fields, and reject layout-invalid cycles/tails;
3. collect function signatures and roles;
4. lower helpers and indexed stages to Kernel IR;
5. infer buffer mutability from effects;
6. reject helper recursion and verify Kernel IR;
7. lower each exported function to a Flow IR public program; and
8. verify the complete Flow IR module.

Lowering happens during checking because concrete choices are coupled to
meaning. A literal needs a resolved type before it becomes a constant; a
buffer projection becomes a typed place; local rebinding becomes structured
SSA results; a `run` buffer becomes a resource/version edge; and a program
shape becomes a checked host-evaluable expression.

## 4. Two target-independent IRs

### Kernel IR: per-invocation semantics

`src/ir` represents helpers and indexed stages. Its core distinction is:

```text
Value<T>       immutable computed data
Place<T>       typed path to buffer or workgroup memory
Block          structured instruction region with a terminator
```

Kernel IR contains resolved logical Tach types, helper calls, intrinsics,
loads/stores, atomics, barriers, workgroup declarations, and structured
`If`/`Loop`/`Scope` regions. Logical index parameters are ordinary `uint32`
values. Bindings, descriptor sets, padding, target pointer types, and builtin
variable names do not appear.

### Flow IR: public program semantics

`src/flow` represents host-callable work around indexed stages:

```text
Program
  public parameters
  external and transient resources
  resource versions
  shape expression DAG
  ordered dispatches
```

An exported indexed function synthesizes one Flow program with one launch-axis
shape and one dispatch. An exported unindexed function lowers its source
`const`, `transient`, and `run` declarations directly.

Resource versions state dataflow explicitly. A mutable dispatch consumes one
version and produces another. The verifier knows whether an initial or
transient version is defined and rejects a read before a complete definition.
Shapes refer to public uint values, runtime lengths, launch axes, constants,
or checked arithmetic.

This layer is why bindings can expose one command for a multi-stage operation
without encoding a dispatch graph by hand in TypeScript.

## 5. Verification and analysis

`ir.Verify` checks the complete Kernel IR contract: unique IDs, structured
availability, exact types, function roles, buffer/place paths, access rights,
control results, loop carriers, runtime arrays, intrinsics, atomics, barriers,
workgroup memory, and returns.

`src/ir/uniformity.go` derives whether values and control are uniform within a
workgroup. Constants and plain stage values are uniform; coordinates, mutable
loads, and atomic results are varying. The verifier rejects barriers nested
under varying control.

`flow.Verify` first verifies its Kernel IR, then checks public program names,
parameters, resources, dense IDs, version chains, shape DAGs, stage references,
argument types, resource non-aliasing, definition-before-read, and final
versions.

Verification is a production boundary. Semantic analysis, optimization,
backend lowering, emission, and binding generation do not accept malformed IR
as a recoverable variant.

## 6. Target-neutral optimization

`src/opt.OptimizeLogical` currently optimizes the module's Kernel IR and then
re-verifies the Flow module. `OptimizeKernel` applies, per function:

1. common value/place elimination;
2. loop-invariant code motion;
3. commoning again;
4. conservative loop buffer-value promotion;
5. final commoning; and
6. dead pure definition removal to a fixed point.

The passes know Tach types, structured dominance, helper purity, access
inference, and effects. They have no WGSL or SPIR-V cases.

Loop buffer promotion can replace a repeated load/update/store with a lazy
loop-carried value and one final write. It stops at ambiguous access,
synchronization, atomics, early exits, competing touches, or internally
defined places. Zero-trip behavior remains unchanged.

Flow IR currently has no rewrite pipeline beyond construction-time shape
interning and verification. Dispatch planning occurs per target. In
particular, no pass fuses distinct Flow dispatches.

## 7. Target executable planning

`src/backend.Lower` clones the optimized logical module for one target profile
and creates an `Executable` containing:

- a target-private Kernel IR module;
- one `PhysicalKernel` per program dispatch;
- one `ProgramPlan` per public program; and
- target identity and limits.

For every dispatch, planning:

1. clones its logical stage;
2. optionally internalizes safe command repetition;
3. removes unused value parameters;
4. assigns a private `_tach_kN` entry;
5. chooses or verifies a portable workgroup;
6. assigns dense storage bindings;
7. builds the shared ABI parameter block;
8. lowers logical coordinate requirements; and
9. applies coordinate optimization.

The target module also contains cloned helpers. Physical stage clones can
differ because program arguments, repeat mode, parameter pruning, or later
target passes can differ even when their source stage is shared.

The current one-kernel-per-dispatch policy is deliberately simple and
deterministic. The implementation carries a `DECISION` comment: deduplication
requires a verified Kernel IR identity and must earn itself through real
shader-size or pipeline evidence.

### Repeat internalization

A one-dispatch plan may move `repeat` inside each invocation when the stage has
no loop or synchronization and every buffer access is exactly the current 1D
coordinate. Planning adds a repeat value parameter and wraps the stage body in
a Kernel IR loop. The public result remains equivalent because invocations do
not communicate across repetitions.

Other plans retain program repetition. This optimization removes dispatch
overhead without claiming general kernel fusion.

### Transients and barriers

Transient planning computes first/last dispatch use and greedily assigns an
allocation color not overlapping an earlier live range. The runtime allocates
one capacity per color.

SPIR-V plans insert barrier steps between adjacent dispatches when the first
writes a resource touched by the second, plus an optional barrier between
repeated program iterations. WebGPU records ordered dispatches in one compute
pass and relies on WebGPU's pass execution model.

## 8. Coordinate lowering

`src/backend/coordinates.go` begins with each named logical index mapped to a
global coordinate. It then recognizes exact workgroup-local expressions:

```text
coordinate % matchingWorkgroupDimension
localX + localY * width + localZ * width * height
```

These can become target local-coordinate or local-linear inputs. Unused target
inputs and now-dead arithmetic disappear. If an expression is not an exact
match, it remains ordinary arithmetic over the global logical coordinate.

Provider builtin names exist only in target lowering and emission, never in
source, Flow IR, or logical Kernel IR.

## 9. Layout and parameter ABI

`src/layout` computes the canonical host-visible representation: scalar/vector
size and alignment, 16-byte struct alignment, field offsets, nested extents,
array strides, runtime tails, and checked 32-bit sizes.

`src/abi` owns private entry names and immutable value blocks. It flattens each
remaining physical-stage value parameter into numeric leaves, replacing bool
leaves with physical `uint32`, then applies the same layout engine. Storage
bindings precede the optional parameter block in group/set `0`.

The plan drives:

- WGSL storage wrappers and uniform blocks;
- SPIR-V offsets, strides, descriptors, and physical aggregate types;
- generated target metadata;
- TypeScript buffer codecs and parameter packing; and
- the native Vulkan harness.

No target independently recalculates ABI offsets.

## 10. WGSL backend

`src/wgsl` receives a verified Web `Executable` and:

1. indexes physical and helper function coordinate requirements;
2. emits structs, resources, parameter blocks, helpers, and private entries;
3. maps structured Kernel IR directly to structured WGSL; and
4. reparses the exact generated WGSL subset with its in-tree validator.

Fixed resources use aligned wrappers; runtime tails retain natural stride.
Plain values are reconstructed from one uniform block at entry. Values become
expressions or compiler locals; places become access expressions; `If` and
`Loop` stay structured.

The in-tree validator checks Tach's serialization contract. The Chromium
harness provides the independent implementation check.

## 11. SPIR-V backend

`src/spirv` emits SPIR-V 1.3 with Logical addressing, Shader capability, and
GLSL.std.450 where required. It owns result IDs, logical/physical types,
interface variables, decorations, structured CFG construction, phi nodes,
access chains, atomics, barriers, and extended instructions.

Host-visible StorageBuffer/uniform aggregates use decorated physical types.
SSA, helpers, and Workgroup memory use logical undecorated types. Field-wise
conversion prevents padding and physical bool words from entering value
semantics.

WGSL guarantees zeroed workgroup variables; native SPIR-V does not. The
SPIR-V emitter therefore generates a first-local-invocation zeroing prologue
and synchronization before source instructions.

The validator independently decodes the binary and checks header/sections,
IDs, capabilities, types, layouts, decorations, CFG, predecessors, dominance,
phi edges, structured merges, memory operations, atomics, barriers,
intrinsics, and used interfaces. `spirv-val` and Vulkan provide external
checks.

## 12. Bindings and documentation

`src/bindings.Generate` consumes the optimized `flow.Module`, requested target
executables, and WGSL text. It creates:

- schema-1 metadata with public programs and target plans;
- an ES module embedding WGSL and the Web plan; and
- declarations for source types and exported programs.

Generated JavaScript imports only `defineModule` from
`@depths/tach/internal`. Public applications import `tach` and `TachError` from
`@depths/tach`. `ValidateGenerated` checks exports, declarations, metadata,
layouts, references, and imports before compilation succeeds.

Documentation follows a deliberately acyclic path:

```text
@docs source
  -> semantic Documentation model
  -> target-neutral JSON description from Go
  -> Markdown rendering and TypeScript syntax in tach-ts
```

The Go compiler describes Tach types, functions, coordinates, buffers,
access, and returns without importing or spelling TypeScript. `tach-ts` owns
JSDoc/Markdown presentation and the generated usage sample.

## 13. WebGPU runtime

The runtime in `tach-ts/src` has one `Session` and two ownership forms:

```text
tach(callback)       scoped: wait and close on exit
tach(options?)       persistent: caller invokes idle/close
```

Generated `defineModule` consumes the embedded shader, public-program table,
and Web plan. A generated function validates public buffers and options, then
creates an opaque `ComputeCommand`. Preparation materializes buffers,
evaluates shapes, sizes transients, compiles referenced pipelines, and packs
parameter blocks.

`submit(first, ...rest)` serializes submission calls, prepares commands,
records every program step in argument order into one compute pass and command
buffer, and queues once. It does not wait for device completion.

The session owns:

- resident `ComputeBuffer` state and codecs;
- per-device generated shader modules and pipelines;
- cached bind groups by layout and resource range;
- a geometrically growing aligned uniform arena;
- geometrically growing scratch buffers by transient color;
- retired buffers awaiting a safe `idle()` boundary; and
- error scopes, uncaptured errors, and device-loss state.

`idle()`, materialized readback, and scoped-session exit wait for completion.
`close()` and buffer `destroy()` are idempotent.

## 14. Validation boundaries

| Boundary | Question |
|---|---|
| lexer/parser | Is source spelling and grammar valid? |
| semantic analysis | Do declarations and expressions have valid Tach meaning? |
| Kernel IR verifier | Are per-invocation values, places, control, and effects sound? |
| Flow verifier | Are programs, shapes, resources, versions, and dispatches sound? |
| optimizer post-verify | Did rewrites preserve both IR contracts? |
| backend verifier | Are physical kernels and target plans internally consistent? |
| WGSL validator | Did Tach serialize its supported WGSL shape correctly? |
| SPIR-V validator | Is the binary structurally, semantically, and ABI valid? |
| generated validator | Do metadata, JS, declarations, and plans agree? |
| browser harness | Does Chromium WebGPU compile and execute generated modules? |
| native harness | Do `spirv-val` and Vulkan execute target plans correctly? |

Each catches faults at the earliest owner; external harnesses challenge the
compiler's assumptions against real implementations.

## 15. Package responsibilities and dependencies

```text
src/source       spans and source errors
src/lexer        tokens
src/ast          source-shaped nodes
src/parser       grammar
src/types        logical types and domain predicates
src/ir           Kernel IR, verification, access, uniformity, use counts
src/flow         Flow IR, resource versions, shapes, verification
src/sema         language checking and both IR lowerings
src/opt          target-neutral Kernel IR optimization
src/layout       canonical host-visible layout
src/abi          private names and parameter blocks
src/backend      target executable planning and coordinate lowering
src/wgsl         WGSL emission and validation
src/spirv        SPIR-V emission, decoding, validation, disassembly
src/bindings     metadata, generated modules/declarations, descriptions
src/compiler     end-to-end compiler API and artifact writing
main.go          native CLI
tach-ts/src      compiler delivery, docs renderer, WebGPU runtime
```

The dependency graph points from primitive semantics toward orchestration.
The front end never imports a backend; Go never imports the TypeScript renderer;
the runtime never reverse-engineers a shader. `go list -deps ./...` is part of
the repository's cycle check.

## 16. Test architecture

Tests mirror ownership:

- lexer, parser, semantic, Kernel IR, Flow IR, optimizer, layout, backend,
  binding, and emitter tests cover local contracts and rejection cases;
- compiler tests check target artifact sets, deterministic cross-target
  output, documentation descriptions, baseline desugaring, and all maintained
  examples;
- SPIR-V mutation tests corrupt valid modules and require rejection;
- `browser-test` compiles all examples and checks generated WebGPU execution;
- `spirv-test` runs the same corpus through external validation and Vulkan;
- `showcase-ts` runs five large two-stage GPU workloads and a separate
  medium GPU-versus-single-threaded-TypeScript profile; and
- `dupl` and `deadcode` provide structural duplication and reachability audits
  beyond behavioral tests.

A shared semantic change belongs first in the lowest owning test and then in
both executable harnesses when it affects target behavior.

## 17. Extension rule

Before adding a feature, find its first real owner:

1. Existing syntax or a helper expresses it: add nothing.
2. Source convenience has existing semantics: lower to existing IR.
3. New per-invocation portable meaning: extend Kernel IR, verification,
   effects, optimization, and both emitters.
4. New public dispatch/resource meaning: extend Flow IR, verification, target
   plans, metadata, and both runtimes.
5. Target representation improvement: keep it in backend planning/emission.
6. Byte, binding, launch, name, or lifetime change: update the single ABI
   owner and every generated/native consumer together.

This keeps beginner syntax ergonomic without pretending orchestration,
parallel memory, or provider representation is simpler than it is.
