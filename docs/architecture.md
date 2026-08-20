# Tach compiler and runtime architecture

This is how Tach is built, not how to write a first kernel. If you want to
author GPU work from TypeScript, start with the [root README](../README.md),
the [language guide](language.md), the [examples](../examples/README.md),
and the [TypeScript guide](../tach-ts/README.md). Come here when you need
to know why a view is not a canvas, why WebGPU and Vulkan share one pack
but not one container, or where a change belongs.

Tach has one project compiler, two target-independent intermediate
representations, two target executable plans, two shader emitters, one
canonical host layout, and one managed runtime with WebGPU and Tach-owned
Vulkan drivers. The split keeps beginner syntax small: imports, multi-step
programs, and display views have real compiler objects instead of living
as generated glue.

For exact source rules, read [the language guide](language.md). For internal
data models, read [the IR guide](ir.md). For bytes and host execution, read
[the ABI guide](abi.md).

The shortest accurate model is:

```text
project loading owns filesystem identity, imports, DAGs, and canonical order
Kernel IR owns per-invocation portable meaning
Flow IR owns public programs, dispatch dependencies, and terminal views
target plans own physical kernels, bindings, scratch, barriers, and view output
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
10. Flow dispatch order and boundaries are explicit inputs to target planning.
11. One project-global declaration namespace feeds one merged IR and one
    cohesive dual-host artifact set.
12. Parallel scheduling may change elapsed time, never diagnostics or bytes.
13. A display view is terminal Flow meaning; texture or packed-buffer output
    is a target choice.
14. Buffers are session-owned state; generated commands are opaque recipes
    whose usable session is determined only by the buffers they reference.

These rules let `export function scale[i]` stay a one-screen beginner feature
while the compiler internally represents its public command, logical stage,
private shader entry, launch source, and host resources separately.

## 2. End-to-end pipeline

```text
nearest tach.json
    |
    v
strict manifest + canonical one-tier source discovery
    |
    v
concurrent lexer/recovering parser -> one AST per kernel
    |
    v
imports + global names + kernel DAG + collapsed module DAG
    |
    v
global headers -> direct-import semantic environments
    |
    +-------------------------+
    |                         |
    v                         v
Kernel IR                  Flow IR
helpers + stages      programs + resources
    verify            + versions + shapes/views
    |                         |
    +------------+------------+
                 v
     canonical project merge and optimization
                 |
       +---------+---------+
       |                   |
       v                   v
 WebGPU planning       Vulkan/SPIR-V planning
 + physical kernels    + physical kernels
 + transients          + transients/barriers
 + coordinate pass     + coordinate pass
       |                   |
       v                   v
 one WGSL module        one SPIR-V 1.6 module
 + validation           + binary validation
       +---------+---------+
                 v
 schema-2 runtime metadata + schema-2 project description
                 |
                 v
 TypeScript JS/declaration/Markdown/package rendering
 + sibling shader URLs + contract validation
                 |
                 v
 atomic build replacement
```

`src/compiler.Build`, `Check`, and `Describe` are the three internal Go entry
points used by the private native engine. They share project discovery and the
same semantic pipeline. `Build` and `Check` lower and validate both targets;
only `Build` stages their artifacts. `Describe` stops before optimization,
target lowering, and execution-metadata emission after producing a trustworthy
documentation model.

No consumer reparses another artifact to recover meaning. Bindings consume IR
and executable plans, not WGSL. The SPIR-V validator and diagnostic decoder
consume emitted bytes, never private emitter state. Ordinary builds discard
the two private JSON descriptions; `tach build --verbose` relocates them and
the IR/plan/disassembly views under `build/diagnostics/`.

## 3. Front end

### Source, lexer, parser, and AST

`src/compiler/project.go` first canonicalizes the nearest project root, parses
the strict manifest, discovers exactly `<module>/<kernel>.tach`, rejects
misplaced/case-colliding/physically duplicated sources, and assigns canonical
forward-slash identities. It resolves imports without concatenating or
rewriting source, checks project-global declaration uniqueness, and validates
both dependency DAGs.

`src/source` owns positions, primary and related spans, ordered diagnostic
sets, and rendering inputs. `src/lexer` owns Unicode identifiers, strings used
by imports and `@docs`, suffix-free numbers, preserved line-comment trivia,
punctuation, and operators. Lexing advances after invalid UTF-8 input and
returns all independently recoverable lexical diagnostics.

`src/parser` builds `src/ast`. The AST retains source roles:

- file and declaration attributes;
- explicit whole-file imports;
- types and fields;
- helpers, indexed stages, and exported functions;
- ordinary statements and structured control;
- `run` domains, `transient<T>(length)` expressions, and terminal
  `view<srgb8>` returns; and
- exact source spans and comment trivia needed by formatting.

The parser recovers at declaration and statement boundaries so one broken
kernel does not suppress diagnostics from siblings. It decides grammar only. It does not resolve types, decide whether an
export is baseline sugar or explicit orchestration, infer resource access, or
assign target representation.

### Semantic analysis

`src/sema` is the language authority. Project-global names are collected
before local interfaces or bodies, while each source file receives only its
own and directly imported declarations. Its order matters:

1. normalize and validate structured documentation;
2. collect type names, resolve fields, and reject layout-invalid cycles/tails;
3. collect function signatures and roles;
4. lower helpers and indexed stages to Kernel IR;
5. infer buffer mutability from effects;
6. reject helper recursion and verify Kernel IR;
7. lower each exported function, including any terminal view, to a Flow IR
   public program; and
8. verify the complete Flow IR module.

Parsing, type-field resolution, signature checking, function lowering, and
program lowering use bounded goroutines. Results occupy canonical input slots
and merge only after workers finish, so map iteration and completion order
cannot affect diagnostics or artifacts. One worker and `GOMAXPROCS` workers
are byte-for-byte regression-tested; the entire suite also runs under the Go
race detector.

Lowering happens during checking because concrete choices are coupled to
meaning. A literal needs a resolved type before it becomes a constant; a
buffer projection becomes a typed place; local rebinding becomes structured
SSA results; a `run` buffer becomes a resource/version edge; and a program
shape becomes a checked host-evaluable expression.

Numeric inference has one expression-local authority here. It resolves all
operands of an operator or intrinsic together using explicit types, expected
context, concrete siblings, intrinsic domains, then defaults. This makes
operand order irrelevant without importing later-use, whole-program, host, or
backend knowledge. Inferred `vec(...)` construction and scalar broadcast are
then lowered to ordinary typed composites.
Plain and compound assignment enter this same resolver; assignment syntax does
not carry a second operand-typing algorithm.

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

Inference is absent from this IR: each value already has one concrete type.
`vec(...)` and scalar broadcast require no new operation because semantic
lowering expresses both with the existing `Composite` instruction.
Intrinsic signatures live with their Kernel IR kinds and are consumed by both
semantic lowering and IR verification, so admissible types have one authority.

`Continue` and `Break` terminators carry the loop values for their exact CFG
edge, so early transfer remains valid SSA rather than source-level control that
a backend must rediscover. `fma` remains one target-neutral typed intrinsic.

### Flow IR: public program semantics

`src/flow` represents host-callable work around indexed stages:

```text
Program
  public parameters
  external and transient resources
  resource versions
  shape expression DAG
  ordered dispatches
  optional terminal view
```

An exported indexed function synthesizes one Flow program with one launch-axis
shape and one dispatch. An exported unindexed function lowers its source
`const`, `transient`, and `run` declarations directly. A `view<srgb8>` program
also records the final runtime `vec<float32, 4>` resource version and checked width
and height shapes. It may have no external resource when it constructs the
complete frame in a transient.

Resource versions state dataflow explicitly. A mutable dispatch consumes one
version and produces another. The verifier knows whether an initial or
transient version is defined and rejects a read before a complete definition.
Shapes refer to public uint values, runtime lengths, launch axes, constants,
or checked arithmetic. Views reuse the same shapes; they do not introduce a
second extent language.

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
versions. For a view it additionally proves the format, source element type,
exact final defined version, and width/height shapes.

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
particular, no pass fuses distinct Flow dispatches. Terminal view projection
may be folded into the final dispatch under a separate exact proof because it
is target representation of the view, not inter-stage fusion.

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
9. applies coordinate optimization and optional terminal-view lowering.

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

Other plans retain program repetition. The internalized case removes repeat
dispatch overhead while preserving the proved Flow semantics.

### Transients and barriers

Transient planning computes first/last dispatch use and greedily assigns an
allocation color not overlapping an earlier live range. The runtime allocates
one capacity per color.

SPIR-V plans insert barrier steps between adjacent dispatches when the first
writes a resource touched by the second, plus an optional barrier between
repeated program iterations. WebGPU records ordered dispatches in one compute
pass and relies on WebGPU's pass execution model.

### Terminal views

Every view plan records its checked extent, output allocation color and
binding, terminal projection step, and whether projection was fused. Planning
folds conversion into the final stage only when a transient is written
completely at the exact current 1D coordinate, the domain equals the transient
length and `width * height`, and no earlier use requires the result. The
transient and standalone projection then disappear.

All other valid views receive one target-owned projection kernel that reads
the final float pixel resource. Both targets first lower each pixel to one
packed RGBA8 `uint32` word: IEC sRGB on RGB, clamp-only alpha, then
`uint32(channel * 255 + 0.5)` with R, G, B, A in low-to-high bytes. WGSL
unpacks that word with `unpack4x8unorm` into an `rgba8unorm` storage texture so
`present` can write a 2D image. SPIR-V stores the word in packed scratch. View commands cannot
use repeat; repeating a display recipe has no useful externally visible
intermediate result and complicates the terminal resource contract.

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
array strides, runtime tails, and checked 32-bit sizes. SPIR-V Workgroup and
Input `Aligned` operands use logical pointee alignment instead of this floor.

`src/abi` owns private entry names and immutable value blocks. It flattens each
remaining physical-stage value parameter into numeric leaves, replacing bool
leaves with physical `uint32`, then applies the same layout engine. Storage
bindings precede the optional parameter block in group/set `0`.

The plan drives:

- WGSL storage wrappers and uniform blocks;
- SPIR-V offsets, strides, descriptors, and physical aggregate types;
- generated target metadata;
- TypeScript buffer codecs and parameter packing; and
- the Tach-owned Vulkan runtime.

No target independently recalculates ABI offsets.

Binary16 follows this same path: the logical type remains `float16`, canonical
host layout assigns 2-byte scalar storage and vector-derived alignment, and
metadata carries `f16`/exact `f16Bits`. The TypeScript codec uses
`Float16Array` and little-endian `DataView` binary16 operations rather than
widening the storage contract. The only provider alignment wrinkle is a scalar
`float16[]` whose logical byte extent is not divisible by four, either directly
or after a struct prefix. WebGPU/Vulkan transfers need four-byte capacity and
WebGPU buffer bindings need a four-byte size. Drivers pad physical capacity,
while target planning injects the metadata-derived logical length and
runtime-tail path for source `.length`. Both lowerings consume that exact value,
so physical padding cannot become a phantom logical element.

## 10. WGSL backend

`src/wgsl` receives a verified Web `Executable` and:

1. indexes physical and helper function coordinate requirements;
2. emits structs, resources, parameter blocks, helpers, and private entries;
3. maps structured Kernel IR directly to structured WGSL, including the
   shared view-pack helper, carrier assignments before `break`/`continue`, and
   the WGSL `fma` builtin;
4. stores a fused or standalone view by unpacking the packed word with
   `unpack4x8unorm` into an `rgba8unorm` storage texture; and
5. reparses the exact generated WGSL subset with its in-tree validator.

Fixed resources use aligned wrappers. A direct runtime array uses a
natural-alignment wrapper; a runtime-tail struct is the storage root in WGSL
and the decorated `Block` in SPIR-V, because neither representation may nest it
inside another resource struct.
Plain values are reconstructed from one uniform block at entry. Values become
expressions or compiler locals; places become access expressions; `If` and
`Loop` stay structured.

The in-tree validator checks Tach's serialization contract. The Chromium
harness provides the independent implementation check.

If Kernel IR contains `float16`, emission adds `enable f16;` and target
metadata requires `shader-f16`. The browser host requests this optional feature
when the adapter offers it and rejects preparation otherwise. Float32-only
projects emit neither the directive nor the requirement.

## 11. SPIR-V backend

`src/spirv` emits SPIR-V 1.6 for the Vulkan 1.3 floor with Logical addressing,
Shader plus VulkanMemoryModel capabilities, the Vulkan memory model, and
GLSL.std.450 math where required. It owns result IDs,
logical/physical types,
interface variables, decorations, structured CFG construction, phi nodes,
access chains, atomics, barriers, and extended instructions.

Early loop transfers become edges to the structured continuation or merge
block and contribute their carried values to that block's `OpPhi` nodes.
`fma` becomes GLSL.std.450 `Fma`; neither mapping changes its target-neutral
source type contract.

Host-visible StorageBuffer/uniform aggregates use decorated physical types.
SSA, helpers, and Workgroup memory use logical undecorated types. Field-wise
conversion prevents padding and physical bool words from entering value
semantics.

For views, planning rewrites a proven terminal store, or adds a projection
entry, through the same pack sequence used by WGSL. SPIR-V stores that packed
`uint32` for the native runtime without importing browser texture semantics
into logical IR.

Tach requires Vulkan's `shaderZeroInitializeWorkgroupMemory` and
`vulkanMemoryModel` features. The SPIR-V emitter gives every Workgroup
variable an `OpConstantNull` initializer, so no synthetic invocation, store
loop, or barrier alters the source program. Shared loads and stores are NonPrivate. Aligned follows the host ABI on
Uniform/StorageBuffer and the logical pointee on Workgroup/Input.
Storage-buffer atomics use QueueFamily scope.
Synchronization2 is the inter-dispatch synchronization floor.

Binary16 emission uses 16-bit `OpTypeFloat`, exact binary16 constants, and
`OpFConvert` for explicit `float16`/`float32` conversion. The emitter derives
`Float16`, `StorageBuffer16BitAccess`, and
`UniformAndStorageBuffer16BitAccess` capabilities from actual IR/interface
use. The Vulkan host enables supported Vulkan 1.1/1.2 optional feature fields
once at device creation, while the native module wire carries the exact subset
each module requires. Unsupported Float16 work fails before shader-module
creation; ordinary modules retain the existing Vulkan 1.3 floor.

The validator independently decodes the binary and checks header/sections,
IDs, capabilities, types, layouts, decorations, CFG, predecessors, dominance,
phi edges, structured merges, memory operations, atomics, barriers,
intrinsics, and used interfaces. `spirv-val` and Vulkan provide external
checks.

## 12. Bindings and documentation

`src/bindings.Generate` consumes the optimized project-wide `flow.Module` and
both executable plans. It creates schema-2 metadata with public programs,
public view flags, buffer/texture binding kinds, and both target plans,
including each terminal view step and extent.

Runtime metadata and the project description currently both use version `2`,
but they are distinct closed protocols. Runtime metadata is embedded in
generated JavaScript and drives execution. The project description exists only
between the Go engine and TypeScript build/docs renderer. Consumers identify
them by their owning boundary and full validated shape, never by the number
alone.

The Deno-first TypeScript layer consumes that checked metadata plus the project
description and creates one ES module, one declaration module, and one package
manifest. The ES module embeds metadata and sibling URLs for both
`kernel.wgsl.gz` and `kernel.spv`; it does not embed or parse shader source.

Generated JavaScript imports only `defineModule` from
`@depths/tach/internal`. Public applications import `tach` and `TachError` from
`@depths/tach`. Metadata validation in Go and generation-time validation in
TypeScript check program counts, exports, declarations, layouts, plan
references, profile facts, and the exact package inventory before installation.

Documentation follows a deliberately acyclic path:

```text
@docs source
  -> semantic Documentation model
  -> target-neutral JSON description from Go
  -> Markdown rendering and TypeScript syntax in tach-ts
```

The Go compiler's schema-2 project description groups canonical kernels by
module and describes Tach types, function roles, coordinates, buffers, access,
returns, documentation, project identity, and JavaScript-package identity
without importing or spelling TypeScript. `tach-ts` owns JSDoc/Markdown presentation,
the generated usage sample, module-document filenames, and npm metadata.

Diagnostics cross the native/TypeScript boundary as one schema-1 JSON envelope
on the private compiler's stderr. This keeps stdout reserved for project and
runtime descriptions while allowing successful operations to carry warnings.
Each record has severity, stable code, byte offset plus line/column span,
message, captured source line, optional help, and related source locations.
The public TypeScript layer validates the envelope once. It then exposes the
same records through `ProjectResult.diagnostics`, `CompilerError.diagnostics`,
Markdown-like terminal rendering, or the public CLI's `--json` result. These
structured records are the sole cross-layer message model. Go error strings
remain an internal compiler and test representation; neither backend owns a
diagnostic model or public renderer.

### Artifact transaction

The public TypeScript CLI owns a uniquely named staging directory beside the
project's fixed `build/` child. The native engine writes both shader artifacts
and private project/runtime descriptions into the already-created empty stage.
TypeScript consumes those descriptions, adds the singular JS/declaration
facade, package manifest, and documentation, removes or relocates private
metadata, then checks the exact inventory before any final path changes.

Replacement renames the prior build to a sibling backup, renames the complete
stage into place, and removes the backup only after success. A failed compiler,
renderer, package validation, inventory check, or pre-commit filesystem action
leaves the previous complete build untouched. The docs-only route copies the
current build into a stage, replaces only README/module documentation, and
commits through the same boundary.

## 13. Unified runtime and host drivers

The runtime in `tach-ts/src` has one `Session` and two ownership forms:

```text
tach(callback)       scoped: wait and close on exit
tach(options?)       persistent: caller invokes idle/close
```

Generated `defineModule` consumes public programs, host layouts, two plans, and
two shader URLs. A generated function validates public buffers and options,
then creates an opaque `ComputeCommand`, or `ComputeView` for a view
program. Preparation materializes buffers, evaluates shapes, sizes transients
and view output, compiles referenced pipelines, and packs parameter blocks.

`prepare(first, ...rest)` compiles and validates recipes without dispatching.
`submit(first, ...rest)` serializes submission calls, prepares commands, and
passes host-neutral prepared commands to the selected driver in argument order.
It does not wait for device completion. Submitting a view runs its terminal
projection into driver-owned offscreen output without CPU readback.

Browser `present(canvas, view)` runs the same recipe directly into the current
same-sized WebGPU canvas texture and waits for submitted work. The wait gives a
CPU-driven frame loop bounded backpressure instead of allowing unbounded frame
queueing. The Vulkan driver executes views into packed scratch through
`submit`, but rejects `present` because Tach currently owns no native surface.

The shared session owns:

- host values and generated codecs;
- resident opaque driver-buffer handles;
- buffer ownership, command recipe state, and lifecycle state; and
- serialized submission order and deferred failures.

A `ComputeBuffer` belongs to exactly one session. A recipe captures only its
arguments and may be prepared or executed by any session that owns every
referenced buffer. Consequently a scalar-only view recipe is owner-neutral and
can be reused across sessions; a buffer-backed recipe is constrained by those
buffers without acquiring separate command ownership.

The WebGPU driver owns its adapter/device, WGSL fetch, shader modules,
pipelines, bind groups, aligned uniform arena, scratch by transient color,
retired buffers, error scopes, uncaptured errors, and device-loss state. The
Vulkan driver owns the packaged FFI library/session, SPIR-V modules, native
buffers/submissions, and error translation. Native code owns Vulkan 1.3 device
selection (Synchronization2, zero-initialized workgroup memory, Vulkan memory
model), device-local storage, staging transfers, lazy pipelines,
descriptor/command/fence pools, a mapped parameter arena, scratch, and
Synchronization2 barriers. Deno retains one process-wide native-library
handle because unloading a Go shared runtime is unsafe; logical Tach sessions
and all their GPU objects still close independently.

`idle()`, materialized readback, and scoped-session exit wait for completion.
`close()` and buffer `destroy()` are idempotent.

## 14. Validation boundaries

| Boundary | Question |
|---|---|
| project discovery | Is the nearest manifest strict and every source at one canonical module/kernel identity? |
| import graphs | Do targets exist, remain directly scoped, and form kernel and module DAGs? |
| lexer/parser | Is source spelling and grammar valid? |
| semantic analysis | Do declarations and expressions have valid Tach meaning? |
| Kernel IR verifier | Are per-invocation values, places, control, and effects sound? |
| Flow verifier | Are programs, shapes, resources, versions, dispatches, and terminal views sound? |
| optimizer post-verify | Did rewrites preserve both IR contracts? |
| backend verifier | Are physical kernels and target plans internally consistent? |
| WGSL validator | Did Tach serialize its supported WGSL shape correctly? |
| SPIR-V validator | Is the binary structurally, semantically, and ABI valid? |
| generated validator | Do metadata, view contracts, JS, declarations, and plans agree? |
| output transaction | Is the staged inventory complete, confined to this project's `build`, and replaceable as one unit? |
| browser harness | Does Chromium WebGPU compile and execute generated modules? |
| native harness | Do `spirv-val` and Vulkan execute target plans correctly? |

Each catches faults at the earliest owner; external harnesses challenge the
compiler's assumptions against real implementations.

## 15. Package responsibilities and dependencies

```text
src/source       spans and ordered diagnostics
src/lexer        tokens and preserved line-comment trivia
src/ast          source-shaped declarations and imports
src/parser       recovering grammar
src/types        logical types and domain predicates
src/ir           Kernel IR, verification, access, uniformity, use counts
src/flow         Flow IR, resource versions, shapes, verification
src/sema         language checking and both IR lowerings
src/opt          target-neutral Kernel IR optimization
src/layout       canonical host-visible layout
src/abi          private names and parameter blocks
src/backend      target executable planning, coordinate lowering, target profile
src/wgsl         WGSL emission and validation
src/spirv        SPIR-V emission, decoding, validation, and summaries
src/bindings     target metadata and target-neutral project descriptions
src/compiler     project discovery, DAGs, formatting, pipeline, native staging
main.go          private native machine-operation dispatcher
tach-ts/src      public API, shared session, WebGPU/Vulkan drivers,
                 compiler orchestration, output transaction, and docs
native           Tach-owned Vulkan 1.3 FFI implementation
```

The dependency graph points from primitive semantics toward orchestration.
The front end never imports a backend; Go never imports the TypeScript renderer;
the runtime never reverse-engineers a shader. `go list ./...` resolves the
complete repository package graph as part of the cycle check.

## 16. Test architecture

Tests mirror ownership:

- lexer, parser, semantic, Kernel IR, Flow IR, optimizer, layout, backend,
  binding, and emitter tests cover local contracts and rejection cases;
- compiler tests check strict manifests, one-tier discovery, import visibility,
  both DAGs, global names, error recovery, formatter transactions, exact unified
  artifact sets, wide one/many-worker determinism, complete multi-file error
  aggregation, all four function forms, documentation descriptions, and all
  maintained projects in canonical format;
- bounded fuzz properties challenge lexer progress and spans, parser recovery
  determinism, source-facing semantic failures, and formatter token
  preservation, reparsing, and idempotence;
- binding tests corrupt each runtime-plan seam, while TypeScript compiler tests
  challenge exact package shape and build/docs transaction rollback;
- SPIR-V mutation tests corrupt valid modules and require rejection;
- `browser-test` builds the example project once and checks every generated
  endpoint through its exact generated WGSL in WebGPU, including fused and
  fallback views, exact 8-bit swatch presentation, sustained CPU-selected
  canvas presentation, contextual numeric/vector inference, scalar broadcast,
  nearest-loop early exits and skips, FP16/FP32 `fma`, Float16
  math/storage/parameters, an odd direct f16 array, and
  a prefixed f16 runtime tail;
- `deno-test` independently builds the same example project, validates its
  SPIR-V for Vulkan 1.3, and runs every exported program through Deno/Vulkan,
  including fused/fallback offscreen projection, the same swatch pair,
  owner-neutral recipes, repeated logical sessions, and the same loop,
  contextual inference, multiply-add, and Float16 seams;
- `showcase-ts` builds eight workload kernels plus one shared color file and
  runs eleven host-neutral rendering, mathematical, and physics workloads
  through both WebGPU and Vulkan, including matched FP32/FP16 matrix,
  data-dependent complex recurrence, and arithmetic-dense oscillator pairs;
  browser renderers use direct canvas presentation while native renderers
  exercise packed view projection;
  and
- `dupl`, `deadcode`, and `staticcheck` provide structural duplication,
  whole-program reachability, and correctness audits beyond behavioral tests.

A shared semantic change belongs first in the lowest owning test and then in
both executable harnesses when it affects target behavior.

## 17. Extension rule

Before adding a feature, find its first real owner:

1. Existing syntax or a helper expresses it: add nothing.
2. Source convenience has existing semantics: lower to existing IR.
3. New per-invocation portable meaning: extend Kernel IR, verification,
   effects, optimization, and both emitters.
4. New public dispatch, resource, or terminal-result meaning: extend Flow IR,
   verification, target plans, metadata, and both runtimes.
5. Target representation improvement: keep it in backend planning/emission.
6. Byte, binding, launch, name, or lifetime change: update the single ABI
   owner and every generated/native consumer together.

This keeps beginner syntax ergonomic without pretending orchestration,
parallel memory, or provider representation is simpler than it is.
