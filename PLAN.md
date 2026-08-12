# Atomic whole-program fusion revamp

This is the implementation contract for one breaking, repository-wide change-set. It is not a migration plan, a sequence of releases, or a list of optional phases. The branch is complete only when the new source model, both IR tiers, both target planners, both shader emitters, generated bindings, the WebGPU runtime, the Vulkan harness, examples, tests, and documentation land together and the old one-export/one-dispatch contract is gone.

## 1. Outcome and accounting

Tach will compile a public logical GPU program, not assume that every public name is one physical shader entry point. The final pipeline is:

```text
.tach source
  -> parse and type-check helpers, indexed stages, and public programs
  -> verified Kernel IR (one invocation's semantics)
  -> verified Flow IR (resource versions, domains, and ordered stage graph)
  -> portable Kernel IR optimization + portable graph fusion
  -> clone once per requested target
       -> target-profile fusion + workgroup selection + kernelization
       -> verified executable plan
       -> WGSL or SPIR-V emission and validation
  -> public-program bindings + target execution metadata validation
  -> one opaque ComputeCommand that may encode one or many dispatches
```

The change earns its cost because neither target API promises to combine separate dispatches. WebGPU defines each `dispatchWorkgroups()` as its own usage scope, while a compute pipeline selects one compute entry point; Vulkan likewise defines a compute pipeline as one static compute shader stage and records each `vkCmdDispatch` explicitly. Drivers may optimize within those contracts, but Tach must synthesize one shader invocation when it wants to remove an intermediate storage round-trip or a dispatch boundary. See the [WebGPU synchronization rules](https://gpuweb.github.io/gpuweb/#synchronization), [WGSL resource interfaces](https://gpuweb.github.io/gpuweb/wgsl/#resource-interface), [Vulkan compute pipelines](https://docs.vulkan.org/spec/latest/chapters/pipelines.html#pipelines-compute), and [Vulkan shader interfaces](https://docs.vulkan.org/spec/latest/chapters/interfaces.html#interfaces-resources).

Keep these user-facing strengths:

- The existing baseline remains exactly valid and remains the first example:

  ```tach
  export function scale[i](values: buffer<float32[]>, factor: float32) {
    if (i < values.length) {
      values[i] *= factor;
    }
  }
  ```

- A generated public function still accepts positional typed values and buffers and returns an opaque `ComputeCommand`.
- `gpu.buffer`, `gpu.submit`, resident sessions, canonical host layout, inferred buffer access, explicit logical coordinates, explicit atomics/barriers/shared memory, and target-neutral source semantics remain.
- An indexed export keeps the existing `LaunchOptions<rank>` contract, including 1D runtime-array length inference and the current one-workgroup fallback.

Delete these implementation assumptions rather than preserving them behind adapters:

- exported name equals physical WGSL/SPIR-V entry-point name;
- exported function equals one dispatch;
- buffer parameters become module-global logical resources;
- buffer bindings are module-global and permanently assigned before kernelization;
- semantic analysis chooses an automatic workgroup before a target exists;
- metadata is `{types, resources, kernels}`;
- one command owns one pipeline, one binding set, one parameter block, and one launch extent;
- a SPIR-V-only build can omit the metadata required to execute it.

There is no old-schema parser, compatibility mode, feature flag, dual runtime path, deprecation alias, or source rewriter. Tach is pre-0.1: rebuild all generated artifacts from `.tach` source.

## 2. Final source language

### 2.1 Declaration forms

Use one coherent `function` spelling with indices and `export` determining the role:

```text
function helper(...) -> value computation
function stage[i](...) -> private indexed GPU stage
export function kernel[i](...) -> indexed stage plus public one-stage program sugar
export function program(...) -> public static orchestration program
```

All names share the existing function namespace; overloading and recursion remain invalid. An indexed function has one to three immutable `uint32` coordinates, at least one `buffer<T>` parameter, no return value, and the existing kernel-body language. Indexed stages may be called only by a program `run` statement. An exported indexed stage may also be used by another program; exporting it merely adds the source-compatible one-stage public program.

`@workgroup(...)` is valid on any indexed stage, exported or private. It is invalid on helpers and non-indexed public programs.

Any stage that declares `shared<T>` or executes `workgroupBarrier()`/`bufferBarrier()` must have an explicit `@workgroup`. Workgroup membership affects those operations' semantics, so allowing each target to choose a different automatic size there would violate portability. Stages without workgroup-scoped semantics may remain `auto` and be selected after fusion.

### 2.2 Program syntax

Add only two orchestration constructs: compiler-owned transients and an explicitly sized stage run.

```tach
function scaleInto[i](
  input: buffer<float32[]>,
  output: buffer<float32[]>,
  factor: float32,
) {
  if (i < input.length && i < output.length) {
    output[i] = input[i] * factor;
  }
}

function addBiasInto[i](
  input: buffer<float32[]>,
  output: buffer<float32[]>,
  bias: float32,
) {
  if (i < input.length && i < output.length) {
    output[i] = input[i] + bias;
  }
}

export function transform(
  input: buffer<float32[]>,
  output: buffer<float32[]>,
  factor: float32,
  bias: float32,
) {
  const count = min(input.length, output.length);
  const scaled = transient<float32>(count);
  run scaleInto(input, scaled, factor) over count;
  run addBiasInto(scaled, output, bias) over count;
}
```

The compact grammar additions are final, not provisional:

```text
declaration       := type-decl | function-decl
function-decl     := {attribute} ["export"] "function" IDENT [indices]
                     parameters [":" type] block

program-statement := program-const ";" | run-statement ";"
program-const     := "const" IDENT "=" (shape-expression | transient-expression)
transient-expression := "transient" "<" type ">" "(" shape-expression ")"
run-statement     := "run" IDENT "(" [program-argument {"," program-argument}] ")"
                     "over" domain
domain            := shape-expression
                   | "[" shape-expression "," shape-expression
                         ["," shape-expression] "]"
```

`run` and `over` are contextual words, as existing source words are identifier tokens interpreted by the parser. Do not add lexer token kinds just to spell them.

### 2.3 Deliberately static program bodies

A non-indexed exported program is a host-evaluable, straight-line graph description. Its body permits only immutable `const` declarations and `run` statements. Reject `let`, assignments, ordinary expression statements, helper calls, `if`, `while`, `for`, `return`, shared variables, atomics, and barriers there. Those remain available inside indexed stages. This boundary gives the compiler a complete graph without inventing a second general-purpose host language or requiring runtime graph tracing.

A public program must have at least one external `buffer<T>` parameter so a generated command has an owning Tach session. Reject a transient-only/no-buffer public program instead of adding a second session-bound command-construction API.

A shape expression has checked `uint32` semantics and may contain:

- a non-negative integer literal;
- a public `uint32` value parameter or nested struct field;
- `.length` on a public or transient path whose final type is a runtime array;
- a previously declared shape `const`;
- parentheses and `+`, `-`, `*`, `/`, `%`;
- `min`, `max`, and `ceilDiv`.

Evaluation rejects overflow, underflow, division/remainder by zero, non-integral host values, values above `uint32`, and a zero final domain or transient element count. `ceilDiv(a, b)` is defined only for positive `b`. Plain-value errors detectable without buffer sizes are reported when the command is constructed; `.length`-dependent checks occur during command preparation after buffer materialization.

A program argument is intentionally smaller than a kernel expression:

- a buffer formal receives one external or transient resource symbol;
- a value formal receives a public value parameter, a nested field of one, a context-typed scalar/bool literal, or a shape expression/const when the formal is `uint32`;
- no program expression reads buffer contents or constructs arbitrary GPU values.

Expose a value as a public parameter instead of building a host expression language for rare cases.

`run`, `over`, `transient`, and `ceilDiv` are reserved contextual spellings in their owning positions; a source declaration may not reuse them as a public/function name. `min` and `max` retain their existing intrinsic reservation. This prevents a helper name from changing the meaning of a program graph.

`transient<T>(count)` creates a compiler-owned runtime array resource with logical stage type `buffer<T[]>`. `T` must have a fixed host-shareable footprint and must not contain an atomic or runtime array. A transient cannot be a public parameter, returned, read back, or stored in a struct. It exists only within one logical command and may be storage-eliminated by fusion.

### 2.4 Launch and repeat semantics

Every explicit `run` has a rank-1, rank-2, or rank-3 `over` domain matching the stage rank. A public orchestration program therefore needs no host `size` option. Its generated final option is:

```ts
export interface CommandOptions {
  readonly repeat?: number;
}
```

Keep the current entry point for baseline kernels:

```ts
export interface LaunchOptions<Size extends LaunchSize = LaunchSize>
  extends CommandOptions {
  readonly size?: Size;
}
```

`repeat` always means repeat the entire logical public program in source order. It never means repeat each physical step independently. A target may replace the repetitions with an invocation-local loop only after proving that every iteration dependency is same-invocation and that the program contains no atomics, collective barriers, shared-memory protocol, or cross-invocation access. Otherwise the executable plan repeats its complete dispatch/barrier sequence.

`repeat` defaults to `1` and command construction requires a positive `uint32`; this single limit applies whether execution remains host-recorded or is internalized into a shader, so optimization cannot change accepted inputs.

An exported indexed function lowers to a generated one-stage Flow program whose domain is the existing host launch symbol. Omitted 1D `size` still uses the first compatible runtime-array buffer length; otherwise omission still selects exactly one chosen workgroup. This is syntax sugar, not a separate compiler/runtime path.

### 2.5 Program memory contract

- Different public buffer parameters of one program are distinct non-aliasing objects. In-place work uses one public buffer parameter across runs. The managed runtime checks the rule once when constructing the command; native callers have the same obligation.
- A stage may not receive one logical resource in two different buffer formals. This preserves the existing per-kernel non-aliasing rule after graph composition.
- Runs are semantically ordered bulk-synchronous dispatches. All writes from one run happen before dependent accesses in the next run. An unfused WebGPU plan gets this from ordered dispatch commands; a Vulkan plan emits the required compute-to-compute memory barrier.
- A transient begins uninitialized. Its first read must be dominated by a proven complete definition: exactly one non-atomic store for every transient element in the run domain, with no read-before-write path. The access/domain prover recognizes the ordinary guarded form (`i < transient.length`) as complete when `over transient.length` implies the guard. Reject a graph that cannot prove initialization; do not silently clear scratch storage.
- Each repetition must completely define a transient before that repetition reads it. Scratch bytes from a prior command or repetition have no semantic value.
- Source order is observable for external buffers. Reordering or fusing runs is legal only through the dependence proof below.

## 3. The two logical IR levels

Keep the existing `src/ir` package path. Renaming it would touch every compiler package without adding capability. Its documented role changes from an ambiguously named whole module to **Kernel IR**, while new `src/flow` owns inter-kernel meaning.

### 3.1 Kernel IR: one invocation

Retain the current typed SSA values, typed places, structured `If`/`Loop`, helpers, effects, uniformity analysis, and source spans. Make resources function-local:

```text
Kernel Module
  Structs
  Functions

Function
  Kind = helper | stage
  Indices
  BufferParams {name, type, inferred access, span}
  ValueParams
  SourceParams {buffer index | value ID}
  WorkgroupConstraint = auto | explicit(x,y,z)
  WorkgroupVars
  Body
```

Delete `Module.Resources`, `Function.Compute`, `Function.KernelParams`, and module-resource indices. `PlaceRoot` names a buffer parameter of its containing stage. Buffer access is inferred independently for each stage, so identical source parameter names in unrelated stages no longer collide and physical binding assignment remains a later concern.

Automatic workgroup size is represented as `auto`; semantic analysis does not insert `256`, `16x16`, or `8x8x4`. An explicit attribute retains its checked portable constraint. A concrete workgroup belongs to a physical kernel in the target executable plan.

Add one structured composition primitive because sequentially combining stages must not let `return;` from the first stage skip later stages:

```text
scope {
  ...
  exit_scope
}
```

`Scope` owns a void region; `ExitScope` may terminate any nested block inside its nearest scope. Kernelization wraps each component stage in a scope and rewrites that stage's void returns to `ExitScope`. WGSL lowers it to a compiler-generated one-iteration `loop`/`break`; SPIR-V lowers it to a structured merge and branches. Source gains no `scope`, `break`, or `continue` syntax.

### 3.2 Flow IR: a public command graph

Add `src/flow` with these stable concepts:

```text
Flow Module
  KernelModule -> verified Kernel IR
  Programs[]

Program
  public name and source span
  positional public parameters (buffer | value)
  logical resources (external | transient)
  immutable shape-expression DAG
  ordered Dispatch nodes
  initial/final resource versions

Dispatch
  referenced stage
  1..3D symbolic domain
  source-ordered buffer/value arguments
  consumed resource versions
  produced resource versions for every mutable/atomic argument
```

Use monotonically allocated, program-local `ResourceID`, `VersionID`, `ShapeID`, and `DispatchID`; zero is invalid. External buffer parameters create initial versions. A read-only stage consumes but does not replace a version. A mutable or atomic stage consumes the current version and produces the next version even when the physical storage is in-place. Transients begin with an explicit undefined version, and the verifier requires a complete-definition edge before a readable version exists. This memory-SSA form is the single owner for RAW, WAR, and WAW ordering.

Flow IR contains logical types, stage references, source names, and symbolic extents only. It contains no binding numbers, physical entry names, target limits, WebGPU usage flags, Vulkan barriers, scratch allocation colors, or parameter byte offsets.

The Flow verifier checks:

- unique public names and IDs, valid source spans, and deterministic order;
- public parameter/resource correspondence and the non-aliasing declaration contract;
- valid acyclic version chains with one producer per defined version;
- exact stage rank and buffer/value argument types;
- shape expression typing, references, arithmetic operators, and rank;
- transient element legality, initialization dominance, complete-definition evidence, and non-escape;
- every external mutable result is represented by the program's final version;
- every stage reference names a verified Kernel IR stage, never a value helper;
- single-stage sugar and explicit programs obey the same invariants.

`flow.Clone` and `ir.Clone` perform explicit deep copies using standard-library slices/maps. Each target gets an independent graph and kernel module; target fusion must never mutate the portable result or another target's plan.

### 3.3 Diagnostic dump

The `.tir`/`tach ir` output becomes one deterministic diagnostic with named sections:

```text
=== optimized logical program ===
Flow IR, resource versions, domains, and surviving dispatches

=== kernel templates ===
optimized helper/stage Kernel IR and access summaries

=== web executable ===
physical kernels, private entries, workgroups, bindings, scratch colors, steps

=== spirv executable ===
the independently selected SPIR-V plan
```

`tach ir` compiles `all` and prints all four sections. Text remains diagnostic-only and is byte deterministic.

## 4. Access and dependence analysis

Add one derived Kernel IR analysis in `src/ir`; do not add user access/fusion annotations. It lives beside uniformity/use analysis because Flow verification needs complete-write evidence before optimization and because importing an optimizer from a semantic verifier would invert package ownership. For every stage memory operation, record:

```text
Access
  buffer formal
  root field path
  kind = read | write | atomic
  exact index map or opaque
  normalized control predicate or opaque
  value definition for writes
  source span
```

An exact index map is a tuple of normalized affine `uint32` expressions over logical coordinates and uniform symbols (value parameters, constants, and array lengths). Recognize coordinate permutation, addition/subtraction of constants, and multiplication by a constant. Preserve modular `uint32` meaning; claim invertibility only for a proven permutation/unit-stride map over the stated domain. Anything data-dependent, non-affine, or ambiguous becomes `opaque`, never an optimistic guess.

Normalize the common bounds predicate fragment: conjunctions/disjunctions of coordinate comparisons against domain extents and uniform values, early-return negations, and exact equalities. Use interval/domain implication to prove that a guarded store covers a launch domain. An unsupported predicate is opaque.

Summaries also derive:

- stage effects: pure, ordinary memory, atomic, workgroup/barrier;
- per-buffer read/write/atomic status;
- exact complete-write maps;
- backward pure slices that compute each stored value;
- instruction count, peak live SSA estimate, helper closure, shared bytes, and existing workgroup constraint.

The analyzer is pure and returns a summary for one verified function. Flow verification invokes it to prove transient initialization; fusion invokes it for dependences and costs. Recompute summaries after any kernel rewrite. An optimizer may cache them by function pointer only within one invocation; no stale summary crosses a clone or mutation.

## 5. One legality engine, two profitability tiers

There must be one fusion legality implementation. The portable pass and both target lowerings call it with different policies; WGSL and SPIR-V must not grow separate dependence logic.

### 5.1 Legality

For a proposed adjacent dispatch group, first construct all cross-stage RAW, WAR, and WAW dependences from Flow versions and access summaries. Fusion is legal only when every dependence is one of:

- exact same-invocation access that can be preserved by sequential scoped bodies;
- a transient producer value forwarded directly into a consumer load;
- a pure producer slice recomputed at the consumer's exact affine access coordinate;
- disjoint resources or statically disjoint root field paths;
- an ordering-only dependence whose relative order remains identical inside each invocation and which has no cross-invocation edge.

Reject fusion when any dependence can cross workgroups or invocations, when an opaque access touches a shared dependency, or when removing the dispatch boundary would remove required device-wide visibility. A workgroup barrier is not a substitute for a dispatch boundary unless the proof establishes that the complete dependence is contained in the same selected workgroup; this sprint intentionally does not synthesize that tiled case.

Atomics, existing shared variables, `workgroupBarrier`, `bufferBarrier`, and opaque synchronization effects are fusion boundaries. Two ordinary stages with opaque indexing may be horizontally fused only when all of their resources are disjoint. Explicit workgroups must be identical, or one side must be `auto` and satisfy the other's size. Two incompatible explicit workgroups are never fused.

Preserve stage statement order, integer wrap behavior, floating operation order, edge predicates, and early-return behavior. Do not reassociate arithmetic as part of fusion.

“Same domain” means equal rank and structurally equal canonical shape DAGs. Direct sequential/horizontal fusion requires that equality. Affine recomputation may use the consumer domain only when the prover shows every substituted producer coordinate lies inside the producer domain and satisfies its defining-store predicate. A recomputed producer must have no observable writes other than the eliminated single-consumer transient and no atomic/barrier/shared effect.

### 5.2 Implemented fusion forms

The atomic revamp is not complete until all four forms work:

1. **Vertical transient elimination.** Fuse producer/consumer runs with compatible domains, forward exact same-index values, remove dead transient stores/loads, and delete a transient whose final use disappears.
2. **Sequential in-place fusion.** Fuse adjacent same-domain updates to one external resource when every cross-stage access is same-invocation; forward the prior stored value into the next load where possible.
3. **Horizontal fusion.** Combine adjacent same-domain stages with disjoint effects into one physical invocation, unioning live buffers and values while preserving each stage in its own scope.
4. **Affine producer recomputation.** At the target tier only, eliminate a single-consumer transient by cloning a side-effect-free producer backward slice at each consumer access coordinate, including small neighbor/stencil reads, when the cost policy accepts the duplicated work.

After composition, renumber IDs deterministically, rerun commoning/LICM/DCE, infer buffer access again, rebuild access summaries, verify Kernel IR, rewrite Flow versions/dispatches/transient lifetimes, and verify Flow IR.

Candidates are contiguous dispatch groups only; this sprint does not reorder the Flow graph. Scan in source order, choose the profitable legal candidate with the lexicographically smallest dispatch-ID tuple, rewrite, then restart until a full scan makes no change. Portable profitability is the exact zero-duplication rule below. Target profitability uses integer `opt.FusionPolicy` costs: removed dispatches and eliminated transient scalar words are benefits; cloned scalar-equivalent instructions and added peak-live values are costs; a rewrite must have positive score, stay below the profile's hard instruction/live/binding/workgroup limits, and remove at least one dispatch or physical transient. Profile constants are named, committed, and decision-tested; no wall-clock timing enters compilation. This makes candidate choice deterministic without confusing a cost estimate with legality.

General reductions, scans, global prefix operations, data-dependent gather/scatter fusion, cyclic graphs, runtime branching, target runtime JIT, profile-guided recompilation, auto-tuning, and synthesized shared-memory tiling are non-goals. Their required proofs are not needed for the profitable memory-bound chains above. Put this required comment next to the static cost model:

```text
// DECISION: This static model handles bounded affine fusion against guaranteed
// target limits. Replace its weights with measured profiles/autotuning only when
// Tach has a device-profile input; legality must remain in the shared prover.
```

### 5.3 Portable IR tier

Run after ordinary Kernel IR optimization and before target cloning. Accept only zero-duplication rewrites that remove a transient round-trip and/or dispatch, fit Tach's guaranteed cross-target limits, and do not make a workgroup choice. This tier performs exact vertical forwarding, exact same-invocation in-place fusion, and plainly disjoint horizontal fusion. Its output has one meaning for both targets.

### 5.4 Per-target lowering tier

Each target clones the portable module and invokes the same engine with the `opt.FusionPolicy` projected from its `backend.Profile`; `opt` never imports `backend`. The profile supplies guaranteed workgroup dimensions/invocations, storage-binding count, uniform-block bytes, workgroup storage, conservative instruction/live-value ceilings, and recomputation weights. The Web profile uses WebGPU guaranteed limits; the SPIR-V profile uses the supported Vulkan 1.1/core contract. Target profitability may make different physical kernels but cannot relax legality.

This tier performs affine recomputation, retries legal groups rejected by portable resource pressure, selects concrete workgroups, and decides whether a dynamic `repeat` can become an invocation-local loop. Offline builds use guaranteed profiles, not vendor guesses. No new dependency or runtime shader compiler is introduced.

## 6. Executable target plan and physical ABI

Extend `src/backend` to own a verified target executable plan. It is the only input to shader emission and execution metadata:

```text
Executable
  Target
  PhysicalKernels[]
  ProgramPlans[] in public-program order

PhysicalKernel
  private entry point
  concrete workgroup size
  fully kernelized Kernel IR function
  per-entry storage bindings
  optional physical parameter block
  used target coordinate inputs

ProgramPlan
  public program index
  transient allocation/color table
  ordered DispatchStep / BarrierStep list
  optional between-repetitions barrier
  repeat mode = whole-program | invocation-loop
```

Kernelization clones or composes stage bodies, converts component returns to scoped exits, drops unused buffer/value formals, assigns dense physical formals, and gives each physical kernel a deterministic `_tach_kN` entry. Public program names never enter shader ABI. Delete `abi.KernelEntry`; add one private-name function covered by injectivity/reserved-name tests.

Emit one physical kernel per surviving Flow dispatch in this change-set; do not add structural kernel deduplication while rewriting the ABI. Put this comment at the allocation site:

```text
// DECISION: Physical kernels are one-per-surviving-dispatch for deterministic,
// unambiguous plans. Deduplicate by verified Kernel IR hash only if real modules
// show shader-size/pipeline duplication worth the extra identity machinery.
```

Binding coordinates restart for every physical entry point: live storage buffers receive group/set `0`, bindings `0..N-1`, and an optional uniform parameter block receives binding `N`. Resource globals still have unique private symbol names. WGSL permits binding reuse across disjoint entry-point resource interfaces, and Vulkan explicitly permits overlapping set/binding decorations subject to statically used entry points. Validate uniqueness within each physical kernel, not across the module, and prove the behavior in the in-tree validators plus Chromium and `spirv-val`.

Parameter planning runs after kernelization and includes only live physical scalar/vector/struct leaves. A physical block records field types and offsets; each `DispatchStep` supplies an equal-length ordered list of field sources, where a source is a public parameter/path, a context-typed literal, an evaluated shape expression, or the implicit repeat count. This keeps a physical kernel reusable without baking one program call site's values into it. Keep one block per physical kernel, bool-as-`uint32`, 16-byte rounding, the 16 KiB portable ceiling, and the existing canonical layout engine. Do not copy whole public structs when only some leaves survive fusion.

The workgroup chooser honors explicit constraints exactly. For `auto`, start from the current rank defaults (`256`, `16x16`, `8x8x4`) clipped by the target profile, then reduce deterministically if the fused kernel's resource/shared limits require it. Coordinate lowering receives the selected physical size as an argument; it no longer reads a semantic-stage default.

A Vulkan `BarrierStep` names the exact logical resources requiring visibility between adjacent dispatches. Emit `vkCmdPipelineBarrier` from compute-shader writes to subsequent compute-shader reads/writes with buffer memory barriers. WebGPU has no explicit barrier step; ordered dispatches in one compute pass preserve the Flow order and each dispatch is its own usage scope.

Invocation-local repeat is valid only when kernelization has produced exactly one dispatch step and no barrier step, and the shared dependence proof establishes that all cross-repetition effects are same-invocation. Backend lowering then wraps that physical kernel body in a uniform loop whose bound comes from `ValueSource { kind: "repeat" }`; otherwise `repeat` remains a host-side loop around the entire plan. A zero or missing runtime repeat never reaches the executable because command validation normalizes it to a positive `uint32` first.

### 6.1 Package and API boundary

Keep the import graph acyclic and make ownership executable, not merely documentary:

```text
parser -> ast/source
ir -> types/source
flow -> ir/types/source
sema -> ast/flow/ir/types
opt -> flow/ir
abi -> ir/layout/types
backend -> opt/flow/ir/abi/layout
wgsl, spirv -> backend plus lower-level IR/layout utilities
bindings -> flow/backend/layout/types
compiler -> parser/sema/opt/wgsl/spirv/bindings
```

The orchestration contract is `sema.CheckAndLower` producing a verified `*flow.Module`, `opt.OptimizeLogical` mutating only that logical module, `backend.Lower(logical, profile)` deep-cloning and returning a verified `*backend.Executable`, and each target emitter accepting only its executable. `wgsl.Lower` and `spirv.Lower` are thin target-profile constructors around `backend.Lower`; they do not own fusion. Bindings receive the optimized logical module plus the completed target executable(s). In particular, `flow` does not import `opt`, `opt` does not import `backend`, emitters do not plan ABI, and runtimes never reconstruct compiler decisions.

## 7. Metadata and generated contract

Replace the old schema outright with versioned schema `1`:

```ts
interface Metadata {
  schema: 1;
  types: TypeMetadata[];
  programs: PublicProgramMetadata[];
  targets: {
    web?: TargetPlanMetadata;
    spirv?: TargetPlanMetadata;
  };
}

interface PublicProgramMetadata {
  name: string;
  parameters: Array<{
    name: string;
    kind: "buffer" | "value";
    type: string;
    resource?: number;
  }>;
  resources: ExternalResourceMetadata[];
  launch?: {
    dimensions: 1 | 2 | 3;
    inferFromResource?: number;
  };
}

interface TargetPlanMetadata {
  kernels: PhysicalKernelMetadata[];
  programs: ProgramPlanMetadata[];
}

interface PhysicalKernelMetadata {
  entryPoint: string;
  workgroupSize: [number, number, number];
  bindings: Array<{
    group: 0;
    binding: number;
    access: "read" | "read_write";
    type: string;
    minimumByteSize: number;
  }>;
  parameterBlock?: {
    group: 0;
    binding: number;
    byteSize: number;
    fields: Array<{
      type: string;
      byteOffset: number;
    }>;
  };
}

interface ProgramPlanMetadata {
  program: number;
  transients: Array<{
    type: string;
    stride: number;
    alignment: number;
    minimumByteSize: number;
    length: ShapeExpression;
    color: number;
    firstStep: number;
    lastStep: number;
  }>;
  steps: Array<DispatchStep | BarrierStep>;
  repeatBarrier?: BarrierStep;
  repeat: "program" | "invocation-loop";
}

interface DispatchStep {
  kind: "dispatch";
  kernel: number;
  domain: ShapeExpression[];
  resources: Array<{
    binding: number;
    source: { kind: "external"; resource: number }
          | { kind: "transient"; transient: number };
  }>;
  parameters: ValueSource[];
}

interface BarrierStep {
  kind: "barrier";
  resources: Array<{
    kind: "external" | "transient";
    resource: number;
  }>;
}
```

External resources deliberately have no physical group, binding, or access: those belong to each physical kernel. They carry the one canonical host layout already used by generated codecs:

```ts
interface ExternalResourceMetadata {
  name: string;
  type: string;
  byteSize?: number;
  alignment: number;
  runtime: boolean;
  runtimeOffset?: number;
  runtimeStride?: number;
  minimumByteSize: number;
  layout: HostLayout;
}
```

For a transient, `type` is its runtime-array buffer type, `stride`/`alignment` come from the same canonical layout engine, and `minimumByteSize === stride` because source counts are positive. `length * stride` uses checked arithmetic in both runtimes. The evaluator unions are exact:

```ts
type ShapeExpression =
  | { op: "constant"; value: number }
  | { op: "parameter"; parameter: number; path: string[] }
  | { op: "resourceLength"; resource: number; path: string[] }
  | { op: "launchAxis"; axis: 0 | 1 | 2 }
  | { op: "add" | "sub" | "mul" | "div" | "rem" | "min" | "max" | "ceilDiv";
      left: ShapeExpression; right: ShapeExpression };

type ValueSource =
  | { kind: "parameter"; parameter: number; path: string[] }
  | { kind: "bool"; value: boolean }
  | { kind: "i32" | "u32" | "f32Bits"; value: number }
  | { kind: "shape"; expression: ShapeExpression }
  | { kind: "repeat" };
```

Every numeric JSON field is range-checked. `f32Bits` carries the exact IEEE-754 word, including signed zero, instead of depending on JSON decimal round trips. `launchAxis` is valid only for an indexed-export sugar program and resolves through its explicit/inferred/default launch contract. Tagged forms avoid string evaluation in either runtime.

Target program plans and public programs have identical order and exact one-to-one indices. Physical kernels have no public `name`. A dispatch has exactly one resource source per physical storage binding and one value source per physical parameter field (zero when the kernel has no block). `ValidateGenerated` must walk every reference, expression, binding, layout, parameter field/source, transient lifetime, target plan, JS export, and declaration signature. It must reject unknown tags and extra/missing target plans. Do not retain the old `resources`/`kernels` top-level fields.

`repeatBarrier` is present only in a SPIR-V whole-program plan when the last dispatch of one repetition writes a resource read/written by the next repetition's first dependent dispatch. The Vulkan executor emits it only between repetitions, never after the final one. WebGPU plans omit it because ordered dispatch commands provide the required boundary. Invocation-loop plans omit it because their proof eliminated the dispatch boundary legally.

An invocation-loop plan has exactly one dispatch, no barriers or physical transients live across an iteration, and exactly one `repeat` value source in that kernel's parameter block. A whole-program plan has no `repeat` value source. `ValidateGenerated` enforces both shapes so a runtime cannot accidentally apply both repeat mechanisms.

Generated JavaScript embeds WGSL, public program descriptors, and only `targets.web`. Generated declarations export every source public program. Indexed exports end in `$launch?: LaunchOptions<...>`; orchestration exports end in `$options?: CommandOptions`. Both return `ComputeCommand` and call one private `$tach.command(programIndex, args, options)`.

Artifact sets become:

| target | written artifacts |
|---|---|
| `web` | `.wgsl`, `.js`, `.d.ts` |
| `spirv` | `.spv`, `.tach.json` |
| `all` | `.tir`, `.wgsl`, `.spv`, `.spvasm`, `.js`, `.d.ts`, `.tach.json` |

An `all` metadata file contains both target plans; a SPIR-V build contains only `targets.spirv`. A web build keeps its web plan embedded in generated JavaScript and does not add a redundant JSON file. Stale-artifact removal follows these exact sets.

## 8. WebGPU runtime execution

Keep the public runtime small. `ComputeCommand`, `ComputeBuffer`, `Tach`, `tach`, submission ordering, readback, and ownership stay. Internals change from one kernel to one public program:

1. Generated command construction validates positional arguments and program-wide non-aliasing, snapshots plain values/options, and derives the owning session from the first external buffer.
2. Preparation materializes external buffers with the existing codecs, evaluates checked shape trees, sizes/colors transients, packs every physical parameter block, and asynchronously caches all physical pipelines referenced by the plan.
3. Recording iterates the plan in order in the existing single compute pass, selecting the physical pipeline/bind groups and dispatching each step. `repeat=R` wraps the entire step iteration unless the plan says invocation-loop.
4. No host-visible object is created for a transient and no transient is decoded/read back.

Add a session scratch pool of storage buffers indexed by plan color. A prepared command reports abstract `{color, byteSize}` requirements; `Session.#record` takes the maximum for each color across the ordered submission, resolves/grows the actual buffers once, then supplies them to each command encoder. A color's required capacity is the maximum evaluated byte size assigned to it in that prepared submission. Grow geometrically, bind offset `0` with the exact logical byte range, reuse under queue order, and invalidate bind groups that reference a replaced buffer. Liveness coloring may share a color only when transient live intervals do not overlap and no physical dispatch binds both.

Do not immediately destroy a replaced scratch or parameter arena while earlier queue work may still use it. Move replaced buffers to a retired list and reclaim them asynchronously after the next `queue.onSubmittedWorkDone`; `idle()` waits and reclaims, while synchronous `close()` preserves its current contract by immediately destroying all live/retired allocations and then the device. Put a `DECISION:` comment beside the unbounded live bind-group/scratch-color caches naming the current session-lifetime ceiling and the measured-profile/LRU upgrade path.

Parameter upload remains one aligned session arena with dynamic uniform offsets, but one prepared command now contributes zero or more blocks in physical step order. Bind-group cache keys already include layout, buffer identity, offset, and size; extend them naturally to scratch ranges rather than adding a second cache.

## 9. Native Vulkan execution

The native harness becomes an executable-plan consumer rather than a one-kernel smoke path. Given a public program name and host arguments, it must:

- select `targets.spirv.programs[publicIndex]`;
- allocate/upload every external resource and validate public non-aliasing;
- evaluate the same tagged shape expressions with checked `uint32` arithmetic;
- allocate transient colors at their maximum required byte sizes;
- create/cache every referenced compute pipeline and its per-entry descriptor layout;
- pack per-kernel blocks from `ValueSource` fields;
- record each dispatch with its own descriptor set and workgroup counts;
- emit each metadata barrier as a compute-write to compute-read/write buffer barrier;
- repeat the complete plan in order;
- submit once and read back only external buffers.

The harness remains deliberately simple and test-owned: host-visible memory and per-case pipeline setup are acceptable. Mark that ceiling with a `DECISION:` comment and name persistent device-local staging/caches as the upgrade only if harness profiling ever matters.

## 10. Atomic file-by-file change ledger

Every path below is part of the same merge. “Unchanged” is an audited decision, not permission to ignore a failing caller discovered during implementation.

### 10.1 New files

- `src/flow/flow.go` — define Flow module/program/resource/version/shape/dispatch structs, tagged shape operations, deterministic ID allocators, deep cloning, and the textual dump. Keep these closely related operations in one file rather than creating one file per node kind.
- `src/flow/verify.go` — implement the complete Flow verifier and initialization/version/domain checks described above.
- `src/flow/flow_test.go` — directly corrupt IDs, versions, stage arguments, domains, transient definitions, and final versions; require targeted verifier failures and deterministic dump order.
- `src/sema/program.go` — lower non-indexed exported bodies and indexed-export sugar to Flow IR. It owns the restricted shape/value argument checker, transient construction, `run` checking, public-name validation inputs, and resource-version construction. This separation is earned because the existing `sema.go` is already the large invocation-language checker and program statements use different semantics.
- `src/parser/parser_test.go` — test all four function declaration forms, contextual `run ... over`, rank domains, transient generic syntax, attributes, and malformed/ambiguous forms at the grammar owner.
- `src/ir/access.go` — build exact/opaque access maps, predicates, complete-write proofs, pure backward slices, effect summaries, and cost metrics from verified Kernel IR. Flow verification and optimization share this one analysis.
- `src/ir/access_test.go` — directly cover identity/permuted/offset/field access, bounds implication, complete writes, pure slices, opaque fallback, synchronization effects, and deterministic summaries.
- `src/opt/fusion.go` — contain the single dependence prover, deterministic contiguous-candidate selection, legality result with rejection reason, Kernel IR composition/renumbering, Flow rewrite, portable policy, and the backend-independent `FusionPolicy`/target-policy entry point.
- `src/opt/fusion_test.go` — table-test each legal form and every hard boundary. Assert transformed dispatch/transient counts, rewritten bodies, rejection reasons, verification after rewrite, and byte-deterministic output.
- `src/backend/program.go` — define `Profile`, `Executable`, `PhysicalKernel`, dispatch/barrier/transient plan nodes, target cloning/lowering, workgroup selection, kernelization, parameter-plan attachment, liveness coloring, verification, and diagnostic dump. These are one target-planning responsibility and should not become a new package hierarchy.
- `src/backend/program_test.go` — verify profile limits and deterministic fusion costs, explicit/auto workgroup resolution, dense private names/bindings, parameter liveness, transient colors, adjacent and between-repetition Vulkan barriers, WebGPU step order, both repeat modes, and rejection of corrupted plans.
- `examples/fusion.tach` — executable corpus module with two public programs: a two-stage transient scale/bias chain that must become one dispatch with no physical transient on both targets, and a neighbor-dependent external-buffer chain that must remain two ordered dispatches (and a Vulkan barrier). This single file proves both optimization and fallback without multiplying fixtures.

No other new package is authorized. In particular, do not add a generic graph framework, polyhedral library, target plugin system, cache library, JSON evaluator dependency, or `src/kernel` rename.

### 10.2 Front end and logical types

- `src/ast/ast.go` — replace separate `FuncDecl`/`ComputeDecl` with one function declaration carrying `Exported`, optional indices, attributes, optional return type, and body. Add `RunStmt`, `Domain`, and `TransientExpr`. Reuse ordinary expression nodes for shape expressions/arguments, but represent `transient<T>(...)` explicitly because the existing expression grammar has no generic-call node and must not confuse `<`/`>` with comparisons.
- `src/lexer/token.go` — unchanged. `run`, `over`, and `transient` stay contextual identifiers.
- `src/lexer/lexer.go` — unchanged except for a bug found by new syntax tests; no keyword table or token boilerplate.
- `src/lexer/lexer_test.go` — retain numeric coverage; add nothing merely because contextual words were introduced.
- `src/parser/parser.go` — unify function parsing, permit indices with or without `export`, require indices or no indices according to the declaration shape, parse contextual `run`/vector domains and `transient<T>(shape)` without weakening ordinary indexing/comparisons, and keep attributes attached for semantic validation. Parsing does not otherwise decide whether a body is a program.
- `src/source/source.go` — unchanged; existing spans/errors cover every new node.
- `src/types/types.go` — keep `buffer` and `transient` out of the value type graph. Add only a reusable `IsTransientElement` predicate if semantic lowering and Flow verification would otherwise duplicate `fixed footprint && host shareable && !atomic && !runtime`; use existing predicates underneath it.

### 10.3 Semantic lowering

- `src/sema/sema.go` — make `CheckAndLower` return a verified `*flow.Module`; collect one unified function namespace; lower helpers and every indexed function into function-local-buffer Kernel IR; infer access per stage; retain all expression/control/atomic/uniformity rules; represent automatic workgroups as unresolved constraints; require explicit workgroups for shared/barrier semantics; and delegate only public-program bodies/sugar to `program.go`. Remove module-global resource append logic and all `ComputeDecl` branches. Finish in the strict order `ir.Verify` then `flow.Verify` so Flow's access proofs only inspect valid Kernel IR.
- `src/sema/sema_test.go` — preserve helper/kernel semantics and add private-stage acceptance, public program lowering, indexed-export sugar equivalence, program-wide duplicate names, invalid stage calls, illegal program statements/arguments, shape typing, transient element restrictions, resource alias-at-one-run rejection, incomplete transient initialization, and explicit workgroup use on private stages.

### 10.4 Kernel IR and verification

- `src/ir/ir.go` — remove module resources and compute/public ABI fields; add `FunctionKind`, function-local `BufferParam`/source-parameter mapping, `WorkgroupConstraint`, `Scope`, `ExitScope`, deep clone support, and updated deterministic dump spelling (`stage`, `buffer %bN`, `workgroup(auto|...)`). Rename fields such as `PlaceRoot.Resource` to `Buffer` so stale module-resource assumptions fail to compile rather than survive semantically. Access summary node types shared by verifier/optimizer also live in this package.
- `src/ir/verify.go` — validate local buffer roots/access, helper-versus-stage restrictions, unresolved versus explicit workgroups, scope nesting/exits, and every old SSA/place/control/effect invariant. Accept unresolved automatic size only in logical stage IR; physical plan verification, not this verifier, requires a concrete size.
- `src/ir/uniformity.go` — seed stage buffer loads as varying using local roots and traverse `Scope`; keep barrier rejection under varying control unchanged.
- `src/ir/uses.go` — count operands through scopes/exits and local roots; retain one shared use-count implementation for optimizers/backends.
- `src/ir/verify_test.go` — replace the obsolete `KernelParams` mapping test with local-buffer/source-parameter integrity tests and add invalid scope exit, helper buffer, and unresolved-physical boundary cases.

### 10.5 Optimization and target planning

- `src/opt/opt.go` — change the public entry to optimize a Flow module's Kernel IR, adapt immutable-resource and loop-promotion logic to function-local buffers, traverse `Scope`, and expose the existing commoning/LICM/promotion/DCE sequence for newly synthesized physical functions. Invoke portable fusion only after initial per-stage cleanup and verify both IR levels before/after.
- `src/opt/opt_test.go` — port every existing pass test to local buffers and add scope traversal/cleanup assertions; keep fine-grained fusion cases in `fusion_test.go`.
- `src/backend/coordinates.go` — accept a concrete workgroup tuple from `PhysicalKernel`, not `Function.Workgroup`; traverse scopes and retain the exact current local-coordinate recognition.
- `src/wgsl/lower.go` — replace the coordinate-only wrapper with `backend.Lower(logical, WebProfile)` and WGSL-specific coordinate expression selection over the returned executable. It must run target-tier fusion through the shared backend/optimizer path, never a WGSL fuser.
- `src/spirv/lower.go` — mirror the same call with `SPIRVProfile`; retain only SPIR-V input-kind selection after the shared executable exists.

### 10.6 Layout and ABI

- `src/layout/layout.go` — unchanged. Transients and physical resources consume the existing checked canonical layout; do not add a second scratch layout.
- `src/layout/layout_test.go` — retain all layout tests. Add a runtime-array stride/checked-size assertion only if `backend.program` cannot use an existing covered result.
- `src/abi/names.go` — delete `KernelEntry`; keep `Mangle`; add deterministic private entry generation `_tach_k<decimal index>` and validation that it cannot collide with emitted helper/type/resource names or WGSL reserved double-underscore names.
- `src/abi/names_test.go` — retain mangling tests and cover dense private entry names, target portability, and collision boundaries.
- `src/abi/parameters.go` — plan a block for one kernelized `ir.Function` at an explicitly supplied binding, flatten only live logical leaves, and keep the layout/bool/16 KiB rules. Remove the module-wide `ParameterPlan` and `len(module.Resources)` binding calculation. Per-dispatch field sources belong to `backend.ProgramPlan`, not this physical layout owner.
- `src/abi/parameters_test.go` — assert dead physical leaves disappear, binding follows live storage count, bool representation remains `uint32`, empty blocks disappear, offsets stay canonical, and the 16 KiB rejection remains.

### 10.7 WGSL backend

- `src/wgsl/emitter.go` — accept a verified `backend.Executable`; emit physical-kernel-specific resource globals and parameter blocks, private `_tach_kN` entries, concrete workgroups, scopes, and only the helper closure each physical kernel needs. Map `PlaceRoot.Buffer` through the physical binding table. Stop planning ABI or iterating logical module resources inside the emitter.
- `src/wgsl/validator.go` — validate the emitted subset with multiple compute entries and overlapping group/binding pairs only when their static entry-point closures are disjoint. Check each entry's declared workgroup, resource interface, binding access/type, and private name against the executable plan.
- `src/wgsl/emitter_test.go` — retain semantic emission coverage and add fused single-entry output, unfused multiple entries, per-entry binding reuse, scoped early returns, target-selected workgroups, live-only parameters/resources, and validator mutations for an interface/plan mismatch.

### 10.8 SPIR-V backend

- `src/spirv/emitter.go` — consume the verified executable; allocate resource globals/parameter blocks per physical kernel, allow overlapping set/binding decorations, emit `_tach_kN` entry points and concrete `LocalSize`, map local buffer roots, lower scopes to structured merges, and compute each entry's exact used input/resource closure. Remove internal parameter planning and all module-resource/public-name assumptions.
- `src/spirv/validator.go` — retain independent binary decoding/CFG/layout/dominance validation and add executable-aware validation of every physical entry, concrete local size, statically reachable descriptor interface, allowed overlapping bindings, access decorations, parameter block, and scope-generated structured control.
- `src/spirv/validator_mutation_test.go` — retain all current corruptions and add missing/wrong physical entry, wrong local size, descriptor reachable from the wrong entry, incompatible overlapping binding, and malformed scope merge mutations.
- `src/spirv/emitter_test.go` — port existing tests and add fused/unfused entry counts, per-entry descriptor reuse, scoped returns, live parameter fields, target workgroups, and exact interface assertions.
- `src/spirv/spec.go` — unchanged unless scope lowering needs an already-supported branch/merge constant named; add no opcode abstraction unrelated to emitted instructions.
- `src/spirv/disasm.go` — unchanged functionally; its existing decoder must naturally print the new private entries and multiple plans.

### 10.9 Bindings, metadata, and compiler orchestration

- `src/bindings/generator.go` — replace old metadata structs/builders with schema `1`; generate public-program JS/TypeScript from Flow IR; serialize web and/or SPIR-V executable plans; reuse the current host-layout/code-generation helpers; embed tagged shape/value sources; and rewrite `ValidateGenerated` to cross-check the complete graph-to-target contract. Delete the parallel `runtimeResource`/`runtimeKernel` translation if the new metadata structs can be marshaled directly into the generated module; one descriptor representation is the goal.
- `src/bindings/generator_test.go` — replace top-level resource/kernel expectations with public programs and physical plans. Test baseline declaration compatibility, orchestration `CommandOptions`, private entries, live-only ABI, transient metadata, target omission rules, tagged expression validation, public names, typed arrays, atomics, and rejection of every dangling/cross-target reference.
- `src/compiler/compiler.go` — orchestrate parse → sema Flow/Kernel lowering → portable optimization → requested target lowerings/emission → target-plan-aware binding/metadata generation. Change backend results to carry their executable plan. Build combined deterministic IR diagnostics only for `all`; generate SPIR-V metadata for `spirv`; update exact artifact write/stale-removal sets; validate all synchronized artifacts before returning.
- `src/compiler/compiler_test.go` — keep language/backend regression coverage while replacing obsolete public-entry/module-binding tests. Add exact target artifact sets, baseline sugar equivalence, program graph dump, portable and target fusion counts, transient elimination, unsafe fallback, Vulkan barrier metadata, target plan independence, private name/binding reuse, workgroup timing, whole-program repeat, metadata schema, generated signatures, byte determinism, and compilation of every maintained documentation fence.
- `main.go` — keep commands and flags; update `ir` wording to Flow + Kernel + target plans, `check` summaries to report public programs/physical kernels/dispatches, and artifact help for SPIR-V metadata.

### 10.10 TypeScript package and runtime

- `tach-ts/src/runtime.ts` — export `CommandOptions`; make `LaunchOptions` extend it; change prepared commands to carry ordered parameter chunks, abstract scratch requirements, and execution steps; aggregate/resolve scratch once per submission; add liveness-color allocation, safe retirement/reclamation for grown parameter/scratch buffers, and plan recording while preserving one compute pass/queue submission and all ownership/error behavior.
- `tach-ts/src/internal.ts` — replace `ModuleDefinition.resources/kernels` with schema-checked public programs plus the web target plan; reuse all existing host codecs/materialization; implement tagged shape/value evaluation; construct a command by public program index; prepare all referenced pipelines/bindings/parameter blocks/transients; encode ordered steps and whole-program repeat; and remove one-kernel defaults/dispatch loops. This file executes plans but never makes a fusion decision.
- `tach-ts/src/index.ts` — publicly re-export `CommandOptions` alongside the existing runtime types.
- `tach-ts/src/error.ts` — unchanged; shape, plan, transient, and pipeline failures fit existing `kernel`, `buffer`, `gpu-validation`, and lifecycle categories.
- `tach-ts/src/compiler.ts` — no API change; its existing target union remains. Confirm that the returned artifact list now includes SPIR-V metadata through tests rather than adding wrapper logic.
- `tach-ts/tests/runtime.test.mjs` — update the controlled WebGPU device to observe multiple pipelines/dispatches/storage scratch. Preserve all current lifecycle/packing/caching/error tests and add graph argument validation, checked shapes, whole-program versus invocation-loop repeat ordering, exact fused/unfused dispatch sequences, transient growth/reuse/coloring, no transient readback, per-entry bind layouts, multi-block offsets, and retirement only after completion.
- `tach-ts/tests/compiler.test.mjs` — update artifact expectations (`spirv` now writes `.spv` + `.tach.json`) and verify generated baseline and orchestration declarations.
- `tach-ts/README.md` — rewrite the mental model from “generated kernel” to “generated program” without making the simple scale tutorial harder; document private stages, program syntax, `CommandOptions`, whole-program repeat, transients, multi-dispatch preparation/caching, and the new artifact/metadata boundary.
- `tach-ts/cli.mjs`, `tach-ts/scripts/install.mjs`, `tach-ts/package.json`, and `tach-ts/tsconfig.json` — unchanged; no command, installer, dependency, export-map, or compiler-setting change is required.

### 10.11 Browser harness

- `browser-test/scripts/build-examples.mjs` — read schema `1`, list public programs and per-target physical kernel/dispatch/transient counts in the manifest, and still require all seven `all` artifacts.
- `browser-test/app.mjs` — render program and physical-plan counts instead of old kernel/resource counts.
- `browser-test/index.html` — rename the affected table headings; no layout redesign.
- `browser-test/tests/interface.spec.mjs` — validate schema keys/version, public export names independent from private entries, baseline versus orchestration option shapes, program-wide non-aliasing, one command owning multiple physical steps, scratch cleanup, pipeline/bind-group counts, and whole-program repeat order in the controlled device.
- `browser-test/tests/webgpu.spec.mjs` — execute every existing example plus both `fusion.tach` programs on real Chromium WebGPU; assert expected results, zero shader errors, the fused program's one physical dispatch/no transient, and the neighbor program's two-dispatch fallback.
- `browser-test/README.md` — document that the harness checks logical programs and compiler-selected physical plans.
- `browser-test/server.mjs`, `browser-test/markdown-reporter.mjs`, `browser-test/playwright.config.mjs`, and `browser-test/package.json` — unchanged unless a renamed manifest field is displayed; no dependency or runner change.

### 10.12 Vulkan harness

- `spirv-test/vulkan.go` — replace old metadata structs and `dispatch` with schema-1 program execution. Build every referenced per-entry pipeline/layout, allocate external and colored transient buffers, pack value sources, bind step-specific descriptors, record dispatch/barrier sequences plus `repeatBarrier` only between repetitions, and read back externals. Reuse existing Vulkan setup, memory allocation, validation capture, and result helpers; fix the shared executor once rather than adding a special fused path.
- `spirv-test/cases_test.go` — make cases name a public program and provide public buffer/value arguments, optional indexed launch size, and repeat instead of one prepacked physical parameter block. Add expected outputs for both `fusion.tach` programs.
- `spirv-test/harness_test.go` — allow multiple program cases per source while still proving every example source/public program has coverage; run `spirv-val`, execute the selected plan, and report physical kernel/dispatch counts.
- `spirv-test/README.md` — replace one-pipeline caller instructions with the executable-plan/barrier/transient obligations.

### 10.13 Showcase

- `showcase-ts/kernels/benchmarks.tach` — express `integrateParticles` as a public graph over two private stages with a compiler-owned transient while preserving its numerical result and public positional arguments. The first stage computes predicted positions; the second commits them. Leave the other four exported indexed workloads as baseline syntax coverage.
- `showcase-ts/src/benchmarks.ts` — call the graph without a host `size` option, keep measurement boundaries identical, and include its logical/physical dispatch information in the result/report so the benchmark proves fusion occurred rather than merely timing it.
- `showcase-ts/tests/showcase.spec.mjs` — continue checking five real workloads and additionally assert the particle program is a fused one-dispatch plan with correct output.
- `showcase-ts/README.md` — explain which workload exercises automatic inter-kernel fusion and how the report distinguishes logical runs from physical dispatches.
- `showcase-ts/benchmark-report.md` — regenerate from an actual showcase run after the implementation; never hand-edit claimed performance numbers.
- `showcase-ts/scripts/build-kernel.mjs`, `showcase-ts/src/main.ts`, `showcase-ts/src/style.css`, `showcase-ts/src/vite-env.d.ts`, `showcase-ts/index.html`, `showcase-ts/markdown-reporter.mjs`, `showcase-ts/playwright.config.mjs`, `showcase-ts/playwright.gpu.config.mjs`, `showcase-ts/playwright.full.config.mjs`, `showcase-ts/package.json`, and `showcase-ts/tsconfig.json` — unchanged unless the new measured result field is rendered; make the minimum display/test edit in the existing owner rather than adding UI machinery.

### 10.14 Examples and documentation

- `examples/scalars.tach` — intentionally unchanged; it is the permanent proof that the simplest syntax did not regress.
- `examples/atomics.tach`, `examples/bitwise.tach`, `examples/control.tach`, `examples/for.tach`, `examples/math.tach`, and `examples/particles.tach` — unchanged source, recompiled through the new indexed-export sugar path. Their continued execution proves no compatibility shim is needed.
- `README.md` — keep scale first, then add the smallest complete private-stage/public-program example; replace the pipeline, compiler-output, ABI-name, repository-map, and optimization descriptions with Flow IR → Kernel IR → per-target executable plans. State plainly that drivers do not perform Tach's inter-dispatch fusion.
- `docs/language.md` — rewrite module/function/kernel sections and compact grammar; specify program-body restrictions, shape arithmetic/errors, transient initialization/lifetime, `run` domains, stage/export rules, launch options, repeat, non-aliasing, and deliberate non-goals. Retain every existing invocation-language rule.
- `docs/ir.md` — document both Flow IR and Kernel IR end to end, memory SSA, access maps, scopes, fusion legality/forms, portable versus target policies, workgroup selection, executable plans, dumps, and both verifier boundaries. Remove every module-global resource example.
- `docs/architecture.md` — replace the pipeline diagram/package ownership/validation table; show one shared legality owner called at two profitability tiers and explain compiler/runtime/native responsibilities for multi-step commands.
- `docs/abi.md` — replace public-entry and module-binding contracts with public programs/private physical kernels; specify schema `1`, per-entry bindings, live parameter fields, tagged evaluators, transient colors, dispatch/barrier/repeat execution, artifact sets, and native obligations.
- `browser-test/README.md`, `spirv-test/README.md`, `showcase-ts/README.md`, and `tach-ts/README.md` receive the focused edits listed above; compiler documentation-fence tests must include all maintained guides.

### 10.15 Repository and release files

- `package.json`, `package-lock.json`, `go.mod`, and `go.sum` — unchanged. The standard libraries, existing WebGPU types, and existing Vulkan dependency cover the revamp; no new dependency is approved.
- `release.sh` — unchanged. It packages whatever the compiler/runtime tests validate; there is no migration artifact to ship.
- `webgpu-playwright.mjs` — unchanged; real-browser invocation remains the same.
- `LICENSE` — unchanged.
- `AGENTS.md` — private, untracked, read-only, and outside the change-set.

## 11. Required test matrix

Tests must prove semantics and the optimization decision separately.

### 11.1 Source and IR ownership

- Parse and lower the exact baseline `export function scale[i]` and prove its Flow program is equivalent to an explicit one-run program.
- Reject every illegal declaration-role combination and program statement at parser/sema, with source spans.
- Mutation-test Kernel and Flow verifiers independently; no backend may accept malformed input.
- Assert access summaries for identity, permutation, constant offset, guarded full write, field path, opaque gather, atomics, barriers, and early returns.

### 11.2 Fusion legality and profitability

For each candidate, assert both the decision and a stable reason:

- identity transient chain: fuse, remove transient, one dispatch;
- same-index in-place chain: fuse and forward;
- disjoint same-domain stages: horizontally fuse;
- affine single-consumer neighbor read with cheap pure producer: target-tier recompute;
- expensive producer over recompute budget: remain separate;
- equally scored legal candidates: choose the earliest stable dispatch-ID tuple;
- cross-invocation external dependence: remain separate;
- mismatched domains/ranks or incompatible explicit workgroups: remain separate;
- opaque shared-resource indexing, atomics, shared memory, or barriers: remain separate;
- resource/instruction/uniform/workgroup limit overflow: split before emission;
- early return in the first fused stage: second scope still runs where its own predicate permits;
- repeat internalization: only same-invocation programs get one physical loop; all others repeat the whole plan.
- cross-repetition Vulkan dependence: emit the exact barrier between repetitions and not after the last.

Every accepted rewrite must pass both verifiers and the ordinary Kernel optimizer. Every rejected rewrite must still produce a valid multi-dispatch executable.

### 11.3 Cross-artifact ABI

- Public names/parameter order/types match JS and `.d.ts`; private entries are dense and absent from the public surface.
- Each physical entry's WGSL and SPIR-V resource interface exactly matches its target-plan bindings even when another entry reuses `(group/set 0, binding N)`.
- Only live values appear in each physical parameter block; each dispatch supplies exactly matching field sources, and every source evaluates/encodes identically in TypeScript and Go.
- Flow domains, target workgroups, metadata workgroup counts, runtime dispatches, and Vulkan dispatches agree.
- Transient byte sizes use canonical stride with checked multiplication; colors never overlap live ranges.
- `spirv` writes executable metadata; `web`, `spirv`, and `all` remove all stale siblings exactly.
- Two complete identical compilations are byte-for-byte identical for every artifact.

### 11.4 Real execution

- Controlled TypeScript tests inspect exact pipeline, bind-group, scratch-buffer, parameter-upload, dispatch, repeat, retirement, and destruction calls.
- Chromium executes all baseline programs, the fused transient program, and the forced multi-dispatch neighbor program with no compilation/uncaptured errors.
- `spirv-val --target-env vulkan1.1` accepts every module; Vulkan executes the same programs, including a real inter-dispatch buffer barrier in the forced fallback.
- The showcase returns correct results and reports one physical dispatch for its two-stage particle program.

## 12. Completion gates for the single merge

The implementation is done only when all of these are true in one tree:

1. `gofmt` has formatted every touched Go file and `go test -count=1 ./src/...` plus `go vet ./...` pass.
2. `npm run compiler`, `npm run check`, and `npm test --workspace=@depths/tach` pass with no generated drift.
3. Browser interface and real WebGPU suites pass; Vulkan/`spirv-val` suites pass on the documented capable host.
4. All seven maintained example modules plus `fusion.tach` are covered by both executable harnesses; every maintained Tach documentation fence compiles.
5. The fused corpus case has one physical entry/dispatch and no transient allocation on both targets; the unsafe case has two dispatches and the SPIR-V plan contains the exact barrier.
6. Grep finds no surviving logical use of `Module.Resources`, `KernelParams`, `Compute bool`, `KernelEntry`, public-name-equals-entry validation, top-level metadata `resources`/`kernels`, or the runtime's one-kernel command loop.
7. There is one dependence/legality engine in `src/opt`; target packages contain profitability/profile data and representation lowering only.
8. No old metadata structs, compatibility parser, migration flag, duplicate shape evaluator, duplicate host layout, new dependency, placeholder TODO, or disabled/skipped replacement test remains.
9. Documentation describes the code that landed, including non-goals and exact failure behavior; generated benchmark/report files contain only results from actual runs.
10. `git diff --check` is clean, unrelated user changes are untouched, and private `AGENTS.md` remains unmodified/untracked.

## 13. Atomic implementation work graph

This is dependency order inside one branch, not phased delivery: unify declarations and local-buffer Kernel IR; add Flow lowering/verification; add access summaries and shared fusion; add executable planning; switch both emitters and validators; switch metadata/bindings and both runtimes; update corpus/harnesses/docs; run the complete matrix. No intermediate commit is considered releasable, and no temporary adapter is permitted to make an intermediate state look supported.
