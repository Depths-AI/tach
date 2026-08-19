# Tach IR: Flow programs and Kernel templates

This page is for contributors who need the compiler's portable meaning,
not for application authors. Kernel authors do not write this notation.
If you are learning to call Tach from TypeScript, use the
[language guide](language.md) and the [examples](../examples/README.md).

Tach has two target-independent intermediate representations:

- **Flow IR** describes host-callable programs, resources, shapes, versions,
  ordered dispatches, and optional terminal views.
- **Kernel IR** describes the portable work performed by one invocation of an
  indexed stage or helper.

The project frontend resolves imports and merges every kernel into one
`flow.Module` containing one `ir.Module`. Target lowering combines those
representations into parallel WebGPU and Vulkan executable plans containing
private physical kernels. IR dump methods remain contributor diagnostics and test
oracles rather than a public command, emitted artifact, or accepted input
language.

The identity boundary is:

```text
public program -> Flow dispatch -> logical stage -> physical target kernel
               -> optional Flow view -> target projection
```

Source-file identity is deliberately absent from symbol names after checking.
Project-global uniqueness makes an unqualified type or function name a true
project identity. Imports control which declarations may be resolved while
lowering a source file; they do not create aliases, qualified symbols, linker
records, or per-file IR modules. Canonical merge order is module identity,
kernel identity, then source declaration order.

## 1. One baseline through every representation

```tach
export function scale[i](values: buffer<float32[]>, factor: float32) {
  if (i < values.length) {
    values[i] *= factor;
  }
}
```

Its optimized logical-program dump is:

```text
program @scale(values=%r1: buffer<float32[]>, factor: float32) {
  resource %r1 values kind=external initial=%v1 final=%v2
  version %v1 resource=%r1 previous=%v0 producer=%d0 defined=true
  version %v2 resource=%r1 previous=%v1 producer=%d1 defined=true
  shape %s1 = launch(0)
  dispatch %d1 @scale over [%s1]
}
```

The corresponding Kernel IR template has this shape:

```text
stage @scale[i=%1](values=%b0: float32[] access=mutable, factor=%2: float32) workgroup(auto) {
  &p1 = place.buffer %b0 : float32[]
  %3 = array_length &p1
  %4 = < %1, %3 : bool
  if %4 -> [] {
    &p3 = place.index &p1, %1 : float32
    %5 = load &p3 : float32
    %6 = * %5, %2 : float32
    store &p3, %6
    yield []
  } else {
    yield []
  }
  return
}
```

Target planning then creates one private `_tach_k0` kernel, selects
`256 x 1 x 1`, maps `%s1` to the host launch axis, and records one dispatch.
The public name `scale` stays in metadata and generated bindings, not in the
shader entry ABI.

## 2. Flow module and public programs

A `flow.Module` contains:

```text
Kernel          shared Kernel IR module
Programs        ordered public Flow programs
Documentation   normalized structured source docs
```

Every exported function creates one public program. An exported indexed
function synthesizes a one-dispatch program; an exported unindexed function
lowers explicit orchestration.

A `Program` records:

```text
Name, Span
Indexed, Rank
Parameters
Resources
Versions
Shapes
Dispatches
View              optional format/source/version/extent
```

IDs for resources, versions, shapes, and dispatches are dense, nonzero, and
local to one program.

### Public parameters

A parameter has a name, type, source span, and kind:

| Kind | Meaning |
|---|---|
| `BufferParameter` | maps to one external resource ID |
| `ValueParameter` | immutable constructible host value |

Parameter order is the generated API order. Buffer resources additionally
remember the public parameter index.

### Resources

A resource records:

```text
ID, Name, Kind, Type, Span
Parameter              external public index, or -1
Length                 transient shape
Initial, Final         version IDs
```

`External` resources come from public buffers and start defined. `Transient`
resources come from `transient<T>(length)` and start undefined. Their logical
type is a runtime array even though target plans later assign reusable scratch
allocations.

Resource identity is stronger than a type. Different public buffers and
different transients are non-aliasing resources until target scratch coloring
proves that non-overlapping transient lifetimes can share storage.

### Versions

Every resource has an initial version. A mutable dispatch consumes the current
version and produces a later one:

```text
Version
  Resource
  Previous
  Producer dispatch
  Defined
```

`Defined` is not merely “some write occurred.” A previously undefined
transient becomes defined only if the stage access summary proves a complete
write for the dispatch domain. A later stage cannot read an undefined version.

This is currently a linear version chain over ordered dispatches, not a
general control-flow SSA graph: public-program source has no branch or loop.

### Shapes

Shapes are interned expression DAG nodes with these operations:

| Operation | Inputs |
|---|---|
| `constant` | one `uint32` literal |
| `parameter` | public value index plus struct-field path |
| `resourceLength` | resource ID plus runtime-array field path |
| `launchAxis` | baseline host launch axis |
| `add`, `sub`, `mul`, `div`, `rem` | two shapes |
| `min`, `max`, `ceilDiv` | two shapes |

Identical structural nodes are shared. Shapes remain symbolic until the shared
TypeScript runtime has public values, materialized resource byte lengths, and
optional launch coordinates. Its one checked `uint32` evaluator prepares both
WebGPU and Vulkan commands.

### Dispatches

A dispatch contains:

```text
ID, Stage, Span
Domain[]           one shape per stage coordinate
Buffers[]          formal -> resource, input version, optional output version
Values[]           formal -> checked value source
```

Buffer arguments are stored in formal order and cannot repeat a resource.
Value sources are a public parameter/path, literal bool/integer/float bits,
shape, or backend-added repeat count.

Flow IR names a logical stage, not a physical kernel index. Target planning
creates the latter.

### Views

A view is a terminal program result, not a Kernel IR value:

```text
View
  Format       SRGB8
  Source       runtime float32x4 resource
  Input        exact final resource version
  Width        checked shape
  Height       checked shape
  Span
```

The source must be an external or transient runtime array of linear RGBA
pixels. `Input` identifies the version produced by the ordered program, so a
backend cannot accidentally project an earlier write. Width and height reuse
the same symbolic shape DAG and checked host evaluator as dispatch domains and
transient lengths.

Views deliberately do not contain a canvas, texture, native surface, byte
packing rule, or presentation operation. Those are target/runtime concerns.
This separation lets a scalar-only public program allocate its frame as a
transient, run ordinary stages, and return a view without inventing a public
output buffer.

## 3. Flow verification

`flow.Verify` first verifies the shared Kernel IR, then proves:

- unique public program names and at least one dispatch per program;
- at least one external buffer for ordinary programs, while view programs may
  be scalar-only, plus valid constructible value parameters;
- dense, valid resource/version/shape/dispatch IDs;
- valid initial/final versions and monotonic version chains;
- acyclic, well-typed shape expressions;
- stage existence, domain rank, and exact argument counts;
- buffer/resource type equality and per-dispatch non-aliasing;
- current-version consumption and correct mutable output versions;
- definition before every read; and
- valid public/literal/shape sources for stage value formals; and
- a supported view format, runtime `float32x4` source, exact defined final
  source version, and valid width/height shapes when a view is present.

Because definition and version checks live here, neither backend runtime needs
to rediscover program dataflow from shader text.

## 4. Kernel module and functions

An `ir.Module` contains logical struct types and functions. Resources are not
module-global: every indexed stage has its own ordered `BufferParams`.

Function roles are:

| Kind | Contents |
|---|---|
| `Helper` | immutable value parameters, constructible result, pure body |
| `Stage` | logical indices, buffer/value source mapping, workgroup constraint, void body |

A stage records:

- `Indices`: one to three named `uint32` values;
- `BufferParams`: logical type, inferred `read`/`mutable` access, and span;
- `Params`: immutable non-buffer values;
- `SourceParams`: original interleaved source order;
- `Workgroup`: explicit dimensions or an automatic constraint;
- `WorkgroupVars`: shared allocations; and
- `Body`: one structured block.

No function records target binding numbers, entry decorations, or physical
parameter fields.

## 5. Values and places

Kernel IR deliberately distinguishes immutable data from addressable memory.

### Values

A `ValueID` is unique and nonzero across an entire function, including nested
regions. Value-producing instructions are:

| Instruction | Meaning |
|---|---|
| `Const` | canonical typed scalar literal |
| `Unary`, `Binary` | resolved Tach operator |
| `Convert` | explicit numeric conversion |
| `Composite` | vector or struct construction |
| `Extract`, `VectorIndex` | constant or dynamic value projection |
| `Call` | direct pure helper call |
| `Intrinsic` | portable math operation |
| `Load` | read through a place |
| `ArrayLength` | runtime-array length through a place |
| `Atomic` | atomic result except store |

Every result carries its resolved type. Emitters never repeat literal
inference or overload selection.

### Places

A `PlaceID` identifies a typed path to GPU memory, not a first-class pointer.

| Instruction | Meaning |
|---|---|
| `PlaceRoot` | stage buffer formal |
| `PlaceWorkgroup` | shared allocation |
| `PlaceField` | struct field or constant vector lane |
| `PlaceIndex` | array/runtime-array/vector index |

`Load`, `Store`, `ArrayLength`, and atomics consume places. A place chain keeps
the root buffer/workgroup identity, element type, and access rights available
for verification and optimization.

Ordinary `let` variables do not become places. Rebinding creates new SSA values
and structured results. Only external/shared memory is addressable.

## 6. Structured control flow

A block is an instruction list plus exactly one terminator:

| Terminator | Role |
|---|---|
| `Yield` | return values from an `If` branch or loop condition |
| `Continue` | provide next loop-carried values |
| `Return` | leave helper/stage, optionally with helper value |
| `ExitScope` | leave a backend-created scope without leaving the stage |
| `Unreachable` | explicit terminal state |

### Conditionals

`If` owns a condition, then/else blocks, and typed results. Ternaries normally
produce one result; statement branches may produce several when surrounding
locals are rebound.

```text
%result = if %condition -> float32 {
  yield [%then]
} else {
  yield [%else]
}
```

WGSL materializes results in private mutable locals. SPIR-V creates merge
blocks and `OpPhi`. Kernel IR chooses neither representation.

### Loops

`Loop` owns initial values, loop parameters, a condition region, a body region,
and final results:

```text
loop params=[(%index <- %initial), (%sum <- %zero)] {
  cond
    yield [%keepGoing]
  body
    continue [%nextIndex, %nextSum]
}
```

The condition yields one bool. The body supplies one next value per loop
parameter. When the condition is false, current parameters become results.
Source `while` and `for`, plus safe backend repeat internalization, share this
model.

`Scope` lets backend-created wrappers rewrite an ordinary stage `return` into
`ExitScope`, preserving early return inside an outer repeat loop.

Short-circuit `&&`, `||`, and conditional expressions lower to `If`, so only
the selected region executes.

## 7. Types, effects, and access summaries

Kernel IR uses only logical Tach types: scalars, numeric vectors, named
structs, atomics, fixed/runtime arrays, and void. Host padding, WGSL wrappers,
SPIR-V pointer/storage types, and descriptor decorations are later physical
representations.

`float16` and its two-, three-, and four-lane vectors are ordinary logical
floating types, not packed integers or backend annotations. Constants carry
canonical source text until emission; semantic checking proves they are finite
and inside the binary16 literal range, then emission rounds them to binary16
with round-to-nearest-even. `Convert` is the sole boundary between binary16 and
other numeric widths, so neither optimizer nor backend may silently widen an
f16 expression.

Observable effects are stores, atomics, and barriers. Loads from mutable or
shared memory cannot be freely reused. Loads from inferred read-only buffers
can be commoned when place identity and structured dominance match.

`ir.AnalyzeAccess` summarizes each stage buffer:

- whether it reads, writes, or performs atomics;
- whether its writes completely define a dispatch result;
- affine index information relative to logical coordinates; and
- global stage effects involving barriers/workgroup memory.

Flow definition proofs, repeat internalization, access inference, and later
planning consume this shared analysis rather than scanning source again.

## 8. Atomics, barriers, and uniformity

An `Atomic` records the portable operation, place, logical integer type, input,
and optional result. Scope and memory-semantics operands are backend-owned.

`Barrier` records workgroup-memory or buffer-memory synchronization. Uniformity
analysis propagates a conservative property through values, helper calls,
structured control, and loop fixed points. A barrier is legal only when all
enclosing decisions are uniform across the workgroup.

Zero-initialized workgroup storage is a language invariant rather than a
Kernel IR instruction. WGSL supplies it natively. SPIR-V emits a null
Workgroup-variable initializer, and the Vulkan host requires the corresponding
Vulkan 1.3 feature.

## 9. Kernel verification

`ir.Verify` checks, among other invariants:

- valid module types, function roles, and workgroup constraints;
- unique nonzero value/place definitions and structured availability;
- exact operand, result, yield, loop-carrier, call, and return types;
- valid source-parameter mappings and at least one stage buffer;
- legal buffer/workgroup roots and field/index projections;
- read/write access through every place;
- constructible, host-shareable, runtime, and shared-memory restrictions;
- intrinsic signatures and conversion rules;
- atomic place/type/access rules;
- compute-only workgroup effects; and
- barrier uniformity.

Semantic lowering verifies before Flow construction. The optimizer verifies
before and after its pipeline. Backend and binding boundaries verify again.

## 10. Target-independent Kernel optimization

`opt.OptimizeKernel` runs:

```text
verify
common values and places
hoist loop invariants
common values and places
promote conservative loop buffer values
common values and places
remove dead pure definitions to a fixed point
verify
```

### Commoning

Exact repeated constants, unary/binary expressions, conversions, intrinsics,
extracts, pure calls, small composites, place paths, array lengths, and
read-only loads reuse a dominating definition. Mutable loads and effects do
not.

Composites and calls wider than four inputs currently skip keyed commoning to
avoid extra hot-path allocation; correctness is unchanged and dead uses can
still disappear.

### Loop-invariant motion

Pure instructions with loop-external inputs move before the loop. An immutable
load in the condition may move because the condition always executes. A
body-only load cannot be eagerly moved across a zero-trip loop.

### Loop buffer-value promotion

A single unambiguous load/update/store path may become a lazy loop-carried
value and one conditional writeback. Promotion stops at synchronization,
atomics, early exits, competing touches, multiple candidates, or places
defined inside the loop.

### Dead definitions

Unused pure values and place paths are removed repeatedly. Stores, atomics,
barriers, and other effects remain when their result is unused.

Flow IR currently receives verification after this Kernel rewrite but no
general rewrite pipeline. Terminal view fusion is a target-planning decision,
not a logical Flow rewrite.

## 11. From logical stages to executable plans

Each backend clones Flow and Kernel IR. For every Flow dispatch it creates a
`PhysicalKernel`:

```text
Entry
Function clone
Workgroup
Storage bindings
Optional parameter block
Coordinate mapping
Optional logical lengths for scalar f16 runtime arrays
```

It can internalize safe repeat, prune now-unused value parameters, select a
portable workgroup, assign dense binding numbers, plan the value block, and
replace exact workgroup-local coordinate arithmetic.

A `ProgramPlan` maps the public program to:

```text
Transients[]       size/lifetime/color
Steps[]            dispatch or target barrier
RepeatBarrier      optional cross-iteration synchronization
Repeat             program | invocation-loop
View               optional terminal projection step and extent
```

A dispatch step refers to a physical-kernel index, domain shapes, resource
sources, and parameter sources. Physical identity is therefore target-specific
and downstream of logical dispatch identity.

A scalar `float16[]`, direct or trailing a struct prefix, may require physical
padding at a host/backend boundary. When its stage uses `.length`, target
planning supplies the checked logical element count and tail path as a hidden
value. WGSL and SPIR-V both consume that value instead of deriving Tach
semantics from the physical binding range. No IR-level padding element is
invented.

The current planner creates one physical kernel for every ordinary dispatch.
It does not deduplicate identical stage clones or fuse adjacent source
dispatches. A narrow terminal-view rule is the exception: when the last
dispatch completely and exclusively writes one transient element at its exact
1D coordinate, its domain equals `width * height`, and the transient is not
otherwise needed, planning rewrites that final store through the shared pack
sequence. Otherwise planning adds one standalone projection kernel. Both
plans implement the same Flow view and the same packed RGBA8 `uint32` word.
WGSL then unpacks those bytes with `unpack4x8unorm` into an `rgba8unorm` texture store. SPIR-V
stores the word in a storage buffer.

## 12. Backend mapping

| Logical concept | WGSL | SPIR-V |
|---|---|---|
| physical entry | private `@compute fn` | private `OpEntryPoint` |
| logical coordinate | selected builtin expression | selected interface builtin |
| plain stage value | reconstructed uniform-block leaf | reconstructed uniform-block leaf |
| immutable value | expression or private local | SSA result ID |
| buffer/shared place | access expression | pointer/access chain |
| load/store | expression/assignment | `OpLoad` / `OpStore` (host-ABI Aligned vs logical Workgroup/Input) |
| `If` result | private mutable local | merge block and `OpPhi` |
| `Loop` carrier | generated mutable local | header `OpPhi` |
| intrinsic | WGSL builtin | core or GLSL.std.450 instruction |
| `float16` | `f16` after `enable f16;` | 16-bit `OpTypeFloat` plus exact capabilities |
| f16/f32 conversion | constructor conversion | `OpFConvert` |
| atomic | WGSL atomic builtin | `OpAtomic*` at QueueFamily or Workgroup |
| source barrier | WGSL barrier | `OpControlBarrier` with availability/visibility |
| plan barrier | ordered WebGPU pass dispatches | `vkCmdPipelineBarrier2` in Tach's Vulkan 1.3 runtime |
| view output | `rgba8unorm` storage texture | packed `uint32[]` RGBA8 scratch |
| terminal view conversion | unpack packed word to `textureStore` | store packed `uint32` |

Backend coordinate optimization may map exact modulo/row-major expressions to
local coordinate inputs and remove dead global inputs. It never changes
logical source/IR names.

## 13. Extension rule

Source sugar belongs in lowering when existing IR already expresses its
meaning. New per-invocation semantics must earn a Kernel IR instruction/type,
verification, effects, optimization rules, and both backend mappings. New
multi-dispatch or terminal-result semantics must earn a Flow node, verifier
rule, target-plan representation, metadata, and both runtime implementations.

Target-only instruction selection, physical layout, and dispatch planning
belong after logical IR. Provider syntax and opcodes never become portable
meaning merely to enable an optimization.
