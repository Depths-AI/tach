# Tach Core IR

Tach Core IR is the compiler's target-independent contract. Semantic analysis
lowers source into it; optimization proves and rewrites it; WGSL, SPIR-V, and
host-binding generation all consume it. It contains Tach semantics, never
WGSL syntax, SPIR-V opcodes, or backend binding coordinates.

The `.tir` artifact and `tach ir` command print this IR for inspection. Textual
IR is diagnostic output, not another accepted input language.

Kernel authors do not need to learn this notation. When debugging a dump,
remember two marks: `%3` is an immutable value and `&p3` is a path to memory.
The rest of this guide explains why that distinction makes control flow,
optimization, and two very different shader backends fit one semantic model.

## 1. From source to Core IR

Consider a complete kernel:

```tach
export function scale[i](
  data: buffer<float32[]>,
  factor: float32,
) {
  if (i < data.length) {
    data[i] *= factor;
  }
}
```

Its optimized dump has this shape:

```text
buffer @0 data: float32[] access=mutable

compute @scale[i=%1](data=@0, factor=%2: float32) workgroup(256,1,1) {
  &p1 = place.resource @0 : float32[]
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

This example exposes the central distinctions:

- `%N` identifies an immutable typed value;
- `&pN` identifies a typed addressable place;
- buffers live at module scope;
- the kernel's logical coordinate is an ordinary `uint32` parameter;
- the plain `factor` input is an ordinary immutable value;
- memory reads and writes are explicit; and
- control flow remains structured.

## 2. Module model

An IR module contains three ordered collections:

```text
Module
  Structs
  Resources
  Functions
```

Structs retain logical field names and types. Resources are the module's
external buffers. Functions include value helpers and exported compute entry
points.

### Buffer resources

Each buffer records:

```text
name
type       logical buffer type
access     read | mutable
source span
```

Buffer order is module-global and deterministic. A kernel keeps a
`KernelParams` list that preserves source parameter order and maps each entry
to either a module buffer or an SSA value parameter. Physical bindings and byte
layout are derived later by the ABI layer; neither is encoded in a logical IR
type. Buffer access is inferred from lowered effects, and the verifier rejects
stores through read-only roots.

### Functions

A helper function records typed value parameters, a result type, and a body.
A kernel records:

- one to three logical index parameters;
- its three-axis workgroup size;
- its ordinary immutable value parameters;
- its source-order buffer/value mapping;
- its workgroup-memory declarations; and
- a `void` body.

Logical indices are explicit `uint32` parameters in Core IR. There is no Core IR
notion of a WGSL builtin variable or SPIR-V `BuiltIn` decoration. A backend
privately chooses the target input needed to produce each used coordinate.

## 3. Values and places

Keeping values and memory locations separate is the foundation of the IR.

### Values

A value is immutable, typed, and defined once. `ValueID` zero is reserved;
ordinary definitions use nonzero IDs. IDs are unique across a whole function,
including nested branches and loops, so a printed `%12` has one meaning within
that function.

Value-producing instructions are:

| Instruction | Meaning |
|---|---|
| `Const` | canonical scalar literal |
| `Unary` | typed unary operation |
| `Binary` | typed binary operation |
| `Convert` | explicit numeric conversion |
| `Composite` | vector or struct assembly |
| `Extract` | constant field or lane extraction from a value |
| `VectorIndex` | dynamic lane extraction from a vector value |
| `Call` | direct pure helper call |
| `Intrinsic` | backend-independent math operation |
| `Load` | read a value from a place |
| `ArrayLength` | query a runtime array through a place |
| `Atomic` | atomic result, except for `atomicStore` |

An instruction stores its result type directly. Backends never have to infer a
literal type or reconstruct overload resolution.

### Places

A place describes a path to addressable GPU memory. It is not a first-class
pointer and supports no pointer arithmetic, casting, or comparison.

| Instruction | Meaning |
|---|---|
| `PlaceRoot` | root a path at a buffer resource |
| `PlaceWorkgroup` | root a path at a shared variable |
| `PlaceField` | project a struct field or constant vector lane |
| `PlaceIndex` | index an array, runtime array, or vector dynamically |

`Load`, `Store`, `ArrayLength`, and atomic instructions consume places. The
place chain makes the root buffer, address space, element type, and write
permission recoverable at every memory operation.

This distinction also explains source locals. Rebinding a source `let` does
not create a hidden memory cell. It creates new values and structured merge or
loop results. Only buffers and shared variables become places. Plain kernel
parameters are values from their first Core IR appearance onward.

## 4. Structured control flow

Core IR preserves source structure instead of flattening control into a
general control-flow graph.

### Blocks and terminators

A block is an instruction list followed by exactly one terminator:

| Terminator | Role |
|---|---|
| `Yield` | return values from an `if` branch or loop condition region |
| `Continue` | supply the next values of all loop-carried parameters |
| `Return` | leave a helper or kernel |
| `Unreachable` | compiler-owned terminal state |

Child regions cannot fall through accidentally; their terminator states how
control and values return to the owning instruction.

### Conditionals

An `If` owns a condition value, a then block, an else block, and zero or more
typed results:

```text
%result = if %condition -> float32 {
  ...
  yield [%thenValue]
} else {
  ...
  yield [%elseValue]
}
```

A ternary expression produces one result. A statement `if` can produce several
results when it rebinds several surrounding locals. Both branches yield the
same arity and types.

WGSL lowering materializes these results with compiler-private mutable locals.
SPIR-V lowering creates merge blocks and `OpPhi` instructions. Core IR needs
neither representation.

### Loops

A `Loop` owns initial values, typed loop parameters, a condition region, a body
region, and final results:

```text
loop params=[(%index <- %initialIndex), (%sum <- %initialSum)] {
  cond
    ...
    yield [%keepGoing]
  body
    ...
    continue [%nextIndex, %nextSum]
}
```

The condition yields one `bool`. The body's `Continue` supplies one next value
for every parameter. When the condition is false, the current parameters
become the loop results. Source `while` and `for` both lower to this single
form.

Loop parameters are the semantic equivalent of loop-header phi nodes. WGSL
uses generated mutable locals; SPIR-V uses `OpPhi`. This is why optimization
can reason about a loop once without adopting either backend's mechanics.

### Short-circuiting and early return

`&&`, `||`, and conditional expressions lower through structured `If`
regions, so only the selected operand or branch executes. A source `return`
terminates its current block directly. Semantic analysis rejects unreachable
source statements and loops without a continuing path.

## 5. Types and representation boundaries

Core IR uses resolved logical Tach types:

```text
bool, int32, uint32, float32
numeric vectors with 2, 3, or 4 lanes
named structs
atomic<int32>, atomic<uint32>
fixed arrays
runtime arrays
void
```

Logical types express program meaning. They do not contain host padding,
descriptor decorations, WGSL wrapper structs, or SPIR-V storage-class pointer
types.

The layout package independently computes host-visible alignment, offsets,
strides, and minimum buffer sizes. The ABI planner also flattens each kernel's
plain values into one physical parameter block without changing Core IR. Each
backend establishes an explicit boundary between logical values and physical
buffer/block memory. This prevents padding and backend storage rules from
contaminating helper signatures or optimization.

## 6. Effects and synchronization

The instructions with observable memory or execution effects are:

- `Store`;
- every `Atomic` operation;
- `BarrierWorkgroup`; and
- `BarrierBuffer`.

Loads from mutable buffers and shared memory are not freely reusable because an
effect may change the addressed value. Loads from inferred read-only buffers
may be reused when the place and structured dominance match. Plain kernel
parameters require no memory load at all; they are immutable SSA values.

### Atomics

An atomic instruction records a Tach operation, an atomic place, an underlying
`int32` or `uint32` type, and any value operand/result. The portable operation set
is load, store, add, subtract, minimum, maximum, bitwise and/or/xor, and
exchange. Target scope and memory-semantics operands are backend decisions, not
Core IR operands.

### Barriers and uniformity

Barrier instructions record the memory domain being synchronized. The
uniformity analysis classifies each available value conservatively as uniform
or varying within a workgroup and follows structured branches plus loop-carried
fixed points.

Constants and plain kernel parameters are uniform. Logical coordinates,
buffer/shared-memory loads, and atomic results are varying. A barrier is valid
only when all enclosing control decisions are uniform.

### Workgroup zero initialization

Zero initialization is a Tach semantic invariant, not an explicit Core IR
instruction. WGSL already provides the required behavior. SPIR-V lowering
emits a first-local-invocation initialization prologue followed by a barrier
before source instructions execute.

## 7. Verification

`ir.Verify` is the executable contract for Core IR. It checks, among other
invariants:

- well-formed modules, buffers, functions, and workgroup sizes;
- unique, nonzero value and place definitions;
- availability of every operand at its structured use;
- exact operand, result, branch-yield, and loop-carrier types;
- direct helper-call signatures and valid returns;
- valid buffer and shared-memory roots;
- field, array, runtime-array, and vector projections;
- place-root access and legal loads/stores;
- runtime-sized and nonconstructible type restrictions;
- atomic object, type, address-space, and access rules;
- intrinsic signatures;
- compute-only workgroup operations; and
- barrier control-flow uniformity.

Semantic lowering verifies its result through the optimizer's precondition.
The optimizer verifies again after all target-independent passes. WGSL,
SPIR-V, and binding generation also refuse an invalid module at their
boundaries. A backend therefore never treats malformed IR as a recoverable
input variant.

## 8. Target-independent optimization

`opt.Run` applies one fixed pipeline to every function:

```text
verify
common values and places
loop-invariant code motion
common values and places
loop buffer-value promotion
common values and places
dead definition elimination to a fixed point
verify
```

### Common values and places

The pass reuses exact repeated constants, unary/binary operations, conversions,
intrinsics, extracts, pure helper calls, and small composites where structured
dominance makes reuse valid. It also reuses identical place paths, array
lengths, and loads from inferred read-only buffers.

Mutable loads, atomics, stores, and barriers are not commoned. Composites and
calls wider than four operands currently skip commoning to avoid allocating a
key in the hot path; they remain correct and may still become dead.

### Loop-invariant code motion

Pure instructions whose operands are defined outside a loop move before it.
An immutable buffer load from the condition region can move because the
condition executes before deciding whether to enter. A body-only load is not
eagerly hoisted across a possible zero-trip loop.

### Loop buffer-value promotion

The optimizer can replace a repeated load/update/store of one unambiguous
buffer place with a loop-carried value. It performs the first load lazily,
keeps subsequent iterations in SSA, and writes back only if the loop ran.

Promotion is deliberately conservative. It stops when the loop contains
synchronization, atomics, an early exit, another touch of the same buffer,
multiple candidate loads/stores, or a place defined inside the loop. This
preserves memory order and zero-trip behavior without speculative alias
analysis.

### Dead definitions

Unused side-effect-free values and unused place paths are removed repeatedly
until use counts stop changing. Effects remain even when their returned value
is unused.

## 9. Backend-specific lowering and optimization

After Core optimization, each backend creates a private lowered program. The
current shared backend analysis begins with every logical index as a global
coordinate and then recognizes two exact Tach expressions:

```text
const localX = x % workgroupWidth;
const local = localX + localY * workgroupWidth
            + localZ * workgroupWidth * workgroupHeight;
```

When the constants exactly match the kernel's workgroup dimensions, the
backend replaces the first form with a local coordinate and the row-major form
with a local linear coordinate. Dead coordinate arithmetic and unused target
inputs disappear. If the expression does not match exactly, it remains normal
Tach arithmetic over the global logical coordinate.

This is the second optimization stage:

```text
Core IR optimization       semantics shared by every target
backend IR optimization    representation choices for one target
```

The source and Core IR never acquire provider-specific invocation objects to
obtain this optimization.

## 10. Backend mapping

| Core concept | WGSL lowering | SPIR-V lowering |
|---|---|---|
| logical index | selected builtin input expression | selected interface builtin |
| plain kernel value | reconstructed from one parameter block | reconstructed from one parameter block |
| immutable value | expression or generated local | SSA result ID |
| buffer/shared place | access expression | pointer plus access chain |
| load/store | value read/assignment | `OpLoad` / `OpStore` |
| structured `If` | WGSL `if` | selection merge, branches, `OpPhi` |
| structured `Loop` | generated locals and `loop` | loop merge, branches, `OpPhi` |
| intrinsic | WGSL builtin | core or GLSL.std.450 operation |
| atomic | WGSL atomic builtin | `OpAtomic*` |
| barrier | WGSL barrier builtin | `OpControlBarrier` |

WGSL's validator reparses the exact emitted subset and checks serialization.
The SPIR-V validator independently decodes the binary and checks structure,
types, layout, control flow, dominance, phi edges, effects, and the exact used
entry-point interface.

## 11. Extension rule

A new source convenience should lower into existing IR whenever its semantics
already fit. `for`, ternary expressions, compound assignment, and swizzles do
this today.

A genuinely new portable capability must first earn a target-independent type,
instruction, verifier rule, and optimizer effect rule. Only then should each
backend lower it. Backend spellings and encodings never become Tach source or
Core IR semantics.
