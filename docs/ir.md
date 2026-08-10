# Tach Core IR

Tach Core IR is the single target-neutral optimization and validation boundary between the source language and all code generators.

## 1. Core model

The IR has five primary concepts:

```text
Type
Value<T>
Place<Space, T, Access>
Region/Block
Resource
```

Values are SSA. Places denote addressable memory. Regions preserve structured control. Resources define the kernel ABI.

## 2. Values

Every ordinary expression computes an immutable typed value with a function-global ID.

Representative value instructions:

```text
const
builtin
unary
binary
convert
composite
extract
intrinsic
call
load
```

Result IDs are unique across the complete function, including nested structured regions. The verifier enforces this independently of ID allocation.

## 3. Places

Places are paths to memory and have no general pointer arithmetic.

```text
place.root       resource root
place.workgroup  workgroup-memory root
place.field      struct field projection
place.index      array/runtime-array/vector-lane projection
```

Effecting operations consume places:

```text
load
store
atomic.*
```

This gives the verifier direct access to address space, stored type, and access mode at every memory operation.

## 4. Structured conditionals

Core IR `if` owns child blocks and may return values.

Source:

```tach
const v = cond ? a : b;
```

Conceptual IR:

```text
%v = if %cond -> T {
  then { yield %a }
  else { yield %b }
}
```

A statement conditional that rebinds lexical locals is represented the same way, with one result per merged local.

WGSL realizes merge results through generated mutable temporaries. SPIR-V realizes them through merge blocks and `OpPhi`.

## 5. Structured loops

A loop owns condition and body regions plus explicit loop-carried parameters.

Conceptually:

```text
%final = loop (%i = %initialI, %acc = %initialAcc) -> T {
  condition {
    %c = ...
    yield %c
  }
  body {
    ...
    continue %nextI, %nextAcc
  }
}
```

The loop header parameters are the semantic counterpart of SPIR-V loop-header phis. The WGSL backend materializes them as mutable locals.

`while` and source `for` both lower into this one representation.

## 6. Terminators

Current internal block terminators are:

```text
yield       return values from a child structured region
continue    provide next loop-carried values
return      leave a function/kernel
unreachable compiler-owned terminal state
```

Structured region ownership makes generic branch labels unnecessary in Core IR.

## 7. Resources

A resource records:

```text
name
kind           uniform | storage
access         read | read_write
group
binding
logical type
```

Physical layout is queried through the ABI/layout layer rather than embedded into logical types.

Compute entry points also record workgroup dimensions and ambient builtin use.

## 8. Workgroup memory

A workgroup declaration records a typed memory object attached to a compute kernel. Fixed arrays and atomics are legal here according to `types.IsWorkgroupStorable`.

Accesses use the same place operations as storage resources, keeping one memory model across the IR.

## 9. Intrinsics

Math intrinsics are semantic IR instructions. They are not SPIR-V extended-op numbers or WGSL function-name strings.

For example:

```text
%r = intrinsic sin %x
%l = intrinsic length %v
%d = intrinsic dot %a, %b
```

The WGSL backend maps them to WGSL builtins. The SPIR-V backend maps them either to core instructions such as `OpDot` or supported GLSL.std.450 extended instructions.

## 10. Portable shifts

Tach defines 32-bit shifts with modulo-32 shift counts. Semantic lowering makes this explicit:

```text
normalized = count & 31u
result     = value << normalized
```

For vector shifts the mask is vector-shaped. This normalization occurs before optimization/backend lowering.

## 11. Atomics and barriers

Atomics are explicit effecting instructions with a place operand and typed value operands/results.

Barriers are explicit semantic effects:

```text
barrier.workgroup
barrier.storage
```

They carry no target-specific scope constants in Core IR. SPIR-V scope/memory-semantics operands are selected by the SPIR-V backend from these semantic operations.

## 12. Verification

`ir.Verify` enforces invariants including:

- valid module/function/kernel declarations
- function-global SSA ID uniqueness
- instruction operand/result typing
- structured region result/terminator typing
- loop carrier consistency
- resource/place type and access correctness
- valid array/vector/field projections
- runtime-array restrictions
- atomic object restrictions
- atomic address-space/access correctness
- intrinsic signatures
- compute-only builtin/workgroup behavior
- barrier placement through uniformity analysis

The verifier is run before optimization, after optimization, and again by the compiler pipeline before code generation.

## 13. Uniformity

Uniformity is derived from IR rather than encoded as source type syntax.

The analysis walks SSA dependencies and structured control, including loop-carried fixed points. Barrier validation consumes that result.

This keeps ordinary arithmetic/type signatures small while still making collective-control legality a compiler-owned semantic property.

## 14. Optimization

The optimizer is intentionally a Core IR transformation layer.

The current dead-value pass:

1. recursively counts SSA uses across nested regions and terminators;
2. removes unused instructions known to be side-effect-free;
3. repeats to a fixed point;
4. leaves loads, calls, structured constructs, memory operations, atomics, and barriers conservative;
5. verifies IR before and after the pass set.

Future passes can compose at this boundary without changing WGSL/SPIR-V frontend semantics.

## 15. Backend lowering summary

```text
Tach Core IR              WGSL                         SPIR-V
──────────────────────    ─────────────────────────    ─────────────────────────
SSA value                 expression/local            result <id>
resource place            global variable reference   pointer + OpAccessChain
load                       value use                   OpLoad
store                      assignment                  OpStore
structured if             if                          OpSelectionMerge + branch
if results                 temporary assignment        OpPhi
loop carriers              mutable locals              loop-header OpPhi
loop                       loop                        OpLoopMerge + branches
builtin                    @builtin input              BuiltIn decoration
resource group             @group                      DescriptorSet
resource binding           @binding                    Binding
math intrinsic             builtin call                core/GLSL.std.450 op
atomic                     atomic builtin              OpAtomic*
barrier                    barrier builtin             OpControlBarrier
```
