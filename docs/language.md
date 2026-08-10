# Tach language reference

This document describes the source surface implemented by the current Tach compiler.

## 1. Module declarations

A module consists of struct type declarations, pure helper functions, and exported compute kernels.

```tach
type Pair = {
  x: f32,
  y: f32,
};

fn magnitudeSquared(v: vec2f): f32 {
  return dot(v, v);
}

@workgroupSize(64)
export compute run(data: storage<vec2f[], read_write>) {
  // ...
}
```

## 2. Types

### Scalars

```text
bool
i32
u32
f32
```

### Vectors

```text
vec2f vec3f vec4f
vec2u vec3u vec4u
vec2i vec3i vec4i
```

Vector lanes are available through `x`, `y`, `z`, `w` and through indexing.

```tach
const p = vec3f(1.0, 2.0, 3.0);
const y = p.y;
const z = p[2u];
```

Storage/workgroup vector lanes are addressable as places:

```tach
data[i].x = 1.0;
data[i][1u] = 2.0;
```

### Structs

```tach
type Particle = {
  position: vec4f,
  velocity: vec4f,
};
```

Construction uses object-shaped syntax:

```tach
const p: Particle = {
  position: vec4f(0.0, 0.0, 0.0, 1.0),
  velocity: vec4f(1.0, 0.0, 0.0, 0.0),
};
```

### Atomics

```text
atomic<i32>
atomic<u32>
```

Atomic objects are memory objects. They are accessed with atomic intrinsics rather than ordinary whole-object loads/stores.

### Arrays

Runtime-sized arrays are used for storage resources:

```text
Particle[]
u32[]
```

Fixed arrays are used for workgroup memory:

```text
u32[256]
atomic<u32>[64]
```

## 3. Numeric literals

Tach accepts decimal, hexadecimal, and binary integer literals, `_` separators, decimal floating exponents, and explicit suffixes.

```tach
const a = 42u;
const b = 0xffu;
const c = 0b1010_0001u;
const d = 1.25;
const e = 6.022e2;
```

Literal spelling is canonicalized during semantic lowering, so downstream IR contains concrete typed constants.

## 4. Variables

`const` binds an immutable local value:

```tach
const n = 64u;
```

`let` creates a rebindable lexical local:

```tach
let sum = 0.0;
sum += x;
```

A Tach `let` does not imply stack memory. The compiler turns local rebinding into SSA values and structured region/loop results.

Optional type annotations are supported:

```tach
const scale: f32 = 0.5;
let index: u32 = 0u;
```

## 5. Functions

Helper functions are value-oriented and pure with respect to kernel resources:

```tach
fn integrate(p: Particle, dt: f32): Particle {
  return {
    position: p.position + p.velocity * dt,
    velocity: p.velocity,
  };
}
```

Return type can use `:` or `->`:

```tach
fn square(x: f32): f32 { return x * x; }
fn cube(x: f32) -> f32 { return x * x * x; }
```

A missing return annotation means `void`.

## 6. Compute kernels

```tach
@workgroupSize(256)
export compute update(
  state: storage<State[], read_write>,
  params: uniform<Params>,
) {
  // ...
}
```

`@workgroupSize(x)`, `@workgroupSize(x, y)`, and `@workgroupSize(x, y, z)` define local workgroup dimensions.

### Compute builtins

Compute kernels have ambient typed builtins:

```text
globalId      : vec3u
localId       : vec3u
localIndex    : u32
workgroupId   : vec3u
numWorkgroups : vec3u
```

They are compiler inputs, not resources.

## 7. Resources

### Storage buffers

```tach
values: storage<f32[], read>
out: storage<f32[], read_write>
```

Storage access is part of the type and is enforced by semantic/IR validation.

### Uniform buffers

```tach
params: uniform<Params>
```

Uniform resources are read-only.

Every exported compute kernel has at least one storage parameter. Besides
making the kernel's result observable, that storage buffer carries the
`GPUDevice` used by the generated direct JavaScript function.

### Binding locations

Bindings may be explicit:

```tach
@group(2) @binding(5) data: storage<u32[], read_write>
```

When omitted, Tach assigns module-global group/binding locations deterministically.

A Tach group maps to WGSL `@group` and the SPIR-V descriptor set. A Tach binding maps to WGSL `@binding` and the SPIR-V binding decoration.

## 8. Runtime array length

Runtime storage arrays expose `.length`:

```tach
if (globalId.x < values.length) {
  // ...
}
```

## 9. Workgroup memory

Workgroup variables are declared inside a compute kernel:

```tach
workgroup scratch: f32[256];
workgroup counters: atomic<u32>[64];
```

They lower to module-scope workgroup storage in target shader representations while remaining lexically associated with the kernel in Tach source/IR.

Every Workgroup variable has its type's zero value when kernel execution begins.
WGSL provides this guarantee directly. The SPIR-V backend emits an explicit,
uniform initialization prologue followed by a Workgroup barrier, so the same
rule holds on the Vulkan 1.1 baseline without optional device features.

## 10. Control flow

### `if` / `else if` / `else`

```tach
if (x < 0.0) {
  y = -x;
} else if (x > 1.0) {
  y = 1.0;
} else {
  y = x;
}
```

Branches that rebind locals produce structured SSA merge values in Core IR.

### Ternary expression

```tach
const sign = x < 0.0 ? -1.0 : 1.0;
```

Ternaries lower to value-producing structured conditionals.

### `while`

```tach
let i = 0u;
let total = 0u;
while (i < count) {
  total += values[i];
  i++;
}
```

Rebound locals become loop-carried SSA values.

### `for`

Tach supports canonical counted loops:

```tach
for (let i = 0u; i < count; i++) {
  out[i] = i;
}
```

The initializer is a `const`/`let` declaration. The update is assignment, compound assignment, `++`, or `--`. The construct lowers into the same Core IR loop representation as `while`.

### `return`

```tach
if (i >= values.length) {
  return;
}
```

Helper functions return typed values; compute kernels return `void`.

## 11. Operators

### Arithmetic

```text
+  -  *  /  %
```

### Comparison

```text
==  !=  <  <=  >  >=
```

### Boolean

```text
!  &&  ||
```

`&&` and `||` preserve short-circuit evaluation through structured IR.

### Integer bitwise

```text
~  &  |  ^  <<  >>
```

Shifts have Tach-defined modulo-32 counts. Semantic lowering normalizes the count before either backend sees the operation.

### Assignment

```text
=  +=  -=  *=  /=  %=
&= |= ^= <<= >>=
++ --
```

Assignments may rebind mutable locals or store through writable places.

## 12. Constructors and conversions

Scalar conversions use type-call syntax:

```tach
const x = f32(i);
const u = u32(x);
const s = i32(u);
```

Vectors use the same shape:

```tach
const a = vec3f(1.0, 2.0, 3.0);
const b = vec4u(1u, 2u, 3u, 4u);
```

## 13. Math intrinsics

### Scalar/vector floating point

```text
abs
floor
ceil
trunc
sin
cos
tan
exp
exp2
log
log2
sqrt
inverseSqrt
pow
```

`abs` also supports signed integers of matching scalar/vector shape.

### Integer bounds

```text
min
max
clamp
```

These operate on integer scalar/vector values in Tach's current portable semantic profile.

### Vector math

```text
dot
length
distance
cross
normalize
```

`cross` is defined for `vec3f`; the other vector operations use floating vectors with matching shapes.

## 14. Atomics

Tach exposes:

```text
atomicLoad(place)
atomicStore(place, value)
atomicAdd(place, value)
atomicSub(place, value)
atomicMin(place, value)
atomicMax(place, value)
atomicAnd(place, value)
atomicOr(place, value)
atomicXor(place, value)
atomicExchange(place, value)
```

Atomic operations require `atomic<i32>` or `atomic<u32>` memory in writable storage or workgroup address space.

## 15. Barriers

```tach
workgroupBarrier();
storageBarrier();
```

Tach's uniformity analysis verifies barrier control-flow legality before backend lowering.

## 16. Host-visible values

Host-shareable values are numeric scalars, numeric vectors, structs composed from host-shareable fields, atomics in storage, and storage runtime arrays.

`bool` remains a control/value type and does not have a Tach host ABI representation.

Physical layout is compiler-owned; see [`abi.md`](abi.md).
