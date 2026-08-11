# Tach language reference

This document is the source-language contract implemented by the current Tach
compiler. It is written for kernel authors; target encodings and host byte
layout are covered separately in [the ABI reference](abi.md).

## 1. A complete kernel

```tach
type Params = {
  dt: f32,
  count: u32,
};

type Particle = {
  position: f32x4,
  velocity: f32x4,
};

fn integrateParticle(p: Particle, dt: f32): Particle {
  return {
    position: p.position + p.velocity * dt,
    velocity: p.velocity,
  };
}

export compute integrate[i](
  particles: buffer<Particle[]>,
  params: uniform<Params>,
) {
  if (i >= params.count) {
    return;
  }
  particles[i] = integrateParticle(particles[i], params.dt);
}
```

A module may contain named struct types, value-only helper functions, and
exported compute kernels. Declaration order does not restrict helper calls or
type references: semantic analysis first collects declarations, then resolves
and lowers them. Recursive value types and recursive function call cycles are
rejected.

## 2. Lexical form

Identifiers begin with a Unicode letter or `_` and continue with Unicode
letters, digits, or `_`. Public type, kernel, and resource-parameter names must
also be portable JavaScript/TypeScript identifiers because every successful
compilation emits host bindings. Compiler builtin names are reserved as call
targets.

Whitespace is insignificant. Tach supports line comments and nested block
comments:

```tach
// One line.

/* A block comment.
   /* Nested blocks are valid. */
*/
```

Statements end in `;`. A trailing comma is accepted in parameter and
struct-literal lists. Type fields may be separated by `,` or `;`, including a
trailing separator; the examples use commas consistently.

## 3. Module declarations

### Struct declarations

```tach
type Pair = {
  x: f32,
  y: f32,
};
```

Field names must be unique. Structs are value types: recursive structs are not
allowed. A runtime array may occur only once, as the final field, and its
element must have a fixed footprint:

```tach
type Samples = {
  count: u32,
  values: f32[],
};
```

Such a runtime-tailed struct can be addressed in a buffer but cannot be loaded,
constructed, passed, or returned as one value.

### Helper functions

```tach
fn square(x: f32): f32 {
  return x * x;
}

fn observe(x: f32) {
  const ignored = x;
}
```

Parameters and results must be constructible value types. An omitted result
annotation means `void`. A non-void helper must return on every path; a void
helper receives an implicit final `return` when needed.

Helpers cannot receive resources, declare workgroup memory, use barriers, or
call a compute entry point. They can call other helpers, but recursion is
rejected. Those rules make helpers pure with respect to GPU-visible state.

### Compute kernels

```tach
export compute clear[i](values: buffer<u32[]>) {
  if (i < values.length) {
    values[i] = 0;
  }
}
```

Every compute kernel:

- is exported;
- declares one, two, or three logical coordinate names in `[...]`;
- receives only `buffer<T>` and `uniform<T>` parameters;
- has at least one buffer parameter; and
- returns `void`.

Compute entry points cannot be called from Tach code.

## 4. Logical coordinates and workgroups

Coordinates are immutable `u32` values whose names and rank belong to the
kernel:

```tach
export compute line[i](out: buffer<u32[]>) {
  if (i < out.length) {
    out[i] = i;
  }
}

export compute image[x, y](out: buffer<f32x4[]>) {
  const pixel = y * 1920 + x;
  if (pixel < out.length) {
    out[pixel] = f32x4(0, 0, 0, 1);
  }
}

export compute volume[x, y, z](out: buffer<f32[]>) {
  // x, y, and z are ordinary Tach values.
}
```

The body does not see target invocation objects or provider-specific builtins.
At dispatch time the host supplies a logical extent of exactly the same rank.
Workgroup counts are rounded up, so edge invocations can fall outside that
extent. Kernels must guard buffer and problem-domain bounds themselves.

When no attribute is present, rank selects a portable 256-invocation default:

| Rank | Default workgroup |
|---:|---:|
| 1 | `256, 1, 1` |
| 2 | `16, 16, 1` |
| 3 | `8, 8, 4` |

Use `@workgroup(...)` only when an algorithm depends on a tile shape or shared
memory protocol:

```tach
@workgroup(16, 16)
export compute tiled[x, y](out: buffer<f32[]>) {
  // ...
}
```

The attribute takes one to `rank` positive integer literals. Omitted trailing
axes become `1`. Tach's portable limits are:

- `x <= 256`;
- `y <= 256`;
- `z <= 64`; and
- `x * y * z <= 256`.

These are compiler guarantees, not WebGPU spellings. For example, a tiled
kernel may write `x % 16` and `y % 16`; backend optimization can recognize and
replace that arithmetic with its target's efficient local-coordinate input.

## 5. Types

### Scalar types

| Type | Meaning |
|---|---|
| `bool` | control-flow boolean |
| `i32` | signed 32-bit integer |
| `u32` | unsigned 32-bit integer |
| `f32` | 32-bit floating point |
| `void` | absence of a helper result; not a value or field type |

`bool` is a value/control type but is not host-shareable, so it cannot appear
in a buffer or uniform layout.

### Vector types

```text
f32x2  f32x3  f32x4
i32x2  i32x3  i32x4
u32x2  u32x3  u32x4
```

Vectors contain two to four lanes of one numeric scalar type. Read a single
lane, a multi-lane swizzle, or a dynamically indexed lane:

```tach
const p = f32x4(1, 2, 3, 4);
const first = p.x;
const reversed = p.wzyx;
const lane = p[index];
```

Swizzle letters are `x`, `y`, `z`, and `w` and must exist in the source vector.
Multi-lane swizzles are values. A single lane of an addressable resource or
workgroup vector is also an assignable place:

```tach
values[i].x = 1.0;
values[i][1] = 2.0;
```

### Struct types

Struct values contain their declared fields and use contextual object-shaped
construction:

```tach
type Color = {
  rgb: f32x3,
  alpha: f32,
};

const color: Color = {
  alpha: 1,
  rgb: f32x3(0.2, 0.4, 0.8),
};
```

Field order in a literal is irrelevant. Every field must appear exactly once;
missing, duplicate, and unknown fields are errors. A struct literal needs an
expected struct type from an annotation, assignment, argument, or return.

### Atomic types

```text
atomic<i32>
atomic<u32>
```

An atomic is a memory object, not an ordinary value. It may live in a buffer or
workgroup variable and is accessed only through atomic operations. Whole-value
construction, load, store, parameters, and return are rejected.

### Arrays

Runtime arrays have no source-level fixed count:

```tach
values: buffer<f32[]>
particles: buffer<Particle[]>
```

They are buffer-only addressable types and expose `.length`:

```tach
if (i < values.length) {
  values[i] = 0.0;
}
```

Fixed arrays carry a positive literal count and are currently workgroup-memory
types:

```tach
workgroup tile: f32[256];
workgroup counters: atomic<u32>[64];
```

Array and vector indexing accepts an `i32` or `u32` scalar. Tach does not insert
dynamic bounds checks.

### Type domains

The compiler distinguishes three useful domains:

- **constructible values:** scalars, vectors, and fixed-footprint structs;
- **host-shareable values:** numeric scalars, numeric vectors, atomics, structs
  composed from them, and runtime arrays/tails; and
- **workgroup-storable values:** numeric scalars, vectors, atomics, fixed
  arrays, and fixed-footprint structs composed from them.

Runtime-sized types are addressable but not constructible. Fixed arrays are
workgroup-storable but are not currently admitted to host resources.

## 6. Resources

### Buffers

```tach
values: buffer<f32[]>
state: buffer<State>
```

Buffers may be read or mutated. Source does not declare an access qualifier;
semantic lowering infers read-only versus mutable access from stores and
non-load atomic operations. The verified access becomes part of generated
WGSL, SPIR-V, metadata, and host pipeline layout.

Atomic-containing storage is physically read/write even when the only source
operation is `atomicLoad`, because the target resource class must support
atomic access.

### Uniforms

```tach
params: uniform<Params>
factor: uniform<f32>
```

Uniforms are read-only and fixed-size. They cannot contain runtime arrays or
atomics.

### Identity and aliasing

Parameters are positional and retain their source names in generated host
functions. Physical group/set and binding numbers are compiler-owned and do
not appear in Tach source.

Different resource parameters are non-aliasing. Managed TypeScript bindings
reject passing one `ComputeBuffer` to two parameters of the same command;
native callers must honor the same rule. In-place algorithms should use one
buffer parameter and read and write through it.

## 7. Numeric literals and inference

Numeric literals carry values, not target-language suffixes:

```tach
const decimal = 42;
const separated = 1_000_000;
const hexadecimal = 0xff00_ff00;
const binary = 0b1010_0001;
const fraction = 1.25;
const exponent = 6.022e2;
```

Suffixes such as `0u`, `1i`, and `1.0f` are rejected. Context supplies the type
whenever an assignment, annotation, argument, constructor, branch, or operator
already requires one. Without context:

- a non-negative whole-number literal is `u32`;
- a decimal or exponent literal is `f32`; and
- unary `-` gives an otherwise untyped whole literal `i32` context.

Inference is independent of operand order. For example, both `1 + 2.0` and
`2.0 + 1` are `f32`, and `condition ? 1 : -2` has matching `i32` branches.

Use scalar constructors for explicit conversion:

```tach
const signed = i32(unsignedValue);
const unsigned = u32(signedValue);
const real = f32(integerValue);
```

There are no general implicit numeric conversions. Contextual typing of a
literal and defined scalar-to-vector broadcasting are the only conveniences.
All literals are range-checked and canonicalized before entering Core IR.

## 8. Constructors

### Scalar conversion

`i32(x)`, `u32(x)`, and `f32(x)` accept one numeric scalar. Converting between
`i32` and `u32` preserves the 32-bit pattern; integer/float conversions use the
target-independent operation selected by Tach Core IR.

### Vector construction

A vector constructor accepts numeric scalars and vectors, flattens them in
order, converts each lane to the destination scalar type, and requires exactly
the destination lane count:

```tach
const splat = f32x4(1);                  // [1, 1, 1, 1]
const joined = f32x4(f32x2(1, 2), 3, 4);
const converted = u32x3(1.0, 2.0, 3.0);
```

One scalar splats to every lane. Passing one already matching vector is an
identity operation.

## 9. Variables and scope

`const` creates an immutable local binding:

```tach
const radius = 4.0;
```

`let` creates a rebindable local binding:

```tach
let total = 0.0;
total += sample;
```

Both may have an explicit type annotation. A `let` does not imply stack or
private memory: the compiler rewrites local rebinding into SSA values, branch
results, and loop-carried values.

Names cannot shadow an existing name in the active scope. Branch-local names
do not escape their branch. A `for` initializer is scoped to that loop;
mutations of locals that existed before the loop remain visible afterward.

Only resources and workgroup variables are addressable GPU memory. Ordinary
locals are values.

## 10. Expressions and operators

### Postfix expressions

Function/constructor calls, member access, and indexing compose from left to
right:

```tach
shape(value)
particle.position.x
blocks[i].values[j]
```

Call targets are direct function, intrinsic, atomic, barrier, or type names;
function values and method calls are not part of the language.

### Unary operators

| Operator | Operand |
|---|---|
| `!` | `bool` |
| `-` | `i32`, `f32`, or a vector of either |
| `~` | `i32`, `u32`, or an integer vector |

Unsigned negation is deliberately rejected.

### Binary operators

| Family | Operators | Accepted operands |
|---|---|---|
| arithmetic | `+ -` | matching numeric scalars/vectors; scalar broadcast with a vector |
| multiplication | `*` | matching numeric values, vector/scalar, or scalar/vector |
| division | `/` | matching numeric values; scalar broadcast with a vector |
| remainder | `%` | matching numeric scalars |
| comparison | `== != < <= > >=` | matching numeric scalars; result is `bool` |
| boolean | `&& ||` | `bool`; short-circuiting |
| bitwise | `& \| ^` | matching integer scalars/vectors; scalar broadcast with a vector |
| shift | `<< >>` | integer scalar/vector shifted by unsigned count of matching shape |

For vector shifts, a scalar `u32` count is broadcast. Every 32-bit shift masks
its count to the low five bits, so Tach defines large counts modulo 32 before a
backend sees them. Right shift is arithmetic for `i32` and logical for `u32`.

Operator precedence, from lowest to highest, is:

```text
?:
||
&&
|
^
&
== !=
< <= > >=
<< >>
+ -
* / %
unary ! - ~
postfix call, member, index
```

Binary operators are left-associative. The conditional operator is
right-associative through its expression branches.

### Assignment

```text
=  +=  -=  *=  /=  %=
&=  |=  ^=  <<=  >>=
++  --
```

Plain `=` and compound assignments work on `let` bindings and addressable
resource/workgroup places. `++` and `--` add or subtract one using the target's
existing numeric type. Assigning to `const` is an error.

## 11. Control flow

### Conditional statements

```tach
if (value < 0.0) {
  value = -value;
} else if (value > 1.0) {
  value = 1.0;
} else {
  value += 0.5;
}
```

Conditions must be `bool`. Locals rebound by either branch become structured
merge values.

### Conditional expressions

```tach
const sign = value < 0.0 ? -1.0 : 1.0;
```

Both branches must produce the same concrete type. Evaluation is
short-circuiting: only the selected branch executes.

### `while`

```tach
let i = 0;
let sum = 0.0;
while (i < count) {
  sum += values[i];
  i++;
}
```

Rebound locals become explicit loop-carried values. A loop body must retain a
continuing path; an unconditionally returning body is rejected.

### `for`

```tach
for (let lane = 0; lane < 4; lane++) {
  sum += values[base + lane];
}
```

The initializer must be a `let` declaration. The condition must be `bool`. The
update must be assignment, compound assignment, `++`, or `--`. `for` is source
sugar and lowers to exactly the same structured loop representation as
`while`.

Source-level `break` and `continue` are not currently part of Tach.

### Return

```tach
if (i >= values.length) {
  return;
}
```

Compute kernels use bare `return;`. Helpers return a value matching their
declared result, or bare `return;` when `void`. Statements after an
unconditional return are rejected as unreachable.

## 12. Math intrinsics

Intrinsics are Tach operations with backend-independent type rules. Names are
reserved and cannot be redefined.

### Floating scalar/vector operations

The following accept `f32` or any `f32xN` and preserve the input type:

```text
floor  ceil  trunc
sin    cos   tan
exp    exp2  log  log2
sqrt   rsqrt
```

`abs` accepts `i32`, `f32`, `i32xN`, or `f32xN` and preserves its type.

`pow(base, exponent)` accepts two matching `f32` scalar/vector values. When
the first argument is a vector, a scalar second argument is broadcast.

### Integer bounds

```text
min(a, b)
max(a, b)
clamp(value, low, high)
```

These accept matching `i32`/`u32` scalars or integer vectors and preserve the
type. A scalar bound is broadcast when the first argument is a vector.
Floating `min`, `max`, and `clamp` are intentionally unavailable until Tach
defines one portable NaN and signed-zero policy.

### Vector operations

| Intrinsic | Operands | Result |
|---|---|---|
| `dot(a, b)` | matching `f32xN` | `f32` |
| `length(v)` | `f32xN` | `f32` |
| `distance(a, b)` | matching `f32xN` | `f32` |
| `cross(a, b)` | matching `f32x3` | `f32x3` |
| `normalize(v)` | `f32xN` | same vector type |

## 13. Atomics

Atomic operations require an addressable `atomic<i32>` or `atomic<u32>` place:

| Operation | Meaning | Result |
|---|---|---|
| `atomicLoad(place)` | read | current value |
| `atomicStore(place, value)` | write | `void` |
| `atomicAdd(place, value)` | add | previous value |
| `atomicSub(place, value)` | subtract | previous value |
| `atomicMin(place, value)` | minimum | previous value |
| `atomicMax(place, value)` | maximum | previous value |
| `atomicAnd(place, value)` | bitwise AND | previous value |
| `atomicOr(place, value)` | bitwise OR | previous value |
| `atomicXor(place, value)` | bitwise XOR | previous value |
| `atomicExchange(place, value)` | replace | previous value |

Storage-buffer atomics use device scope; workgroup-memory atomics use
workgroup scope. Atomic operations are relaxed. Ordering between invocations
comes from explicit barriers or the dependency semantics of the atomic
operation itself.

## 14. Workgroup memory and barriers

Workgroup variables are declared inside a compute kernel and are shared by the
invocations of one workgroup:

```tach
@workgroup(64)
export compute reduce[i](out: buffer<u32[]>) {
  workgroup partial: u32[64];
  const lane = i % 64;

  partial[lane] = i;
  workgroupBarrier();
  // ...
}
```

Every workgroup variable has its type's zero value before source instructions
execute. Tach establishes this invariant in each backend.

Two barriers are available:

| Operation | Synchronized memory |
|---|---|
| `workgroupBarrier()` | workgroup memory within the workgroup |
| `bufferBarrier()` | storage-buffer memory within the workgroup |

Both are execution barriers and take no arguments. Every invocation in the
workgroup must reach a barrier in uniform control flow. Tach proves this
conservatively in Core IR. A barrier controlled by a logical coordinate,
mutable buffer/workgroup load, or atomic result is rejected. Constants,
uniform-address uniform loads, and helper results derived only from uniform
arguments can remain uniform.

## 15. Compact grammar

This EBNF summarizes the complete grammatical structure. The precedence table
above and the type/semantic restrictions in the rest of this document remain
normative:

```text
module        := { declaration }
declaration   := type-decl | function-decl | compute-decl

type-decl     := "type" IDENT "=" "{" fields "}" [";"]
fields        := [field {field-separator field} [field-separator]]
field         := IDENT ":" type
field-separator := "," | ";"

function-decl := "fn" IDENT parameters [":" type] block
compute-decl  := {attribute} "export" "compute" IDENT indices parameters block
attribute     := "@" "workgroup" "(" NUMBER {"," NUMBER} ")"
indices       := "[" IDENT {"," IDENT} "]"
parameters    := "(" [parameter {"," parameter} [","]] ")"
parameter     := IDENT ":" type

type          := IDENT ["<" type {"," type} ">"] ["[" [NUMBER] "]"]

block         := "{" {statement} "}"
statement     := variable-decl ";"
               | workgroup-decl ";"
               | simple-statement ";"
               | if-statement
               | while-statement
               | for-statement
               | return-statement ";"

variable-decl := ("const" | "let") IDENT [":" type] "=" expression
workgroup-decl := "workgroup" IDENT ":" type
simple-statement := expression
                  | expression assignment-op expression
                  | expression ("++" | "--")
assignment-op := "=" | "+=" | "-=" | "*=" | "/=" | "%="
               | "&=" | "|=" | "^=" | "<<=" | ">>="

if-statement  := "if" "(" expression ")" block
                 ["else" (if-statement | block)]
while-statement := "while" "(" expression ")" block
for-statement := "for" "(" for-init ";" expression ";" for-update ")" block
for-init      := "let" IDENT [":" type] "=" expression
for-update    := expression assignment-op expression
               | expression ("++" | "--")
return-statement := "return" [expression]

expression    := primary {postfix} {binary-op expression}
                 ["?" expression ":" expression]
primary       := NUMBER | "true" | "false" | IDENT
               | ("!" | "-" | "~") expression
               | "(" expression ")"
               | struct-literal
postfix       := "(" [expression {"," expression}] ")"
               | "." IDENT
               | "[" expression "]"
struct-literal := "{" [literal-field {"," literal-field} [","]] "}"
literal-field := IDENT ":" expression
```

The parser requires parentheses around `if`, `while`, and `for` headers.

## 16. Deliberate boundaries

The current language has no pointers, pointer arithmetic, target binding
annotations, ambient invocation identifiers, recursion, resource aliasing,
source `break`/`continue`, or provider-specific extensions. These omissions
keep one semantic program valid for both WGSL/WebGPU and SPIR-V/Vulkan.
