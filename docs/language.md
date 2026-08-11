# The Tach language

This guide teaches the complete Tach source language. The first half is for
writing normal kernels. The second half is a reference for advanced parallel
memory, exact operator rules, and grammar.

Tach deliberately resembles TypeScript, but it is not TypeScript executed on a
GPU. It is a small GPU language with TypeScript-shaped declarations, control
flow, objects, and expressions. Its compiler owns the hardware mapping.

## 1. One useful program

Start with a particle update:

```tach
type Params = {
  dt: float32,
  count: uint32,
};

type Particle = {
  position: float32x4,
  velocity: float32x4,
};

function advance(particle: Particle, dt: float32): Particle {
  return {
    position: particle.position + particle.velocity * dt,
    velocity: particle.velocity,
  };
}

export function integrate[i](
  particles: buffer<Particle[]>,
  params: Params,
) {
  if (i >= params.count || i >= particles.length) {
    return;
  }

  particles[i] = advance(particles[i], params.dt);
}
```

Read it as ordinary typed code:

- `type` declares object-shaped value types.
- `function` declares a private helper.
- `export function` declares a host-callable GPU kernel.
- `[i]` names the logical coordinate for one invocation.
- `buffer<Particle[]>` is an array in GPU storage.
- `Params` is an ordinary immutable input value.
- each invocation checks its coordinate and updates one particle.

The brackets after a kernel name are Tach's one deliberate extension to a
normal function declaration. They keep GPU coordinates visible without
pretending that the host passes them or exposing a target invocation object.

## 2. Modules and functions

A `.tach` file may contain three declarations:

```text
type Name = { ... };
function helper(...) { ... }
export function kernel[coordinates](...) { ... }
```

Declaration order does not matter. A function may call a helper written later,
and a type may refer to another type written later. Recursive value types and
recursive function calls are rejected.

### Helpers

Helpers operate only on values:

```tach
function square(value: float32): float32 {
  return value * value;
}

function observe(value: float32) {
  const ignored = value;
}
```

Parameters are typed. A returned value needs an explicit result type. Omitting
the result type means `void`. Every path through a non-void helper must return.

Helpers may call other helpers and math intrinsics. They cannot receive
buffers, touch shared memory, use barriers, or call a kernel.
That makes a helper a predictable value computation on every backend.

### Kernels

A kernel is always an exported function:

```tach
export function clear[i](values: buffer<uint32[]>) {
  if (i < values.length) {
    values[i] = 0;
  }
}
```

Every kernel:

- has one, two, or three coordinate names in `[...]`;
- receives at least one `buffer<T>` plus any number of plain value parameters;
- returns no value; and
- is called only by generated host code, never by another Tach function.

## 3. Coordinates and launch size

Coordinates are immutable `uint32` values. Choose names that fit the problem:

```tach
export function line[index](out: buffer<uint32[]>) {
  if (index < out.length) {
    out[index] = index;
  }
}

export function image[x, y](out: buffer<uint32[]>) {
  const width = 1920;
  const pixel = y * width + x;
  if (pixel < out.length) {
    out[pixel] = x ^ y;
  }
}

export function volume[x, y, z](out: buffer<uint32[]>) {
  const index = x + y * 64 + z * 64 * 64;
  if (index < out.length) {
    out[index] = x + y + z;
  }
}
```

The host supplies a logical size with the same rank:

```ts
line(out, { size: 1_000_000 })
image(out, { size: [1920, 1080] })
volume(out, { size: [64, 64, 64] })
```

Tach rounds the launch up to whole workgroups. Extra invocations can therefore
exist at the edge. Check the problem boundary before indexing, as the examples
do. Tach does not insert dynamic bounds checks.

The compiler chooses these defaults:

| Rank | Default workgroup |
|---:|---:|
| 1D | `256 x 1 x 1` |
| 2D | `16 x 16 x 1` |
| 3D | `8 x 8 x 4` |

Most code should accept the default. If an algorithm truly depends on a tile
shape, place `@workgroup(...)` before the exported function:

```tach
@workgroup(16, 16)
export function tiled[x, y](out: buffer<uint32[]>) {
  const width = 1024;
  const index = y * width + x;
  if (index < out.length) {
    out[index] = x + y;
  }
}
```

The attribute accepts one through `rank` positive integer literals. Missing
axes are `1`. The portable limits are `x <= 256`, `y <= 256`, `z <= 64`, and
at most 256 invocations in one workgroup.

Coordinates remain logical Tach values. A backend may internally recognize
expressions such as `x % 16` and use an efficient target input, but no WGSL or
SPIR-V name enters source or Core IR.

## 4. Data types

### Scalars

| Type | Meaning |
|---|---|
| `bool` | `true` or `false`, used for values and control flow |
| `int32` | signed 32-bit integer |
| `uint32` | unsigned 32-bit integer |
| `float32` | 32-bit floating point |
| `void` | no helper result; not a storable value |

There is intentionally no `number` type. A GPU buffer and both shader targets
must agree on exact width and interpretation. `bool` is a normal value and
kernel-parameter type. It has no direct buffer representation; the compiler
privately encodes bool fields when building a kernel parameter block.

### Vectors

Tach has two-, three-, and four-lane vectors:

```text
float32x2  float32x3  float32x4
int32x2  int32x3  int32x4
uint32x2  uint32x3  uint32x4
```

Construct and read them directly:

```tach
function rearrange(): float32x4 {
  const value = float32x4(1, 2, 3, 4);
  const first = value.x;
  const reversed = value.wzyx;
  return float32x4(first, reversed.xyz);
}
```

Swizzles use `x`, `y`, `z`, and `w`. A one-lane access produces a scalar; a
multi-lane access produces a vector. Vectors also support dynamic indexing:
`value[lane]`.

A constructor accepts scalars and vectors, flattens them in order, converts
lanes to the destination scalar type, and requires exactly the destination
lane count:

```tach
function vectors(): float32x4 {
  const splat = float32x4(1);
  const joined = float32x4(float32x2(1, 2), 3, 4);
  return splat + joined;
}
```

One scalar splats to all lanes. Scalar arithmetic with a vector also broadcasts
where the operator table below allows it.

### Structs

Named structs use TypeScript-shaped object types and contextual object values:

```tach
type Color = {
  rgb: float32x3,
  alpha: float32,
};

function opaque(rgb: float32x3): Color {
  return {
    alpha: 1,
    rgb: rgb,
  };
}
```

Literal field order does not matter. Every field must appear exactly once. A
literal gets its struct type from a variable annotation, assignment, argument,
or return type.

Structs are value types, so they cannot contain themselves recursively.

### Arrays

`T[]` is a runtime-sized array:

```text
buffer<float32[]>
buffer<Particle[]>
```

It may live only in a buffer, either directly or as the final field of a
struct. It exposes `.length` and cannot be loaded or constructed as one whole
value.

`T[N]` is a fixed array with a positive literal count. Fixed arrays currently
belong to shared memory:

```text
float32[256]
atomic<uint32>[64]
```

Array and vector indices may be `int32` or `uint32`.

### Atomics

`atomic<int32>` and `atomic<uint32>` describe synchronized integer objects. They
may live in a buffer or shared variable. They are accessed through atomic operations,
not ordinary whole-value loads, stores, arguments, or returns.

## 5. Buffers and value parameters

Kernel parameters use their types to state the host contract:

```tach
type Params = {
  scale: float32,
  count: uint32,
};

export function scale[i](
  values: buffer<float32[]>,
  params: Params,
) {
  if (i < params.count && i < values.length) {
    values[i] *= params.scale;
  }
}
```

### `buffer<T>`

A buffer is GPU storage. The body may read or write it. Tach infers read-only
versus mutable access from actual stores and atomic operations, then emits that
decision consistently in WGSL, SPIR-V, metadata, and WebGPU layout.

### Plain value parameters

A parameter without `buffer<...>` is an immutable value, exactly as it looks:

```tach
export function addBias[i](values: buffer<float32[]>, bias: float32) {
  if (i < values.length) {
    values[i] += bias;
  }
}
```

Plain parameters may be `bool`, numeric scalars, numeric vectors, or
fixed-footprint structs composed from those types. They cannot be reassigned,
contain a runtime array, or contain an atomic. The compiler packs every plain
parameter of one kernel into one internal parameter block. That storage choice
is ABI machinery, not Tach syntax.

### Buffer identity

Parameters are positional and keep their names in generated functions.
Physical group, set, and binding numbers are compiler-owned and never appear in
source.

Two different buffer parameters mean two non-aliasing memory objects. The
managed TypeScript runtime rejects passing one `ComputeBuffer` to both. For an
in-place algorithm, declare one buffer and read and write through it.

## 6. Literals, conversion, and inference

Write numbers the way a TypeScript developer expects:

```tach
function literals(): float32x4 {
  const decimal = 42;
  const separated = 1_000_000;
  const hexadecimal = 0xff00_ff00;
  const binary = 0b1010_0001;
  const fraction = 1.25;
  const exponent = 6.022e2;
  return float32x4(float32(decimal), float32(separated), float32(binary), fraction + exponent);
}
```

Shader suffixes such as `0u`, `1i`, and `1.0f` are rejected. Context chooses a
literal type when an annotation, assignment, argument, constructor, branch, or
operator already requires one. Without context:

- a non-negative whole number is `uint32`;
- a fraction or exponent is `float32`; and
- unary `-` gives a whole-number literal `int32` context.

Inference does not depend on operand order: `1 + 2.0` and `2.0 + 1` are both
`float32`.

Conversions are explicit function-shaped constructors:

```tach
function convert(unsigned: uint32, signed: int32): float32 {
  const a = int32(unsigned);
  const b = uint32(signed);
  return float32(a) + float32(b);
}
```

`int32`/`uint32` conversion preserves the 32-bit pattern. Integer/float conversion
uses Tach's defined target-neutral operation. There are no general implicit
numeric conversions.

## 7. Variables and expressions

`const` is immutable. `let` may be rebound:

```tach
function sumFour(values: float32x4): float32 {
  let total = 0.0;
  for (let lane = 0; lane < 4; lane++) {
    total += values[lane];
  }
  return total;
}
```

Both forms may include a type annotation. Names cannot shadow another active
name. Branch-local names do not escape their branch. A `for` initializer is
scoped to its loop.

Rebinding a `let` does not promise a private memory cell. Tach may represent it
as an immutable value carried through branches or loops. Buffers and shared
variables are the addressable memory in the language.

### Postfix expressions

Calls, member access, and indexing compose from left to right:

```text
shape(value)
particle.position.x
blocks[i].values[j]
```

Calls are direct. Tach has no function values or methods.

### Unary and binary operators

| Family | Operators | Operands |
|---|---|---|
| unary logic | `!` | `bool` |
| unary numeric | `-` | `int32`, `float32`, or matching vectors |
| unary bitwise | `~` | integer scalar/vector |
| arithmetic | `+ - * /` | matching numeric values; defined scalar/vector broadcast |
| remainder | `%` | matching numeric scalars |
| comparison | `== != < <= > >=` | matching numeric scalars; yields `bool` |
| bool | `&& ||` | `bool`, short-circuiting |
| bitwise | `& \| ^` | matching integer values; defined scalar/vector broadcast |
| shifts | `<< >>` | integer values with unsigned counts of matching shape |

Unsigned negation is rejected. Every 32-bit shift masks its count to the low
five bits, so large shifts have the same modulo-32 meaning on every backend.
Right shift is arithmetic for `int32` and logical for `uint32`.

From lowest to highest, precedence is:

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
call, member, index
```

Binary operators associate left. The conditional expression selects only one
branch.

### Assignment

Supported forms are:

```text
=  +=  -=  *=  /=  %=  &=  |=  ^=  <<=  >>=  ++  --
```

Assignments work on `let` bindings and addressable buffer/shared-memory places.
Single vector lanes are assignable through `.x` or `[index]`. A `const` cannot
be assigned.

## 8. Control flow

Tach keeps the familiar structured forms.

### Conditions

```tach
function clampUnit(value: float32): float32 {
  if (value < 0.0) {
    return 0.0;
  } else if (value > 1.0) {
    return 1.0;
  } else {
    return value;
  }
}
```

`if` and `while` conditions must be `bool`.

The conditional expression is also available:

```tach
function sign(value: float32): float32 {
  return value < 0.0 ? -1.0 : 1.0;
}
```

Both result branches must have the same concrete type.

### Loops

```tach
function accumulated(count: uint32): uint32 {
  let index = 0;
  let total = 0;
  while (index < count) {
    total += index;
    index++;
  }
  return total;
}
```

```tach
function counted(): uint32 {
  let total = 0;
  for (let index = 0; index < 4; index++) {
    total += index;
  }
  return total;
}
```

A `for` initializer must be `let`; its update must be an assignment, compound
assignment, `++`, or `--`. `for` and `while` share one compiler loop model.
`break` and `continue` are not currently part of Tach.

### Return

A helper returns its declared type. A void helper or kernel uses `return;`.
Statements after an unconditional return are rejected as unreachable.

## 9. Math intrinsics

Intrinsics are free functions with Tach-defined type rules. Their names are
reserved.

These preserve a `float32` scalar or vector type:

```text
floor  ceil  trunc
sin    cos   tan
exp    exp2  log  log2
sqrt   rsqrt
```

`abs` accepts signed integer or floating scalar/vector values.

`pow(base, exponent)` accepts matching floating values. A scalar exponent may
broadcast over a vector base.

`min`, `max`, and `clamp` currently accept integer scalars/vectors. Floating
forms wait for one explicit portable NaN and signed-zero policy.

Vector-specific operations are:

| Function | Input | Result |
|---|---|---|
| `dot(a, b)` | matching `float32xN` | `float32` |
| `length(value)` | `float32xN` | `float32` |
| `distance(a, b)` | matching `float32xN` | `float32` |
| `cross(a, b)` | matching `float32x3` | `float32x3` |
| `normalize(value)` | `float32xN` | same vector type |

## 10. Shared memory, atomics, and barriers

This section is advanced. You do not need it for independent per-element or
per-pixel kernels.

### Shared memory

An uninitialized `shared<T>` declaration creates memory shared by the
invocations in one workgroup:

```tach
@workgroup(64)
export function reduce[i](out: buffer<uint32[]>) {
  let partial: shared<uint32[64]>;
  const lane = i % 64;

  partial[lane] = i;
  workgroupBarrier();

  if (lane == 0 && i < out.length) {
    out[i] = partial[0];
  }
}
```

`shared<T>` is valid only on a top-level `let` in a kernel body. The allocation
begins at the type's zero value before source instructions run. Tach establishes
this on both backends.

### Atomic operations

Each operation receives an addressable atomic place:

| Operation | Effect | Result |
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

Buffer atomics use device scope; shared-memory atomics use workgroup scope. Atomic
operations are relaxed. Use dependencies and barriers for ordering.

### Barriers

| Operation | Synchronizes |
|---|---|
| `workgroupBarrier()` | workgroup memory within one workgroup |
| `bufferBarrier()` | storage-buffer memory within one workgroup |

Every invocation in a workgroup must reach a barrier together. Tach rejects a
barrier under control that can differ by coordinate, mutable memory load, or
atomic result. Constants, plain kernel values, and helpers derived only from
such values can remain uniform. Here “uniform” is the parallel-control
property “equal for every invocation,” not a Tach type or source storage class.

## 11. Lexical and scope rules

Identifiers begin with a Unicode letter or `_`, then contain Unicode letters,
digits, or `_`. Public type, kernel, and parameter names emitted to
JavaScript/TypeScript must also be portable ASCII identifiers and avoid
reserved host-language names.

Whitespace is insignificant. Line comments and nested block comments are
supported:

```text
// one line

/* outer
   /* nested */
*/
```

Statements end with `;`. Parameter and object-literal lists accept a trailing
comma. Type fields may use commas or semicolons; this guide uses commas.

The compiler distinguishes these domains:

- constructible values: scalars, vectors, and fixed-footprint structs;
- buffer-shareable values: numeric scalars/vectors, atomics, suitable structs,
  and runtime arrays/tails; and
- workgroup-storable values: numeric scalars/vectors, atomics, fixed arrays,
  and suitable fixed-footprint structs.

Runtime-sized types are addressable but not constructible. Fixed arrays are
currently shared-memory-only. These restrictions prevent a source value from
claiming a representation that one target cannot honor.

## 12. Compact grammar

This EBNF summarizes syntax. The type and semantic rules above still apply:

```text
module        := { declaration }
declaration   := type-decl | helper-decl | kernel-decl

type-decl     := "type" IDENT "=" "{" fields "}" [";"]
fields        := field {field-separator field} [field-separator]
field         := IDENT ":" type
field-separator := "," | ";"

helper-decl   := "function" IDENT parameters [":" type] block
kernel-decl   := {attribute} "export" "function" IDENT indices parameters block
attribute     := "@" "workgroup" "(" NUMBER {"," NUMBER} ")"
indices       := "[" IDENT {"," IDENT} "]"
parameters    := "(" [parameter {"," parameter} [","]] ")"
parameter     := IDENT ":" type

type          := IDENT ["<" type {"," type} ">"] ["[" [NUMBER] "]"]

block         := "{" {statement} "}"
statement     := variable-decl ";"
               | shared-decl ";"
               | simple-statement ";"
               | if-statement
               | while-statement
               | for-statement
               | return-statement ";"

variable-decl := ("const" | "let") IDENT [":" type] "=" expression
shared-decl   := "let" IDENT ":" "shared" "<" type ">"
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

Parentheses are required around `if`, `while`, and `for` headers.

## 13. Deliberate boundaries

Tach currently has no pointers, pointer arithmetic, binding annotations,
ambient invocation objects, recursion, resource aliasing, `break`, `continue`,
or provider-specific extensions.

Those are not missing WGSL conveniences. They keep one source program and one
Core IR meaning valid for both WebGPU/WGSL and Vulkan/SPIR-V. If a capability
can be implemented as an optimization, it belongs inside the compiler rather
than in source syntax.
