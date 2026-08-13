# The Tach language

Tach is a small GPU language with TypeScript-shaped declarations, objects,
control flow, and expressions. It is not TypeScript executed on a GPU: types,
memory, dispatch, and synchronization have deliberately narrower portable
rules.

This guide starts with the one-function path, then defines explicit
multi-stage programs, the value language, parallel memory, and the complete
grammar. The compiler accepts only what this document assigns a meaning.

## 1. The one-function path

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

Read it as ordinary typed code:

- `type` declares an object-shaped value type;
- `export function` makes a host-callable program;
- `[i]` names one logical GPU coordinate;
- `buffer<float32[]>` is runtime-sized GPU storage; and
- `Params` is an immutable host value.

An exported indexed function is baseline syntax sugar. It simultaneously
defines an indexed GPU stage and a public program that dispatches that stage
once over the host-provided launch size. No orchestration syntax is required
for the common one-kernel case.

## 2. Modules and function roles

A `.tach` file contains types and functions. Declaration order is irrelevant:
types and functions may refer to later declarations. Recursive value types and
recursive call graphs are rejected.

There are three function roles:

```text
function helper(values...): Result { ... }
function stage[i](buffers..., values...) { ... }
export function program(buffers..., values...) { ... }
```

The fourth spelling is the baseline shorthand:

```text
export function program[i](buffers..., values...) { ... }
```

### Helpers

A helper is private and has no coordinate list:

```tach
function square(value: float32): float32 {
  return value * value;
}

export function squares[i](input: buffer<float32[]>, output: buffer<float32[]>) {
  if (i < input.length && i < output.length) {
    output[i] = square(input[i]);
  }
}
```

Helper parameters and results are constructible values. Helpers cannot receive
buffers, access shared memory, use barriers, call indexed stages, or recurse.
Omitting a result annotation means `void`; every path through a non-void helper
must return its declared type.

### Indexed stages

An indexed function is GPU work for one logical coordinate:

```tach
function copy[i](input: buffer<float32[]>, output: buffer<float32[]>) {
  output[i] = input[i];
}

export function copyAll(input: buffer<float32[]>, output: buffer<float32[]>) {
  run copy(input, output) over input.length;
}
```

A stage has one to three coordinates, at least one buffer parameter, any
number of constructible value parameters, and no result. It may call helpers,
but it cannot be called as a value function. A private stage is reachable only
through `run`; an exported indexed stage receives an implicit one-dispatch
public program of the same name.

### Public programs

An exported function without coordinates is host-callable orchestration. Its
body contains only untyped `const` declarations and `run` statements. It must
have at least one external buffer parameter and at least one `run`.

```tach
function multiply[i](
  input: buffer<float32[]>,
  scratch: buffer<float32[]>,
  factor: float32,
) {
  scratch[i] = input[i] * factor;
}

function addBias[i](
  scratch: buffer<float32[]>,
  output: buffer<float32[]>,
  bias: float32,
) {
  output[i] = scratch[i] + bias;
}

export function transform(
  input: buffer<float32[]>,
  output: buffer<float32[]>,
  count: uint32,
  factor: float32,
  bias: float32,
) {
  const scratch = transient<float32>(count);
  run multiply(input, scratch, factor) over count;
  run addBias(scratch, output, bias) over count;
}
```

Program declarations do not execute per invocation and have no ordinary
statements, mutable locals, loops, returns, or `@workgroup`. They describe a
checked dispatch graph; indexed stages contain the actual per-invocation code.

## 3. Structured documentation and comments

`@docs(...)` is structured compiler input, not an opaque comment blob. A
module attribute comes first and ends with `;`:

```tach
@docs(
  title("Vector scaling"),
  summary("Scales a GPU-resident vector in place.")
);

@docs(
  summary("Multiplies every in-range value."),
  coordinate(i, "Zero-based value index."),
  param(values, "Values updated in place."),
  param(factor, "Multiplier applied to each value.")
)
export function scale[i](values: buffer<float32[]>, factor: float32) {
  // Rounded workgroups may extend beyond the runtime array.
  if (i < values.length) {
    values[i] *= factor;
  }
}
```

Every `@docs` requires exactly one non-empty `summary`.

| Context | Additional clauses |
|---|---|
| module | `title("...")` |
| type | `field(name, "...")` |
| any function | `param(name, "...")` |
| indexed function | `coordinate(name, "...")` |
| value-returning helper | `returns("...")` |

Each optional clause may occur once per referenced member. Names are unquoted
identifiers and must resolve in that declaration; unknown and duplicate names
are compile errors. A void helper, indexed stage, or public program cannot use
`returns`.

Documentation on a public program describes its host API. Documentation on a
private stage describes the internal indexed operation. An exported indexed
function supplies both roles through one declaration.

Generated `.d.ts` output carries summaries, member descriptions, coordinates,
and inferred buffer access as JSDoc. `tach docs FILE.tach` renders Markdown to
standard output from a target-neutral compiler description and adds a
TypeScript usage example from that same validated API.

`//` starts a single-line implementation comment. It may appear wherever
whitespace is accepted. Tach has no block-comment form.

## 4. Coordinates, workgroups, and launch size

Every coordinate is an immutable `uint32`. Names are local to the stage:

```tach
export function volume[x, y, z](out: buffer<uint32[]>) {
  const index = x + y * 64 + z * 64 * 64;
  if (index < out.length) {
    out[index] = x + y + z;
  }
}
```

For exported indexed shorthand, the host provides a logical size with exactly
the coordinate rank:

```ts
line(out, { size: 1_000_000 })
image(out, { size: [1920, 1080] })
volume(out, { size: [64, 64, 64] })
```

If a 1D shorthand omits `size`, the runtime infers it from the first
runtime-sized public buffer when one exists. Otherwise omission launches one
workgroup. A 2D or 3D size normally must be explicit.

For explicit programs, each `run` supplies its stage domain and the generated
host function accepts only `CommandOptions`, not a launch size.

Tach rounds each domain axis up to whole workgroups. Extra coordinates may
execute at the edge. Source owns bounds guards.

Default workgroups are:

| Rank | Workgroup |
|---:|---:|
| 1D | `256 x 1 x 1` |
| 2D | `16 x 16 x 1` |
| 3D | `8 x 8 x 4` |

Use `@workgroup` on an indexed stage only when its algorithm requires an exact
shape:

```tach
@workgroup(16, 16)
export function tiled[x, y](out: buffer<uint32[]>, width: uint32) {
  const index = y * width + x;
  if (index < out.length) {
    out[index] = x + y;
  }
}
```

The attribute accepts one through `rank` positive integer literals; omitted
axes are `1`. Portable limits are `x <= 256`, `y <= 256`, `z <= 64`, and at
most 256 invocations per workgroup. A stage using shared memory or a barrier
must state an explicit workgroup.

## 5. Program shapes and transient storage

A `run` domain is one shape for a 1D stage or a bracketed list for 2D/3D:

```tach
function fill[x, y](out: buffer<uint32[]>, width: uint32) {
  out[y * width + x] = x + y;
}

export function fillImage(out: buffer<uint32[]>, width: uint32, height: uint32) {
  run fill(out, width) over [width, height];
}
```

A checked shape is a `uint32` expression composed from:

- a `uint32` literal;
- a public `uint32` parameter or nested struct field;
- `.length` on a public runtime array or runtime-array field;
- a preceding shape `const`;
- `+`, `-`, `*`, `/`, and `%`; or
- `min(a, b)`, `max(a, b)`, and `ceilDiv(a, b)`.

Shape arithmetic is evaluated by the host runtime with checked `uint32`
results. Underflow, overflow, division by zero, and a zero dispatch dimension
are runtime errors.

Program `const` declarations either name shapes or allocate transient storage:

```tach
function write[i](scratch: buffer<float32[]>) {
  scratch[i] = float32(i);
}

function read[i](scratch: buffer<float32[]>, output: buffer<float32[]>) {
  output[i] = scratch[i];
}

export function roundTrip(output: buffer<float32[]>, count: uint32) {
  const blocks = ceilDiv(count, 256);
  const rounded = blocks * 256;
  const scratch = transient<float32>(rounded);
  run write(scratch) over rounded;
  run read(scratch, output) over count;
}
```

`transient<T>(length)` yields a program-local `buffer<T[]>`. `T` must have a
fixed, host-shareable, non-atomic footprint. A transient has no host argument
or readback handle. The compiler proves that every read is preceded by a
defining dispatch and assigns non-overlapping lifetimes to reusable scratch
allocations.

Stage buffer arguments in `run` must directly name a public buffer or
transient. The same resource cannot fill two buffer formals of one stage. A
value argument may be a matching public value or nested field, a supported
literal, or a checked shape when the formal is `uint32`.

Every `run` remains a distinct physical dispatch today. Tach does not perform
inter-stage fusion.

## 6. Data types

### Scalars and vectors

| Type | Meaning |
|---|---|
| `bool` | `true` or `false`; value-only outside parameter blocks |
| `int32` | signed 32-bit integer |
| `uint32` | unsigned 32-bit integer |
| `float32` | IEEE 754 single precision |
| `void` | absence of a helper result |

There is no `number`: host and shader code must agree on width and
interpretation.

Numeric vectors have two, three, or four lanes:

```text
float32x2  float32x3  float32x4
int32x2    int32x3    int32x4
uint32x2   uint32x3   uint32x4
```

Constructors flatten scalar and vector arguments and require exactly the
destination lane count. One scalar splats:

```tach
function vectorValue(): float32x4 {
  const joined = float32x4(float32x2(1, 2), 3, 4);
  return joined + float32x4(0.5);
}
```

Swizzles use `x`, `y`, `z`, and `w`. One lane yields a scalar; several lanes
yield a vector. `value[index]` dynamically selects one lane.

### Structs

```tach
type Color = {
  rgb: float32x3,
  alpha: float32,
};

function opaque(rgb: float32x3): Color {
  return { alpha: 1, rgb: rgb };
}
```

Every field appears exactly once in a struct literal; literal order is
irrelevant. Context from an annotation, assignment, argument, or result
determines the named type. Structs are value types and cannot contain a cycle.

### Arrays and atomics

`T[]` is a runtime array. It may appear directly in a buffer or as the final
field of one struct. It exposes `.length` through a place and cannot be loaded,
constructed, passed, or returned as a whole value.

`T[N]` is a positive-literal fixed array. Fixed arrays currently belong to
shared memory, not host values or buffers.

`atomic<int32>` and `atomic<uint32>` are synchronized integer objects. They
may occur in host buffers or shared memory and are accessed only by atomic
operations, never by ordinary whole-value loads or stores.

## 7. Buffers, values, and identity

An indexed stage may read or write `buffer<T>` parameters. Tach infers access
from stores and non-load atomic operations, then uses that result in both
backends, generated metadata, WebGPU layouts, and documentation.

Plain parameters are immutable constructible values: bool, numeric scalars,
numeric vectors, or fixed-footprint structs recursively composed from them.
They cannot contain runtime arrays or atomics. The ABI later packs their leaves
into one private parameter block per physical stage.

Different public buffer parameters represent distinct memory. The managed
runtime rejects the same `ComputeBuffer` in two positions of one command. A
single stage also cannot receive one public/transient resource through two
buffer formals. Express in-place work with one parameter.

Source never assigns group, set, or binding numbers.

## 8. Literals, conversion, and inference

Numbers use ordinary spelling:

```tach
function literalValue(): float32 {
  const decimal = 42;
  const separated = 1_000_000;
  const hexadecimal = 0xff00_ff00;
  const binary = 0b1010_0001;
  const fraction = 1.25;
  const exponent = 6.022e2;
  return float32(decimal + separated + hexadecimal + binary) + fraction + exponent;
}
```

Suffixes such as `0u`, `1i`, and `1.0f` are rejected. Context selects the
concrete type. Without context, a non-negative whole number is `uint32`, a
fraction or exponent is `float32`, and unary `-` gives a whole literal
`int32` context. Mixed literal pairs infer independent of operand order.

`int32(value)`, `uint32(value)`, and `float32(value)` perform explicit numeric
conversion. Integer-to-integer conversion preserves the low 32-bit pattern.
General implicit conversions do not exist.

## 9. Variables, expressions, and assignment

`const` is immutable; `let` may be rebound. Either may carry a type annotation:

```tach
function sumFour(values: float32x4): float32 {
  let total: float32 = 0;
  for (let lane = 0; lane < 4; lane++) {
    total += values[lane];
  }
  return total;
}
```

Names cannot shadow another active name. Branch-local names do not escape
their branch. A `for` initializer is scoped to its loop. Rebinding a `let`
does not promise memory; the compiler represents locals as immutable values
carried through structured control.

Calls, member access, and indexing compose left to right. Calls are direct;
Tach has no function values or methods.

| Family | Operators and operands |
|---|---|
| unary | `!` bool; `-` signed numeric; `~` integer scalar/vector |
| add/subtract | `+ -` matching numeric values; scalar/vector broadcast |
| multiply | `*` matching numeric values or scalar/vector broadcast |
| divide | `/` matching numeric values; scalar/vector broadcast |
| remainder | `%` matching numeric scalars |
| comparison | `== != < <= > >=` matching numeric scalars, yielding bool |
| logic | `&& ||` bool with short-circuiting |
| bitwise | `& \| ^` matching integers; scalar/vector broadcast |
| shifts | `<< >>` integer scalar/vector with unsigned scalar or lane-wise counts |

Unsigned negation is invalid. Every shift masks its count to the low five bits;
signed right shift is arithmetic and unsigned right shift is logical.

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

Binary operators associate left. A conditional expression evaluates only its
selected branch and requires equal result types.

Assignments are `=`, `+=`, `-=`, `*=`, `/=`, `%=`, `&=`, `|=`, `^=`,
`<<=`, `>>=`, `++`, and `--`. Targets are mutable locals or addressable
buffer/shared places. A vector lane selected by `.x` or `[index]` is
addressable when its base is.

## 10. Control flow

`if`, `while`, and `for` headers require parentheses. Conditions are `bool`.
Both branches of a conditional expression have one concrete type.

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

A `for` initializer is a `let`; its update is an assignment, compound
assignment, `++`, or `--`. `break` and `continue` are not source constructs.

A helper returns its declared type. A void helper or indexed stage may use
`return;`. Statements after an unconditional return are rejected.

## 11. Math intrinsics

Intrinsic names are reserved free functions.

These preserve a `float32` scalar or vector type:

```text
floor  ceil  trunc
sin    cos   tan
exp    exp2  log  log2
sqrt   rsqrt
```

`abs` accepts `int32` or `float32` scalar/vector values. `pow` accepts matching
floating values and may broadcast a scalar exponent across a vector base.

`min`, `max`, and `clamp` accept integer scalars/vectors. Their floating forms
remain unavailable until Tach defines one portable NaN and signed-zero policy.

| Function | Input | Result |
|---|---|---|
| `dot(a, b)` | matching `float32xN` | `float32` |
| `length(value)` | `float32xN` | `float32` |
| `distance(a, b)` | matching `float32xN` | `float32` |
| `cross(a, b)` | two `float32x3` | `float32x3` |
| `normalize(value)` | `float32xN` | same vector type |

`ceilDiv` is not a stage/helper intrinsic; it exists only in public-program
shape expressions.

## 12. Shared memory, atomics, and barriers

An uninitialized top-level stage declaration `let name: shared<T>;` allocates
zero-initialized workgroup memory:

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

The declaration must be a direct child of the stage body and requires explicit
`@workgroup`. Both targets guarantee zero initialization before source
instructions.

Atomic operations receive an addressable atomic place:

| Operation | Effect | Result |
|---|---|---|
| `atomicLoad(place)` | read | current value |
| `atomicStore(place, value)` | write | void |
| `atomicAdd`, `atomicSub` | arithmetic update | previous value |
| `atomicMin`, `atomicMax` | ordered update | previous value |
| `atomicAnd`, `atomicOr`, `atomicXor` | bitwise update | previous value |
| `atomicExchange` | replacement | previous value |

Buffer atomics use device scope; shared atomics use workgroup scope. Atomic
operations are relaxed.

`workgroupBarrier()` synchronizes workgroup memory within one workgroup.
`bufferBarrier()` synchronizes storage-buffer memory within one workgroup.
Every invocation must reach a barrier together. Tach rejects barriers under
control derived from coordinates, mutable memory loads, atomic results, or
other varying values. Here *uniform* means equal across a workgroup, not a
source type or storage class.

## 13. Lexical and scope rules

Identifiers begin with a Unicode letter or `_`, then contain Unicode letters,
digits, or `_`. Public type, program, and parameter names emitted to
JavaScript/TypeScript must also be portable ASCII identifiers and avoid
reserved host-language names.

Whitespace is insignificant. Statements end with `;`. Parameter, argument,
attribute, and object-literal lists accept a trailing comma. Type fields use
`,` or `;`.

String literals exist only as structured-documentation arguments; they are not
runtime values.

The value domains are:

- constructible: scalars, vectors, and fixed-footprint structs;
- host-shareable: numeric scalars/vectors, atomics, compatible structs, and
  runtime arrays/tails; and
- workgroup-storable: numeric scalars/vectors, atomics, fixed arrays, and
  compatible fixed-footprint structs.

Runtime types are addressable but not constructible. Fixed arrays are
currently shared-memory-only. These boundaries prevent a source value from
claiming a representation unavailable on one target.

## 14. Compact grammar

The grammar below summarizes syntax; semantic restrictions above still apply.

```text
module          := [docs-attribute ";"] {declaration}
declaration     := {attribute} (type-decl | function-decl)

type-decl       := "type" IDENT "=" "{" fields "}" [";"]
fields          := field {field-separator field} [field-separator]
field           := IDENT ":" type
field-separator := "," | ";"

function-decl   := ["export"] "function" IDENT [indices]
                   parameters [":" type] block
indices         := "[" IDENT {"," IDENT} "]"
parameters      := "(" [parameter {"," parameter} [","]] ")"
parameter       := IDENT ":" type

attribute       := "@" "workgroup" "(" NUMBER {"," NUMBER} ")"
                 | docs-attribute
docs-attribute  := "@" "docs" "(" docs-clause {"," docs-clause} [","] ")"
docs-clause     := IDENT "(" [IDENT ","] STRING ")"

type            := IDENT ["<" type {"," type} ">"] ["[" [NUMBER] "]"]

block           := "{" {statement} "}"
statement       := variable-decl ";"
                 | shared-decl ";"
                 | run-statement ";"
                 | simple-statement ";"
                 | if-statement | while-statement | for-statement
                 | return-statement ";"

variable-decl   := ("const" | "let") IDENT [":" type] "=" expression
shared-decl     := "let" IDENT ":" "shared" "<" type ">"
run-statement   := "run" IDENT arguments "over" domain
domain          := expression | "[" expression {"," expression} "]"
simple-statement := expression
                  | expression assignment-op expression
                  | expression ("++" | "--")
assignment-op   := "=" | "+=" | "-=" | "*=" | "/=" | "%="
                 | "&=" | "|=" | "^=" | "<<=" | ">>="

if-statement    := "if" "(" expression ")" block
                   ["else" (if-statement | block)]
while-statement := "while" "(" expression ")" block
for-statement   := "for" "(" for-init ";" expression ";" for-update ")" block
for-init        := "let" IDENT [":" type] "=" expression
for-update      := expression assignment-op expression
                 | expression ("++" | "--")
return-statement := "return" [expression]

expression      := primary {postfix} {binary-op expression}
                   ["?" expression ":" expression]
primary         := NUMBER | STRING | "true" | "false" | IDENT
                 | "transient" "<" type ">" "(" expression ")"
                 | ("!" | "-" | "~") expression
                 | "(" expression ")" | struct-literal
postfix         := arguments | "." IDENT | "[" expression "]"
arguments       := "(" [expression {"," expression} [","]] ")"
struct-literal  := "{" [literal-field {"," literal-field} [","]] "}"
literal-field   := IDENT ":" expression
```

## 15. Deliberate boundaries

Tach currently has no pointers, pointer arithmetic, binding annotations,
ambient invocation objects, recursion, resource aliasing, `break`, `continue`,
block comments, imports, or provider extensions.

Public programs express multiple dispatches and temporary resources, but not
arbitrary host control flow. Distinct `run` statements are not fused. These
boundaries keep one source meaning valid for WebGPU/WGSL and Vulkan/SPIR-V and
leave target adaptation inside the compiler.
