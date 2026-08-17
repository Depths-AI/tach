# The Tach language

Tach is a small GPU language with TypeScript-shaped declarations, objects,
control flow, and expressions. It is not TypeScript executed on a GPU: types,
memory, dispatch, and synchronization have deliberately narrower portable
rules.

This guide starts with the one-function path inside a project, then defines
imports, explicit multi-stage programs, display views, the value language,
parallel memory, formatting, and the complete grammar. The compiler accepts
only what this document assigns a meaning.

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

- `type` declares an object-shaped value type, the way a TypeScript
  `type` does;
- `export function` makes a function TypeScript can call;
- `[i]` names the index of this run;
- `buffer<float32[]>` is an array that lives on the GPU; and
- `Params` is an immutable value supplied from TypeScript for this call.

An exported indexed function is the common case. It is both "what happens
at index `i`" and "launch that work once over a size the host provides."
You do not need a separate orchestration function until one operation has
several ordered steps.

The important mental model is "write one element, launch many elements."
Each run receives its own `i`, but every run executes the same body. Runs
are not executed in source order. Correct code gives each run its own
output slot, or deliberately coordinates shared writes with atomics or a
later second stage.

A buffer stays on the GPU. It is not a TypeScript array copied back after
each call. The host creates the handle once, builds one or more recipes,
and submits them. `read()` is an explicit wait-and-copy. Leaving
intermediate state in buffers, or in program-local scratch, is what lets
the GPU do sustained work without shipping every frame through JavaScript.

## 2. Projects, modules, imports, and function roles

Tach compiles projects, never isolated files. The nearest ancestor
`tach.json` is the project root. Every source lives exactly one tier below it:

```text
simulation/
  tach.json
  data/
    particles.tach
  physics/
    integrate.tach
    simulate.tach
```

An immediate directory containing `.tach` files is a module; each immediate
`.tach` file is a kernel file. Root-level or more deeply nested Tach sources
are errors. The filesystem defines the module set-there is no source list,
glob, or output path in the manifest.

The manifest separates Tach identity from the name of the generated JavaScript
package:

```json
{
  "name": "simulation",
  "version": "0.1.0",
  "javascript": {
    "package": "@studio/simulation"
  },
  "docs": {
    "title": "Simulation",
    "summary": "GPU simulation and rendering kernels."
  }
}
```

Every field is required and non-empty. `version` is Semantic Versioning and
is shared by both hosts; `javascript.package` is only the generated npm package
identity and never appears in Tach imports. `docs.title` and `docs.summary`
form the generated package README. Unknown fields, duplicate JSON keys,
multiple JSON values, invalid versions, and invalid npm package names are
errors. The manifest cannot enumerate modules or configure source depth,
output paths, formatting, optimization, or target-specific language rules.

Discovery is deterministic. Root-level and more deeply nested `.tach` files
are rejected; the generated `build` directory, non-source files, and directory
symlinks are not modules. Module and kernel identities that differ only by
case, or two paths naming the same physical kernel file, are rejected. Kernel
identities must also be valid canonical import spellings, so filesystem names
cannot create unreachable kernels. They are sorted bytewise with `/`
separators before parsing, checking, documentation, and emission.

A kernel file may begin with file documentation, followed by one contiguous
import block:

```tach
@docs(title("Integration"), summary("Particle integration stages."));

import "data/particles";
```

The only import form is a string containing the extensionless
`<module>/<kernel>` identity. It imports the complete file. There are no named,
default, wildcard, aliased, package, conditional, dynamic, or re-export forms.
The spelling contains exactly one `/`. Empty segments, `.`, `..`, `.tach`,
backslashes, drive/URL prefixes, npm-style `@scope/package` names, missing
targets, duplicate imports, and self imports are rejected. Imports are
contiguous, precede declarations, and have no initialization or side effects.

Declarations in the current file and its direct imports are visible. Imports
are deliberately non-transitive: if `physics/simulate` directly names a type
owned by `data/particles`, it must import `data/particles` even when another
import already depends on that file. `export` never changes Tach visibility;
it controls only generated JS/TypeScript exposure.

All top-level type and function names share one project-global namespace and
must be unique even across unrelated modules. Declaration order within a file
is irrelevant: types and functions may refer to later visible declarations.
Recursive value types and recursive call graphs are rejected.

Both dependency views must be acyclic. The kernel graph has one node per file.
The stricter module graph collapses files by directory, preventing two modules
from depending on each other in opposite directions even when the individual
file edges do not form a cycle. This gives the compiler deterministic,
parallelizable dependency branches without changing source visibility.

There are three fundamental function roles, plus one spelling that covers
the common case:

```text
function helper(values...): Result { ... }     // ordinary value helper
function stage[i](buffers..., values...) { }   // private GPU work
export function program(...) { ... }           // several stages, or a view
export function program[i](...) { ... }        // the usual one-launch kernel
```

Start with the last form. Introduce a helper when math repeats. Introduce
a private stage plus an exported program when one TypeScript call must run
several launches in order.

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

An exported function without `[i]` is a recipe of ordered steps. TypeScript
calls it once. Its body is not a pixel loop: it names scratch, then
`run`s stages, and may finish with `return view(...)`. Every program has
at least one `run`. An ordinary program also takes at least one public
buffer; a view program may paint its whole image in scratch instead.

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
statements, mutable locals, loops, or `@workgroup`. Apart from the required
final return of a declared view, they contain no returns. They describe a
checked dispatch graph; indexed stages contain the actual per-invocation code.

Every struct type is generated into the TypeScript API. An unexported indexed
stage is Tach-internal; an exported indexed stage and an exported unindexed
program are host endpoints. An explicit orchestration program is not callable
from Tach and cannot be used as a `run` target.

### Display views

A view is a finished picture, not a `<canvas>` and not a DOM node.
`view<srgb8>` is the one display result type. It belongs only to an
exported, unindexed program and is constructed by the program's final
statement:

```tach
function paint[i](pixels: buffer<float32x4[]>, width: uint32, height: uint32) {
  if (i < pixels.length) {
    const x = i % width;
    const y = i / width;
    pixels[i] = float32x4(
      float32(x) / float32(width),
      float32(y) / float32(height),
      0.25,
      1,
    );
  }
}

export function gradient(width: uint32, height: uint32): view<srgb8> {
  const pixels = transient<float32x4>(width * height);
  run paint(pixels, width, height) over pixels.length;
  return view(pixels, width, height);
}
```

The first `view(...)` argument directly names a runtime
`buffer<float32x4[]>` or transient containing linear RGBA. Width and height
are checked program shapes. The source resource's exact final version must be
defined by the preceding dispatches. At preparation, both dimensions and their
product must be positive and the source must contain at least
`width * height` pixels; any extra source elements are not displayed. You
write linear floating-point RGBA, the way a renderer thinks about color.
Tach converts RGB with the IEC sRGB transfer, clamps alpha to `[0, 1]`,
and packs both to 8-bit channels. The browser stores those bytes as a
canvas texture so `present` can draw them. Native Vulkan stores the same
bytes in a packed buffer. Source does not pack bytes or name a canvas,
texture, or native window.

The generated function returns `ComputeView`, a subtype of `ComputeCommand`.
`submit(view)` computes the picture and leaves it on the GPU. In a
browser, `gpu.present(canvas, view)` draws it on a same-sized canvas and
waits until that frame is done, so a render loop cannot queue forever.
Deno/Vulkan can still `submit` the same view; it has no window, so
`present` rejects there.

View programs are ordinary recipes. They may take session-owned public
buffers, or only scalar/struct values and internal transients. The latter form
is naturally reusable by any compatible Tach session because it captures no
buffer owner. A view cannot use command `repeat`; construct and present the
chosen recipe once per frame instead.

## 3. Structured documentation and comments

`@docs(...)` is structured compiler input, not an opaque comment blob. A
kernel-file attribute comes first, may precede imports, and ends with `;`:

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
| kernel file | `title("...")` |
| type | `field(name, "...")` |
| any function | `param(name, "...")` |
| indexed function | `coordinate(name, "...")` |
| value-returning helper or view program | `returns("...")` |

Each optional clause may occur once per referenced member. Names are unquoted
identifiers and must resolve in that declaration; unknown and duplicate names
are compile errors. A void helper, indexed stage, or ordinary public program
cannot use `returns`; a `view<srgb8>` program can and should describe its view.

Documentation on a public program describes its host API. Documentation on a
private stage describes the internal indexed operation. An exported indexed
function supplies both roles through one declaration.

Generated `index.d.ts` output carries summaries, member descriptions,
coordinates, and inferred buffer access as JSDoc. `tach docs` validates the
project and atomically refreshes `build/README.md` plus one
`build/docs/<module>.md` file per module without recompiling either GPU
artifact.
The README's TypeScript usage is derived from the same target-neutral checked
ABI and is compiled under strict TypeScript settings in the repository suite.
Every successful `tach build` refreshes the same documentation.

`//` starts a single-line implementation comment. It may appear wherever
whitespace is accepted. Tach has no block-comment form.

## 4. Coordinates, workgroups, and launch size

A coordinate is the index of this run: `i` for a line, `[x, y]` for an
image, `[x, y, z]` for a volume. Each one is an immutable `uint32`. Names
are local to the stage:

Hardware groups runs into **workgroups**. You can ignore that word until
you need a small scratchpad that only one team shares. Tach picks a
default size; extra runs past the array still happen, which is why
kernels keep a bounds check.

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
run at the edge. Your kernel owns the bounds check.

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

A public program is a short recipe of ordered steps, not a loop over
pixels. `run stage(...) over count` means: launch that stage across
`count` indices, then go on. `transient<T>(length)` is scratch the
TypeScript side never sees: allocated for this program, thrown away after
it, and not zero-filled. A later step may read it only after an earlier
step has written it.

A `run` domain is one size for a 1D stage or a bracketed list for 2D/3D:

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

Program `const` declarations either name shapes or allocate transient storage.
The same shapes also define view width and height:

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

Every `run` contributes one ordered physical dispatch to the program plan.
For a view program, the final `return view(...)` records a terminal projection
after those dispatches. When the final dispatch completely writes one
transient pixel per output pixel, target planning can fold projection into
that dispatch. Otherwise it appends a target-owned projection kernel. This
changes physical work, not source meaning.

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

`view<srgb8>` is a program result contract, not a value available to helpers,
stages, buffers, structs, locals, or expressions other than the final
`return view(...)`. `srgb8` is the sole format and is likewise not an ordinary
source type.

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

Most kernels never need this section. If each run owns a different slot,
you are done. Reach for the tools below only when runs must meet: a
count, a histogram, a reduction inside one team of at most 256.

An uninitialized top-level stage declaration `let name: shared<T>;` is a
whiteboard for that team. It is zero before your code runs:

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

Atomics on a public buffer are visible to every run on the device. Atomics
on shared memory are visible only inside this team. Updates are relaxed:
they do not imply a wait. Use a barrier when the next line must see
everyone else's write.

A normal `total += 1` from many runs can lose updates. `atomicAdd` cannot.

`workgroupBarrier()` waits until every run in this team has reached that
line, then continues with shared memory visible.
`bufferBarrier()` does the same for buffer memory inside that team.
Everyone must arrive. Tach rejects a barrier inside a branch that
different runs might take differently — for example anything derived from
the coordinate `i`, from a buffer load, or from an atomic result. *Uniform*
here means "the same decision for every run in the team," not a type.

## 13. Lexical and scope rules

Identifiers begin with a Unicode letter or `_`, then contain Unicode letters,
digits, or `_`. Type names, exported function names, and parameters of exported
functions must also be portable ASCII JavaScript/TypeScript identifiers and
avoid TypeScript keywords. Type names additionally cannot be `Float32Array`,
`Int32Array`, `Uint32Array`, or `ReadonlyArray`, because those spellings retain
their host-collection meanings in generated signatures. Runtime API names such
as `ComputeBuffer` remain safe because declarations import them through
compiler-private `$...` aliases. Private helper and stage names retain the
broader Tach identifier rule; backend-private names are mangled deterministically.

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
module          := [docs-attribute ";"] {import-decl} {declaration}
import-decl     := "import" STRING ";"
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

## 15. Canonical formatting

`tach fmt` discovers the nearest project and formats every kernel as one
transaction. A lexical or syntactic error prevents all writes. The formatter
preserves string contents, `//` comments, import order, and declaration order;
it performs no semantic rewrites.

The fixed style is UTF-8 with LF line endings, one final newline, two-space
indentation, semicolon-terminated statements, spaces around binary operators,
and one blank line between top-level declarations. Imports remain contiguous,
one per line, with one blank line before declarations. Lists target 100 columns;
multiline lists use one item per line and a trailing comma. An indivisible
identifier, string, or comment may exceed the target.

## 16. Deliberate boundaries

Tach currently has no pointers, pointer arithmetic, binding annotations,
ambient invocation objects, recursion, resource aliasing, `break`, `continue`,
block comments, cross-project imports, named imports, re-exports, deeper source
trees, or provider extensions.

Public programs express multiple dispatches and temporary resources, but not
arbitrary host control flow. Views express linear floating-point color and a
checked extent, not a provider presentation object. These boundaries keep one
source meaning valid for WebGPU/WGSL and Vulkan/SPIR-V and leave target
adaptation inside the compiler.
