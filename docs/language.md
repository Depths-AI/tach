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
are errors. The filesystem defines the module set; there is no source list,
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

All top-level constant, type, and function names share one project-global
namespace and must be unique even across unrelated modules. Declaration order
within a file is irrelevant: module constants, types, and functions may refer
to later visible declarations of the appropriate kind. Recursive value types,
constant graphs, and call graphs are rejected.

Both dependency views must be acyclic. The kernel graph has one node per file.
The stricter module graph collapses files by directory, preventing two modules
from depending on each other in opposite directions even when the individual
file edges do not form a cycle. This gives the compiler deterministic,
parallelizable dependency branches without changing source visibility.

Module constants are Tach-only compile-time values:

```tach
const tileWidth: uint32 = 16;
const tileArea = tileWidth * tileWidth;
const up: vec<float32, 3> = normalize(vec(0.0, 2.0, 0.0));
```

They use the same direct-import visibility as types and functions. Their order
within a file does not matter, but their dependency graph must be acyclic.
They are neither exported nor emitted into the generated JavaScript,
TypeScript declarations, runtime metadata, WGSL, or SPIR-V as named objects.
The compiler substitutes their evaluated scalar or vector values wherever
they are used. Section 9 defines the complete constant algebra.

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
  if (i < input.length && i < output.length) {
    output[i] = input[i];
  }
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
  if (i < input.length && i < scratch.length) {
    scratch[i] = input[i] * factor;
  }
}

function addBias[i](
  scratch: buffer<float32[]>,
  output: buffer<float32[]>,
  bias: float32,
) {
  if (i < scratch.length && i < output.length) {
    output[i] = scratch[i] + bias;
  }
}

export function transform(
  input: buffer<float32[]>,
  output: buffer<float32[]>,
  count: uint32,
  factor: float32,
  bias: float32,
) {
  let scratch = transient<float32>(count);
  run multiply(input, scratch, factor) over count;
  run addBias(scratch, output, bias) over count;
}
```

Program declarations do not execute per invocation and have no branches,
loops, assignment statements, or `@workgroup`. Their bodies contain
compile-time `const` declarations, runtime shape or transient `let`
declarations, `run` statements, and the required final return of a declared
view. They describe a checked dispatch graph; indexed stages contain the actual
per-invocation code.

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
function paint[i](pixels: buffer<vec<float32, 4>[]>, width: uint32, height: uint32) {
  if (i < pixels.length) {
    let x = i % width;
    let y = i / width;
    pixels[i] = vec(
      float32(x) / float32(width),
      float32(y) / float32(height),
      0.25,
      1,
    );
  }
}

export function gradient(width: uint32, height: uint32): view<srgb8> {
  let pixels = transient<vec<float32, 4>>(width * height);
  run paint(pixels, width, height) over pixels.length;
  return view(pixels, width, height);
}
```

The first `view(...)` argument directly names a runtime
`buffer<vec<float32, 4>[]>` or transient containing linear RGBA. Width and height
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
Constants do not accept `@docs`: they are compiler inputs, not generated API
members. Use a precise name and an adjacent `//` comment when the value needs
implementation context.

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
are local to the stage.

Hardware groups runs into **workgroups**. You can ignore that word until
you need a small scratchpad that only one team shares. Tach picks a
default size; extra runs past the array still happen, which is why
kernels keep a bounds check.

```tach
export function volume[x, y, z](out: buffer<uint32[]>) {
  let index = x + y * 64 + z * 64 * 64;
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
  let index = y * width + x;
  if (index < out.length) {
    out[index] = x + y;
  }
}
```

The attribute accepts one through `rank` positive compile-time `uint32`
expressions; omitted axes are `1`. Portable limits are `x <= 256`, `y <= 256`,
`z <= 64`, and at most 256 invocations per workgroup. A stage using shared
memory or a barrier must state an explicit workgroup.

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
  let index = y * width + x;
  if (index < out.length) {
    out[index] = x + y;
  }
}

export function fillImage(out: buffer<uint32[]>, width: uint32, height: uint32) {
  run fill(out, width) over [width, height];
}
```

A checked shape is a `uint32` expression composed from:

- a `uint32` literal;
- a public `uint32` parameter or nested struct field;
- `.length` on a public runtime array or runtime-array field;
- a preceding runtime shape `let`;
- `+`, `-`, `*`, `/`, and `%`; or
- `min(a, b)`, `max(a, b)`, and `ceilDiv(a, b)`.

Shape arithmetic is evaluated by the host runtime with checked `uint32`
results. Underflow, overflow, division by zero, and a zero dispatch dimension
are runtime errors.

Program `let` declarations name runtime shapes or allocate transient storage.
A `const` in the same body is still a Tach-only scalar or vector known during
compilation; it cannot depend on a program parameter, buffer length, or earlier
runtime `let`. The same runtime shapes also define view width and height:

```tach
function write[i](scratch: buffer<float32[]>) {
  if (i < scratch.length) {
    scratch[i] = float32(i);
  }
}

function read[i](scratch: buffer<float32[]>, output: buffer<float32[]>) {
  if (i < scratch.length && i < output.length) {
    output[i] = scratch[i];
  }
}

export function roundTrip(output: buffer<float32[]>, count: uint32) {
  let blocks = ceilDiv(count, 256);
  let rounded = blocks * 256;
  let scratch = transient<float32>(rounded);
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
value argument may be a matching public value or nested field, a compile-time
scalar/vector expression, or a checked shape when the formal is `uint32`.
Compile-time arguments specialize the physical stage and never become runtime
parameter-block fields.

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
| `float16` | IEEE 754 binary16 |
| `float32` | IEEE 754 single precision |
| `void` | absence of a helper result |

There is no `number`: host and shader code must agree on width and
interpretation.

`view<srgb8>` is a program result contract, not a value available to helpers,
stages, buffers, structs, locals, or expressions other than the final
`return view(...)`. `srgb8` is the sole format and is likewise not an ordinary
source type.

`vec<T, N>` is the sole vector type syntax. `T` is one numeric scalar and `N`
is exactly `2`, `3`, or `4`:

```text
vec<float16, 2>  vec<float16, 3>  vec<float16, 4>
vec<float32, 2>  vec<float32, 3>  vec<float32, 4>
vec<int32, 2>    vec<int32, 3>    vec<int32, 4>
vec<uint32, 2>   vec<uint32, 3>   vec<uint32, 4>
```

`vec(...)` is the sole vector value constructor. It flattens numeric scalar
and vector arguments and must receive exactly two, three, or four total lanes:

```tach
function vectorValue(): vec<float32, 4> {
  let joined: vec<float32, 4> = vec(vec(1, 2), 3, 4);
  return joined + 0.5;
}
```

Its element type comes from the surrounding expression or a concrete
argument:

```tach
function inferredVectors(direction: vec<float32, 3>): vec<float32, 4> {
  let moved = direction + vec(1, 2, 3); // vec<float32, 3> from addition
  return vec(moved, 1); // vec<float32, 4> from the result
}
```

When neither source exists, whole-number lanes default to `uint32` and a
fraction or exponent defaults the vector to `float32`. `vec` never converts a
typed value and has no one-argument splat form. Repeat a scalar explicitly or
rely on documented scalar/vector operator and intrinsic broadcast. To change
a vector's element type, convert its scalar lanes explicitly and rebuild it
with `vec(...)`.

Swizzles use `x`, `y`, `z`, and `w`. One lane yields a scalar; several lanes
yield a vector. `value[index]` dynamically selects one lane.

### Structs

```tach
type Color = {
  rgb: vec<float32, 3>,
  alpha: float32,
};

function opaque(rgb: vec<float32, 3>): Color {
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
Backend-required padding never contributes another logical element, including
for a scalar `float16[]` after a fixed struct prefix.

`T[N]` is a fixed array whose length is a positive compile-time `uint32`
expression. Fixed arrays currently belong to shared memory, not host values or
buffers.

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
  let decimal = 42;
  let separated = 1_000_000;
  let hexadecimal = 0xff00_ff00;
  let binary = 0b1010_0001;
  let fraction = 1.25;
  let exponent = 6.022e2;
  return float32(decimal + separated + hexadecimal + binary) + fraction + exponent;
}
```

Suffixes such as `0u`, `1i`, and `1.0f` are rejected. Inference is local to one
expression and deterministic. Explicit types and conversions win, followed by
the expected assignment, argument, return, field, or result type; concrete
sibling operands; intrinsic requirements; and finally defaults. A
non-negative whole number defaults to `uint32`, a fraction or exponent to
`float32`, and unary `-` gives a whole literal `int32` context. An all-literal
floating intrinsic defaults to `float32`; `abs(1)` defaults to `int32`.
Operands are resolved collectively, so source order cannot change the result.

`int32(value)`, `uint32(value)`, `float16(value)`, and `float32(value)` perform
explicit numeric conversion. Integer-to-integer conversion preserves the low
32-bit pattern. A literal used in a `float16` context must be finite and within
binary16's `-65504` to `65504` range. Fractions still infer `float32` when no
context exists, so use an annotation or `float16(...)` when binary16 is
intended. General implicit conversions do not exist.

## 9. Variables, expressions, and assignment

Tach has one runtime local declaration, `let`, and one compile-time declaration,
`const`. They are different execution categories, not mutable and immutable
spellings for the same runtime value.

`let` evaluates where the surrounding function runs and may be reassigned. It
may carry a type annotation:

```tach
function sumFour(values: vec<float32, 4>): float32 {
  let total: float32 = 0;
  for (let lane = 0; lane < 4; lane++) {
    total += values[lane];
  }
  return total;
}
```

Writing `let fixed = 4;` still declares a runtime local, even though an
optimizer may later fold it. Tach has no second runtime-immutability keyword.
Function parameters and coordinates cannot be assigned. Names cannot shadow
another active local name. Branch-local names do not escape their branch. A
`for` initializer is always a `let` scoped to its loop. Rebinding a `let` does
not promise memory; the compiler represents locals as values carried through
structured control.

`const` evaluates completely in the compiler and may produce only `bool`, a
numeric scalar, or a numeric vector:

```tach
const tileWidth: uint32 = 16; // module scope; visible through direct imports

@workgroup(tileWidth)
export function tiled[i](out: buffer<uint32[]>) {
  const tileArea = tileWidth * tileWidth; // lexical; earlier constants only
  const tint = normalize(vec(3.0, 4.0, 0.0));
  let scratch: shared<uint32[tileArea]>;
  let lane = i % tileArea;
  scratch[lane] = i;
  workgroupBarrier();
  if (i < out.length) {
    out[i] = uint32(tint.x * float32(scratch[lane]));
  }
}
```

A module constant may refer to visible module constants in any declaration
order. Imported constants require the file's direct import, exactly like an
imported type or helper. A lexical constant may refer to module constants and
earlier constants in its active lexical scope; it cannot refer forward. Cycles
are errors and report the complete dependency chain.

The constant algebra deliberately reuses Tach's ordinary expression typing:

- literals and constant identifiers;
- `!`, unary `-`, and `~`;
- arithmetic, comparisons, short-circuit logic, bitwise operations, and shifts;
- the lazy `condition ? then : else` expression;
- numeric scalar conversions;
- `vec(...)`, vector indexing, and swizzles; and
- pure value intrinsics: numeric math, `fma`, and vector geometry.

Struct literals, runtime arrays, buffers, parameters, coordinates, `let`
bindings, transient allocation, barriers, atomics, and user-function calls are
not constant expressions. This is not a macro or general compile-time
programming language: it has no loops, declarations inside expressions,
conditional compilation, or host-configurable specialization.

Evaluation uses the declared or inferred Tach type at every operation. Integer
addition, subtraction, multiplication, negation, and bitwise operations use
32-bit wrapping semantics; shifts use the low five count bits. Integer division
or remainder by zero and signed `-2147483648 / -1` are errors. Float16 and
Float32 operations round back to their type; a result that is NaN, infinite, or
outside that type's finite range is an error. Conversions obey section 8,
including low-32-bit preservation between integer types. A conditional or
short-circuit expression evaluates only its selected branch.

The evaluated value is substituted at each use. A constant used for
`@workgroup`, a shared fixed-array length, a loop bound, ordinary math, or a
`run` value argument therefore has one meaning. A constant passed through a
program specializes that physical stage before backend lowering; it is absent
from the generated TypeScript signature and runtime parameter block. An unused
module constant or local constant is reported by the ordinary warning pass.

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
assignment, `++`, or `--`. `break;` exits the nearest enclosing loop.
`continue;` starts its next iteration; in a `for`, the update runs before the
condition is tested again. Either statement may appear inside nested `if` or
scope blocks, but only within a `while` or `for`. Statements after an
unconditional `break`, `continue`, or `return` are rejected as unreachable.

```tach
function boundedSum(limit: uint32): float32 {
  let total = 0.0;
  for (let i = 0; i < limit; i++) {
    if (i == 0) {
      continue;
    }
    total = fma(float32(i), 0.5, total);
    if (total > 100.0) {
      break;
    }
  }
  return total;
}
```

A helper returns its declared type. A void helper or indexed stage may use
`return;`. Statements after an unconditional return are rejected.

## 11. Math intrinsics

Intrinsic names are reserved free functions.

These preserve their `float16` or `float32` scalar/vector type:

```text
floor  ceil  trunc
sin    cos   tan
exp    exp2  log  log2
sqrt   rsqrt
```

`abs` accepts `int32`, `float16`, or `float32` scalar/vector values. `pow`
accepts matching floating values and may broadcast a scalar exponent across a
vector base.

`fma(a, b, c)` accepts `float16` or `float32` values and computes `a * b + c`,
component by component for a vector. Vector operands must have the same width;
scalars broadcast to that width. The arguments are inferred together, so
`fma(value, 2, vec(1, 1, 1))` naturally follows a concrete three-lane `value`.
It deliberately expresses a multiply-add operation to WGSL and SPIR-V. A
backend or device may execute it as a fused instruction or as separate
multiply and add operations; Tach promises the portable operation, not one
physical instruction or one universal intermediate-rounding rule.

`min`, `max`, and `clamp` accept every numeric scalar/vector type and broadcast
scalar arguments to a shared vector width. Their exact portable definitions
are:

```text
min(a, b)         = b if b < a, otherwise a
max(a, b)         = b if a < b, otherwise a
clamp(x, low, hi) = min(max(x, low), hi)
```

These definitions settle the awkward floating cases without backend guesses:
an unordered comparison is false, equal values preserve the first operand
(including its signed zero), and inverted clamp limits produce `hi`.

| Function | Input | Result |
|---|---|---|
| `dot(a, b)` | matching `vec<float16, N>` or `vec<float32, N>` | component type |
| `length(value)` | floating vector | component type |
| `distance(a, b)` | matching floating vectors | component type |
| `cross(a, b)` | matching three-lane floating vectors | same vector type |
| `normalize(value)` | floating vector | same vector type |

`float16` is an optional hardware feature, not an emulated storage format.
`tach build` emits both targets and records the exact requirement. Opening a
WebGPU session requests `shader-f16` when the adapter exposes it; preparing a
Float16 command fails clearly if it is unavailable. The Vulkan host likewise
enables and checks `shaderFloat16` plus the 16-bit storage/uniform features
actually required by the compiled project. Projects that use no `float16` do
not acquire these requirements.

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
  let lane = i % 64;

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
| `atomicCompareExchange` | replace only when the old value equals `expected` | previous value |

`atomicCompareExchange(place, expected, replacement)` is strong: returning
`expected` means the replacement occurred, while any other result proves the
place held that returned value. WGSL's weak primitive is retried internally,
so source code never handles spurious failure or a backend-specific result
structure.

Atomics on a public buffer are visible to every run on the device. Atomics
on shared memory are visible only inside this team. Updates are relaxed:
they do not imply a wait. Use a barrier when the next line must see
everyone else's write.

A normal `total += 1` from many runs can lose updates. `atomicAdd` cannot.

`workgroupBarrier()` waits until every run in this team has reached that
line, then continues with shared memory visible.
`bufferBarrier()` does the same for buffer memory inside that team.
Everyone must arrive. Tach rejects a barrier inside a branch that
different runs might take differently - for example anything derived from
the coordinate `i`, from a buffer load, or from an atomic result. *Uniform*
here means "the same decision for every run in the team," not a type.

## 13. Lexical and scope rules

Identifiers begin with a Unicode letter or `_`, then contain Unicode letters,
digits, or `_`. Type names, exported function names, and parameters of exported
functions must also be portable ASCII JavaScript/TypeScript identifiers and
avoid TypeScript keywords. Type names additionally cannot be `Float16Array`, `Float32Array`,
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
declaration     := const-decl | {attribute} (type-decl | function-decl)

const-decl      := "const" IDENT [":" type] "=" expression ";"

type-decl       := "type" IDENT "=" "{" fields "}" [";"]
fields          := field {field-separator field} [field-separator]
field           := IDENT ":" type
field-separator := "," | ";"

function-decl   := ["export"] "function" IDENT [indices]
                   parameters [":" type] block
indices         := "[" IDENT {"," IDENT} "]"
parameters      := "(" [parameter {"," parameter} [","]] ")"
parameter       := IDENT ":" type

attribute       := "@" "workgroup" "(" expression {"," expression} ")"
                 | docs-attribute
docs-attribute  := "@" "docs" "(" docs-clause {"," docs-clause} [","] ")"
docs-clause     := IDENT "(" [IDENT ","] STRING ")"

type            := (IDENT | "vec" "<" type "," NUMBER ">")
                   ["[" [expression] "]"]

block           := "{" {statement} "}"
statement       := const-decl
                 | let-decl ";"
                 | shared-decl ";"
                 | run-statement ";"
                 | simple-statement ";"
                 | if-statement | while-statement | for-statement
                 | "break" ";" | "continue" ";"
                 | return-statement ";"

let-decl        := "let" IDENT [":" type] "=" expression
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

Inference never crosses an expression boundary or inspects later uses, other
functions, generated host code, or backend behavior. Parameters, results,
struct fields, buffer elements, and public interfaces therefore stay explicit
and readable. Every value has one concrete type before typed IR is created.

Tach currently has no pointers, pointer arithmetic, binding annotations,
ambient invocation objects, recursion, resource aliasing, block comments,
cross-project imports, named imports, re-exports, deeper source
trees, or provider extensions.

Public programs express multiple dispatches and temporary resources, but not
arbitrary host control flow. Views express linear floating-point color and a
checked extent, not a provider presentation object. These boundaries keep one
source meaning valid for WebGPU/WGSL and Vulkan/SPIR-V and leave target
adaptation inside the compiler.
