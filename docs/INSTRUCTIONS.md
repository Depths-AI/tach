# Tach application instructions for AI coding agents

This document is context for AI coding agents that use Tach as a programming
language through the published `@depths/tach` npm package. Its subject is writing
Tach applications: project layout, source semantics, compiler commands, generated
packages, TypeScript integration, the unified WebGPU/Vulkan runtime, correctness,
debugging, and performance.

Treat every rule below as part of one language-and-tooling contract. Do not infer
TypeScript, WGSL, GLSL, CUDA, or native Vulkan behavior where Tach defines a
narrower rule. Prefer the simplest Tach form that expresses the application:
an exported indexed function for one dispatch, private indexed stages plus one
exported program for ordered multi-dispatch work, and helpers for reusable pure
value computation. Use a `view<srgb8>` program when the result is a displayable
frame; do not route pixels through CPU readback merely to draw them.

## 1. Operational model

Tach is a small typed language for portable GPU compute. Tach source describes:

- typed values and storage;
- one-, two-, or three-dimensional indexed GPU work;
- pure helper computation;
- workgroup-local memory, atomics, and synchronization; and
- ordered programs made from indexed stages and transient GPU storage; and
- terminal display views from linear floating-point RGBA.

The compiler owns shader entry points, bindings, padding, byte layout, launch
geometry, target validation, generated TypeScript signatures, and the physical
execution plan. Application code must not assign descriptor numbers, declare
provider built-ins, hand-pack padding, edit generated shaders, or reach into
backend buffers owned by the Tach runtime.

An application has two distinct source layers and one host-neutral generated
boundary:

```text
Tach project                         TypeScript application
-----------                          ----------------------
<module>/<kernel>.tach   build ->   one generated recipe facade
tach.json                            @depths/tach runtime
                                     browser -> WebGPU/WGSL + canvas present
                                     Deno -> Vulkan/SPIR-V + offscreen views
```

Keep responsibilities on the correct side:

- Tach performs parallel numerical work and declares dispatch structure.
- TypeScript owns user interaction, files, networking, clocks, frame loops,
  error presentation, and decisions that require ordinary host control flow.
- Generated functions construct opaque commands; they do not execute work.
- A Tach session owns GPU resources and executes recipes in order.

The minimum runtime sequence is always:

```text
gpu.buffer(hostValue)       -> session-owned ComputeBuffer
generatedProgram(arguments) -> opaque ComputeCommand or ComputeView recipe
gpu.submit(command)         -> prepare and queue GPU work
gpu.present(canvas, view)   -> browser execution, display, and backpressure
buffer.read() / gpu.idle()  -> wait for completion when required
```

## 2. Five-minute complete example

Compilation and server execution use Deno. Browser execution requires WebGPU;
Deno execution requires Tach's supported Vulkan 1.3 native host.
Install the compiler and runtime in the consuming npm application:

```sh
npm install @depths/tach
```

Create a Tach project root with this `tach.json`:

```json
{
  "name": "scaling",
  "version": "0.1.0",
  "javascript": {
    "package": "@example/scaling"
  },
  "docs": {
    "title": "Scaling kernels",
    "summary": "Typed GPU scaling used by the application."
  }
}
```

Create `kernels/scale.tach` one directory below the project root:

```tach
@docs(
  title("Vector scaling"),
  summary("Scales a GPU-resident vector in place."),
);

@docs(
  summary("Multiplies every in-range value."),
  coordinate(i, "Zero-based value index."),
  param(values, "Values updated in place."),
  param(factor, "Multiplier applied to each value."),
)
export function scale[i](values: buffer<float32[]>, factor: float32) {
  // A rounded dispatch may contain lanes beyond the runtime array.
  if (i < values.length) {
    values[i] *= factor;
  }
}
```

Run commands anywhere inside the Tach project; the nearest ancestor
`tach.json` is discovered automatically:

```sh
npx tach fmt
npx tach check
npx tach build
```

The build creates `build/index.js`, `build/index.d.ts`, cohesive compressed
`build/kernel.wgsl.gz` and `build/kernel.spv`, a generated package manifest,
and generated Markdown.
Use the runtime and generated project as separate imports:

```ts
import { tach } from "@depths/tach";
import { scale } from "./build/index.js";

const result = await tach(async (gpu) => {
  const values = gpu.buffer(new Float32Array([1, 2, 3, 4]));
  await gpu.submit(scale(values, 2));
  return values.read();
});

console.log(result); // Float32Array [2, 4, 6, 8]
```

This example already demonstrates the full boundary: Tach owns GPU work;
TypeScript creates host data, opens a session, constructs a command, submits it,
and reads the result.

## 3. Tach projects are not npm projects

A Tach project is identified by `tach.json`. It is not defined by
`package.json`, npm workspaces, TypeScript configuration, or JavaScript source.
It may live at an npm application root or in a subdirectory, but those are
organizational choices outside the Tach language.

The compiler always operates on a complete Tach project. There is no supported
single-file compilation mode. Every project command walks upward from its
current working directory to the nearest `tach.json`, then discovers and
processes the whole project. Therefore:

- run the CLI from the intended project or one of its descendants;
- do not pass a `.tach` filename to a command;
- use separate project roots when an application needs independent packages;
- remember that a nested `tach.json` starts a different Tach project; and
- do not expect an outer manifest to include a nested project.

An application repository may use a layout such as:

```text
app/
  package.json
  src/
    main.ts
  gpu/
    tach.json
    data/
      particles.tach
    physics/
      integrate.tach
      simulate.tach
    build/
```

Run `npx tach build` from `app/gpu` or any descendant. TypeScript can then import
`../gpu/build/index.js`, or the generated `build/` package can be linked,
installed, packed, or published under its configured JavaScript package name.

## 4. The manifest contract

The project manifest has one fixed shape:

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

All fields are required and non-empty:

| Field | Meaning |
|---|---|
| `name` | Tach project identity |
| `version` | SemVer shared by every generated artifact |
| `javascript.package` | npm identity of the generated host-neutral package |
| `docs.title` | heading of the generated package README |
| `docs.summary` | introductory text of the generated package README |

`name` and `javascript.package` deliberately describe different identities. Tach
imports never contain either value; they identify files inside the current
project. The JavaScript package name is used only when consuming or publishing the
generated npm package.

The manifest rejects unknown fields, duplicate JSON keys, multiple JSON values,
invalid Semantic Versions, and invalid npm package names. It does not support:

- a list or glob of source files;
- module declarations;
- custom source depth;
- output-directory selection;
- formatter or optimization switches;
- target-specific language semantics; or
- dependencies on other Tach projects.

Do not invent any such fields. The filesystem is the source inventory and the
fixed `build/` directory is the output.

## 5. Filesystem and module layout

Every Tach source file must have the exact shape:

```text
<project root>/<module>/<kernel>.tach
```

There is exactly one directory tier between the project root and every source
file. An immediate directory containing one or more immediate `.tach` files is
a module. The compiler rejects root-level Tach files and Tach files nested more
deeply.

Valid:

```text
tach.json
math/
  vectors.tach
  transforms.tach
physics/
  particles.tach
  constraints.tach
```

Invalid:

```text
root-kernel.tach
physics/integration/euler.tach
```

Modules and kernels are discovered automatically; never list them in
`tach.json`. Non-Tach files do not become source. The compiler-owned `build/`
directory is not a module.

The logical identity of `physics/integrate.tach` is `physics/integrate`.
Hyphens are valid in filesystem identities because imports use strings, for
example `import "rigid-body/solve-contacts";`. A Tach declaration name still
follows identifier rules and cannot contain a hyphen.

Avoid identities that differ only by case. Tach rejects case-folding
collisions so a project behaves consistently across case-sensitive and
case-insensitive filesystems. Keep module and kernel filenames lowercase and
stable unless the application has a strong reason not to.

## 6. Imports

Tach has exactly one import form:

```tach
import "data/particles";
```

The string is the extensionless `<module>/<kernel>` identity of one file in the
same Tach project. The import brings the complete file into direct visibility.
It is not a JavaScript import and has no runtime initialization or side effect.

The following are invalid:

```tach
import "particles";                 // Missing module.
import "./data/particles";          // Relative spelling.
import "data/particles.tach";       // Extension must be implicit.
import "data/more/particles";       // Too deeply nested.
import "@scope/package";            // npm identity.
import "C:/project/particles";      // Filesystem path.
import "https://example.test/x";    // URL.
```

There are no named, default, wildcard, aliased, conditional, dynamic, or
re-export forms. Duplicate imports, missing targets, self-imports, empty path
segments, `.`, `..`, backslashes, drive prefixes, and URL-like spellings are
errors.

Imports must form one contiguous block after optional file documentation and
before all declarations:

```tach
@docs(
  title("Integration"),
  summary("Particle integration stages."),
);

import "data/particles";
import "math/vectors";

// Declarations begin here.
```

Import order has no semantic effect, and `tach fmt` preserves the existing
order rather than sorting it.

## 7. Visibility is direct and non-transitive

A Tach file can name declarations from:

1. the same file; and
2. every file it imports directly.

Imports are intentionally non-transitive. If `physics/simulate` names
`Particle`, and `Particle` is declared in `data/particles`, then
`physics/simulate` must import `data/particles` itself. It does not matter that
another imported file also imports `data/particles`.

Use this rule mechanically:

```text
If this file spells a top-level name owned elsewhere,
this file imports that name's owner directly.
```

An import exposes every constant, type, helper, and indexed stage in its target
file to Tach source. The `export` keyword does not control this visibility.
`export` only controls whether a function becomes a generated
JavaScript/TypeScript recipe constructor.

Declaration order within a file is irrelevant. A declaration may refer to a
later visible module constant, type, or helper. This does not permit cycles:
constant dependencies, recursive value types, and recursive call graphs are
rejected with their dependency chain.

## 8. One project-global declaration namespace

All top-level constant, type, and function names must be unique across the
entire Tach project, even when their files are unrelated and never imported
together. Constants, types, helpers, private stages, and public functions share
this global naming constraint.

For example, these two declarations collide:

```text
math/vectors.tach       function normalizeValue(...)
physics/forces.tach     function normalizeValue(...)
```

Module qualification does not disambiguate declarations in source. There is
no `math.normalizeValue`, alias import, or namespace object. Choose names that
are concise but project-distinct, such as `normalizeVector` and
`normalizeForceDirection`.

This also means that moving a declaration between files does not require
renaming its uses; update direct imports to point at the new owner. Before
adding a top-level name, search the entire Tach project, not only the current
module.

Public names have an additional host-language constraint. Type names, exported
function names, and exported parameter names must be portable ASCII
JavaScript/TypeScript identifiers and cannot be TypeScript keywords. Type names
also cannot be `Float16Array`, `Float32Array`, `Int32Array`, `Uint32Array`, or
`ReadonlyArray`.
Private helpers and stages may use the broader Tach Unicode identifier syntax,
but portable ASCII remains the least surprising application convention.

A Tach identifier begins with a Unicode letter or `_`, followed by Unicode
letters, digits, or `_`. Keywords and reserved intrinsic/type spellings cannot
be repurposed where the grammar requires an ordinary name. Filesystem module
and kernel identities are strings rather than identifiers, which is why a
hyphen is legal in `rigid-body/solve-contacts` but illegal in a function name.

## 9. Dependency graphs must be DAGs

Imports must be acyclic in two views:

- The kernel graph has one node per `.tach` file.
- The module graph collapses every file in the same directory into one node.

The module rule is stricter. These edges are invalid even if the individual
files do not form a literal file cycle:

```text
physics/integrate -> data/particles
data/packing      -> physics/constants
```

After collapsing by directory, `physics -> data -> physics` is a module cycle.
Organize dependencies in one direction. A reliable application structure is:

```text
foundation -> imports nothing
data       -> imports foundation
math       -> imports foundation
physics    -> imports data and math
rendering  -> imports data, math, and selected physics declarations
pipelines  -> imports lower-level stages and exposes public programs
```

This is an example direction, not a required module vocabulary. The required
property is that no dependency points back upward. If two modules need each
other, extract genuinely shared declarations into a lower module or move the
coupled files into one module. Do not attempt a re-export or alias workaround;
neither exists.

## 10. Source file anatomy

A kernel file has this order:

```text
optional file-level @docs(...);
zero or more contiguous imports
zero or more documented declarations
```

A representative file is:

```tach
@docs(
  title("Particle integration"),
  summary("Advances particle state with reusable value helpers."),
);

import "simulation/types";

@docs(
  summary("Advances one particle without mutating the input value."),
  param(particle, "State at the beginning of the step."),
  param(dt, "Elapsed simulation time."),
  returns("State at the end of the step."),
)
function integrateParticle(particle: Particle, dt: float32): Particle {
  return {
    position: particle.position + particle.velocity * dt,
    velocity: particle.velocity,
  };
}

@docs(
  summary("Advances every active particle in place."),
  coordinate(i, "Particle index."),
  param(particles, "Particle state updated in place."),
  param(dt, "Elapsed simulation time."),
)
export function integrate[i](particles: buffer<Particle[]>, dt: float32) {
  if (i < particles.length) {
    particles[i] = integrateParticle(particles[i], dt);
  }
}
```

Strings are documentation-only; Tach has no runtime string type. Statements
end with semicolons. Type fields may use commas or semicolons. Trailing commas
are accepted in parameter, argument, attribute, and object-literal lists.

## 11. Structured documentation

`@docs(...)` is checked source metadata, not an unstructured comment blob.
Every `@docs` attribute requires exactly one non-empty `summary`.

Allowed clauses depend on attachment context:

| Context | Allowed clauses beyond `summary` |
|---|---|
| kernel file | `title("...")` |
| type | `field(name, "...")` |
| any function | `param(name, "...")` |
| indexed function | `coordinate(name, "...")` |
| value-returning helper or view program | `returns("...")` |

File documentation is the first construct and ends with a semicolon:

```tach
@docs(
  title("Particle data"),
  summary("Shared values used by the simulation module."),
);
```

Declaration documentation immediately precedes the declaration and has no
semicolon after the attribute:

```tach
@docs(
  summary("Position and velocity for one particle."),
  field(position, "Current world-space position."),
  field(velocity, "Velocity applied during integration."),
)
type Particle = {
  position: vec<float32, 4>,
  velocity: vec<float32, 4>,
};
```

Names passed to `field`, `param`, and `coordinate` are unquoted identifiers.
They must resolve in the attached declaration. Unknown names and duplicate
clauses for one member are compile errors. `returns` is invalid on a void
helper, indexed stage, or ordinary orchestration program. A
`view<srgb8>` program may use it to describe the display result.
Module constants cannot carry `@docs`: they are compiler-only implementation
values with no generated API surface. Explain a non-obvious constant locally
with `//`, and document the public behavior it enables on the relevant type or
function.

Documentation on a public program describes its host API; documentation on a
private stage describes indexed work. An exported indexed function represents
both roles in one declaration, so document its host parameters and coordinates
together.

Write documentation for the semantic contract, not a restatement of syntax.
For buffers, say whether data is read, written, or updated in place. For
coordinates, state the indexing interpretation. For a program parameter,
state units, range, and relationship to dispatch shape where relevant.

## 12. Generated documentation behavior

Structured source documentation has two application-visible outputs:

- generated `index.d.ts` JSDoc on public types and recipe constructors; and
- generated Markdown in `build/README.md` and `build/docs/<module>.md`.

The root README uses `tach.json.docs.title` and `tach.json.docs.summary` and
links one document per discovered module. Each module document has one section
per kernel file in alphabetical order and includes documented types and
functions. Public function documentation also carries inferred buffer access.
Generated examples use the checked public TypeScript ABI, so they do not invent
host signatures separately from compilation.

Use:

```sh
npx tach docs
```

to refresh Markdown without recompiling or replacing existing target artifacts.
A successful `tach build` always refreshes the same documentation. `tach docs`
still parses and validates the project; it is not a text-only renderer.

Do not hand-edit generated Markdown. Change `tach.json` or source `@docs`, then
regenerate. Documentation failures are compilation failures because stale
field, parameter, or coordinate references represent an invalid API contract.

## 13. Inline comments

`//` is the only inline comment syntax:

```tach
// Rounded workgroups can execute coordinates beyond the logical domain.
if (i < values.length) {
  values[i] *= factor;
}
```

Block comments do not exist. Do not emit `/* ... */`, JSDoc syntax, or nested
comment markers in Tach source.

Use `@docs` for stable API and declaration meaning. Use `//` for a local reason
that helps a reader understand an implementation choice, such as an edge
guard, numerical assumption, synchronization protocol, unit conversion, or
non-obvious index mapping. Avoid comments that merely narrate the next line.

The formatter preserves comment text. An indivisible comment may exceed the
normal formatting width, but concise single-line comments are easier for both
people and agents to maintain.

## 14. Function roles at a glance

Tach has four source spellings and three semantic roles:

| Source form | Meaning | Host exposure |
|---|---|---|
| `function helper(...)` | pure per-invocation value helper | none |
| `function stage[i](...)` | private indexed GPU stage | none |
| `export function kernel[i](...)` | indexed stage plus one-dispatch program | recipe constructor |
| `export function program(...)` | explicit ordered orchestration, optionally returning a view | command or view constructor |

Choose by required work:

1. If one indexed dispatch solves the operation, use an exported indexed
   function.
2. If repeated value computation improves clarity, add private helpers.
3. If the operation needs multiple dispatches, scratch storage, or a display
   view, write private indexed stages and one exported unindexed program.
4. Do not export a private implementation stage merely to call it from another
   Tach file. Direct imports already provide Tach visibility.

All struct types appear in generated TypeScript by default. `export` applies
only to functions. An unexported indexed function remains usable as a `run`
target in directly importing files. A helper remains a value function. An
exported unindexed program is host-only orchestration and cannot be called by
Tach source.

## 15. Pure value helpers

A helper has no coordinate list and is never exposed to JavaScript:

```tach
function squaredLength(value: vec<float32, 3>): float32 {
  return dot(value, value);
}
```

Helper parameters and returns must be constructible values: booleans, numeric
scalars, numeric vectors, or fixed-footprint structs recursively made from
those values. A helper cannot:

- receive or access `buffer<T>`;
- declare or access `shared<T>`;
- use barriers or atomics through memory;
- call an indexed stage or orchestration program;
- recurse directly or indirectly; or
- return a runtime array, fixed array, atomic, buffer, or shared value.

A missing result annotation means `void`. Every path through a non-void helper
must return exactly its declared type. Helpers may call other visible helpers
and use ordinary expressions, local variables, branches, and loops. Use helpers
to centralize numerical formulas and transformations, not to simulate host
dispatch control.

## 16. Indexed stages

An indexed stage describes the work performed by one logical GPU invocation:

```tach
function copy[i](input: buffer<float32[]>, output: buffer<float32[]>) {
  if (i < input.length && i < output.length) {
    output[i] = input[i];
  }
}
```

An indexed stage:

- has one, two, or three coordinate names;
- treats each coordinate as immutable `uint32`;
- has at least one buffer parameter;
- may have constructible immutable value parameters;
- has no return type and cannot return a value;
- may call visible value helpers;
- may use storage, shared memory, atomics, and barriers under their rules; and
- cannot be called as an expression.

An unexported stage executes only when a public program names it in `run`.
The stage can live in the program's file or in a directly imported file.

`return;` may end one invocation early. This is safe for ordinary stages, but
never place an early return on a path that would cause only part of a workgroup
to skip a later barrier.

## 17. Exported indexed shorthand

The baseline Tach entry point is:

```tach
export function scale[i](values: buffer<float32[]>, factor: float32) {
  if (i < values.length) {
    values[i] *= factor;
  }
}
```

This one declaration means both:

- an indexed stage named `scale`; and
- a public one-dispatch program named `scale`.

The generated TypeScript function preserves source parameter order and adds a
rank-specific final `LaunchOptions` parameter.

The generated call constructs a command; it does not run the stage. For a
simple map, image pass, particle update, or independent per-element transform,
this form is preferred over a one-stage explicit program.

Use `export` only when the TypeScript application should construct this
operation directly. Exporting a function does not make it more visible to Tach
source and does not change its numerical semantics.

## 18. Explicit orchestration programs

Use an exported function without coordinates when one host operation requires
multiple ordered dispatches or compiler-managed scratch:

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

An orchestration program body contains compile-time `const` declarations,
runtime shape and transient `let` declarations, and `run` statements, plus a
final return when it declares `view<srgb8>`. Every program has at least one
`run`. An ordinary program has at least one external buffer parameter; a view
program may use only plain parameters and a transient frame. It cannot contain
general value locals, assignments, `if`, loops, barriers, `@workgroup`,
arbitrary host logic, or any other return. A `const` here obeys the same
compiler-only rules as every other constant; it cannot depend on a program
parameter or shape `let`.

Each `run` targets a private indexed stage and creates one ordered dispatch.
An explicit program cannot be called from Tach, cannot be a `run` target, and
does not execute once per GPU invocation. It describes the dispatch plan that
the generated host command will execute.

### View programs

`view<srgb8>` is the only public display result. It is valid only on an
exported unindexed program, and the last statement must be exactly the
form `return view(pixels, width, height);`:

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

`pixels` directly names a public buffer or transient of runtime
`vec<float32, 4>[]`. Width and height are positive checked shape expressions. Their
product cannot exceed the available pixel count. The source resource's exact
final version must be defined, so a transient frame needs a preceding complete
write.

Pixels are linear RGBA. Backend lowering clamps RGB, applies the IEC sRGB
transfer, clamps alpha, and quantizes every channel with
`uint32(channel * 255 + 0.5)` into one packed RGBA8 `uint32` word. Both
targets share that word. WebGPU unpacks it with `unpack4x8unorm` into an
`rgba8unorm` texture so `present` can write a 2D image; Vulkan stores the word
in packed scratch. Tach
source never packs bytes or names a texture, surface, canvas, WebGPU object,
or Vulkan object. Target planning may fold projection into a proven final
one-pixel-per-index dispatch and remove the full-frame float transient;
otherwise it adds a projection stage. This is an implementation choice with
identical source and host behavior.

The generated function returns `ComputeView`, which extends `ComputeCommand`.
Use `submit(view)` for offscreen projection on either host. Use
`present(canvas, view)` for direct browser display and completion-backed frame
backpressure. Deno/Vulkan has no native presentation surface and rejects
`present`; it still executes the same view through `submit`. View recipes
reject `repeat`; construct the CPU-selected frame recipe once per frame.

## 19. Export and generated API rules

`export` has exactly one role: it gates host recipe constructors. Every named
struct type is represented in generated TypeScript regardless of whether a
public function uses it.

There is no `export type` form. There is no default export. There are no
re-exports. The generated package has one entry point containing all public
types and exported functions from the complete Tach project. Ordinary exports
return `ComputeCommand`; view exports return `ComputeView`.

An exported source parameter name becomes a generated TypeScript parameter
name. Avoid names likely to confuse application code. Compiler-generated final
options use collision-safe naming when needed, and runtime types are imported
through private aliases so a Tach type named `ComputeBuffer` does not collide.

## 20. Coordinates and logical domains

Coordinates appear after an indexed function name:

```tach
function line[i](out: buffer<uint32[]>) { ... }
function image[x, y](out: buffer<vec<float32, 4>[]>) { ... }
function volume[x, y, z](out: buffer<float32[]>) { ... }
```

Each coordinate is an immutable `uint32` local. It is supplied by the GPU
invocation and is not a host parameter. Names have no built-in semantics;
`x`, `y`, `z`, `i`, or domain-specific names are equivalent.

For exported indexed shorthand, TypeScript supplies a logical launch size of
matching rank:

```ts
line(out, { size: 1_000_000 });
image(out, { size: [1920, 1080] });
volume(out, { size: [64, 64, 64] });
```

For an explicit orchestration program, every `run` states its domain in Tach:

```tach
run lineStage(out) over count;
run imageStage(out, width) over [width, height];
run volumeStage(out, width, height) over [width, height, depth];
```

The number of domain components must equal the stage coordinate rank.

## 21. Workgroups

Without an attribute, Tach chooses portable defaults:

| Stage rank | Default workgroup dimensions |
|---:|---:|
| 1D | `256 x 1 x 1` |
| 2D | `16 x 16 x 1` |
| 3D | `8 x 8 x 4` |

Use defaults unless the algorithm depends on an exact local group shape.
Declare an explicit shape with `@workgroup`:

```tach
@workgroup(16, 16)
export function shade[x, y](pixels: buffer<vec<float32, 4>[]>, width: uint32) {
  let index = y * width + x;
  if (index < pixels.length) {
    pixels[index] = vec(0, 0, 0, 1);
  }
}
```

The attribute is valid only on indexed stages. It accepts one through the
stage-rank number of positive compile-time `uint32` expressions; omitted axes
are `1`. Portable limits are:

```text
x <= 256
y <= 256
z <= 64
x * y * z <= 256
```

Any stage using shared memory or a barrier must declare `@workgroup`
explicitly. Do not tune workgroup dimensions speculatively. Choose a shape
because shared-array dimensions, lane mapping, or measured application behavior
requires it.

## 22. Rounded dispatches and mandatory edge guards

Logical sizes are rounded upward to whole workgroups. If a logical 1D domain is
1,000 and the workgroup size is 256, 1,024 invocations may execute. Tach does
not synthesize dynamic bounds checks.

Guard every access that may receive an edge coordinate:

```tach
export function clear[i](values: buffer<float32[]>) {
  if (i >= values.length) {
    return;
  }
  values[i] = 0;
}
```

For a flattened 2D buffer, guard both logical axes or the final index as the
algorithm requires. Do not assume a buffer length proves both axes are valid
when row stride, padding, or a subregion makes the mapping more restrictive.

## 23. Launch-size inference

Only exported indexed shorthand accepts host launch size.

For a 1D shorthand, omitting `size` uses the first runtime-sized public buffer
when one exists:

```ts
await gpu.submit(scale(values, 2));
```

If no runtime-sized public resource provides a length, omission launches
exactly one workgroup. This matters for fixed-size buffer types and kernels
whose logical count differs from storage length. Supply `{ size: count }`
whenever inference would be ambiguous to a reader or incorrect for the domain.

Two- and three-dimensional shorthand normally requires explicit tuple size:

```ts
await gpu.submit(image(pixels, width, height, { size: [width, height] }));
```

Every size component must be a positive JavaScript safe integer within the
supported unsigned range, and tuple rank must match coordinates exactly.

Explicit orchestration programs never accept a host `size`; their `run`
domains are fully defined in Tach source. Passing launch size to such a program
is a host API error and a TypeScript type error.

## 24. Shape expressions in programs

Each explicit-program `run` domain and each view width/height is a checked
`uint32` shape expression. It may use:

- a `uint32` literal;
- a public `uint32` parameter;
- nested `uint32` fields of public structs;
- `.length` on a public runtime array or runtime-array field;
- an earlier runtime shape `let`;
- `+`, `-`, `*`, `/`, or `%`; and
- shape-only `min(a, b)`, `max(a, b)`, and `ceilDiv(a, b)`.

Example:

```tach
function initialize[i](scratch: buffer<float32[]>) {
  if (i < scratch.length) {
    scratch[i] = float32(i);
  }
}

function finish[i](
  scratch: buffer<float32[]>,
  output: buffer<float32[]>,
) {
  if (i < scratch.length && i < output.length) {
    output[i] = sqrt(scratch[i]);
  }
}

export function process(
  output: buffer<float32[]>,
  width: uint32,
  height: uint32,
) {
  let pixels = width * height;
  let groups = ceilDiv(pixels, 256);
  let rounded = groups * 256;
  let scratch = transient<float32>(rounded);
  run initialize(scratch) over rounded;
  run finish(scratch, output) over pixels;
}
```

Shape evaluation occurs on the host with checked unsigned 32-bit arithmetic.
Underflow, overflow, division or remainder by zero, and a zero dispatch
dimension are runtime errors. Do not rely on wraparound in shape expressions.
`ceilDiv` exists only here; it is not a helper/stage math intrinsic.

Arguments of `run` are also intentionally constrained. A buffer argument must
directly name one public buffer parameter or one earlier transient declaration;
arbitrary indexing, conditionals, and buffer-producing expressions are invalid.
A value argument may be a matching public value, a nested field of one, a
compile-time constant expression, or, when the stage formal is `uint32`, an
earlier checked runtime shape. A constant argument specializes the physical
stage and never becomes runtime parameter metadata. These restrictions keep
orchestration declarative and make all resource and parameter sources knowable
before GPU execution.

## 25. Transient storage

An explicit program allocates private scratch with:

```tach
let scratch = transient<float32>(count);
```

This creates a program-local `buffer<float32[]>`. The TypeScript caller does
not pass or receive a handle for it. A transient element type must have a fixed,
host-shareable, non-atomic footprint.

Use transients for intermediate arrays whose lifetime is entirely inside one
public operation. Keep caller-owned state as a public buffer. This distinction
usually makes an API clearer:

```text
state needed across commands or frames -> public ComputeBuffer
scratch needed only between program runs -> transient<T>(length)
```

A transient read must be preceded by an earlier dispatch that defines it.
Every stage buffer argument in `run` directly names a public buffer or a
transient. A resource cannot fill two buffer parameters of the same stage.
Unlike shared workgroup memory, a transient has no zero-initialization
guarantee; its defining stage must write every element a later stage may read.

Transient storage is reusable across non-overlapping internal lifetimes, but
this is not a reason to split one logical scratch value into aliases. Express
the real dataflow and let the runtime manage physical scratch.

Every source `run` remains an ordered physical dispatch. Do not assume distinct
stages are fused. Terminal view projection alone may be folded into its final
writer under an exact proof; that does not fuse two source `run` statements.
If a value can remain a local inside one indexed stage, that is materially
different from writing it to a transient between dispatches.

## 26. Scalar types

Tach scalar types are explicit:

| Type | Meaning |
|---|---|
| `bool` | `true` or `false`; value-only outside parameter blocks |
| `int32` | signed 32-bit integer |
| `uint32` | unsigned 32-bit integer |
| `float16` | IEEE 754 binary16 |
| `float32` | IEEE 754 single precision |
| `void` | absence of a helper result |

There is no Tach `number`, `f64`, `i64`, arbitrary precision integer, string,
null, undefined, optional value, enum, union, tuple type, class, interface, or
generic user-defined type.

`view<srgb8>` is a terminal public-program contract, not a constructible value
type. It cannot be used for a local, helper result, parameter, field, buffer,
array, or indexed-stage result. `srgb8` is the only view format and is not an
ordinary type on its own.

Use `uint32` for coordinates, counts, lengths, indices, masks whose ordering is
unsigned, and program shapes. Use `int32` for signed integer arithmetic. Use
`float32` for general numerical work. Use `float16` deliberately when its
half-sized storage, bandwidth, or hardware throughput matters and the
algorithm tolerates its much smaller range and precision. Host TypeScript
represents all four numeric scalars as `number`. Generated codecs enforce
integer ranges, boolean representation, aggregate shape, and each declared
floating width. A Float16 buffer uses `Float16Array` on the host and remains
binary16 through arithmetic and both shader backends; it is not storage sugar
for widened float32.

Float16 is an optional GPU feature. WebGPU execution requires `shader-f16`.
Vulkan execution requires `shaderFloat16` and the relevant 16-bit
storage/uniform feature according to how the project uses the type. A host
session enables supported Float16 functionality when it opens; each generated
module records its exact requirements, and command preparation checks them. A
Float16 command fails on unsupported hardware, while a module without Float16
retains the ordinary feature floor.

`bool` can be passed as a plain parameter and used in local values. It has no
direct storage-buffer representation, so do not place it in buffer-backed
structs or arrays. `vec<bool, N>` is a value-only lane mask: use it in
constants, locals, helper parameters/results, and value-only structs, but not
as a public parameter, buffer-backed value, or shared value. If persistent
boolean-like storage is needed, use a documented `uint32` convention such as
`0` and `1`, load it, and compare to derive a mask.

## 27. Vectors and boolean masks

`vec<T, N>` is Tach's sole vector type syntax. `T` is a scalar and `N`
must be `2`, `3`, or `4`:

```text
vec<float16, 2>  vec<float16, 3>  vec<float16, 4>
vec<float32, 2>  vec<float32, 3>  vec<float32, 4>
vec<int32, 2>    vec<int32, 3>    vec<int32, 4>
vec<uint32, 2>   vec<uint32, 3>   vec<uint32, 4>
vec<bool, 2>     vec<bool, 3>     vec<bool, 4>
```

`vec(...)` is the sole vector value constructor. It flattens scalar and vector
arguments of one element type to exactly two, three, or four total lanes:

```tach
let a = vec(1, 2, 3, 4);
let b = vec(vec(1, 2), 3, 4);
let allHalf = vec(0.5, 0.5, 0.5, 0.5);
let alternating = vec(true, false, true, false);
```

Context determines its element type and the arguments determine its width:

```tach
function direction(): vec<float16, 3> {
  return vec(1, 2, 3);
}

function extend(value: vec<float32, 3>): vec<float32, 4> {
  return vec(value, 1);
}
```

The surrounding type or a concrete argument constrains `vec`; otherwise whole
lanes default to `uint32` and any fraction/exponent defaults the vector to
`float32`. It never converts a typed value. There is no one-argument splat and
no alternate named vector constructor. Repeat a scalar explicitly, use a
documented scalar/vector broadcast operation, or explicitly convert each lane
and rebuild the vector.

Swizzles use `x`, `y`, `z`, and `w`:

```tach
let horizontal = value.xz;
let alpha = color.w;
```

One selected lane yields a scalar; multiple lanes yield a matching vector.
`value[index]` dynamically selects a lane. A vector lane is addressable when
its base is an addressable mutable local, buffer place, or shared place.

Most arithmetic supports equal-type numeric vectors and documented
scalar/vector broadcast. Numeric vector comparisons produce same-width boolean
masks. Combine masks with `!`, `&`, `|`, or `^`; reduce them with `all`/`any`;
choose lanes with `select`. Do not assume matrix types or arbitrary swizzle
alphabets exist.

## 28. Struct types and literals

Structs are named object-shaped value types:

```tach
type Particle = {
  position: vec<float32, 4>,
  velocity: vec<float32, 4>,
};
```

Construct a struct with an object literal in a context that determines its
named type:

```tach
function makeParticle(position: vec<float32, 4>, velocity: vec<float32, 4>): Particle {
  return {
    velocity: velocity,
    position: position,
  };
}
```

Every declared field must appear exactly once. Literal field order is
irrelevant. Declaration field order is relevant to host memory layout and must
not be casually reordered for buffer-backed types.

Structs are value types. They may nest other compatible structs. Recursive
struct definitions are invalid. A fixed-footprint constructible struct may be
passed to and returned from helpers and supplied as a plain public parameter.
A host-shareable struct may be stored in a buffer. A struct with a runtime
array may place that array only in its final field and is a storage shape, not a
whole constructible value.

All structs are emitted as readonly TypeScript shapes. Match exact field names
when creating host values; generated codecs reject missing or extra fields,
wrong vector/array widths, malformed values, and out-of-range integers.

## 29. Runtime and fixed arrays

`T[]` is a runtime-sized array. It can appear:

- directly as a buffer payload, as in `buffer<float32[]>`; or
- as the final field of one host-shareable struct.

It exposes `.length` only through an addressable place:

```tach
export function copy[i](
  input: buffer<float32[]>,
  output: buffer<float32[]>,
) {
  if (i < input.length && i < output.length) {
    output[i] = input[i];
  }
}
```

A runtime array cannot be constructed, loaded, assigned, passed, or returned
as one whole value. Read and write individual elements. A materialized runtime
resource must contain at least one complete element; zero-length or partially
packed runtime resources are invalid.

`T[N]` is a fixed array whose length is a positive compile-time `uint32`
expression. Fixed arrays currently exist only in shared workgroup memory:

```tach
let partial: shared<float32[64]>;
```

Do not use fixed arrays as ordinary locals, helper values, public parameters,
or buffer payloads.

## 30. Buffer parameters

An indexed stage accesses GPU storage through `buffer<T>` parameters:

```tach
export function saxpy[i](
  x: buffer<float32[]>,
  y: buffer<float32[]>,
  alpha: float32,
) {
  if (i < x.length && i < y.length) {
    y[i] = alpha * x[i] + y[i];
  }
}
```

Tach infers whether each public buffer is read, written, read-write, or atomic
from the complete public program. Source never states storage access modes,
binding groups, descriptor sets, binding indices, or storage classes.

Plain function parameters are immutable values. Buffer contents are addressable
storage and may be read or written according to source operations. A stage must
have at least one buffer parameter; pure value computation belongs in a helper.

The same logical public buffer may flow through multiple stages of one explicit
program. Dispatches are ordered, so later stages observe earlier completed
writes under the program plan. This does not make unsynchronized writes within
one stage safe; per-invocation concurrency rules still apply.

## 31. Non-aliasing and in-place work

Different public buffer parameters of one command promise distinct storage.
The managed runtime rejects this:

```ts
const values = gpu.buffer(new Float32Array(1024));
await gpu.submit(copy(values, values)); // Invalid: two buffer positions alias.
```

This rule applies even if an algorithm would happen to read one region and
write another. Tach has no public alias declaration or sub-buffer view that can
prove disjoint ranges.

Express genuine in-place work with one buffer parameter:

```tach
export function squareInPlace[i](values: buffer<float32[]>) {
  if (i < values.length) {
    values[i] *= values[i];
  }
}
```

Inside an explicit program, one public or transient resource likewise cannot
fill two buffer formals of the same stage invocation. Redesign the stage to use
one formal when it intentionally accesses one storage object in multiple ways.

Different commands may of course reuse the same `ComputeBuffer`; persistent
GPU state depends on that. The restriction is on distinct buffer positions of
one public command and one physical stage, not on sequential command use.

## 32. Host-visible storage representation

The runtime accepts ordinary TypeScript values and performs validated layout
conversion. Application code should use generated declarations rather than
manually calculating offsets. A few representation facts matter when choosing
host data:

| Tach storage value | TypeScript representation |
|---|---|
| `int32`, `uint32`, `float16`, `float32` | `number` |
| storage atomic | `number` |
| numeric vector | readonly numeric tuple |
| named struct | generated readonly object |
| scalar runtime array | matching typed array or readonly number array |
| 2-/4-lane vector array | flat matching typed array or tuple array |
| 3-lane vector array | tuple array |

Three-lane vectors have padded element stride in arrays. A flat typed array
would falsely imply tightly packed triples, so use generated tuple-array types:

```ts
const positions: readonly (readonly [number, number, number])[] = [
  [0, 0, 0],
  [1, 2, 3],
];
```

Scalar, two-lane, and four-lane runtime arrays are tightly packed and can use
matching `Float16Array`, `Float32Array`, `Int32Array`, or `Uint32Array` values. Readback
preserves an accepted typed-array representation where possible.

Struct field order determines layout even though object-literal order does
not. Do not construct padding fields. Generated codecs validate exact fields,
integer ranges, vector widths, array completeness, and all required strides.

Use the generated declaration as the final authority for nested shapes. A
fixed buffer such as `buffer<Config>` receives a `ComputeBuffer<Config>` whose
host object has every generated field. A direct `buffer<uint32[]>` naturally
accepts `Uint32Array` or a readonly number array. Atomic leaves are initialized
with ordinary integer numbers, not JavaScript atomic wrapper objects. Runtime
tails remain fields of their containing host object; they are not passed as a
second buffer. Plain `bool` parameters are JavaScript booleans, never numeric
truthiness values.

## 33. Practical storage layout rules

Most applications should trust generated types and codecs, but these rules
explain memory-size differences and prevent incorrect typed-array assumptions:

| Tach type | Size | Alignment |
|---|---:|---:|
| `float16` | 2 | 2 |
| `int32`, `uint32`, `float32`, integer atomic | 4 | 4 |
| `vec<float16, 2>`, `vec<float16, 3>`, `vec<float16, 4>` | 4 / 6 / 8 | 4 / 8 / 8 |
| 32-bit two-lane numeric vector | 8 | 8 |
| 32-bit three-lane numeric vector | 12 | 16 |
| 32-bit four-lane numeric vector | 16 | 16 |

A host-visible struct has at least 16-byte alignment. Each field begins at its
required alignment; the final struct extent is rounded to its alignment. These
sizes apply to buffers and parameter blocks, not to workgroup `shared` objects.
For example:

```tach
type Particle = {
  position: vec<float32, 3>,
  mass: float32,
  velocity: vec<float32, 3>,
};
```

has `position` at byte 0, `mass` at byte 12, `velocity` at byte 16, and a
32-byte final extent. Host objects still contain only the three declared
fields.

Runtime-array element stride is the element size rounded to its alignment.
A direct odd-length `float16[]` remains tightly packed at two bytes per element;
the runtime/backend privately handles four-byte transfer and WebGPU binding
padding without changing `.length` or exposing a phantom element.
A trailing scalar `float16[]` receives the same treatment when its struct
prefix leaves the complete logical byte extent off a four-byte boundary.
A struct may have one runtime-array tail after a fixed prefix. Multi-byte
values use little-endian representation. A fixed-size public buffer's physical
allocation is rounded to 16 bytes; the host value still contains only its
logical fields. Runtime arrays retain their natural stride so padding never
becomes a phantom element. The managed runtime owns these details; use them to
estimate memory, not to bypass the generated codec.

## 34. Numeric literals

Tach uses ordinary unsuffixed numeric spelling:

```tach
let decimal = 42;
let separated = 1_000_000;
let hexadecimal = 0xff00_ff00;
let binary = 0b1010_0001;
let fraction = 1.25;
let exponent = 6.022e2;
```

Shader suffixes such as `0u`, `1i`, and `1.0f` are invalid. Inference is
expression-local. Its deterministic precedence is explicit types and
conversions, expected assignment/argument/return/field/result context,
concrete sibling operands, intrinsic domains, then defaults. Without context:

- a non-negative whole literal infers `uint32`;
- a fractional or exponent literal infers `float32`; and
- unary minus gives a whole literal signed `int32` context.

An all-literal floating intrinsic defaults to `float32`; `abs(1)` defaults to
`int32`. Multi-operand expressions and intrinsics resolve their operands
collectively, so swapping a contextual literal from left to right cannot
change the result.

There is no unconstrained `float16` inference. A binary16 literal receives its
type from an annotation, parameter/result/assignment context, or
`float16(...)`; it must be finite and within `-65504` to `65504`.

Use an annotation or explicit conversion when a literal's intended domain is
not obvious:

```tach
let signed: int32 = -1;
let count: uint32 = 64;
let half: float16 = 0.5;
let scale: float32 = 2;
```

Literal range is checked. Do not use JavaScript numeric suffixes, bigint
syntax, `NaN`, `Infinity`, or hexadecimal floating-point syntax.

## 35. Explicit numeric conversion

The only general numeric conversions are constructor-shaped:

```tach
int32(value)
uint32(value)
float16(value)
float32(value)
```

Tach does not silently mix arbitrary numeric types. Convert deliberately at
domain boundaries:

```tach
let normalized = float32(i) / float32(count);
let cell = uint32(floor(position));
```

Integer-to-integer conversion preserves the low 32-bit pattern. Therefore
`uint32(-1)` is a bit-pattern conversion, not a range clamp. Conversion from
floating point to integer should be used only with values whose range and rounding have
been made explicit by the algorithm.

`float16(float32Value)` explicitly narrows with IEEE binary16 rounding;
`float32(float16Value)` explicitly widens. Neither conversion changes the
precision already lost by prior binary16 computation.

`vec(...)` builds vectors; the four functions above convert scalars. Vector
conversion is deliberately lane-explicit: extract and convert each component,
then rebuild it with `vec(...)`. Tach has no second vector constructor or
implicit aggregate conversion path.

Host TypeScript numbers are validated when a command is prepared or a buffer
is materialized. Passing `1.5` to a generated `uint32` parameter remains an
error even though TypeScript represents it as `number`.

## 36. Variables and lexical scope

Tach has one runtime local declaration, `let`, and one compile-time declaration,
`const`. They are different execution categories, not two immutability choices
for the same runtime value.

`let` evaluates where its surrounding function or program runs and may be
reassigned. It may have a type annotation:

```tach
let lanes: uint32 = 4;
let total: float32 = 0;
```

Even `let fixed = 4;` is a runtime local; later optimization may fold it, but
the source declaration makes no compile-time promise. Function parameters and
coordinates cannot be assigned. A `for` initializer is always a loop-scoped
`let`.

`const` is evaluated completely by the compiler and may produce only a scalar
or vector, including a boolean mask:

```tach
const tileWidth: uint32 = 16;

@workgroup(tileWidth)
export function tiled[i](out: buffer<uint32[]>) {
  const tileArea = tileWidth * tileWidth;
  const direction = normalize(vec(3.0, 4.0, 0.0));
  let partial: shared<uint32[tileArea]>;
  let lane = i % tileArea;
  partial[lane] = i;
  workgroupBarrier();
  if (i < out.length) {
    out[i] = uint32(direction.x * float32(partial[lane]));
  }
}
```

A module constant may refer to visible module constants in any declaration
order. Imported constants require a direct import, just like imported types and
functions. A local constant may refer to module constants and earlier constants
in its active lexical scope; local forward references are invalid. Constant
cycles are errors and report the dependency chain.

Constant expressions use ordinary Tach typing and exactly this algebra:

- literals and constant identifiers;
- unary `!`, `-`, and `~`;
- arithmetic, comparisons, short-circuit logic, bitwise operations, and shifts;
- the lazy conditional expression;
- numeric scalar conversions;
- `vec(...)`, vector indexing, and swizzles; and
- pure numeric, `fma`, vector-geometry, and mask intrinsics.

They cannot use structs, runtime arrays, buffers, parameters, coordinates,
`let` bindings, transient allocation, barriers, atomics, or user-function
calls. Tach constants are deliberately not macros, conditional compilation, or
a general compile-time programming language.

Evaluation obeys the declared or inferred Tach type at every operation.
Integer arithmetic wraps to 32 bits and shifts mask the count to five bits.
Division or remainder by zero and signed minimum divided by `-1` are compile
errors. Float16 and Float32 round back to their type after every operation;
NaN, infinity, and values outside the finite range are errors. Lazy logic and
conditionals evaluate only the selected branch.

The compiler substitutes the evaluated value at every use. The same constant
may therefore drive `@workgroup`, a shared-array length, a loop bound, ordinary
math, or a program `run` argument. A constant passed to a stage specializes the
physical stage before either backend is lowered; it appears in neither the
generated TypeScript signature nor the runtime parameter block. Module
constants have no JavaScript/TypeScript export surface. Unused module and local
constants are warnings.

Names cannot shadow another active name. This is invalid:

```tach
let value = 1;
if (condition) {
  let value = 2;
}
```

Branch-local declarations do not escape their branch. Top-level shared
declarations follow their stage scope but must appear directly in the stage
body.

Tach locals are values, not JavaScript objects with reference identity.
Reassigning a struct or vector local replaces its value. Buffer and shared
places are memory and obey parallel access rules.

## 37. Operators

Supported operator families are deliberately narrow:

| Family | Supported operands |
|---|---|
| unary `!` | `bool` or `vec<bool, N>` |
| unary `-` | signed numeric scalar/vector |
| unary `~` | integer scalar/vector |
| `+ - * /` | matching numeric values; documented scalar/vector broadcast |
| `%` | matching numeric scalars |
| `== != < <= > >=` | matching numeric scalars/vectors; vector result is `vec<bool, N>`; equality also accepts booleans |
| `&& ||` | `bool`, with short-circuit evaluation |
| `& \| ^` | matching integer or boolean values; scalar/vector broadcast |
| `<< >>` | integer scalar/vector with unsigned scalar or lane-wise counts |

Unsigned negation is invalid. A shift masks its count to the low five bits.
Signed right shift is arithmetic; unsigned right shift is logical.

Do not assume JavaScript coercion. Both operands must satisfy Tach's exact
typing and broadcast rules. There is no string concatenation, nullish
coalescing, optional chaining, exponentiation operator, identity comparison,
or overloaded user operator.

The conditional expression is lazy and requires equal branch result types:

```tach
let safe = denominator == 0 ? 0 : numerator / denominator;
```

Only the selected branch evaluates.

## 38. Precedence and associativity

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

Binary operators associate left. Use parentheses when an expression mixes
bitwise, comparison, and arithmetic operators or when a reader might import
JavaScript assumptions:

```tach
let index = (y * width) + x;
let active = (flags & mask) != 0;
```

Function calls, member access, and indexing compose from left to right. Calls
are always direct names; Tach has no function values, closures, methods,
constructors with object identity, or dynamic dispatch.

## 39. Assignment and addressable places

Supported assignments are:

```text
=  +=  -=  *=  /=  %=  &=  |=  ^=  <<=  >>=  ++  --
```

Targets must be mutable locals or addressable buffer/shared places. Examples:

```tach
total += value;
values[i] *= factor;
particle.position.x = nextX;
lane++;
```

A member or vector lane is addressable only when its base is addressable. A
field selected from a temporary value is not a persistent place. Atomics are
never accessed with ordinary assignment; use atomic operations.

Compound assignment evaluates according to the target type and operator
rules. Do not use postfix increment for a value result; treat `++` and `--` as
statements that mutate their target.

## 40. Control flow

Tach supports `if`, `while`, and `for`. Conditions must be `bool`, and headers
use parentheses:

```tach
if (value > 0) {
  total += value;
} else if (value < 0) {
  total -= value;
} else {
  total += 1;
}
```

```tach
let index: uint32 = 0;
while (index < count) {
  total += index;
  index++;
}
```

```tach
for (let lane: uint32 = 0; lane < 4; lane++) {
  total += vector[lane];
}
```

A `for` initializer is a `let` declaration. Its update is assignment,
compound assignment, `++`, or `--`. `break;` exits the nearest enclosing
`while` or `for`. `continue;` starts the nearest loop's next iteration; in a
`for`, its update executes first. Both may sit inside nested branches and
scopes, but neither may occur outside a loop. Code after an unconditional
`break`, `continue`, or `return` is unreachable and rejected.

```tach
for (let i = start; i < values.length; i += stride) {
  if (values[i] == 0.0) {
    continue;
  }
  total = fma(values[i], scale, total);
  if (total >= threshold) {
    break;
  }
}
```

Tach has no labeled transfer, `switch`, exceptions, `do while`, `for of`, or
`for in`. A transfer always targets the lexically nearest loop.

A helper returns its declared value; a void helper or indexed stage may use
`return;`. Statements after an unconditional return are rejected. Public
orchestration programs do not support ordinary control flow at all.

## 41. Math intrinsics

Mask intrinsics complete vector comparison:

```tach
let inside = point >= lower & point <= upper;
let clipped = select(inside, point, 0.0);
if (all(inside) || any(clipped != point)) {
  // Conditions remain scalar bool.
}
```

- `all(mask)` returns true only when every `vec<bool, N>` lane is true.
- `any(mask)` returns true when at least one lane is true.
- `select(mask, whenTrue, whenFalse)` returns an `N`-lane numeric or boolean
  vector and broadcasts scalar arms. Its arguments are mask-first in Tach.

Mask logic and `select` are eager. Both operands/arms execute before the result
is formed. They cannot guard an invalid load, division, or other operation that
must not happen. Scalar `&&`, `||`, and `condition ? true : false` are the lazy
forms; reduce a mask before using it as their condition.

These free functions preserve their `float16` or `float32` scalar/vector type:

```text
floor  ceil  trunc
sin    cos   tan
exp    exp2  log  log2
sqrt   rsqrt
```

Additional rules:

- `abs` accepts `int32`, `float16`, or `float32` scalars/vectors.
- `pow` accepts matching floating values and can broadcast a scalar exponent
  across a vector base.
- `fma(a, b, c)` accepts `float16` or `float32` values and computes
  component-wise `a * b + c`; equal-width vectors may mix with scalars, which
  broadcast to that width.
- `min`, `max`, and `clamp` accept numeric scalars/vectors and broadcast scalar
  arguments to a shared vector width.

Bounds have exact comparison-shaped semantics:

```text
min(a, b)         = b if b < a, otherwise a
max(a, b)         = b if a < b, otherwise a
clamp(x, low, hi) = min(max(x, low), hi)
```

An unordered floating comparison is false. Equal operands preserve the first
operand, including signed zero. Inverted clamp limits produce `hi`. Do not
substitute a host library's differently specified NaN or inverted-limit rule
when writing an exact CPU oracle.

Geometric intrinsics are:

| Function | Input | Result |
|---|---|---|
| `dot(a, b)` | matching `vec<float16, N>` or `vec<float32, N>` | component type |
| `length(value)` | floating vector | component type |
| `distance(a, b)` | matching floating vectors | component type |
| `cross(a, b)` | matching three-lane floating vectors | same vector type |
| `normalize(value)` | floating vector | same vector type |

Intrinsic names are reserved. Do not redefine them as types or functions.
Remember that both floating widths have finite precision and target-portable
GPU math may not be bit-identical to CPU double precision.
`fma` explicitly carries multiply-add intent through WGSL and SPIR-V, but a
backend or device may still choose fused hardware or separate operations. Do
not rely on a particular physical instruction count or intermediate rounding.

## 42. Shared workgroup memory

Declare workgroup-local memory only as an uninitialized direct child of an
indexed stage body:

```tach
@workgroup(64)
export function blockFirstValues[i](
  input: buffer<uint32[]>,
  out: buffer<uint32[]>,
) {
  let partial: shared<uint32[64]>;
  let lane = i % 64;

  partial[lane] = i < input.length ? input[i] : 0;
  workgroupBarrier();

  let block = i / 64;
  if (lane == 0 && block < out.length) {
    out[block] = partial[0];
  }
}
```

The first runtime-sized buffer makes an omitted 1D launch size follow
`input.length`; rounded edge lanes contribute zero before every lane reaches
the barrier.

Shared memory:

- is allocated once per workgroup, not once per invocation or dispatch;
- is visible only to invocations in that workgroup;
- is zero-initialized before source instructions;
- requires an explicit `@workgroup` attribute; and
- may contain numeric scalars/vectors, integer atomics, compatible structs,
  and fixed arrays of workgroup-storable values.

Do not assume one workgroup can see another workgroup's shared state. A fixed
array length and workgroup dimensions must agree with the algorithm's lane
mapping; the language does not infer that relationship.

## 43. Atomic storage and operations

Atomic types are:

```tach
atomic<int32>
atomic<uint32>
```

They may appear in host buffers or shared memory. Access an atomic place only
through:

| Operation | Effect | Result |
|---|---|---|
| `atomicLoad(place)` | read current value | current value |
| `atomicStore(place, value)` | replace | `void` |
| `atomicAdd`, `atomicSub` | arithmetic update | previous value |
| `atomicMin`, `atomicMax` | ordered update | previous value |
| `atomicAnd`, `atomicOr`, `atomicXor` | bitwise update | previous value |
| `atomicExchange` | replace | previous value |
| `atomicCompareExchange` | conditional replace | previous value |

`atomicCompareExchange(place, expected, replacement)` atomically reads the
place, stores `replacement` only if the old value equals `expected`, and
returns that old value. It is strong: a returned value different from
`expected` proves the comparison failed because the place held that value.
Tach retries WGSL's weak primitive internally, so never write a source retry
loop for spurious failure and never expect a backend-shaped result structure.

Example global accumulation:

```tach
type Counters = {
  total: atomic<uint32>,
};

export function countPositive[i](
  values: buffer<float32[]>,
  counters: buffer<Counters>,
) {
  if (i < values.length && values[i] > 0) {
    atomicAdd(counters.total, 1);
  }
}
```

Buffer atomics use WebGPU device scope and Vulkan queue-family scope; shared
atomics have workgroup scope. Tach has one compute queue, so those buffer
scopes are the same visibility: every later dispatch on the device sees the
update. Atomic operations are relaxed. An atomic update makes one location
race-safe; it does not create a global barrier or make surrounding non-atomic
data ordered.
Buffer atomics persist like all other buffer state: repeated commands continue
from the previous value. Initialize or clear counters explicitly when a public
operation requires a fresh accumulation.

## 44. Barriers and uniform control

Tach provides:

```tach
workgroupBarrier();
bufferBarrier();
```

`workgroupBarrier()` synchronizes workgroup memory among invocations of one
workgroup. `bufferBarrier()` synchronizes storage-buffer memory among
invocations of one workgroup. Neither synchronizes separate workgroups.

Every invocation in a workgroup must reach a barrier together. A barrier is
invalid under control derived from a coordinate, mutable memory load, atomic
result, or another varying value. This pattern is invalid:

```tach
if (i % 2 == 0) {
  workgroupBarrier();
}
```

Uniform means equal for all invocations in the workgroup. It is a control-flow
property, not a declared source type. Put edge checks after the final required
barrier, or structure early work so all lanes still reach synchronization.

An ordered `run` boundary in an explicit program is the mechanism for
device-wide stage sequencing. Do not attempt to synchronize all workgroups
inside one dispatch with `bufferBarrier()`.

## 45. Parallel correctness and races

Indexed invocations may execute in any order. Workgroups may overlap in time.
Never derive correctness from an assumed scheduling order, invocation order,
or stable mapping to hardware cores.

A stage is straightforwardly race-free when each invocation writes only a
location uniquely determined by its coordinate and reads immutable or
non-overlapping data. Examples include element maps, image pixels, and particle
updates with one particle per invocation.

When multiple invocations can write one location:

- use an atomic if the operation has an atomic formulation;
- use workgroup shared memory and barriers for workgroup-local cooperation; or
- split the operation into ordered dispatches with an intermediate buffer.

Avoid read-modify-write sequences on ordinary storage when another invocation
can touch the same location. A bounds guard prevents out-of-range access but
does not prevent two in-range invocations from racing.

Floating reductions through integer atomics are not automatically available.
Design a multi-stage reduction or a numerically justified representation. Be
explicit about associativity: parallel floating arithmetic can produce valid
rounding differences from a serial CPU order.

## 46. Canonical formatting

Format the entire Tach project with:

```sh
npx tach fmt
```

There is no file-specific formatting command. The formatter discovers the
nearest project and applies one transaction: if any source has a lexical or
syntactic failure, no file is written.

Canonical style is:

- UTF-8;
- LF line endings;
- exactly one final newline;
- two-space indentation;
- semicolon-terminated statements;
- spaces around binary operators;
- one blank line between top-level declarations;
- contiguous imports, one per line, followed by one blank line; and
- lists targeting 100 columns, with one item per line and a trailing comma when
  multiline.

Formatting preserves string contents, `//` comments, import order, and
declaration order. It performs no semantic rewrite. Agents should run it rather
than approximating style by hand. Review the resulting diff because formatting
is project-wide.

## 47. CLI command surface

The npm package exposes this complete public command set:

```text
tach build [--verbose] [--json]
tach check [--json]
tach docs [--json]
tach fmt [--json]
tach instructions [--details <section>...]
tach version
tach help
tach --help
tach -h
```

Project commands operate on the nearest complete Tach project and accept no
source-file argument.

| Command | Use |
|---|---|
| `tach build` | validate and replace `build/` with the complete dual-host package |
| `tach build --verbose` | build the same package plus compiler diagnostics |
| `tach check` | validate the whole project through both hosts, write nothing |
| `tach docs` | validate and refresh only generated Markdown |
| `tach fmt` | transactionally format every kernel |
| `tach instructions` | print the dense AI-agent language and tooling guide |
| `tach instructions --details N [N ...]` | print only the requested numbered detail sections |
| `tach version` | report installed Tach version |
| `tach help`, `--help`, `-h` | show command help |

`--json` is valid on `build`, `check`, `docs`, and `fmt`. It emits one
schema-1 result to stdout and suppresses human prose. Read `ok` first, then
inspect `diagnostics[]`; each item provides `severity`, stable `code`, exact
`span`, `message`, optional source/help, and related locations. Prefer this
surface when operating as an agent. Without `--json`, Tach prints the same
records as Markdown-like source reports. Errors reject the command; warnings
are successful, actionable observations and never alter generated output.

Every command except `instructions`, `version`, and help operates on the nearest
complete Tach project and accepts no source-file argument. `instructions` is
available anywhere: call it without options for the compact first context, then
request one or more referenced sections as space-separated positive integers.
The requested order is preserved and duplicate numbers are emitted once.
Both layers are immutable assets inside the installed `@depths/tach` package;
retrieval performs no project discovery, native compiler execution, or network
request.

Use `npx tach ...` in an npm application to select the installed package
binary. The table above is the complete command surface; project commands
accept neither source paths nor backend selection.

## 48. Validation workflow

For ordinary application changes, use the project-wide order `fmt -> check ->
build`; the quick start gives the exact `npx tach` commands.

`tach check` intentionally checks both WebGPU and Vulkan/SPIR-V
compatibility without writes. It catches project layout, import graph, syntax,
documentation, typing, resource, dispatch, target, generated-binding, and
package-description failures relevant to the public project.

`tach build` always emits both executable artifacts and the singular package
facade. Use `--verbose` only when inspecting compiler-owned IR, executable
plans, runtime/project descriptions, or SPIR-V disassembly. Verbosity never
changes executable semantics.

Run `tach docs` when only `@docs` or manifest documentation changed and compiled
target files should remain untouched. Run a full build before distributing the
package so executable output and documentation are one current set.

In CI, at minimum run `tach check`, then one ordinary build for any published
or deployed package. Type-check the consuming TypeScript code
against generated `index.d.ts`; never duplicate generated signatures in a
handwritten declaration file.

## 49. Build output

Build output always lives at `<project>/build`. An ordinary build contains
exactly:

```text
build/
  package.json
  index.js
  index.d.ts
  kernel.wgsl.gz
  kernel.spv
  README.md
  docs/
    <module>.md
```

There is one `docs/<module>.md` for each discovered module. `index.js` and
`index.d.ts` expose all project types and exported functions through one entry
point. Deterministic gzip `kernel.wgsl.gz` and SPIR-V 1.6 `kernel.spv` contain
corresponding physical stages for WebGPU and Vulkan 1.3. The browser runtime
decompresses WGSL before shader-module creation.

`--verbose` additionally writes `diagnostics/flow.ir`, logical and per-target
Kernel IR, both plan JSON files, private project/runtime JSON descriptions, and
`kernel.spvasm`. These files are observations, never runtime inputs.

The build is replaced atomically as one compiler-owned set. Never edit files in
`build/`, copy only selected generated files over an older build, or expect
custom files placed there to survive. Store application-authored wrappers,
tests, assets, and deployment scripts outside `build/`.

## 50. Consuming the generated package

During local development, import the generated entry directly:

```ts
import { tach } from "@depths/tach";
import { integrate } from "../gpu/build/index.js";
import type { Particle } from "../gpu/build/index.js";
```

Use `import type` whenever a generated name is needed only as a TypeScript type.

For package-style consumption, link, install, pack, or publish the generated
`build/` directory under `tach.json.javascript.package`:

```ts
import { tach } from "@depths/tach";
import { integrate } from "@studio/simulation";
```

The runtime and generated project remain separate imports. The generated
package does not re-export `tach`. It declares an exact `@depths/tach`
dependency because its generated JavaScript must match that runtime version.
The consuming application should also declare and import `@depths/tach`
directly when it uses the public runtime.

Never import `@depths/tach/internal`. That subpath exists only for code emitted
by the matching Tach compiler. Do not load either shader manually or
reconstruct bindings; use generated recipe constructors. The same import runs
through WebGPU in a browser and Tach's Vulkan runtime in Deno.

## 51. Programmatic build tooling for Deno

Deno application build scripts may use `@depths/tach/compiler`:

```ts
import {
  build,
  check,
  compilerPath,
  format,
} from "@depths/tach/compiler";

const cwd = Deno.cwd();
await format({ cwd });
await check({ cwd });
await build({ cwd, verbose: true });
await compilerPath();
```

`cwd` is a starting directory for nearest-project discovery, not a source file.
`build`, `check`, and `docs` return the canonical project root and checked
project description. The same entry exports `docs({ cwd })` for the docs-only
route. `build` accepts only the optional `verbose` flag; there is no target
selection. Options may provide an
environment overlay when build tooling needs controlled process settings.

Do not import this entry into browser code. Use it only in Deno build scripts,
task runners, or CI orchestration. Prefer the CLI when no
programmatic composition is needed; it is the smaller and more portable
application setup.

## 52. Generated TypeScript signatures

Source parameter order becomes generated host parameter order. Buffer
parameters become `ComputeBuffer<HostShape>`. Plain Tach values become their
generated TypeScript representations. A compiler-generated options parameter
comes last.

For indexed shorthand:

```tach
export function scale[i](values: buffer<float32[]>, factor: float32) { ... }
```

the declaration is equivalent in shape to:

```ts
declare function scale(
  values: ComputeBuffer<Float32Array | readonly number[]>,
  factor: number,
  launch?: LaunchOptions<number>,
): ComputeCommand;
```

For explicit orchestration:

```tach
export function transform(
  input: buffer<float32[]>,
  output: buffer<float32[]>,
  count: uint32,
) { ... }
```

the declaration ends in `CommandOptions`, not `LaunchOptions`:

```ts
declare function transform(
  input: ComputeBuffer<Float32Array | readonly number[]>,
  output: ComputeBuffer<Float32Array | readonly number[]>,
  count: number,
  options?: CommandOptions,
): ComputeCommand;
```

Always inspect generated `index.d.ts` rather than guessing the exact host shape
for nested structs, runtime tails, or vectors.

For an exported view program with source signature
`frame(params: FrameParams): view<srgb8>`, the declaration returns
`ComputeView` rather than `ComputeCommand`:

```ts
declare function frame(
  params: FrameParams,
  options?: CommandOptions,
): ComputeView;
```

`ComputeView` is assignable to `ComputeCommand`, so the recipe can be prepared
or submitted; its extra brand is what permits browser presentation.

## 53. Runtime session choices

Import the public runtime from the package root:

```ts
import { tach } from "@depths/tach";
```

Choose one of two ownership styles. `tach(work, options?)` creates a scoped job;
`tach(options?)` creates a persistent session. Both accept the same
`TachOptions` for portable adapter preference.

### Scoped job

```ts
const result = await tach(async (gpu) => {
  const input = gpu.buffer(initial);
  const output = gpu.buffer(new Float32Array(initial.length));
  await gpu.submit(transform(input, output, initial.length));
  return output.read();
});
```

The callback form opens a session, runs the callback, waits for queued work,
returns the callback result, and closes owned resources. Return host data, not
a `ComputeBuffer`; any buffer belongs to the closing session.

Generated recipes are not owned by the session. A scalar-only recipe can be
reused in separate scoped jobs. A recipe referencing a `ComputeBuffer` can run
only in that buffer's owner, which the session checks before GPU work.

### Persistent session

```ts
const gpu = await tach({
  powerPreference: "high-performance",
});

try {
  const state = gpu.buffer(initialState);
  for (let frame = 0; frame < 1_000; frame++) {
    await gpu.submit(step(state, 1 / 60));
  }
  await gpu.idle();
} finally {
  gpu.close();
}
```

Use persistent ownership for frame loops, simulations, solvers, services, or
benchmarks. The caller owns synchronization and closure.

## 54. Runtime public API

The application-facing shape is:

```ts
interface TachOptions {
  readonly powerPreference?: "low-power" | "high-performance";
}

interface TachAdapterInfo {
  readonly backend: "webgpu" | "vulkan";
  readonly name: string;
  readonly vendor?: string;
  readonly architecture?: string;
  readonly type?: "integrated" | "discrete" | "virtual" | "cpu" | "unknown";
}

interface Tach {
  readonly adapter: TachAdapterInfo;
  buffer<T>(value: T): ComputeBuffer<T>;
  prepare(
    first: ComputeCommand,
    ...rest: readonly ComputeCommand[]
  ): Promise<void>;
  submit(
    first: ComputeCommand,
    ...rest: readonly ComputeCommand[]
  ): Promise<void>;
  present(canvas: PresentationCanvas, view: ComputeView): Promise<void>;
  idle(): Promise<void>;
  close(): void;
}

interface ComputeCommand {
  // Opaque generated recipe.
}

interface ComputeView extends ComputeCommand {
  // Opaque recipe with one terminal srgb8 view.
}

interface PresentationCanvas {
  readonly width: number;
  readonly height: number;
  getContext(context: "webgpu"): unknown;
}

interface ComputeBuffer<T> {
  write(value: T): void;
  read(): Promise<T>;
  destroy(): void;
}
```

The browser passes `powerPreference` to WebGPU adapter selection. Deno uses the
same preference when ranking compatible Vulkan devices. It is not a guarantee.
`adapter` deliberately exposes portable observations rather than a raw WebGPU
or Vulkan object.

`ComputeBuffer` intentionally hides its underlying backend allocation. Application
code cannot bind it manually, mix it with another session, or choose a physical
layout. The runtime package root exports `tach`, `TachError`, and their public
types; generated program functions come from the generated project package.

`prepare` validates arguments, materializes and uploads referenced buffers,
allocates required scratch, and compiles backend pipelines, but does not
dispatch. Use it only when eager warmup is useful; `submit` and
`present` prepare automatically. `present` is browser-only, requires canvas
dimensions exactly equal to the view extent, executes the complete recipe,
writes the current WebGPU canvas texture, and waits for that frame. Deno
reports a structured `vulkan-unavailable` error because the current native
host has no presentation surface.

## 55. Buffer creation and materialization

`gpu.buffer(value)` creates a session-owned logical handle and stores a
structured clone of the host value. It does not immediately allocate GPU
memory or select a Tach layout.

The first submitted command that uses the handle:

1. selects the generated resource layout for that parameter;
2. validates and packs the host value;
3. creates and uploads GPU storage;
4. fixes the handle's layout interpretation; and
5. fixes its byte length.

This lazy behavior has useful consequences:

- before first use, `write(value)` may change the future array length;
- after first use, `write(value)` must pack to the same byte length;
- one handle cannot later be interpreted as another Tach buffer layout; and
- create a new handle to resize materialized storage.

Do not mutate the original host value expecting the GPU buffer to observe it.
Call `write` explicitly. For frequently updated resident state, keep one handle
and write same-shaped data only when the CPU actually owns a new value.

## 56. Buffer reads, writes, and destruction

`buffer.write(value)` updates the handle's host/GPU content under its current
lifecycle and layout rules. It is synchronous at the JavaScript call boundary;
the runtime coordinates the actual upload with ordered session work.

`await buffer.read()` is a completion and readback boundary. For materialized
storage it waits for earlier session submissions, copies GPU data to readable
storage, decodes it, and returns a host clone. For a never-submitted handle it
returns a clone of the stored host value without allocating GPU storage.

`buffer.destroy()` is idempotent. After destruction, the handle cannot be read,
written, or submitted. Closing a session destroys its owned resources. Passing
a buffer to another session is a lifecycle error.

Avoid readback in a hot loop unless the CPU truly needs the data. If the next
operation is another Tach command, keep the buffer resident and submit the next
command directly.

## 57. Command construction

Calling a generated function creates an opaque recipe. Ordinary
programs return `ComputeCommand`; view programs return `ComputeView`:

```ts
const command = scale(values, 2);
```

It validates immediately available arguments and records what preparation will
need. It does not submit or execute GPU work. Execute only through:

```ts
await gpu.submit(command);
```

Generated recipe construction checks buffer-handle shape, non-aliasing,
options, and plain values that can be packed immediately.
Buffer arguments remain live handles. Object and array value arguments are
retained until command preparation, so do not mutate them between construction
and submission:

```ts
const params = { dt: 1 / 60, count: 1000 };
const command = step(particles, params);
// Do not mutate params here.
await gpu.submit(command);
```

Accidentally writing `await scale(...)` throws a targeted error rather than
silently pretending the recipe executed. A recipe does not belong to a
session. At `prepare`, `submit`, or `present`, the executing session requires
ownership of every buffer captured by that recipe. A scalar-only recipe has no
such owner constraint and may be reused across sessions.

## 58. Launch and command options

Every generated command supports:

```ts
interface CommandOptions {
  readonly repeat?: number;
}
```

`repeat` is a positive integer in the supported unsigned range and defaults to
`1`. It repeats the complete public program. For a multi-stage program, that
means the entire ordered sequence repeats, not each stage independently.
View recipes reject any repeat other than `1`; present the selected recipe
once per frame.

Exported indexed shorthand also supports:

```ts
type LaunchSize =
  | number
  | readonly [x: number, y: number]
  | readonly [x: number, y: number, z: number];

interface LaunchOptions<Size extends LaunchSize = LaunchSize>
  extends CommandOptions {
  readonly size?: Size;
}
```

Examples:

```ts
scale(values, 2, { size: valuesCount });
shade(pixels, width, height, { size: [width, height] });
step(state, dt, { repeat: 8 });
```

For one safe 1D dispatch, Tach may execute repeat as an invocation-local loop
when independence rules permit it. Application semantics remain complete
program repetition. Do not write code that depends on physical dispatch count.

## 59. Submission ordering and batching

Submit one or more commands:

```ts
await gpu.submit(
  integrate(particles, params),
  resolveConstraints(particles, constraints),
  shade(particles, pixels, { size: [width, height] }),
);
```

Commands are prepared, then recorded in argument order and queued together.
Ordering is preserved: later commands observe the effects of earlier commands
according to their declared resources and plans.

Concurrent JavaScript calls to `submit` on one session are serialized in call
order. Even so, prefer explicit batching when operations are naturally one
unit: it reduces host submission overhead and makes ordering obvious.

`await gpu.submit(...)` waits for command preparation and queue submission. It
does not wait for the GPU to finish. This is intentional and permits the CPU to
continue without a synchronization stall.

A `ComputeView` remains a command, so `submit(view)` executes all source stages
and terminal color projection on both hosts. WebGPU targets a reusable
offscreen texture; Vulkan targets reusable packed scratch. Neither path copies
the frame to CPU memory. Submission is useful for native computation, warmup,
or offscreen work, but it does not draw a browser canvas.

For browser display:

```ts
await gpu.present(canvas, frame(params));
```

`present` accepts exactly one `ComputeView`, checks that canvas and view extents
match, serializes with earlier session work, and waits for the submitted frame.
That final wait is deliberate backpressure: a CPU-driven animation can choose
a different generated view recipe each frame without queuing frames without
bound. The view writes directly to the current WebGPU texture; no
`buffer.read()`, CPU RGBA conversion, `putImageData`, or upload is involved.

Do not pass an empty command list, a promise, a raw GPU command, a recipe whose
buffers belong to another session, or an object imitating `ComputeCommand`.
Only generated recipes are valid.

## 60. Completion boundaries

Use a completion boundary only when the application needs one:

- `await gpu.idle()` waits for all earlier work in the session.
- `await buffer.read()` waits for earlier work and performs readback.
- `await gpu.present(...)` waits for the displayed browser frame.
- returning from `tach(async gpu => ...)` waits before session closure.

`gpu.close()` is synchronous and idempotent teardown. In a persistent session,
call `await gpu.idle()` first when successful completion must be observed:

```ts
try {
  await gpu.submit(command);
  await gpu.idle();
} finally {
  gpu.close();
}
```

Do not add `idle()` after every submit. That serializes CPU and GPU work and
destroys the main benefit of asynchronous queues. Do not read a buffer merely
to prove a command completed when no CPU consumer needs the result.

Device loss invalidates the session. Recover by opening a new session and
recreating application state; a lost device's buffers cannot be transplanted.

## 61. Error handling

Asynchronous runtime and tooling failures use `TachError`:

```ts
import { TachError } from "@depths/tach";

try {
  await runGpuWork();
} catch (error) {
  if (error instanceof TachError) {
    console.error(error.code, error.operation, error.message);
  }
  throw error;
}
```

Public error codes are:

```text
webgpu-unavailable
adapter-unavailable
device-request-failed
device-lost
gpu-validation
gpu-out-of-memory
gpu-internal
vulkan-unavailable
vulkan-profile
native
buffer
kernel
lifecycle
user
compiler-platform
compiler-install
compiler-execution
```

Codes distinguish WebGPU/Vulkan capability and acquisition, native profile and
execution, buffer and command
contracts, lifecycle misuse, scoped callback failures, and compiler delivery.
The original cause is retained. `operation` identifies a relevant operation
when available.

Always use `try/finally` around persistent sessions. Scoped callback exceptions
are normalized as `user` failures after cleanup. GPU errors may surface at
submission or a later synchronization boundary because device work is
asynchronous; preserve the first useful error context rather than replacing it
with a generic message.

## 62. Residency and performance

The first use of a program may upload buffers, create shaders and pipelines,
allocate scratch, build binding state, and grow parameter storage. Warm use
reuses resident resources. Therefore a meaningful hot-path design should:

1. open one persistent session;
2. create long-lived buffers once;
3. submit the complete operation at least once before timing steady state;
4. keep evolving state on the GPU;
5. batch related commands;
6. avoid `read()` and `idle()` until the CPU needs a result; and
7. include realistic data sizes and useful work when benchmarking.

For display loops, return `view<srgb8>` and call `present` instead of reading a
float frame to the CPU, converting it, and uploading it again. Keep only
interaction and frame-recipe selection on the CPU. `present` supplies the
necessary completion-backed backpressure, so do not add a second `idle()` per
frame.

Do not infer GPU throughput from a scoped job that includes adapter creation,
pipeline compilation, upload, synchronization, and readback. Those costs may
matter for a one-shot application, but they answer a different question from
resident execution throughput.

Minimize dispatch count when one stage can naturally compute the result in
locals. Use explicit multi-stage programs where device-wide ordering or large
intermediate state is genuinely required. Distinct `run` statements are
distinct dispatches; there is no promise of automatic inter-stage fusion.

## 63. Data-transfer design

GPU acceleration depends on useful compute relative to transfer and launch
cost. Before writing a kernel, classify each value:

| Lifetime | Recommended representation |
|---|---|
| tiny immutable invocation input | plain Tach parameter |
| large input or evolving state | persistent `ComputeBuffer` |
| program-only intermediate | `transient<T>` |
| final result needed by CPU | buffer followed by one readback |
| final image needed by browser display | `view<srgb8>` followed by `present` |
| final image needed only offscreen/native | `view<srgb8>` followed by `submit` |
| intermediate used only by another kernel | resident buffer, no readback |

Avoid uploading an entire large state every dispatch. Avoid reading it back
between stages. If the CPU must inspect a small statistic, consider computing
a compact GPU result rather than transferring the full dataset.

Plain parameter objects are packed for each command; keep them small and
fixed-footprint. Runtime arrays belong in buffers, not plain values. Reuse
buffer handles across frames and use `write` only for host-originated changes.

## 64. Numerical robustness and portability

Tach deliberately exposes 32-bit GPU values rather than JavaScript's broad
`number` abstraction. Design numerical contracts in those terms. Host integers
must be finite, integral, and in the declared signed or unsigned range.
Coordinates, lengths, and explicit-program shapes are `uint32`; validate
application dimensions before constructing commands rather than allowing a
large JavaScript product to become an invalid host value.

Shape arithmetic is checked by the runtime, but ordinary stage arithmetic is
GPU computation. Keep integer operations in range and guard divisors instead of
depending on overflow or division-by-zero behavior. Multiplication used for a
flattened index deserves particular scrutiny:

```tach
let index = (z * height + y) * width + x;
```

The expression is `uint32`; the application must keep the maximum logical
extent representable. Bounds-checking the final buffer access does not repair an
index that overflowed before the comparison.

`float32` has about seven decimal digits of precision and a smaller range than
the double-precision arithmetic normally performed by JavaScript. `float16`
has roughly three decimal digits, maximum finite magnitude 65504, and far less
headroom for intermediate results. Store or compute in binary16 only after the
algorithm's error and dynamic-range budget justifies it; convert explicitly at
an intentional width boundary rather than relying on widening. Guard
domain-sensitive operations when inputs can be exceptional: `sqrt` and `rsqrt` need a
non-negative domain, logarithms need positive inputs, division needs a nonzero
denominator, and `normalize` needs a policy for zero-length vectors. Tach does
not provide a magic finite-value mode or implicit clamp.

Parallel floating reductions may reorder additions, so compare against a CPU
reference with a justified absolute/relative tolerance rather than exact bit
equality. Tolerance must reflect the algorithm and result scale; it is not a
blanket permission for large error. Integer and bitwise results should normally
be exact. Atomic integer totals are exact when the algorithm itself does not
overflow, while the order in which individual invocations win an exchange may
be nondeterministic.

Write portable algorithms against Tach semantics only. Do not depend on one
browser's workgroup scheduling, one GPU's transcendental approximation, native
subgroup behavior, unexposed shader features, or JavaScript double precision
inside Tach source. `tach check` proves target acceptance, not that an
ill-conditioned numerical method is accurate for the application's data.

## 65. Application correctness testing

Test at three separate boundaries: project validity, generated TypeScript ABI,
and real GPU results. `tach check` is the mandatory static gate, but it does not
replace execution tests. A consuming application should build the complete
package, type-check its actual calls against generated `index.d.ts`, and run
representative commands through every deployed host: WebGPU in browsers and
Vulkan in Deno.

For every indexed operation, include domain sizes around workgroup boundaries:

```text
1
workgroupSize - 1
workgroupSize
workgroupSize + 1
several workgroups plus a partial final group
```

For 2D/3D work, use unequal dimensions and independently partial axes; square
images hide transposed indices and swapped launch tuples. Exercise a logical
count smaller than storage length when the public API permits prefix processing.
Test the smallest valid runtime resource because zero-element runtime resources
are invalid after materialization.

Construct small deterministic inputs with an obvious CPU reference. Compare
integer, bitwise, copy, and layout-sensitive results exactly. Compare floating
results with an explicit tolerance and include zero, sign changes, large/small
magnitudes, and intrinsic-domain boundaries relevant to the algorithm. For
struct buffers, vary every field so a swapped field or lane cannot accidentally
pass. For three-lane vector arrays, exercise more than one element to expose
stride errors.

Multi-stage programs need tests that would fail if dispatch order, shape, or
transient initialization were wrong. Use non-identity factors and nonzero
intermediates; a zero-filled fixture often allows a skipped stage to pass.
Atomic tests should assert invariants and totals, not which invocation won an
intentionally unordered race. Shared-memory tests should include a partial edge
workgroup and ensure inactive lanes contribute the algorithm's identity while
still reaching barriers.

Also test host contracts that the application relies upon: buffer reuse across
sequential commands, same-sized `write`, one final `read`, command batching,
repeat semantics, and persistent-session cleanup. Test expected failures only
when they protect a real public boundary, such as aliasing two buffer parameters
or supplying the wrong launch rank. Do not create vanity tests for syntax the
compiler already rejects unless the application wraps and translates that
diagnostic.

For views, exercise a scalar-only transient frame and a caller-owned pixel
buffer. The former covers terminal fusion and owner-neutral recipe reuse; the
latter covers standalone projection and buffer ownership. Also exercise a
tiny known-color swatch on both paths so the shared pack can be checked as
exact 8-bit display bytes, not only as a non-empty PNG. Use unequal extents,
verify the source has at least `width * height` pixels, compare the displayed
canvas to expected output in a real browser, submit native views without
readback in the hot path, and test extent mismatch plus non-view rejection.

## 66. Measuring Tach applications

Name the measurement before collecting it. Four useful but different numbers
are:

- cold end-to-end latency: session acquisition, compilation, upload, dispatch,
  completion, and optional readback;
- warm resident latency: existing session/buffers/pipelines through completion;
- transfer-inclusive operation latency: required upload or readback plus work;
- application frame or transaction latency: all real CPU and GPU work in scope.

The public runtime does not make `await gpu.submit(...)` a GPU timer: that
promise ends after preparation and queue submission. A wall-clock interval that
must include device completion needs `gpu.idle()` or a required `buffer.read()`
at its end. A durable warm resident measurement can use this shape:

```ts
const gpu = await tach();
try {
  const state = gpu.buffer(initialState);

  await gpu.submit(workload(state));
  await gpu.idle();

  const start = performance.now();
  for (let sample = 0; sample < sampleCount; sample++) {
    await gpu.submit(workload(state));
  }
  await gpu.idle();
  const milliseconds = performance.now() - start;
  console.log({ milliseconds, perOperation: milliseconds / sampleCount });
} finally {
  gpu.close();
}
```

This is resident submit-to-completion wall time; it still includes host command
construction, preparation, and queueing. Do not label it exact device execution
time. `sampleCount` must be a positive value selected before measurement. If the
application has access to a separate standards-compliant GPU
timestamp facility, define its scope precisely rather than mixing it with the
managed runtime's wall clock.

Warm every public program and representative resource layout used in the timed
path. Allocate and initialize buffers before the interval unless transfers are
part of the question. Read exact results once after timing to prove work was not
optimized away or misconfigured. Use workloads large enough that fixed launch
noise does not dominate, report input dimensions, command/repeat counts,
completion boundary, backend/adapter, and whether power state was controlled.

Use `repeat` only when repeated complete-program semantics match the workload.
A safe one-dispatch program may internalize repeat, so repeat and multiple host
commands answer different overhead questions even when numerical output is the
same. Compare like with like: resident GPU baseline versus resident GPU change,
not GPU execution versus a TypeScript loop or cold setup. Report raw samples or
distribution statistics; do not turn a noisy microbenchmark into a pass/fail
claim.

For a displayed browser frame, `await gpu.present(canvas, view)` is already a
completion boundary and measures recipe preparation, dispatch, projection,
submission, and completion. Do not append `idle()`. Compare it with another
GPU-resident presentation path, not a CPU readback path unless transfer cost is
the explicit question. On Vulkan, time `submit(view)` through `idle()` because
native presentation is not part of the current API.

## 67. Package integration and distribution

The Tach source project and consuming npm project may share a repository without
sharing identity. When `package.json` and `tach.json` share a root, the npm
package can expose scripts such as:

```json
{
  "scripts": {
    "gpu:fmt": "tach fmt",
    "gpu:check": "tach check",
    "gpu:build": "tach build"
  }
}
```

Install `@depths/tach` with the package manager instead of copying a version
placeholder into the manifest by hand.

Run commands with a working directory whose nearest `tach.json` is the intended
project. If the manifest lives in `gpu/`, invoke the CLI from that directory or
pass its resolved directory as the Deno API's `cwd` option in an application
build script. Tach's CLI itself has no project-path flag.

For an application-local build, import `gpu/build/index.js` directly and ensure
the Tach build runs before TypeScript type-checking or bundling. For independent
distribution, treat `gpu/build/` as the generated npm package: its
`package.json` carries `javascript.package`, the Tach project version, entry points,
and the matching `@depths/tach` dependency. Pack or publish that directory only after a
fresh successful build. Consumers then install the generated package and
`@depths/tach`, importing runtime and kernels separately.

Do not copy only `index.js`, rewrite its dependency, or replace either shader
behind a previously generated module. Generated JavaScript, declarations,
both shader modules, package metadata, and docs are one versioned unit. Do not run a
package formatter or declaration generator over `build/`; any desired wrapper
belongs in application-authored source outside it.

One build is already complete for browser and Deno deployments. Preserve its
whole inventory when packing, copying, or publishing; never create partial
per-host packages.

## 68. Unified browser and Deno host boundary

`tach build` emits one package containing `index.js`, `index.d.ts`, compressed
WGSL, and SPIR-V 1.6. The facade embeds both executable plans and URLs to the
two sibling shader files. There is no target flag and no separate server
facade.

The managed `@depths/tach` API detects the host. A browser uses WebGPU and WGSL;
Deno uses Tach's packaged Vulkan 1.3 runtime and SPIR-V. Both consume identical
recipes, buffers, host layouts, program ordering, views, and lifecycle rules.
Tach
source and application calls therefore contain no Vulkan descriptors, WebGPU
objects, target conditionals, or provider-specific branches.

Execution is unified; presentation capability is explicit. Browsers can call
`present` because a WebGPU canvas is available. Deno can submit the same view
for offscreen packed output but has no Tach-owned Vulkan surface. This is one
view language and ABI with distinct host capabilities, not two generated
facades or source paths.

The Vulkan host requires an x86-64 Windows or Linux runtime, Vulkan API 1.3,
robust buffer access, Synchronization2, zero-initialized workgroup memory, and
the Vulkan memory model.
Unavailable or incompatible devices fail with structured Tach errors. Do not
load `kernel.spv` yourself: the managed driver owns profile enforcement,
pipelines, descriptors, staging, barriers, submission reuse, and readback.

## 69. Pattern: one-dimensional map

Use exported indexed shorthand when each output element depends only on the
same input index and immutable values:

```tach
@docs(
  title("Affine mapping"),
  summary("Applies a scale and bias to a runtime vector."),
);

@docs(
  summary("Transforms every in-range element."),
  coordinate(i, "Element index."),
  param(input, "Source values."),
  param(output, "Transformed values."),
  param(scale, "Multiplier."),
  param(bias, "Additive bias."),
)
export function affine[i](
  input: buffer<float32[]>,
  output: buffer<float32[]>,
  scale: float32,
  bias: float32,
) {
  if (i < input.length && i < output.length) {
    output[i] = input[i] * scale + bias;
  }
}
```

TypeScript:

```ts
const input = gpu.buffer(new Float32Array(source));
const output = gpu.buffer(new Float32Array(source.length));
await gpu.submit(affine(input, output, 2, 0.5));
const result = await output.read();
```

Omitted 1D size is inferred from the first runtime-sized buffer, `input`.
Pass `{ size: logicalCount }` if only a prefix should execute.

## 70. Pattern: two-dimensional image work

Represent the image extent explicitly, keep row-major mapping visible, and
return a view when the result is for display. A flattened final stage gives
each invocation one complete pixel and enables terminal projection fusion:

```tach
@docs(
  title("Image gradient"),
  summary("Produces a normalized procedural display view."),
);

@docs(
  summary("Describes the image extent."),
  field(width, "Image width in pixels."),
  field(height, "Image height in pixels."),
)
type ImageParams = {
  width: uint32,
  height: uint32,
};

@docs(
  summary("Shades one row-major output pixel."),
  coordinate(i, "Linear pixel index."),
  param(pixels, "Linear RGBA output."),
  param(params, "Image extent."),
)
function shadeGradient[i](pixels: buffer<vec<float32, 4>[]>, params: ImageParams) {
  if (i < pixels.length) {
    let x = i % params.width;
    let y = i / params.width;
    pixels[i] = vec(
      float32(x) / float32(params.width),
      float32(y) / float32(params.height),
      0.25,
      1,
    );
  }
}

@docs(
  summary("Builds a complete display-ready gradient."),
  param(params, "Image extent."),
  returns("The complete sRGB display view."),
)
export function imageGradient(params: ImageParams): view<srgb8> {
  let pixels = transient<vec<float32, 4>>(params.width * params.height);
  run shadeGradient(pixels, params) over pixels.length;
  return view(pixels, params.width, params.height);
}
```

```ts
await gpu.present(canvas, imageGradient({
  width: canvas.width,
  height: canvas.height,
}));
```

The caller allocates no frame buffer and performs no readback. The transient
exists in the source dataflow, but the compiler can remove its physical
full-frame allocation because the final 1D stage writes exactly one complete
pixel at `i`, runs over the transient length, and that length equals
`width * height`. If an algorithm already has caller-owned or multi-use float
pixels, return that buffer instead; the same host API remains correct and a
standalone projection is used.

## 71. Pattern: shared types and helpers across files

Place shared declarations in a lower dependency file.

`data/particles.tach`:

```tach
@docs(
  title("Particle data"),
  summary("Defines state shared by simulation kernels."),
);

@docs(
  summary("Position and velocity for one particle."),
  field(position, "Current position."),
  field(velocity, "Current velocity."),
)
type Particle = {
  position: vec<float32, 4>,
  velocity: vec<float32, 4>,
};

@docs(
  summary("Advances one particle value."),
  param(particle, "State before integration."),
  param(dt, "Elapsed simulation time."),
  returns("State after integration."),
)
function advanceParticle(particle: Particle, dt: float32): Particle {
  return {
    position: particle.position + particle.velocity * dt,
    velocity: particle.velocity,
  };
}
```

`physics/integrate.tach`:

```tach
@docs(
  title("Particle integration"),
  summary("Advances particle state with the shared data model."),
);

import "data/particles";

@docs(
  summary("Advances all in-range particles in place."),
  coordinate(i, "Particle index."),
  param(particles, "Particle state updated in place."),
  param(dt, "Elapsed simulation time."),
)
export function integrateParticles[i](
  particles: buffer<Particle[]>,
  dt: float32,
) {
  if (i < particles.length) {
    particles[i] = advanceParticle(particles[i], dt);
  }
}
```

The helper needs no `export`; direct import makes it visible in Tach. The
generated host API exposes `Particle` and `integrateParticles`, not
`advanceParticle`.

## 72. Pattern: public pipeline importing private stages

Private stages may be organized by operation and imported into one public
pipeline. `passes/scale.tach` owns `scalePass`, `passes/bias.tach` owns
`biasPass`, and `pipelines/transform.tach` composes them:

```tach
import "passes/bias";
import "passes/scale";

export function transformValues(
  input: buffer<float32[]>,
  output: buffer<float32[]>,
  count: uint32,
  factor: float32,
  bias: float32,
) {
  let scratch = transient<float32>(count);
  run scalePass(input, scratch, factor) over count;
  run biasPass(scratch, output, bias) over count;
}
```

Both imported files declare ordinary `function stage[i](...)` stages with
project-global names. They need no `export`, so the generated package exposes
only `transformValues`. The pipeline imports each owner directly; an import by
one pass would not make the other pass transitively visible.

## 73. Pattern: persistent ping-pong state

When one step reads all old state while writing separate new state, keep two
resident handles and swap their host roles between commands. This satisfies
non-aliasing without copying through the CPU:

```ts
async function simulate(initialState: Float32Array, steps: number) {
  const gpu = await tach();
  try {
    let current = gpu.buffer(initialState);
    let next = gpu.buffer(new Float32Array(initialState.length));

    for (let stepIndex = 0; stepIndex < steps; stepIndex++) {
      await gpu.submit(simulateStep(current, next, 1 / 60));
      [current, next] = [next, current];
    }

    return await current.read();
  } finally {
    gpu.close();
  }
}
```

The handles remain distinct within every `simulateStep` command. Swapping
JavaScript variables does not move GPU data. The final read is the only
completion/readback boundary; the scoped result is host data, not a
session-owned handle.

## 74. Pattern: workgroup reduction

A reduction requires exact lane cooperation and usually multiple stages. A
single workgroup partial stage can look like:

```tach
@workgroup(64)
function reduceBlocks[i](
  input: buffer<float32[]>,
  partials: buffer<float32[]>,
) {
  let sharedValues: shared<float32[64]>;
  let lane = i % 64;
  sharedValues[lane] = i < input.length ? input[i] : 0;
  workgroupBarrier();

  let stride: uint32 = 32;
  while (stride > 0) {
    if (lane < stride) {
      sharedValues[lane] += sharedValues[lane + stride];
    }
    workgroupBarrier();
    stride /= 2;
  }

  let block = i / 64;
  if (lane == 0 && block < partials.length) {
    partials[block] = sharedValues[0];
  }
}
```

All lanes reach every barrier. Out-of-range lanes contribute the additive
identity rather than returning early. A complete reduction needs subsequent
dispatches until one result remains; encode those fixed stages in an explicit
program when their domains can be expressed from public shapes. Do not use
`bufferBarrier()` as a global reduction barrier.

## 75. Debugging source failures

Start with the earliest source diagnostic, fix its underlying issue, rerun
`tach fmt`, then rerun `tach check`. Recovery may report later diagnostics
whose context was damaged by the first syntax error.

Classify failures in this order:

1. Project discovery: manifest, source depth, identity, or case collision.
2. Import graph: malformed identity, missing direct import, duplicate edge, or
   kernel/module cycle.
3. Syntax: misplaced imports, missing punctuation, unsupported construct, or
   invalid attribute placement.
4. Names and types: global collision, visibility, shadowing, type mismatch, or
   illegal value/storage type.
5. Stage/program contract: wrong function role, invalid `run`, shape, transient,
   view source/extent/final return, resource alias, workgroup, or barrier
   control.
6. Target portability: a construct has no valid meaning for both targets.
7. Generated package/runtime: host shape, ownership, lifecycle, or WebGPU
   failure.

Do not edit generated WGSL to work around a source failure. Do not add a second
Tach declaration file that duplicates types. Resolve the error in the project
source or host call that owns it.

## 76. Debugging runtime failures

When a command fails, inspect these facts before changing numerical code:

- Was WebGPU available and was an adapter/device acquired?
- Does every `ComputeBuffer` belong to the active session?
- Was any handle destroyed or session closed?
- Is the same handle passed to two buffer parameters?
- Does the host value match generated `index.d.ts` exactly?
- Are integers finite, integral, and in the required 32-bit range?
- Does a write preserve materialized byte length and layout?
- Does launch size have the exact rank and positive components?
- Can a program shape underflow, overflow, divide by zero, or become zero?
- Was an opaque command accidentally awaited instead of submitted?
- For `present`, is this a browser and does the canvas exactly match the view?
- Did a GPU error surface later at `read()`, `idle()`, or `present()`?

Use `TachError.code`, `operation`, message, and cause together. Preserve the
failing public command name and input dimensions in application logs. Avoid
logging entire large buffers unless a small reproducer requires it.

## 77. Debugging wrong numerical output

If execution succeeds but results are wrong, check:

1. Edge guards and flattening formulas.
2. Host launch rank and logical size.
3. `uint32` versus `int32` arithmetic and explicit conversions.
4. Struct field meaning and declaration order.
5. Three-lane vector host representation.
6. Mutation of a retained parameter object before submission.
7. Intra-stage races or an invalid assumption about workgroup order.
8. Missing barrier within a cooperating workgroup.
9. Incorrect identity value for out-of-range reduction lanes.
10. Floating-order differences mistaken for logic errors.
11. A transient read before the stage that fully defines it.
12. A TypeScript buffer initialized with the wrong logical units or count.

Build a tiny deterministic input, run one operation, and read only the minimum
output needed to locate the first incorrect value. Then restore resident,
large-input behavior after correctness is established. Do not benchmark a
debug readback path and call it steady-state GPU time.

## 78. Unsupported assumptions to reject

Do not generate Tach code that assumes any of the following exist:

- isolated-file compilation;
- root-level or deeply nested `.tach` source;
- cross-project, npm, URL, relative, named, aliased, or transitive imports;
- namespaces, overloads, methods, closures, or function values;
- pointers, references, pointer arithmetic, or user binding annotations;
- recursion, labeled transfers, `switch`, exceptions, or block comments;
- strings as runtime values;
- implicit general numeric conversion or JavaScript coercion;
- whole-program, later-use, host-TypeScript, or backend-dependent inference;
- matrices, 64-bit numbers, or floating atomics;
- public resource aliasing;
- global synchronization inside one dispatch;
- arbitrary host control flow inside an orchestration program;
- automatic fusion of distinct `run` stages;
- direct access to runtime-owned `GPUBuffer` objects;
- native Vulkan surface presentation through the current Deno runtime;
- hand editing or partial preservation of generated `build/`; or
- importing application APIs from `@depths/tach/internal`.

When a requested design depends on one of these, reshape the application using
supported helpers, stages, ordered programs, atomics, TypeScript host logic, or
separate Tach projects. Do not invent compatibility syntax.

## 79. Compact syntax grammar

This grammar summarizes syntax; semantic restrictions in earlier sections
still apply:

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
docs-attribute  := "@" "docs" "(" docs-clause
                   {"," docs-clause} [","] ")"
docs-clause     := IDENT "(" [IDENT ","] STRING ")"

type            := (IDENT | "vec" "<" type "," NUMBER ">")
                   ["[" [expression] "]"]

block           := "{" {statement} "}"
statement       := const-decl
                 | let-decl ";"
                 | shared-decl ";"
                 | run-statement ";"
                 | simple-statement ";"
                 | if-statement
                 | while-statement
                 | for-statement
                 | "break" ";"
                 | "continue" ";"
                 | return-statement ";"

let-decl        := "let" IDENT [":" type] "=" expression
shared-decl     := "let" IDENT ":" "shared" "<" type ">"
run-statement   := "run" IDENT arguments "over" domain
domain          := expression
                 | "[" expression {"," expression} "]"

simple-statement := expression
                  | expression assignment-op expression
                  | expression ("++" | "--")
assignment-op   := "=" | "+=" | "-=" | "*=" | "/=" | "%="
                 | "&=" | "|=" | "^=" | "<<=" | ">>="

if-statement    := "if" "(" expression ")" block
                   ["else" (if-statement | block)]
while-statement := "while" "(" expression ")" block
for-statement   := "for" "(" for-init ";" expression ";"
                   for-update ")" block
for-init        := "let" IDENT [":" type] "=" expression
for-update      := expression assignment-op expression
                 | expression ("++" | "--")
return-statement := "return" [expression]

expression      := primary {postfix} {binary-op expression}
                   ["?" expression ":" expression]
primary         := NUMBER | STRING | "true" | "false" | IDENT
                 | "transient" "<" type ">" "(" expression ")"
                 | ("!" | "-" | "~") expression
                 | "(" expression ")"
                 | struct-literal
postfix         := arguments | "." IDENT | "[" expression "]"
arguments       := "(" [expression {"," expression} [","]] ")"
struct-literal  := "{" [literal-field {"," literal-field} [","]] "}"
literal-field   := IDENT ":" expression
```

`STRING` is usable only where documentation or import syntax permits it. The
grammar shows shape, not permission: for example `transient` and `run` are
restricted to public orchestration programs, while `shared` is restricted to
indexed stages. The only generic public result is `view<srgb8>`, and its only
constructor use is the final `return view(pixels, width, height)` of that
exported unindexed program.

## 80. Application-authoring workflow for agents

When asked to add GPU work to an application, follow this sequence:

1. Locate the nearest intended `tach.json` and inspect its generated JavaScript package
   name.
2. Inventory existing modules, kernel identities, global declaration names,
   constants, types, helpers, stages, and exported programs.
3. Decide whether the operation is one dispatch, a true multi-stage program,
   or a display view over either form.
4. Reuse directly visible types/helpers or add the exact direct imports needed.
5. Keep dependency edges acyclic at both file and module level.
6. Define host data representation before choosing Tach buffer types.
7. Separate compiler-known constants, runtime values, persistent caller state,
   and program-local transient scratch.
8. Write bounds guards for rounded domains.
9. Prove each write is race-free or explicitly synchronized/atomic.
10. Add structured `@docs` for file, types, and functions; use `//` only for
    local implementation reasoning.
11. Run project-wide formatting.
12. Run targetless checking.
13. Build the complete dual-host package.
14. Inspect generated `index.d.ts` and update the TypeScript caller to match it.
15. Exercise the real runtime path with representative data and explicit
    completion only where output is consumed; present views directly rather
    than reading frames back.

Do not start by editing generated JavaScript, TypeScript declarations, WGSL,
SPIR-V, or Markdown.

## 81. Design checklist before writing a kernel

Answer these questions explicitly in reasoning:

- What is one logical invocation responsible for?
- What coordinate rank matches the data domain?
- Which accesses can occur beyond the logical edge after workgroup rounding?
- Does each invocation own its output, or can writes collide?
- Which inputs are small immutable values versus large resident buffers?
- Which values are truly compiler-known and should be `const`, rather than
  runtime parameters or `let` bindings?
- Is output updated in place, and if so can the API use one buffer parameter?
- Does a helper improve reuse without touching memory?
- Is device-wide sequencing necessary, requiring multiple stages?
- Which intermediates must persist to the caller, and which are transient?
- Is the result a display frame, and can its final writer satisfy one complete
  `vec<float32, 4>` pixel per linear index?
- Can every `run` shape be expressed with checked public `uint32` data?
- Does synchronization stay within a workgroup, or is another dispatch needed?
- What exact TypeScript host shape will initialize each public buffer?
- When does the CPU genuinely need completion or readback?

If these answers are unclear, the kernel API is not yet well formed.

## 82. Source review checklist

Before considering Tach source complete, verify:

- Every file is exactly `<module>/<kernel>.tach`.
- Imports are extensionless, direct, contiguous, and project-local.
- Both dependency views remain acyclic.
- Every top-level name is project-global unique.
- Every imported constant has a direct import, every local constant depends
  only on visible constants, and constant dependencies are acyclic.
- Public names are portable TypeScript identifiers.
- `export` appears only on intended host endpoints.
- Helpers use only constructible values and do not recurse.
- Indexed stages have one to three coordinates, buffers, no value result, and
  complete bounds reasoning.
- Explicit programs contain compile-time `const`, shape/transient `let`, `run`,
  and an optional required final view return.
- Every view source is runtime `vec<float32, 4>`, fully defined, and large enough for
  its positive width and height.
- Every inferred numeric expression has an obvious local type; ambiguous
  domain changes use an annotation or explicit constructor.
- Every shape is checked-`uint32` safe and nonzero for valid calls.
- Runtime arrays appear only in supported storage positions.
- Fixed arrays appear only in shared memory and use a positive compile-time
  `uint32` length.
- Atomic places use only atomic operations.
- Shared/barrier stages state explicit workgroups.
- Every barrier is under uniform control and reached by every lane.
- No stage relies on cross-workgroup synchronization or scheduling order.
- Every `@docs` reference resolves and describes meaningful semantics.
- Inline comments use `//` only.
- `tach fmt` and `tach check` succeed.

## 83. TypeScript integration checklist

Before considering the host integration complete, verify:

- `@depths/tach` is a direct application dependency.
- The runtime comes from `@depths/tach`, while programs come from the generated
  project entry.
- No application import references `@depths/tach/internal`.
- Generated `index.d.ts`, not a copied signature, governs host calls.
- Every buffer is created by the session that submits its commands.
- Distinct buffer parameters receive distinct handles.
- Host objects and tuples match generated readonly shapes exactly.
- Three-lane runtime vector arrays use tuple arrays.
- Integer values are finite, integral, and within the required range.
- Materialized buffer writes preserve layout and byte length.
- Command parameter objects are not mutated before submission.
- Generated calls are passed to `gpu.submit`, not awaited directly.
- Display recipes return `ComputeView` and use browser `present` without frame
  readback; native offscreen views use `submit`.
- Every presented canvas exactly matches the view extent.
- Launch size has exact rank and positive components.
- Persistent sessions close in `finally`.
- `read()` or `idle()` appears only at a real completion requirement.
- Device loss leads to session and state recreation.

## 84. Performance review checklist

For a hot path, verify:

- The session, pipelines, and buffers are warmed before steady-state timing.
- Large evolving state remains GPU-resident.
- Upload and readback are excluded from pure execution timing unless the
  measured application path genuinely requires them.
- Related commands are batched in one submit where practical.
- There is no `idle()` between commands without a CPU dependency.
- One indexed stage performs locally composable work without unnecessary
  transient round trips.
- Multi-stage dispatches represent real device-wide dependencies.
- Workgroup shape is default unless shared-memory geometry or measurement
  justifies an explicit shape.
- Input size and work density are representative enough to dominate fixed
  overhead.
- Correctness validation reads exact results outside the timed hot path.
- Display paths use `present` directly and do not add redundant per-frame
  readback or `idle()`.
- Comparisons distinguish cold startup, transfer-inclusive latency, resident
  GPU execution, and end-to-end application latency.

Never claim acceleration from TypeScript wall time that mixes unrelated setup,
or from workloads too small to measure reliably.

## 85. Final compact mental model

Remember the complete system in seven statements:

1. A Tach project is one strict manifest plus one-tier module directories;
   project commands operate on that complete project.
2. Imports expose whole files directly, names are project-global, and both file
   and module dependencies are DAGs.
3. `const` is compiler-evaluated scalar/vector algebra with no host surface;
   `let` is the sole mutable runtime local.
4. Helpers compute values, indexed stages compute one coordinate, exported
   indexed functions are one-dispatch host sugar, and exported unindexed
   functions orchestrate ordered stages and may return a terminal display
   view.
5. Buffers hold persistent GPU storage, transients hold program-local scratch,
   and parallel writes require ownership, atomics, workgroup synchronization,
   or another dispatch.
6. `tach build` generates one browser/Deno package; `@depths/tach` opens
   sessions, creates buffers, executes generated recipes, presents browser
   views, and owns WebGPU or Vulkan details.
7. GPU work is asynchronous and resident by default: submission is not
   completion, generated calls are not execution, and readback is never free.

When uncertain, prefer explicit types, direct imports, one clear ownership path,
one bounds guard, one public operation, generated declarations, and the
project-wide `fmt -> check -> build` workflow.
