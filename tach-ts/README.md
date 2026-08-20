# `@depths/tach`

Tach is an ergonomic, typed GPU programming language for TypeScript developers.
Write the parallel work in small `.tach` files, build once, and call the
generated functions from ordinary TypeScript.

You do not write WGSL, SPIR-V, bind groups, descriptor sets, staging buffers, or
pipeline layouts. Tach compiles the same program for browser WebGPU and native
Vulkan 1.3, generates the TypeScript types, and provides one managed runtime for
both.

```tach
export function scale[i](values: buffer<float32[]>, factor: float32) {
  if (i < values.length) {
    values[i] *= factor;
  }
}
```

```ts
import { tach } from "@depths/tach";
import { scale } from "./build/index.js";

const answer = await tach(async (gpu) => {
  const values = gpu.buffer(new Float32Array([1, 2, 3, 4]));
  await gpu.submit(scale(values, 10));
  return values.read();
});

console.log([...answer]); // [10, 20, 30, 40]
```

The generated `scale(...)` call is a typed recipe. `gpu.submit(...)` executes
it. The input remains on the GPU until the explicit `read()`.

This README is the complete application guide. It starts with no assumed GPU
knowledge, then progressively covers projects, the language, display views,
multi-stage computation, runtime lifetime, portability, correctness, and
performance.

## 1. What Tach is for

Tach targets general-purpose GPU work in TypeScript applications: numerical
simulation, physics, image generation, procedural rendering, mesh computation,
signal processing, matrix work, spatial transforms, and other large parallel
calculations. It is designed around consumer integrated and discrete GPUs, not
only specialized accelerator servers.

Tach sits between a high-level tensor library and a low-level GPU API:

- unlike a tensor framework, you write the per-element logic yourself,
  including irregular access and explicit stages;
- unlike WebGPU or Vulkan, you do not own shaders, bind groups, pipelines,
  or a second native backend; and
- unlike a shader string preprocessor, Tach is its own checked project:
  modules, types, programs, docs, and one generated package.

Tach is not a graphics-pipeline API, an AI framework, a CPU fallback system, or
a promise that every small loop belongs on a GPU. It is the compact path from a
typed parallel recipe to portable GPU execution and optional display output.

## 2. Install

Deno must be installed and available on `PATH`; the published `tach` command is
a Deno executable. Browser users still need Deno for project tooling, but not
inside the browser application.

```sh
npm install @depths/tach
```

The package contains:

- the `tach` project command;
- the browser WebGPU and Deno Vulkan runtime;
- the native Tach compiler resolver;
- public TypeScript declarations; and
- built-in instructions for AI coding agents.

Use the local command through `npx`:

```sh
npx tach version
npx tach help
```

Tach is Deno-first. Browser applications can use any ESM development server or
bundler that preserves the generated package and its sibling shader asset.
Native execution uses Deno because Tach's Vulkan host is loaded through Deno
FFI.

## 3. Your first Tach project

A Tach project is a directory containing one `tach.json` and one or more module
directories. Its identity and build rules do not come from npm. A Tach project
may share a directory with the TypeScript application's `package.json` and
`node_modules`; the two project models remain independent.

Create this layout:

```text
my-kernels/
  tach.json
  math/
    scale.tach
```

### 3.1 Describe the project

Save this as `tach.json`:

```json
{
  "name": "my-kernels",
  "version": "0.1.0",
  "javascript": {
    "package": "@example/my-kernels"
  },
  "docs": {
    "title": "My kernels",
    "summary": "GPU calculations used by my application."
  }
}
```

The fields have deliberately narrow roles:

| Field | Meaning |
|---|---|
| `name` | Tach project identity |
| `version` | SemVer shared by all generated outputs |
| `javascript.package` | npm identity of the generated package |
| `docs.title` | generated README title |
| `docs.summary` | generated README introduction |

The compiler discovers source files. Do not list modules, files, output paths,
targets, or compiler settings in the manifest.

### 3.2 Write one kernel

Save this as `math/scale.tach`:

```tach
export function scale[i](values: buffer<float32[]>, factor: float32) {
  if (i < values.length) {
    values[i] *= factor;
  }
}
```

Read it as:

- `export`: expose this function to JavaScript and TypeScript;
- `[i]`: run many invocations, each with its own unsigned integer coordinate;
- `buffer<float32[]>`: GPU-resident, runtime-sized floating-point storage;
- `factor`: one ordinary value shared by all invocations; and
- the bounds check: ignore extra invocations created by workgroup rounding.

Each valid invocation owns one output element, so no two invocations race.

### 3.3 Check, format, and build

Run these commands anywhere below `my-kernels/`:

```sh
npx tach fmt
npx tach check
npx tach build
```

Tach finds the nearest ancestor `tach.json`. Every command operates on the
whole project, never on one source file.

`tach build` atomically replaces `build/`:

```text
build/
  package.json
  index.js
  index.d.ts
  kernel.wgsl.gz
  kernel.spv
  README.md
  docs/
    math.md
```

This is one generated ESM package:

- `index.js` contains every exported recipe constructor and execution plan;
- `index.d.ts` contains every generated type and function signature;
- `kernel.wgsl.gz` contains the compressed browser program;
- `kernel.spv` contains the SPIR-V 1.6 native program;
- `README.md` and `docs/*.md` are generated documentation; and
- `package.json` declares the generated package and its `@depths/tach`
  dependency.

Treat the entire directory as compiler-owned. Never edit one generated file or
mix files from different builds.

### 3.4 Call it from TypeScript

Place `build/` where the application can import and serve it as one directory.
Then:

```ts
import { tach } from "@depths/tach";
import { scale } from "./build/index.js";

const gpu = await tach();
const values = gpu.buffer(new Float32Array([1, 2, 3, 4]));

try {
  await gpu.submit(scale(values, 3));
  console.log([...await values.read()]);
} finally {
  gpu.close();
}
```

This same TypeScript source selects WebGPU in a browser and Vulkan in Deno.
Application code does not choose a backend.

On a supported x86-64 Windows or Linux Vulkan host, save the example as
`app.ts` beside `tach.json`, then execute it natively:

```sh
deno run --allow-ffi --allow-read app.ts
```

For a browser, import the same module from application code served by the
browser's normal ESM development server or bundle.

## 4. The GPU model, without shader jargon

A CPU normally follows one control flow through one item after another. A GPU
is useful when a large collection of items can receive similar work at once.

In the `scale` kernel:

```text
invocation 0 -> values[0]
invocation 1 -> values[1]
invocation 2 -> values[2]
...
```

An **invocation** is one run of an indexed Tach function. Its coordinate
says which element, pixel, particle, or cell that run owns.

A **dispatch** launches those runs. Hardware groups them into
**workgroups** of up to 256. Workgroups matter when neighbors share a
small scratchpad; until then, accept Tach's defaults.

A **buffer** is typed storage that can stay on the GPU across many dispatches.
Moving data between TypeScript and the GPU costs time, so useful applications
usually:

1. create large buffers once;
2. submit many recipes that reuse them;
3. keep intermediate results on the GPU; and
4. read only when the CPU genuinely needs an answer.

A **recipe** is the object returned by a generated function. Constructing a
recipe performs no GPU work:

```ts
const command = scale(values, 2); // describe work
await gpu.submit(command);        // queue work
await values.read();              // wait, transfer, and decode
```

Do not write `await scale(...)`. Tach deliberately reports that mistake.

### Submission is not completion

`await gpu.submit(command)` means resources were prepared and work was queued.
It does not wait for an idle GPU. Waiting after every dispatch would throw away
useful overlap.

Completion is explicit:

- `await buffer.read()` waits, transfers, and decodes that result;
- `await gpu.idle()` waits for all earlier work;
- leaving `tach(async gpu => ...)` waits before closing; and
- `await gpu.present(canvas, view)` waits for the presented browser frame.

## 5. The smallest useful language

Most first kernels need only coordinates, buffers, scalar parameters, an edge
guard, arithmetic, and assignment.

### 5.1 One-dimensional work

```tach
export function add[i](
  left: buffer<float32[]>,
  right: buffer<float32[]>,
  output: buffer<float32[]>,
) {
  if (i < output.length && i < left.length && i < right.length) {
    output[i] = left[i] + right[i];
  }
}
```

Call it with three distinct buffers:

```ts
const left = gpu.buffer(new Float32Array([1, 2, 3]));
const right = gpu.buffer(new Float32Array([10, 20, 30]));
const output = gpu.buffer(new Float32Array(3));

await gpu.submit(add(left, right, output));
console.log([...await output.read()]); // [11, 22, 33]
```

For a 1D exported indexed function, Tach can infer the launch length from the
first runtime-sized buffer. An explicit launch is also possible:

```ts
await gpu.submit(add(left, right, output, { size: outputLength }));
```

### 5.2 Two-dimensional work

Coordinates can have one, two, or three axes:

```tach
export function brighten[x, y](
  pixels: buffer<vec<float32, 4>[]>,
  width: uint32,
  height: uint32,
  amount: float32,
) {
  if (x < width && y < height) {
    const i = y * width + x;
    if (i < pixels.length) {
      pixels[i] = pixels[i] + vec(amount, amount, amount, 0.0);
    }
  }
}
```

```ts
await gpu.submit(
  brighten(pixels, width, height, 0.1, { size: [width, height] }),
);
```

Tach stores this image as a row-major flat array because buffers are linear
memory. The two coordinates make ownership easier to understand.

### 5.3 Edge guards are normal

GPU dispatches round each axis up to whole workgroups. If the requested width is
1,001 and the workgroup width is 16, the physical launch includes extra
coordinates. Guard before every access that might fall outside the logical
domain.

Default workgroup sizes are:

| Coordinate rank | Default |
|---|---|
| `[i]` | `256 x 1 x 1` |
| `[x, y]` | `16 x 16 x 1` |
| `[x, y, z]` | `8 x 8 x 4` |

Use `@workgroup(...)` only when the algorithm needs exact geometry, shared
memory, or barriers. A workgroup may contain at most 256 invocations.

## 6. Functions: four roles, one consistent rule

Two independent questions classify every function. Indexed means it runs
once per GPU coordinate. Exported means TypeScript can call it.

| Indexed? | Exported? | Role |
|---|---|---|
| no | no | ordinary helper, like a private `function` in TypeScript |
| yes | no | private GPU stage, called only from a Tach program |
| yes | yes | the common case: one stage, one launch, one generated function |
| no | yes | a program that runs several stages, or returns a display view |

`export` has one meaning: generate a JavaScript/TypeScript recipe constructor.
It does not control visibility inside Tach. Private helpers and stages are
visible to files that import them.

### 6.1 Pure helpers

A private unindexed function computes one value and can be called by indexed
stages:

```tach
type Particle = {
  position: vec<float32, 4>,
  velocity: vec<float32, 4>,
};

function advance(particle: Particle, dt: float32): Particle {
  return {
    position: particle.position + particle.velocity * dt,
    velocity: particle.velocity,
  };
}

export function integrate[i](
  particles: buffer<Particle[]>,
  dt: float32,
) {
  if (i < particles.length) {
    particles[i] = advance(particles[i], dt);
  }
}
```

Helpers operate on constructible values. They do not accept buffers, access
shared memory, use barriers, call indexed functions, or recurse.

### 6.2 Private stages

A private indexed function describes one dispatch stage. It is available to
programs in the same file and to files that import its owner, but it is not
generated in `index.js`.

### 6.3 Public indexed shorthand

An exported indexed function is the shortest route from one kernel to one host
recipe. It accepts `LaunchOptions` on the TypeScript side.

### 6.4 Public programs

An exported unindexed function assembles ordered `run` statements into one host
recipe. It accepts `CommandOptions`, not a host launch size, because its domains
are defined in Tach.

Public programs are host entry points. Tach functions cannot call them.
Their bodies contain only immutable shape declarations, transient declarations,
ordered `run` statements, and the required final view return when applicable.
Ordinary branching and loops belong inside indexed stages or value helpers.

## 7. Multi-stage programs and private scratch

Real algorithms often need globally ordered steps. Write private indexed stages
and expose one unindexed program:

```tach
function doubleValues[i](
  input: buffer<float32[]>,
  scratch: buffer<float32[]>,
) {
  if (i < input.length && i < scratch.length) {
    scratch[i] = input[i] * 2.0;
  }
}

function addOne[i](
  scratch: buffer<float32[]>,
  output: buffer<float32[]>,
) {
  if (i < scratch.length && i < output.length) {
    output[i] = scratch[i] + 1.0;
  }
}

export function transform(
  input: buffer<float32[]>,
  output: buffer<float32[]>,
  count: uint32,
) {
  const scratch = transient<float32>(count);
  run doubleValues(input, scratch) over count;
  run addOne(scratch, output) over count;
}
```

```ts
await gpu.submit(transform(input, output, count));
```

`transient<T>(length)` creates program-private GPU storage. It is:

- allocated and reused by the runtime;
- unavailable to TypeScript;
- not zero-initialized; and
- valid only after an earlier stage has defined the elements being read.

Every `run` is a distinct ordered dispatch. The boundary supplies device-wide
sequencing. A workgroup barrier cannot replace it because a workgroup barrier
synchronizes only invocations in one workgroup.

Program shape expressions use checked `uint32` arithmetic. Zero extents,
underflow, overflow, division by zero, and invalid resource lengths are runtime
errors instead of silently malformed launches.

A shape may use `uint32` literals, public `uint32` parameters or nested fields,
runtime-array `.length`, an earlier shape `const`, arithmetic, and `min`, `max`,
or `ceilDiv`. A 2D or 3D stage uses `over [x, y]` or `over [x, y, z]`.
Stage buffer arguments directly name a public buffer or transient, keeping
resource identity explicit.

## 8. Display a GPU result without reading pixels through the CPU

If you computed an image in a buffer and called `read()` every frame, the
whole frame would copy into JavaScript and then back onto the GPU to draw.
Tach instead lets a program return a picture. You write linear
floating-point RGBA. Tach converts that to 8-bit sRGB. In the browser,
`present` draws it on a canvas. The pixels never visit the CPU.

A view is not a `<canvas>` and not a DOM node. `view<srgb8>` is the
finished image as a result type:

```tach
type Frame = {
  width: uint32,
  height: uint32,
  time: float32,
};

function shade[i](pixels: buffer<vec<float32, 4>[]>, frame: Frame) {
  if (i < pixels.length) {
    const x = i % frame.width;
    const y = i / frame.width;
    const u = float32(x) / float32(frame.width);
    const v = float32(y) / float32(frame.height);
    const pulse = 0.5 + 0.5 * sin(frame.time);
    pixels[i] = vec(u * pulse, v, 0.25, 1.0);
  }
}

export function render(frame: Frame): view<srgb8> {
  const pixels = transient<vec<float32, 4>>(frame.width * frame.height);
  run shade(pixels, frame) over pixels.length;
  return view(pixels, frame.width, frame.height);
}
```

This result type is valid only on an exported unindexed program. Its final
statement is exactly `return view(pixels, width, height);`, where `pixels` names
the final defined version of a linear `vec<float32, 4>[]` buffer. The buffer may be a
program-private transient, as above, or a public caller-owned buffer.

The source stays backend-neutral:

- RGB values are linear floating point;
- alpha is floating point;
- the compiler owns IEC sRGB conversion, clamping, and RGBA8 projection;
- no browser canvas, WebGPU texture, Vulkan image, or byte packing appears in
  Tach source.

Width and height must be positive, their checked product must fit, and an
unfused source must contain at least `width * height` complete pixels. For each
RGB channel, lowering clamps the linear input to `[0, 1]` and applies:

```text
channel <= 0.0031308 ? 12.92 * channel
                     : 1.055 * pow(channel, 1 / 2.4) - 0.055
```

Alpha is clamped without that transfer. Channels are then rounded to one
packed RGBA8 `uint32` word with `uint32(channel * 255 + 0.5)`. Both backends
share that word. WebGPU stores it as `rgba8unorm` texels so `present` can
write a canvas texture; Vulkan stores the word in packed scratch.

When the last stage already writes each pixel once, and nothing else needs
the float buffer, Tach can convert in that same stage. Otherwise it adds a
separate conversion step. TypeScript does not change. A buffer you still
own naturally takes the separate step, so you can keep using those linear
pixels.

### 8.1 Present in a browser

```ts
import { tach } from "@depths/tach";
import { render } from "./build/index.js";

const canvas = document.querySelector("canvas");
if (!(canvas instanceof HTMLCanvasElement)) throw new Error("Missing canvas");

canvas.width = 1920;
canvas.height = 1080;

const gpu = await tach({ powerPreference: "high-performance" });
let running = true;

async function frames(): Promise<void> {
  const started = performance.now();
  while (running) {
    await gpu.present(canvas, render({
      width: canvas.width,
      height: canvas.height,
      time: (performance.now() - started) / 1000,
    }));
    await new Promise<void>((resolve) => requestAnimationFrame(() => resolve()));
  }
}

try {
  await frames();
} finally {
  gpu.close();
}
```

`present(canvas, view)`:

- requires the canvas and view extents to match;
- executes the whole recipe;
- projects directly into the current WebGPU canvas texture;
- avoids GPU-to-CPU frame readback and re-upload; and
- resolves after completion, providing natural frame backpressure.

Only one presentation may be active on a session at a time. Await it before
starting the next frame.

### 8.2 Execute a view offscreen

`ComputeView` extends `ComputeCommand`, so both hosts accept:

```ts
await gpu.submit(render({ width, height, time }));
await gpu.idle();
```

Both hosts compute the picture and leave it on the GPU. Neither copies
pixels to JavaScript. Deno has no window, so `gpu.present(...)` rejects
there. Building the recipe and submitting it stay identical.

Views cannot use `repeat`; one view recipe defines one frame.

## 9. Projects, modules, imports, and names

Every source path is exactly:

```text
<module>/<kernel>.tach
```

One directory tier is required. Root-level files and deeper nesting are
rejected. For example:

```text
simulation/
  types.tach
  integrate.tach
rendering/
  procedural.tach
```

Import a complete Tach file:

```text
import "simulation/types";
```

The `.tach` extension is implicit. Imports do not use relative dots, npm names,
URLs, aliases, named bindings, wildcards, or re-exports.

A source file has one canonical order:

1. optional file-level `@docs(...)`;
2. one contiguous block of imports; and
3. type and function declarations.

Type, function, parameter, field, and coordinate names use portable
identifiers. Module and kernel identities are import strings, so their folder
and source filenames may contain dashes.
`tach fmt` supplies the canonical whitespace, punctuation, and line breaking.
The language accepts `//` comments only; block comments do not exist.

Visibility is explicit and non-transitive:

- a function can use declarations in its own file;
- it can use declarations owned by files it directly imports;
- importing a file does not also expose that file's imports; and
- `export` does not affect Tach-side visibility.

All top-level type and function names share one project-global namespace.
Duplicate names are errors even when they live in unrelated modules. This keeps
the singular generated TypeScript namespace obvious.

Both file dependencies and collapsed module dependencies must be DAGs. Tach
rejects file cycles and opposite cross-module edges. Independent subtrees can
therefore be checked and compiled concurrently before being merged into one
package.

## 10. Types and their TypeScript shapes

### 10.1 Scalars and vectors

| Tach | Meaning | TypeScript value |
|---|---|---|
| `bool` | Boolean | `boolean` |
| `int32` | signed 32-bit integer | `number` |
| `uint32` | unsigned 32-bit integer | `number` |
| `float16` | IEEE 754 binary16 | `number` |
| `float32` | 32-bit floating point | `number` |
| `void` | absence of a helper result | no host value |
| `vec<int32, N>` | signed vector, `N` = 2, 3, or 4 | readonly numeric tuple |
| `vec<uint32, N>` | unsigned vector, `N` = 2, 3, or 4 | readonly numeric tuple |
| `vec<float16, N>` | binary16 vector, `N` = 2, 3, or 4 | readonly numeric tuple |
| `vec<float32, N>` | floating vector, `N` = 2, 3, or 4 | readonly numeric tuple |

`vec<T, N>` is the only vector type spelling. `vec(...)` is the only vector
value constructor and flattens exactly two, three, or four lanes:

```text
vec(1.0, 1.0, 1.0, 1.0)
vec(1.0, 2.0, 3.0, 4.0)
vec(vec(1.0, 2.0), vec(3.0, 4.0))
```

Use `vec(...)` when the surrounding expression or a typed argument should
determine the vector type:

```tach
function direction(): vec<float16, 3> {
  return vec(1, 2, 3);
}

function color(rgb: vec<float32, 3>): vec<float32, 4> {
  return vec(rgb, 1);
}
```

If nothing constrains its element type, whole lanes default to `uint32` and a
fraction or exponent makes the vector `float32`. `vec` never converts an
already typed value and has no single-scalar splat form. Repeat a scalar, use a
documented broadcast operation, or explicitly convert each lane before
rebuilding the vector.

### 10.2 Structs

Named structs become generated readonly TypeScript object types:

```tach
type Particle = {
  position: vec<float32, 4>,
  velocity: vec<float32, 4>,
  mass: float32,
};

export function accelerate[i](
  particles: buffer<Particle[]>,
  force: vec<float32, 4>,
) {
  if (i < particles.length) {
    const particle = particles[i];
    particles[i] = {
      position: particle.position,
      velocity: particle.velocity + force / particle.mass,
      mass: particle.mass,
    };
  }
}
```

The generated API accepts:

```ts
type Particle = {
  readonly position: readonly [number, number, number, number];
  readonly velocity: readonly [number, number, number, number];
  readonly mass: number;
};
```

Tach computes storage offsets, alignment, padding, strides, and runtime tails.
Do not put manual padding fields in source or host objects.

### 10.3 Arrays

- `T[]` is a runtime-sized storage array. It appears directly in a buffer or as
  the final field of a buffered struct.
- `T[N]` is a fixed array used for workgroup-shared memory.
- Runtime arrays do not move around as ordinary values.

Scalar arrays use matching typed arrays naturally:

| Tach buffer element | Convenient host storage |
|---|---|
| `float16[]` | `Float16Array` |
| `float32[]` | `Float32Array` |
| `int32[]` | `Int32Array` |
| `uint32[]` | `Uint32Array` |
| `vec<float16, 2>[]`, `vec<float16, 4>[]` | flat `Float16Array` or tuple array |
| `vec<float32, 2>[]`, `vec<float32, 4>[]` | flat `Float32Array` or tuple array |
| three-lane vector array | tuple array |
| struct array | readonly object array |

Three-lane storage has a padded stride. Tach therefore requires tuple arrays
instead of pretending a tightly packed typed array has the same layout.
Boolean values are available to helpers and parameter blocks, but are not
ordinary storage-buffer elements.

An odd-length direct `float16[]` remains logically exact even though WebGPU
requires four-byte physical buffer and copy alignment. Tach handles that
padding privately; `.length` never includes a phantom element.
The same guarantee applies to a final scalar `float16[]` after a struct prefix.

### 10.4 Numbers and conversion

Tach has no shader-style numeric suffixes. Inference is confined to one
expression and resolves explicit types, expected context, concrete sibling
operands, intrinsic domains, then defaults. Therefore operand order does not
change an inferred type:

- unconstrained nonnegative whole literals infer `uint32`;
- negative whole literals infer `int32`;
- fractions and exponents infer `float32`; and
- explicit `int32(...)`, `uint32(...)`, `float16(...)`, and `float32(...)`
  conversions resolve ambiguity.

An all-literal floating intrinsic defaults to `float32`; `abs(1)` defaults to
`int32`. Inference never consults later uses, other functions, generated
TypeScript, or backend behavior.

Unconstrained fractions infer `float32`; binary16 is chosen by a `float16`
annotation, parameter/result context, or explicit constructor. Float16
literals must be finite and within `-65504` to `65504`. Tach does not silently
widen Float16 buffers or arithmetic to float32.

Do not assume JavaScript coercion. Values are packed against the generated
signature; integer ranges, exact fields, vector widths, and array shapes are
validated.

## 11. Variables, control flow, and math

`const` declares an immutable local. `let` declares a mutable local. Shadowing
is rejected so every name has one obvious meaning.

Indexed functions support:

- arithmetic, numeric comparison, logical, bitwise, shift, conditional, and
  assignment operators;
- field, lane, and array access;
- `if`/`else`;
- `while`;
- C-style `for`;
- nearest-loop `break` and `continue`;
- `return`; and
- typed struct and vector construction.

Portable scalar/vector intrinsics include:

| Family | Operations |
|---|---|
| integer bounds | `min`, `max`, `clamp` |
| magnitude | `abs`, `sqrt`, `rsqrt` |
| rounding | `floor`, `ceil`, `trunc` |
| trigonometry | `sin`, `cos`, `tan` |
| exponential | `exp`, `exp2`, `log`, `log2`, `pow` |
| multiply-add | `fma` |
| vector geometry | `dot`, `cross`, `length`, `distance`, `normalize` |

`floor`, `ceil`, `trunc`, trigonometric, exponential, `sqrt`, and `rsqrt`
preserve a `float16` or `float32` scalar/vector type. `abs` accepts signed
integer or floating scalar/vector values. `pow` accepts matching floating
values and can broadcast a scalar exponent across a vector base.

`fma(a, b, c)` accepts `float16` or `float32` values and computes `a * b + c`,
component by component. Equal-width vector arguments may mix with scalars,
which broadcast to that width. It carries explicit multiply-add intent into
both generated targets. The adapter may use one fused hardware instruction or
separate multiply and add operations, so do not assume one physical
instruction or one universal intermediate-rounding rule.

`min`, `max`, and `clamp` currently accept integer scalar/vector values so Tach
does not silently choose a cross-backend floating-point NaN policy. `cross`
accepts a matching `vec<float16, 3>` or `vec<float32, 3>`; the other geometry operations
accept matching floating vectors and return their component width. `break`
exits the nearest enclosing loop. `continue` advances its nearest loop and, in
a `for`, performs the update before testing the condition again. Neither may
appear outside a loop. Tach has no labeled transfer, function values, methods,
or recursion.

Use `tach check` as the authority for exact overloads. `float32` has about
seven decimal digits of precision; `float16` has roughly three and a maximum
finite magnitude of 65504. Both have less precision and range than
JavaScript's number type, and parallel floating-point reductions need not
reproduce serial CPU order bit for bit.

## 12. Parallel correctness

GPU runs may finish in any order. A `for` loop in TypeScript does not.
Correct code gives every write one of these foundations:

1. this run uniquely owns the destination, the way `scale` owns `values[i]`;
2. an atomic operation updates a shared integer even if others update it
   too;
3. a barrier makes one team of at most 256 runs wait for each other; or
4. a later `run` stage in a program sequences work after the previous
   stage has finished.

A bounds check prevents writing off the end of an array. It does not
prevent two valid runs from writing the same slot.

### 12.1 Shared memory

`shared<T>` is a whiteboard for one team. It is zero-initialized, declared
at the top of an indexed stage, and requires an explicit `@workgroup`
size. The next team, and the next stage, cannot see it.

### 12.2 Atomics

`atomic<int32>` and `atomic<uint32>` support atomic load, store, exchange,
add, subtract, minimum, maximum, and bitwise updates. Use them when many
runs must update one integer. Atomic state in a public buffer survives
across commands.

### 12.3 Barriers

`workgroupBarrier()` waits until every run in this team has reached that
line, then lets them all continue, with shared memory visible.
`bufferBarrier()` does the same for buffer memory inside that team.

Every run in the team must reach the barrier. A barrier inside
`if (i == 0)` will hang: the other runs never arrive. Neither barrier
waits for a different team.

## 13. Structured documentation and generated API docs

Tach source is the documentation authority. `@docs(...)` is structured and
checked, not an arbitrary blob. `//` is the only inline comment syntax.

```tach
@docs(
  title("Scaling"),
  summary("Scales floating-point values on the GPU."),
);

@docs(
  summary("Controls one scaling operation."),
  field(factor, "Multiplier applied to each input."),
)
type ScaleParams = {
  factor: float32,
};

@docs(
  summary("Multiplies every in-range value."),
  coordinate(i, "Value index."),
  param(values, "Values updated in place."),
  param(params, "Scaling parameters."),
)
export function documentedScale[i](
  values: buffer<float32[]>,
  params: ScaleParams,
) {
  // Rounded launches can include coordinates outside the array.
  if (i < values.length) {
    values[i] *= params.factor;
  }
}
```

Every `@docs` annotation requires `summary`. Its declaration context determines
the additional valid clauses:

| Context | Clauses |
|---|---|
| file | optional `title` |
| type | checked `field` entries |
| function | checked `param` entries |
| indexed function | function clauses plus checked `coordinate` entries |
| value helper or view program | function clauses plus `returns` |

The compiler carries descriptions into:

- JSDoc in generated `index.d.ts`;
- the generated project `README.md`; and
- one generated Markdown file per module.

`tach docs` refreshes only those Markdown files without recompiling shader or
binding artifacts. `tach build` always refreshes everything.

## 14. The generated TypeScript API

All named Tach structs are generated as exported TypeScript types. Only
`export function` declarations become recipe constructors.

For the first example, generated declarations look conceptually like:

```ts
import type {
  ComputeBuffer,
  ComputeCommand,
  LaunchOptions,
} from "@depths/tach";

export function scale(
  values: ComputeBuffer<Float32Array | readonly number[]>,
  factor: number,
  launch?: LaunchOptions<number>,
): ComputeCommand;
```

A view program returns `ComputeView`. An orchestration program accepts
`CommandOptions`. Compiler-private aliases in the real declaration file keep
user-defined Tach names from colliding with runtime type names.

Generated JavaScript imports `@depths/tach/internal`. That is a private
compiler/runtime protocol. Application code must import only:

```ts
import { tach, TachError } from "@depths/tach";
import { myProgram, type MyType } from "./build/index.js";
```

Do not import `@depths/tach/internal`, load shader files yourself, or construct
imitation recipes.

## 15. Sessions and lifetime

Tach provides scoped and persistent sessions.

### 15.1 Scoped session

Use a callback for finite work that returns host data:

```ts
const output = await tach(async (gpu) => {
  const input = gpu.buffer(new Float32Array([1, 2, 3]));
  const right = gpu.buffer(new Float32Array([10, 20, 30]));
  const result = gpu.buffer(new Float32Array(3));
  await gpu.submit(add(input, right, result));
  return result.read();
}, { powerPreference: "high-performance" });
```

The callback form opens the session, runs the callback, waits for submitted
work, closes every owned resource, and returns the callback result. Return host
data, not a session-owned buffer.

### 15.2 Persistent session

Use a persistent session for frame loops, simulations, solvers, or services:

```ts
const gpu = await tach({ powerPreference: "high-performance" });
const state = gpu.buffer(initialState);

try {
  for (let stepIndex = 0; stepIndex < 1_000; stepIndex++) {
    await gpu.submit(simulate(state, 1 / 60));
  }
  await gpu.idle();
} finally {
  gpu.close();
}
```

One long-lived session reuses shader modules, pipelines, buffers, descriptors or
bind groups, parameter storage, submission resources, and transient scratch.
`close()` is idempotent. Call `idle()` first when successful completion must be
established before teardown.

## 16. Buffers

```ts
interface ComputeBuffer<T> {
  write(value: T): void;
  read(): Promise<T>;
  destroy(): void;
}
```

`gpu.buffer(value)` clones the initial host value but allocates no physical GPU
storage yet. First submitted use supplies the compiler-generated layout codec
and materializes the backend resource.

Tach infers whether every stage buffer is read-only, write-only, read/write, or
atomic. Source never declares binding numbers, descriptor sets, access flags,
or provider usage masks.

Before materialization, `write(value)` may change the byte length. After
materialization, writes must preserve exact byte length and layout. Allocate a
new buffer when the shape changes.

`write()` validates and packs immediately, then schedules the backend upload.
`read()` waits for earlier work, transfers the resource, decodes it, and returns
a clone. Reading a never-used handle returns its cloned host value without
allocating GPU memory.

`destroy()` and session `close()` are idempotent. Using a destroyed handle, a
handle from another session, or a handle with an incompatible generated layout
raises a typed lifecycle or buffer error.

### Buffer identity and aliasing

Two distinct public buffer parameters in one recipe must receive distinct
`ComputeBuffer` handles:

```ts
// Invalid if copy has separate input and output buffer parameters:
copy(values, values);
```

This keeps access planning portable and unambiguous. Express intentional
in-place work with one Tach buffer parameter.

## 17. Recipes, preparation, submission, and ownership

Generated calls return opaque `ComputeCommand` recipes. View programs return
the narrower `ComputeView`.

Recipes are not session-owned. Their captured buffers are. At preparation or
execution, every captured buffer must belong to the executing session. A
scalar-only recipe, such as the transient-backed procedural view above, can be
reused across compatible sessions.

Recipe arguments are retained until execution. Do not mutate object or array
value arguments after construction:

```ts
const params = { factor: 2 };
const command = configuredScale(values, params);
params.factor = 100; // avoid mutation after recipe construction
await gpu.submit(command);
```

### Prepare without executing

```ts
await gpu.prepare(firstRecipe, secondRecipe);
```

`prepare` validates ownership, materializes resources, loads shaders, creates
required pipelines, and warms reusable runtime state without dispatching the
program. Use it before a latency-sensitive loop.

### Submit in order

```ts
await gpu.submit(
  first(values),
  second(values),
  third(values),
);
```

One submission preserves recipe and internal stage order. Concurrent calls to
`submit()` are serialized by the session.

## 18. Launch and repeat options

All ordinary recipes accept:

```ts
interface CommandOptions {
  readonly repeat?: number;
}
```

`repeat` is a positive `uint32` integer and repeats the complete program. Views
reject it because one view is one frame.

Only exported indexed shorthand accepts launch size:

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

| Tach coordinates | Host size |
|---|---|
| `[i]` | `number` |
| `[x, y]` | `readonly [number, number]` |
| `[x, y, z]` | `readonly [number, number, number]` |

Every component is a positive safe integer and rank must match. A 1D shorthand
can infer size from its first runtime-sized public buffer. Without inference or
an explicit size, Tach launches one workgroup.

Public orchestration programs derive all domains from Tach source and therefore
accept only `CommandOptions`.

## 19. Runtime API

```ts
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
```

Session creation accepts:

```ts
interface TachOptions {
  readonly powerPreference?: "low-power" | "high-performance";
}
```

The preference influences adapter selection but is not a guarantee.

`gpu.adapter` reports backend-neutral facts:

```ts
interface TachAdapterInfo {
  readonly backend: "webgpu" | "vulkan";
  readonly name: string;
  readonly vendor?: string;
  readonly architecture?: string;
  readonly type?: "integrated" | "discrete" | "virtual" | "cpu" | "unknown";
}
```

No raw WebGPU device or Vulkan handle escapes the runtime.

## 20. Browser and Deno use the same facade

There is one generated `index.js`, one `index.d.ts`, and one command ABI.
Backend selection depends only on the host:

| Host | Execution |
|---|---|
| browser global with WebGPU | fetch and decompress sibling `kernel.wgsl.gz` |
| Deno global | load sibling `kernel.spv` through Tach's Vulkan 1.3 host |
| unsupported environment | structured availability error |

The complete generated directory must remain available at runtime. Browser
servers and bundlers must preserve the WGSL URL beside the generated module.
Deliver `kernel.wgsl.gz` as unchanged binary content rather than declaring an
HTTP `Content-Encoding`; the Tach runtime performs the gzip decompression.
Deno needs read access to the generated SPIR-V and FFI access to the packaged
native library.

### Deno execution

```sh
deno run --allow-ffi --allow-read app.ts
```

The Vulkan host currently ships for x86-64 Windows and Linux. It requires a
Vulkan 1.3 loader and device with Synchronization2,
`shaderZeroInitializeWorkgroupMemory`, and `vulkanMemoryModel`.

Float16 is genuinely optional hardware functionality. A host session enables
supported Float16 functionality when it opens. Each generated module records
whether it needs WebGPU `shader-f16` and Vulkan `shaderFloat16`, plus only the
16-bit storage/uniform features its interfaces use. Command preparation fails
with a typed error when the selected adapter cannot satisfy those exact module
requirements. Projects without `float16` carry no optional Float16 requirement.

The compiler itself is distributed for:

- x86-64 and arm64 Windows;
- x86-64 and arm64 Linux; and
- Apple-silicon macOS.

Browser execution depends on browser WebGPU availability rather than the native
Vulkan host matrix.

## 21. Errors

Public failures are `TachError`:

```ts
import { TachError } from "@depths/tach";

try {
  await gpu.submit(command);
  await gpu.idle();
} catch (error) {
  if (error instanceof TachError) {
    console.error(error.code, error.operation, error.message);
  }
  throw error;
}
```

The exact public codes are:

```ts
type TachErrorCode =
  | "webgpu-unavailable"
  | "adapter-unavailable"
  | "device-request-failed"
  | "device-lost"
  | "gpu-validation"
  | "gpu-out-of-memory"
  | "gpu-internal"
  | "vulkan-unavailable"
  | "vulkan-profile"
  | "native"
  | "buffer"
  | "kernel"
  | "lifecycle"
  | "user"
  | "compiler-platform"
  | "compiler-install"
  | "compiler-execution";
```

`code` classifies availability, device, GPU validation, Vulkan profile, native,
buffer, kernel, lifecycle, callback, and compiler failures. `operation`
identifies the failing runtime operation when available. `cause` preserves the
original error.

Deferred GPU failures can surface at the next submission or completion
boundary. Do not assume the line that observes an asynchronous device error is
the line that caused it.

Device loss invalidates the session. Create a new session and recreate
application state; Tach does not hide recovery behind stale handles.

## 22. Tooling

The public command surface is intentionally small:

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

| Command | Contract |
|---|---|
| `build` | validate and atomically replace the complete dual-backend package |
| `build --verbose` | build normally and add diagnostic IR/plan artifacts |
| `check` | validate the entire WebGPU and Vulkan pipeline without writes |
| `docs` | refresh only generated Markdown while preserving compiled output |
| `fmt` | transactionally format every source file in the project |
| `instructions` | print compact AI-agent guidance |
| `instructions --details N ...` | retrieve exact numbered deep sections |
| `version` | print the installed Tach version |
| `help` | print command help |

`check` covers project discovery, naming, imports, both DAGs, recovering
parsing, semantics, IR verification and optimization, WGSL, SPIR-V 1.6,
generated bindings, and documentation rendering.

`fmt` and generated writes are transactional. One invalid file prevents a
partially updated project.

### Errors, warnings, and machine output

Errors reject faulty projects. Warnings do not reject or change output; they
identify only statically proven dead work or suspicious GPU access/control
patterns. Each diagnostic has a stable `code`, an exact UTF-8 source span, the
source line, related locations when several sites form one issue, and a
specific `help` action when Tach can recommend one safely.

Ordinary commands render a compact Markdown-like terminal report:

```text
## warning [unused-binding]

local "discarded" is never used

--> kernels/image.tach:8:3
  |
8 |   const discarded = 1;
  |   ^^^^^^^^^^^^^^^^^^^^

- help: remove the binding or use its value
```

Add `--json` to `build`, `check`, `docs`, or `fmt` for one JSON value on
stdout and no human prose. The top-level record has `schema: 1`, `ok`,
`command`, and `diagnostics`; a non-compiler usage/runtime failure instead adds
an `error` object. Every diagnostic contains `severity`, `code`, `message`,
`span`, optional `source` and `help`, and optional `related` locations. This is
the preferred interface for agents, editors, and build automation.

Current warnings are deliberately conservative:

| Code | Proven condition |
|---|---|
| `unused-import` | no declaration from an imported file is referenced |
| `unused-binding` | a parameter, index, local, or shared variable is never used |
| `unreachable-function` | a private function is outside every exported dependency graph |
| `discarded-value` | a pure expression is evaluated and ignored |
| `constant-condition` | an `if` or conditional expression is fixed, or a loop is statically false |
| `zero-dispatch` | a literal launch axis prevents a stage from running |
| `no-effect-kernel` | a reachable kernel cannot affect externally visible memory |
| `constant-write-index` | an unconditional non-atomic write address is invocation-independent |
| `strided-access` | adjacent one-dimensional invocations provably use a non-unit buffer stride |

Tach does not warn about folklore such as a particular workgroup multiple,
integer division, or a possible `fma`: those depend on the algorithm, target,
or driver and would train users to ignore the compiler. Related locations fold
repeated instances of one proven access pattern into one report.

### Verbose diagnostics

`tach build --verbose` adds:

```text
build/diagnostics/
  flow.ir
  kernel.ir
  kernel.spvasm
  project.json
  runtime.json
  spirv.kernel.ir
  spirv.plan.json
  web.kernel.ir
  web.plan.json
```

These files explain compilation. They are not application inputs and must not
be shipped in place of ordinary artifacts.

## 23. Deno compiler API

Deno-native build tools can call project operations directly:

```ts
import {
  build,
  check,
  CompilerError,
  compilerPath,
  docs,
  format,
  renderDiagnostics,
} from "@depths/tach/compiler";

const cwd = Deno.cwd();
const checked = await check({ cwd });
if (checked.diagnostics.length) {
  console.warn(renderDiagnostics(checked.diagnostics));
}
await build({ cwd, verbose: true });
await docs({ cwd });
await format({ cwd });
console.log(await compilerPath());
```

This API operates on whole projects. It does not expose the private compiler
protocol and must not be included in browser bundles. A script using it needs
Deno read, write, environment, subprocess, and network permissions; `-A` is the
simple choice for a trusted local build script.

Successful `build`, `check`, and `docs` calls return warnings in
`ProjectResult.diagnostics`. A source failure throws `CompilerError`, whose
`diagnostics` property contains the machine records and whose message is the
same report produced by `renderDiagnostics`. Catch `CompilerError` when a tool
needs to distinguish source diagnostics from installation, subprocess, or
filesystem failures.

Compiler resolution checks an explicit `TACH_BIN`, a package-local binary, the
repository development binary, then the exact release asset for the installed
package version and platform. Release downloads are checksum-verified and
placed atomically. An invalid explicit path is an error, not a silent fallback.

## 24. Performance that matters

GPU speed comes from enough independent work and from avoiding transfers and
synchronization, not from changing a small loop into GPU syntax.

Start with these rules:

1. keep one persistent session for repeated work;
2. create large resident buffers once;
3. reuse those handles across recipes;
4. use transients for private intermediate state;
5. batch naturally related recipes in one `submit`;
6. call `prepare` before latency-sensitive measurement;
7. keep readback outside hot loops;
8. use views for displayed frames; and
9. synchronize only where the CPU requires completion.

First execution can include shader loading and decompression, validation,
pipeline creation, buffer upload, parameter allocation, and scratch growth.
Warm execution reuses those objects.

For a defensible benchmark:

1. create one persistent session;
2. allocate inputs once;
3. prepare and warm the complete measured recipe;
4. start the timer;
5. submit representative work;
6. end with `idle()`, `read()`, or `present()`, according to the real application
   boundary; and
7. validate results outside the timed interval.

Do not time only `submit()` and call it GPU execution time. Do not compare a GPU
path that includes upload/readback against a CPU loop that includes neither.

## 25. Common mistakes

### Awaiting a recipe

```ts
await scale(values, 2); // wrong
await gpu.submit(scale(values, 2)); // correct
```

### Reading every intermediate

Keep intermediates in resident buffers or Tach transients. Read only final
values needed by TypeScript.

### Omitting edge guards

Physical dispatch sizes are rounded. Guard coordinates before access.

### Assuming a workgroup barrier is global

It synchronizes only one workgroup. Use another `run` stage for device-wide
ordering.

### Reusing one handle for two parameters

Distinct buffer parameters cannot alias. Model in-place work with one parameter.

### Mutating recipe arguments

Plain objects and arrays are retained until execution. Treat them as immutable
after recipe construction.

### Returning a buffer from a scoped session

The session closes before the caller can use it. Return `await buffer.read()`
instead.

### Measuring queue submission

`submit` does not mean completion. End measurement at the boundary your
application actually needs.

### Reading a frame merely to display it

Return `view<srgb8>` and use browser `present`.

### Editing generated output

Change Tach source or `tach.json`, then rebuild the complete directory.

## 26. A practical authoring route

For each new GPU operation:

1. Decide what one invocation owns.
2. Choose one, two, or three coordinates.
3. Define host-facing scalar, vector, struct, and buffer shapes.
4. Add bounds guards for rounded dispatches.
5. Prove that writes are unique, atomic, or synchronized.
6. Start with one exported indexed function.
7. Introduce helpers only for reusable value computation.
8. Introduce private stages and a program only when global sequencing is needed.
9. Introduce transients only for program-private intermediate storage.
10. Return a view when the final result is a displayed image.
11. Run `tach fmt`, `tach check`, then `tach build`.
12. Import the generated declarations and let TypeScript expose host-shape
    mistakes.
13. Execute representative boundary sizes and validate actual GPU results.

Useful test sizes sit immediately below, at, and above workgroup boundaries:
`255/256/257`, `15/16/17`, unequal 2D axes, and partial final workgroups.

## 27. AI-agent programming support

The npm package carries its own version-matched language context:

```sh
npx tach instructions
npx tach instructions --details 18 24 45 54 59 70
```

The default response is a dense complete introduction with precise section
pointers. `--details` returns only requested authoritative chunks from the
larger guide. This lets an AI coding agent load the minimum relevant language
context without guessing from generic shader knowledge.

Instructions do not require a Tach project or native compiler download.

## 28. Where to go deeper

This README is sufficient to create, build, execute, display, document, and
optimize a Tach project. The repository references define compiler-facing
details:

- [language reference](https://github.com/Depths-AI/tach/blob/master/docs/language.md);
- [examples guide](https://github.com/Depths-AI/tach/blob/master/examples/README.md);
- [architecture](https://github.com/Depths-AI/tach/blob/master/docs/architecture.md);
- [intermediate representations](https://github.com/Depths-AI/tach/blob/master/docs/ir.md);
- [generated ABI](https://github.com/Depths-AI/tach/blob/master/docs/abi.md); and
- [full AI-agent application guide](https://github.com/Depths-AI/tach/blob/master/docs/INSTRUCTIONS.md).

`@depths/tach` is licensed under `AGPL-3.0-only`.
