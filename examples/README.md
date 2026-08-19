# Tach examples

If you write TypeScript for the web, you already know how to *use* a GPU
through a library. These files are how Tach asks you to *describe* GPU work
yourself: small functions that look like TypeScript, compiled once, then
called from ordinary application code.

This folder is not an app and not a speed test. It is a set of fourteen short
programs that together cover the language you will actually write. Each
program exists because it teaches one idea that is easy to miss if you only
ever saw `array.map`. Read them in order the first time. After that, jump
to the file that matches the thing you are trying to build.

You do not need prior GPU, shader, or graphics experience. You do need to
be comfortable with TypeScript functions, types, and `async`/`await`.

## The one idea everything else sits on

A CPU usually follows one instruction stream. It walks an array with a
`for` loop, one index after another.

A GPU is useful when the same work can happen at many indices at once.
Instead of "loop over the array," you write "here is what happens at index
`i`," and Tach launches that function across the whole array.

```tach
export function scale[i](data: buffer<float32[]>, factor: float32) {
  if (i < data.length) {
    data[i] *= factor;
  }
}
```

Read `[i]` as: this function runs many times, and each run gets its own
`i`. If the array has a million numbers, you are describing one
multiplication, not a million-line program. Tach turns that description
into GPU work and a typed TypeScript function you can call.

Three words are enough for almost every file here:

- A **stage** is the work for one index (or one pixel, or one particle).
- A **buffer** is an array that lives on the GPU. It stays there until you
  explicitly `read()` it back into JavaScript.
- A **recipe** is the object a generated function returns. Building a
  recipe does no GPU work. `gpu.submit(recipe)` queues the work.
  `buffer.read()` waits and brings the result back.

That last split is the performance rule in everyday language: send small
inputs and commands to the GPU, leave large data there, and read only what
the page actually needs.

## What you will see in every kernel

**A bounds check.** GPUs launch work in fixed-size groups. If you have 4
numbers and the group size is 256, 252 extra runs still happen, with `i`
past the end of the array. The `if (i < data.length)` you will keep seeing
is not timid style. It is how you ignore those extra runs. Forgetting it
is the most common first bug.

**Typed numbers.** There is no JavaScript `number`. `float16` is a 16-bit
float, `float32` is a 32-bit float, `int32` / `uint32` are 32-bit integers,
and `float32x4` is four floats in one value (a pixel, a position, a color).
Binary16 is explicit because it trades range and precision for half-sized
storage and hardware throughput where supported.

**`export` means "call me from TypeScript."** A function without `export`
is private GPU glue. After `tach build`, only the exported names appear in
`build/index.js`.

**You still write TypeScript around it.** The `.tach` file never talks to
the DOM, never fetches, never opens a canvas. Application code does that
with `@depths/tach` and the generated functions.

## The project

```text
examples/
  tach.json          project name and generated package name
  core/              everyday language
    scalars.tach     multiply float32 and float16 arrays
    bitwise.tach     integer bit operations
    control.tach     loops and branches
    for.tach         for-loops and vectors
    math.tach        sin, length, and friends
    view.tach        turn pixels into something a canvas can show
  simulation/        slightly larger pieces
    types.tach       shared Particle type
    atomics.tach     many workers updating one counter
    particles.tach   move particles using the shared type
```

`tach.json` is the whole manifest. Tach finds every `<folder>/<file>.tach`
under it. There is no source list and no webpack config for kernels.

| You call this from TypeScript | It lives in | In one sentence |
|---|---|---|
| `scale` | `core/scalars.tach` | Multiply every number by a factor. |
| `scaleFloat16` | `core/scalars.tach` | Keep the same operation in binary16 end to end. |
| `float16Math` | `core/scalars.tach` | Exercise binary16 scalar, vector, and geometry math. |
| `halveFloat16` | `core/scalars.tach` | Carry a binary16 constant through an orchestration plan. |
| `bitwise` | `core/bitwise.tach` | Do the same bit math at every index. |
| `transform` | `core/control.tach` | Walk a strided slice, add, maybe write back. |
| `reduceLanes` | `core/for.tach` | Sum each group of four integers. |
| `math` | `core/math.tach` | Run the standard math functions at each index. |
| `gradient` | `core/view.tach` | Paint a gradient and return a displayable image. |
| `gradientInto` | `core/view.tach` | Paint that gradient into a buffer you still own. |
| `swatch` | `core/view.tach` | Paint four known colors as a tiny image. |
| `swatchInto` | `core/view.tach` | Same four colors, into a buffer you still own. |
| `accumulate` | `simulation/atomics.tach` | Count cooperating workers safely. |
| `integrate` | `simulation/particles.tach` | Move each particle one time step. |

The names `paint`, `stamp`, and `integrateParticle` are private. They are
helpers the public functions call. You will not import them from
JavaScript.

## `scale` - start here

`core/scalars.tach` is the whole mental model in one screen.

You already know this TypeScript:

```ts
for (let i = 0; i < data.length; i++) {
  data[i] *= factor;
}
```

The Tach version says the same thing about *one* `i`, then lets the GPU
run many `i`s at once. The data lives in a `buffer`, so it can stay on the
GPU across several `submit` calls. The harness does exactly that: it
scales the same buffer by 2, then 3, then 4, and only then reads. The
array `[1, 2, 3, 4]` becomes `[24, 48, 72, 96]`.

This file is first because every other example reuses its shape: a
coordinate, a buffer, a small constant from TypeScript, and a bounds
check. If `scale` makes sense, you can read the rest.

## The Float16 trio - smaller values, real tradeoffs

`scaleFloat16` repeats `scale` with `float16` data so the difference stays
visible: TypeScript supplies a `Float16Array`, every stored element occupies
two bytes, and arithmetic remains binary16 through WGSL and SPIR-V.
`float16Math` then exercises scalar, vector, transcendental, and geometric
operations without silently widening them. Its `Float16Series` input also
proves that a prefixed runtime tail keeps its exact `.length` when physical GPU
alignment needs extra bytes. `halveFloat16` carries an exact binary16 constant
through a public orchestration plan, covering the same type at the multi-stage
boundary.

Tach checks the optional WebGPU and Vulkan Float16 requirements automatically,
including direct and prefixed-tail byte extents that require private physical
padding. Use binary16 when its storage, bandwidth, or arithmetic benefit matters
and the algorithm can tolerate roughly three decimal digits of precision and a
maximum finite magnitude of 65504. Smaller is not automatically more accurate
or faster.

## `bitwise` - 32-bit integers are not JavaScript numbers

JavaScript has one number type. Bit shifts on it are defined, but they are
defined for JS, not for a 32-bit GPU lane. `core/bitwise.tach` writes one
unsigned result using shifts, or, xor, not, and a signed right shift.

Two rules surprise people coming from TypeScript:

1. Shift counts wrap at 32. `<< 40` is `<< 8`. The kernel uses oversized
   counts on purpose so both backends have to get that wrap right.
2. A signed right shift keeps the sign. `-64 >> 3` still looks like a
   negative number, then it is converted to unsigned for storage.

The TypeScript test recomputes the same expression with `>>> 0` and
expects an exact match, no epsilon. This file is here so you see that
"integer" in Tach means a real 32-bit word, not a `number` that happens
to be whole.

## `transform` - loops and branches are normal

`core/control.tach` looks like ordinary structured code: a `while`, an
`if` / `else`, a `bool` flag from the host.

Each run starts at its own `start` and then hops by 64 (`start`,
`start + 64`, `start + 128`, …). Think of 64 cashiers, each taking every
64th customer. They add until the running total would exceed 1000, cap or
bump the result, and write it back only if `enabled` is true.

It is here because people assume GPU code cannot really branch. It can.
You write the `if` you mean. What you must not do is assume the 64
workers finish in a particular order, or that a `bool` from TypeScript is
a JavaScript boolean on the GPU. Tach packs that flag for you. The
harness checks the exact piecewise result so the loop, the cap, the else
branch, and the flag all have to work together.

The `@workgroup(64)` above the function is the team size those 64-apart
strides are built for. You can ignore workgroups until you need this kind
of teamwork. `scale` never mentions them.

## `reduceLanes` - vectors and `for`

A `uint32x4` is four unsigned integers in one value. `core/for.tach`
loads four neighbors, then sums them with a `for` loop that indexes the
vector as `values[lane]`.

Why not write `values.x + values.y + values.z + values.w`? Because that
would not test the thing you will want: picking a lane with a variable.
The early `return` when fewer than four values remain is the same edge
rule as `scale`, applied to a reduction that consumes four inputs per
output.

The harness puts `1, 2, 3, 4` at the front of a long zeroed array and
expects `10` at index 0. One complete group, one number you can check in
your head.

## `math` - the functions you already know

`core/math.tach` is a tour of Tach's math library: `sin`, `cos`, `sqrt`,
`length`, `normalize`, `dot`, `cross`, `pow`, rounding, and integer
`min` / `max` / `clamp`. Each invocation seeds a 3D vector from its
index and writes a `float32x4` result.

The names match `Math` and everyday graphics code on purpose. Two
differences matter:

- These run per invocation on 32-bit floats, not on JavaScript's 64-bit
  `number`.
- Tach does not yet offer floating-point `min` / `max` / `clamp`, because
  those need a single rule for `NaN` that both backends share. The kernel
  uses the integer forms so it does not pretend that rule exists.

The test compares against a CPU version with a small tolerance. `sin` on
a GPU does not have to match `Math.sin` bit for bit. It has to be the
same function closely enough that you can trust the kernel is doing math,
not skipping it.

## `view.tach` - making a picture without reading pixels

This is the file frontend developers usually came for.

If you computed an image in a buffer and called `read()` every frame, you
would copy the whole frame to JavaScript and then back onto the GPU to
draw it. That is the slow path. Tach instead lets a program *return an
image*. You write linear floating-point RGBA, the way a renderer thinks
about color. Tach turns that into 8-bit sRGB for a screen. In the
browser you call `gpu.present(canvas, view)` and the pixels never visit
the CPU.

A view is not a `<canvas>` and not a DOM node. It is the finished picture
as a result type: `view<srgb8>`. Showing it is a separate step.

```text
gradient({ width, height, bias })   describe the picture
gpu.submit(view)                    compute it, leave it on the GPU
gpu.present(canvas, view)           compute it onto this canvas, then wait
```

`present` needs a real canvas, so it exists in the browser. Deno on
Vulkan can still `submit` the same view; it just has no window to show
it in yet.

The file has four exported programs because two independent questions
each have two answers.

**Who owns the pixels?**

- `gradient` creates the frame itself. TypeScript only passes width,
  height, and a color bias. The recipe holds no buffer, so any Tach
  session can run it. This is how a procedural background works: the
  page does not allocate an image array.
- `gradientInto` writes the same gradient into a buffer *you* created.
  After drawing, you can `read()` those floats if the CPU needs them.
  This is how you show a picture and still keep the linear pixels for
  later GPU work.

**How sure are we the colors are right?**

Gradients are good for seeing a frame and for stressing launch sizes.
They are a poor unit test, because "a purple-ish PNG" can hide an
off-by-one in color conversion. `swatch` and `swatchInto` paint a 2 x 2
image whose colors are only 0 or 1. After Tach converts to 8-bit sRGB,
those become the bytes 0 and 255. The browser test draws both swatches
and checks the actual PNG bytes. That is why the tiny image exists.

`paint` and `stamp` are the private "color this pixel" functions. The
exported programs only say *where the pixels live* and *how big the
image is*. That split is the one you will copy: keep shading in an
indexed stage, keep "this is a frame" in a short exported program.

A note on alpha: the swatch uses alpha `1` because a browser canvas
presented by Tach is opaque. A zero alpha would test the compositor, not
Tach's color conversion.

## `types.tach` and `integrate` - sharing a type across files

Real projects do not put every type next to every kernel.

`simulation/types.tach` defines `Particle` (position and velocity) and
`ParticleParams` (a time step and a live count). It exports nothing to
JavaScript as a function. The structs still appear in the generated
`.d.ts`, because application code needs to create those objects.

`simulation/particles.tach` starts with:

```tach
import "simulation/types";
```

That is the only import form: the file identity, no `./`, no `.tach`, no
named bindings. Imports are not transitive. If a third file wants
`Particle`, it must import `simulation/types` itself. That is stricter
than TypeScript, and it is on purpose. You always know where a name
comes from.

`integrateParticle` is a helper: ordinary value in, ordinary value out,
no buffers. `integrate` is the GPU stage. It loads particle `i`, calls
the helper, and stores the result. The helper is the math you would
extract in TypeScript. The stage is the part that knows about the array
and the index.

The `i >= params.count` check is the "live subset" rule. You may
allocate room for more particles than are alive this frame. Extra GPU
runs must not move the unused slots. The harness places two particles,
steps `dt = 0.5`, and checks that each position moved by half of its
velocity. You can do that arithmetic on paper.

## `accumulate` - when workers must meet

Until this file, every invocation owns a different slot and never talks
to its neighbors. That is the easy, fast case.

Sometimes they must meet. A histogram, a count, a lock-free flag: many
workers update the same integer. `simulation/atomics.tach` is that
pattern, stripped down.

64 workers share a small scratch array that only they can see. Each one
adds 1 to its own scratch slot. The first worker to touch a slot sees
`0` and adds 1 to a global `total`. Then they all continue. Because
there are 64 slots and 64 workers, `total` becomes 64.

Two new tools appear:

- `shared<...>` is a whiteboard for this team of 64, thrown away when
  the team finishes. The next team of 64 gets a fresh one.
- `atomicAdd` is "add to this integer even if someone else is adding at
  the same time." A normal `total += 1` from many workers can lose
  updates. An atomic add cannot.

They also wait together (`workgroupBarrier`) so nobody uses the
whiteboard before it is set up. Everyone has to reach that wait. You
cannot put it inside `if (i == 0)` for only some workers; the others
would wait forever.

This file is last among the language examples because you should not
reach for atomics until a unique slot, or a later second stage, is not
enough. Most first kernels never need them.

## How these files get used

`tach build` in this directory writes one package: JavaScript functions,
TypeScript types, a browser shader, and a native shader. Application
code imports `scale`, `gradient`, and the rest from that package, and
imports `tach` from `@depths/tach`. The same TypeScript runs in a
browser (WebGPU) and in Deno (Vulkan). You do not choose a backend.

The repository's browser and Deno tests compile this project and call
every public function. They are how we know the fourteen programs still
mean the same thing on both hosts. A separate project, `showcase-ts`,
measures large workloads. It is not this folder.

To explore locally, from the repository root:

```sh
npx tach --help
```

Or open any `.tach` file here, run `npx tach check` from this directory,
then `npx tach build`. Import `./build/index.js` from a small TypeScript
file the way the root README does for `scale`.
