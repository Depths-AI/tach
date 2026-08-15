<p align="center">
  <img src="tach.svg" alt="Tach logo" width="144">
</p>

<p align="center">
  <a href="https://www.npmjs.com/package/@depths/tach"><img src="https://img.shields.io/npm/v/%40depths%2Ftach?style=flat-square" alt="npm version"></a>
  <a href="LICENSE"><img src="https://img.shields.io/github/license/Depths-AI/tach?style=flat-square" alt="license"></a>
  <a href="docs/architecture.md"><img src="https://img.shields.io/badge/targets-WebGPU%20%7C%20Vulkan%201.3-111827?style=flat-square" alt="WebGPU and Vulkan 1.3"></a>
</p>

# Tach

Tach is a small, typed language for general-purpose GPU programming from
TypeScript applications. Its goal is to make directly authored GPU compute
ergonomic for TypeScript developers. Kernels look familiar, while Tach
handles the GPU-specific work that normally surrounds them: shader entry
points, bindings, memory layout, launch geometry, resource lifetimes, target
validation, and host bindings.

```tach
export function scale[i](
  values: buffer<float32[]>,
  factor: float32,
) {
  if (i < values.length) {
    values[i] *= factor;
  }
}
```

One Tach project currently produces validated WGSL for browser WebGPU,
validated SPIR-V 1.6 for Vulkan 1.3, and a typed JavaScript interface for both.
Application code uses the same buffers, commands, and generated functions on
either host; it does not contain a WebGPU/Vulkan branch. A native MSL backend
is planned for an upcoming version, extending the same model to Apple GPUs
without changing Tach source into Metal-specific code.

Tach is deliberately a compute language rather than a graphics API, tensor
framework, or thin shader wrapper. It is intended for simulation, procedural
generation, numerical work, image processing, custom rendering compute,
physics, and other algorithms whose parallel work is worth expressing
directly.

## Why Tach exists

JavaScript and TypeScript developers have excellent ways to *use* the GPU, but
there is a large gap between high-level libraries and low-level GPU APIs.

At one end, a framework can provide tensors, operators, automatic
differentiation, or an entire rendering model. That is wonderfully productive
when the problem fits the framework. At the other end, WebGPU provides direct
and portable access to GPU resources, but the application must own shader
source, layouts, bind groups, pipeline construction, command encoding,
readback, validation, and usually a separate answer for native execution.

Tach occupies the space between those ends. The algorithm remains an explicit
kernel, but the ceremony becomes compiler-owned. The language is narrow enough
to validate consistently across its targets and familiar enough that a
TypeScript developer does not first need to become a graphics programmer.

That focus also explains how Tach relates to several nearby projects:

| Project | What it is designed around | How Tach differs |
|---|---|---|
| [CubeCL](https://github.com/tracel-ai/cubecl) | A Rust language extension, JIT compiler, and runtime family aimed at high-performance compute across a broad set of backends | CubeCL offers more backends and more scope for device-specific specialization. Tach offers a standalone, ahead-of-time language, a simpler baseline-kernel entry point, and npm/Deno ergonomics for TypeScript applications. Tach's optimizer and backend range are younger and deliberately narrower. |
| [wgpu](https://wgpu.rs/) | A Rust implementation of the WebGPU API for native and web applications | wgpu gives applications direct control over devices, resources, pipelines, and shader modules. Tach generates the shader and ABI and manages routine dispatch plumbing. That removes substantial ceremony, but it also exposes less low-level control and fewer backend-specific features. |
| [zgpu](https://github.com/zig-gamedev/zgpu) | A Zig helper layer over Dawn native WebGPU, shaped for Zig game and graphics development | zgpu makes direct WebGPU integration convenient for Zig programs and fits naturally into a graphics engine. Tach starts from typed compute kernels and exposes commands to TypeScript without application-owned WebGPU objects. Tach is less suitable when the program needs to control the surrounding graphics pipeline itself. |
| [PyTorch](https://pytorch.org/) | A tensor and machine-learning framework with a large operator ecosystem, automatic differentiation, and accelerator runtimes | PyTorch is vastly richer for standard tensor and ML work, and its mature operators are usually the right choice there. Tach has no tensor framework, model stack, autograd, or comparable library ecosystem. It instead allows the custom parallel operation itself-including irregular, simulation, and non-ML workloads-to be expressed directly. |

These are not interchangeable products and the distinctions are intentional.
Choose PyTorch when the problem is fundamentally tensor and ML work. Choose
wgpu or zgpu when the application should own the GPU API. Choose CubeCL when a
Rust-centered, aggressively specialized multi-backend compute stack fits the
project. Tach is for directly authored GPU compute that should feel native in
a TypeScript codebase and run through one API in browsers and on Vulkan.

Tach optimizes first for a low-friction language, compile-time validation, one
portable ABI, explicit multi-stage compute, and a small host API. It does not
claim that portable kernels automatically match hand-tuned CUDA, Metal, or
backend-specific libraries. It does not yet offer CUDA, ROCm, native MSL,
tensor-core intrinsics, a graphics pipeline, CPU fallback, or the ecosystems of
the older projects above. Those limits are the cost of keeping the current
model coherent and approachable while its optimizer and backend coverage grow.

## Install

Install the runtime and compiler from npm:

```sh
npm install @depths/tach
```

Tach tooling and native execution use Deno. Browser execution requires WebGPU;
native execution currently requires x86-64 Windows or Linux with a Vulkan 1.3
loader and compatible device. The compiler itself is available for x64 and
arm64 Windows and Linux, and Apple-silicon macOS.

The npm package provides both the `tach` command and the `@depths/tach`
runtime. The compiler binary for the current operating system and architecture
is resolved automatically.

## Create a project

A Tach project has one `tach.json` and one or more module directories. Each
module contains `.tach` files directly; deeper source nesting is intentionally
not part of the project model.

```text
scaling/
  tach.json
  kernels/
    scale.tach
```

Create `tach.json`:

```json
{
  "name": "scaling",
  "version": "0.1.0",
  "javascript": {
    "package": "@example/scaling"
  },
  "docs": {
    "title": "Scaling kernels",
    "summary": "Small GPU transforms used by the application."
  }
}
```

Save the opening kernel as `kernels/scale.tach`:

```tach
@docs(
  summary("Multiply every value in a buffer by one factor."),
  coordinate(i, "Index of the value to update."),
  param(values, "Values updated in place."),
  param(factor, "Multiplier applied to every value."),
)
export function scale[i](
  values: buffer<float32[]>,
  factor: float32,
) {
  // Rounded-up dispatches may contain an invocation beyond the buffer.
  if (i < values.length) {
    values[i] *= factor;
  }
}
```

Run project commands anywhere beneath `tach.json`:

```sh
npx tach fmt
npx tach check
npx tach build
```

`fmt` formats every kernel in the project. `check` validates the complete
project through both targets without writing output. `build` writes one
cohesive generated package:

```text
build/
  package.json
  index.js
  index.d.ts
  kernel.wgsl
  kernel.spv
  README.md
  docs/
    kernels.md
```

Generated files are one unit and should not be edited by hand.

## Run a kernel

Import the Tach runtime and generated function separately:

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

The host model has three ordinary operations:

1. `gpu.buffer(value)` creates session-owned GPU state.
2. A generated function such as `scale(...)` constructs a command.
3. `gpu.submit(...)` queues commands; `read()` or `idle()` waits for their
   completion.

The callback form of `tach()` owns the complete session lifetime. For
long-lived state, open and close the session explicitly:

```ts
import { tach } from "@depths/tach";
import { step } from "./build/index.js";

const gpu = await tach();
const state = gpu.buffer(initialState);

try {
  for (let frame = 0; frame < 1_000; frame++) {
    await gpu.submit(step(state, 1 / 60));
  }
  await gpu.idle();
} finally {
  gpu.close();
}
```

`submit()` waits for preparation and queue submission, not GPU completion.
Readback, `idle()`, and the end of a scoped session are completion boundaries.

## The source model

Tach keeps simple kernels short without making larger algorithms opaque:

| Source form | Meaning | Generated host API |
|---|---|---|
| `function helper(...)` | Value helper called by kernels | None |
| `function stage[i](...)` | Private indexed GPU stage | None |
| `export function kernel[i](...)` | Indexed stage with a synthesized one-dispatch program | Typed command constructor |
| `export function program(...)` | Explicit orchestration of one or more stages | Typed command constructor |

The brackets declare logical invocation coordinates. A two-dimensional kernel
can use `[x, y]`; the host then supplies a matching logical size:

```tach
export function image[x, y](pixels: buffer<float32x4[]>) {
  const width = 1920;
  const pixel = y * width + x;
  if (pixel < pixels.length) {
    pixels[pixel] = float32x4(
      float32(x) / 1920.0,
      float32(y) / 1080.0,
      0.5,
      1.0,
    );
  }
}
```

```ts
await gpu.submit(image(pixels, { size: [1920, 1080] }));
```

Tach rounds logical sizes to complete workgroups, so kernels guard edge
invocations before indexing. Parameters are either `buffer<T>` GPU storage or
immutable values packed by the compiler. The core value types are `bool`,
`int32`, `uint32`, `float32`, and two-, three-, or four-lane numeric vectors.
Struct types are always emitted into the TypeScript API.

Files import other project files by extensionless module/kernel identity:

```tach
import "data/particles";
```

Declarations from the current file and directly imported files are visible.
Imports are not transitive, names are unique across the project, and the
compiler rejects cyclic file or module dependencies.

## Compose multi-stage work

An exported indexed function is the concise path for one dispatch. When an
operation needs several stages, export an orchestration function instead:

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

The host still receives one command constructor:

```ts
await gpu.submit(transform(input, output, count, 2, 0.5));
```

The compiler checks stage calls, domains, transient lifetimes, storage access,
and synchronization before either target is emitted.

## Documentation and agent guidance

Structured `@docs(...)` declarations are checked against the type or function
they describe and become documentation in generated Markdown and TypeScript
declarations. `//` is the single inline-comment form.

Every successful build refreshes the generated package documentation. Refresh
only documentation with:

```sh
npx tach docs
```

The npm package also contains a language and tooling guide designed for coding
agents. Its short index is available without a project or compiler invocation:

```sh
npx tach instructions
npx tach instructions --details 6 7
```

Other available commands are shown by `npx tach help`.

## Learn the complete language

- [Language guide](docs/language.md) - syntax, types, expressions, imports,
  programs, memory, synchronization, documentation, and diagnostics.
- [TypeScript guide](tach-ts/README.md) - runtime behavior, buffers, commands,
  execution, readback, errors, and generated bindings.
- [AI-agent guide](docs/INSTRUCTIONS.md) - the complete language and tooling
  reference intended for programmatic consumption.
