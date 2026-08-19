# Tach large GPU showcase

This standalone npm workspace measures nine fixed, sustained GPU workloads from
one Tach project through Chromium/WebGPU and Deno/Vulkan 1.3. It is a
calculator, not a performance gate: it records what the selected adapters did,
with one workload profile, five samples, raw observations, and no pass/fail
budget.

The showcase answers a practical beginner question: what useful work can a
TypeScript application describe in Tach while shaders, bindings, scratch, color
conversion, and "am I on WebGPU or Vulkan?" stay out of application code?

## One project, two hosts

Each workload lives in its own `.tach` file. `kernels/mesh` and
`kernels/procedural` directly import `shared/color`, so the build also exercises
whole-file project imports. One `tach build` merges all eight files into one
JavaScript facade, one declaration file, compressed WGSL, and SPIR-V 1.6.

The shared TypeScript workload implementation imports all nine generated recipes
through `build/index.js`. Browser and Deno runners open one persistent Tach
session and call the same functions. Application code never names a shader
entry, a bind group, or a canvas format.

## Workloads

| Category    | Kernel             | Fixed useful work                                                                                                                                                     |
| ----------- | ------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Rendering   | `procedural.tach`  | 1920 x 1080 analytic scene; up to 72 traversal steps, eight soft-shadow samples, three ambient-occlusion samples, material lighting, fog, and spatial post-processing |
| Rendering   | `mesh.tach`        | 1920 x 1080 compute renderer over terrain and 244 torus-knot, twisted-ribbon, superquadric, and organic elements: 162,481 vertices and 312,064 triangles              |
| Mathematics | `matrix.tach`      | FP32 2048 x 2048 dense matrix multiplication with 16 x 16 cooperative shared-memory tiles: 17,179,869,184 floating-point operations                                   |
| Mathematics | `matrix.tach`      | Identical matrix dimensions, tiling, inputs, and operation count in FP16, with half the matrix storage                                                                |
| Mathematics | `monte-carlo.tach` | 1,048,576 geometric Brownian paths, each advanced through 64 Box-Muller Gaussian time steps                                                                           |
| Physics     | `particles.tach`   | 2,097,152 particles, 64 integration steps, and four moving softened inverse-square attractors per step                                                                |
| Physics     | `wave.tach`        | 2048 x 2048 height/velocity grid advanced through 64 five-point stencil steps                                                                                         |
| Physics     | `oscillators.tach` | FP32 ensemble of 1,048,576 coupled conservative nonlinear oscillators advanced through 512 register-resident symplectic steps                                         |
| Physics     | `oscillators.tach` | Identical oscillator ensemble and integration work in FP16, with half the state storage                                                                               |

The matrix pair isolates precision under the same tiled memory-access pattern.
The oscillator pair instead keeps state in registers for 512 steps between one
initial read and one final write. Its 7,516,192,768 counted floating-point
operations therefore emphasize sustained arithmetic rather than memory
bandwidth. The reports preserve both results even when a backend does not make
FP16 faster.

The procedural renderer runs traversal, lighting, and post-processing. The mesh
renderer runs vertex projection, visibility clear, per-triangle bounding-box
rasterization with atomic depth ownership, perspective-correct normal/material
reconstruction, lighting, and post-processing. The mesh is a heterogeneous world
of arbitrary curved elements rather than a regular proxy grid.

Both rendering programs return `view<srgb8>` from linear `float32x4` pixels.
They never pack display bytes in Tach source. Tach converts those floats to
8-bit sRGB. The browser draws that picture on a canvas; Deno computes the same
bytes offscreen. When the last pixel stage already writes each pixel once,
conversion happens there and the extra full-frame float buffer disappears.

The renderer paths differ only at the real host capability boundary:

- Chromium calls `gpu.present(canvas, view)`, producing the measured frame
  directly on a 1920 x 1080 WebGPU canvas and waiting for completion.
- Deno calls `gpu.submit(view)` followed by `gpu.idle()`, producing the same
  backend-neutral view offscreen through Vulkan because no native surface is
  part of the current runtime.

No graphics pipeline recreates either frame.

## Timing contract

Each backend reuses one adapter and Tach session for its complete run. Each
workload:

1. constructs and uploads resident inputs outside timing;
2. calls `prepare` and completes one untimed warmup;
3. records five timed samples;
4. reports the median and every raw sample;
5. reads only the buffers required for post-timing numerical validation; and
6. destroys its public buffers before constructing the next workload.

For non-display work and native views, one sample is:

```text
gpu.submit(recipe) through gpu.idle()
```

For browser rendering, one sample is:

```text
gpu.present(canvas, view)
```

`present` already waits for the frame, so adding `idle()` would duplicate the
completion boundary. Both forms include per-run host encoding, queue
submission, and GPU completion. They exclude pipeline preparation, initial
allocation/upload, post-timing readback, PNG encoding, report generation, and
validation. The managed JavaScript runtime and, on native, the Deno FFI boundary
remain part of the observation.

Throughput uses work intrinsic to each algorithm:

- complete output pixels per second for procedural rendering;
- measured candidate fragments per second for mesh rendering;
- floating-point operations per second for both matrix precisions;
- stochastic path-steps per second for Monte Carlo;
- attractor interactions per second for particles;
- stencil cell updates per second for the wave field;
- floating-point operations per second for both oscillator precisions.

Renderer FPS is `1000 / median milliseconds`. It is a calculated observation,
not an acceptance threshold.

## Correctness evidence

Validation exists to prevent timings from describing skipped or empty work:

- browser procedural and mesh canvases are captured after timing as exact PNGs;
- mesh coverage and visibility buffers must report nonzero candidate fragments
  and visible pixels on both hosts;
- both matrix variants compare six dispersed cells with direct reference dot
  products over the exact host inputs, including their binary16 quantization;
- Monte Carlo checks finite non-negative payoffs and bounded distribution
  statistics;
- sampled particles must remain finite and inside the simulation volume; and
- sampled wave state must remain finite, active, and stable;
- both oscillator variants must retain finite, nonzero, bounded state and energy
  after the warmup and all timed integration passes.

The math and physics checks read actual GPU output only after all timed samples.
Mesh counters likewise affect reported throughput only after timing. The
procedural Vulkan run proves complete view dispatch/projection and successful
completion; the current public native API intentionally does not expose its
driver-owned packed view scratch for image capture.

## Run

From the repository root:

```sh
npm run compiler
npm run native
npm run build --workspace=@depths/tach
npm run benchmark --workspace=@tach/showcase-ts
```

Static validation without hardware execution is:

```sh
npm run check --workspace=@tach/showcase-ts
```

The build script compiles the Tach project once, copies the complete generated
package locally, bundles the browser entry with Deno, and places the exact
`kernel.wgsl.gz` beside it. Deno imports the same generated package and uses its
SPIR-V plan.

## Reports

Every successful benchmark replaces ignored local reports with:

```text
showcase-ts/reports/gpu.json
showcase-ts/reports/gpu.md
showcase-ts/reports/webgpu-procedural.png
showcase-ts/reports/webgpu-mesh.png
```

JSON is the machine-readable source and contains separate WebGPU and Vulkan host
records. Markdown compares both backends, then records adapter, timing contract,
raw samples, medians, throughput, FPS, validation facts, and workload details.
PNGs are exact post-timing browser canvas captures. Reports describe one local
run and remain ignored by Git.
