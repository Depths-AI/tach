# Tach large GPU showcase

This standalone npm workspace measures six fixed, sustained GPU workloads from
one generated Tach project package through both Chromium/WebGPU and
Deno/Vulkan 1.3. It has one workload profile and no performance verdicts. The
harness records what each selected adapter did.

Each workload lives in its own `.tach` kernel file. `kernels/mesh` and
`kernels/procedural` directly import shared color helpers from
`shared/color`, exercising project imports in addition to six independent
public programs. One project build merges all seven files into one
JavaScript/TypeScript facade, one WGSL module, and one SPIR-V 1.6 module. One
host-neutral workload implementation imports all six public programs from
`build/index.js`; the browser and Deno runners each execute it in one
persistent Tach session through the same `tach` API.

## Workloads

| Category | Kernel | Fixed work |
|---|---|---|
| Rendering | `procedural.tach` | 1920 x 1080 analytic scene; up to 72 traversal steps, eight soft-shadow samples, three ambient-occlusion samples, material lighting, fog, and spatial post-processing |
| Rendering | `mesh.tach` | 1920 x 1080 compute-only renderer over terrain and 244 torus-knot, twisted-ribbon, superquadric, and organic elements: 162,481 vertices and 312,064 triangles |
| Mathematics | `matrix.tach` | 2048 x 2048 dense matrix multiplication with 16 x 16 cooperative shared-memory tiles: 17,179,869,184 floating-point operations |
| Mathematics | `monte-carlo.tach` | 1,048,576 geometric Brownian paths, each evolved through 64 Box-Muller Gaussian time steps |
| Physics | `particles.tach` | 2,097,152 particles, 64 integration steps, and four moving softened inverse-square attractors per step |
| Physics | `wave.tach` | 2048 x 2048 height/velocity grid advanced through 64 five-point stencil steps |

The procedural renderer dispatches trace, lighting, and post-processing
stages. The mesh renderer dispatches vertex projection, visibility clear,
per-triangle bounding-box rasterization with atomic depth ownership,
perspective-correct reconstruction of smooth normals, material identity and
roughness, heterogeneous lighting, and post-processing. Both frames are
produced entirely by compute work; no browser graphics pipeline recreates
their contents.

The mesh report includes exact GPU-produced coverage counters: total candidate
fragments, candidates per output and visible pixel, and visible coverage. It
does not infer overdraw from triangle count.

## Timing

Each backend reuses one adapter and Tach session for its complete run. Each
workload:

1. allocates and uploads its resident inputs outside timing;
2. completes one untimed warmup;
3. records five independent samples;
4. reports the median and all raw samples;
5. reads back and validates output after timing; and
6. destroys its buffers before the next workload is constructed.

Every sample measures:

```text
gpu.submit(command) through gpu.idle()
```

This includes command preparation, backend encoding, queue submission, and GPU
completion. It excludes initial allocation/upload, output readback, PNG
encoding, validation, and report writing. The managed runtime and respective
browser or native FFI boundary remain part of each observation.

Throughput uses work intrinsic to each algorithm:

- complete pixels per second for procedural rendering;
- covered candidate fragments per second for mesh rendering;
- floating-point multiply/add operations per second for matrix algebra;
- stochastic path-steps per second for Monte Carlo;
- attractor interactions per second for particles; and
- stencil cell updates per second for the wave field.

FPS is `1000 / median milliseconds` for the two complete-frame renderers. It
is a calculated rate, not an acceptance threshold.

## Validation

Validation protects the measurements from empty or broken work without timing
a second implementation:

- both renderers require opaque, spatially varied 1080p output;
- matrix multiplication compares six dispersed cells with direct reference
  dot products;
- Monte Carlo checks finite non-negative payoffs and bounded distribution
  statistics;
- particle samples must remain finite and inside the simulation volume; and
- wave samples must remain finite, active, and stable.

The renderer readbacks are copied byte-for-byte into PNGs after timing. The
math and physics checks read actual GPU output after all timed samples.

## Run

From the repository root:

```sh
npm run compiler
npm run native
npm run build --workspace=@depths/tach
npm run benchmark --workspace=@tach/showcase-ts
```

`npm run check --workspace=@tach/showcase-ts` lints and type-checks both host
entries and the shared workload implementation without running hardware work.
The build script compiles the Tach project once, copies the complete package
locally, bundles the browser entry with Deno, and places the exact WGSL beside
it. Deno directly imports the same generated package and uses its SPIR-V.

## Reports

Every successful benchmark writes:

```text
showcase-ts/reports/gpu.json
showcase-ts/reports/gpu.md
showcase-ts/reports/webgpu-procedural.png
showcase-ts/reports/webgpu-mesh.png
showcase-ts/reports/vulkan-procedural.png
showcase-ts/reports/vulkan-mesh.png
```

JSON is the machine-readable source and contains separate host records.
Markdown puts both backends in one comparison table and then records their
adapter, timing contract, raw samples, medians, throughput, FPS, validation,
and full workload details. Each PNG is the exact post-timing readback from its
named backend. Reports are local observations and ignored by Git.
