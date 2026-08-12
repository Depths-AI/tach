# Tach TypeScript performance showcase

This private Vite workspace compares Tach-generated WebGPU kernels with the
same algorithms written in single-threaded TypeScript over typed arrays. It is
both an executable performance demonstration and a stress consumer of the
public `@depths/tach` binding contract.

The suite covers five different compute shapes:

- particle integration fuses repeated steps while keeping state register-resident;
- Mandelbrot escape uses a two-dimensional dispatch and divergent loops;
- matrix multiplication tiles through workgroup memory and barriers;
- Black-Scholes pricing exercises a generated helper plus `log`, `exp`,
  `sqrt`, and branching; and
- procedural composition calculates and packs every RGBA pixel of a full-HD
  scene from gradients, signed-distance shapes, stars, terrain, and a
  perspective grid.

Each generated kernel constructs a `ComputeCommand` and accepts one optional
`LaunchOptions` object. `size` sets the logical invocation dimensions;
`repeat` records repeated executions of that command inside
one compute pass and queue submission:

```ts
const count = 1 << 20;
const positions = gpu.buffer(new Float32Array(count));
const velocities = gpu.buffer(new Float32Array(count));

await gpu.submit(integrateParticles(
  positions,
  velocities,
  { dt: 0.001, count, steps: 128 },
  { size: count },
));
await gpu.idle();
```

`steps` is an ordinary suffix-free Tach loop. The target-neutral optimizer
recognizes its safe repeated buffer update, loads position and velocity only on
the first executed iteration, carries them and the already-resident `dt` value
as SSA values, and commits the position once after the loop. A zero-iteration
loop performs no memory access. The loop itself remains source-visible because replacing ordered
dispatches with an invocation-local loop is invalid for kernels with
cross-invocation reads, atomics, or barriers; `repeat` retains its literal
ordered-dispatch meaning.

`submit()` itself stops at queue submission. The explicit `idle()` is part of
this benchmark because a wall-clock sample needs completed GPU work; a frame
loop deliberately omits it and submits the next frame through the same
long-lived session.

Scalar runtime arrays accept `Float32Array`, `Uint32Array`, and `Int32Array`
directly. The runtime preserves that representation on readback and transfers
their backing bytes without an element-by-element JavaScript packing pass.

## Measurement contract

Before timing, the harness completes native compilation, WGSL module and
pipeline creation, the initial buffer upload, parameter-arena creation,
bind-group creation, and warmup. Each GPU sample then measures one
application-visible `submit()` plus `idle()` pair, including command encoding,
submission, and `queue.onSubmittedWorkDone()`.

Both full and GPU-only modes run the same five-sample GPU phase. All five GPU
workloads and readbacks finish before full mode begins any TypeScript baseline,
so CPU heating cannot bias later measurements on a shared CPU/iGPU package.

The TypeScript baseline performs the same full repeated workload on the main
thread. Both sides run five samples in the full profile and report medians. The
comparison is intentionally against ordinary TypeScript, not native SIMD,
WebAssembly, a worker pool, or a vendor math library.

The full profile runs:

| Workload | Timed batch |
| --- | --- |
| Particle integration | 1,048,576 scalar components, 128 steps |
| Mandelbrot escape | 768 x 768 pixels, limit 192, 4 dispatches |
| Tiled matrix multiply | 256 x 256 matrices, 4 dispatches |
| Black-Scholes pricing | 1,048,576 options, 8 dispatches |
| Procedural RGBA composition | 1920 x 1080 pixels, 3 dispatches |

## Commands and reports

From the repository root:

```sh
npm ci
npm run compiler
npm run install:browser --workspace=@tach/showcase-ts
npm run benchmark:gpu --workspace=@tach/showcase-ts
npm run benchmark --workspace=@tach/showcase-ts
```

`benchmark:gpu` runs that identical full-size GPU phase but skips every
TypeScript reference workload and correctness comparison. It writes
the ignored `showcase-ts/test-report.md`, so rapid compiler iteration does not
overwrite the tracked comparison baseline.

`benchmark` builds the generated module and production site, runs the full
hardware benchmark in headless Chromium, verifies every GPU result against its
TypeScript counterpart, and writes `showcase-ts/benchmark-report.md`. The
Markdown report records the adapter, workload sizes, dispatch counts, medians,
speedups, throughput, and correctness results. The full report is tracked as a
hardware baseline so later optimization work can be compared directly.

Every Playwright run also exports the procedural canvas from its exact
1920-by-1080 backing store as `procedural-scene.png` and attaches it to the
Playwright result. The PNG and HTML report are ignored local artifacts: use
them for qualitative inspection, while the per-pixel comparison remains the
correctness assertion.

`npm test --workspace=@tach/showcase-ts` runs a smaller three-sample profile for
regression testing and writes the same report shape to the ignored
`showcase-ts/test-report.md`, leaving the tracked full benchmark untouched.
`npm run dev --workspace=@tach/showcase-ts` opens the full interactive
showcase. Results are hardware-, driver-, browser-, power-, and thermal-state
dependent.
