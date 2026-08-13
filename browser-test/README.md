# Tach browser correctness harness

This private Playwright workspace compiles every maintained example and runs
its generated JavaScript module through Chromium WebGPU. It exercises the same
`@depths/tach` package surface an application uses; it does not call Go from
the browser or maintain handwritten shader fixtures.

The seven examples cover atomics/shared memory, bitwise operations, structured
control, `for` lowering, math intrinsics and vectors, structs, and scalar
runtime arrays. Each case checks:

- generated WGSL reports no Chromium compilation errors;
- execution produces the expected typed data;
- no uncaptured WebGPU error occurs; and
- buffers, commands, parameters, submission, and scoped cleanup use the public
  runtime contract.

An additional case submits commands with different parameter values in one
submission and across submissions to check parameter isolation and ordering.

## Setup

From the repository root:

```sh
npm ci
npm run compiler
npm run install:browser --workspace=@tach/browser-test
```

The package compiler resolver finds `dist/tach` or `dist/tach.exe` in this
development checkout. `TACH_BIN` can select an explicit executable.

## Run

```sh
npm test --workspace=@tach/browser-test
```

Playwright's global setup deletes and regenerates `browser-test/build/` by
calling the public Node compiler API once for each `examples/*.tach` file. The
test then starts the local server, loads those generated `.js` and `.wgsl`
artifacts, and executes serially with one Chromium worker.

Chromium is launched headless with WebGPU enabled. Hardware is preferred;
`--enable-unsafe-swiftshader` permits Chromium's software implementation on a
GPU-less host. Software fallback is appropriate here only because all shaders
are trusted local compiler output. The first case logs whether adapter metadata
looks hardware-accelerated or software-emulated.

For an interactive browser window:

```sh
npm run test:headed --workspace=@tach/browser-test
```

The harness uses Playwright's line reporter and leaves no custom Markdown
report. Failures, traces, and standard Playwright artifacts use the workspace's
ignored output directories.

## Inspect generated examples

Build fixtures and run the static server without the test runner:

```sh
npm run build:examples --workspace=@tach/browser-test
npm run start --workspace=@tach/browser-test
```

The server listens at <http://127.0.0.1:4173>. It serves only files beneath the
harness or built `@depths/tach` package roots, rejects path traversal, disables
caching, and enables cross-origin isolation headers.
