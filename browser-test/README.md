# Tach browser correctness harness

This private Playwright workspace builds the maintained multi-module
`examples` Tach project once and runs its unified generated JavaScript module
through Chromium WebGPU. It exercises the same
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

Playwright's global setup calls the public project `build()` API once from the
`examples` root. The compiler atomically regenerates `examples/build/` with
one `index.js`, one `index.d.ts`, and one `kernel.wgsl` for every exported
example endpoint. The test server maps that tree to `/build/`; tests import the
single `/build/index.js`, fetch the exact standalone WGSL module, and execute
serially with one Chromium worker.

This arrangement checks more than seven isolated algorithms: it exercises
one-tier project discovery, cross-file imports used by the particle example,
global binding generation, multiple physical shader entries in one WGSL
module, and public-program lookup through one package facade.

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
harness, `examples/build`, or built `@depths/tach` package roots, rejects path
traversal, disables caching, and enables cross-origin isolation headers.
