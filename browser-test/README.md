# Tach browser harness

This private npm workspace exercises Tach's generated direct JavaScript
functions, TypeScript-facing data contract, `@depths/tach` lifecycle, and
WebGPU kernels. It consumes the exact same package surface as an application;
it does not contain or expose a Go library.

## Setup

Install the root workspace, build the native development compiler, and install
Chromium:

```sh
npm ci
npm run compiler
npm run install:browser --workspace=@tach/browser-test
```

`@depths/tach/compiler` resolves `dist/tach` in this repository. `TACH_BIN`
selects an explicit compiler on any operating system.

## Commands

- `npm test --workspace=@tach/browser-test` compiles every example and always
  runs the complete interface and
  WebGPU execution suite. Chromium prefers a physical adapter and automatically
  falls back to its bundled CPU-backed SwiftShader adapter when necessary. The
  report labels the run `hardware-accelerated` or `software-emulated` using
  `GPUAdapterInfo.isFallbackAdapter` plus known software-adapter identities.
- `npm run start --workspace=@tach/browser-test` serves the inspection UI at
  <http://127.0.0.1:4173> after
  `npm run build:examples --workspace=@tach/browser-test`.

SwiftShader processes only trusted local shaders in this harness because
enabling Chromium's software fallback reduces its normal security guarantees.
No command changes between the GPU-less VPS and a GPU-equipped Windows machine:
run the workspace test on both and compare the reported execution mode.

Every run writes a terminal-friendly Markdown summary to `test-report.md` and
the richer Playwright report to `playwright-report/index.html`. Both reports are
generated locally and ignored by Git.

The harness builds generated fixtures into `browser-test/build/`; npm packages,
browser reports, traces, and generated fixtures are intentionally ignored by
Git.
