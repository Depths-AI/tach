# Tach browser harness

This is a standalone npm project for exercising Tach's generated browser
bindings and WebGPU kernels. It consumes the repository's CLI as an external
executable; it does not contain or expose a Go library.

## Setup

Build the native Tach CLI from the repository root, then install this project's
locked npm dependencies and Chromium:

```sh
go build -o bin/tach .
cd browser-test
npm ci
npm run install:browser
```

On Windows, place `tach.exe` in `..\bin\tach.exe` or set `TACH_BIN` to its
location. The same override works on Unix.

## Commands

- `npm test` compiles every example and always runs the complete interface and
  WebGPU execution suite. Chromium prefers a physical adapter and automatically
  falls back to its bundled CPU-backed SwiftShader adapter when necessary. The
  report labels the run `hardware-accelerated` or `software-emulated` using
  `GPUAdapterInfo.isFallbackAdapter` and includes the adapter identity.
- `npm run start` serves the inspection UI at <http://127.0.0.1:4173> after
  `npm run build:examples`.

SwiftShader processes only trusted local shaders in this harness because
enabling Chromium's software fallback reduces its normal security guarantees.
No command changes between the GPU-less VPS and a GPU-equipped Windows machine:
run `npm test` on both and compare the reported execution mode.

Every run writes a terminal-friendly Markdown summary to `test-report.md` and
the richer Playwright report to `playwright-report/index.html`. Both reports are
generated locally and ignored by Git.

The harness builds generated fixtures into `browser-test/build/`; npm packages,
browser reports, traces, and generated fixtures are intentionally ignored by
Git.
