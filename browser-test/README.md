# Tach browser correctness harness

This private workspace asks a product question: do the sixteen public
programs in `examples/` work from the published `@depths/tach` API in
real Chrome WebGPU? It is not a unit-test framework. A Deno script
builds the example project, serves one page, launches Chromium, and
checks results the way an application would: generated functions,
`submit`, `present`, buffer `read()`, and a PNG of the canvas.

It shares no runner or handwritten shader with the native harness. Both
harnesses import the same generated `index.js`.

## Build boundary

`scripts/build.ts` calls `@depths/tach/compiler` on `examples/`, copies the
complete generated package into ignored `generated/`, bundles `src/main.ts`,
and places the exact `kernel.wgsl.gz` beside the browser module. The browser
imports every public endpoint through the generated project `index.js` and the
runtime through `@depths/tach`; it never imports compiler internals or
handwritten WGSL.

The local server exposes only the page module and compressed generated shader.
The WebGPU driver decompresses WGSL, validates schema-2 execution metadata,
creates pipelines, and executes the same recipe facade used by Deno/Vulkan.

## What it proves

The sixteen canonical programs cover the language a TypeScript app will
actually call. In everyday terms:

- workers sharing a counter;
- 32-bit shifts and masks;
- loops, branches, nearest-loop `break` and `continue`, vectors, and math;
- scalar FP32 and component-wise vector FP16 `fma` execution;
- binary16 storage, parameters, arithmetic, readback through `Float16Array`,
  and exact `.length` for both direct and struct-tail padding cases;
- a type imported from another file;
- several recipes on one buffer, including `prepare` then `submit`;
- a picture the program paints itself, drawn in one step;
- the same picture written into a buffer you still own, then drawn;
- a 2 x 2 of known colors on both of those paths, decoded from the
  presented PNG with `colorSpaceConversion: "none"` so the 8-bit
  channels can be checked exactly.

One scoped session first checks exact integer results and tolerance-bounded
floating results through decoded buffer readback. It then prepares and submits
the owner-neutral scalar view offscreen, presents both swatches, presents the
caller-owned gradient fallback, and verifies that each caller-owned linear
float buffer was actually written.

The sustained display seam constructs 32 CPU-selected recipes, alternating
the scalar/transient and caller-owned forms, and invokes `present` concurrently
on one 128 x 72 canvas. Session serialization preserves call order;
completion-backed presentation prevents unbounded GPU queueing. The final
canvas is captured as PNG and must be non-empty. Together with the two swatch
frames and the initial gradient fallback, success reports 35 displayed frames
and sixteen public programs.

This distinguishes three contracts that are easy to conflate:

```text
generated call       describe the picture, do not run it
submit(view)          compute it, leave it on the GPU
present(canvas, view) compute it onto this canvas, then wait
```

## Browser runner

`test.ts` launches an installed Chrome or Chromium with WebGPU enabled, opens
the local page, and polls one promise through the DevTools Protocol. It has no
browser-test framework or bundler dependency. Set `CHROME_BIN` only when the
browser is outside the standard platform locations.

The runner requires exactly sixteen programs, 35 presented frames, and a
non-trivial PNG. It prints the selected adapter and those counts, then closes
Chrome, aborts the server, and removes its temporary profile even after
failure. Process status and assertion diagnostics are the test contract; the
harness writes no report Markdown.

## Run

From the repository root:

```sh
npm ci --ignore-scripts
npm test --workspace=@tach/browser-test
```

The workspace consumes local `@depths/tach` through the root npm workspace.
`npm run build` must have produced `dist/tach.exe` or `dist/tach`;
alternatively `TACH_BIN` may identify the exactly matching compiler.

Useful narrower commands are:

```sh
npm run build --workspace=@tach/browser-test
npm run check --workspace=@tach/browser-test
```

`check` lints and type-checks the standalone harness. `test` rebuilds the
canonical example project and performs real WebGPU execution and presentation.
