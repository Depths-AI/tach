# Tach browser correctness harness

This private npm workspace proves that the canonical `examples/` Tach project
works through the public `@depths/tach` API in Chromium WebGPU. It is a
self-contained Deno harness: it owns project generation, bundling, its
loopback-only server, Chrome launch, DevTools Protocol client, GPU assertions,
canvas capture, and cleanup. It shares no runner or shader fixture with the
native harness.

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

The nine canonical programs cover:

- shared workgroup memory and integer atomics;
- shifts, masks, complements, and unsigned behavior;
- branches, loops, compound assignment, vectors, and math intrinsics;
- direct imports, project-global types, structs, and orchestration;
- runtime-array launch inference, batching, repeat, and parameter isolation;
- explicit `prepare` followed by submission;
- a scalar-only `view<srgb8>` whose final transient writer is fused with
  texture projection; and
- a caller-owned pixel-buffer view using standalone projection.

One scoped session first checks exact integer results and tolerance-bounded
floating results through decoded buffer readback. It then prepares and submits
the owner-neutral scalar view offscreen, presents the caller-owned fallback,
and verifies that its linear float buffer was actually written.

The sustained display seam constructs 32 CPU-selected recipes, alternating
the scalar/transient and caller-owned forms, and invokes `present` concurrently
on one 128 x 72 canvas. Session serialization preserves call order;
completion-backed presentation prevents unbounded GPU queueing. The final
canvas is captured as PNG and must be non-empty. Together with the initial
fallback presentation, success reports 33 displayed frames and nine public
programs.

This distinguishes three contracts that are easy to conflate:

```text
generated call       opaque recipe, no execution
submit(view)          offscreen GPU projection, no CPU readback
present(canvas, view) full GPU recipe, direct canvas output, frame completion
```

## Browser runner

`test.ts` launches an installed Chrome or Chromium with WebGPU enabled, opens
the local page, and polls one promise through the DevTools Protocol. It has no
browser-test framework or bundler dependency. Set `CHROME_BIN` only when the
browser is outside the standard platform locations.

The runner requires exactly nine programs, 33 presented frames, and a
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
