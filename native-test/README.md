# Tach native Vulkan correctness harness

This private workspace asks the same product question as the browser
harness, on the other host: do the eighteen public programs in `examples/`
work from the published `@depths/tach` API in Deno through Tach's Vulkan
1.3 runtime? Application TypeScript is the same. It never chooses a
shader file, a descriptor, or a Vulkan pipeline.

## Build and validation boundary

`build.ts` calls `@depths/tach/compiler` on `examples/` and copies the complete
generated package into ignored `generated/`. Both `kernel.wgsl.gz` and
`kernel.spv` remain part of that singular package even though this harness
executes SPIR-V. Before hardware execution, the test independently validates
the binary with Khronos `spirv-val --target-env vulkan1.3`.

`examples.ts` then imports all eighteen public programs from one generated
`index.js`. Host detection selects the packaged Tach-owned native library and
Vulkan plan. The TypeScript calls, generated declarations, structured values,
buffer handles, recipe objects, and ordering rules are identical to the WebGPU
path.

## What it proves

The language programs check the same everyday surface as the browser harness:
shared counters, 32-bit bit ops, loops with nearest-loop `break` and `continue`,
scalar FP32 and vector FP16 `fma`, binary16 storage and arithmetic, exact
`.length` for direct and struct-tail padding cases, a type imported from another
file, contextual literals and vectors, scalar/vector intrinsic broadcast,
boolean-vector comparison and lane logic, eager selection, `all`/`any`
reduction with exact numeric output,
several recipes on one buffer, and `prepare` before `submit`.

The view programs compute pictures without copying a frame to the CPU:

- `gradient(params)` paints a frame from width, height, and a bias only.
  Tach keeps that picture on the GPU. Any session can run the recipe
  because it holds no buffer.
- `gradientInto(pixels, params)` writes the same gradient into a buffer
  you own, then converts it for display. You can still `read()` the
  linear floats.
- `swatch()` and `swatchInto(pixels)` do the same pair on a 2 x 2 of
  known colors, so the linear source of the fallback can be checked.
- The same `gradient` recipe is reused in a second Tach session, proving
  recipes are not tied to a session the way buffers are.

Each session runs the prepared gradient, both swatches, 32 alternating
gradient recipes, and one last fallback whose source is read only to
confirm the write. The whole corpus runs twice. The reported 72 frames
are offscreen Vulkan results, not a window. Deno has no `present`
surface today. The display bytes are the same math the browser shows;
this host cannot yet put them on a screen.

Running two scoped sessions also proves that closing one releases its buffers
and Vulkan objects while the process-wide Deno FFI library remains safe for the
next session.

## Requirements

- Deno;
- Khronos `spirv-val`;
- a Vulkan 1.3 loader and x86-64 Windows or Linux device with Synchronization2,
  zero-initialized workgroup memory, and the Vulkan memory model. Float16
  examples additionally require shader, storage-buffer, and uniform-buffer
  16-bit support; and
- the matching Tach native library in the local `@depths/tach` workspace.

The repository builds the native library against official Vulkan SDK headers
and loader. No third-party JavaScript Vulkan package is involved.

## Run

From the repository root:

```sh
npm ci --ignore-scripts
npm run native-bindings
npm test --workspace=@tach/native-test
```

The test grants Deno only FFI and read access beyond npm resolution. It prints
the selected physical adapter, eighteen-program count, and 72 projected frames.
Assertion failures and process status are authoritative; no test report is
generated.

For static validation without hardware execution:

```sh
npm run check --workspace=@tach/native-test
```

This lints and type-checks the standalone harness against the local package
facade.
