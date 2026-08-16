# Tach Deno/Vulkan correctness harness

This private npm workspace proves that the canonical `examples/` Tach project
works in Deno through Tach's Vulkan 1.3 host. It imports the same generated
functions and `tach` API as the browser harness. Application code never
selects shader files, descriptor bindings, memory types, or Vulkan pipelines.

## Build and validation boundary

`build.ts` calls `@depths/tach/compiler` on `examples/` and copies the complete
generated package into ignored `generated/`. Both `kernel.wgsl.gz` and
`kernel.spv` remain part of that singular package even though this harness
executes SPIR-V. Before hardware execution, the test independently validates
the binary with Khronos `spirv-val --target-env vulkan1.3`.

`examples.ts` then imports all nine public programs from one generated
`index.js`. Host detection selects the packaged Tach-owned native library and
Vulkan plan. The TypeScript calls, generated declarations, structured values,
buffer handles, recipe objects, and ordering rules are identical to the WebGPU
path.

## What it proves

The ordinary language corpus checks:

- workgroup memory and atomics;
- bitwise and unsigned semantics;
- branches, loops, compound assignment, vectors, and intrinsics;
- cross-file imports, struct layout, and explicit orchestration;
- runtime arrays, batching, repeat, and distinct parameter blocks; and
- eager `prepare` followed by real execution.

The view corpus checks both physical plans without copying a frame to the CPU:

- `gradient(params)` has no public buffer. Its transient final frame is folded
  into packed RGBA8 output, proving scalar-only procedural recipes;
- `gradientInto(pixels, params)` writes a session-owned float buffer, then uses
  standalone projection; and
- one scalar `ComputeView` recipe is reused across two separate Tach sessions,
  proving that recipes are owner-neutral while buffers remain session-owned.

Each session executes one prepared scalar view, a batch of 32 alternating
fused/fallback views, and one final fallback whose source buffer is read only
to verify its write. The complete language and view corpus runs twice. The
reported 68 projected frames are offscreen Vulkan compute results, not native
surface presentation; Deno intentionally has no `present` surface today.

Running two scoped sessions also proves that closing one releases its buffers
and Vulkan objects while the process-wide Deno FFI library remains safe for the
next session.

## Requirements

- Deno;
- Khronos `spirv-val`;
- a Vulkan 1.3 loader and compatible x86-64 Windows or Linux device; and
- the matching Tach native library in the local `@depths/tach` workspace.

The repository builds the native library against official Vulkan SDK headers
and loader. No third-party JavaScript Vulkan package is involved.

## Run

From the repository root:

```sh
npm ci --ignore-scripts
npm run native
npm test --workspace=@tach/deno-test
```

The test grants Deno only FFI and read access beyond npm resolution. It prints
the selected physical adapter, nine-program count, and 68 projected frames.
Assertion failures and process status are authoritative; no test report is
generated.

For static validation without hardware execution:

```sh
npm run check --workspace=@tach/deno-test
```

This lints and type-checks the standalone harness against the local package
facade.
