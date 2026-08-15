# Tach Deno/Vulkan correctness harness

This private npm workspace executes the canonical `examples/` Tach project in
Deno through the same `@depths/tach` import used by browser applications. Host
selection resolves to Tach's packaged Vulkan 1.3 runtime; generated program
imports, commands, buffers, options, and TypeScript declarations remain
identical to the WebGPU harness.

`build.ts` calls `@depths/tach/compiler`, copies the complete generated package
to local ignored `generated/`, and leaves both `kernel.wgsl` and `kernel.spv`
intact. The test validates `kernel.spv` with Khronos `spirv-val` using
`--target-env vulkan1.3`, then `examples.ts` executes all seven exported
programs on the selected physical device.

The semantic corpus checks atomics/shared memory, bitwise behavior, control
flow, loops, vectors and math intrinsics, cross-file struct imports, runtime
arrays, batched parameters, and sequential Tach sessions. Running the whole
corpus twice is deliberate: it proves that closing a logical session releases
its buffers and native Vulkan objects while the process-wide Deno FFI library
remains safe for later sessions.

## Requirements

- Deno;
- Khronos `spirv-val`;
- a Vulkan 1.3 loader and compatible x86-64 Windows or Linux device; and
- the matching Tach native library built into the local `@depths/tach` workspace
  package.

The repository build obtains the native library from the official Vulkan SDK
headers and loader. No third-party JavaScript Vulkan package is involved.

## Run

From the repository root:

```sh
npm ci --ignore-scripts
npm run native
npm test --workspace=@tach/deno-test
```

The test grants Deno only FFI and read access beyond normal npm resolution. It
prints the selected Vulkan adapter after both sessions agree. It generates no
test report: assertion failures and process status are authoritative.

For static validation without hardware execution:

```sh
npm run check --workspace=@tach/deno-test
```

This lints and type-checks the harness against the local package facade.
