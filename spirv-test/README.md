# Tach SPIR-V harness

This native harness exercises the same seven Tach examples and expected logic
as `browser-test`, but consumes Tach's SPIR-V output directly through Vulkan.
For every example it:

1. runs Tach's complete compiler pipeline in memory;
2. validates the emitted module with Khronos `spirv-val` for Vulkan 1.1;
3. creates a native Vulkan compute pipeline from the emitted `.spv`;
4. binds host-visible buffers according to Tach's reflection metadata;
5. dispatches the kernel and asserts the readback values; and
6. fails on messages from `VK_LAYER_KHRONOS_validation` when the layer is
   installed.

The harness prefers discrete, integrated, and virtual GPUs in that order, then
falls back to a CPU Vulkan implementation such as Mesa Lavapipe. Its report
labels the selected path `hardware-accelerated` or `software-emulated` from the
Vulkan device type and known software-driver identities.

## Requirements

- Go 1.23 or newer with CGO enabled
- a C compiler
- a Vulkan 1.1 loader and compute-capable ICD
- Khronos SPIR-V Tools (`spirv-val`)
- Khronos Vulkan validation layers (strongly recommended)

On Ubuntu, the development additions to an existing Mesa Vulkan installation
are:

```sh
sudo apt install build-essential libvulkan1 mesa-vulkan-drivers spirv-tools vulkan-validationlayers
```

On Windows, install the Vulkan SDK and ensure its runtime and tools are on
`PATH`. A hardware Vulkan driver is sufficient; no software fallback is
required when a compatible GPU is available.

## Run

From the repository root:

```sh
go test -count=1 -v ./spirv-test
```

The equivalent root shortcut is `npm run test:spirv`.

Every run writes `spirv-test/test-report.md`, which is ignored by Git and is
readable directly on a headless host. `go test -count=1 ./...` also includes
this harness, so local release validation exercises both native SPIR-V/Vulkan
and browser WGSL/WebGPU execution.
