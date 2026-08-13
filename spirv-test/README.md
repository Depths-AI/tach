# Tach SPIR-V/Vulkan correctness harness

This native Go harness compiles the same seven maintained examples as
`browser-test`, consumes schema-1 SPIR-V target plans, and executes them through
Vulkan. It validates the complete native-host contract rather than assuming a
public source function maps directly to one shader entry.

For every example it:

1. compiles the `.tach` source for both targets in memory;
2. runs Khronos `spirv-val --target-env vulkan1.1` on emitted SPIR-V;
3. decodes Tach metadata and selects the named public program plan;
4. allocates external resources and transient scratch colors;
5. creates every referenced private physical pipeline and descriptor layout;
6. evaluates shapes and packs parameter blocks from recorded value sources;
7. records dispatches, barriers, and repeat behavior in one command buffer;
8. executes through Vulkan and compares every expected readback value; and
9. fails on new `VK_LAYER_KHRONOS_validation` messages when that layer is
   available.

The harness prefers discrete, integrated, and virtual GPUs in that order, then
accepts a CPU Vulkan implementation such as Mesa Lavapipe. Its report labels
the selected device `hardware-accelerated` or `software-emulated` from Vulkan
device type and known software-driver identities.

## Requirements

- Go 1.26.5 with CGO enabled;
- a C compiler;
- a Vulkan 1.1 loader and compute-capable ICD;
- Khronos SPIR-V Tools (`spirv-val`); and
- Khronos Vulkan validation layers, strongly recommended.

On Ubuntu with Mesa:

```sh
sudo apt install build-essential libvulkan1 mesa-vulkan-drivers spirv-tools vulkan-validationlayers
```

On Windows, install the Vulkan SDK and ensure its runtime, headers, libraries,
validation layer, and tools are discoverable. A compatible hardware driver is
sufficient; no software ICD is required when a GPU is available.

## Run

From the repository root:

```sh
go test -count=1 -v ./spirv-test
```

The equivalent npm shortcut is:

```sh
npm run test:spirv
```

`go test -count=1 ./...` also includes this native harness and therefore has
the same Vulkan/CGO/`spirv-val` requirements.

Every run writes ignored `spirv-test/test-report.md`, including:

- pass/fail counts and elapsed time;
- host OS/architecture and Go version;
- `spirv-val` version and target environment;
- Vulkan device identity, API version, mode, and vendor/device IDs;
- validation-layer availability; and
- per-example SPIR-V byte size, duration, and failure detail.

The report is written even when setup or execution fails, making it suitable
for a headless machine where the terminal log is ephemeral.
