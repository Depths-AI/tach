# Tach resource and host ABI

Tach owns one resource ABI shared by WGSL/WebGPU and SPIR-V/Vulkan output.

## 1. Resource identity

Each compute parameter becomes a resource record with:

```text
kind       uniform | storage
access     read | read_write
group      u32
binding    u32
logical type
physical binding layout
```

Bindings may be specified in source:

```tach
@group(0) @binding(3) data: storage<Data[], read_write>
```

or deterministically allocated from the module-global binding space by the compiler.

The mapping is direct:

```text
group   -> WGSL @group     -> SPIR-V DescriptorSet
binding -> WGSL @binding   -> SPIR-V Binding
```

## 2. Entry-point naming

Exported kernel names preserve their source spelling across Tach, WGSL,
SPIR-V, reflection, JavaScript, and TypeScript. A source kernel named
`integrate` is therefore an `integrate` shader entry point and an exported
`integrate` JavaScript function. Compiler-private names may be transformed,
but no private naming scheme leaks into the public contract.

## 3. Logical vs physical types

A logical Tach type is never rewritten to contain artificial padding fields.

For example:

```tach
type Particle = {
  position: vec3f,
  mass: f32,
  velocity: vec3f,
};
```

remains that logical shape to semantic analysis and optimization.

`src/layout` separately computes its physical representation:

```text
Particle
  align  = 16
  size   = 32
  position offset = 0
  mass     offset = 12
  velocity offset = 16
```

That physical layout then drives both shader backends and host packers.

## 4. Scalar/vector layout

Current portable ABI layout:

| Type | Size | Alignment |
|---|---:|---:|
| `i32` | 4 | 4 |
| `u32` | 4 | 4 |
| `f32` | 4 | 4 |
| `vec2<T>` | 8 | 8 |
| `vec3<T>` | 12 | 16 |
| `vec4<T>` | 16 | 16 |

`T` is a 32-bit numeric scalar.

`bool` has no host ABI representation.

Struct alignment has a 16-byte minimum. Struct size is rounded up to its alignment. Nested struct placement reserves the full aligned nested extent.

## 5. Arrays

Fixed-array stride is:

```text
roundUp(elementAlignment, elementSize)
```

Runtime storage arrays use the same natural element stride and are permitted only as a final struct tail when nested.

A runtime-tail layout records:

```text
fixed prefix size
runtime member offset
element alignment
element stride
```

Generated host packing requires enough bytes for at least the ABI minimum binding size.

## 6. Resource wrappers

Host-visible fixed-size resource bindings use a physical wrapper whose first field has at least 16-byte alignment. This makes a scalar uniform such as `uniform<f32>` have:

```text
logical value size  = 4
physical binding size = 16
```

Reflection deliberately records both concepts.

## 7. Reflection metadata

Every build emits `<name>.tach.json` with compiler-owned metadata describing the generated program.

The metadata is deliberately plain and unversioned while Tach is under active
development. It includes:

- source kernel name and identical backend entry-point name
- workgroup dimensions
- resource group/binding
- resource kind/access
- logical layout information
- physical/minimum binding size
- runtime-array tail/stride information where applicable

The generated JS module passes the same compiler data to `@depths/tach`; the
runtime never parses WGSL to recover reflection.

## 8. JavaScript and TypeScript bindings

The generated public interface mirrors Tach source directly:

```ts
import { tach } from "@depths/tach";
import { integrate, type Particle } from "./particles.js";

const result = await tach(async (gpu) => {
  const particles = gpu.buffer(initialParticles);
  await integrate(particles, { dt: 0.5, count: initialParticles.length });
  return particles.read();
});
```

Named Tach structs become TypeScript interfaces. Exported kernels become
same-named positional functions. Storage parameters become the single
`ComputeBuffer<T>` host abstraction; uniforms remain ordinary typed values.
The enclosing `tach(...)` scope owns its adapter, device, buffers, queued work,
and cleanup, so a kernel call needs no module/context object or synthetic device
parameter. The scope resolves to either `{ ok: true, value }` or `{ ok: false,
error }`; WebGPU, compiler, lifecycle, and application failures therefore have
one explicit error-as-data boundary.

Buffers are lazy: their first kernel use supplies the exact compiler layout,
at which point `@depths/tach` packs the value and creates the physical WebGPU
buffer. Pipelines and bind-group layouts are also created lazily and cached per
device. Calls infer their logical invocation count from the first runtime-sized
storage buffer and accept an optional final size only when an explicit count or
`[x, y, z]` is required.

Packing and unpacking remain private implementation details. They use
`DataView`, compiler-computed byte offsets, and compiler-recorded runtime
strides. Generated JavaScript imports only `@depths/tach/internal` and exports
only the source kernels; its declarations import `ComputeBuffer` from
`@depths/tach`. The generated binding contract is validated before `Compile`
succeeds.

## 9. SPIR-V physical representation

The SPIR-V backend lowers host-visible ABI layout into explicit decorations,
including member offsets and array stride where required. Resource variables
carry descriptor set/binding decorations matching the Tach ABI. Logical SSA
types and Workgroup-memory types remain undecorated; aggregate resource
loads/stores are lowered field-by-field so physical buffer types never enter the
logical value domain.

Tach's SPIR-V validator decodes the binary and checks both directions of this
rule: descriptor-reachable aggregates require the exact Tach ABI layout, while
Workgroup-reachable aggregates must not carry explicit layout decorations.

## 10. WGSL physical representation

The WGSL backend emits resource declarations/wrappers compatible with Tach's physical binding layout. The generated host packer and WebGPU binding metadata use the same minimum sizes.

This keeps the byte contract compiler-owned across browser and native targets.
