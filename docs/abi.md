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

Tach defines backend-neutral entry-point names through `src/abi`.

A source kernel named:

```text
integrate
```

is exported to both shader backends as:

```text
__tach_k_integrate
```

Generated reflection records that exact ABI name.

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

The metadata includes:

- ABI/schema version
- source/kernel names
- backend entry-point name
- workgroup dimensions
- resource group/binding
- resource kind/access
- logical layout information
- physical/minimum binding size
- runtime-array tail/stride information where applicable

The generated JS runtime consumes the same compiler data; it does not parse WGSL to recover reflection.

## 8. JavaScript packing

Generated JS uses `DataView`-based packers from the Tach physical layout.

For a struct/resource value the generated code writes fields at compiler-computed byte offsets. Runtime resources derive the final allocation size from the runtime element count and compiler-recorded stride.

The generated binding contract is validated before `Compile` succeeds.

## 9. SPIR-V physical representation

The SPIR-V backend lowers ABI layout into explicit decorations, including member offsets and array stride where required. Resource variables carry descriptor set/binding decorations matching the Tach ABI.

Tach's SPIR-V validator decodes the binary and checks those decorations against the module/type/resource model rather than trusting the emitter's in-memory structures.

## 10. WGSL physical representation

The WGSL backend emits resource declarations/wrappers compatible with Tach's physical binding layout. The generated host packer and WebGPU binding metadata use the same minimum sizes.

This keeps the byte contract compiler-owned across browser and native targets.
