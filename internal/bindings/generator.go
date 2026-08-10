package bindings

import (
	"encoding/json"
	"fmt"
	"sort"
	"strings"
	"unicode"

	"pine/internal/abi"
	"pine/internal/ir"
	"pine/internal/layout"
	"pine/internal/types"
)

const metadataFormat = "pine.module.v1"

type Artifacts struct {
	JavaScript   string
	Declarations string
	MetadataJSON []byte
}

type Metadata struct {
	Format    string         `json:"format"`
	ABI       ABI            `json:"abi"`
	Types     []TypeMetadata `json:"types"`
	Resources []ResourceMeta `json:"resources"`
	Kernels   []KernelMeta   `json:"kernels"`
}

type ABI struct {
	Layout     string `json:"layout"`
	Endianness string `json:"endianness"`
}

type TypeMetadata struct {
	Name    string      `json:"name"`
	Size    uint32      `json:"size"`
	Align   uint32      `json:"align"`
	Runtime bool        `json:"runtime"`
	Fields  []FieldMeta `json:"fields"`
}

type FieldMeta struct {
	Name   string `json:"name"`
	Type   string `json:"type"`
	Offset uint32 `json:"offset"`
}

type ResourceMeta struct {
	Index          int    `json:"index"`
	Name           string `json:"name"`
	Group          uint32 `json:"group"`
	Binding        uint32 `json:"binding"`
	Kind           string `json:"kind"`
	Access         string `json:"access"`
	Type           string `json:"type"`
	ValueSize      uint32 `json:"valueSize"`
	BindingSize    uint32 `json:"bindingSize,omitempty"`
	Align          uint32 `json:"align"`
	Runtime        bool   `json:"runtime"`
	RuntimeOffset  uint32 `json:"runtimeOffset,omitempty"`
	RuntimeStride  uint32 `json:"runtimeStride,omitempty"`
	MinBindingSize uint32 `json:"minBindingSize"`
}

type KernelMeta struct {
	Name          string               `json:"name"`
	EntryPoint    string               `json:"entryPoint"`
	WorkgroupSize [3]uint32            `json:"workgroupSize"`
	Resources     []KernelResourceMeta `json:"resources"`
}

type KernelResourceMeta struct {
	Param    string `json:"param"`
	Resource int    `json:"resource"`
	Group    uint32 `json:"group"`
	Binding  uint32 `json:"binding"`
}

// Generate creates browser-native JavaScript bindings, TypeScript declarations,
// and machine-readable ABI metadata directly from verified Pine IR. Reflection
// is never reconstructed from WGSL.
func Generate(m *ir.Module, wgslSource string) (*Artifacts, error) {
	if err := ir.Verify(m); err != nil {
		return nil, err
	}
	meta, err := buildMetadata(m)
	if err != nil {
		return nil, err
	}
	metaJSON, err := json.MarshalIndent(meta, "", "  ")
	if err != nil {
		return nil, err
	}
	js, err := emitJavaScript(m, wgslSource, meta, metaJSON)
	if err != nil {
		return nil, err
	}
	dts, err := emitDeclarations(m, meta)
	if err != nil {
		return nil, err
	}
	if err := ValidateGenerated(js, dts, metaJSON); err != nil {
		return nil, fmt.Errorf("Pine binding self-validation failed: %w", err)
	}
	return &Artifacts{JavaScript: js, Declarations: dts, MetadataJSON: append(metaJSON, '\n')}, nil
}

func buildMetadata(m *ir.Module) (*Metadata, error) {
	out := &Metadata{
		Format: metadataFormat,
		ABI:    ABI{Layout: "pine-portable-v1", Endianness: "little"},
		Types:  []TypeMetadata{}, Resources: []ResourceMeta{}, Kernels: []KernelMeta{},
	}
	for _, t := range m.Structs {
		l, err := layout.Of(t)
		if err != nil {
			return nil, err
		}
		tm := TypeMetadata{Name: t.Name, Size: l.Size, Align: l.Align, Runtime: l.Runtime}
		for i, f := range t.Fields {
			tm.Fields = append(tm.Fields, FieldMeta{Name: f.Name, Type: f.Type.String(), Offset: l.Fields[i].Offset})
		}
		out.Types = append(out.Types, tm)
	}
	for i, r := range m.Resources {
		l, err := layout.Of(r.Type)
		if err != nil {
			return nil, fmt.Errorf("resource %s: %w", r.Name, err)
		}
		rm := ResourceMeta{Index: i, Name: r.Name, Group: r.Group, Binding: r.Binding, Type: r.Type.String(), ValueSize: l.Size, Align: l.Align, Runtime: l.Runtime}
		if r.Kind == ir.Uniform {
			rm.Kind = "uniform"
		} else {
			rm.Kind = "storage"
		}
		if r.Access == ir.ReadWrite {
			rm.Access = "read_write"
		} else {
			rm.Access = "read"
		}
		if l.Runtime {
			off, stride, err := runtimeTail(r.Type)
			if err != nil {
				return nil, err
			}
			rm.RuntimeOffset, rm.RuntimeStride = off, stride
			// WebGPU runtime arrays always have at least one element at dispatch.
			rm.MinBindingSize = off + stride
		} else {
			// WGSL resources are emitted through a compiler-owned wrapper struct
			// whose first member has @align(16). For fixed-size logical values the
			// wrapper's physical store size is therefore rounded to 16 bytes.
			rm.BindingSize = roundUp(16, l.Size)
			rm.MinBindingSize = rm.BindingSize
		}
		out.Resources = append(out.Resources, rm)
	}
	for _, f := range m.Functions {
		if !f.Compute {
			continue
		}
		km := KernelMeta{Name: f.Name, EntryPoint: abi.KernelEntry(f.Name), WorkgroupSize: f.Workgroup}
		seen := map[[2]uint32]string{}
		for _, rp := range f.ResourceParams {
			if rp.Resource < 0 || rp.Resource >= len(m.Resources) {
				return nil, fmt.Errorf("kernel %s resource index out of range", f.Name)
			}
			r := m.Resources[rp.Resource]
			key := [2]uint32{r.Group, r.Binding}
			if prev, ok := seen[key]; ok {
				return nil, fmt.Errorf("kernel %s parameters %s and %s alias group=%d binding=%d", f.Name, prev, rp.Name, r.Group, r.Binding)
			}
			seen[key] = rp.Name
			km.Resources = append(km.Resources, KernelResourceMeta{Param: rp.Name, Resource: rp.Resource, Group: r.Group, Binding: r.Binding})
		}
		out.Kernels = append(out.Kernels, km)
	}
	return out, nil
}

func roundUp(align, n uint32) uint32 {
	if align == 0 {
		return n
	}
	return (n + align - 1) / align * align
}

func runtimeTail(t *types.Type) (offset, stride uint32, err error) {
	l, err := layout.Of(t)
	if err != nil {
		return 0, 0, err
	}
	if t.Kind == types.RuntimeArray {
		return 0, l.Stride, nil
	}
	if t.Kind != types.Struct || len(t.Fields) == 0 {
		return 0, 0, fmt.Errorf("runtime host type %s has no trailing runtime array", t)
	}
	last := len(t.Fields) - 1
	if t.Fields[last].Type.Kind != types.RuntimeArray {
		return 0, 0, fmt.Errorf("runtime host type %s violates Pine trailing-array invariant", t)
	}
	fl := l.Fields[last]
	return fl.Offset, fl.Layout.Stride, nil
}

func safeIdent(s string) string {
	var b strings.Builder
	for i, r := range s {
		ok := r == '_' || r == '$' || unicode.IsLetter(r) || (i > 0 && unicode.IsDigit(r))
		if ok && r < 128 {
			b.WriteRune(r)
		} else {
			fmt.Fprintf(&b, "_x%x_", r)
		}
	}
	if b.Len() == 0 {
		return "pine"
	}
	if unicode.IsDigit(rune(b.String()[0])) {
		return "_" + b.String()
	}
	return b.String()
}

func jsQuote(s string) string { b, _ := json.Marshal(s); return string(b) }

func emitJavaScript(m *ir.Module, wgslSource string, meta *Metadata, metaJSON []byte) (string, error) {
	var b strings.Builder
	b.WriteString("// Generated by Pine. Browser-native WebGPU bindings.\n")
	b.WriteString("// Pine owns the shader/resource ABI; no runtime reflection is performed.\n\n")
	fmt.Fprintf(&b, "export const wgsl = %s;\n", jsQuote(wgslSource))
	fmt.Fprintf(&b, "export const metadata = Object.freeze(%s);\n\n", string(metaJSON))
	b.WriteString(`function __align4(n) { return (n + 3) & ~3; }
function __target(target, byteOffset, size) {
  if (target === undefined || target === null) {
    const buffer = new ArrayBuffer(size);
    return { buffer, byteOffset: 0, bytes: new Uint8Array(buffer) };
  }
  if (ArrayBuffer.isView(target)) {
    const base = target.byteOffset + byteOffset;
    if (base + size > target.byteOffset + target.byteLength) throw new RangeError("Pine pack target is too small");
    return { buffer: target.buffer, byteOffset: base, bytes: new Uint8Array(target.buffer, base, size) };
  }
  if (byteOffset + size > target.byteLength) throw new RangeError("Pine pack target is too small");
  return { buffer: target, byteOffset, bytes: new Uint8Array(target, byteOffset, size) };
}
function __view(t) { return new DataView(t.buffer, t.byteOffset); }
function __binding(value) {
  if (value && typeof value === "object" && Object.prototype.hasOwnProperty.call(value, "buffer")) return value;
  return { buffer: value };
}
function __workgroups(value) {
  const v = typeof value === "number" ? [value, 1, 1] : value;
  if (!Array.isArray(v) || v.length < 1 || v.length > 3) throw new TypeError("workgroups must be a positive integer or [x, y?, z?]");
  const x = v[0], y = v[1] ?? 1, z = v[2] ?? 1;
  for (const n of [x, y, z]) if (!Number.isInteger(n) || n <= 0) throw new RangeError("workgroup counts must be positive integers");
  return [x, y, z];
}
function __upload(device, bytes, usage, label) {
  const size = Math.max(4, __align4(bytes.byteLength));
  const buffer = device.createBuffer({ label, size, usage, mappedAtCreation: true });
  new Uint8Array(buffer.getMappedRange()).set(bytes);
  buffer.unmap();
  return buffer;
}

`)
	// Internal ABI writers and public struct packers.
	for _, t := range m.Structs {
		if err := emitWriter(&b, t); err != nil {
			return "", err
		}
	}
	for _, t := range m.Structs {
		if err := emitStructPacker(&b, t); err != nil {
			return "", err
		}
	}
	for _, r := range m.Resources {
		if err := emitWriter(&b, r.Type); err != nil {
			return "", err
		}
	}
	for i, r := range m.Resources {
		if err := emitResourcePacker(&b, i, r, meta.Resources[i]); err != nil {
			return "", err
		}
	}

	// Runtime WebGPU kernel object. Bind-group layouts are generated from Pine's
	// own ABI metadata, including empty groups needed to preserve group indices.
	b.WriteString(`class PineKernel {
  constructor(device, shaderModule, info) {
    this.device = device;
    this.metadata = info;
    const grouped = new Map();
    let maxGroup = -1;
    for (const p of info.resources) {
      const r = metadata.resources[p.resource];
      maxGroup = Math.max(maxGroup, r.group);
      let entries = grouped.get(r.group);
      if (!entries) grouped.set(r.group, entries = []);
      entries.push({
        binding: r.binding,
        visibility: GPUShaderStage.COMPUTE,
        buffer: {
          type: r.kind === "uniform" ? "uniform" : (r.access === "read_write" ? "storage" : "read-only-storage"),
          minBindingSize: r.minBindingSize,
        },
      });
    }
    this.bindGroupLayouts = [];
    for (let group = 0; group <= maxGroup; group++) {
      const entries = grouped.get(group) ?? [];
      entries.sort((a, b) => a.binding - b.binding);
      this.bindGroupLayouts.push(device.createBindGroupLayout({ label: "Pine " + info.name + " group " + group, entries }));
    }
    const layout = device.createPipelineLayout({ label: "Pine " + info.name + " layout", bindGroupLayouts: this.bindGroupLayouts });
    this.pipeline = device.createComputePipeline({
      label: "Pine " + info.name,
      layout,
      compute: { module: shaderModule, entryPoint: info.entryPoint },
    });
  }
  bind(resources) {
    const grouped = new Map();
    for (const p of this.metadata.resources) {
      const r = metadata.resources[p.resource];
      if (!(p.param in resources)) throw new TypeError("missing Pine resource " + p.param);
      let entries = grouped.get(r.group);
      if (!entries) grouped.set(r.group, entries = []);
      entries.push({ binding: r.binding, resource: __binding(resources[p.param]) });
    }
    const groups = [];
    for (let group = 0; group < this.bindGroupLayouts.length; group++) {
      const entries = grouped.get(group) ?? [];
      entries.sort((a, b) => a.binding - b.binding);
      groups.push(this.device.createBindGroup({ label: "Pine " + this.metadata.name + " group " + group, layout: this.bindGroupLayouts[group], entries }));
    }
    return groups;
  }
  encodeBound(pass, bindGroups, workgroups) {
    const [x, y, z] = __workgroups(workgroups);
    pass.setPipeline(this.pipeline);
    for (let i = 0; i < bindGroups.length; i++) pass.setBindGroup(i, bindGroups[i]);
    pass.dispatchWorkgroups(x, y, z);
  }
  encode(pass, resources, workgroups) { this.encodeBound(pass, this.bind(resources), workgroups); }
  dispatch(encoder, resources, workgroups, passDescriptor) {
    const pass = encoder.beginComputePass(passDescriptor);
    this.encode(pass, resources, workgroups);
    pass.end();
  }
}

export function createPineProgram(device) {
  const shaderModule = device.createShaderModule({ label: "Pine shader module", code: wgsl });
  const kernels = Object.create(null);
  for (const info of metadata.kernels) kernels[info.name] = new PineKernel(device, shaderModule, info);
  return Object.freeze({ device, shaderModule, kernels: Object.freeze(kernels), metadata });
}
`)
	return b.String(), nil
}

func writerName(t *types.Type) string {
	switch t.Kind {
	case types.I32:
		return "__write_i32"
	case types.U32:
		return "__write_u32"
	case types.F32:
		return "__write_f32"
	case types.Atomic:
		return writerName(t.Elem)
	case types.Vector:
		return fmt.Sprintf("__write_vec%d_%s", t.Lanes, safeIdent(t.Elem.String()))
	case types.Struct:
		return "__write_" + safeIdent(t.Name)
	case types.FixedArray:
		return fmt.Sprintf("__write_array_%d_%s", t.Count, safeIdent(typeToken(t.Elem)))
	case types.RuntimeArray:
		return "__write_runtime_" + safeIdent(typeToken(t.Elem))
	}
	return "__write_invalid"
}
func typeToken(t *types.Type) string {
	return strings.NewReplacer("<", "_", ">", "_", "[", "_", "]", "_", ",", "_", " ", "_").Replace(t.String())
}

func emitWriter(b *strings.Builder, t *types.Type) error {
	name := writerName(t)
	// Emit dependencies lazily through generated inline recursion is hard to dedupe;
	// scalar/vector/runtime helpers are emitted once from a registry below via a
	// local recursive walk per struct guarded by text-name set at generator level.
	_ = name
	return emitWriterSet(b, t, map[string]bool{})
}

// writerEmitted is intentionally encoded into the generated text marker search
// rather than global Go state, keeping Generate reentrant.
func emitWriterSet(b *strings.Builder, t *types.Type, seen map[string]bool) error {
	key := writerName(t)
	if strings.Contains(b.String(), "function "+key+"(") {
		return nil
	}
	if seen[key] {
		return nil
	}
	seen[key] = true
	switch t.Kind {
	case types.I32:
		fmt.Fprintf(b, "function %s(v, o, x) { v.setInt32(o, x, true); }\n", key)
	case types.U32:
		fmt.Fprintf(b, "function %s(v, o, x) { v.setUint32(o, x, true); }\n", key)
	case types.F32:
		fmt.Fprintf(b, "function %s(v, o, x) { v.setFloat32(o, x, true); }\n", key)
	case types.Atomic:
		return emitWriterSet(b, t.Elem, seen)
	case types.Vector:
		if err := emitWriterSet(b, t.Elem, seen); err != nil {
			return err
		}
		fmt.Fprintf(b, "function %s(v, o, x) {", key)
		for i := 0; i < t.Lanes; i++ {
			fmt.Fprintf(b, " %s(v, o + %d, x[%d]);", writerName(t.Elem), i*4, i)
		}
		b.WriteString(" }\n")
	case types.FixedArray:
		if err := emitWriterSet(b, t.Elem, seen); err != nil {
			return err
		}
		l, err := layout.Of(t)
		if err != nil {
			return err
		}
		fmt.Fprintf(b, "function %s(v, o, x) { if (x.length !== %d) throw new RangeError(\"Pine fixed array expects %d elements\"); for (let i = 0; i < %d; i++) %s(v, o + i * %d, x[i]); }\n", key, t.Count, t.Count, t.Count, writerName(t.Elem), l.Stride)
	case types.RuntimeArray:
		if err := emitWriterSet(b, t.Elem, seen); err != nil {
			return err
		}
		l, err := layout.Of(t)
		if err != nil {
			return err
		}
		fmt.Fprintf(b, "function %s(v, o, x) { for (let i = 0; i < x.length; i++) %s(v, o + i * %d, x[i]); }\n", key, writerName(t.Elem), l.Stride)
	case types.Struct:
		for _, f := range t.Fields {
			if err := emitWriterSet(b, f.Type, seen); err != nil {
				return err
			}
		}
		l, err := layout.Of(t)
		if err != nil {
			return err
		}
		fmt.Fprintf(b, "function %s(v, o, x) {\n", key)
		for i, f := range t.Fields {
			fmt.Fprintf(b, "  %s(v, o + %d, x[%s]);\n", writerName(f.Type), l.Fields[i].Offset, jsQuote(f.Name))
		}
		b.WriteString("}\n")
	default:
		return fmt.Errorf("cannot generate host writer for %s", t)
	}
	return nil
}

func dynamicSizeExpr(t *types.Type, value string) (string, error) {
	l, err := layout.Of(t)
	if err != nil {
		return "", err
	}
	if !l.Runtime {
		return fmt.Sprintf("%d", l.Size), nil
	}
	if t.Kind == types.RuntimeArray {
		return fmt.Sprintf("(%s.length * %d)", value, l.Stride), nil
	}
	last := len(t.Fields) - 1
	fl := l.Fields[last]
	return fmt.Sprintf("(%d + %s[%s].length * %d)", fl.Offset, value, jsQuote(t.Fields[last].Name), fl.Layout.Stride), nil
}

func emitStructPacker(b *strings.Builder, t *types.Type) error {
	l, err := layout.Of(t)
	if err != nil {
		return err
	}
	safe := safeIdent(t.Name)
	sizeExpr, err := dynamicSizeExpr(t, "value")
	if err != nil {
		return err
	}
	if l.Runtime {
		fmt.Fprintf(b, "export function byteSize%s(value) { return %s; }\n", safe, sizeExpr)
	} else {
		fmt.Fprintf(b, "export const byteSize%s = %d;\n", safe, l.Size)
	}
	fmt.Fprintf(b, "export function pack%s(value, target, byteOffset = 0) {\n", safe)
	if l.Runtime {
		fmt.Fprintf(b, "  const size = byteSize%s(value);\n", safe)
	} else {
		fmt.Fprintf(b, "  const size = byteSize%s;\n", safe)
	}
	b.WriteString("  const t = __target(target, byteOffset, size);\n")
	fmt.Fprintf(b, "  %s(__view(t), 0, value);\n", writerName(t))
	b.WriteString("  return t.bytes;\n}\n\n")
	return nil
}

func resourceHelperName(i int, r ir.Resource) string {
	return fmt.Sprintf("%s_g%d_b%d", safeIdent(r.Name), r.Group, r.Binding)
}
func resourceUsageExpr(r ir.Resource) string {
	if r.Kind == ir.Uniform {
		return "GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST"
	}
	return "GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC"
}

func emitResourcePacker(b *strings.Builder, i int, r ir.Resource, rm ResourceMeta) error {
	h := resourceHelperName(i, r)
	sizeExpr, err := dynamicSizeExpr(r.Type, "value")
	if err != nil {
		return err
	}
	if rm.Runtime {
		fmt.Fprintf(b, "export function byteSize_%s(value) { return %s; }\n", h, sizeExpr)
	} else {
		fmt.Fprintf(b, "export const byteSize_%s = %d;\n", h, rm.BindingSize)
	}
	fmt.Fprintf(b, "export function pack_%s(value, target, byteOffset = 0) {\n", h)
	if rm.Runtime {
		fmt.Fprintf(b, "  const size = byteSize_%s(value);\n", h)
		fmt.Fprintf(b, "  if (size < %d) throw new RangeError(%s);\n", rm.MinBindingSize, jsQuote("Pine runtime resource "+r.Name+" requires at least one element"))
	} else {
		fmt.Fprintf(b, "  const size = byteSize_%s;\n", h)
	}
	b.WriteString("  const t = __target(target, byteOffset, size);\n")
	fmt.Fprintf(b, "  %s(__view(t), 0, value);\n", writerName(r.Type))
	b.WriteString("  return t.bytes;\n}\n")
	fmt.Fprintf(b, "export function create_%s(device, value, usage = %s) { return __upload(device, pack_%s(value), usage, %s); }\n", h, resourceUsageExpr(r), h, jsQuote("Pine "+r.Name))
	fmt.Fprintf(b, "export function write_%s(device, buffer, value, offset = 0) { const bytes = pack_%s(value); device.queue.writeBuffer(buffer, offset, bytes.buffer, bytes.byteOffset, bytes.byteLength); return bytes.byteLength; }\n\n", h, h)
	return nil
}

func emitDeclarations(m *ir.Module, meta *Metadata) (string, error) {
	var b strings.Builder
	b.WriteString("// Generated by Pine. TypeScript declarations for the browser bindings.\n\n")
	for _, t := range m.Structs {
		fmt.Fprintf(&b, "export interface %s {\n", safeIdent(t.Name))
		for _, f := range t.Fields {
			fmt.Fprintf(&b, "  %s: %s;\n", safeIdent(f.Name), tsType(f.Type))
		}
		b.WriteString("}\n\n")
	}
	b.WriteString(`export type PineBufferBinding = GPUBuffer | GPUBufferBinding;
export interface PineKernel<R> {
  readonly metadata: Readonly<Record<string, unknown>>;
  readonly pipeline: GPUComputePipeline;
  bind(resources: R): readonly GPUBindGroup[];
  encodeBound(pass: GPUComputePassEncoder, bindGroups: readonly GPUBindGroup[], workgroups: number | readonly [number, number?, number?]): void;
  encode(pass: GPUComputePassEncoder, resources: R, workgroups: number | readonly [number, number?, number?]): void;
  dispatch(encoder: GPUCommandEncoder, resources: R, workgroups: number | readonly [number, number?, number?], passDescriptor?: GPUComputePassDescriptor): void;
}

`)
	for _, k := range meta.Kernels {
		tn := safeIdent(k.Name) + "Resources"
		fmt.Fprintf(&b, "export interface %s {\n", tn)
		for _, p := range k.Resources {
			fmt.Fprintf(&b, "  %s: PineBufferBinding;\n", safeIdent(p.Param))
		}
		b.WriteString("}\n\n")
	}
	for _, t := range m.Structs {
		l, _ := layout.Of(t)
		safe := safeIdent(t.Name)
		if l.Runtime {
			fmt.Fprintf(&b, "export function byteSize%s(value: %s): number;\n", safe, safe)
		} else {
			fmt.Fprintf(&b, "export const byteSize%s: %d;\n", safe, l.Size)
		}
		fmt.Fprintf(&b, "export function pack%s(value: %s, target?: ArrayBuffer | ArrayBufferView, byteOffset?: number): Uint8Array;\n", safe, safe)
	}
	b.WriteString("\n")
	for i, r := range m.Resources {
		h := resourceHelperName(i, r)
		val := tsType(r.Type)
		if meta.Resources[i].Runtime {
			fmt.Fprintf(&b, "export function byteSize_%s(value: %s): number;\n", h, val)
		} else {
			fmt.Fprintf(&b, "export const byteSize_%s: %d;\n", h, meta.Resources[i].BindingSize)
		}
		fmt.Fprintf(&b, "export function pack_%s(value: %s, target?: ArrayBuffer | ArrayBufferView, byteOffset?: number): Uint8Array;\n", h, val)
		fmt.Fprintf(&b, "export function create_%s(device: GPUDevice, value: %s, usage?: GPUBufferUsageFlags): GPUBuffer;\n", h, val)
		fmt.Fprintf(&b, "export function write_%s(device: GPUDevice, buffer: GPUBuffer, value: %s, offset?: number): number;\n", h, val)
	}
	b.WriteString("\nexport const wgsl: string;\nexport const metadata: Readonly<Record<string, unknown>>;\n")
	b.WriteString("export interface PineProgram {\n  readonly device: GPUDevice;\n  readonly shaderModule: GPUShaderModule;\n  readonly metadata: typeof metadata;\n  readonly kernels: {\n")
	for _, k := range meta.Kernels {
		fmt.Fprintf(&b, "    readonly %s: PineKernel<%sResources>;\n", safeIdent(k.Name), safeIdent(k.Name))
	}
	b.WriteString("  };\n}\nexport function createPineProgram(device: GPUDevice): PineProgram;\n")
	return b.String(), nil
}

func tsType(t *types.Type) string {
	switch t.Kind {
	case types.I32, types.U32, types.F32, types.Atomic:
		return "number"
	case types.Vector:
		parts := make([]string, t.Lanes)
		for i := range parts {
			parts[i] = "number"
		}
		return "readonly [" + strings.Join(parts, ", ") + "]"
	case types.Struct:
		return safeIdent(t.Name)
	case types.FixedArray:
		return "readonly " + tsType(t.Elem) + "[]"
	case types.RuntimeArray:
		return "readonly " + tsType(t.Elem) + "[]"
	}
	return "never"
}

// ValidateGenerated validates the cross-artifact contract owned by Pine: JSON
// metadata shape/version, mandatory JS exports, balanced generated delimiters,
// and declaration/export correspondence. It intentionally validates Pine's
// generated grammar rather than accepting arbitrary JavaScript.
func ValidateGenerated(js, dts string, metaJSON []byte) error {
	var m Metadata
	if err := json.Unmarshal(metaJSON, &m); err != nil {
		return fmt.Errorf("metadata JSON: %w", err)
	}
	if m.Format != metadataFormat {
		return fmt.Errorf("metadata format %q", m.Format)
	}
	if m.ABI.Layout != "pine-portable-v1" || m.ABI.Endianness != "little" {
		return fmt.Errorf("unexpected ABI descriptor")
	}
	for _, needle := range []string{"export const wgsl =", "export const metadata =", "export function createPineProgram(device)", "class PineKernel"} {
		if !strings.Contains(js, needle) {
			return fmt.Errorf("JavaScript missing %q", needle)
		}
	}
	if err := balancedGeneratedJS(js); err != nil {
		return err
	}
	for _, k := range m.Kernels {
		if !strings.Contains(dts, "interface "+safeIdent(k.Name)+"Resources") {
			return fmt.Errorf("declarations missing kernel %s", k.Name)
		}
	}
	pairs := map[[2]uint32]bool{}
	for _, r := range m.Resources {
		p := [2]uint32{r.Group, r.Binding}
		if pairs[p] {
			return fmt.Errorf("duplicate metadata binding %v", p)
		}
		pairs[p] = true
		if r.Runtime && r.MinBindingSize != r.RuntimeOffset+r.RuntimeStride {
			return fmt.Errorf("runtime resource %s minimum binding size invariant failed", r.Name)
		}
		if !r.Runtime && (r.BindingSize == 0 || r.MinBindingSize != r.BindingSize || r.BindingSize < r.ValueSize || r.BindingSize%16 != 0) {
			return fmt.Errorf("fixed resource %s binding-size invariant failed", r.Name)
		}
	}
	return nil
}

func balancedGeneratedJS(s string) error {
	var stack []rune
	inString := rune(0)
	escape := false
	inLine := false
	inBlock := false
	inTemplate := false
	for i := 0; i < len(s); i++ {
		c := rune(s[i])
		n := rune(0)
		if i+1 < len(s) {
			n = rune(s[i+1])
		}
		if inLine {
			if c == '\n' {
				inLine = false
			}
			continue
		}
		if inBlock {
			if c == '*' && n == '/' {
				inBlock = false
				i++
			}
			continue
		}
		if inString != 0 {
			if escape {
				escape = false
				continue
			}
			if c == '\\' {
				escape = true
				continue
			}
			if c == inString {
				inString = 0
			}
			continue
		}
		if inTemplate {
			if escape {
				escape = false
				continue
			}
			if c == '\\' {
				escape = true
				continue
			}
			if c == '`' {
				inTemplate = false
			}
			continue
		}
		if c == '/' && n == '/' {
			inLine = true
			i++
			continue
		}
		if c == '/' && n == '*' {
			inBlock = true
			i++
			continue
		}
		if c == '\'' || c == '"' {
			inString = c
			continue
		}
		if c == '`' {
			inTemplate = true
			continue
		}
		switch c {
		case '(', '[', '{':
			stack = append(stack, c)
		case ')', ']', '}':
			if len(stack) == 0 {
				return fmt.Errorf("generated JS closes %c without opener", c)
			}
			o := stack[len(stack)-1]
			stack = stack[:len(stack)-1]
			if (o == '(' && c != ')') || (o == '[' && c != ']') || (o == '{' && c != '}') {
				return fmt.Errorf("generated JS mismatched %c and %c", o, c)
			}
		}
	}
	if inString != 0 || inBlock || inTemplate {
		return fmt.Errorf("generated JS has unterminated literal/comment")
	}
	if len(stack) != 0 {
		return fmt.Errorf("generated JS has unclosed delimiter %c", stack[len(stack)-1])
	}
	return nil
}

// StableResourceOrder is exposed for consumers that want to reproduce Pine's
// canonical descriptor ordering in native bindings.
func StableResourceOrder(meta *Metadata) []ResourceMeta {
	out := append([]ResourceMeta(nil), meta.Resources...)
	sort.Slice(out, func(i, j int) bool {
		if out[i].Group != out[j].Group {
			return out[i].Group < out[j].Group
		}
		return out[i].Binding < out[j].Binding
	})
	return out
}
