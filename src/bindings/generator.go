package bindings

import (
	"encoding/json"
	"fmt"
	"strings"
	"unicode"

	"tach/src/abi"
	"tach/src/ir"
	"tach/src/layout"
	"tach/src/types"
)

type Artifacts struct {
	JavaScript   string
	Declarations string
	MetadataJSON []byte
}

// Metadata is a plain description of a compiled Tach module. Generated
// bindings and metadata are rebuilt together directly from Tach source.
type Metadata struct {
	Types     []TypeMetadata `json:"types"`
	Resources []ResourceMeta `json:"resources"`
	Kernels   []KernelMeta   `json:"kernels"`
}

type TypeMetadata struct {
	Name      string      `json:"name"`
	ByteSize  uint32      `json:"byteSize"`
	Alignment uint32      `json:"alignment"`
	Runtime   bool        `json:"runtime"`
	Fields    []FieldMeta `json:"fields"`
}

type FieldMeta struct {
	Name       string `json:"name"`
	Type       string `json:"type"`
	ByteOffset uint32 `json:"byteOffset"`
}

type ResourceMeta struct {
	Name            string `json:"name"`
	Group           uint32 `json:"group"`
	Binding         uint32 `json:"binding"`
	Kind            string `json:"kind"`
	Access          string `json:"access"`
	Type            string `json:"type"`
	ByteSize        uint32 `json:"byteSize,omitempty"`
	Alignment       uint32 `json:"alignment"`
	Runtime         bool   `json:"runtime"`
	RuntimeOffset   uint32 `json:"runtimeOffset,omitempty"`
	RuntimeStride   uint32 `json:"runtimeStride,omitempty"`
	MinimumByteSize uint32 `json:"minimumByteSize"`
}

type KernelMeta struct {
	Name          string               `json:"name"`
	EntryPoint    string               `json:"entryPoint"`
	Dimensions    uint32               `json:"dimensions"`
	WorkgroupSize [3]uint32            `json:"workgroupSize"`
	Resources     []KernelResourceMeta `json:"resources"`
}

type KernelResourceMeta struct {
	Name     string `json:"name"`
	Resource int    `json:"resource"`
}

// Generate emits a JavaScript module whose public surface is a direct
// translation of Tach: structs become TypeScript interfaces and exported
// compute kernels become functions with positional parameters. The generated
// module delegates WebGPU lifecycle and host-data plumbing to @depths/tach.
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
	js, err := emitJavaScript(m, wgslSource, meta)
	if err != nil {
		return nil, err
	}
	dts, err := emitDeclarations(m, meta)
	if err != nil {
		return nil, err
	}
	if err := ValidateGenerated(js, dts, metaJSON); err != nil {
		return nil, fmt.Errorf("Tach binding self-validation failed: %w", err)
	}
	return &Artifacts{
		JavaScript:   js,
		Declarations: dts,
		MetadataJSON: append(metaJSON, '\n'),
	}, nil
}

func buildMetadata(m *ir.Module) (*Metadata, error) {
	out := &Metadata{
		Types:     []TypeMetadata{},
		Resources: []ResourceMeta{},
		Kernels:   []KernelMeta{},
	}
	for _, t := range m.Structs {
		l, err := layout.Of(t)
		if err != nil {
			return nil, err
		}
		item := TypeMetadata{
			Name:      t.Name,
			ByteSize:  l.Size,
			Alignment: l.Align,
			Runtime:   l.Runtime,
			Fields:    []FieldMeta{},
		}
		for index, field := range t.Fields {
			item.Fields = append(item.Fields, FieldMeta{
				Name:       field.Name,
				Type:       field.Type.String(),
				ByteOffset: l.Fields[index].Offset,
			})
		}
		out.Types = append(out.Types, item)
	}
	for resourceIndex, resource := range m.Resources {
		l, err := layout.Of(resource.Type)
		if err != nil {
			return nil, fmt.Errorf("resource %s: %w", resource.Name, err)
		}
		item := ResourceMeta{
			Name:      resource.Name,
			Group:     0,
			Binding:   uint32(resourceIndex),
			Type:      resource.Type.String(),
			Alignment: l.Align,
			Runtime:   l.Runtime,
		}
		if resource.Kind == ir.Uniform {
			item.Kind = "uniform"
		} else {
			item.Kind = "storage"
		}
		if resource.Access == ir.Mutable || types.ContainsAtomic(resource.Type) {
			item.Access = "read_write"
		} else {
			item.Access = "read"
		}
		if l.Runtime {
			offset, stride, err := runtimeTail(resource.Type)
			if err != nil {
				return nil, err
			}
			item.RuntimeOffset = offset
			item.RuntimeStride = stride
			item.MinimumByteSize = offset + stride
		} else {
			// Both shader backends wrap fixed resources to this portable size.
			item.ByteSize = roundUp(16, l.Size)
			item.MinimumByteSize = item.ByteSize
		}
		out.Resources = append(out.Resources, item)
	}
	for _, function := range m.Functions {
		if !function.Compute {
			continue
		}
		item := KernelMeta{
			Name:          function.Name,
			EntryPoint:    abi.KernelEntry(function.Name),
			Dimensions:    uint32(len(function.Indices)),
			WorkgroupSize: function.Workgroup,
			Resources:     []KernelResourceMeta{},
		}
		for _, parameter := range function.ResourceParams {
			if parameter.Resource < 0 || parameter.Resource >= len(m.Resources) {
				return nil, fmt.Errorf("kernel %s resource index out of range", function.Name)
			}
			item.Resources = append(item.Resources, KernelResourceMeta{
				Name:     parameter.Name,
				Resource: parameter.Resource,
			})
		}
		out.Kernels = append(out.Kernels, item)
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
		return 0, 0, fmt.Errorf("runtime host type %s violates Tach's trailing-array invariant", t)
	}
	field := l.Fields[last]
	return field.Offset, field.Layout.Stride, nil
}

type hostLayout struct {
	Kind    string            `json:"kind"`
	Size    uint32            `json:"size,omitempty"`
	Stride  uint32            `json:"stride,omitempty"`
	Count   uint32            `json:"count,omitempty"`
	Runtime bool              `json:"runtime,omitempty"`
	Elem    *hostLayout       `json:"elem,omitempty"`
	Fields  []hostLayoutField `json:"fields,omitempty"`
}

type hostLayoutField struct {
	Name   string      `json:"name"`
	Offset uint32      `json:"offset"`
	Type   *hostLayout `json:"type"`
}

func describeHostLayout(t *types.Type) (*hostLayout, error) {
	if t == nil {
		return nil, fmt.Errorf("nil host type")
	}
	if t.Kind == types.Atomic {
		return describeHostLayout(t.Elem)
	}
	l, err := layout.Of(t)
	if err != nil {
		return nil, err
	}
	description := &hostLayout{
		Size:    l.Size,
		Stride:  l.Stride,
		Runtime: l.Runtime,
	}
	switch t.Kind {
	case types.I32:
		description.Kind = "i32"
	case types.U32:
		description.Kind = "u32"
	case types.F32:
		description.Kind = "f32"
	case types.Vector:
		description.Kind = "vector"
		description.Count = uint32(t.Lanes)
		description.Elem, err = describeHostLayout(t.Elem)
	case types.FixedArray:
		description.Kind = "array"
		description.Count = t.Count
		description.Elem, err = describeHostLayout(t.Elem)
	case types.RuntimeArray:
		description.Kind = "runtime"
		description.Elem, err = describeHostLayout(t.Elem)
	case types.Struct:
		description.Kind = "struct"
		description.Fields = []hostLayoutField{}
		for index, field := range t.Fields {
			fieldDescription, fieldErr := describeHostLayout(field.Type)
			if fieldErr != nil {
				return nil, fieldErr
			}
			description.Fields = append(description.Fields, hostLayoutField{
				Name:   field.Name,
				Offset: l.Fields[index].Offset,
				Type:   fieldDescription,
			})
		}
	default:
		return nil, fmt.Errorf("cannot describe host type %s", t)
	}
	if err != nil {
		return nil, err
	}
	return description, nil
}

type runtimeResource struct {
	Name            string      `json:"name"`
	Group           uint32      `json:"group"`
	Binding         uint32      `json:"binding"`
	Kind            string      `json:"kind"`
	Access          string      `json:"access"`
	ByteSize        uint32      `json:"byteSize,omitempty"`
	MinimumByteSize uint32      `json:"minimumByteSize"`
	Runtime         bool        `json:"runtime"`
	Layout          *hostLayout `json:"layout"`
}

func runtimeResources(m *ir.Module, meta *Metadata) ([]runtimeResource, error) {
	out := make([]runtimeResource, len(m.Resources))
	for index, resource := range m.Resources {
		description, err := describeHostLayout(resource.Type)
		if err != nil {
			return nil, fmt.Errorf("resource %s: %w", resource.Name, err)
		}
		item := meta.Resources[index]
		out[index] = runtimeResource{
			Name:            item.Name,
			Group:           item.Group,
			Binding:         item.Binding,
			Kind:            item.Kind,
			Access:          item.Access,
			ByteSize:        item.ByteSize,
			MinimumByteSize: item.MinimumByteSize,
			Runtime:         item.Runtime,
			Layout:          description,
		}
	}
	return out, nil
}

func jsQuote(s string) string {
	value, _ := json.Marshal(s)
	return string(value)
}

func emitJavaScript(m *ir.Module, wgslSource string, meta *Metadata) (string, error) {
	resources, err := runtimeResources(m, meta)
	if err != nil {
		return "", err
	}
	resourcesJSON, err := json.Marshal(resources)
	if err != nil {
		return "", err
	}
	kernelsJSON, err := json.Marshal(meta.Kernels)
	if err != nil {
		return "", err
	}
	for _, kernel := range meta.Kernels {
		if err := validateKernelExportName(kernel.Name); err != nil {
			return "", err
		}
		hasBuffer := false
		for _, parameter := range kernel.Resources {
			if !isASCIIIdentifier(parameter.Name) || typeScriptKeywords[parameter.Name] {
				return "", fmt.Errorf(
					"Tach parameter name %q is not a portable JavaScript identifier",
					parameter.Name,
				)
			}
			if m.Resources[parameter.Resource].Kind == ir.Buffer {
				hasBuffer = true
			}
		}
		if !hasBuffer {
			return "", fmt.Errorf("Tach kernel %q has no buffer parameter to carry its GPUDevice", kernel.Name)
		}
	}

	var b strings.Builder
	b.WriteString("// Generated by Tach. Typed WebGPU module.\n")
	b.WriteString("// Rebuild this file from its .tach source; its public API mirrors that source.\n\n")
	b.WriteString("import { defineModule as $defineModule } from \"@depths/tach/internal\";\n\n")
	b.WriteString("const $tach = $defineModule({\n")
	fmt.Fprintf(&b, "  source: %s,\n", jsQuote(wgslSource))
	fmt.Fprintf(&b, "  resources: %s,\n", resourcesJSON)
	fmt.Fprintf(&b, "  kernels: %s,\n", kernelsJSON)
	b.WriteString("});\n\n")
	for index, kernel := range meta.Kernels {
		fmt.Fprintf(&b, "export function %s(", kernel.Name)
		for parameterIndex, parameter := range kernel.Resources {
			if parameterIndex > 0 {
				b.WriteString(", ")
			}
			b.WriteString(parameter.Name)
		}
		if len(kernel.Resources) > 0 {
			b.WriteString(", ")
		}
		b.WriteString("$dispatch) {\n")
		fmt.Fprintf(&b, "  return $tach.dispatch(%d, [", index)
		for parameterIndex, parameter := range kernel.Resources {
			if parameterIndex > 0 {
				b.WriteString(", ")
			}
			b.WriteString(parameter.Name)
		}
		b.WriteString("], $dispatch);\n}\n\n")
	}
	return b.String(), nil
}

var typeScriptKeywords = map[string]bool{
	"any": true, "as": true, "asserts": true, "async": true, "await": true, "bigint": true,
	"boolean": true, "break": true, "case": true, "catch": true, "class": true, "const": true,
	"constructor": true, "continue": true, "declare": true, "default": true, "delete": true,
	"do": true, "else": true, "enum": true, "export": true, "extends": true, "false": true,
	"finally": true, "for": true, "from": true, "function": true, "get": true, "if": true,
	"implements": true, "import": true, "in": true, "infer": true, "instanceof": true,
	"interface": true, "is": true, "keyof": true, "let": true, "module": true, "namespace": true,
	"never": true, "new": true, "null": true, "number": true, "object": true, "of": true,
	"package": true, "private": true, "protected": true, "public": true, "readonly": true,
	"require": true, "return": true, "set": true, "static": true, "string": true, "super": true,
	"switch": true, "symbol": true, "this": true, "throw": true, "true": true, "try": true,
	"type": true, "typeof": true, "undefined": true, "unique": true, "unknown": true,
	"using": true, "var": true, "void": true, "while": true, "with": true, "yield": true,
}

func isASCIIIdentifier(s string) bool {
	if s == "" {
		return false
	}
	for index, r := range s {
		if r >= 128 || !(r == '_' || unicode.IsLetter(r) || index > 0 && unicode.IsDigit(r)) {
			return false
		}
	}
	return true
}

func typeScriptTypeName(name string) (string, error) {
	if !isASCIIIdentifier(name) || typeScriptKeywords[name] {
		return "", fmt.Errorf("Tach type name %q is not a portable TypeScript type identifier", name)
	}
	if name == "ComputeBuffer" || name == "ComputeDispatch" || name == "DispatchOptions" {
		return "", fmt.Errorf("Tach type name %q conflicts with an imported runtime type", name)
	}
	return name, nil
}

func validateKernelExportName(name string) error {
	if !isASCIIIdentifier(name) || typeScriptKeywords[name] {
		return fmt.Errorf("Tach kernel name %q is not a portable JavaScript export name", name)
	}
	return nil
}

func tsProperty(name string) string {
	if isASCIIIdentifier(name) {
		return name
	}
	return jsQuote(name)
}

func emitDeclarations(m *ir.Module, meta *Metadata) (string, error) {
	for _, t := range m.Structs {
		if _, err := typeScriptTypeName(t.Name); err != nil {
			return "", err
		}
	}
	for _, kernel := range meta.Kernels {
		if err := validateKernelExportName(kernel.Name); err != nil {
			return "", err
		}
	}

	var b strings.Builder
	b.WriteString("// Generated by Tach. Typed WebGPU module.\n\n")
	b.WriteString("import type { ComputeBuffer, ComputeDispatch, DispatchOptions } from \"@depths/tach\";\n\n")
	for _, t := range m.Structs {
		name, _ := typeScriptTypeName(t.Name)
		fmt.Fprintf(&b, "export interface %s {\n", name)
		for _, field := range t.Fields {
			fmt.Fprintf(&b, "  readonly %s: %s;\n", tsProperty(field.Name), tsType(field.Type))
		}
		b.WriteString("}\n\n")
	}
	for _, kernel := range meta.Kernels {
		fmt.Fprintf(&b, "export function %s(\n", kernel.Name)
		for _, parameter := range kernel.Resources {
			resource := m.Resources[parameter.Resource]
			parameterType := tsType(resource.Type)
			if resource.Kind == ir.Buffer {
				parameterType = "ComputeBuffer<" + parameterType + ">"
			}
			fmt.Fprintf(&b, "  %s: %s,\n", parameter.Name, parameterType)
		}
		dispatchSize := []string{"", "number", "readonly [x: number, y: number]", "readonly [x: number, y: number, z: number]"}[kernel.Dimensions]
		fmt.Fprintf(&b, "  $dispatch?: DispatchOptions<%s>,\n", dispatchSize)
		b.WriteString("): ComputeDispatch;\n\n")
	}
	return b.String(), nil
}

func tsType(t *types.Type) string {
	switch t.Kind {
	case types.I32, types.U32, types.F32, types.Atomic:
		return "number"
	case types.Vector:
		parts := make([]string, t.Lanes)
		for index := range parts {
			parts[index] = "number"
		}
		return "readonly [" + strings.Join(parts, ", ") + "]"
	case types.Struct:
		return t.Name
	case types.FixedArray:
		return "readonly " + tsType(t.Elem) + "[]"
	case types.RuntimeArray:
		if typed := tsTypedArray(t.Elem); typed != "" {
			if t.Elem.Kind == types.Vector {
				return typed + " | ReadonlyArray<" + tsType(t.Elem) + ">"
			}
			return typed + " | readonly " + tsType(t.Elem) + "[]"
		}
		if t.Elem.Kind == types.Vector {
			return "ReadonlyArray<" + tsType(t.Elem) + ">"
		}
		return "readonly " + tsType(t.Elem) + "[]"
	}
	return "never"
}

func tsTypedArray(t *types.Type) string {
	if t.Kind == types.Atomic {
		t = t.Elem
	}
	if t.Kind == types.Vector {
		if t.Lanes == 3 {
			return ""
		}
		t = t.Elem
	}
	switch t.Kind {
	case types.I32:
		return "Int32Array"
	case types.U32:
		return "Uint32Array"
	case types.F32:
		return "Float32Array"
	}
	return ""
}

// ValidateGenerated checks the compiler-owned cross-artifact invariants. It is
// intentionally scoped to generated output rather than arbitrary JavaScript.
func ValidateGenerated(js, dts string, metaJSON []byte) error {
	var metadata Metadata
	if err := json.Unmarshal(metaJSON, &metadata); err != nil {
		return fmt.Errorf("metadata JSON: %w", err)
	}
	for _, needle := range []string{
		"import { defineModule as $defineModule } from \"@depths/tach/internal\"",
		"const $tach = $defineModule({",
	} {
		if !strings.Contains(js, needle) {
			return fmt.Errorf("JavaScript missing %q", needle)
		}
	}
	if !strings.Contains(dts, "import type { ComputeBuffer, ComputeDispatch, DispatchOptions } from \"@depths/tach\"") {
		return fmt.Errorf("TypeScript declarations are missing runtime type imports")
	}
	pairs := map[[2]uint32]bool{}
	for _, resource := range metadata.Resources {
		pair := [2]uint32{resource.Group, resource.Binding}
		if pairs[pair] {
			return fmt.Errorf("duplicate metadata binding %v", pair)
		}
		pairs[pair] = true
		if resource.Runtime &&
			resource.MinimumByteSize != resource.RuntimeOffset+resource.RuntimeStride {
			return fmt.Errorf(
				"runtime resource %s minimum byte-size invariant failed",
				resource.Name,
			)
		}
		if !resource.Runtime &&
			(resource.ByteSize == 0 ||
				resource.MinimumByteSize != resource.ByteSize ||
				resource.ByteSize%16 != 0) {
			return fmt.Errorf("fixed resource %s byte-size invariant failed", resource.Name)
		}
	}
	for index, kernel := range metadata.Kernels {
		if kernel.Dimensions < 1 || kernel.Dimensions > 3 {
			return fmt.Errorf("kernel %s has invalid logical dimension count %d", kernel.Name, kernel.Dimensions)
		}
		if kernel.EntryPoint != kernel.Name {
			return fmt.Errorf("kernel %s has mangled entry point %q", kernel.Name, kernel.EntryPoint)
		}
		export := "export function " + kernel.Name + "("
		if !strings.Contains(js, export) || !strings.Contains(dts, export) {
			return fmt.Errorf("generated bindings are missing kernel function %s", kernel.Name)
		}
		dispatch := fmt.Sprintf("return $tach.dispatch(%d", index)
		if !strings.Contains(js, dispatch) {
			return fmt.Errorf("generated binding %s does not construct a compute dispatch", kernel.Name)
		}
	}
	return nil
}
