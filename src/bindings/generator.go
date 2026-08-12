package bindings

import (
	"encoding/json"
	"fmt"
	"strings"
	"unicode"

	"tach/src/backend"
	"tach/src/flow"
	"tach/src/ir"
	"tach/src/layout"
	"tach/src/types"
)

type Artifacts struct {
	JavaScript   string
	Declarations string
	MetadataJSON []byte
}

type Metadata struct {
	Schema   int                 `json:"schema"`
	Types    []TypeMetadata      `json:"types"`
	Programs []PublicProgramMeta `json:"programs"`
	Targets  TargetMetadata      `json:"targets"`
}
type TypeMetadata struct {
	Name   string      `json:"name"`
	Fields []FieldMeta `json:"fields"`
}
type FieldMeta struct {
	Name string `json:"name"`
	Type string `json:"type"`
}
type PublicProgramMeta struct {
	Name       string                 `json:"name"`
	Parameters []PublicParameterMeta  `json:"parameters"`
	Resources  []ExternalResourceMeta `json:"resources"`
	Launch     *LaunchMeta            `json:"launch,omitempty"`
}
type PublicParameterMeta struct {
	Name     string `json:"name"`
	Kind     string `json:"kind"`
	Type     string `json:"type"`
	Resource *int   `json:"resource,omitempty"`
}
type LaunchMeta struct {
	Dimensions        int  `json:"dimensions"`
	InferFromResource *int `json:"inferFromResource,omitempty"`
}
type ExternalResourceMeta struct {
	Name            string      `json:"name"`
	Type            string      `json:"type"`
	ByteSize        uint32      `json:"byteSize,omitempty"`
	Alignment       uint32      `json:"alignment"`
	Runtime         bool        `json:"runtime"`
	RuntimeOffset   uint32      `json:"runtimeOffset,omitempty"`
	RuntimeStride   uint32      `json:"runtimeStride,omitempty"`
	MinimumByteSize uint32      `json:"minimumByteSize"`
	Layout          *HostLayout `json:"layout"`
}
type TargetMetadata struct {
	Web   *TargetPlanMeta `json:"web,omitempty"`
	SPIRV *TargetPlanMeta `json:"spirv,omitempty"`
}
type TargetPlanMeta struct {
	Kernels  []PhysicalKernelMeta `json:"kernels"`
	Programs []ProgramPlanMeta    `json:"programs"`
}
type PhysicalKernelMeta struct {
	EntryPoint     string              `json:"entryPoint"`
	WorkgroupSize  [3]uint32           `json:"workgroupSize"`
	Bindings       []BindingMeta       `json:"bindings"`
	ParameterBlock *ParameterBlockMeta `json:"parameterBlock,omitempty"`
}
type BindingMeta struct {
	Group           uint32 `json:"group"`
	Binding         uint32 `json:"binding"`
	Access          string `json:"access"`
	Type            string `json:"type"`
	MinimumByteSize uint32 `json:"minimumByteSize"`
}
type ParameterBlockMeta struct {
	Group    uint32               `json:"group"`
	Binding  uint32               `json:"binding"`
	ByteSize uint32               `json:"byteSize"`
	Fields   []ParameterFieldMeta `json:"fields"`
}
type ParameterFieldMeta struct {
	Type       string      `json:"type"`
	ByteOffset uint32      `json:"byteOffset"`
	Layout     *HostLayout `json:"layout"`
}
type ProgramPlanMeta struct {
	Program       int             `json:"program"`
	Transients    []TransientMeta `json:"transients"`
	Steps         []StepMeta      `json:"steps"`
	RepeatBarrier *StepMeta       `json:"repeatBarrier,omitempty"`
	Repeat        string          `json:"repeat"`
}
type TransientMeta struct {
	Type            string          `json:"type"`
	Stride          uint32          `json:"stride"`
	Alignment       uint32          `json:"alignment"`
	MinimumByteSize uint32          `json:"minimumByteSize"`
	Length          ShapeExpression `json:"length"`
	Color           int             `json:"color"`
	FirstStep       int             `json:"firstStep"`
	LastStep        int             `json:"lastStep"`
}
type StepMeta struct {
	Kind       string               `json:"kind"`
	Kernel     int                  `json:"kernel"`
	Domain     []ShapeExpression    `json:"domain,omitempty"`
	Resources  []ResourceSourceMeta `json:"resources"`
	Parameters []ValueSource        `json:"parameters,omitempty"`
}
type ResourceSourceMeta struct {
	Binding  uint32 `json:"binding"`
	Kind     string `json:"kind"`
	Resource int    `json:"resource"`
}
type ShapeExpression struct {
	Op        string           `json:"op"`
	Value     uint32           `json:"value,omitempty"`
	Parameter int              `json:"parameter"`
	Resource  int              `json:"resource"`
	Path      []string         `json:"path,omitempty"`
	Axis      uint8            `json:"axis"`
	Left      *ShapeExpression `json:"left,omitempty"`
	Right     *ShapeExpression `json:"right,omitempty"`
}
type ValueSource struct {
	Kind       string           `json:"kind"`
	Parameter  int              `json:"parameter"`
	Path       []string         `json:"path,omitempty"`
	Value      any              `json:"value,omitempty"`
	Expression *ShapeExpression `json:"expression,omitempty"`
}

type HostLayout struct {
	Kind    string            `json:"kind"`
	Size    uint32            `json:"size,omitempty"`
	Stride  uint32            `json:"stride,omitempty"`
	Count   uint32            `json:"count,omitempty"`
	Runtime bool              `json:"runtime,omitempty"`
	Elem    *HostLayout       `json:"elem,omitempty"`
	Fields  []HostLayoutField `json:"fields,omitempty"`
}
type HostLayoutField struct {
	Name   string      `json:"name"`
	Offset uint32      `json:"offset"`
	Type   *HostLayout `json:"type"`
}

func Generate(logical *flow.Module, web, spirv *backend.Executable, wgslSource string) (*Artifacts, error) {
	if err := flow.Verify(logical); err != nil {
		return nil, err
	}
	metadata, err := buildMetadata(logical, web, spirv)
	if err != nil {
		return nil, err
	}
	metadataJSON, err := json.MarshalIndent(metadata, "", "  ")
	if err != nil {
		return nil, err
	}
	metadataJSON = append(metadataJSON, '\n')
	artifacts := &Artifacts{MetadataJSON: metadataJSON}
	if web != nil {
		artifacts.JavaScript, err = emitJavaScript(metadata, wgslSource)
		if err != nil {
			return nil, err
		}
		artifacts.Declarations, err = emitDeclarations(logical)
		if err != nil {
			return nil, err
		}
		if err := ValidateGenerated(artifacts.JavaScript, artifacts.Declarations, metadataJSON); err != nil {
			return nil, fmt.Errorf("Tach binding self-validation failed: %w", err)
		}
	} else if err := ValidateMetadata(metadata); err != nil {
		return nil, err
	}
	return artifacts, nil
}

func buildMetadata(logical *flow.Module, web, spirv *backend.Executable) (*Metadata, error) {
	metadata := &Metadata{Schema: 1, Types: []TypeMetadata{}, Programs: []PublicProgramMeta{}}
	for _, t := range logical.Kernel.Structs {
		item := TypeMetadata{Name: t.Name, Fields: []FieldMeta{}}
		for _, field := range t.Fields {
			item.Fields = append(item.Fields, FieldMeta{Name: field.Name, Type: field.Type.String()})
		}
		metadata.Types = append(metadata.Types, item)
	}
	for _, program := range logical.Programs {
		item := PublicProgramMeta{Name: program.Name, Parameters: []PublicParameterMeta{}, Resources: []ExternalResourceMeta{}}
		external := map[flow.ResourceID]int{}
		for _, resource := range program.Resources {
			if resource.Kind != flow.External {
				continue
			}
			index := len(item.Resources)
			external[resource.ID] = index
			description, err := externalResource(resource)
			if err != nil {
				return nil, err
			}
			item.Resources = append(item.Resources, description)
		}
		for _, parameter := range program.Parameters {
			p := PublicParameterMeta{Name: parameter.Name, Type: parameter.Type.String()}
			if parameter.Kind == flow.BufferParameter {
				p.Kind = "buffer"
				index := external[parameter.Resource]
				p.Resource = &index
			} else {
				p.Kind = "value"
			}
			item.Parameters = append(item.Parameters, p)
		}
		if program.Indexed {
			item.Launch = &LaunchMeta{Dimensions: program.Rank}
			if program.Rank == 1 {
				for i, resource := range item.Resources {
					if resource.Runtime {
						index := i
						item.Launch.InferFromResource = &index
						break
					}
				}
			}
		}
		metadata.Programs = append(metadata.Programs, item)
	}
	var err error
	if web != nil {
		metadata.Targets.Web, err = targetMetadata(web)
		if err != nil {
			return nil, err
		}
	}
	if spirv != nil {
		metadata.Targets.SPIRV, err = targetMetadata(spirv)
		if err != nil {
			return nil, err
		}
	}
	if err := ValidateMetadata(metadata); err != nil {
		return nil, err
	}
	return metadata, nil
}

func externalResource(resource flow.Resource) (ExternalResourceMeta, error) {
	l, err := layout.Of(resource.Type)
	if err != nil {
		return ExternalResourceMeta{}, err
	}
	host, err := describeHostLayout(resource.Type)
	if err != nil {
		return ExternalResourceMeta{}, err
	}
	out := ExternalResourceMeta{Name: resource.Name, Type: resource.Type.String(), Alignment: l.Align, Runtime: l.Runtime, Layout: host}
	if l.Runtime {
		out.RuntimeOffset, out.RuntimeStride, err = runtimeTail(resource.Type)
		out.MinimumByteSize = out.RuntimeOffset + out.RuntimeStride
	} else {
		out.ByteSize = roundUp(16, l.Size)
		out.MinimumByteSize = out.ByteSize
	}
	return out, err
}

func targetMetadata(executable *backend.Executable) (*TargetPlanMeta, error) {
	if err := backend.Verify(executable); err != nil {
		return nil, err
	}
	target := &TargetPlanMeta{Kernels: []PhysicalKernelMeta{}, Programs: []ProgramPlanMeta{}}
	for _, kernel := range executable.PhysicalKernels {
		item := PhysicalKernelMeta{EntryPoint: kernel.Entry, WorkgroupSize: kernel.Workgroup, Bindings: []BindingMeta{}}
		for _, binding := range kernel.Bindings {
			access := "read"
			if binding.Access == ir.Mutable || types.ContainsAtomic(binding.Type) {
				access = "read_write"
			}
			item.Bindings = append(item.Bindings, BindingMeta{Group: 0, Binding: binding.Binding, Access: access, Type: binding.Type.String(), MinimumByteSize: binding.MinimumByteSize})
		}
		if block := kernel.Parameters; block != nil {
			item.ParameterBlock = &ParameterBlockMeta{Group: 0, Binding: block.Binding, ByteSize: block.Layout.Size, Fields: []ParameterFieldMeta{}}
			for _, field := range block.Fields {
				host, err := parameterLayout(field.Logical)
				if err != nil {
					return nil, err
				}
				item.ParameterBlock.Fields = append(item.ParameterBlock.Fields, ParameterFieldMeta{Type: field.Logical.String(), ByteOffset: field.Offset, Layout: host})
			}
		}
		target.Kernels = append(target.Kernels, item)
	}
	for _, plan := range executable.Programs {
		program := executable.Logical.Programs[plan.Program]
		item := ProgramPlanMeta{Program: plan.Program, Transients: []TransientMeta{}, Steps: []StepMeta{}, Repeat: "program"}
		if plan.Repeat == backend.RepeatInvocationLoop {
			item.Repeat = "invocation-loop"
		}
		for _, transient := range plan.Transients {
			expression, err := shapeExpression(program, transient.Length)
			if err != nil {
				return nil, err
			}
			item.Transients = append(item.Transients, TransientMeta{Type: transient.Type.String(), Stride: transient.Stride, Alignment: transient.Alignment, MinimumByteSize: transient.MinimumByteSize, Length: expression, Color: transient.Color, FirstStep: transient.FirstStep, LastStep: transient.LastStep})
		}
		for _, step := range plan.Steps {
			description, err := stepMetadata(program, executable.PhysicalKernels, step)
			if err != nil {
				return nil, err
			}
			item.Steps = append(item.Steps, description)
		}
		if len(plan.RepeatBarrier) > 0 {
			description := barrierMetadata(plan.RepeatBarrier)
			item.RepeatBarrier = &description
		}
		target.Programs = append(target.Programs, item)
	}
	return target, nil
}

func stepMetadata(program *flow.Program, kernels []backend.PhysicalKernel, step backend.Step) (StepMeta, error) {
	if step.Kind == backend.BarrierStepKind {
		return barrierMetadata(step.Barrier), nil
	}
	out := StepMeta{Kind: "dispatch", Kernel: step.Kernel, Domain: []ShapeExpression{}, Resources: []ResourceSourceMeta{}, Parameters: []ValueSource{}}
	for _, id := range step.Domain {
		expression, err := shapeExpression(program, id)
		if err != nil {
			return out, err
		}
		out.Domain = append(out.Domain, expression)
	}
	for _, resource := range step.Resources {
		kind := "external"
		if resource.Kind == backend.TransientSource {
			kind = "transient"
		}
		out.Resources = append(out.Resources, ResourceSourceMeta{Binding: resource.Binding, Kind: kind, Resource: resource.Resource})
	}
	block := kernels[step.Kernel].Parameters
	if block != nil {
		for _, field := range block.Fields {
			if field.Parameter < 0 || field.Parameter >= len(step.Parameters) {
				return out, fmt.Errorf("kernel parameter field references missing dispatch value")
			}
			source, err := valueSource(program, step.Parameters[field.Parameter], field.Path)
			if err != nil {
				return out, err
			}
			out.Parameters = append(out.Parameters, source)
		}
	}
	return out, nil
}

func barrierMetadata(resources []backend.BarrierResource) StepMeta {
	out := StepMeta{Kind: "barrier", Resources: []ResourceSourceMeta{}}
	for _, resource := range resources {
		kind := "external"
		if resource.Kind == backend.TransientSource {
			kind = "transient"
		}
		out.Resources = append(out.Resources, ResourceSourceMeta{Kind: kind, Resource: resource.Resource})
	}
	return out
}

func shapeExpression(program *flow.Program, id flow.ShapeID) (ShapeExpression, error) {
	shape := program.Shape(id)
	if shape == nil {
		return ShapeExpression{}, fmt.Errorf("invalid shape %d", id)
	}
	out := ShapeExpression{Path: append([]string(nil), shape.Path...)}
	switch shape.Op {
	case flow.ShapeConstant:
		out.Op, out.Value = "constant", shape.Value
	case flow.ShapeParameter:
		out.Op, out.Parameter = "parameter", shape.Parameter
	case flow.ShapeResourceLength:
		out.Op, out.Resource = "resourceLength", externalResourceIndex(program, shape.Resource)
	case flow.ShapeLaunchAxis:
		out.Op, out.Axis = "launchAxis", shape.Axis
	default:
		out.Op = shape.Op.String()
		left, err := shapeExpression(program, shape.Left)
		if err != nil {
			return out, err
		}
		right, err := shapeExpression(program, shape.Right)
		if err != nil {
			return out, err
		}
		out.Left, out.Right = &left, &right
	}
	return out, nil
}

func externalResourceIndex(program *flow.Program, id flow.ResourceID) int {
	index := 0
	for _, resource := range program.Resources {
		if resource.Kind == flow.External {
			if resource.ID == id {
				return index
			}
			index++
		}
	}
	return -1
}

func valueSource(program *flow.Program, argument flow.ValueArgument, fieldPath []string) (ValueSource, error) {
	switch argument.Kind {
	case flow.ValueParameterRef:
		return ValueSource{Kind: "parameter", Parameter: argument.Parameter, Path: append(append([]string(nil), argument.Path...), fieldPath...)}, nil
	case flow.ValueBool:
		return ValueSource{Kind: "bool", Value: argument.Bits != 0}, nil
	case flow.ValueI32:
		return ValueSource{Kind: "i32", Value: int32(argument.Bits)}, nil
	case flow.ValueU32:
		return ValueSource{Kind: "u32", Value: argument.Bits}, nil
	case flow.ValueF32Bits:
		return ValueSource{Kind: "f32Bits", Value: argument.Bits}, nil
	case flow.ValueShape:
		expression, err := shapeExpression(program, argument.Shape)
		return ValueSource{Kind: "shape", Expression: &expression}, err
	case flow.ValueRepeat:
		return ValueSource{Kind: "repeat"}, nil
	default:
		return ValueSource{}, fmt.Errorf("invalid value source")
	}
}

func emitJavaScript(metadata *Metadata, wgsl string) (string, error) {
	definition := struct {
		Shader   string              `json:"shader"`
		Schema   int                 `json:"schema"`
		Types    []TypeMetadata      `json:"types"`
		Programs []PublicProgramMeta `json:"programs"`
		Target   *TargetPlanMeta     `json:"target"`
	}{wgsl, metadata.Schema, metadata.Types, metadata.Programs, metadata.Targets.Web}
	encoded, err := json.Marshal(definition)
	if err != nil {
		return "", err
	}
	var b strings.Builder
	b.WriteString("// Generated by Tach.\nimport { defineModule as $defineModule } from \"@depths/tach/internal\";\n\nconst $tach = $defineModule(")
	b.Write(encoded)
	b.WriteString(");\n\n")
	for index, program := range metadata.Programs {
		if err := validateExportName(program.Name); err != nil {
			return "", err
		}
		fmt.Fprintf(&b, "export function %s(...$args) { const $options = $args.length > %d ? $args.pop() : undefined; return $tach.command(%d, $args, $options); }\n", program.Name, len(program.Parameters), index)
	}
	return b.String(), nil
}

func emitDeclarations(logical *flow.Module) (string, error) {
	var b strings.Builder
	b.WriteString("// Generated by Tach. Typed WebGPU module.\n\nimport type { CommandOptions, ComputeBuffer, ComputeCommand, LaunchOptions } from \"@depths/tach\";\n\n")
	for _, t := range logical.Kernel.Structs {
		if err := validateExportName(t.Name); err != nil {
			return "", err
		}
		fmt.Fprintf(&b, "export type %s = {\n", t.Name)
		for _, field := range t.Fields {
			fmt.Fprintf(&b, "  readonly %s: %s;\n", tsProperty(field.Name), tsType(field.Type))
		}
		b.WriteString("};\n\n")
	}
	for _, program := range logical.Programs {
		if err := validateExportName(program.Name); err != nil {
			return "", err
		}
		fmt.Fprintf(&b, "export function %s(\n", program.Name)
		for _, parameter := range program.Parameters {
			t := tsType(parameter.Type)
			if parameter.Kind == flow.BufferParameter {
				t = "ComputeBuffer<" + t + ">"
			}
			fmt.Fprintf(&b, "  %s: %s,\n", parameter.Name, t)
		}
		if program.Indexed {
			size := []string{"", "number", "readonly [x: number, y: number]", "readonly [x: number, y: number, z: number]"}[program.Rank]
			fmt.Fprintf(&b, "  $launch?: LaunchOptions<%s>,\n", size)
		} else {
			b.WriteString("  $options?: CommandOptions,\n")
		}
		b.WriteString("): ComputeCommand;\n\n")
	}
	return b.String(), nil
}

func ValidateGenerated(js, declarations string, metadataJSON []byte) error {
	var metadata Metadata
	if err := json.Unmarshal(metadataJSON, &metadata); err != nil {
		return err
	}
	if err := ValidateMetadata(&metadata); err != nil {
		return err
	}
	if !strings.Contains(js, "defineModule as $defineModule") || !strings.Contains(declarations, "CommandOptions, ComputeBuffer, ComputeCommand, LaunchOptions") {
		return fmt.Errorf("generated module is missing runtime contract")
	}
	for _, program := range metadata.Programs {
		needle := "export function " + program.Name + "("
		if !strings.Contains(js, "export function "+program.Name) || !strings.Contains(declarations, needle) {
			return fmt.Errorf("missing public program %s", program.Name)
		}
	}
	return nil
}

func ValidateMetadata(metadata *Metadata) error {
	if metadata == nil || metadata.Schema != 1 {
		return fmt.Errorf("metadata schema/programs are invalid")
	}
	if metadata.Targets.Web == nil && metadata.Targets.SPIRV == nil {
		return fmt.Errorf("metadata contains no target plan")
	}
	for _, target := range []*TargetPlanMeta{metadata.Targets.Web, metadata.Targets.SPIRV} {
		if target == nil {
			continue
		}
		if len(target.Programs) != len(metadata.Programs) {
			return fmt.Errorf("target program count mismatch")
		}
		for i, program := range target.Programs {
			if program.Program != i || (program.Repeat != "program" && program.Repeat != "invocation-loop") {
				return fmt.Errorf("invalid target program %d", i)
			}
			for _, step := range program.Steps {
				if step.Kind == "dispatch" {
					if step.Kernel < 0 || step.Kernel >= len(target.Kernels) {
						return fmt.Errorf("invalid kernel reference")
					}
				} else if step.Kind != "barrier" {
					return fmt.Errorf("invalid step kind")
				}
			}
		}
		for i, kernel := range target.Kernels {
			if kernel.EntryPoint != fmt.Sprintf("_tach_k%d", i) || kernel.WorkgroupSize[0] == 0 || kernel.WorkgroupSize[1] == 0 || kernel.WorkgroupSize[2] == 0 {
				return fmt.Errorf("invalid physical kernel %d", i)
			}
			for j, binding := range kernel.Bindings {
				if binding.Group != 0 || binding.Binding != uint32(j) {
					return fmt.Errorf("kernel %d bindings are not dense", i)
				}
			}
		}
	}
	return nil
}

func describeHostLayout(t *types.Type) (*HostLayout, error) {
	if t.Kind == types.Atomic {
		return describeHostLayout(t.Elem)
	}
	l, err := layout.Of(t)
	if err != nil {
		return nil, err
	}
	out := &HostLayout{Size: l.Size, Stride: l.Stride, Runtime: l.Runtime}
	switch t.Kind {
	case types.Bool:
		out.Kind = "bool"
		out.Size = 4
	case types.I32:
		out.Kind = "i32"
	case types.U32:
		out.Kind = "u32"
	case types.F32:
		out.Kind = "f32"
	case types.Vector:
		out.Kind = "vector"
		out.Count = uint32(t.Lanes)
		out.Elem, err = describeHostLayout(t.Elem)
	case types.FixedArray:
		out.Kind = "array"
		out.Count = t.Count
		out.Elem, err = describeHostLayout(t.Elem)
	case types.RuntimeArray:
		out.Kind = "runtime"
		out.Elem, err = describeHostLayout(t.Elem)
	case types.Struct:
		out.Kind = "struct"
		out.Fields = []HostLayoutField{}
		for i, field := range t.Fields {
			child, e := describeHostLayout(field.Type)
			if e != nil {
				return nil, e
			}
			out.Fields = append(out.Fields, HostLayoutField{Name: field.Name, Offset: l.Fields[i].Offset, Type: child})
		}
	default:
		return nil, fmt.Errorf("cannot describe host type %s", t)
	}
	return out, err
}
func parameterLayout(t *types.Type) (*HostLayout, error) {
	if t.Kind == types.Bool {
		return &HostLayout{Kind: "bool", Size: 4}, nil
	}
	return describeHostLayout(t)
}
func runtimeTail(t *types.Type) (uint32, uint32, error) {
	l, e := layout.Of(t)
	if e != nil {
		return 0, 0, e
	}
	if t.Kind == types.RuntimeArray {
		return 0, l.Stride, nil
	}
	if t.Kind != types.Struct || len(t.Fields) == 0 {
		return 0, 0, fmt.Errorf("runtime type %s has no tail", t)
	}
	i := len(t.Fields) - 1
	return l.Fields[i].Offset, l.Fields[i].Layout.Stride, nil
}
func roundUp(a, n uint32) uint32 {
	if a == 0 {
		return n
	}
	return (n + a - 1) / a * a
}
func tsType(t *types.Type) string {
	switch t.Kind {
	case types.Bool:
		return "boolean"
	case types.I32, types.U32, types.F32, types.Atomic:
		return "number"
	case types.Vector:
		parts := make([]string, t.Lanes)
		for i := range parts {
			parts[i] = "number"
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
func tsProperty(name string) string {
	if isASCIIIdentifier(name) {
		return name
	}
	encoded, _ := json.Marshal(name)
	return string(encoded)
}
func validateExportName(name string) error {
	if !isASCIIIdentifier(name) || typeScriptKeywords[name] {
		return fmt.Errorf("%q is not a valid generated export", name)
	}
	return nil
}
func isASCIIIdentifier(s string) bool {
	if s == "" {
		return false
	}
	for i, r := range s {
		if r > unicode.MaxASCII || !(r == '_' || r == '$' || unicode.IsLetter(r) || i > 0 && unicode.IsDigit(r)) {
			return false
		}
	}
	return true
}

var typeScriptKeywords = map[string]bool{"break": true, "case": true, "class": true, "const": true, "continue": true, "default": true, "delete": true, "do": true, "else": true, "export": true, "extends": true, "false": true, "finally": true, "for": true, "function": true, "if": true, "import": true, "in": true, "instanceof": true, "new": true, "null": true, "return": true, "super": true, "switch": true, "this": true, "throw": true, "true": true, "try": true, "typeof": true, "var": true, "void": true, "while": true, "with": true, "yield": true}
