package bindings

import (
	"encoding/json"
	"fmt"

	"tach/src/backend"
	"tach/src/flow"
	"tach/src/ir"
	"tach/src/layout"
	"tach/src/types"
)

type Artifacts struct {
	Metadata     *Metadata
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
	View       bool                   `json:"view,omitempty"`
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
	Vulkan   string               `json:"vulkan,omitempty"`
	SPIRV    string               `json:"spirv,omitempty"`
	Features []string             `json:"features,omitempty"`
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
	Kind            string `json:"kind"`
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
	View          *ViewMeta       `json:"view,omitempty"`
}
type ViewMeta struct {
	Format      string          `json:"format"`
	Step        StepMeta        `json:"step"`
	Width       ShapeExpression `json:"width"`
	Height      ShapeExpression `json:"height"`
	OutputColor int             `json:"outputColor"`
	Output      uint32          `json:"output"`
	Fused       bool            `json:"fused"`
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

func Generate(logical *flow.Module, web, spirv *backend.Executable) (*Artifacts, error) {
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
	return &Artifacts{Metadata: metadata, MetadataJSON: metadataJSON}, nil
}

func buildMetadata(logical *flow.Module, web, spirv *backend.Executable) (*Metadata, error) {
	metadata := &Metadata{Schema: 2, Types: []TypeMetadata{}, Programs: []PublicProgramMeta{}}
	for _, t := range logical.Kernel.Structs {
		item := TypeMetadata{Name: t.Name, Fields: []FieldMeta{}}
		for _, field := range t.Fields {
			item.Fields = append(item.Fields, FieldMeta{Name: field.Name, Type: field.Type.String()})
		}
		metadata.Types = append(metadata.Types, item)
	}
	for _, program := range logical.Programs {
		item := PublicProgramMeta{Name: program.Name, Parameters: []PublicParameterMeta{}, Resources: []ExternalResourceMeta{}, View: program.View != nil}
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
	target.Features = backend.RequiredFeatures(executable)
	if executable.Target == backend.SPIRV {
		target.Vulkan = backend.VulkanVersion
		target.SPIRV = backend.SPIRVVersion
	}
	for _, kernel := range executable.PhysicalKernels {
		item := PhysicalKernelMeta{EntryPoint: kernel.Entry, WorkgroupSize: kernel.Workgroup, Bindings: []BindingMeta{}}
		for _, binding := range kernel.Bindings {
			access := "read"
			if binding.Access == ir.Mutable || types.ContainsAtomic(binding.Type) {
				access = "read_write"
			}
			kind, name := "buffer", binding.Type.String()
			if binding.Texture {
				kind, name = "texture", "rgba8unorm"
			}
			item.Bindings = append(item.Bindings, BindingMeta{Group: 0, Binding: binding.Binding, Access: access, Type: name, MinimumByteSize: binding.MinimumByteSize, Kind: kind})
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
		if plan.View != nil {
			width, err := shapeExpression(program, plan.View.Width)
			if err != nil {
				return nil, err
			}
			height, err := shapeExpression(program, plan.View.Height)
			if err != nil {
				return nil, err
			}
			step, err := stepMetadata(program, executable.PhysicalKernels, plan.View.Step)
			if err != nil {
				return nil, err
			}
			view := &ViewMeta{Format: "srgb8", Step: step, Width: width, Height: height, OutputColor: plan.View.OutputColor, Output: plan.View.Output, Fused: plan.View.Fused}
			item.View = view
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
	case flow.ValueShape:
		expression, err := shapeExpression(program, argument.Shape)
		return ValueSource{Kind: "shape", Expression: &expression}, err
	case flow.ValueRepeat:
		return ValueSource{Kind: "repeat"}, nil
	case flow.ValueConstant:
		return ValueSource{}, fmt.Errorf("compile-time constant reached runtime metadata")
	default:
		return ValueSource{}, fmt.Errorf("invalid value source")
	}
}

func ValidateMetadata(metadata *Metadata) error {
	if metadata == nil || metadata.Schema != 2 {
		return fmt.Errorf("metadata schema/programs are invalid")
	}
	if metadata.Targets.Web == nil || metadata.Targets.SPIRV == nil {
		return fmt.Errorf("metadata must contain web and SPIR-V target plans")
	}
	if metadata.Targets.Web.Vulkan != "" || metadata.Targets.Web.SPIRV != "" || !validFeatures(metadata.Targets.Web.Features, backend.ShaderF16) {
		return fmt.Errorf("web target contains a Vulkan profile")
	}
	spv := metadata.Targets.SPIRV
	if spv.Vulkan != backend.VulkanVersion || spv.SPIRV != backend.SPIRVVersion || !validFeatures(spv.Features, backend.Synchronization2, backend.ZeroInitializeWorkgroupMemory, backend.VulkanMemoryModel, backend.ShaderFloat16, backend.StorageBuffer16BitAccess, backend.UniformAndStorage16BitAccess) || len(spv.Features) < 3 || spv.Features[0] != backend.Synchronization2 || spv.Features[1] != backend.ZeroInitializeWorkgroupMemory || spv.Features[2] != backend.VulkanMemoryModel || len(spv.Features) > 3 && spv.Features[3] != backend.ShaderFloat16 {
		return fmt.Errorf("SPIR-V target profile is invalid")
	}
	if (len(metadata.Targets.Web.Features) > 0) != (len(spv.Features) > 3) {
		return fmt.Errorf("target Float16 requirements differ")
	}
	for targetIndex, target := range []*TargetPlanMeta{metadata.Targets.Web, metadata.Targets.SPIRV} {
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
			if (program.View != nil) != metadata.Programs[i].View {
				return fmt.Errorf("public and target view contracts differ")
			}
			if program.View != nil {
				view := program.View
				if view.Format != "srgb8" || view.Step.Kind != "dispatch" || view.Step.Kernel < 0 || view.Step.Kernel >= len(target.Kernels) || view.OutputColor < 0 || view.Width.Op == "" || view.Height.Op == "" {
					return fmt.Errorf("invalid view plan")
				}
				kernel := target.Kernels[view.Step.Kernel]
				outputKind := "buffer"
				if targetIndex == 0 {
					outputKind = "texture"
				}
				if view.Output >= uint32(len(kernel.Bindings)) || kernel.Bindings[view.Output].Kind != outputKind {
					return fmt.Errorf("invalid view projection kernel")
				}
				for _, resource := range view.Step.Resources {
					if resource.Binding == view.Output {
						return fmt.Errorf("view output is also an input")
					}
				}
				if !view.Fused && (len(kernel.Bindings) != 2 || len(view.Step.Resources) != 1 || kernel.ParameterBlock == nil || len(kernel.ParameterBlock.Fields) != 2) {
					return fmt.Errorf("invalid standalone view projection")
				}
			}
		}
		for i, kernel := range target.Kernels {
			if kernel.EntryPoint != fmt.Sprintf("_tach_k%d", i) || kernel.WorkgroupSize[0] == 0 || kernel.WorkgroupSize[1] == 0 || kernel.WorkgroupSize[2] == 0 {
				return fmt.Errorf("invalid physical kernel %d", i)
			}
			for j, binding := range kernel.Bindings {
				if binding.Group != 0 || binding.Binding != uint32(j) || binding.Kind != "buffer" && binding.Kind != "texture" {
					return fmt.Errorf("kernel %d bindings are not dense", i)
				}
			}
		}
	}
	return nil
}

func validFeatures(features []string, allowed ...string) bool {
	last := -1
	for _, feature := range features {
		index := -1
		for i, candidate := range allowed {
			if feature == candidate {
				index = i
				break
			}
		}
		if index <= last {
			return false
		}
		last = index
	}
	return true
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
	case types.F16:
		out.Kind = "f16"
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
