package spirv

import (
	"encoding/json"
	"fmt"

	"tach/foundation"
	"tach/ir"
)

type runtimeMetadata struct {
	Vulkan   string            `json:"vulkan"`
	SPIRV    string            `json:"spirv"`
	Features []string          `json:"features,omitempty"`
	Kernels  []kernelMetadata  `json:"kernels"`
	Programs []programMetadata `json:"programs"`
}
type kernelMetadata struct {
	EntryPoint     string                  `json:"entryPoint"`
	WorkgroupSize  [3]uint32               `json:"workgroupSize"`
	Bindings       []bindingMetadata       `json:"bindings"`
	ParameterBlock *parameterBlockMetadata `json:"parameterBlock,omitempty"`
}
type bindingMetadata struct {
	Group           uint32 `json:"group"`
	Binding         uint32 `json:"binding"`
	Access          string `json:"access"`
	Type            string `json:"type"`
	MinimumByteSize uint32 `json:"minimumByteSize"`
	Kind            string `json:"kind"`
}
type parameterBlockMetadata struct {
	Group    uint32                   `json:"group"`
	Binding  uint32                   `json:"binding"`
	ByteSize uint32                   `json:"byteSize"`
	Fields   []parameterFieldMetadata `json:"fields"`
}
type parameterFieldMetadata struct {
	Type       string      `json:"type"`
	ByteOffset uint32      `json:"byteOffset"`
	Layout     *hostLayout `json:"layout"`
}
type programMetadata struct {
	Program       int                 `json:"program"`
	Transients    []transientMetadata `json:"transients"`
	Steps         []stepMetadata      `json:"steps"`
	RepeatBarrier *stepMetadata       `json:"repeatBarrier,omitempty"`
	Repeat        string              `json:"repeat"`
	View          *viewMetadata       `json:"view,omitempty"`
}
type viewMetadata struct {
	Format      string          `json:"format"`
	Step        stepMetadata    `json:"step"`
	Width       shapeExpression `json:"width"`
	Height      shapeExpression `json:"height"`
	OutputColor int             `json:"outputColor"`
	Output      uint32          `json:"output"`
	Fused       bool            `json:"fused"`
}
type transientMetadata struct {
	Type            string          `json:"type"`
	Stride          uint32          `json:"stride"`
	Alignment       uint32          `json:"alignment"`
	MinimumByteSize uint32          `json:"minimumByteSize"`
	Length          shapeExpression `json:"length"`
	Color           int             `json:"color"`
	FirstStep       int             `json:"firstStep"`
	LastStep        int             `json:"lastStep"`
}
type stepMetadata struct {
	Kind       string             `json:"kind"`
	Kernel     int                `json:"kernel"`
	Domain     []shapeExpression  `json:"domain,omitempty"`
	Resources  []resourceMetadata `json:"resources"`
	Parameters []valueSource      `json:"parameters,omitempty"`
}
type resourceMetadata struct {
	Binding  uint32 `json:"binding"`
	Kind     string `json:"kind"`
	Resource int    `json:"resource"`
}
type shapeExpression struct {
	Op        string           `json:"op"`
	Value     uint32           `json:"value,omitempty"`
	Parameter int              `json:"parameter"`
	Resource  int              `json:"resource"`
	Path      []string         `json:"path,omitempty"`
	Axis      uint8            `json:"axis"`
	Left      *shapeExpression `json:"left,omitempty"`
	Right     *shapeExpression `json:"right,omitempty"`
}
type valueSource struct {
	Kind       string           `json:"kind"`
	Parameter  int              `json:"parameter"`
	Path       []string         `json:"path,omitempty"`
	Expression *shapeExpression `json:"expression,omitempty"`
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

func encodeRuntime(executable *plan) ([]byte, error) {
	metadata, err := describePlan(executable)
	if err != nil {
		return nil, err
	}
	if err := validateRuntime(metadata, executable.Logical); err != nil {
		return nil, err
	}
	return json.Marshal(metadata)
}

func describePlan(executable *plan) (*runtimeMetadata, error) {
	target := &runtimeMetadata{Vulkan: vulkanVersion, SPIRV: spirvVersion, Features: requiredFeatures(executable), Kernels: []kernelMetadata{}, Programs: []programMetadata{}}
	for _, kernel := range executable.PhysicalKernels {
		item := kernelMetadata{EntryPoint: kernel.Entry, WorkgroupSize: kernel.Workgroup, Bindings: []bindingMetadata{}}
		for _, binding := range kernel.Bindings {
			access := "read"
			if binding.Access == ir.Mutable || foundation.ContainsAtomic(binding.Type) {
				access = "read_write"
			}
			item.Bindings = append(item.Bindings, bindingMetadata{Group: 0, Binding: binding.Binding, Access: access, Type: binding.Type.String(), MinimumByteSize: binding.MinimumByteSize, Kind: "buffer"})
		}
		parameterBlock, err := describeParameterBlock(kernel.Parameters)
		if err != nil {
			return nil, err
		}
		item.ParameterBlock = parameterBlock
		target.Kernels = append(target.Kernels, item)
	}
	for _, plan := range executable.Programs {
		program := executable.Logical.Programs[plan.Program]
		item := programMetadata{Program: plan.Program, Transients: []transientMetadata{}, Steps: []stepMetadata{}, Repeat: "program"}
		if plan.Repeat == repeatInvocationLoop {
			item.Repeat = "invocation-loop"
		}
		for _, transient := range plan.Transients {
			length, err := describeShape(program, transient.Length)
			if err != nil {
				return nil, err
			}
			item.Transients = append(item.Transients, transientMetadata{Type: transient.Type.String(), Stride: transient.Stride, Alignment: transient.Alignment, MinimumByteSize: transient.MinimumByteSize, Length: length, Color: transient.Color, FirstStep: transient.FirstStep, LastStep: transient.LastStep})
		}
		for _, step := range plan.Steps {
			description, err := describeStep(program, executable.PhysicalKernels, step)
			if err != nil {
				return nil, err
			}
			item.Steps = append(item.Steps, description)
		}
		if len(plan.RepeatBarrier) > 0 {
			barrier := describeBarrier(plan.RepeatBarrier)
			item.RepeatBarrier = &barrier
		}
		if plan.View != nil {
			step, err := describeStep(program, executable.PhysicalKernels, plan.View.step)
			if err != nil {
				return nil, err
			}
			width, err := describeShape(program, plan.View.Width)
			if err != nil {
				return nil, err
			}
			height, err := describeShape(program, plan.View.Height)
			if err != nil {
				return nil, err
			}
			item.View = &viewMetadata{Format: "srgb8", Step: step, Width: width, Height: height, OutputColor: plan.View.OutputColor, Output: plan.View.Output, Fused: plan.View.Fused}
		}
		target.Programs = append(target.Programs, item)
	}
	return target, nil
}

func describeStep(program *ir.Program, kernels []physicalKernel, step step) (stepMetadata, error) {
	if step.Kind == barrierStepKind {
		return describeBarrier(step.Barrier), nil
	}
	domain, err := describeDomain(program, step.Domain)
	if err != nil {
		return stepMetadata{}, err
	}
	out := stepMetadata{Kind: "dispatch", Kernel: step.Kernel, Domain: domain, Resources: []resourceMetadata{}}
	for _, resource := range step.Resources {
		kind := "external"
		if resource.Kind == transientSource {
			kind = "transient"
		}
		out.Resources = append(out.Resources, resourceMetadata{Binding: resource.Binding, Kind: kind, Resource: resource.Resource})
	}
	if block := kernels[step.Kernel].Parameters; block != nil {
		out.Parameters = make([]valueSource, 0, len(block.Fields))
		for _, field := range block.Fields {
			if field.Parameter < 0 || field.Parameter >= len(step.Parameters) {
				return stepMetadata{}, fmt.Errorf("kernel parameter field references missing dispatch value")
			}
			source, err := describeValue(program, step.Parameters[field.Parameter], field.Path)
			if err != nil {
				return stepMetadata{}, err
			}
			out.Parameters = append(out.Parameters, source)
		}
	} else {
		out.Parameters = []valueSource{}
	}
	return out, err
}

func describeBarrier(resources []barrierResource) stepMetadata {
	out := stepMetadata{Kind: "barrier", Resources: []resourceMetadata{}}
	for _, resource := range resources {
		kind := "external"
		if resource.Kind == transientSource {
			kind = "transient"
		}
		out.Resources = append(out.Resources, resourceMetadata{Kind: kind, Resource: resource.Resource})
	}
	return out
}

// DECISION: Physical kernels are one-per-surviving-dispatch for deterministic,
// unambiguous plans. Deduplicate by verified Kernel IR hash only if real modules
// show shader-size/pipeline duplication worth the extra identity machinery.
