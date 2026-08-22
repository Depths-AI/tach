package wgsl

import (
	"fmt"
	"tach/foundation"
	"tach/ir"
)

const shaderF16 = "shader-f16"

type storageBinding struct {
	Buffer          int
	Binding         uint32
	Access          ir.Access
	Type            *foundation.Type
	MinimumByteSize uint32
	Texture         bool
}

type physicalKernel struct {
	Entry          string
	Function       *ir.Function
	Workgroup      [3]uint32
	Bindings       []storageBinding
	Parameters     *ir.HostParameterBlock
	Coordinates    *ir.Coordinates
	Projection     bool
	FusedView      bool
	ViewBinding    int
	ViewWidth      ir.ValueID
	ViewHeight     ir.ValueID
	LogicalLengths map[int]ir.ValueID
}

type resourceSourceKind uint8

const (
	externalSource resourceSourceKind = iota + 1
	transientSource
)

type resourceSource struct {
	Binding  uint32
	Kind     resourceSourceKind
	Resource int
}

type stepKind uint8

const dispatchStepKind stepKind = 1

type step struct {
	Kind       stepKind
	Kernel     int
	Domain     []ir.ShapeID
	Resources  []resourceSource
	Parameters []ir.ValueArgument
}

type transient struct {
	Type            *foundation.Type
	Stride          uint32
	Alignment       uint32
	MinimumByteSize uint32
	Length          ir.ShapeID
	Color           int
	FirstStep       int
	LastStep        int
}

type repeatMode uint8

const (
	repeatProgram repeatMode = iota + 1
	repeatInvocationLoop
)

type programPlan struct {
	Program    int
	Transients []transient
	Steps      []step
	Repeat     repeatMode
	View       *viewPlan
}

type viewPlan struct {
	step        step
	Width       ir.ShapeID
	Height      ir.ShapeID
	OutputColor int
	Output      uint32
	Fused       bool
}

type plan struct {
	Logical         *ir.Module
	KernelModule    *ir.KernelModule
	PhysicalKernels []physicalKernel
	Programs        []programPlan
}

func requiredFeatures(executable *plan) []string {
	if ir.UsesKind(executable.KernelModule, foundation.Float16Kind) {
		return []string{shaderF16}
	}
	return nil
}

func planModule(logical *ir.Module) (*plan, error) {
	cloned := ir.Clone(logical)
	executable := &plan{Logical: cloned, KernelModule: &ir.KernelModule{Structs: append([]*foundation.Type(nil), cloned.Kernel.Structs...)}}
	for _, function := range cloned.Kernel.Functions {
		if function.Kind == ir.Helper {
			executable.KernelModule.Functions = append(executable.KernelModule.Functions, ir.CloneFunction(function))
		}
	}
	for _, program := range cloned.Programs {
		if program.View != nil {
			executable.KernelModule.Functions = append(executable.KernelModule.Functions, ir.ViewHelpers()...)
			break
		}
	}
	viewKernel := -1
	for programIndex, program := range cloned.Programs {
		plan := programPlan{Program: programIndex, Repeat: repeatProgram}
		fusedDispatch, fusedBinding := ir.FusibleView(cloned, program)
		omitted := ir.ResourceID(0)
		if fusedDispatch >= 0 {
			omitted = program.View.Source
		}
		kernelForDispatch := make([]int, len(program.Dispatches))
		valuesForDispatch := make([][]ir.ValueArgument, len(program.Dispatches))
		invocationRepeat := program.View == nil && len(program.Dispatches) == 1 && ir.CanInternalizeRepeat(cloned.Kernel.Function(program.Dispatches[0].Stage))
		for dispatchIndex, dispatch := range program.Dispatches {
			stage := cloned.Kernel.Function(dispatch.Stage)
			if stage == nil {
				return nil, fmt.Errorf("program %s dispatch references missing stage %s", program.Name, dispatch.Stage)
			}
			function := ir.CloneFunction(stage)
			values := append([]ir.ValueArgument(nil), dispatch.Values...)
			if invocationRepeat {
				if err := ir.InternalizeRepeat(function); err != nil {
					return nil, err
				}
				values = append(values, ir.ValueArgument{Formal: len(values), Kind: ir.ValueFromRepeat})
			}
			values, err := ir.SpecializeParameters(function, values)
			if err != nil {
				return nil, err
			}
			logicalLengths := ir.AppendLogicalLengths(function, &values, program, &dispatch)
			var viewWidth, viewHeight ir.ValueID
			if dispatchIndex == fusedDispatch {
				if err := ir.FuseView(function, fusedBinding); err != nil {
					return nil, err
				}
				viewWidth, viewHeight = ir.AppendViewExtent(function, &values, program.View)

			}
			valuesForDispatch[dispatchIndex] = values
			kernelIndex := len(executable.PhysicalKernels)
			function.Name = ir.PrivateEntryName(kernelIndex)
			workgroup, err := chooseWorkgroup(function)
			if err != nil {
				return nil, err
			}
			physical := physicalKernel{Entry: function.Name, Function: function, Workgroup: workgroup, LogicalLengths: logicalLengths}
			for buffer, parameter := range function.BufferParams {
				minimum, err := foundation.MinimumByteSize(parameter.Type)
				if err != nil {
					return nil, fmt.Errorf("kernel %s buffer %s: %w", function.Name, parameter.Name, err)
				}
				physical.Bindings = append(physical.Bindings, storageBinding{Buffer: buffer, Binding: uint32(buffer), Access: parameter.Access, Type: parameter.Type, MinimumByteSize: minimum})
			}
			if dispatchIndex == fusedDispatch {
				physical.FusedView, physical.ViewBinding = true, fusedBinding
				physical.ViewWidth, physical.ViewHeight = viewWidth, viewHeight
				physical.Bindings[fusedBinding].Texture = true
				physical.Bindings[fusedBinding].MinimumByteSize = 0
			}
			physical.Parameters, err = ir.PlanHostParameters(function, uint32(len(physical.Bindings)))
			if err != nil {
				return nil, err
			}
			physical.Coordinates, err = ir.ResolveCoordinates(function)
			if err != nil {
				return nil, err
			}
			ir.OptimizeCoordinates(function, workgroup, physical.Coordinates)
			executable.KernelModule.Functions = append(executable.KernelModule.Functions, function)
			executable.PhysicalKernels = append(executable.PhysicalKernels, physical)
			kernelForDispatch[dispatchIndex] = kernelIndex
		}
		plan.Transients = planTransients(program, omitted)
		for dispatchIndex, dispatch := range program.Dispatches {
			step := step{Kind: dispatchStepKind, Kernel: kernelForDispatch[dispatchIndex], Domain: append([]ir.ShapeID(nil), dispatch.Domain...), Parameters: valuesForDispatch[dispatchIndex]}
			for _, argument := range dispatch.Buffers {
				resource := program.Resource(argument.Resource)
				source := resourceSource{Binding: uint32(argument.Formal), Resource: resourceIndex(program, resource, omitted)}
				if resource.Kind == ir.ExternalResourceKind {
					source.Kind = externalSource
				} else {
					source.Kind = transientSource
				}
				step.Resources = append(step.Resources, source)
			}
			plan.Steps = append(plan.Steps, step)
		}
		if invocationRepeat {
			plan.Repeat = repeatInvocationLoop
		}
		if program.View != nil {
			view := program.View
			color := 0
			for _, transient := range plan.Transients {
				if transient.Color >= color {
					color = transient.Color + 1
				}
			}
			terminal := &viewPlan{Width: view.Width, Height: view.Height, OutputColor: color}
			if fusedDispatch >= 0 {
				step := plan.Steps[len(plan.Steps)-1]
				plan.Steps = plan.Steps[:len(plan.Steps)-1]
				for index, resource := range step.Resources {
					if resource.Binding == uint32(fusedBinding) {
						step.Resources = append(step.Resources[:index], step.Resources[index+1:]...)
						break
					}
				}
				terminal.step, terminal.Output, terminal.Fused = step, uint32(fusedBinding), true
			} else {
				resource := program.Resource(view.Source)
				source := resourceSource{Binding: 0, Resource: resourceIndex(program, resource, omitted), Kind: externalSource}
				if resource.Kind == ir.TransientResourceKind {
					source.Kind = transientSource
				}
				if viewKernel < 0 {
					kernel, err := projectionKernel()
					if err != nil {
						return nil, err
					}
					viewKernel = len(executable.PhysicalKernels)
					kernel.Entry = ir.PrivateEntryName(viewKernel)
					kernel.Function.Name = kernel.Entry
					executable.PhysicalKernels = append(executable.PhysicalKernels, kernel)
					executable.KernelModule.Functions = append(executable.KernelModule.Functions, kernel.Function)
				}
				terminal.Output, terminal.step = 1, step{Kind: dispatchStepKind, Kernel: viewKernel, Domain: []ir.ShapeID{view.Width, view.Height}, Resources: []resourceSource{source}, Parameters: []ir.ValueArgument{{Formal: 0, Kind: ir.ValueFromShape, Shape: view.Width}, {Formal: 1, Kind: ir.ValueFromShape, Shape: view.Height}}}
			}
			plan.View = terminal
		}
		executable.Programs = append(executable.Programs, plan)
	}
	if err := verify(executable); err != nil {
		return nil, err
	}
	return executable, nil
}

func chooseWorkgroup(function *ir.Function) ([3]uint32, error) {
	size, valid := ir.ChooseWorkgroup(function, [3]uint32{256, 256, 64}, 256)
	if !valid {
		return size, fmt.Errorf("stage %s workgroup exceeds WebGPU limits", function.Name)
	}
	return size, nil
}

func resourceIndex(program *ir.Program, resource *ir.Resource, omitted ir.ResourceID) int {
	index := 0
	for _, candidate := range program.Resources {
		if candidate.ID == omitted {
			continue
		}
		if candidate.Kind == resource.Kind {
			if candidate.ID == resource.ID {
				return index
			}
			index++
		}
	}
	return -1
}

func planTransients(program *ir.Program, omitted ir.ResourceID) []transient {
	out := make([]transient, 0)
	for _, lifetime := range ir.AnalyzeTransientLifetimes(program, omitted) {
		resource := program.Resource(lifetime.Resource)
		layout, _ := foundation.LayoutOf(resource.Type)
		out = append(out, transient{Type: resource.Type, Stride: layout.Stride, Alignment: layout.Align, MinimumByteSize: layout.Stride, Length: resource.Length, Color: lifetime.Color, FirstStep: lifetime.FirstStep, LastStep: lifetime.LastStep})
	}
	return out
}

func projectionKernel() (physicalKernel, error) {
	function := ir.ViewProjectionFunction()
	physical := physicalKernel{Function: function, Workgroup: function.Workgroup.Size, Projection: true}
	physical.Bindings = []storageBinding{
		{Buffer: 0, Binding: 0, Access: ir.Read, Type: function.BufferParams[0].Type, MinimumByteSize: 16},
		{Buffer: 1, Binding: 1, Access: ir.Mutable, Type: function.BufferParams[1].Type, MinimumByteSize: 0, Texture: true},
	}
	var err error
	physical.Parameters, err = ir.PlanHostParameters(function, 2)
	if err != nil {
		return physicalKernel{}, err
	}
	physical.Coordinates, err = ir.ResolveCoordinates(function)
	return physical, err
}
