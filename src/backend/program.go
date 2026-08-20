package backend

import (
	"fmt"

	"tach/src/abi"
	"tach/src/flow"
	"tach/src/ir"
	"tach/src/layout"
	"tach/src/types"
)

type Target string

const (
	Web   Target = "web"
	SPIRV Target = "spirv"

	VulkanVersion             = "1.3"
	SPIRVVersion              = "1.6"
	SPIRVBinaryVersion uint32 = 0x00010600

	Synchronization2              = "synchronization2"
	ZeroInitializeWorkgroupMemory = "shaderZeroInitializeWorkgroupMemory"
	VulkanMemoryModel             = "vulkanMemoryModel"
	ShaderF16                     = "shader-f16"
	ShaderFloat16                 = "shaderFloat16"
	StorageBuffer16BitAccess      = "storageBuffer16BitAccess"
	UniformAndStorage16BitAccess  = "uniformAndStorageBuffer16BitAccess"
	unitHelper                    = "$tach_unit"
	srgbHelper                    = "$tach_srgb"
)

type Profile struct {
	Target             Target
	MaxWorkgroup       [3]uint32
	MaxInvocations     uint32
	MaxStorageBindings int
	MaxUniformBytes    uint32
	MaxSharedBytes     uint32
}

var WebProfile = Profile{Target: Web, MaxWorkgroup: [3]uint32{256, 256, 64}, MaxInvocations: 256, MaxStorageBindings: 8, MaxUniformBytes: 16 * 1024, MaxSharedBytes: 16 * 1024}
var SPIRVProfile = Profile{Target: SPIRV, MaxWorkgroup: [3]uint32{256, 256, 64}, MaxInvocations: 256, MaxStorageBindings: 8, MaxUniformBytes: 16 * 1024, MaxSharedBytes: 16 * 1024}

type StorageBinding struct {
	Buffer          int
	Binding         uint32
	Access          ir.Access
	Type            *types.Type
	MinimumByteSize uint32
	Texture         bool
}

type PhysicalKernel struct {
	Entry          string
	Function       *ir.Function
	Workgroup      [3]uint32
	Bindings       []StorageBinding
	Parameters     *abi.ParameterBlock
	Coordinates    *Coordinates
	Projection     bool
	FusedView      bool
	ViewBinding    int
	ViewWidth      ir.ValueID
	ViewHeight     ir.ValueID
	LogicalLengths map[int]ir.ValueID
}

type ResourceSourceKind uint8

const (
	ExternalSource ResourceSourceKind = iota + 1
	TransientSource
)

type ResourceSource struct {
	Binding  uint32
	Kind     ResourceSourceKind
	Resource int
}

type StepKind uint8

const (
	DispatchStepKind StepKind = iota + 1
	BarrierStepKind
)

type BarrierResource struct {
	Kind     ResourceSourceKind
	Resource int
}

type Step struct {
	Kind       StepKind
	Kernel     int
	Domain     []flow.ShapeID
	Resources  []ResourceSource
	Parameters []flow.ValueArgument
	Barrier    []BarrierResource
}

type Transient struct {
	Resource        flow.ResourceID
	Type            *types.Type
	Stride          uint32
	Alignment       uint32
	MinimumByteSize uint32
	Length          flow.ShapeID
	Color           int
	FirstStep       int
	LastStep        int
}

type RepeatMode uint8

const (
	RepeatProgram RepeatMode = iota + 1
	RepeatInvocationLoop
)

type ProgramPlan struct {
	Program       int
	Transients    []Transient
	Steps         []Step
	RepeatBarrier []BarrierResource
	Repeat        RepeatMode
	View          *ViewPlan
}

type ViewPlan struct {
	Step        Step
	Width       flow.ShapeID
	Height      flow.ShapeID
	OutputColor int
	Output      uint32
	Fused       bool
}

type Executable struct {
	Target          Target
	Logical         *flow.Module
	KernelModule    *ir.Module
	PhysicalKernels []PhysicalKernel
	Programs        []ProgramPlan
}

func RequiredFeatures(executable *Executable) []string {
	if executable.Target == Web {
		if ir.UsesKind(executable.KernelModule, types.F16) {
			return []string{ShaderF16}
		}
		return nil
	}
	features := []string{Synchronization2, ZeroInitializeWorkgroupMemory, VulkanMemoryModel}
	if !ir.UsesKind(executable.KernelModule, types.F16) {
		return features
	}
	features = append(features, ShaderFloat16)
	storage, uniform := false, false
	for _, kernel := range executable.PhysicalKernels {
		for _, binding := range kernel.Bindings {
			storage = storage || types.Contains(binding.Type, types.F16)
		}
		uniform = uniform || kernel.Parameters != nil && types.Contains(kernel.Parameters.Type, types.F16)
	}
	if storage {
		features = append(features, StorageBuffer16BitAccess)
	}
	if uniform {
		features = append(features, UniformAndStorage16BitAccess)
	}
	return features
}

func (e *Executable) IndexFunctions() (map[*ir.Function]*Coordinates, map[*ir.Function]*PhysicalKernel, error) {
	coordinates := map[*ir.Function]*Coordinates{}
	kernels := map[*ir.Function]*PhysicalKernel{}
	for i := range e.PhysicalKernels {
		kernel := &e.PhysicalKernels[i]
		coordinates[kernel.Function] = kernel.Coordinates
		kernels[kernel.Function] = kernel
	}
	for _, function := range e.KernelModule.Functions {
		if coordinates[function] != nil {
			continue
		}
		lowered, err := LowerCoordinates(function)
		if err != nil {
			return nil, nil, err
		}
		coordinates[function] = lowered
	}
	return coordinates, kernels, nil
}

func Lower(logical *flow.Module, profile Profile) (*Executable, error) {
	if profile.Target != Web && profile.Target != SPIRV {
		return nil, fmt.Errorf("invalid target profile %q", profile.Target)
	}
	cloned := flow.Clone(logical)
	executable := &Executable{Target: profile.Target, Logical: cloned, KernelModule: &ir.Module{Structs: append([]*types.Type(nil), cloned.Kernel.Structs...)}}
	for _, function := range cloned.Kernel.Functions {
		if function.Kind == ir.Helper {
			executable.KernelModule.Functions = append(executable.KernelModule.Functions, cloneFunction(function))
		}
	}
	for _, program := range cloned.Programs {
		if program.View != nil {
			executable.KernelModule.Functions = append(executable.KernelModule.Functions, unitFunction(), srgbFunction())
			break
		}
	}
	viewKernel := -1
	for programIndex, program := range cloned.Programs {
		plan := ProgramPlan{Program: programIndex, Repeat: RepeatProgram}
		fusedDispatch, fusedBinding := fusibleView(cloned, program)
		omitted := flow.ResourceID(0)
		if fusedDispatch >= 0 {
			omitted = program.View.Source
		}
		kernelForDispatch := make([]int, len(program.Dispatches))
		valuesForDispatch := make([][]flow.ValueArgument, len(program.Dispatches))
		invocationRepeat := program.View == nil && len(program.Dispatches) == 1 && canInternalizeRepeat(cloned.Kernel.Function(program.Dispatches[0].Stage))
		for dispatchIndex, dispatch := range program.Dispatches {
			stage := cloned.Kernel.Function(dispatch.Stage)
			if stage == nil {
				return nil, fmt.Errorf("program %s dispatch references missing stage %s", program.Name, dispatch.Stage)
			}
			function := cloneFunction(stage)
			values := append([]flow.ValueArgument(nil), dispatch.Values...)
			if invocationRepeat {
				if err := internalizeRepeat(function); err != nil {
					return nil, err
				}
				values = append(values, flow.ValueArgument{Formal: len(values), Kind: flow.ValueRepeat})
			}
			values = pruneUnusedParameters(function, values)
			logicalLengths := appendLogicalLengths(function, &values, program, &dispatch)
			var viewWidth, viewHeight ir.ValueID
			if dispatchIndex == fusedDispatch {
				if err := fuseView(function, fusedBinding); err != nil {
					return nil, err
				}
				if profile.Target == Web {
					viewWidth, viewHeight = appendViewExtent(function, &values, program.View)
				}
			}
			valuesForDispatch[dispatchIndex] = values
			kernelIndex := len(executable.PhysicalKernels)
			function.Name = abi.PrivateEntry(kernelIndex)
			workgroup, err := chooseWorkgroup(function, profile)
			if err != nil {
				return nil, err
			}
			physical := PhysicalKernel{Entry: function.Name, Function: function, Workgroup: workgroup, LogicalLengths: logicalLengths}
			for buffer, parameter := range function.BufferParams {
				minimum, err := minimumByteSize(parameter.Type)
				if err != nil {
					return nil, fmt.Errorf("kernel %s buffer %s: %w", function.Name, parameter.Name, err)
				}
				physical.Bindings = append(physical.Bindings, StorageBinding{Buffer: buffer, Binding: uint32(buffer), Access: parameter.Access, Type: parameter.Type, MinimumByteSize: minimum})
			}
			if dispatchIndex == fusedDispatch {
				physical.FusedView, physical.ViewBinding = true, fusedBinding
				physical.ViewWidth, physical.ViewHeight = viewWidth, viewHeight
				if profile.Target == Web {
					physical.Bindings[fusedBinding].Texture = true
					physical.Bindings[fusedBinding].MinimumByteSize = 0
				}
			}
			physical.Parameters, err = abi.PlanParameters(function, uint32(len(physical.Bindings)))
			if err != nil {
				return nil, err
			}
			physical.Coordinates, err = LowerCoordinates(function)
			if err != nil {
				return nil, err
			}
			OptimizeCoordinates(function, workgroup, physical.Coordinates)
			executable.KernelModule.Functions = append(executable.KernelModule.Functions, function)
			executable.PhysicalKernels = append(executable.PhysicalKernels, physical)
			kernelForDispatch[dispatchIndex] = kernelIndex
		}
		plan.Transients = planTransients(program, omitted)
		for dispatchIndex, dispatch := range program.Dispatches {
			if profile.Target == SPIRV && dispatchIndex > 0 {
				if barrier := between(program, program.Dispatches[dispatchIndex-1], dispatch, omitted); len(barrier) > 0 {
					plan.Steps = append(plan.Steps, Step{Kind: BarrierStepKind, Barrier: barrier})
				}
			}
			step := Step{Kind: DispatchStepKind, Kernel: kernelForDispatch[dispatchIndex], Domain: append([]flow.ShapeID(nil), dispatch.Domain...), Parameters: valuesForDispatch[dispatchIndex]}
			for _, argument := range dispatch.Buffers {
				resource := program.Resource(argument.Resource)
				source := ResourceSource{Binding: uint32(argument.Formal), Resource: resourceIndex(program, resource, omitted)}
				if resource.Kind == flow.External {
					source.Kind = ExternalSource
				} else {
					source.Kind = TransientSource
				}
				step.Resources = append(step.Resources, source)
			}
			plan.Steps = append(plan.Steps, step)
		}
		if invocationRepeat {
			plan.Repeat = RepeatInvocationLoop
		}
		if profile.Target == SPIRV && program.View == nil && len(program.Dispatches) > 0 {
			plan.RepeatBarrier = between(program, program.Dispatches[len(program.Dispatches)-1], program.Dispatches[0], omitted)
		}
		if program.View != nil {
			view := program.View
			color := 0
			for _, transient := range plan.Transients {
				if transient.Color >= color {
					color = transient.Color + 1
				}
			}
			terminal := &ViewPlan{Width: view.Width, Height: view.Height, OutputColor: color}
			if fusedDispatch >= 0 {
				step := plan.Steps[len(plan.Steps)-1]
				plan.Steps = plan.Steps[:len(plan.Steps)-1]
				if len(plan.Steps) > 0 && plan.Steps[len(plan.Steps)-1].Kind == BarrierStepKind {
					plan.Steps = plan.Steps[:len(plan.Steps)-1]
				}
				for index, resource := range step.Resources {
					if resource.Binding == uint32(fusedBinding) {
						step.Resources = append(step.Resources[:index], step.Resources[index+1:]...)
						break
					}
				}
				terminal.Step, terminal.Output, terminal.Fused = step, uint32(fusedBinding), true
			} else {
				if viewKernel < 0 {
					kernel, err := projectionKernel(profile.Target)
					if err != nil {
						return nil, err
					}
					viewKernel = len(executable.PhysicalKernels)
					kernel.Entry = abi.PrivateEntry(viewKernel)
					kernel.Function.Name = kernel.Entry
					executable.PhysicalKernels = append(executable.PhysicalKernels, kernel)
					executable.KernelModule.Functions = append(executable.KernelModule.Functions, kernel.Function)
				}
				resource := program.Resource(view.Source)
				source := ResourceSource{Binding: 0, Resource: resourceIndex(program, resource, omitted), Kind: ExternalSource}
				if resource.Kind == flow.Transient {
					source.Kind = TransientSource
				}
				terminal.Step = Step{Kind: DispatchStepKind, Kernel: viewKernel, Domain: []flow.ShapeID{view.Width, view.Height}, Resources: []ResourceSource{source}, Parameters: []flow.ValueArgument{{Formal: 0, Kind: flow.ValueShape, Shape: view.Width}, {Formal: 1, Kind: flow.ValueShape, Shape: view.Height}}}
				terminal.Output = 1
			}
			plan.View = terminal
		}
		executable.Programs = append(executable.Programs, plan)
	}
	if err := Verify(executable); err != nil {
		return nil, err
	}
	return executable, nil
}

func appendLogicalLengths(function *ir.Function, values *[]flow.ValueArgument, program *flow.Program, dispatch *flow.Dispatch) map[int]ir.ValueID {
	lengths := map[int]ir.ValueID{}
	next := maxValue(function) + 1
	for buffer, parameter := range function.BufferParams {
		path, ok := f16RuntimePath(parameter.Type)
		if !ok || !usesBufferLength(function.Body, buffer, map[ir.PlaceID]bool{}) {
			continue
		}
		for _, argument := range dispatch.Buffers {
			if argument.Formal != buffer {
				continue
			}
			shape := program.AddShape(flow.Shape{Op: flow.ShapeResourceLength, Resource: argument.Resource, Path: path, Span: dispatch.Span})
			formal := len(function.Params)
			function.Params = append(function.Params, ir.Param{Name: fmt.Sprintf("__tach_length_%d", buffer), ID: next, Type: types.TU32})
			function.SourceParams = append(function.SourceParams, ir.SourceParam{Name: function.Params[formal].Name, Kind: ir.SourceValue, Value: next, Buffer: -1})
			*values = append(*values, flow.ValueArgument{Formal: formal, Kind: flow.ValueShape, Shape: shape})
			lengths[buffer], next = next, next+1
			break
		}
	}
	return lengths
}

func f16RuntimePath(t *types.Type) ([]string, bool) {
	if t.Kind == types.RuntimeArray {
		return nil, t.Elem.Kind == types.F16
	}
	if t.Kind == types.Struct && len(t.Fields) > 0 {
		tail := t.Fields[len(t.Fields)-1]
		if tail.Type.Kind == types.RuntimeArray && tail.Type.Elem.Kind == types.F16 {
			return []string{tail.Name}, true
		}
	}
	return nil, false
}

func usesBufferLength(block *ir.Block, buffer int, places map[ir.PlaceID]bool) bool {
	for _, instruction := range block.Instrs {
		switch item := instruction.(type) {
		case *ir.PlaceRoot:
			places[item.Result] = item.Buffer == buffer
		case *ir.PlaceField:
			places[item.Result] = places[item.Base]
		case *ir.PlaceIndex:
			places[item.Result] = places[item.Base]
		case *ir.ArrayLength:
			if places[item.Place] {
				return true
			}
		case *ir.If:
			if usesBufferLength(item.Then, buffer, places) || usesBufferLength(item.Else, buffer, places) {
				return true
			}
		case *ir.Loop:
			if usesBufferLength(item.Cond, buffer, places) || usesBufferLength(item.Body, buffer, places) {
				return true
			}
		case *ir.Scope:
			if usesBufferLength(item.Body, buffer, places) {
				return true
			}
		}
	}
	return false
}

func canInternalizeRepeat(function *ir.Function) bool {
	if function == nil || containsLoop(function.Body) {
		return false
	}
	summary := ir.AnalyzeAccess(function)
	if summary.Effects.Atomic || summary.Effects.Workgroup || summary.Effects.Barrier {
		return false
	}
	for _, buffer := range summary.Buffers {
		for _, access := range buffer.Accesses {
			if len(access.Indices) != 1 || !access.Indices[0].Exact || access.Indices[0].Constant != 0 || access.Indices[0].Coefficient != [3]int64{1, 0, 0} {
				return false
			}
		}
	}
	return true
}

func containsLoop(block *ir.Block) bool {
	if block == nil {
		return false
	}
	for _, instruction := range block.Instrs {
		switch x := instruction.(type) {
		case *ir.Loop:
			return true
		case *ir.If:
			if containsLoop(x.Then) || containsLoop(x.Else) {
				return true
			}
		case *ir.Scope:
			if containsLoop(x.Body) {
				return true
			}
		}
	}
	return false
}

func internalizeRepeat(function *ir.Function) error {
	next := maxValue(function)
	next++
	repeat := ir.Param{Name: "__tach_repeat", ID: next, Type: types.TU32}
	function.Params = append(function.Params, repeat)
	function.SourceParams = append(function.SourceParams, ir.SourceParam{Name: repeat.Name, Kind: ir.SourceValue, Value: repeat.ID, Buffer: -1})
	if !rewriteReturns(function.Body) {
		return fmt.Errorf("stage %s has a value return", function.Name)
	}
	next++
	zero := next
	next++
	result := next
	next++
	parameter := next
	next++
	condition := next
	next++
	one := next
	next++
	incremented := next
	original := function.Body
	function.Body = &ir.Block{
		Instrs: []ir.Instr{
			&ir.Const{Result: zero, Type: types.TU32, Raw: "0", Span: function.Span},
			&ir.Loop{
				Results: []ir.Result{{ID: result, Type: types.TU32}},
				Params:  []ir.LoopParam{{ID: parameter, Type: types.TU32, Init: zero}},
				Cond: &ir.Block{Instrs: []ir.Instr{
					&ir.Binary{Result: condition, Type: types.TBool, Op: "<", Left: parameter, Right: repeat.ID, Span: function.Span},
				}, Term: &ir.Yield{Values: []ir.ValueID{condition}}},
				Body: &ir.Block{Instrs: []ir.Instr{
					&ir.Scope{Body: original, Span: function.Span},
					&ir.Const{Result: one, Type: types.TU32, Raw: "1", Span: function.Span},
					&ir.Binary{Result: incremented, Type: types.TU32, Op: "+", Left: parameter, Right: one, Span: function.Span},
				}, Term: &ir.Continue{Values: []ir.ValueID{incremented}}},
				Span: function.Span,
			},
		},
		Term: &ir.Return{},
	}
	return nil
}

func rewriteReturns(block *ir.Block) bool {
	if block == nil {
		return true
	}
	for _, instruction := range block.Instrs {
		switch x := instruction.(type) {
		case *ir.If:
			if !rewriteReturns(x.Then) || !rewriteReturns(x.Else) {
				return false
			}
		case *ir.Scope:
			if !rewriteReturns(x.Body) {
				return false
			}
		case *ir.Loop:
			return false
		}
	}
	if ret, ok := block.Term.(*ir.Return); ok {
		if ret.HasValue {
			return false
		}
		block.Term = &ir.ExitScope{}
	}
	return true
}

func maxValue(function *ir.Function) ir.ValueID {
	var maximum ir.ValueID
	see := func(id ir.ValueID) {
		if id > maximum {
			maximum = id
		}
	}
	for _, parameter := range function.Indices {
		see(parameter.ID)
	}
	for _, parameter := range function.Params {
		see(parameter.ID)
	}
	var walk func(*ir.Block)
	walk = func(block *ir.Block) {
		for _, instruction := range block.Instrs {
			if definition, ok := instruction.(ir.ValueDef); ok {
				see(definition.ResultValue())
			}
			switch x := instruction.(type) {
			case *ir.If:
				for _, result := range x.Results {
					see(result.ID)
				}
				walk(x.Then)
				walk(x.Else)
			case *ir.Loop:
				for _, result := range x.Results {
					see(result.ID)
				}
				for _, parameter := range x.Params {
					see(parameter.ID)
				}
				walk(x.Cond)
				walk(x.Body)
			case *ir.Scope:
				walk(x.Body)
			}
		}
	}
	walk(function.Body)
	return maximum
}

func pruneUnusedParameters(function *ir.Function, values []flow.ValueArgument) []flow.ValueArgument {
	uses, _, err := ir.UseCounts(function)
	if err != nil {
		return values
	}
	keptParameters := make([]ir.Param, 0, len(function.Params))
	keptValues := make([]flow.ValueArgument, 0, len(values))
	for index, parameter := range function.Params {
		if uses[parameter.ID] == 0 {
			continue
		}
		keptParameters = append(keptParameters, parameter)
		value := values[index]
		value.Formal = len(keptValues)
		keptValues = append(keptValues, value)
	}
	function.Params = keptParameters
	function.SourceParams = function.SourceParams[:0]
	for index, parameter := range function.BufferParams {
		function.SourceParams = append(function.SourceParams, ir.SourceParam{Name: parameter.Name, Kind: ir.SourceBuffer, Buffer: index})
	}
	for _, parameter := range function.Params {
		function.SourceParams = append(function.SourceParams, ir.SourceParam{Name: parameter.Name, Kind: ir.SourceValue, Value: parameter.ID, Buffer: -1})
	}
	return keptValues
}

func cloneFunction(function *ir.Function) *ir.Function {
	module := ir.Clone(&ir.Module{Functions: []*ir.Function{function}})
	return module.Functions[0]
}

func chooseWorkgroup(function *ir.Function, profile Profile) ([3]uint32, error) {
	if function.Workgroup.Explicit {
		size := function.Workgroup.Size
		product := uint64(1)
		for i, value := range size {
			if value == 0 || value > profile.MaxWorkgroup[i] {
				return size, fmt.Errorf("stage %s workgroup exceeds %s limits", function.Name, profile.Target)
			}
			product *= uint64(value)
		}
		if product > uint64(profile.MaxInvocations) {
			return size, fmt.Errorf("stage %s workgroup exceeds %s invocation limit", function.Name, profile.Target)
		}
		return size, nil
	}
	defaults := [][3]uint32{{256, 1, 1}, {16, 16, 1}, {8, 8, 4}}
	size := defaults[len(function.Indices)-1]
	for {
		product := uint64(size[0]) * uint64(size[1]) * uint64(size[2])
		valid := product <= uint64(profile.MaxInvocations)
		for i := range size {
			valid = valid && size[i] <= profile.MaxWorkgroup[i]
		}
		if valid {
			return size, nil
		}
		for i := len(function.Indices) - 1; i >= 0; i-- {
			if size[i] > 1 {
				size[i] = (size[i] + 1) / 2
				break
			}
		}
	}
}

func minimumByteSize(t *types.Type) (uint32, error) {
	l, err := layout.Of(t)
	if err != nil {
		return 0, err
	}
	if l.Runtime {
		if t.Kind == types.RuntimeArray {
			return l.Stride, nil
		}
		tail := l.Fields[len(l.Fields)-1]
		return tail.Offset + tail.Layout.Stride, nil
	}
	return l.Size, nil
}

func resourceIndex(program *flow.Program, resource *flow.Resource, omitted flow.ResourceID) int {
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

func planTransients(program *flow.Program, omitted flow.ResourceID) []Transient {
	var out []Transient
	for _, resource := range program.Resources {
		if resource.Kind != flow.Transient || resource.ID == omitted {
			continue
		}
		first, last := len(program.Dispatches), -1
		for i, dispatch := range program.Dispatches {
			for _, argument := range dispatch.Buffers {
				if argument.Resource == resource.ID {
					if i < first {
						first = i
					}
					if i > last {
						last = i
					}
				}
			}
		}
		if program.View != nil && program.View.Source == resource.ID {
			last = len(program.Dispatches)
		}
		l, _ := layout.Of(resource.Type)
		color := 0
		for {
			overlap := false
			for _, prior := range out {
				if prior.Color == color && first <= prior.LastStep && prior.FirstStep <= last {
					overlap = true
					break
				}
			}
			if !overlap {
				break
			}
			color++
		}
		out = append(out, Transient{Resource: resource.ID, Type: resource.Type, Stride: l.Stride, Alignment: l.Align, MinimumByteSize: l.Stride, Length: resource.Length, Color: color, FirstStep: first, LastStep: last})
	}
	return out
}

func between(program *flow.Program, before, after flow.Dispatch, omitted flow.ResourceID) []BarrierResource {
	writes := map[flow.ResourceID]bool{}
	touches := map[flow.ResourceID]bool{}
	for _, argument := range before.Buffers {
		if argument.Output != 0 {
			writes[argument.Resource] = true
		}
	}
	for _, argument := range after.Buffers {
		touches[argument.Resource] = true
	}
	var out []BarrierResource
	for _, resource := range program.Resources {
		if writes[resource.ID] && touches[resource.ID] {
			kind := ExternalSource
			if resource.Kind == flow.Transient {
				kind = TransientSource
			}
			out = append(out, BarrierResource{Kind: kind, Resource: resourceIndex(program, &resource, omitted)})
		}
	}
	return out
}

func fusibleView(module *flow.Module, program *flow.Program) (int, int) {
	if program.View == nil || len(program.Dispatches) == 0 {
		return -1, -1
	}
	resource := program.Resource(program.View.Source)
	version := program.Version(program.View.Input)
	last := len(program.Dispatches) - 1
	if resource == nil || resource.Kind != flow.Transient || version == nil || version.Producer != program.Dispatches[last].ID || !shapeProduct(program, resource.Length, program.View.Width, program.View.Height) {
		return -1, -1
	}
	for _, dispatch := range program.Dispatches[:last] {
		for _, argument := range dispatch.Buffers {
			if argument.Resource == resource.ID {
				return -1, -1
			}
		}
	}
	dispatch := program.Dispatches[last]
	stage := module.Kernel.Function(dispatch.Stage)
	if stage == nil || len(stage.Indices) != 1 || len(dispatch.Domain) != 1 || dispatch.Domain[0] != resource.Length {
		return -1, -1
	}
	summary := ir.AnalyzeAccess(stage)
	for _, argument := range dispatch.Buffers {
		if argument.Resource != resource.ID || argument.Output != program.View.Input {
			continue
		}
		access := summary.Buffers[argument.Formal]
		if access.CompleteWrite && len(access.Accesses) == 1 && len(access.Accesses[0].FieldPath) == 0 {
			return last, argument.Formal
		}
	}
	return -1, -1
}

func shapeProduct(program *flow.Program, product, left, right flow.ShapeID) bool {
	shape := program.Shape(product)
	if shape != nil && shape.Op == flow.ShapeMul && (shape.Left == left && shape.Right == right || shape.Left == right && shape.Right == left) {
		return true
	}
	a, b := program.Shape(left), program.Shape(right)
	if a == nil || b == nil {
		return false
	}
	if (product == left && b.Op == flow.ShapeConstant && b.Value == 1) || (product == right && a.Op == flow.ShapeConstant && a.Value == 1) {
		return true
	}
	return shape != nil && shape.Op == flow.ShapeConstant && a.Op == flow.ShapeConstant && b.Op == flow.ShapeConstant && uint64(a.Value)*uint64(b.Value) == uint64(shape.Value)
}

func appendViewExtent(function *ir.Function, values *[]flow.ValueArgument, view *flow.View) (ir.ValueID, ir.ValueID) {
	next := maxValue(function) + 1
	width, height := next, next+1
	for _, parameter := range []ir.Param{{Name: "__tach_view_width", ID: width, Type: types.TU32}, {Name: "__tach_view_height", ID: height, Type: types.TU32}} {
		function.Params = append(function.Params, parameter)
		function.SourceParams = append(function.SourceParams, ir.SourceParam{Name: parameter.Name, Kind: ir.SourceValue, Value: parameter.ID, Buffer: -1})
	}
	formal := len(*values)
	*values = append(*values,
		flow.ValueArgument{Formal: formal, Kind: flow.ValueShape, Shape: view.Width},
		flow.ValueArgument{Formal: formal + 1, Kind: flow.ValueShape, Shape: view.Height},
	)
	return width, height
}

func fuseView(function *ir.Function, binding int) error {
	output := types.Runtime(types.TU32)
	function.BufferParams[binding].Type = output
	places := map[ir.PlaceID]bool{}
	next, stores := maxValue(function)+1, 0
	var rewrite func(*ir.Block) error
	rewrite = func(block *ir.Block) error {
		var instructions []ir.Instr
		for _, instruction := range block.Instrs {
			switch x := instruction.(type) {
			case *ir.PlaceRoot:
				if x.Buffer == binding {
					x.Type, places[x.Result] = output, true
				}
			case *ir.PlaceField:
				if places[x.Base] {
					return fmt.Errorf("fused view output contains a field access")
				}
			case *ir.PlaceIndex:
				if places[x.Base] {
					x.Type, places[x.Result] = types.TU32, true
				}
			case *ir.Store:
				if places[x.Place] {
					packed, value := packRGBA(x.Value, &next)
					instructions = append(instructions, packed...)
					x.Value, stores = value, stores+1
				}
			case *ir.If:
				if err := rewrite(x.Then); err != nil {
					return err
				}
				if err := rewrite(x.Else); err != nil {
					return err
				}
			case *ir.Loop:
				if err := rewrite(x.Cond); err != nil {
					return err
				}
				if err := rewrite(x.Body); err != nil {
					return err
				}
			case *ir.Scope:
				if err := rewrite(x.Body); err != nil {
					return err
				}
			}
			instructions = append(instructions, instruction)
		}
		block.Instrs = instructions
		return nil
	}
	if err := rewrite(function.Body); err != nil {
		return err
	}
	if stores != 1 {
		return fmt.Errorf("fused view stage has %d output stores", stores)
	}
	return nil
}

func projectionKernel(target Target) (PhysicalKernel, error) {
	pixel, pixels := types.Vec(types.TF32, 4), types.Runtime(types.Vec(types.TF32, 4))
	output := types.Runtime(types.TU32)
	function := &ir.Function{
		Kind:    ir.Stage,
		Indices: []ir.Param{{Name: "x", ID: 1, Type: types.TU32}, {Name: "y", ID: 2, Type: types.TU32}},
		Params:  []ir.Param{{Name: "width", ID: 3, Type: types.TU32}, {Name: "height", ID: 4, Type: types.TU32}},
		BufferParams: []ir.BufferParam{
			{Name: "pixels", Type: pixels, Access: ir.Read},
			{Name: "output", Type: output, Access: ir.Mutable},
		},
		SourceParams: []ir.SourceParam{
			{Name: "pixels", Kind: ir.SourceBuffer, Buffer: 0},
			{Name: "output", Kind: ir.SourceBuffer, Buffer: 1},
			{Name: "width", Kind: ir.SourceValue, Value: 3, Buffer: -1},
			{Name: "height", Kind: ir.SourceValue, Value: 4, Buffer: -1},
		},
		Return:    types.TVoid,
		Workgroup: ir.WorkgroupConstraint{Explicit: true, Size: [3]uint32{16, 16, 1}},
	}
	function.Body = projectionBody(pixel)
	outputBytes := uint32(4)
	if target == Web {
		outputBytes = 0
	}
	physical := PhysicalKernel{Function: function, Workgroup: function.Workgroup.Size, Projection: true}
	physical.Bindings = []StorageBinding{
		{Buffer: 0, Binding: 0, Access: ir.Read, Type: pixels, MinimumByteSize: 16},
		{Buffer: 1, Binding: 1, Access: ir.Mutable, Type: output, MinimumByteSize: outputBytes, Texture: target == Web},
	}
	var err error
	physical.Parameters, err = abi.PlanParameters(function, 2)
	if err != nil {
		return PhysicalKernel{}, err
	}
	physical.Coordinates, err = LowerCoordinates(function)
	if err != nil {
		return PhysicalKernel{}, err
	}
	return physical, nil
}

func srgbFunction() *ir.Function {
	return &ir.Function{
		Name:   srgbHelper,
		Kind:   ir.Helper,
		Params: []ir.Param{{Name: "value", ID: 1, Type: types.TF32}},
		Return: types.TF32,
		Body: &ir.Block{
			Instrs: []ir.Instr{
				&ir.Call{Result: 2, Type: types.TF32, Function: unitHelper, Args: []ir.ValueID{1}},
				&ir.Const{Result: 3, Type: types.TF32, Raw: "0.0031308"},
				&ir.Binary{Result: 4, Type: types.TBool, Op: "<=", Left: 2, Right: 3},
				&ir.Const{Result: 5, Type: types.TF32, Raw: "12.92"},
				&ir.Binary{Result: 6, Type: types.TF32, Op: "*", Left: 2, Right: 5},
				&ir.Const{Result: 7, Type: types.TF32, Raw: "0.416666667"},
				&ir.Intrinsic{Result: 8, Type: types.TF32, Kind: ir.IntrinsicPow, Args: []ir.ValueID{2, 7}},
				&ir.Const{Result: 9, Type: types.TF32, Raw: "1.055"},
				&ir.Binary{Result: 10, Type: types.TF32, Op: "*", Left: 8, Right: 9},
				&ir.Const{Result: 11, Type: types.TF32, Raw: "0.055"},
				&ir.Binary{Result: 12, Type: types.TF32, Op: "-", Left: 10, Right: 11},
				&ir.If{Results: []ir.Result{{ID: 13, Type: types.TF32}}, Cond: 4, Then: &ir.Block{Term: &ir.Yield{Values: []ir.ValueID{6}}}, Else: &ir.Block{Term: &ir.Yield{Values: []ir.ValueID{12}}}},
			},
			Term: &ir.Return{Value: 13, HasValue: true},
		},
	}
}

func unitFunction() *ir.Function {
	return &ir.Function{
		Name: unitHelper, Kind: ir.Helper, Params: []ir.Param{{Name: "value", ID: 1, Type: types.TF32}}, Return: types.TF32,
		Body: &ir.Block{
			Instrs: []ir.Instr{
				&ir.Const{Result: 2, Type: types.TF32, Raw: "0.0"},
				&ir.Const{Result: 3, Type: types.TF32, Raw: "1.0"},
				&ir.Binary{Result: 4, Type: types.TBool, Op: ">", Left: 1, Right: 2},
				&ir.Binary{Result: 5, Type: types.TBool, Op: "<", Left: 1, Right: 3},
				&ir.If{Results: []ir.Result{{ID: 6, Type: types.TF32}}, Cond: 5, Then: &ir.Block{Term: &ir.Yield{Values: []ir.ValueID{1}}}, Else: &ir.Block{Term: &ir.Yield{Values: []ir.ValueID{3}}}},
				&ir.If{Results: []ir.Result{{ID: 7, Type: types.TF32}}, Cond: 4, Then: &ir.Block{Term: &ir.Yield{Values: []ir.ValueID{6}}}, Else: &ir.Block{Term: &ir.Yield{Values: []ir.ValueID{2}}}},
			},
			Term: &ir.Return{Value: 7, HasValue: true},
		},
	}
}

func projectionBody(pixel *types.Type) *ir.Block {
	then := &ir.Block{Instrs: []ir.Instr{
		&ir.Binary{Result: 8, Type: types.TU32, Op: "*", Left: 2, Right: 3},
		&ir.Binary{Result: 9, Type: types.TU32, Op: "+", Left: 8, Right: 1},
		&ir.PlaceRoot{Result: 1, Type: types.Runtime(pixel), Buffer: 0},
		&ir.PlaceIndex{Result: 2, Type: pixel, Base: 1, Index: 9},
		&ir.Load{Result: 10, Type: pixel, Place: 2},
	}}
	next := ir.ValueID(11)
	packed, merged := packRGBA(10, &next)
	then.Instrs = append(then.Instrs, packed...)
	then.Instrs = append(then.Instrs,
		&ir.PlaceRoot{Result: 3, Type: types.Runtime(types.TU32), Buffer: 1},
		&ir.PlaceIndex{Result: 4, Type: types.TU32, Base: 3, Index: 9},
		&ir.Store{Place: 4, Value: merged},
	)
	then.Term = &ir.Yield{}
	return &ir.Block{
		Instrs: []ir.Instr{
			&ir.Binary{Result: 5, Type: types.TBool, Op: "<", Left: 1, Right: 3},
			&ir.Binary{Result: 6, Type: types.TBool, Op: "<", Left: 2, Right: 4},
			&ir.Binary{Result: 7, Type: types.TBool, Op: "&&", Left: 5, Right: 6},
			&ir.If{Cond: 7, Then: then, Else: &ir.Block{Term: &ir.Yield{}}},
		},
		Term: &ir.Return{},
	}
}

func packRGBA(value ir.ValueID, next *ir.ValueID) ([]ir.Instr, ir.ValueID) {
	newValue := func() ir.ValueID {
		id := *next
		*next = id + 1
		return id
	}
	var instructions []ir.Instr
	channels := make([]ir.ValueID, 4)
	for index := range channels {
		channels[index] = newValue()
		instructions = append(instructions, &ir.Extract{Result: channels[index], Type: types.TF32, Base: value, Index: index})
	}
	for index := range 3 {
		encoded := newValue()
		instructions = append(instructions, &ir.Call{Result: encoded, Type: types.TF32, Function: srgbHelper, Args: []ir.ValueID{channels[index]}})
		channels[index] = encoded
	}
	alpha := newValue()
	instructions = append(instructions, &ir.Call{Result: alpha, Type: types.TF32, Function: unitHelper, Args: []ir.ValueID{channels[3]}})
	channels[3] = alpha
	scale, half := newValue(), newValue()
	instructions = append(instructions,
		&ir.Const{Result: scale, Type: types.TF32, Raw: "255.0"},
		&ir.Const{Result: half, Type: types.TF32, Raw: "0.5"},
	)
	packed := make([]ir.ValueID, 4)
	for index, channel := range channels {
		multiply, round, convert := newValue(), newValue(), newValue()
		instructions = append(instructions,
			&ir.Binary{Result: multiply, Type: types.TF32, Op: "*", Left: channel, Right: scale},
			&ir.Binary{Result: round, Type: types.TF32, Op: "+", Left: multiply, Right: half},
			&ir.Convert{Result: convert, Type: types.TU32, X: round, From: types.TF32},
		)
		packed[index] = convert
	}
	for index := 1; index < 4; index++ {
		shift, shifted := newValue(), newValue()
		instructions = append(instructions,
			&ir.Const{Result: shift, Type: types.TU32, Raw: fmt.Sprintf("%d", index*8)},
			&ir.Binary{Result: shifted, Type: types.TU32, Op: "<<", Left: packed[index], Right: shift},
		)
		packed[index] = shifted
	}
	merged := packed[0]
	for index := 1; index < 4; index++ {
		result := newValue()
		instructions = append(instructions, &ir.Binary{Result: result, Type: types.TU32, Op: "|", Left: merged, Right: packed[index]})
		merged = result
	}
	return instructions, merged
}

func Verify(executable *Executable) error {
	if executable == nil || executable.Logical == nil || executable.KernelModule == nil {
		return fmt.Errorf("incomplete executable")
	}
	if err := flow.Verify(executable.Logical); err != nil {
		return fmt.Errorf("logical module: %w", err)
	}
	if err := ir.Verify(executable.KernelModule); err != nil {
		return fmt.Errorf("physical kernel module: %w", err)
	}
	entries := map[string]bool{}
	for i, kernel := range executable.PhysicalKernels {
		if kernel.Entry != abi.PrivateEntry(i) || kernel.Function == nil || kernel.Function.Name != kernel.Entry || entries[kernel.Entry] {
			return fmt.Errorf("physical kernel %d has invalid private entry", i)
		}
		entries[kernel.Entry] = true
		if kernel.Workgroup[0] == 0 || kernel.Workgroup[1] == 0 || kernel.Workgroup[2] == 0 || len(kernel.Bindings) != len(kernel.Function.BufferParams) {
			return fmt.Errorf("physical kernel %s has invalid workgroup/bindings", kernel.Entry)
		}
		for binding, descriptor := range kernel.Bindings {
			if descriptor.Buffer != binding || descriptor.Binding != uint32(binding) {
				return fmt.Errorf("physical kernel %s bindings are not dense", kernel.Entry)
			}
		}
		if kernel.Projection && (kernel.FusedView || len(kernel.Bindings) != 2 || kernel.Bindings[0].Texture || kernel.Bindings[1].Texture != (executable.Target == Web)) {
			return fmt.Errorf("physical projection kernel %s is invalid", kernel.Entry)
		}
		if kernel.FusedView {
			if kernel.Projection || kernel.ViewBinding < 0 || kernel.ViewBinding >= len(kernel.Bindings) || kernel.Bindings[kernel.ViewBinding].Texture != (executable.Target == Web) {
				return fmt.Errorf("physical fused view kernel %s is invalid", kernel.Entry)
			}
			if (executable.Target == Web) != (kernel.ViewWidth != 0 && kernel.ViewHeight != 0) {
				return fmt.Errorf("physical fused view kernel %s has invalid extent", kernel.Entry)
			}
		}
	}
	if len(executable.Programs) != len(executable.Logical.Programs) {
		return fmt.Errorf("target program count mismatch")
	}
	for i, plan := range executable.Programs {
		if plan.Program != i || plan.Repeat == 0 {
			return fmt.Errorf("program plan %d is invalid", i)
		}
		for _, step := range plan.Steps {
			if step.Kind == DispatchStepKind && (step.Kernel < 0 || step.Kernel >= len(executable.PhysicalKernels)) {
				return fmt.Errorf("program plan %d references invalid kernel", i)
			}
		}
		logical := executable.Logical.Programs[i].View
		if (plan.View != nil) != (logical != nil) {
			return fmt.Errorf("program plan %d view contract mismatch", i)
		}
		if plan.View != nil {
			view := plan.View
			if view.Step.Kind != DispatchStepKind || view.Step.Kernel < 0 || view.Step.Kernel >= len(executable.PhysicalKernels) || view.OutputColor < 0 || view.Width != logical.Width || view.Height != logical.Height {
				return fmt.Errorf("program plan %d has invalid view", i)
			}
			kernel := executable.PhysicalKernels[view.Step.Kernel]
			if view.Output >= uint32(len(kernel.Bindings)) || view.Fused && (!kernel.FusedView || int(view.Output) != kernel.ViewBinding) || !view.Fused && (!kernel.Projection || view.Output != 1 || len(view.Step.Resources) != 1) {
				return fmt.Errorf("program plan %d has invalid view kernel", i)
			}
			for _, resource := range view.Step.Resources {
				if resource.Binding == view.Output {
					return fmt.Errorf("program plan %d binds the view output as input", i)
				}
			}
			for _, transient := range plan.Transients {
				if transient.Color == view.OutputColor {
					return fmt.Errorf("program plan %d aliases view output and transient", i)
				}
			}
		}
	}
	return nil
}

// DECISION: Physical kernels are one-per-surviving-dispatch for deterministic,
// unambiguous plans. Deduplicate by verified Kernel IR hash only if real modules
// show shader-size/pipeline duplication worth the extra identity machinery.
