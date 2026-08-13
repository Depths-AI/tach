package backend

import (
	"fmt"
	"strings"

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
}

type PhysicalKernel struct {
	Entry       string
	Function    *ir.Function
	Workgroup   [3]uint32
	Bindings    []StorageBinding
	Parameters  *abi.ParameterBlock
	Coordinates *Coordinates
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
}

type Executable struct {
	Target          Target
	Logical         *flow.Module
	KernelModule    *ir.Module
	PhysicalKernels []PhysicalKernel
	Programs        []ProgramPlan
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
	for programIndex, program := range cloned.Programs {
		plan := ProgramPlan{Program: programIndex, Repeat: RepeatProgram}
		kernelForDispatch := make([]int, len(program.Dispatches))
		valuesForDispatch := make([][]flow.ValueArgument, len(program.Dispatches))
		invocationRepeat := len(program.Dispatches) == 1 && canInternalizeRepeat(cloned.Kernel.Function(program.Dispatches[0].Stage))
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
			valuesForDispatch[dispatchIndex] = pruneUnusedParameters(function, values)
			kernelIndex := len(executable.PhysicalKernels)
			function.Name = abi.PrivateEntry(kernelIndex)
			workgroup, err := chooseWorkgroup(function, profile)
			if err != nil {
				return nil, err
			}
			physical := PhysicalKernel{Entry: function.Name, Function: function, Workgroup: workgroup}
			for buffer, parameter := range function.BufferParams {
				minimum, err := minimumByteSize(parameter.Type)
				if err != nil {
					return nil, fmt.Errorf("kernel %s buffer %s: %w", function.Name, parameter.Name, err)
				}
				physical.Bindings = append(physical.Bindings, StorageBinding{Buffer: buffer, Binding: uint32(buffer), Access: parameter.Access, Type: parameter.Type, MinimumByteSize: minimum})
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
		plan.Transients = planTransients(program)
		for dispatchIndex, dispatch := range program.Dispatches {
			if profile.Target == SPIRV && dispatchIndex > 0 {
				if barrier := between(program, program.Dispatches[dispatchIndex-1], dispatch); len(barrier) > 0 {
					plan.Steps = append(plan.Steps, Step{Kind: BarrierStepKind, Barrier: barrier})
				}
			}
			step := Step{Kind: DispatchStepKind, Kernel: kernelForDispatch[dispatchIndex], Domain: append([]flow.ShapeID(nil), dispatch.Domain...), Parameters: valuesForDispatch[dispatchIndex]}
			for _, argument := range dispatch.Buffers {
				resource := program.Resource(argument.Resource)
				source := ResourceSource{Binding: uint32(argument.Formal), Resource: resourceIndex(program, resource)}
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
		if profile.Target == SPIRV && len(program.Dispatches) > 0 {
			plan.RepeatBarrier = between(program, program.Dispatches[len(program.Dispatches)-1], program.Dispatches[0])
		}
		executable.Programs = append(executable.Programs, plan)
	}
	if err := Verify(executable); err != nil {
		return nil, err
	}
	return executable, nil
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
	if t.Kind == types.RuntimeArray {
		return l.Stride, nil
	}
	return l.Size, nil
}

func resourceIndex(program *flow.Program, resource *flow.Resource) int {
	index := 0
	for _, candidate := range program.Resources {
		if candidate.Kind == resource.Kind {
			if candidate.ID == resource.ID {
				return index
			}
			index++
		}
	}
	return -1
}

func planTransients(program *flow.Program) []Transient {
	var out []Transient
	for _, resource := range program.Resources {
		if resource.Kind != flow.Transient {
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

func between(program *flow.Program, before, after flow.Dispatch) []BarrierResource {
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
			out = append(out, BarrierResource{Kind: kind, Resource: resourceIndex(program, &resource)})
		}
	}
	return out
}

func Verify(executable *Executable) error {
	if executable == nil || executable.Logical == nil || executable.KernelModule == nil {
		return fmt.Errorf("incomplete executable")
	}
	if err := flow.Verify(executable.Logical); err != nil {
		return fmt.Errorf("logical module: %w", err)
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
	}
	return nil
}

func Dump(executable *Executable) string {
	var b strings.Builder
	fmt.Fprintf(&b, "target %s\n", executable.Target)
	for i, kernel := range executable.PhysicalKernels {
		fmt.Fprintf(&b, "kernel %d @%s workgroup(%d,%d,%d) bindings=%d\n", i, kernel.Entry, kernel.Workgroup[0], kernel.Workgroup[1], kernel.Workgroup[2], len(kernel.Bindings))
	}
	for _, program := range executable.Programs {
		fmt.Fprintf(&b, "program %d transients=%d repeat=%d\n", program.Program, len(program.Transients), program.Repeat)
		for _, step := range program.Steps {
			if step.Kind == DispatchStepKind {
				fmt.Fprintf(&b, "  dispatch kernel=%d domain=%v\n", step.Kernel, step.Domain)
			} else {
				fmt.Fprintf(&b, "  barrier resources=%v\n", step.Barrier)
			}
		}
	}
	return b.String()
}

// DECISION: Physical kernels are one-per-surviving-dispatch for deterministic,
// unambiguous plans. Deduplicate by verified Kernel IR hash only if real modules
// show shader-size/pipeline duplication worth the extra identity machinery.
