package opt

import (
	"fmt"
	"reflect"

	"tach/src/flow"
	"tach/src/ir"
	"tach/src/types"
)

type FusionPolicy struct {
	Target           bool
	MaxInstructions  int
	MaxLiveValues    int
	MaxBindings      int
	DispatchBenefit  int
	TransientBenefit int
	CloneCost        int
	LiveCost         int
}

func PortablePolicy() FusionPolicy {
	return FusionPolicy{MaxInstructions: 2048, MaxLiveValues: 512, MaxBindings: 8, DispatchBenefit: 1, TransientBenefit: 1}
}

type FusionDecision struct {
	Dispatches []flow.DispatchID
	Legal      bool
	Profitable bool
	Reason     string
}

func Fuse(module *flow.Module, policy FusionPolicy) error {
	if err := flow.Verify(module); err != nil {
		return fmt.Errorf("pre-fusion Flow IR verification: %w", err)
	}
	// DECISION: This static model handles bounded affine fusion against guaranteed
	// target limits. Replace its weights with measured profiles/autotuning only when
	// Tach has a device-profile input; legality must remain in the shared prover.
	for _, program := range module.Programs {
		for {
			changed := false
			for index := 0; index+1 < len(program.Dispatches); index++ {
				decision := decidePair(module, program, index, policy)
				if !decision.Legal || !decision.Profitable {
					continue
				}
				if err := fusePair(module, program, index, policy); err != nil {
					return fmt.Errorf("fuse program %s dispatches %d,%d: %w", program.Name, index+1, index+2, err)
				}
				changed = true
				break
			}
			if !changed {
				break
			}
		}
	}
	if err := OptimizeKernel(module.Kernel); err != nil {
		return err
	}
	if err := flow.Verify(module); err != nil {
		return fmt.Errorf("post-fusion Flow IR verification: %w", err)
	}
	return nil
}

func Decide(module *flow.Module, program *flow.Program, index int, policy FusionPolicy) FusionDecision {
	return decidePair(module, program, index, policy)
}

func decidePair(module *flow.Module, program *flow.Program, index int, policy FusionPolicy) FusionDecision {
	decision := FusionDecision{Reason: "not adjacent"}
	if program == nil || index < 0 || index+1 >= len(program.Dispatches) {
		return decision
	}
	first, second := &program.Dispatches[index], &program.Dispatches[index+1]
	decision.Dispatches = []flow.DispatchID{first.ID, second.ID}
	if !reflect.DeepEqual(first.Domain, second.Domain) {
		decision.Reason = "domains differ"
		return decision
	}
	a, b := module.Kernel.Function(first.Stage), module.Kernel.Function(second.Stage)
	if a == nil || b == nil {
		decision.Reason = "missing stage"
		return decision
	}
	if a.Workgroup.Explicit && b.Workgroup.Explicit && a.Workgroup.Size != b.Workgroup.Size {
		decision.Reason = "explicit workgroups differ"
		return decision
	}
	sa, sb := ir.AnalyzeAccess(a), ir.AnalyzeAccess(b)
	if hasLoop(a.Body) || hasLoop(b.Body) {
		decision.Reason = "structured loop fusion is outside the bounded composer"
		return decision
	}
	if sa.Effects.Atomic || sb.Effects.Atomic || sa.Effects.Workgroup || sb.Effects.Workgroup || sa.Effects.Barrier || sb.Effects.Barrier {
		decision.Reason = "atomic, shared-memory, or barrier boundary"
		return decision
	}
	if sa.InstructionCount+sb.InstructionCount > policy.MaxInstructions || sa.PeakLive+sb.PeakLive > policy.MaxLiveValues {
		decision.Reason = "target instruction/live limit"
		return decision
	}
	firstResources, secondResources := dispatchResources(first), dispatchResources(second)
	shared := false
	eliminated := flow.ResourceID(0)
	for resource, firstFormal := range firstResources {
		secondFormal, ok := secondResources[resource]
		if !ok {
			continue
		}
		shared = true
		logical := program.Resource(resource)
		if logical.Kind == flow.Transient && first.Buffers[firstFormal].Output != 0 && sb.Buffers[secondFormal].Read && sa.Buffers[firstFormal].CompleteWrite && identityOnly(sa.Buffers[firstFormal]) && transientOnlyBetween(program, resource, index) && topLevelForwardable(a, firstFormal, b, secondFormal) {
			if identityOnly(sb.Buffers[secondFormal]) {
				eliminated = resource
				continue
			}
			if policy.Target && affineRecomputable(a, firstFormal, sb.Buffers[secondFormal]) {
				eliminated = resource
				continue
			}
		}
		if !identityOnly(sa.Buffers[firstFormal]) || !identityOnly(sb.Buffers[secondFormal]) {
			decision.Reason = "cross-invocation or opaque shared-resource dependence"
			return decision
		}
	}
	bindings := len(firstResources) + len(secondResources)
	if shared {
		bindings--
	}
	if eliminated != 0 {
		bindings--
	}
	if bindings > policy.MaxBindings {
		decision.Reason = "target binding limit"
		return decision
	}
	decision.Legal = true
	if eliminated != 0 {
		if _, ok := recomputeOffset(sb.Buffers[secondResources[eliminated]]); ok && !identityOnly(sb.Buffers[secondResources[eliminated]]) {
			decision.Reason = "target affine producer recomputation"
		} else {
			decision.Reason = "vertical transient forwarding"
		}
	} else if shared {
		decision.Reason = "same-invocation sequential fusion"
	} else {
		decision.Reason = "disjoint horizontal fusion"
	}
	score := policy.DispatchBenefit - policy.LiveCost*(sa.PeakLive+sb.PeakLive)
	if eliminated != 0 {
		score += policy.TransientBenefit
		if decision.Reason == "target affine producer recomputation" {
			score -= policy.CloneCost * sa.InstructionCount
		}
	}
	decision.Profitable = score > 0
	if decision.Legal && !decision.Profitable {
		decision.Reason = "fusion cost exceeds dispatch and transient benefit"
	}
	return decision
}

func hasLoop(block *ir.Block) bool {
	for _, instruction := range block.Instrs {
		switch x := instruction.(type) {
		case *ir.Loop:
			return true
		case *ir.If:
			if hasLoop(x.Then) || hasLoop(x.Else) {
				return true
			}
		case *ir.Scope:
			if hasLoop(x.Body) {
				return true
			}
		}
	}
	return false
}

func affineRecomputable(producer *ir.Function, output int, consumer ir.BufferSummary) bool {
	if len(producer.Indices) != 1 {
		return false
	}
	if _, ok := recomputeOffset(consumer); !ok {
		return false
	}
	producerSummary := ir.AnalyzeAccess(producer)
	for formal, buffer := range producerSummary.Buffers {
		if formal != output && (buffer.Write || buffer.Atomic) {
			return false
		}
		for _, access := range buffer.Accesses {
			for _, affine := range access.Indices {
				if !affine.Exact {
					return false
				}
			}
		}
	}
	return true
}

func recomputeOffset(summary ir.BufferSummary) (int64, bool) {
	var read *ir.MemoryAccess
	for i := range summary.Accesses {
		access := &summary.Accesses[i]
		if access.Kind != ir.MemoryRead {
			return 0, false
		}
		if read != nil || len(access.Indices) != 1 || !access.Indices[0].Exact || access.Indices[0].Coefficient != [3]int64{1, 0, 0} {
			return 0, false
		}
		read = access
	}
	if read == nil {
		return 0, false
	}
	return read.Indices[0].Constant, true
}

func dispatchResources(dispatch *flow.Dispatch) map[flow.ResourceID]int {
	out := map[flow.ResourceID]int{}
	for i, a := range dispatch.Buffers {
		out[a.Resource] = i
	}
	return out
}
func identityOnly(summary ir.BufferSummary) bool {
	for _, access := range summary.Accesses {
		if len(access.Indices) > 0 && !identityAccess(access.Indices) {
			return false
		}
	}
	return true
}
func identityAccess(indices []ir.Affine) bool {
	if len(indices) != 1 {
		return false
	}
	want := ir.Affine{Exact: true}
	want.Coefficient[0] = 1
	return indices[0] == want
}
func transientOnlyBetween(program *flow.Program, resource flow.ResourceID, index int) bool {
	for i, d := range program.Dispatches {
		for _, a := range d.Buffers {
			if a.Resource == resource && i != index && i != index+1 {
				return false
			}
		}
	}
	return true
}

func topLevelForwardable(producer *ir.Function, producerBuffer int, consumer *ir.Function, consumerBuffer int) bool {
	producerPlaces := map[ir.PlaceID]bool{}
	stores := 0
	for _, in := range producer.Body.Instrs {
		switch x := in.(type) {
		case *ir.PlaceRoot:
			producerPlaces[x.Result] = x.Buffer == producerBuffer
		case *ir.PlaceField:
			producerPlaces[x.Result] = producerPlaces[x.Base]
		case *ir.PlaceIndex:
			producerPlaces[x.Result] = producerPlaces[x.Base]
		case *ir.Store:
			if producerPlaces[x.Place] {
				stores++
			}
		}
	}
	consumerPlaces := map[ir.PlaceID]bool{}
	loads := 0
	for _, in := range consumer.Body.Instrs {
		switch x := in.(type) {
		case *ir.PlaceRoot:
			consumerPlaces[x.Result] = x.Buffer == consumerBuffer
		case *ir.PlaceField:
			consumerPlaces[x.Result] = consumerPlaces[x.Base]
		case *ir.PlaceIndex:
			consumerPlaces[x.Result] = consumerPlaces[x.Base]
		case *ir.Load:
			if consumerPlaces[x.Place] {
				loads++
			}
		}
	}
	return stores == 1 && loads > 0
}

type cloneContext struct {
	values     map[ir.ValueID]ir.ValueID
	places     map[ir.PlaceID]ir.PlaceID
	eliminated map[ir.PlaceID]bool
	bufferMap  map[int]int
	nextValue  ir.ValueID
	nextPlace  ir.PlaceID
	forwarded  *ir.ValueID
	consume    ir.ValueID
}

func (c *cloneContext) value(id ir.ValueID) ir.ValueID {
	if id == 0 {
		return 0
	}
	if value := c.values[id]; value != 0 {
		return value
	}
	c.nextValue++
	c.values[id] = c.nextValue
	return c.nextValue
}
func (c *cloneContext) place(id ir.PlaceID) ir.PlaceID {
	if id == 0 {
		return 0
	}
	if place := c.places[id]; place != 0 {
		return place
	}
	c.nextPlace++
	c.places[id] = c.nextPlace
	return c.nextPlace
}

func fusePair(module *flow.Module, program *flow.Program, index int, policy FusionPolicy) error {
	first, second := program.Dispatches[index], program.Dispatches[index+1]
	a, b := module.Kernel.Function(first.Stage), module.Kernel.Function(second.Stage)
	firstResources, secondResources := dispatchResources(&first), dispatchResources(&second)
	eliminated := flow.ResourceID(0)
	recompute := int64(0)
	sa, sb := ir.AnalyzeAccess(a), ir.AnalyzeAccess(b)
	for resource, af := range firstResources {
		if bf, ok := secondResources[resource]; ok && program.Resource(resource).Kind == flow.Transient && sa.Buffers[af].CompleteWrite && identityOnly(sa.Buffers[af]) && topLevelForwardable(a, af, b, bf) {
			if identityOnly(sb.Buffers[bf]) {
				eliminated = resource
			} else if policy.Target && affineRecomputable(a, af, sb.Buffers[bf]) {
				eliminated = resource
				recompute, _ = recomputeOffset(sb.Buffers[bf])
			}
		}
	}
	name := fmt.Sprintf("_tach_fused_%d", len(module.Kernel.Functions))
	fused := &ir.Function{Name: name, Kind: ir.Stage, Return: types.TVoid, Body: &ir.Block{}, Span: first.Span}
	if a.Workgroup.Explicit {
		fused.Workgroup = a.Workgroup
	} else {
		fused.Workgroup = b.Workgroup
	}
	ctx := &cloneContext{values: map[ir.ValueID]ir.ValueID{}, places: map[ir.PlaceID]ir.PlaceID{}, eliminated: map[ir.PlaceID]bool{}, bufferMap: map[int]int{}}
	for dimension, indexParam := range a.Indices {
		ctx.nextValue++
		fused.Indices = append(fused.Indices, ir.Param{Name: indexParam.Name, ID: ctx.nextValue, Type: indexParam.Type})
		ctx.values[indexParam.ID] = ctx.nextValue
		if dimension < len(b.Indices) {
			ctx.values[b.Indices[dimension].ID] = ctx.nextValue
		}
	}
	if recompute != 0 {
		ctx.nextValue++
		constant := ctx.nextValue
		fused.Body.Instrs = append(fused.Body.Instrs, &ir.Const{Result: constant, Type: types.TU32, Raw: fmt.Sprintf("%d", abs64(recompute)), Span: a.Span})
		ctx.nextValue++
		adjusted := ctx.nextValue
		op := "+"
		if recompute < 0 {
			op = "-"
		}
		fused.Body.Instrs = append(fused.Body.Instrs, &ir.Binary{Result: adjusted, Type: types.TU32, Op: op, Left: fused.Indices[0].ID, Right: constant, Span: a.Span})
		ctx.values[a.Indices[0].ID] = adjusted
	}
	resourceToBuffer := map[flow.ResourceID]int{}
	var fusedBuffers []flow.BufferArgument
	addBuffers := func(stage *ir.Function, dispatch flow.Dispatch) {
		for _, argument := range dispatch.Buffers {
			if argument.Resource == eliminated {
				ctx.bufferMap[argument.Formal] = -1
				continue
			}
			mapped, ok := resourceToBuffer[argument.Resource]
			if !ok {
				mapped = len(fused.BufferParams)
				resourceToBuffer[argument.Resource] = mapped
				parameter := stage.BufferParams[argument.Formal]
				parameter.Name = fmt.Sprintf("b%d", mapped)
				fused.BufferParams = append(fused.BufferParams, parameter)
				fusedBuffers = append(fusedBuffers, flow.BufferArgument{Formal: mapped, Resource: argument.Resource, Input: argument.Input, Output: argument.Output})
			} else if stage.BufferParams[argument.Formal].Access == ir.Mutable {
				fused.BufferParams[mapped].Access = ir.Mutable
				for i := range fusedBuffers {
					if fusedBuffers[i].Formal == mapped {
						fusedBuffers[i].Output = argument.Output
					}
				}
			}
			ctx.bufferMap[argument.Formal] = mapped
		}
	}
	addBuffers(a, first)
	for _, p := range a.Params {
		ctx.nextValue++
		fused.Params = append(fused.Params, ir.Param{Name: fmt.Sprintf("v%d", len(fused.Params)), ID: ctx.nextValue, Type: p.Type})
		ctx.values[p.ID] = ctx.nextValue
	}
	firstValues := append([]flow.ValueArgument(nil), first.Values...)
	firstBlock, err := cloneFusionBlock(a.Body, ctx, true)
	if err != nil {
		return err
	}
	if eliminated != 0 {
		fused.Body.Instrs = append(fused.Body.Instrs, firstBlock.Instrs...)
	} else {
		fused.Body.Instrs = append(fused.Body.Instrs, &ir.Scope{Body: firstBlock, Span: a.Span})
	}
	forwarded := ir.ValueID(0)
	if ctx.forwarded != nil {
		forwarded = *ctx.forwarded
	}
	ctx.values = map[ir.ValueID]ir.ValueID{}
	for dimension, indexParam := range b.Indices {
		ctx.values[indexParam.ID] = fused.Indices[dimension].ID
	}
	ctx.places = map[ir.PlaceID]ir.PlaceID{}
	ctx.eliminated = map[ir.PlaceID]bool{}
	ctx.consume = forwarded
	addBuffers(b, second)
	for _, p := range b.Params {
		ctx.nextValue++
		fused.Params = append(fused.Params, ir.Param{Name: fmt.Sprintf("v%d", len(fused.Params)), ID: ctx.nextValue, Type: p.Type})
		ctx.values[p.ID] = ctx.nextValue
	}
	secondBlock, err := cloneFusionBlock(b.Body, ctx, true)
	if err != nil {
		return err
	}
	fused.Body.Instrs = append(fused.Body.Instrs, &ir.Scope{Body: secondBlock, Span: b.Span})
	fused.Body.Term = &ir.Return{}
	for i, p := range fused.BufferParams {
		fused.SourceParams = append(fused.SourceParams, ir.SourceParam{Name: p.Name, Kind: ir.SourceBuffer, Buffer: i})
	}
	for _, p := range fused.Params {
		fused.SourceParams = append(fused.SourceParams, ir.SourceParam{Name: p.Name, Kind: ir.SourceValue, Value: p.ID, Buffer: -1})
	}
	module.Kernel.Functions = append(module.Kernel.Functions, fused)
	fusedDispatch := flow.Dispatch{ID: first.ID, Stage: name, Domain: append([]flow.ShapeID(nil), first.Domain...), Buffers: fusedBuffers, Values: append(firstValues, second.Values...), Span: first.Span}
	for i := range fusedDispatch.Values {
		fusedDispatch.Values[i].Formal = i
	}
	program.Dispatches = append(program.Dispatches[:index:index], append([]flow.Dispatch{fusedDispatch}, program.Dispatches[index+2:]...)...)
	rebuildProgram(module, program, eliminated)
	return nil
}

func abs64(value int64) int64 {
	if value < 0 {
		return -value
	}
	return value
}

func cloneFusionBlock(block *ir.Block, c *cloneContext, scope bool) (*ir.Block, error) {
	out := &ir.Block{}
	for _, in := range block.Instrs {
		cloned, keep, err := cloneFusionInstr(in, c)
		if err != nil {
			return nil, err
		}
		if keep {
			out.Instrs = append(out.Instrs, cloned)
		}
	}
	switch t := block.Term.(type) {
	case *ir.Return:
		if scope && !t.HasValue {
			out.Term = &ir.ExitScope{}
		} else {
			out.Term = &ir.Return{Value: c.value(t.Value), HasValue: t.HasValue}
		}
	case *ir.Yield:
		values := make([]ir.ValueID, len(t.Values))
		for i, v := range t.Values {
			values[i] = c.value(v)
		}
		out.Term = &ir.Yield{Values: values}
	case *ir.Continue:
		values := make([]ir.ValueID, len(t.Values))
		for i, v := range t.Values {
			values[i] = c.value(v)
		}
		out.Term = &ir.Continue{Values: values}
	case *ir.Unreachable:
		out.Term = &ir.Unreachable{}
	case *ir.ExitScope:
		out.Term = &ir.ExitScope{}
	default:
		return nil, fmt.Errorf("unknown terminator %T", block.Term)
	}
	return out, nil
}

func cloneFusionInstr(in ir.Instr, c *cloneContext) (ir.Instr, bool, error) {
	switch x := in.(type) {
	case *ir.Const:
		y := *x
		y.Result = c.value(x.Result)
		return &y, true, nil
	case *ir.Unary:
		y := *x
		y.Result, y.X = c.value(x.Result), c.value(x.X)
		return &y, true, nil
	case *ir.Binary:
		y := *x
		y.Result, y.Left, y.Right = c.value(x.Result), c.value(x.Left), c.value(x.Right)
		return &y, true, nil
	case *ir.Convert:
		y := *x
		y.Result, y.X = c.value(x.Result), c.value(x.X)
		return &y, true, nil
	case *ir.Composite:
		y := *x
		y.Result = c.value(x.Result)
		y.Values = make([]ir.ValueID, len(x.Values))
		for i, v := range x.Values {
			y.Values[i] = c.value(v)
		}
		return &y, true, nil
	case *ir.Extract:
		y := *x
		y.Result, y.Base = c.value(x.Result), c.value(x.Base)
		return &y, true, nil
	case *ir.VectorIndex:
		y := *x
		y.Result, y.Base, y.Index = c.value(x.Result), c.value(x.Base), c.value(x.Index)
		return &y, true, nil
	case *ir.Call:
		y := *x
		y.Result = c.value(x.Result)
		y.Args = make([]ir.ValueID, len(x.Args))
		for i, v := range x.Args {
			y.Args[i] = c.value(v)
		}
		return &y, true, nil
	case *ir.Intrinsic:
		y := *x
		y.Result = c.value(x.Result)
		y.Args = make([]ir.ValueID, len(x.Args))
		for i, v := range x.Args {
			y.Args[i] = c.value(v)
		}
		return &y, true, nil
	case *ir.PlaceRoot:
		mapped := c.bufferMap[x.Buffer]
		if mapped < 0 {
			c.eliminated[x.Result] = true
			return nil, false, nil
		}
		y := *x
		y.Result, y.Buffer = c.place(x.Result), mapped
		return &y, true, nil
	case *ir.PlaceField:
		if c.eliminated[x.Base] {
			c.eliminated[x.Result] = true
			return nil, false, nil
		}
		y := *x
		y.Result, y.Base = c.place(x.Result), c.place(x.Base)
		return &y, true, nil
	case *ir.PlaceIndex:
		if c.eliminated[x.Base] {
			c.eliminated[x.Result] = true
			return nil, false, nil
		}
		y := *x
		y.Result, y.Base, y.Index = c.place(x.Result), c.place(x.Base), c.value(x.Index)
		return &y, true, nil
	case *ir.PlaceWorkgroup:
		y := *x
		y.Result = c.place(x.Result)
		return &y, true, nil
	case *ir.Load:
		if c.eliminated[x.Place] {
			if c.consume == 0 {
				return nil, false, fmt.Errorf("eliminated transient load has no forwarded value")
			}
			c.values[x.Result] = c.consume
			return nil, false, nil
		}
		y := *x
		y.Result, y.Place = c.value(x.Result), c.place(x.Place)
		return &y, true, nil
	case *ir.Store:
		if c.eliminated[x.Place] {
			value := c.value(x.Value)
			c.forwarded = &value
			return nil, false, nil
		}
		y := *x
		y.Place, y.Value = c.place(x.Place), c.value(x.Value)
		return &y, true, nil
	case *ir.ArrayLength:
		if c.eliminated[x.Place] {
			return nil, false, fmt.Errorf("cannot eliminate transient length read inside stage")
		}
		y := *x
		y.Result, y.Place = c.value(x.Result), c.place(x.Place)
		return &y, true, nil
	case *ir.Atomic:
		y := *x
		y.Result, y.Place, y.Value = c.value(x.Result), c.place(x.Place), c.value(x.Value)
		return &y, true, nil
	case *ir.Barrier:
		y := *x
		return &y, true, nil
	case *ir.If:
		y := *x
		y.Cond = c.value(x.Cond)
		y.Results = make([]ir.Result, len(x.Results))
		for i, r := range x.Results {
			y.Results[i] = ir.Result{ID: c.value(r.ID), Type: r.Type}
		}
		var err error
		y.Then, err = cloneFusionBlock(x.Then, c, false)
		if err != nil {
			return nil, false, err
		}
		y.Else, err = cloneFusionBlock(x.Else, c, false)
		return &y, true, err
	case *ir.Loop:
		return nil, false, fmt.Errorf("loop fusion composition is not representable in the current bounded composer")
	case *ir.Scope:
		y := *x
		var err error
		y.Body, err = cloneFusionBlock(x.Body, c, false)
		return &y, true, err
	default:
		return nil, false, fmt.Errorf("unknown instruction %T", in)
	}
}

func rebuildProgram(module *flow.Module, program *flow.Program, removed flow.ResourceID) {
	resourceMap := map[flow.ResourceID]flow.ResourceID{}
	resources := program.Resources[:0]
	for _, resource := range program.Resources {
		if resource.ID == removed {
			continue
		}
		old := resource.ID
		resource.ID = flow.ResourceID(len(resources) + 1)
		resourceMap[old] = resource.ID
		resources = append(resources, resource)
	}
	program.Resources = resources
	for i := range program.Parameters {
		if program.Parameters[i].Kind == flow.BufferParameter {
			program.Parameters[i].Resource = resourceMap[program.Parameters[i].Resource]
		}
	}
	program.Versions = nil
	current := map[flow.ResourceID]flow.VersionID{}
	for i := range program.Resources {
		resource := &program.Resources[i]
		defined := resource.Kind == flow.External
		id := flow.VersionID(len(program.Versions) + 1)
		program.Versions = append(program.Versions, flow.Version{ID: id, Resource: resource.ID, Defined: defined})
		resource.Initial = id
		current[resource.ID] = id
	}
	for dispatchIndex := range program.Dispatches {
		dispatch := &program.Dispatches[dispatchIndex]
		dispatch.ID = flow.DispatchID(dispatchIndex + 1)
		stage := module.Kernel.Function(dispatch.Stage)
		summary := ir.AnalyzeAccess(stage)
		for i := range dispatch.Buffers {
			argument := &dispatch.Buffers[i]
			argument.Resource = resourceMap[argument.Resource]
			argument.Input = current[argument.Resource]
			argument.Output = 0
			if stage.BufferParams[argument.Formal].Access == ir.Mutable {
				input := program.Version(argument.Input)
				defined := input.Defined || summary.Buffers[argument.Formal].CompleteWrite
				output := flow.VersionID(len(program.Versions) + 1)
				program.Versions = append(program.Versions, flow.Version{ID: output, Resource: argument.Resource, Previous: argument.Input, Producer: dispatch.ID, Defined: defined})
				argument.Output = output
				current[argument.Resource] = output
			}
		}
	}
	for i := range program.Resources {
		program.Resources[i].Final = current[program.Resources[i].ID]
	}
	for i := range program.Shapes {
		if program.Shapes[i].Resource != 0 {
			program.Shapes[i].Resource = resourceMap[program.Shapes[i].Resource]
		}
	}
	program.SyncIDs()
}
