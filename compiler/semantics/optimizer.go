package semantics

import (
	"fmt"
	"tach/foundation"
	"tach/ir"
)

func optimize(module *ir.Module) error {
	if module == nil {
		return fmt.Errorf("nil logical module")
	}
	if err := optimizeKernel(module.Kernel); err != nil {
		return err
	}
	return ir.Verify(module)
}

func optimizeKernel(m *ir.KernelModule) error {
	if err := ir.VerifyKernel(m); err != nil {
		return fmt.Errorf("pre-optimization IR verification: %w", err)
	}
	for _, f := range m.Functions {
		commonValues(m, f)
		hoistLoopInvariants(m, f)
		commonValues(m, f)
		promoteLoopBufferValues(m, f)
		commonValues(m, f)
		if err := deadValues(f); err != nil {
			return fmt.Errorf("optimize function %s: %w", f.Name, err)
		}
	}
	if err := ir.VerifyKernel(m); err != nil {
		return fmt.Errorf("post-optimization IR verification: %w", err)
	}
	return nil
}

type valueKey struct {
	kind         uint8
	typeName, op string
	args         [4]ir.ValueID
	count        uint8
	aux          int
}

type placeKey struct {
	kind  uint8
	base  ir.PlaceID
	index ir.ValueID
	aux   int
}

// commonValues removes repeated pure SSA computations while preserving the
// structured dominance already present in Core IR.
func commonValues(m *ir.KernelModule, f *ir.Function) {
	canonicalizeBlock(f, f.Body, map[ir.ValueID]ir.ValueID{}, map[ir.PlaceID]ir.PlaceID{}, map[ir.PlaceID]int{}, map[valueKey]ir.ValueID{}, map[placeKey]ir.PlaceID{})
}

func hoistLoopInvariants(m *ir.KernelModule, f *ir.Function) {
	resources := placeResourceRoots(f)
	hoistNestedLoops(f, f.Body, resources)
}

func hoistNestedLoops(f *ir.Function, block *ir.Block, resources map[ir.PlaceID]int) {
	out := make([]ir.Instr, 0, len(block.Instrs))
	for _, in := range block.Instrs {
		switch x := in.(type) {
		case *ir.If:
			hoistNestedLoops(f, x.Then, resources)
			hoistNestedLoops(f, x.Else, resources)
		case *ir.Loop:
			hoistNestedLoops(f, x.Cond, resources)
			hoistNestedLoops(f, x.Body, resources)
			out = append(out, takeLoopInvariants(f, x, resources)...)
		case *ir.Scope:
			hoistNestedLoops(f, x.Body, resources)
		}
		out = append(out, in)
	}
	block.Instrs = out
}

func takeLoopInvariants(f *ir.Function, loop *ir.Loop, resources map[ir.PlaceID]int) []ir.Instr {
	values, places := loopDefinitions(loop)
	var hoisted []ir.Instr
	for {
		changed := false
		for _, block := range []*ir.Block{loop.Cond, loop.Body} {
			kept := block.Instrs[:0]
			for _, in := range block.Instrs {
				if loopInvariant(f, in, values, places, resources, block == loop.Cond) {
					hoisted = append(hoisted, in)
					forgetDefinition(in, values, places)
					changed = true
				} else {
					kept = append(kept, in)
				}
			}
			block.Instrs = kept
		}
		if !changed {
			return hoisted
		}
	}
}

func loopDefinitions(loop *ir.Loop) (map[ir.ValueID]bool, map[ir.PlaceID]bool) {
	values := map[ir.ValueID]bool{}
	places := map[ir.PlaceID]bool{}
	for _, parameter := range loop.Params {
		values[parameter.ID] = true
	}
	var collect func(*ir.Block)
	collect = func(block *ir.Block) {
		for _, in := range block.Instrs {
			if definition, ok := in.(ir.ValueDef); ok && definition.ResultValue() != 0 {
				values[definition.ResultValue()] = true
			}
			if definition, ok := in.(ir.PlaceDef); ok {
				places[definition.ResultPlace()] = true
			}
			switch x := in.(type) {
			case *ir.If:
				for _, result := range x.Results {
					values[result.ID] = true
				}
				collect(x.Then)
				collect(x.Else)
			case *ir.Loop:
				for _, parameter := range x.Params {
					values[parameter.ID] = true
				}
				for _, result := range x.Results {
					values[result.ID] = true
				}
				collect(x.Cond)
				collect(x.Body)
			case *ir.Scope:
				collect(x.Body)
			}
		}
	}
	collect(loop.Cond)
	collect(loop.Body)
	return values, places
}

func forgetDefinition(in ir.Instr, values map[ir.ValueID]bool, places map[ir.PlaceID]bool) {
	if definition, ok := in.(ir.ValueDef); ok {
		delete(values, definition.ResultValue())
	}
	if definition, ok := in.(ir.PlaceDef); ok {
		delete(places, definition.ResultPlace())
	}
}

func loopInvariant(f *ir.Function, in ir.Instr, values map[ir.ValueID]bool, places map[ir.PlaceID]bool, resources map[ir.PlaceID]int, allowLoad bool) bool {
	value := func(id ir.ValueID) bool { return id == 0 || !values[id] }
	place := func(id ir.PlaceID) bool { return id == 0 || !places[id] }
	all := func(ids []ir.ValueID) bool {
		for _, id := range ids {
			if !value(id) {
				return false
			}
		}
		return true
	}
	switch x := in.(type) {
	case *ir.Const, *ir.PlaceRoot, *ir.PlaceWorkgroup:
		return true
	case *ir.Unary:
		return value(x.X)
	case *ir.Binary:
		return value(x.Left) && value(x.Right)
	case *ir.Intrinsic:
		return all(x.Args)
	case *ir.Convert:
		return value(x.X)
	case *ir.Composite:
		return all(x.Values)
	case *ir.Extract:
		return value(x.Base)
	case *ir.VectorIndex:
		return value(x.Base) && value(x.Index)
	case *ir.Call:
		return x.Result != 0 && all(x.Args)
	case *ir.PlaceField:
		return place(x.Base)
	case *ir.PlaceIndex:
		return place(x.Base) && value(x.Index)
	case *ir.Load:
		// The condition always executes once. A body load does not execute when
		// the loop has zero iterations, so it must be cached lazily instead.
		return allowLoad && place(x.Place) && immutableResource(f, resources, x.Place)
	case *ir.ArrayLength:
		return place(x.Place)
	default:
		return false
	}
}

func immutableResource(f *ir.Function, resources map[ir.PlaceID]int, place ir.PlaceID) bool {
	buffer, ok := resources[place]
	return ok && buffer >= 0 && buffer < len(f.BufferParams) && f.BufferParams[buffer].Access == ir.Read
}

func placeResourceRoots(f *ir.Function) map[ir.PlaceID]int {
	resources := map[ir.PlaceID]int{}
	var walk func(*ir.Block)
	walk = func(block *ir.Block) {
		for _, in := range block.Instrs {
			switch x := in.(type) {
			case *ir.PlaceRoot:
				resources[x.Result] = x.Buffer
			case *ir.PlaceWorkgroup:
				resources[x.Result] = -1
			case *ir.PlaceField:
				resources[x.Result] = resources[x.Base]
			case *ir.PlaceIndex:
				resources[x.Result] = resources[x.Base]
			case *ir.If:
				walk(x.Then)
				walk(x.Else)
			case *ir.Loop:
				walk(x.Cond)
				walk(x.Body)
			case *ir.Scope:
				walk(x.Body)
			}
		}
	}
	walk(f.Body)
	return resources
}

type promotedBufferValue struct {
	load     *ir.Load
	store    *ir.Store
	paramID  ir.ValueID
	resultID ir.ValueID
	valueID  ir.ValueID
	initID   ir.ValueID
}

func promoteLoopBufferValues(m *ir.KernelModule, f *ir.Function) {
	resources := placeResourceRoots(f)
	next := ir.MaxValueID(f)
	promoteNestedLoops(f, f.Body, resources, &next)
}

func promoteNestedLoops(f *ir.Function, block *ir.Block, resources map[ir.PlaceID]int, next *ir.ValueID) {
	out := make([]ir.Instr, 0, len(block.Instrs))
	for _, in := range block.Instrs {
		switch x := in.(type) {
		case *ir.If:
			promoteNestedLoops(f, x.Then, resources, next)
			promoteNestedLoops(f, x.Else, resources, next)
		case *ir.Loop:
			promoteNestedLoops(f, x.Cond, resources, next)
			promoteNestedLoops(f, x.Body, resources, next)
			before, after := promoteLoop(f, x, resources, next)
			out = append(out, before...)
			out = append(out, in)
			out = append(out, after...)
			continue
		case *ir.Scope:
			promoteNestedLoops(f, x.Body, resources, next)
		}
		out = append(out, in)
	}
	block.Instrs = out
}

func promoteLoop(f *ir.Function, loop *ir.Loop, resources map[ir.PlaceID]int, next *ir.ValueID) (before, after []ir.Instr) {
	continuation, ok := loop.Body.Term.(*ir.Continue)
	if !ok || loopHasUnsafeExitOrSync(loop) {
		return nil, nil
	}
	_, loopPlaces := loopDefinitions(loop)
	loads := map[ir.PlaceID][]*ir.Load{}
	var loadPlaces []ir.PlaceID
	stores := map[ir.PlaceID][]*ir.Store{}
	loadOrder := map[*ir.Load]int{}
	storeOrder := map[*ir.Store]int{}
	for index, in := range loop.Body.Instrs {
		switch x := in.(type) {
		case *ir.Load:
			if len(loads[x.Place]) == 0 {
				loadPlaces = append(loadPlaces, x.Place)
			}
			loads[x.Place] = append(loads[x.Place], x)
			loadOrder[x] = index
		case *ir.Store:
			stores[x.Place] = append(stores[x.Place], x)
			storeOrder[x] = index
		}
	}
	var candidates []promotedBufferValue
	for _, place := range loadPlaces {
		reads := loads[place]
		writes := stores[place]
		resource, exists := resources[place]
		if len(reads) != 1 || loopPlaces[place] || !exists || resource < 0 || resource >= len(f.BufferParams) || !zeroable(reads[0].Type) {
			continue
		}
		var store *ir.Store
		if len(writes) == 0 && immutableResource(f, resources, place) {
			// Cache an invariant read on the first executed iteration.
		} else if len(writes) == 1 &&
			loadOrder[reads[0]] < storeOrder[writes[0]] {
			store = writes[0]
		} else {
			continue
		}
		if loopTouchesResourceElsewhere(loop, resources, resource, reads[0], store) {
			continue
		}
		candidates = append(candidates, promotedBufferValue{load: reads[0], store: store})
	}
	if len(candidates) == 0 {
		return nil, nil
	}
	fresh := func() ir.ValueID {
		*next = *next + 1
		return *next
	}
	didNotRun, didRun := fresh(), fresh()
	ranParameter, ranResult := fresh(), fresh()
	before = append(before,
		&ir.Const{Result: didNotRun, Type: foundation.BoolType, Raw: "false", Span: loop.Span},
		&ir.Const{Result: didRun, Type: foundation.BoolType, Raw: "true", Span: loop.Span},
	)
	replacements := map[ir.ValueID]ir.ValueID{}
	byLoad := map[*ir.Load]int{}
	removedStores := map[*ir.Store]bool{}
	for index := range candidates {
		candidate := &candidates[index]
		candidate.paramID, candidate.resultID, candidate.valueID = fresh(), fresh(), fresh()
		candidate.initID, before = zeroValue(candidate.load.Type, loop.Span, fresh, before)
		replacements[candidate.load.Result] = candidate.valueID
		byLoad[candidate.load] = index
		if candidate.store != nil {
			removedStores[candidate.store] = true
		}
		loop.Params = append(loop.Params, ir.LoopParam{ID: candidate.paramID, Type: candidate.load.Type, Init: candidate.initID})
		loop.Results = append(loop.Results, ir.Result{ID: candidate.resultID, Type: candidate.load.Type})
	}
	loop.Params = append(loop.Params, ir.LoopParam{ID: ranParameter, Type: foundation.BoolType, Init: didNotRun})
	loop.Results = append(loop.Results, ir.Result{ID: ranResult, Type: foundation.BoolType})
	rewriteBlockValues(loop.Body, func(id ir.ValueID) ir.ValueID {
		if replacement := replacements[id]; replacement != 0 {
			return replacement
		}
		return id
	})
	kept := make([]ir.Instr, 0, len(loop.Body.Instrs))
	for _, in := range loop.Body.Instrs {
		if load, ok := in.(*ir.Load); ok {
			if index, found := byLoad[load]; found {
				candidate := candidates[index]
				kept = append(kept, &ir.If{
					Results: []ir.Result{{ID: candidate.valueID, Type: load.Type}},
					Cond:    ranParameter,
					Then:    &ir.Block{Term: &ir.Yield{Values: []ir.ValueID{candidate.paramID}}},
					Else:    &ir.Block{Instrs: []ir.Instr{load}, Term: &ir.Yield{Values: []ir.ValueID{load.Result}}},
					Span:    load.Span,
				})
				continue
			}
		}
		if store, ok := in.(*ir.Store); ok && removedStores[store] {
			continue
		}
		kept = append(kept, in)
	}
	loop.Body.Instrs = kept
	for index := range candidates {
		candidate := &candidates[index]
		value := candidate.valueID
		if candidate.store != nil {
			value = candidate.store.Value
		}
		continuation.Values = append(continuation.Values, value)
	}
	continuation.Values = append(continuation.Values, didRun)
	then := &ir.Block{Term: &ir.Yield{}}
	for _, candidate := range candidates {
		if candidate.store != nil {
			then.Instrs = append(then.Instrs, &ir.Store{Place: candidate.store.Place, Value: candidate.resultID, Span: candidate.store.Span})
		}
	}
	if len(then.Instrs) > 0 {
		after = append(after, &ir.If{Cond: ranResult, Then: then, Else: &ir.Block{Term: &ir.Yield{}}, Span: loop.Span})
	}
	return before, after
}

func zeroable(t *foundation.Type) bool {
	return foundation.IsScalar(t) || t.Kind == foundation.VectorKind
}

func zeroValue(t *foundation.Type, span foundation.Span, fresh func() ir.ValueID, out []ir.Instr) (ir.ValueID, []ir.Instr) {
	lanes := 1
	if t.Kind == foundation.VectorKind {
		lanes = t.Lanes
	}
	id, instructions := ir.MaterializeConstant(&foundation.ConstantValue{Type: t, Bits: make([]uint32, lanes)}, span, fresh)
	return id, append(out, instructions...)
}

func rewriteBlockValues(block *ir.Block, resolve func(ir.ValueID) ir.ValueID) {
	for _, in := range block.Instrs {
		ir.RewriteOperands(in, resolve, func(id ir.PlaceID) ir.PlaceID { return id })
		switch x := in.(type) {
		case *ir.If:
			rewriteBlockValues(x.Then, resolve)
			rewriteBlockValues(x.Else, resolve)
		case *ir.Loop:
			rewriteBlockValues(x.Cond, resolve)
			rewriteBlockValues(x.Body, resolve)
		}
	}
	ir.RewriteTerm(block.Term, resolve)
}

// DECISION: Promotion stops at synchronization, early exits, and any other
// access to the same buffer. Memory SSA plus alias analysis can safely widen
// this ceiling later; guessing would silently reorder observable memory.
func loopHasUnsafeExitOrSync(loop *ir.Loop) bool {
	unsafe := false
	var walk func(*ir.Block, bool)
	walk = func(block *ir.Block, root bool) {
		if _, ok := block.Term.(*ir.Return); ok {
			unsafe = true
		}
		if _, ok := block.Term.(*ir.Unreachable); ok {
			unsafe = true
		}
		if _, ok := block.Term.(*ir.Break); ok {
			unsafe = true
		}
		if _, ok := block.Term.(*ir.Continue); ok && !root {
			unsafe = true
		}
		for _, in := range block.Instrs {
			switch x := in.(type) {
			case *ir.Atomic, *ir.Barrier:
				unsafe = true
			case *ir.If:
				walk(x.Then, false)
				walk(x.Else, false)
			case *ir.Loop:
				walk(x.Cond, false)
				walk(x.Body, false)
			}
		}
	}
	walk(loop.Cond, false)
	walk(loop.Body, true)
	return unsafe
}

func loopTouchesResourceElsewhere(loop *ir.Loop, resources map[ir.PlaceID]int, resource int, allowedLoad *ir.Load, allowedStore *ir.Store) bool {
	touches := false
	var walk func(*ir.Block)
	walk = func(block *ir.Block) {
		for _, in := range block.Instrs {
			switch x := in.(type) {
			case *ir.Load:
				if x != allowedLoad && resources[x.Place] == resource {
					touches = true
				}
			case *ir.Store:
				if x != allowedStore && resources[x.Place] == resource {
					touches = true
				}
			case *ir.Atomic:
				if resources[x.Place] == resource {
					touches = true
				}
			case *ir.If:
				walk(x.Then)
				walk(x.Else)
			case *ir.Loop:
				walk(x.Cond)
				walk(x.Body)
			}
		}
	}
	walk(loop.Cond)
	walk(loop.Body)
	return touches
}
