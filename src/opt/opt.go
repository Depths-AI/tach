package opt

import (
	"fmt"
	"maps"

	"tach/src/ir"
	"tach/src/source"
	"tach/src/types"
)

// Run applies Tach's target-independent Core IR canonicalization passes. Passes
// are deliberately semantics-based: they know nothing about any backend.
func Run(m *ir.Module) error {
	if err := ir.Verify(m); err != nil {
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
	if err := ir.Verify(m); err != nil {
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
func commonValues(m *ir.Module, f *ir.Function) {
	canonicalizeBlock(m, f.Body, map[ir.ValueID]ir.ValueID{}, map[ir.PlaceID]ir.PlaceID{}, map[ir.PlaceID]int{}, map[valueKey]ir.ValueID{}, map[placeKey]ir.PlaceID{})
}

func hoistLoopInvariants(m *ir.Module, f *ir.Function) {
	resources := placeResourceRoots(f)
	hoistNestedLoops(m, f.Body, resources)
}

func hoistNestedLoops(m *ir.Module, block *ir.Block, resources map[ir.PlaceID]int) {
	out := make([]ir.Instr, 0, len(block.Instrs))
	for _, in := range block.Instrs {
		switch x := in.(type) {
		case *ir.If:
			hoistNestedLoops(m, x.Then, resources)
			hoistNestedLoops(m, x.Else, resources)
		case *ir.Loop:
			hoistNestedLoops(m, x.Cond, resources)
			hoistNestedLoops(m, x.Body, resources)
			out = append(out, takeLoopInvariants(m, x, resources)...)
		}
		out = append(out, in)
	}
	block.Instrs = out
}

func takeLoopInvariants(m *ir.Module, loop *ir.Loop, resources map[ir.PlaceID]int) []ir.Instr {
	values, places := loopDefinitions(loop)
	var hoisted []ir.Instr
	for {
		changed := false
		for _, block := range []*ir.Block{loop.Cond, loop.Body} {
			kept := block.Instrs[:0]
			for _, in := range block.Instrs {
				if loopInvariant(m, in, values, places, resources, block == loop.Cond) {
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

func loopInvariant(m *ir.Module, in ir.Instr, values map[ir.ValueID]bool, places map[ir.PlaceID]bool, resources map[ir.PlaceID]int, allowLoad bool) bool {
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
		return allowLoad && place(x.Place) && immutableResource(m, resources, x.Place)
	case *ir.ArrayLength:
		return place(x.Place)
	default:
		return false
	}
}

func immutableResource(m *ir.Module, resources map[ir.PlaceID]int, place ir.PlaceID) bool {
	resource, ok := resources[place]
	return ok && resource >= 0 && resource < len(m.Resources) &&
		(m.Resources[resource].Kind == ir.Uniform || m.Resources[resource].Access == ir.Read)
}

func placeResourceRoots(f *ir.Function) map[ir.PlaceID]int {
	resources := map[ir.PlaceID]int{}
	var walk func(*ir.Block)
	walk = func(block *ir.Block) {
		for _, in := range block.Instrs {
			switch x := in.(type) {
			case *ir.PlaceRoot:
				resources[x.Result] = x.Resource
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

func promoteLoopBufferValues(m *ir.Module, f *ir.Function) {
	resources := placeResourceRoots(f)
	next := maximumValueID(f)
	promoteNestedLoops(m, f.Body, resources, &next)
}

func promoteNestedLoops(m *ir.Module, block *ir.Block, resources map[ir.PlaceID]int, next *ir.ValueID) {
	out := make([]ir.Instr, 0, len(block.Instrs))
	for _, in := range block.Instrs {
		switch x := in.(type) {
		case *ir.If:
			promoteNestedLoops(m, x.Then, resources, next)
			promoteNestedLoops(m, x.Else, resources, next)
		case *ir.Loop:
			promoteNestedLoops(m, x.Cond, resources, next)
			promoteNestedLoops(m, x.Body, resources, next)
			before, after := promoteLoop(m, x, resources, next)
			out = append(out, before...)
			out = append(out, in)
			out = append(out, after...)
			continue
		}
		out = append(out, in)
	}
	block.Instrs = out
}

func promoteLoop(m *ir.Module, loop *ir.Loop, resources map[ir.PlaceID]int, next *ir.ValueID) (before, after []ir.Instr) {
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
		if len(reads) != 1 || loopPlaces[place] || !exists || resource < 0 || resource >= len(m.Resources) || !zeroable(reads[0].Type) {
			continue
		}
		var store *ir.Store
		if len(writes) == 0 && immutableResource(m, resources, place) {
			// Cache an invariant read on the first executed iteration.
		} else if len(writes) == 1 && m.Resources[resource].Kind == ir.Buffer &&
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
		&ir.Const{Result: didNotRun, Type: types.TBool, Raw: "false", Span: loop.Span},
		&ir.Const{Result: didRun, Type: types.TBool, Raw: "true", Span: loop.Span},
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
	loop.Params = append(loop.Params, ir.LoopParam{ID: ranParameter, Type: types.TBool, Init: didNotRun})
	loop.Results = append(loop.Results, ir.Result{ID: ranResult, Type: types.TBool})
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

func zeroable(t *types.Type) bool {
	return types.IsScalar(t) || t.Kind == types.Vector
}

func zeroValue(t *types.Type, span source.Span, fresh func() ir.ValueID, out []ir.Instr) (ir.ValueID, []ir.Instr) {
	if t.Kind != types.Vector {
		id := fresh()
		raw := "0"
		if t.Kind == types.Bool {
			raw = "false"
		} else if t.Kind == types.F32 {
			raw = "0.0"
		}
		return id, append(out, &ir.Const{Result: id, Type: t, Raw: raw, Span: span})
	}
	lane, out := zeroValue(t.Elem, span, fresh, out)
	id := fresh()
	values := make([]ir.ValueID, t.Lanes)
	for index := range values {
		values[index] = lane
	}
	return id, append(out, &ir.Composite{Result: id, Type: t, Values: values, Span: span})
}

func rewriteBlockValues(block *ir.Block, resolve func(ir.ValueID) ir.ValueID) {
	for _, in := range block.Instrs {
		rewriteOperands(in, resolve, func(id ir.PlaceID) ir.PlaceID { return id })
		switch x := in.(type) {
		case *ir.If:
			rewriteBlockValues(x.Then, resolve)
			rewriteBlockValues(x.Else, resolve)
		case *ir.Loop:
			rewriteBlockValues(x.Cond, resolve)
			rewriteBlockValues(x.Body, resolve)
		}
	}
	rewriteTerm(block.Term, resolve)
}

// DECISION: Promotion stops at synchronization, early exits, and any other
// access to the same buffer. Memory SSA plus alias analysis can safely widen
// this ceiling later; guessing would silently reorder observable memory.
func loopHasUnsafeExitOrSync(loop *ir.Loop) bool {
	unsafe := false
	var walk func(*ir.Block)
	walk = func(block *ir.Block) {
		if _, ok := block.Term.(*ir.Return); ok {
			unsafe = true
		}
		if _, ok := block.Term.(*ir.Unreachable); ok {
			unsafe = true
		}
		for _, in := range block.Instrs {
			switch x := in.(type) {
			case *ir.Atomic, *ir.Barrier:
				unsafe = true
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

func maximumValueID(f *ir.Function) ir.ValueID {
	var maximum ir.ValueID
	remember := func(id ir.ValueID) {
		if id > maximum {
			maximum = id
		}
	}
	for _, parameter := range f.Params {
		remember(parameter.ID)
	}
	for _, index := range f.Indices {
		remember(index.ID)
	}
	var walk func(*ir.Block)
	walk = func(block *ir.Block) {
		for _, in := range block.Instrs {
			if definition, ok := in.(ir.ValueDef); ok {
				remember(definition.ResultValue())
			}
			switch x := in.(type) {
			case *ir.If:
				for _, result := range x.Results {
					remember(result.ID)
				}
				walk(x.Then)
				walk(x.Else)
			case *ir.Loop:
				for _, parameter := range x.Params {
					remember(parameter.ID)
				}
				for _, result := range x.Results {
					remember(result.ID)
				}
				walk(x.Cond)
				walk(x.Body)
			}
		}
	}
	walk(f.Body)
	return maximum
}

func canonicalizeBlock(m *ir.Module, b *ir.Block, replacements map[ir.ValueID]ir.ValueID, placeReplacements map[ir.PlaceID]ir.PlaceID, placeResources map[ir.PlaceID]int, available map[valueKey]ir.ValueID, availablePlaces map[placeKey]ir.PlaceID) {
	resolve := func(id ir.ValueID) ir.ValueID {
		for replacements[id] != 0 {
			id = replacements[id]
		}
		return id
	}
	resolvePlace := func(id ir.PlaceID) ir.PlaceID {
		for placeReplacements[id] != 0 {
			id = placeReplacements[id]
		}
		return id
	}
	out := b.Instrs[:0]
	for _, in := range b.Instrs {
		rewriteOperands(in, resolve, resolvePlace)
		switch x := in.(type) {
		case *ir.If:
			canonicalizeBlock(m, x.Then, replacements, placeReplacements, placeResources, maps.Clone(available), maps.Clone(availablePlaces))
			canonicalizeBlock(m, x.Else, replacements, placeReplacements, placeResources, maps.Clone(available), maps.Clone(availablePlaces))
		case *ir.Loop:
			canonicalizeBlock(m, x.Cond, replacements, placeReplacements, placeResources, maps.Clone(available), maps.Clone(availablePlaces))
			canonicalizeBlock(m, x.Body, replacements, placeReplacements, placeResources, maps.Clone(available), maps.Clone(availablePlaces))
		}
		if definition, ok := in.(ir.PlaceDef); ok {
			key, resource := reusablePlace(in, placeResources)
			placeResources[definition.ResultPlace()] = resource
			if previous := availablePlaces[key]; previous != 0 {
				placeReplacements[definition.ResultPlace()] = previous
				continue
			}
			availablePlaces[key] = definition.ResultPlace()
		}
		key, pure := pureValueKey(m, in, placeResources)
		definition, defines := in.(ir.ValueDef)
		if pure && defines && definition.ResultValue() != 0 {
			if previous := available[key]; previous != 0 {
				replacements[definition.ResultValue()] = previous
				continue
			}
			available[key] = definition.ResultValue()
		}
		out = append(out, in)
	}
	b.Instrs = out
	rewriteTerm(b.Term, resolve)
}

func reusablePlace(in ir.Instr, resources map[ir.PlaceID]int) (placeKey, int) {
	switch x := in.(type) {
	case *ir.PlaceRoot:
		return placeKey{kind: 1, aux: x.Resource}, x.Resource
	case *ir.PlaceWorkgroup:
		return placeKey{kind: 2, aux: x.Workgroup}, -1
	case *ir.PlaceField:
		return placeKey{kind: 3, base: x.Base, aux: x.Field}, resources[x.Base]
	case *ir.PlaceIndex:
		return placeKey{kind: 4, base: x.Base, index: x.Index}, resources[x.Base]
	default:
		panic(fmt.Sprintf("non-place definition %T", in))
	}
}

func rewriteOperands(in ir.Instr, resolve func(ir.ValueID) ir.ValueID, resolvePlace func(ir.PlaceID) ir.PlaceID) {
	switch x := in.(type) {
	case *ir.Unary:
		x.X = resolve(x.X)
	case *ir.Binary:
		x.Left, x.Right = resolve(x.Left), resolve(x.Right)
	case *ir.Intrinsic:
		for i := range x.Args {
			x.Args[i] = resolve(x.Args[i])
		}
	case *ir.Convert:
		x.X = resolve(x.X)
	case *ir.Composite:
		for i := range x.Values {
			x.Values[i] = resolve(x.Values[i])
		}
	case *ir.Extract:
		x.Base = resolve(x.Base)
	case *ir.VectorIndex:
		x.Base, x.Index = resolve(x.Base), resolve(x.Index)
	case *ir.Call:
		for i := range x.Args {
			x.Args[i] = resolve(x.Args[i])
		}
	case *ir.PlaceIndex:
		x.Base = resolvePlace(x.Base)
		x.Index = resolve(x.Index)
	case *ir.PlaceField:
		x.Base = resolvePlace(x.Base)
	case *ir.Load:
		x.Place = resolvePlace(x.Place)
	case *ir.Store:
		x.Place = resolvePlace(x.Place)
		x.Value = resolve(x.Value)
	case *ir.ArrayLength:
		x.Place = resolvePlace(x.Place)
	case *ir.Atomic:
		x.Place = resolvePlace(x.Place)
		x.Value = resolve(x.Value)
	case *ir.If:
		x.Cond = resolve(x.Cond)
	case *ir.Loop:
		for i := range x.Params {
			x.Params[i].Init = resolve(x.Params[i].Init)
		}
	}
}

func rewriteTerm(term ir.Term, resolve func(ir.ValueID) ir.ValueID) {
	var values []ir.ValueID
	switch x := term.(type) {
	case *ir.Yield:
		values = x.Values
	case *ir.Continue:
		values = x.Values
	case *ir.Return:
		if x.HasValue {
			x.Value = resolve(x.Value)
		}
	}
	for i := range values {
		values[i] = resolve(values[i])
	}
}

func pureValueKey(m *ir.Module, in ir.Instr, resources map[ir.PlaceID]int) (valueKey, bool) {
	key := valueKey{}
	// DECISION: Four operands cover today's scalar/vector operations. Wider
	// composites and calls skip CSE until profiles justify an allocated key.
	set := func(kind uint8, t string, op string, values ...ir.ValueID) (valueKey, bool) {
		if len(values) > len(key.args) {
			return key, false
		}
		key.kind, key.typeName, key.op, key.count = kind, t, op, uint8(len(values))
		copy(key.args[:], values)
		return key, true
	}
	switch x := in.(type) {
	case *ir.Const:
		return set(1, x.Type.String(), x.Raw)
	case *ir.Unary:
		return set(3, x.Type.String(), x.Op, x.X)
	case *ir.Binary:
		return set(4, x.Type.String(), x.Op, x.Left, x.Right)
	case *ir.Intrinsic:
		return set(5, x.Type.String(), x.Kind.String(), x.Args...)
	case *ir.Convert:
		return set(6, x.Type.String(), x.From.String(), x.X)
	case *ir.Composite:
		return set(7, x.Type.String(), "", x.Values...)
	case *ir.Extract:
		key, _ = set(8, x.Type.String(), "", x.Base)
		key.aux = x.Index
		return key, true
	case *ir.VectorIndex:
		return set(12, x.Type.String(), "", x.Base, x.Index)
	case *ir.Call:
		return set(9, x.Type.String(), x.Function, x.Args...)
	case *ir.Load:
		resource, ok := resources[x.Place]
		if !ok || resource < 0 || resource >= len(m.Resources) ||
			(m.Resources[resource].Kind != ir.Uniform && m.Resources[resource].Access != ir.Read) {
			return key, false
		}
		key, _ = set(10, x.Type.String(), "")
		key.aux = int(x.Place)
		return key, true
	case *ir.ArrayLength:
		key, _ = set(11, x.Type.String(), "")
		key.aux = int(x.Place)
		return key, true
	default:
		return key, false
	}
}

// deadValues recursively removes unused, side-effect-free SSA definitions.
// Recomputing use counts to a fixed point keeps the implementation small and
// catches whole dead expression trees without depending on instruction order.
func deadValues(f *ir.Function) error {
	for {
		values, places, err := ir.UseCounts(f)
		if err != nil {
			return err
		}
		changed := pruneBlock(f.Body, values, places)
		if !changed {
			return nil
		}
	}
}

func pruneBlock(b *ir.Block, values map[ir.ValueID]int, places map[ir.PlaceID]int) bool {
	changed := false
	out := b.Instrs[:0]
	for _, in := range b.Instrs {
		switch x := in.(type) {
		case *ir.If:
			if pruneBlock(x.Then, values, places) {
				changed = true
			}
			if pruneBlock(x.Else, values, places) {
				changed = true
			}
		case *ir.Loop:
			if pruneBlock(x.Cond, values, places) {
				changed = true
			}
			if pruneBlock(x.Body, values, places) {
				changed = true
			}
		}
		if d, ok := in.(ir.ValueDef); ok && isDeadRemovable(in) && values[d.ResultValue()] == 0 {
			changed = true
			continue
		}
		if d, ok := in.(ir.PlaceDef); ok && places[d.ResultPlace()] == 0 {
			changed = true
			continue
		}
		out = append(out, in)
	}
	b.Instrs = out
	return changed
}

func isDeadRemovable(in ir.Instr) bool {
	switch in.(type) {
	case *ir.Const, *ir.Unary, *ir.Binary, *ir.Intrinsic, *ir.Convert, *ir.Composite, *ir.Extract, *ir.VectorIndex, *ir.Call, *ir.Load, *ir.ArrayLength:
		return true
	default:
		return false
	}
}
