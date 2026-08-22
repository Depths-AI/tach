package semantics

import (
	"fmt"
	"maps"

	"tach/ir"
)

func canonicalizeBlock(f *ir.Function, b *ir.Block, replacements map[ir.ValueID]ir.ValueID, placeReplacements map[ir.PlaceID]ir.PlaceID, placeResources map[ir.PlaceID]int, available map[valueKey]ir.ValueID, availablePlaces map[placeKey]ir.PlaceID) {
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
		ir.RewriteOperands(in, resolve, resolvePlace)
		switch x := in.(type) {
		case *ir.If:
			canonicalizeBlock(f, x.Then, replacements, placeReplacements, placeResources, maps.Clone(available), maps.Clone(availablePlaces))
			canonicalizeBlock(f, x.Else, replacements, placeReplacements, placeResources, maps.Clone(available), maps.Clone(availablePlaces))
		case *ir.Loop:
			canonicalizeBlock(f, x.Cond, replacements, placeReplacements, placeResources, maps.Clone(available), maps.Clone(availablePlaces))
			canonicalizeBlock(f, x.Body, replacements, placeReplacements, placeResources, maps.Clone(available), maps.Clone(availablePlaces))
		case *ir.Scope:
			canonicalizeBlock(f, x.Body, replacements, placeReplacements, placeResources, maps.Clone(available), maps.Clone(availablePlaces))
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
		key, pure := pureValueKey(f, in, placeResources)
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
	ir.RewriteTerm(b.Term, resolve)
}

func reusablePlace(in ir.Instr, resources map[ir.PlaceID]int) (placeKey, int) {
	switch x := in.(type) {
	case *ir.PlaceRoot:
		return placeKey{kind: 1, aux: x.Buffer}, x.Buffer
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

func pureValueKey(f *ir.Function, in ir.Instr, resources map[ir.PlaceID]int) (valueKey, bool) {
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
		if !ok || resource < 0 || resource >= len(f.BufferParams) || f.BufferParams[resource].Access != ir.Read {
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
