package ir

import (
	"fmt"

	"tach/src/types"
)

// UsesKind reports whether a type kind occurs anywhere in a module's declared
// or instruction-level type surface.
func UsesKind(module *Module, kind types.Kind) bool {
	for _, item := range module.Structs {
		if types.Contains(item, kind) {
			return true
		}
	}
	var block func(*Block) bool
	block = func(body *Block) bool {
		for _, instruction := range body.Instrs {
			if value, ok := instruction.(ValueDef); ok && types.Contains(value.ResultType(), kind) {
				return true
			}
			if place, ok := instruction.(PlaceDef); ok && types.Contains(place.PlaceType(), kind) {
				return true
			}
			switch item := instruction.(type) {
			case *If:
				for _, result := range item.Results {
					if types.Contains(result.Type, kind) {
						return true
					}
				}
				if block(item.Then) || block(item.Else) {
					return true
				}
			case *Loop:
				for _, result := range item.Results {
					if types.Contains(result.Type, kind) {
						return true
					}
				}
				for _, parameter := range item.Params {
					if types.Contains(parameter.Type, kind) {
						return true
					}
				}
				if block(item.Cond) || block(item.Body) {
					return true
				}
			case *Scope:
				if block(item.Body) {
					return true
				}
			}
		}
		return false
	}
	for _, function := range module.Functions {
		for _, parameter := range function.BufferParams {
			if types.Contains(parameter.Type, kind) {
				return true
			}
		}
		for _, parameter := range function.Params {
			if types.Contains(parameter.Type, kind) {
				return true
			}
		}
		for _, variable := range function.WorkgroupVars {
			if types.Contains(variable.Type, kind) {
				return true
			}
		}
		if types.Contains(function.Return, kind) || block(function.Body) {
			return true
		}
	}
	return false
}

// UseCounts returns every SSA-value and place read in f. Optimizers share this
// accounting so a new IR instruction cannot silently disappear in one stage.
func UseCounts(f *Function) (map[ValueID]int, map[PlaceID]int, error) {
	values := map[ValueID]int{}
	places := map[PlaceID]int{}
	useValue := func(id ValueID) {
		if id != 0 {
			values[id]++
		}
	}
	usePlace := func(id PlaceID) {
		if id != 0 {
			places[id]++
		}
	}
	var block func(*Block) error
	block = func(b *Block) error {
		for _, in := range b.Instrs {
			switch x := in.(type) {
			case *Const, *PlaceRoot, *PlaceWorkgroup, *Barrier:
			case *Unary:
				useValue(x.X)
			case *Binary:
				useValue(x.Left)
				useValue(x.Right)
			case *Intrinsic:
				for _, id := range x.Args {
					useValue(id)
				}
			case *Convert:
				useValue(x.X)
			case *Composite:
				for _, id := range x.Values {
					useValue(id)
				}
			case *Extract:
				useValue(x.Base)
			case *VectorIndex:
				useValue(x.Base)
				useValue(x.Index)
			case *Call:
				for _, id := range x.Args {
					useValue(id)
				}
			case *PlaceField:
				usePlace(x.Base)
			case *PlaceIndex:
				usePlace(x.Base)
				useValue(x.Index)
			case *Load:
				usePlace(x.Place)
			case *Store:
				usePlace(x.Place)
				useValue(x.Value)
			case *ArrayLength:
				usePlace(x.Place)
			case *Atomic:
				usePlace(x.Place)
				useValue(x.Value)
			case *If:
				useValue(x.Cond)
				if err := block(x.Then); err != nil {
					return err
				}
				if err := block(x.Else); err != nil {
					return err
				}
			case *Loop:
				for _, p := range x.Params {
					useValue(p.Init)
				}
				if err := block(x.Cond); err != nil {
					return err
				}
				if err := block(x.Body); err != nil {
					return err
				}
			case *Scope:
				if err := block(x.Body); err != nil {
					return err
				}
			default:
				return fmt.Errorf("unhandled IR instruction %T", in)
			}
		}
		switch x := b.Term.(type) {
		case *Yield:
			for _, id := range x.Values {
				useValue(id)
			}
		case *Continue:
			for _, id := range x.Values {
				useValue(id)
			}
		case *Return:
			if x.HasValue {
				useValue(x.Value)
			}
		case *Unreachable, *ExitScope, nil:
		default:
			return fmt.Errorf("unhandled IR terminator %T", b.Term)
		}
		return nil
	}
	return values, places, block(f.Body)
}
