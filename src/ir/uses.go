package ir

import "fmt"

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
		case *Unreachable, nil:
		default:
			return fmt.Errorf("unhandled IR terminator %T", b.Term)
		}
		return nil
	}
	return values, places, block(f.Body)
}
