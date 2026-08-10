package ir

import (
	"fmt"
)

// Pine owns barrier legality at the semantic IR layer. The analysis is scoped
// to a workgroup: a value is uniform when every invocation in a workgroup is
// guaranteed to observe the same value. This is deliberately conservative for
// mutable storage/workgroup reads while remaining compositional for SSA values.
type uniformPlace struct {
	addr     bool
	resource int // -1 for workgroup memory
}

type uniformEnv struct {
	values map[ValueID]bool
	places map[PlaceID]uniformPlace
}

func newUniformEnv() uniformEnv {
	return uniformEnv{values: map[ValueID]bool{}, places: map[PlaceID]uniformPlace{}}
}
func (e uniformEnv) clone() uniformEnv {
	v := make(map[ValueID]bool, len(e.values))
	for k, x := range e.values {
		v[k] = x
	}
	p := make(map[PlaceID]uniformPlace, len(e.places))
	for k, x := range e.places {
		p[k] = x
	}
	return uniformEnv{values: v, places: p}
}

func verifyUniformity(m *Module, f *Function, fmap map[string]*Function) error {
	e := newUniformEnv()
	_, _, err := analyzeUniformBlock(m, f, f.Body, e, fmap, true, true)
	return err
}

// analyzeUniformBlock returns the environment at the terminator and whether
// control reaching that terminator is workgroup-uniform. checkBarriers is false
// only during loop-carried fixed-point discovery.
func analyzeUniformBlock(m *Module, f *Function, b *Block, e uniformEnv, fmap map[string]*Function, control, checkBarriers bool) (uniformEnv, bool, error) {
	value := func(id ValueID) bool { return e.values[id] }
	for _, in := range b.Instrs {
		switch x := in.(type) {
		case *Const:
			e.values[x.Result] = true
		case *Builtin:
			e.values[x.Result] = x.Kind == WorkgroupID || x.Kind == NumWorkgroups
		case *Unary:
			e.values[x.Result] = value(x.X)
		case *Binary:
			e.values[x.Result] = value(x.Left) && value(x.Right)
		case *Convert:
			e.values[x.Result] = value(x.X)
		case *Composite:
			u := true
			for _, id := range x.Values {
				u = u && value(id)
			}
			e.values[x.Result] = u
		case *Extract:
			e.values[x.Result] = value(x.Base)
		case *Intrinsic:
			u := true
			for _, id := range x.Args {
				u = u && value(id)
			}
			e.values[x.Result] = u
		case *Call:
			u := true
			for _, id := range x.Args {
				u = u && value(id)
			}
			// Pine helpers are value-only and cannot observe compute builtins,
			// resources, workgroup memory, atomics, or barriers. Therefore equal
			// arguments imply equal results across the workgroup.
			if callee := fmap[x.Function]; callee == nil || callee.Compute {
				u = false
			}
			if x.Result != 0 {
				e.values[x.Result] = u
			}
		case *PlaceRoot:
			e.places[x.Result] = uniformPlace{addr: true, resource: x.Resource}
		case *PlaceWorkgroup:
			e.places[x.Result] = uniformPlace{addr: true, resource: -1}
		case *PlaceField:
			p := e.places[x.Base]
			e.places[x.Result] = p
		case *PlaceIndex:
			p := e.places[x.Base]
			p.addr = p.addr && value(x.Index)
			e.places[x.Result] = p
		case *Load:
			p := e.places[x.Place]
			u := false
			if p.resource >= 0 && p.resource < len(m.Resources) && m.Resources[p.resource].Kind == Uniform {
				u = p.addr
			}
			e.values[x.Result] = u
		case *Store:
			// Stores do not define SSA values. Mutable memory is conservatively
			// classified as varying on every subsequent load.
		case *ArrayLength:
			p := e.places[x.Place]
			e.values[x.Result] = p.addr && p.resource >= 0
		case *Atomic:
			if x.Result != 0 {
				e.values[x.Result] = false
			}
		case *Barrier:
			if checkBarriers && !control {
				name := "workgroupBarrier"
				if x.Kind == BarrierStorage {
					name = "storageBarrier"
				}
				return e, control, fmt.Errorf("%s is reached through non-uniform control flow", name)
			}
		case *If:
			condUniform := value(x.Cond)
			branchControl := control && condUniform
			te, tc, err := analyzeUniformBlock(m, f, x.Then, e.clone(), fmap, branchControl, checkBarriers)
			if err != nil {
				return e, control, fmt.Errorf("if then uniformity: %w", err)
			}
			ee, ec, err := analyzeUniformBlock(m, f, x.Else, e.clone(), fmap, branchControl, checkBarriers)
			if err != nil {
				return e, control, fmt.Errorf("if else uniformity: %w", err)
			}
			ty, thenYields := x.Then.Term.(*Yield)
			ey, elseYields := x.Else.Term.(*Yield)
			for i, r := range x.Results {
				u := condUniform
				if thenYields {
					u = u && te.values[ty.Values[i]]
				}
				if elseYields {
					u = u && ee.values[ey.Values[i]]
				}
				e.values[r.ID] = u
			}
			// A normal structured merge reconverges. If a varying branch exits
			// instead of yielding to the merge, only a subset reaches later code.
			if !condUniform && thenYields != elseYields {
				control = false
			}
			_ = tc
			_ = ec
		case *Loop:
			var err error
			e, control, err = analyzeUniformLoop(m, f, x, e, fmap, control, checkBarriers)
			if err != nil {
				return e, control, err
			}
		default:
			return e, control, fmt.Errorf("uniformity analysis does not understand %T", in)
		}
	}
	return e, control, nil
}

func analyzeUniformLoop(m *Module, f *Function, loop *Loop, outer uniformEnv, fmap map[string]*Function, control, checkBarriers bool) (uniformEnv, bool, error) {
	paramUniform := make([]bool, len(loop.Params))
	for i, p := range loop.Params {
		paramUniform[i] = outer.values[p.Init]
	}

	// Uniformity is a descending lattice. Iterate loop-carried values to a fixed
	// point before validating barriers, so second-and-later iterations are
	// accounted for rather than only the first trip through the loop.
	for iter := 0; iter <= len(loop.Params); iter++ {
		le := outer.clone()
		for i, p := range loop.Params {
			le.values[p.ID] = paramUniform[i]
		}
		ce, _, err := analyzeUniformBlock(m, f, loop.Cond, le.clone(), fmap, control, false)
		if err != nil {
			return outer, control, fmt.Errorf("loop condition uniformity: %w", err)
		}
		cy := loop.Cond.Term.(*Yield)
		condUniform := ce.values[cy.Values[0]]
		be, _, err := analyzeUniformBlock(m, f, loop.Body, le.clone(), fmap, control && condUniform, false)
		if err != nil {
			return outer, control, fmt.Errorf("loop body uniformity: %w", err)
		}
		co := loop.Body.Term.(*Continue)
		changed := false
		for i := range paramUniform {
			next := paramUniform[i] && be.values[co.Values[i]]
			if next != paramUniform[i] {
				paramUniform[i] = next
				changed = true
			}
		}
		if !changed {
			break
		}
	}

	le := outer.clone()
	for i, p := range loop.Params {
		le.values[p.ID] = paramUniform[i]
	}
	// Loop-carried values may vary while every invocation still executes the
	// same number of iterations. Control convergence depends on the condition,
	// not on unrelated carried data.
	loopControl := control
	ce, _, err := analyzeUniformBlock(m, f, loop.Cond, le.clone(), fmap, loopControl, checkBarriers)
	if err != nil {
		return outer, control, fmt.Errorf("loop condition uniformity: %w", err)
	}
	cy := loop.Cond.Term.(*Yield)
	condUniform := ce.values[cy.Values[0]]
	be, _, err := analyzeUniformBlock(m, f, loop.Body, le.clone(), fmap, loopControl && condUniform, checkBarriers)
	if err != nil {
		return outer, control, fmt.Errorf("loop body uniformity: %w", err)
	}
	co := loop.Body.Term.(*Continue)
	for i, r := range loop.Results {
		outer.values[r.ID] = paramUniform[i] && condUniform && be.values[co.Values[i]]
	}
	// A varying iteration condition means invocations can leave the loop at
	// different dynamic points. Pine conservatively tracks subsequent control as
	// non-uniform, which keeps barrier legality target-independent.
	if !condUniform {
		control = false
	}
	return outer, control, nil
}

func allTrue(xs []bool) bool {
	for _, x := range xs {
		if !x {
			return false
		}
	}
	return true
}
