package opt

import (
	"fmt"

	"tach/src/ir"
)

// Run applies Tach's target-independent Core IR canonicalization passes. Passes
// are deliberately semantics-based: they know nothing about WGSL or SPIR-V.
func Run(m *ir.Module) error {
	if err := ir.Verify(m); err != nil {
		return fmt.Errorf("pre-optimization IR verification: %w", err)
	}
	for _, f := range m.Functions {
		if err := deadValues(f); err != nil {
			return fmt.Errorf("optimize function %s: %w", f.Name, err)
		}
	}
	if err := ir.Verify(m); err != nil {
		return fmt.Errorf("post-optimization IR verification: %w", err)
	}
	return nil
}

// deadValues recursively removes unused, side-effect-free SSA definitions.
// Recomputing use counts to a fixed point keeps the implementation small and
// catches whole dead expression trees without depending on instruction order.
func deadValues(f *ir.Function) error {
	for {
		uses := map[ir.ValueID]int{}
		if err := countBlockUses(f.Body, uses); err != nil {
			return err
		}
		changed := pruneBlock(f.Body, uses)
		if !changed {
			return nil
		}
	}
}

func countBlockUses(b *ir.Block, uses map[ir.ValueID]int) error {
	use := func(id ir.ValueID) {
		if id != 0 {
			uses[id]++
		}
	}
	for _, in := range b.Instrs {
		switch x := in.(type) {
		case *ir.Const, *ir.Builtin, *ir.PlaceRoot, *ir.PlaceWorkgroup, *ir.PlaceField, *ir.Load, *ir.Barrier, *ir.ArrayLength:
		case *ir.Unary:
			use(x.X)
		case *ir.Binary:
			use(x.Left)
			use(x.Right)
		case *ir.Intrinsic:
			for _, id := range x.Args {
				use(id)
			}
		case *ir.Convert:
			use(x.X)
		case *ir.Composite:
			for _, id := range x.Values {
				use(id)
			}
		case *ir.Extract:
			use(x.Base)
		case *ir.Call:
			for _, id := range x.Args {
				use(id)
			}
		case *ir.PlaceIndex:
			use(x.Index)
		case *ir.Store:
			use(x.Value)
		case *ir.Atomic:
			use(x.Value)
		case *ir.If:
			use(x.Cond)
			if err := countBlockUses(x.Then, uses); err != nil {
				return err
			}
			if err := countBlockUses(x.Else, uses); err != nil {
				return err
			}
		case *ir.Loop:
			for _, p := range x.Params {
				use(p.Init)
			}
			if err := countBlockUses(x.Cond, uses); err != nil {
				return err
			}
			if err := countBlockUses(x.Body, uses); err != nil {
				return err
			}
		default:
			return fmt.Errorf("unhandled IR instruction %T", in)
		}
	}
	switch t := b.Term.(type) {
	case *ir.Yield:
		for _, id := range t.Values {
			use(id)
		}
	case *ir.Continue:
		for _, id := range t.Values {
			use(id)
		}
	case *ir.Return:
		if t.HasValue {
			use(t.Value)
		}
	case *ir.Unreachable, nil:
	default:
		return fmt.Errorf("unhandled IR terminator %T", b.Term)
	}
	return nil
}

func pruneBlock(b *ir.Block, uses map[ir.ValueID]int) bool {
	changed := false
	out := b.Instrs[:0]
	for _, in := range b.Instrs {
		switch x := in.(type) {
		case *ir.If:
			if pruneBlock(x.Then, uses) {
				changed = true
			}
			if pruneBlock(x.Else, uses) {
				changed = true
			}
		case *ir.Loop:
			if pruneBlock(x.Cond, uses) {
				changed = true
			}
			if pruneBlock(x.Body, uses) {
				changed = true
			}
		}
		if d, ok := in.(ir.ValueDef); ok && isDeadRemovable(in) && uses[d.ResultValue()] == 0 {
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
	case *ir.Const, *ir.Builtin, *ir.Unary, *ir.Binary, *ir.Intrinsic, *ir.Convert, *ir.Composite, *ir.Extract:
		return true
	default:
		return false
	}
}
