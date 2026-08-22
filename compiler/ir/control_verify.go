package ir

import (
	"fmt"
	"tach/foundation"
)

func (v *blockVerifier) verifyControlInstruction(in Instr) error {
	m, f, e, fmap, loopTypes := v.module, v.function, v.env, v.functions, v.loopTypes
	defVal, val := v.defineValue, v.value
	switch x := in.(type) {
	case *If:
		ct, err := val(x.Cond)
		if err != nil {
			return err
		}
		if !foundation.Equal(ct, foundation.BoolType) {
			return fmt.Errorf("if condition is %s", ct)
		}
		te, err := verifyBlock(m, f, x.Then, e.clone(), fmap, "yield", loopTypes)
		if err != nil {
			return fmt.Errorf("if then: %w", err)
		}
		ee, err := verifyBlock(m, f, x.Else, e.clone(), fmap, "yield", loopTypes)
		if err != nil {
			return fmt.Errorf("if else: %w", err)
		}
		ty, ok1 := x.Then.Term.(*Yield)
		ey, ok2 := x.Else.Term.(*Yield)
		if len(x.Results) > 0 && !ok1 && !ok2 {
			return fmt.Errorf("value-producing if has no continuing branch")
		}
		if ok1 && len(ty.Values) != len(x.Results) {
			return fmt.Errorf("if then yield arity mismatch")
		}
		if ok2 && len(ey.Values) != len(x.Results) {
			return fmt.Errorf("if else yield arity mismatch")
		}
		for i, r := range x.Results {
			if ok1 {
				a := te.values[ty.Values[i]]
				if !foundation.Equal(a, r.Type) {
					return fmt.Errorf("if then result %d type mismatch", i)
				}
			}
			if ok2 {
				bb := ee.values[ey.Values[i]]
				if !foundation.Equal(bb, r.Type) {
					return fmt.Errorf("if else result %d type mismatch", i)
				}
			}
			if err := defVal(r.ID, r.Type); err != nil {
				return err
			}
		}
	case *Loop:
		le := e.clone()
		for _, p := range x.Params {
			it, err := val(p.Init)
			if err != nil {
				return err
			}
			if !foundation.Equal(it, p.Type) {
				return fmt.Errorf("loop init %s, want %s", it, p.Type)
			}
			if _, exists := le.values[p.ID]; exists {
				return fmt.Errorf("loop param %%%d redefined", p.ID)
			}
			le.values[p.ID] = p.Type
		}
		ce, err := verifyBlock(m, f, x.Cond, le.clone(), fmap, "yield", nil)
		if err != nil {
			return fmt.Errorf("loop condition: %w", err)
		}
		cy, ok := x.Cond.Term.(*Yield)
		if !ok || len(cy.Values) != 1 || !foundation.Equal(ce.values[cy.Values[0]], foundation.BoolType) {
			return fmt.Errorf("loop condition must yield one bool")
		}
		carriedTypes := make([]*foundation.Type, len(x.Params))
		for i, parameter := range x.Params {
			carriedTypes[i] = parameter.Type
		}
		_, err = verifyBlock(m, f, x.Body, le.clone(), fmap, "continue", carriedTypes)
		if err != nil {
			return fmt.Errorf("loop body: %w", err)
		}
		if len(x.Results) != len(x.Params) {
			return fmt.Errorf("loop carried arity mismatch")
		}
		for i, p := range x.Params {
			if !foundation.Equal(x.Results[i].Type, p.Type) {
				return fmt.Errorf("loop carried value %d type mismatch", i)
			}
			if err := defVal(x.Results[i].ID, p.Type); err != nil {
				return err
			}
		}
	case *Scope:
		if f.Kind != Stage {
			return fmt.Errorf("scope outside stage")
		}
		if _, err := verifyBlock(m, f, x.Body, e.clone(), fmap, "exit_scope", loopTypes); err != nil {
			return fmt.Errorf("scope: %w", err)
		}
	default:
		return fmt.Errorf("invalid control instruction %T", in)
	}
	return nil
}
