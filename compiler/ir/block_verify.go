package ir

import (
	"fmt"
	"tach/foundation"
)

type blockVerifier struct {
	module    *KernelModule
	function  *Function
	env       verifyEnv
	functions map[string]*Function
	termKind  string
	loopTypes []*foundation.Type
}

func verifyBlock(module *KernelModule, function *Function, block *Block, environment verifyEnv, functions map[string]*Function, termKind string, loopTypes []*foundation.Type) (verifyEnv, error) {
	if block == nil {
		return environment, fmt.Errorf("nil block")
	}
	verifier := &blockVerifier{module: module, function: function, env: environment, functions: functions, termKind: termKind, loopTypes: loopTypes}
	for _, instruction := range block.Instrs {
		if err := verifier.verifyInstruction(instruction); err != nil {
			return environment, err
		}
	}
	if err := verifier.verifyTerminator(block.Term); err != nil {
		return environment, err
	}
	return verifier.env, nil
}

func (v *blockVerifier) defineValue(id ValueID, t *foundation.Type) error {
	if id == 0 {
		return fmt.Errorf("value id 0 is reserved")
	}
	if _, exists := v.env.values[id]; exists {
		return fmt.Errorf("value %%%d redefined", id)
	}
	v.env.values[id] = t
	return nil
}

func (v *blockVerifier) definePlace(id PlaceID, place placeInfo) error {
	if id == 0 {
		return fmt.Errorf("place id 0 is reserved")
	}
	if _, exists := v.env.places[id]; exists {
		return fmt.Errorf("place &p%d redefined", id)
	}
	v.env.places[id] = place
	return nil
}

func (v *blockVerifier) value(id ValueID) (*foundation.Type, error) {
	t, exists := v.env.values[id]
	if !exists {
		return nil, fmt.Errorf("use of undefined value %%%d", id)
	}
	return t, nil
}

func (v *blockVerifier) place(id PlaceID) (placeInfo, error) {
	place, exists := v.env.places[id]
	if !exists {
		return placeInfo{}, fmt.Errorf("use of undefined place &p%d", id)
	}
	return place, nil
}

func (v *blockVerifier) verifyInstruction(instruction Instr) error {
	switch instruction.(type) {
	case *Const, *Unary, *Binary, *Convert, *Composite, *Extract, *VectorIndex, *Intrinsic, *Call:
		return v.verifyValueInstruction(instruction)
	case *PlaceRoot, *PlaceWorkgroup, *PlaceField, *PlaceIndex, *Load, *Store, *Atomic, *Barrier, *ArrayLength:
		return v.verifyMemoryInstruction(instruction)
	case *If, *Loop, *Scope:
		return v.verifyControlInstruction(instruction)
	default:
		return fmt.Errorf("unknown instruction %T", instruction)
	}
}

func (v *blockVerifier) verifyTerminator(term Term) error {
	if term == nil {
		return fmt.Errorf("block has no terminator")
	}
	f, termKind, loopTypes := v.function, v.termKind, v.loopTypes
	val := v.value
	if term == nil {
		return fmt.Errorf("block has no terminator")
	}
	verifyTransfer := func(kind string, values []ValueID) error {
		if loopTypes == nil {
			return fmt.Errorf("unexpected %s terminator", kind)
		}
		if len(values) != len(loopTypes) {
			return fmt.Errorf("%s carries %d values, want %d", kind, len(values), len(loopTypes))
		}
		for i, id := range values {
			type_, err := val(id)
			if err != nil {
				return err
			}
			if !foundation.Equal(type_, loopTypes[i]) {
				return fmt.Errorf("%s value %d is %s, want %s", kind, i, type_, loopTypes[i])
			}
		}
		return nil
	}
	switch t := term.(type) {
	case *Return:
		if f.Return.Kind == foundation.VoidKind {
			if t.HasValue {
				return fmt.Errorf("void function returns a value")
			}
		} else {
			if !t.HasValue {
				return fmt.Errorf("function returning %s has bare return", f.Return)
			}
			vt, err := val(t.Value)
			if err != nil {
				return err
			}
			if !foundation.Equal(vt, f.Return) {
				return fmt.Errorf("return value is %s, want %s", vt, f.Return)
			}
		}
	case *Yield:
		if termKind != "yield" {
			return fmt.Errorf("unexpected yield terminator")
		}
		for _, id := range t.Values {
			if _, err := val(id); err != nil {
				return err
			}
		}
	case *Continue:
		if err := verifyTransfer("continue", t.Values); err != nil {
			return err
		}
	case *Break:
		if err := verifyTransfer("break", t.Values); err != nil {
			return err
		}
	case *Unreachable:
		// Structured constructs whose every path exits can leave an unreachable merge.
	case *ExitScope:
		if termKind != "exit_scope" && termKind != "yield" && termKind != "continue" {
			return fmt.Errorf("exit_scope outside scope")
		}
	default:
		return fmt.Errorf("unknown terminator %T", term)
	}
	return nil
}
