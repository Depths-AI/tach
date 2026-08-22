package ir

import (
	"fmt"
	"tach/foundation"
)

func (v *blockVerifier) verifyValueInstruction(in Instr) error {
	fmap := v.functions
	defVal, val := v.defineValue, v.value
	switch x := in.(type) {
	case *Const:
		if !foundation.IsScalar(x.Type) {
			return fmt.Errorf("constant %%%d has non-scalar type %s", x.Result, x.Type)
		}
		if err := defVal(x.Result, x.Type); err != nil {
			return err
		}
	case *Unary:
		t, err := val(x.X)
		if err != nil {
			return err
		}
		if !foundation.Equal(t, x.Type) {
			return fmt.Errorf("unary %s type mismatch %s -> %s", x.Op, t, x.Type)
		}
		if x.Op == "!" && !foundation.IsBoolean(t) {
			return fmt.Errorf("! requires bool or boolean vector")
		}
		if x.Op == "-" && !foundation.IsSignedNumeric(t) {
			return fmt.Errorf("unary - requires signed/float numeric")
		}
		if x.Op == "~" && !foundation.IsIntegerLike(t) {
			return fmt.Errorf("unary ~ requires integer scalar/vector")
		}
		if x.Op != "!" && x.Op != "-" && x.Op != "~" {
			return fmt.Errorf("unknown unary op %q", x.Op)
		}
		if err := defVal(x.Result, x.Type); err != nil {
			return err
		}
	case *Binary:
		lt, err := val(x.Left)
		if err != nil {
			return err
		}
		rt, err := val(x.Right)
		if err != nil {
			return err
		}
		if err := verifyBinary(x, lt, rt); err != nil {
			return err
		}
		if err := defVal(x.Result, x.Type); err != nil {
			return err
		}
	case *Convert:
		t, err := val(x.X)
		if err != nil {
			return err
		}
		if !foundation.Equal(t, x.From) {
			return fmt.Errorf("convert source says %s but value is %s", x.From, t)
		}
		if !foundation.IsNumericScalar(t) || !foundation.IsNumericScalar(x.Type) {
			return fmt.Errorf("convert supports scalar numerics")
		}
		if err := defVal(x.Result, x.Type); err != nil {
			return err
		}
	case *Composite:
		if x.Type.Kind == foundation.StructKind {
			if len(x.Values) != len(x.Type.Fields) {
				return fmt.Errorf("struct %s construction has %d values", x.Type, len(x.Values))
			}
			for i, id := range x.Values {
				t, err := val(id)
				if err != nil {
					return err
				}
				if !foundation.Equal(t, x.Type.Fields[i].Type) {
					return fmt.Errorf("struct %s field %s has %s, want %s", x.Type, x.Type.Fields[i].Name, t, x.Type.Fields[i].Type)
				}
			}
		} else if x.Type.Kind == foundation.VectorKind {
			if len(x.Values) != x.Type.Lanes {
				return fmt.Errorf("vector construction has %d components, want %d", len(x.Values), x.Type.Lanes)
			}
			for _, id := range x.Values {
				t, err := val(id)
				if err != nil {
					return err
				}
				if !foundation.Equal(t, x.Type.Elem) {
					return fmt.Errorf("vector component has %s, want %s", t, x.Type.Elem)
				}
			}
		} else {
			return fmt.Errorf("composite result type %s is not constructible", x.Type)
		}
		if err := defVal(x.Result, x.Type); err != nil {
			return err
		}
	case *Extract:
		bt, err := val(x.Base)
		if err != nil {
			return err
		}
		var et *foundation.Type
		if bt.Kind == foundation.StructKind {
			if x.Index < 0 || x.Index >= len(bt.Fields) {
				return fmt.Errorf("struct extract index %d out of range", x.Index)
			}
			et = bt.Fields[x.Index].Type
		} else if bt.Kind == foundation.VectorKind {
			if x.Index < 0 || x.Index >= bt.Lanes {
				return fmt.Errorf("vector extract index %d out of range", x.Index)
			}
			et = bt.Elem
		} else {
			return fmt.Errorf("extract from %s", bt)
		}
		if !foundation.Equal(et, x.Type) {
			return fmt.Errorf("extract type %s, want %s", x.Type, et)
		}
		if err := defVal(x.Result, x.Type); err != nil {
			return err
		}
	case *VectorIndex:
		bt, err := val(x.Base)
		if err != nil {
			return err
		}
		it, err := val(x.Index)
		if err != nil {
			return err
		}
		if bt.Kind != foundation.VectorKind {
			return fmt.Errorf("vector index base is %s", bt)
		}
		if !foundation.IsInteger(it) {
			return fmt.Errorf("vector index is %s, want int32 or uint32", it)
		}
		if !foundation.Equal(x.Type, bt.Elem) {
			return fmt.Errorf("vector index type %s, want %s", x.Type, bt.Elem)
		}
		if err := defVal(x.Result, x.Type); err != nil {
			return err
		}
	case *Intrinsic:
		args := make([]*foundation.Type, len(x.Args))
		for i, id := range x.Args {
			t, err := val(id)
			if err != nil {
				return err
			}
			args[i] = t
		}
		if err := verifyIntrinsic(x, args); err != nil {
			return err
		}
		if err := defVal(x.Result, x.Type); err != nil {
			return err
		}
	case *Call:
		callee := fmap[x.Function]
		if callee == nil {
			return fmt.Errorf("call to unknown function %s", x.Function)
		}
		if callee.Kind == Stage {
			return fmt.Errorf("compute entry point %s cannot be called", x.Function)
		}
		if len(x.Args) != len(callee.Params) {
			return fmt.Errorf("call %s has %d args, want %d", x.Function, len(x.Args), len(callee.Params))
		}
		for i, id := range x.Args {
			t, err := val(id)
			if err != nil {
				return err
			}
			if !foundation.Equal(t, callee.Params[i].Type) {
				return fmt.Errorf("call %s arg %d is %s, want %s", x.Function, i, t, callee.Params[i].Type)
			}
		}
		if !foundation.Equal(x.Type, callee.Return) {
			return fmt.Errorf("call %s result says %s, want %s", x.Function, x.Type, callee.Return)
		}
		if x.Type.Kind != foundation.VoidKind {
			if err := defVal(x.Result, x.Type); err != nil {
				return err
			}
		}
	default:
		return fmt.Errorf("invalid value instruction %T", in)
	}
	return nil
}
