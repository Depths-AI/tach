package ir

import (
	"fmt"
	"tach/foundation"
)

func (v *blockVerifier) verifyMemoryInstruction(in Instr) error {
	f := v.function
	defVal, defPlace, val, place := v.defineValue, v.definePlace, v.value, v.place
	switch x := in.(type) {
	case *PlaceRoot:
		if x.Buffer < 0 || x.Buffer >= len(f.BufferParams) {
			return fmt.Errorf("invalid buffer %d", x.Buffer)
		}
		r := f.BufferParams[x.Buffer]
		if !foundation.Equal(x.Type, r.Type) {
			return fmt.Errorf("buffer place type %s, want %s", x.Type, r.Type)
		}
		if err := defPlace(x.Result, placeInfo{x.Type, x.Buffer}); err != nil {
			return err
		}
	case *PlaceWorkgroup:
		if f.Kind != Stage || x.Workgroup < 0 || x.Workgroup >= len(f.WorkgroupVars) {
			return fmt.Errorf("invalid workgroup place %d", x.Workgroup)
		}
		w := f.WorkgroupVars[x.Workgroup]
		if !foundation.Equal(x.Type, w.Type) {
			return fmt.Errorf("workgroup place type %s, want %s", x.Type, w.Type)
		}
		if err := defPlace(x.Result, placeInfo{x.Type, -1}); err != nil {
			return err
		}
	case *PlaceField:
		bp, err := place(x.Base)
		if err != nil {
			return err
		}
		if bp.ty.Kind != foundation.StructKind || x.Field < 0 || x.Field >= len(bp.ty.Fields) {
			return fmt.Errorf("invalid field place on %s", bp.ty)
		}
		want := bp.ty.Fields[x.Field].Type
		if !foundation.Equal(want, x.Type) {
			return fmt.Errorf("field place type %s, want %s", x.Type, want)
		}
		if err := defPlace(x.Result, placeInfo{x.Type, bp.buffer}); err != nil {
			return err
		}
	case *PlaceIndex:
		bp, err := place(x.Base)
		if err != nil {
			return err
		}
		if bp.ty.Kind != foundation.RuntimeArrayKind && bp.ty.Kind != foundation.FixedArrayKind && bp.ty.Kind != foundation.VectorKind {
			return fmt.Errorf("index place base is %s", bp.ty)
		}
		it, err := val(x.Index)
		if err != nil {
			return err
		}
		if !foundation.Equal(it, foundation.Uint32Type) && !foundation.Equal(it, foundation.Int32Type) {
			return fmt.Errorf("array index is %s", it)
		}
		if !foundation.Equal(x.Type, bp.ty.Elem) {
			return fmt.Errorf("index result %s, want %s", x.Type, bp.ty.Elem)
		}
		if err := defPlace(x.Result, placeInfo{x.Type, bp.buffer}); err != nil {
			return err
		}
	case *Load:
		p, err := place(x.Place)
		if err != nil {
			return err
		}
		if !foundation.IsConstructible(p.ty) {
			return fmt.Errorf("place of type %s cannot be loaded as a value", p.ty)
		}
		if !foundation.Equal(p.ty, x.Type) {
			return fmt.Errorf("load type %s, place is %s", x.Type, p.ty)
		}
		if err := defVal(x.Result, x.Type); err != nil {
			return err
		}
	case *Store:
		p, err := place(x.Place)
		if err != nil {
			return err
		}
		v, err := val(x.Value)
		if err != nil {
			return err
		}
		if !foundation.Equal(p.ty, v) {
			return fmt.Errorf("store %s into %s", v, p.ty)
		}
		if !foundation.IsConstructible(p.ty) {
			return fmt.Errorf("place of type %s cannot be stored as a whole value", p.ty)
		}
		if p.buffer >= 0 {
			r := f.BufferParams[p.buffer]
			if r.Access != Mutable {
				return fmt.Errorf("store through non-writable buffer %s", r.Name)
			}
		}
	case *Atomic:
		p, err := place(x.Place)
		if err != nil {
			return err
		}
		if p.ty.Kind != foundation.AtomicKind || !foundation.Equal(p.ty.Elem, x.Type) || (x.Type.Kind != foundation.Int32Kind && x.Type.Kind != foundation.Uint32Kind) {
			return fmt.Errorf("atomic operation type %s does not match place %s", x.Type, p.ty)
		}
		if p.buffer >= 0 && x.Op != AtomicLoad {
			r := f.BufferParams[p.buffer]
			if r.Access != Mutable {
				return fmt.Errorf("atomic operation through non-writable buffer resource %s", r.Name)
			}
		}
		switch x.Op {
		case AtomicLoad:
			if x.Result == 0 || x.Value != 0 || x.Expected != 0 {
				return fmt.Errorf("atomicLoad result/value shape is invalid")
			}
			if err := defVal(x.Result, x.Type); err != nil {
				return err
			}
		case AtomicStore:
			if x.Result != 0 || x.Value == 0 || x.Expected != 0 {
				return fmt.Errorf("atomicStore result/value shape is invalid")
			}
			vt, err := val(x.Value)
			if err != nil || !foundation.Equal(vt, x.Type) {
				if err != nil {
					return err
				}
				return fmt.Errorf("atomicStore value is %s, want %s", vt, x.Type)
			}
		case AtomicAdd, AtomicSub, AtomicMin, AtomicMax, AtomicAnd, AtomicOr, AtomicXor, AtomicExchange:
			if x.Result == 0 || x.Value == 0 || x.Expected != 0 {
				return fmt.Errorf("atomic read-modify-write result/value shape is invalid")
			}
			vt, err := val(x.Value)
			if err != nil {
				return err
			}
			if !foundation.Equal(vt, x.Type) {
				return fmt.Errorf("atomic operand is %s, want %s", vt, x.Type)
			}
			if err := defVal(x.Result, x.Type); err != nil {
				return err
			}
		case AtomicCompareExchange:
			if x.Result == 0 || x.Value == 0 || x.Expected == 0 {
				return fmt.Errorf("atomicCompareExchange result/operand shape is invalid")
			}
			for _, operand := range []ValueID{x.Expected, x.Value} {
				operandType, err := val(operand)
				if err != nil {
					return err
				}
				if !foundation.Equal(operandType, x.Type) {
					return fmt.Errorf("atomic compare-exchange operand is %s, want %s", operandType, x.Type)
				}
			}
			if err := defVal(x.Result, x.Type); err != nil {
				return err
			}
		default:
			return fmt.Errorf("unknown atomic operation %d", x.Op)
		}
	case *Barrier:
		if f.Kind != Stage {
			return fmt.Errorf("barrier outside compute function")
		}
		if x.Kind != BarrierWorkgroup && x.Kind != BarrierBuffer {
			return fmt.Errorf("unknown barrier kind %d", x.Kind)
		}
	case *ArrayLength:
		p, err := place(x.Place)
		if err != nil {
			return err
		}
		if p.ty.Kind != foundation.RuntimeArrayKind {
			return fmt.Errorf("array_length on %s", p.ty)
		}
		if !foundation.Equal(x.Type, foundation.Uint32Type) {
			return fmt.Errorf("array_length result must be uint32")
		}
		if err := defVal(x.Result, x.Type); err != nil {
			return err
		}
	default:
		return fmt.Errorf("invalid memory instruction %T", in)
	}
	return nil
}
