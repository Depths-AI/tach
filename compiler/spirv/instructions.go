package spirv

import (
	"fmt"
	"tach/foundation"
	"tach/ir"
)

func (s *fnEmitter) emitInstr(in ir.Instr) error {
	if definition, ok := in.(ir.ValueDef); ok {
		lowered := s.b.p.functions[s.f]
		if lowered.Replaced[definition.ResultValue()] {
			return nil
		}
	}
	switch x := in.(type) {
	case *ir.Const:
		if s.b.p.functions[s.f].Uses[x.Result] == 0 {
			return nil
		}
		id, err := s.b.constant(x.Type, x.Raw)
		if err != nil {
			return err
		}
		s.def(x.Result, id, x.Type)
	case *ir.Unary:
		return s.emitUnary(x)
	case *ir.Binary:
		return s.emitBinary(x)
	case *ir.Intrinsic:
		return s.emitIntrinsic(x)
	case *ir.Convert:
		return s.emitConvert(x)
	case *ir.Composite:
		tid, _ := s.b.typeID(x.Type, typeLogical)
		id := s.b.id()
		ops := []uint32{tid, id}
		for _, v := range x.Values {
			sv, err := s.value(v)
			if err != nil {
				return err
			}
			ops = append(ops, sv)
		}
		emit(&s.b.functions, OpCompositeConstruct, ops...)
		s.def(x.Result, id, x.Type)
	case *ir.Extract:
		base, err := s.value(x.Base)
		if err != nil {
			return err
		}
		tid, _ := s.b.typeID(x.Type, typeLogical)
		id := s.b.id()
		emit(&s.b.functions, OpCompositeExtract, tid, id, base, uint32(x.Index))
		s.def(x.Result, id, x.Type)
	case *ir.VectorIndex:
		base, err := s.value(x.Base)
		if err != nil {
			return err
		}
		index, err := s.value(x.Index)
		if err != nil {
			return err
		}
		tid, _ := s.b.typeID(x.Type, typeLogical)
		id := s.b.id()
		emit(&s.b.functions, OpVectorExtractDynamic, tid, id, base, index)
		s.def(x.Result, id, x.Type)
	case *ir.Call:
		fid := s.b.funcIDs[x.Function]
		if fid == 0 {
			return fmt.Errorf("unknown callee %s", x.Function)
		}
		tid, _ := s.b.typeID(x.Type, typeLogical)
		result := s.b.id() // OpFunctionCall always carries a Result <id>, including void calls.
		ops := []uint32{tid, result, fid}
		for _, a := range x.Args {
			v, err := s.value(a)
			if err != nil {
				return err
			}
			ops = append(ops, v)
		}
		emit(&s.b.functions, OpFunctionCall, ops...)
		calls := s.b.calls[s.f.Name]
		if calls == nil {
			calls = map[string]bool{}
			s.b.calls[s.f.Name] = calls
		}
		calls[x.Function] = true
		s.def(x.Result, result, x.Type)
	case *ir.PlaceRoot:
		return s.emitPlaceRoot(x)
	case *ir.PlaceWorkgroup:
		ids := s.b.workgroupIDs[s.f.Name]
		if x.Workgroup < 0 || x.Workgroup >= len(ids) {
			return fmt.Errorf("workgroup index %d out of bounds", x.Workgroup)
		}
		s.useGlobal(ids[x.Workgroup])
		s.places[x.Result] = spvPlace{ptr: ids[x.Workgroup], ty: x.Type, storage: StorageWorkgroup, resource: -1}
	case *ir.PlaceField:
		return s.emitPlaceField(x)
	case *ir.PlaceIndex:
		return s.emitPlaceIndex(x)
	case *ir.Load:
		p, err := s.place(x.Place)
		if err != nil {
			return err
		}
		id, err := s.loadPlace(p, x.Type)
		if err != nil {
			return err
		}
		s.def(x.Result, id, x.Type)
	case *ir.Store:
		p, err := s.place(x.Place)
		if err != nil {
			return err
		}
		v, err := s.value(x.Value)
		if err != nil {
			return err
		}
		if err := s.storePlace(p, v); err != nil {
			return err
		}
	case *ir.Atomic:
		return s.emitAtomic(x)
	case *ir.Barrier:
		return s.emitBarrier(x)
	case *ir.ArrayLength:
		p, err := s.place(x.Place)
		if err != nil {
			return err
		}
		if length, ok := s.b.p.kernels[s.f].LogicalLengths[p.resource]; ok {
			id, err := s.value(length)
			if err != nil {
				return err
			}
			s.def(x.Result, id, foundation.Uint32Type)
			return nil
		}
		if !p.hasArrayLen {
			return fmt.Errorf("runtime-array place lacks OpArrayLength base")
		}
		tid, _ := s.b.typeID(foundation.Uint32Type, typeLogical)
		id := s.b.id()
		emit(&s.b.functions, OpArrayLength, tid, id, p.arrayBase, p.arrayMember)
		s.def(x.Result, id, foundation.Uint32Type)
	case *ir.If:
		return s.emitIf(x)
	case *ir.Loop:
		return s.emitLoop(x)
	case *ir.Scope:
		return s.emitScope(x)
	default:
		return fmt.Errorf("unsupported IR instruction %T", in)
	}
	return nil
}
