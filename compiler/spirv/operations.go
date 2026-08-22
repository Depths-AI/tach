package spirv

import (
	"fmt"
	"tach/foundation"
	"tach/ir"
)

func (s *fnEmitter) emitIntrinsic(x *ir.Intrinsic) error {
	tid, err := s.b.typeID(x.Type, typeLogical)
	if err != nil {
		return err
	}
	args := make([]uint32, len(x.Args))
	for i, a := range x.Args {
		v, err := s.value(a)
		if err != nil {
			return err
		}
		args[i] = v
	}
	if x.Kind == ir.IntrinsicAll || x.Kind == ir.IntrinsicAny {
		op := OpAll
		if x.Kind == ir.IntrinsicAny {
			op = OpAny
		}
		id := s.b.id()
		emit(&s.b.functions, op, tid, id, args[0])
		s.def(x.Result, id, x.Type)
		return nil
	}
	if x.Kind == ir.IntrinsicSelect {
		id := s.b.id()
		emit(&s.b.functions, OpSelect, tid, id, args[0], args[1], args[2])
		s.def(x.Result, id, x.Type)
		return nil
	}
	if x.Kind == ir.IntrinsicDot {
		id := s.b.id()
		emit(&s.b.functions, OpDot, tid, id, args[0], args[1])
		s.def(x.Result, id, x.Type)
		return nil
	}
	if x.Kind == ir.IntrinsicMin || x.Kind == ir.IntrinsicMax || x.Kind == ir.IntrinsicClamp {
		conditionType := foundation.BoolType
		if x.Type.Kind == foundation.VectorKind {
			conditionType = foundation.VectorOf(foundation.BoolType, x.Type.Lanes)
		}
		conditionTypeID, err := s.b.typeID(conditionType, typeLogical)
		if err != nil {
			return err
		}
		kind := scalarKind(x.Type)
		less := OpFOrdLessThan
		if kind == foundation.Int32Kind {
			less = OpSLessThan
		} else if kind == foundation.Uint32Kind {
			less = OpULessThan
		}
		bound := func(minimum bool, left, right uint32) uint32 {
			condition, result := s.b.id(), s.b.id()
			first, second := left, right
			if minimum {
				first, second = right, left
			}
			emit(&s.b.functions, less, conditionTypeID, condition, first, second)
			emit(&s.b.functions, OpSelect, tid, result, condition, right, left)
			return result
		}
		result := bound(x.Kind == ir.IntrinsicMin, args[0], args[1])
		if x.Kind == ir.IntrinsicClamp {
			result = bound(true, result, args[2])
		}
		s.def(x.Result, result, x.Type)
		return nil
	}

	id := s.b.id()
	var inst uint32
	switch x.Kind {
	case ir.IntrinsicAbs:
		if foundation.IsFloatLike(x.Type) {
			inst = GLSL450FAbs
		} else {
			inst = GLSL450SAbs
		}
	case ir.IntrinsicFloor:
		inst = GLSL450Floor
	case ir.IntrinsicCeil:
		inst = GLSL450Ceil
	case ir.IntrinsicTrunc:
		inst = GLSL450Trunc
	case ir.IntrinsicSin:
		inst = GLSL450Sin
	case ir.IntrinsicCos:
		inst = GLSL450Cos
	case ir.IntrinsicTan:
		inst = GLSL450Tan
	case ir.IntrinsicExp:
		inst = GLSL450Exp
	case ir.IntrinsicExp2:
		inst = GLSL450Exp2
	case ir.IntrinsicLog:
		inst = GLSL450Log
	case ir.IntrinsicLog2:
		inst = GLSL450Log2
	case ir.IntrinsicSqrt:
		inst = GLSL450Sqrt
	case ir.IntrinsicRSqrt:
		inst = GLSL450InverseSqrt
	case ir.IntrinsicPow:
		inst = GLSL450Pow
	case ir.IntrinsicFma:
		inst = GLSL450Fma
	case ir.IntrinsicLength:
		inst = GLSL450Length
	case ir.IntrinsicDistance:
		inst = GLSL450Distance
	case ir.IntrinsicCross:
		inst = GLSL450Cross
	case ir.IntrinsicNormalize:
		inst = GLSL450Normalize
	default:
		return fmt.Errorf("unsupported intrinsic %s", x.Kind)
	}
	ops := []uint32{tid, id, s.b.ensureGLSL450(), inst}
	ops = append(ops, args...)
	emit(&s.b.functions, OpExtInst, ops...)
	s.def(x.Result, id, x.Type)
	return nil
}

func (s *fnEmitter) emitAtomic(x *ir.Atomic) error {
	p, err := s.place(x.Place)
	if err != nil {
		return err
	}
	if p.storage != StorageWorkgroup && p.storage != StorageStorageBuffer {
		return fmt.Errorf("atomic place uses invalid storage class %d", p.storage)
	}
	scopeValue := ScopeQueueFamily // QueueFamily avoids vulkanMemoryModelDeviceScope.
	if p.storage == StorageWorkgroup {
		scopeValue = ScopeWorkgroup
	}
	scope, err := s.b.u32Constant(scopeValue)
	if err != nil {
		return err
	}
	semantics, err := s.b.u32Constant(MemorySemanticsRelaxed)
	if err != nil {
		return err
	}
	tid, err := s.b.typeID(x.Type, typeLogical)
	if err != nil {
		return err
	}

	switch x.Op {
	case ir.AtomicLoad:
		id := s.b.id()
		emit(&s.b.functions, OpAtomicLoad, tid, id, p.ptr, scope, semantics)
		s.def(x.Result, id, x.Type)
		return nil
	case ir.AtomicStore:
		value, err := s.value(x.Value)
		if err != nil {
			return err
		}
		emit(&s.b.functions, OpAtomicStore, p.ptr, scope, semantics, value)
		return nil
	}
	if x.Op == ir.AtomicCompareExchange {
		expected, err := s.value(x.Expected)
		if err != nil {
			return err
		}
		value, err := s.value(x.Value)
		if err != nil {
			return err
		}
		id := s.b.id()
		emit(&s.b.functions, OpAtomicCompareExchange, tid, id, p.ptr, scope, semantics, semantics, value, expected)
		s.def(x.Result, id, x.Type)
		return nil
	}

	value, err := s.value(x.Value)
	if err != nil {
		return err
	}
	var op Op
	switch x.Op {
	case ir.AtomicExchange:
		op = OpAtomicExchange
	case ir.AtomicAdd:
		op = OpAtomicIAdd
	case ir.AtomicSub:
		op = OpAtomicISub
	case ir.AtomicMin:
		if x.Type.Kind == foundation.Int32Kind {
			op = OpAtomicSMin
		} else {
			op = OpAtomicUMin
		}
	case ir.AtomicMax:
		if x.Type.Kind == foundation.Int32Kind {
			op = OpAtomicSMax
		} else {
			op = OpAtomicUMax
		}
	case ir.AtomicAnd:
		op = OpAtomicAnd
	case ir.AtomicOr:
		op = OpAtomicOr
	case ir.AtomicXor:
		op = OpAtomicXor
	default:
		return fmt.Errorf("unsupported atomic operation %d", x.Op)
	}
	id := s.b.id()
	emit(&s.b.functions, op, tid, id, p.ptr, scope, semantics, value)
	s.def(x.Result, id, x.Type)
	return nil
}

func (s *fnEmitter) emitBarrier(x *ir.Barrier) error {
	execScope, err := s.b.u32Constant(ScopeWorkgroup)
	if err != nil {
		return err
	}
	memoryScope := execScope
	sem := MemorySemanticsAcquireRelease | MemorySemanticsMakeAvailable | MemorySemanticsMakeVisible
	switch x.Kind {
	case ir.BarrierWorkgroup:
		sem |= MemorySemanticsWorkgroupMemory
	case ir.BarrierBuffer:
		sem |= MemorySemanticsUniformMemory
	default:
		return fmt.Errorf("unsupported barrier kind %d", x.Kind)
	}
	semantics, err := s.b.u32Constant(sem)
	if err != nil {
		return err
	}
	emit(&s.b.functions, OpControlBarrier, execScope, memoryScope, semantics)
	return nil
}

func (s *fnEmitter) emitPlaceRoot(x *ir.PlaceRoot) error {
	resources := s.b.resourceIDs[s.f]
	if x.Buffer < 0 || x.Buffer >= len(resources) {
		return fmt.Errorf("buffer index %d out of bounds", x.Buffer)
	}
	s.useGlobal(resources[x.Buffer])
	storage := uint32(StorageStorageBuffer)
	if x.Type.Kind == foundation.StructKind && foundation.ContainsRuntimeArray(x.Type) {
		s.places[x.Result] = spvPlace{ptr: resources[x.Buffer], ty: x.Type, storage: storage, resource: x.Buffer}
		return nil
	}
	ptrType, err := s.b.pointerID(storage, x.Type)
	if err != nil {
		return err
	}
	zero, err := s.b.u32Constant(0)
	if err != nil {
		return err
	}
	id := s.b.id()
	emit(&s.b.functions, OpAccessChain, ptrType, id, resources[x.Buffer], zero)
	p := spvPlace{ptr: id, ty: x.Type, storage: storage, resource: x.Buffer}
	if x.Type.Kind == foundation.RuntimeArrayKind {
		p.arrayBase = resources[x.Buffer]
		p.arrayMember = 0
		p.hasArrayLen = true
	}
	s.places[x.Result] = p
	return nil
}

func (s *fnEmitter) emitPlaceField(x *ir.PlaceField) error {
	base, err := s.place(x.Base)
	if err != nil {
		return err
	}
	p, err := s.accessField(base, x.Field, x.Type)
	if err != nil {
		return err
	}
	s.places[x.Result] = p
	return nil
}

func (s *fnEmitter) emitPlaceIndex(x *ir.PlaceIndex) error {
	base, err := s.place(x.Base)
	if err != nil {
		return err
	}
	idx, err := s.value(x.Index)
	if err != nil {
		return err
	}
	ptrType, err := s.b.pointerID(base.storage, x.Type)
	if err != nil {
		return err
	}
	id := s.b.id()
	emit(&s.b.functions, OpAccessChain, ptrType, id, base.ptr, idx)
	s.places[x.Result] = spvPlace{ptr: id, ty: x.Type, storage: base.storage, resource: base.resource}
	return nil
}
