package spirv

import (
	"fmt"
)

func (v *validation) validateReferencesAndTypes() error {
	for _, in := range v.m.Instructions {
		a := in.Operands
		ctx := fmt.Sprintf("word %d %s", in.Offset, opName(in.Op))
		switch in.Op {
		case OpName:
			if err := v.requireID(a[0], ctx); err != nil {
				return err
			}
		case OpMemberName:
			t, err := v.requireType(a[0], ctx)
			if err != nil {
				return err
			}
			if t.kind != typeStruct || int(a[1]) >= len(t.members) {
				return fmt.Errorf("%s names invalid struct member %d", ctx, a[1])
			}
		case OpEntryPoint:
			if err := v.requireID(a[1], ctx); err != nil {
				return err
			}
			_, next, _ := literalString(a, 2)
			seen := map[uint32]bool{}
			for _, id := range a[next:] {
				if err := v.requireID(id, ctx+" interface"); err != nil {
					return err
				}
				if _, global := v.globalVars[id]; !global {
					return fmt.Errorf("%s interface id %%%d is not a global variable", ctx, id)
				}
				if seen[id] {
					return fmt.Errorf("%s contains duplicate interface id %%%d", ctx, id)
				}
				seen[id] = true
			}
		case OpExecutionMode:
			if err := v.requireID(a[0], ctx); err != nil {
				return err
			}
		case OpDecorate:
			if err := v.requireID(a[0], ctx); err != nil {
				return err
			}
		case OpMemberDecorate:
			t, err := v.requireType(a[0], ctx)
			if err != nil {
				return err
			}
			if t.kind != typeStruct || int(a[1]) >= len(t.members) {
				return fmt.Errorf("%s decorates invalid member %d", ctx, a[1])
			}
		case OpTypeVector:
			t, err := v.requireType(a[1], ctx)
			if err != nil {
				return err
			}
			if t.kind != typeBool && t.kind != typeInt && t.kind != typeFloat {
				return fmt.Errorf("%s vector element must be bool, integer, or float", ctx)
			}
		case OpTypeArray:
			if _, err := v.requireType(a[1], ctx); err != nil {
				return err
			}
			lt, err := v.requireValue(a[2], ctx)
			if err != nil {
				return err
			}
			li := v.types[lt]
			if li == nil || li.kind != typeInt || li.signed || li.width != 32 {
				return fmt.Errorf("%s array length must be a uint32 constant", ctx)
			}
			if _, ok := v.constants[a[2]]; !ok || v.constants[a[2]] == 0 {
				return fmt.Errorf("%s array length must be a positive constant", ctx)
			}
		case OpTypeRuntimeArray:
			if _, err := v.requireType(a[1], ctx); err != nil {
				return err
			}
		case OpTypeStruct:
			for _, id := range a[1:] {
				if _, err := v.requireType(id, ctx); err != nil {
					return err
				}
			}
		case OpTypePointer:
			if _, err := v.requireType(a[2], ctx); err != nil {
				return err
			}
		case OpTypeFunction:
			for _, id := range a[1:] {
				if _, err := v.requireType(id, ctx); err != nil {
					return err
				}
			}
		case OpConstantTrue, OpConstantFalse:
			t, err := v.requireType(a[0], ctx)
			if err != nil {
				return err
			}
			if t.kind != typeBool {
				return fmt.Errorf("%s requires bool result type", ctx)
			}
		case OpConstant:
			t, err := v.requireType(a[0], ctx)
			if err != nil {
				return err
			}
			if t.kind != typeInt && t.kind != typeFloat {
				return fmt.Errorf("%s scalar constant requires int/float type", ctx)
			}
			if t.width == 16 && a[2] > 0xffff {
				return fmt.Errorf("%s float16 constant has non-zero high bits", ctx)
			}
		case OpConstantComposite:
			t, err := v.requireType(a[0], ctx)
			if err != nil {
				return err
			}
			if t.kind != typeVector && t.kind != typeStruct {
				return fmt.Errorf("%s composite constant has invalid type", ctx)
			}
			for _, id := range a[2:] {
				if _, err := v.requireValue(id, ctx); err != nil {
					return err
				}
			}
		case OpConstantNull:
			t, err := v.requireType(a[0], ctx)
			if err != nil {
				return err
			}
			switch t.kind {
			case typeBool, typeInt, typeFloat, typeVector, typeArray, typeStruct:
			default:
				return fmt.Errorf("%s result type cannot have a null value", ctx)
			}
		case OpVariable:
			pt, err := v.requireType(a[0], ctx)
			if err != nil {
				return err
			}
			if pt.kind != typePointer || pt.storage != a[2] {
				return fmt.Errorf("%s variable result type/storage mismatch", ctx)
			}
			if a[2] == StorageWorkgroup {
				if len(a) != 4 {
					return fmt.Errorf("%s Workgroup variable requires a null initializer", ctx)
				}
				initializerType, err := v.requireValue(a[3], ctx+" initializer")
				if err != nil {
					return err
				}
				if initializerType != pt.elem || v.m.Instructions[v.defs[a[3]]].Op != OpConstantNull {
					return fmt.Errorf("%s Workgroup initializer must be OpConstantNull of the pointee type", ctx)
				}
			} else if len(a) != 3 {
				return fmt.Errorf("%s storage class %d cannot have an initializer in Tach's profile", ctx, a[2])
			}
		case OpFunction:
			if _, err := v.requireType(a[0], ctx); err != nil {
				return err
			}
			if a[2] != FunctionControlNone && a[2] != FunctionControlInline|FunctionControlConst {
				return fmt.Errorf("%s function control outside Tach profile", ctx)
			}
			ft, err := v.requireType(a[3], ctx)
			if err != nil {
				return err
			}
			if ft.kind != typeFunction || ft.ret != a[0] {
				return fmt.Errorf("%s function type return mismatch", ctx)
			}
		case OpFunctionParameter:
			if _, err := v.requireType(a[0], ctx); err != nil {
				return err
			}
		case OpFunctionCall:
			rt, err := v.requireType(a[0], ctx)
			if err != nil {
				return err
			}
			_ = rt
			callee := v.functions[a[2]]
			if callee == nil {
				return fmt.Errorf("%s callee %%%d is not a function", ctx, a[2])
			}
			if callee.ret != a[0] {
				return fmt.Errorf("%s call result type does not match callee", ctx)
			}
			ft := v.types[callee.fnType]
			if ft == nil || ft.kind != typeFunction {
				return fmt.Errorf("%s callee function type missing", ctx)
			}
			if len(a[3:]) != len(ft.params) {
				return fmt.Errorf("%s call has %d args, want %d", ctx, len(a[3:]), len(ft.params))
			}
			for i, id := range a[3:] {
				vt, err := v.requireValue(id, ctx)
				if err != nil {
					return err
				}
				if vt != ft.params[i] {
					return fmt.Errorf("%s arg %d type mismatch", ctx, i)
				}
			}
			v.valueType[a[1]] = a[0]
		case OpExtInst:
			if err := v.validateExtInst(in); err != nil {
				return err
			}
		case OpDot:
			if err := v.validateDot(in); err != nil {
				return err
			}
		case OpLoad:
			rt, err := v.requireType(a[0], ctx)
			if err != nil {
				return err
			}
			_ = rt
			ptid, err := v.requireValue(a[2], ctx)
			if err != nil {
				return err
			}
			pt := v.types[ptid]
			if pt == nil || pt.kind != typePointer || pt.elem != a[0] {
				return fmt.Errorf("%s load pointer/result mismatch", ctx)
			}
			if err := v.validateMemoryAccess(a[3:], pt.storage, pt.elem); err != nil {
				return fmt.Errorf("%s: %w", ctx, err)
			}
			v.valueType[a[1]] = a[0]
		case OpStore:
			ptid, err := v.requireValue(a[0], ctx)
			if err != nil {
				return err
			}
			pt := v.types[ptid]
			if pt == nil || pt.kind != typePointer {
				return fmt.Errorf("%s store target is not pointer", ctx)
			}
			vt, err := v.requireValue(a[1], ctx)
			if err != nil {
				return err
			}
			if pt.elem != vt {
				return fmt.Errorf("%s store value type mismatch", ctx)
			}
			if err := v.validateMemoryAccess(a[2:], pt.storage, pt.elem); err != nil {
				return fmt.Errorf("%s: %w", ctx, err)
			}
			if pt.storage == StorageUniform {
				return fmt.Errorf("%s attempts to store through Uniform pointer", ctx)
			}
			root := v.pointerRoot[a[0]]
			if root != 0 && v.decoration(root).nonWritable {
				return fmt.Errorf("%s stores through NonWritable resource %%%d", ctx, root)
			}
		case OpAtomicLoad, OpAtomicStore, OpAtomicExchange, OpAtomicCompareExchange, OpAtomicIAdd, OpAtomicISub,
			OpAtomicSMin, OpAtomicUMin, OpAtomicSMax, OpAtomicUMax, OpAtomicAnd, OpAtomicOr, OpAtomicXor:
			if err := v.validateAtomic(in); err != nil {
				return err
			}
		case OpControlBarrier:
			if err := v.validateBarrier(in); err != nil {
				return err
			}
		case OpAccessChain:
			if err := v.validateAccessChain(in); err != nil {
				return err
			}
		case OpArrayLength:
			if err := v.validateArrayLength(in); err != nil {
				return err
			}
		case OpCompositeConstruct:
			if err := v.validateCompositeConstruct(in); err != nil {
				return err
			}
		case OpVectorExtractDynamic:
			if err := v.validateVectorExtractDynamic(in); err != nil {
				return err
			}
		case OpCompositeExtract:
			if err := v.validateCompositeExtract(in); err != nil {
				return err
			}
		case OpConvertFToU, OpConvertFToS, OpConvertSToF, OpConvertUToF, OpFConvert, OpBitcast:
			if err := v.validateConversion(in); err != nil {
				return err
			}
		case OpSNegate, OpFNegate, OpLogicalNot, OpNot:
			if err := v.validateUnary(in); err != nil {
				return err
			}
		case OpAny, OpAll:
			if err := v.validateMaskReduction(in); err != nil {
				return err
			}
		case OpIAdd, OpFAdd, OpISub, OpFSub, OpIMul, OpFMul, OpUDiv, OpSDiv, OpFDiv, OpUMod, OpSRem, OpFRem,
			OpVectorTimesScalar, OpLogicalEqual, OpLogicalNotEqual, OpLogicalOr, OpLogicalAnd, OpIEqual, OpINotEqual,
			OpUGreaterThan, OpSGreaterThan, OpUGreaterThanEqual, OpSGreaterThanEqual, OpULessThan, OpSLessThan,
			OpULessThanEqual, OpSLessThanEqual, OpFOrdEqual, OpFOrdNotEqual, OpFOrdLessThan, OpFOrdGreaterThan,
			OpFOrdLessThanEqual, OpFOrdGreaterThanEqual, OpShiftRightLogical, OpShiftRightArithmetic,
			OpShiftLeftLogical, OpBitwiseOr, OpBitwiseXor, OpBitwiseAnd:
			if err := v.validateBinary(in); err != nil {
				return err
			}
		case OpSelect:
			if err := v.validateSelect(in); err != nil {
				return err
			}
		case OpPhi:
			if _, err := v.requireType(a[0], ctx); err != nil {
				return err
			}
			v.valueType[a[1]] = a[0]
			for i := 2; i < len(a); i += 2 {
				vt, err := v.requireValue(a[i], ctx)
				if err != nil {
					return err
				}
				if vt != a[0] {
					return fmt.Errorf("%s phi incoming type mismatch", ctx)
				}
				if err := v.requireID(a[i+1], ctx); err != nil {
					return err
				}
			}
		case OpLoopMerge:
			if err := v.requireID(a[0], ctx); err != nil {
				return err
			}
			if err := v.requireID(a[1], ctx); err != nil {
				return err
			}
			if a[2] != LoopControlNone {
				return fmt.Errorf("%s loop control outside Tach profile", ctx)
			}
		case OpSelectionMerge:
			if err := v.requireID(a[0], ctx); err != nil {
				return err
			}
			if a[1] != SelectionControlNone {
				return fmt.Errorf("%s selection control outside Tach profile", ctx)
			}
		case OpBranch:
			if err := v.requireID(a[0], ctx); err != nil {
				return err
			}
		case OpBranchConditional:
			ct, err := v.requireValue(a[0], ctx)
			if err != nil {
				return err
			}
			if v.types[ct] == nil || v.types[ct].kind != typeBool {
				return fmt.Errorf("%s condition is not bool", ctx)
			}
			if err := v.requireID(a[1], ctx); err != nil {
				return err
			}
			if err := v.requireID(a[2], ctx); err != nil {
				return err
			}
		case OpReturnValue:
			if _, err := v.requireValue(a[0], ctx); err != nil {
				return err
			}
		}
	}
	return nil
}

func (v *validation) validateAccessChain(in Instruction) error {
	a := in.Operands
	ctx := fmt.Sprintf("word %d OpAccessChain", in.Offset)
	rp, err := v.requireType(a[0], ctx)
	if err != nil {
		return err
	}
	if rp.kind != typePointer {
		return fmt.Errorf("%s result type is not pointer", ctx)
	}
	btid, err := v.requireValue(a[2], ctx)
	if err != nil {
		return err
	}
	bp := v.types[btid]
	if bp == nil || bp.kind != typePointer {
		return fmt.Errorf("%s base is not pointer", ctx)
	}
	if bp.storage != rp.storage {
		return fmt.Errorf("%s changes storage class", ctx)
	}
	cur := bp.elem
	for _, idxID := range a[3:] {
		it, err := v.requireValue(idxID, ctx)
		if err != nil {
			return err
		}
		itp := v.types[it]
		if itp == nil || itp.kind != typeInt {
			return fmt.Errorf("%s index is not integer", ctx)
		}
		t := v.types[cur]
		if t == nil {
			return fmt.Errorf("%s indexes non-type %%%d", ctx, cur)
		}
		switch t.kind {
		case typeStruct:
			c, ok := v.constants[idxID]
			if !ok {
				return fmt.Errorf("%s struct index must be constant", ctx)
			}
			idx := uint32(c)
			if int(idx) >= len(t.members) {
				return fmt.Errorf("%s struct index %d out of bounds", ctx, idx)
			}
			cur = t.members[idx]
		case typeArray, typeRuntimeArray:
			cur = t.elem
		case typeVector:
			cur = t.elem
		default:
			return fmt.Errorf("%s cannot index type %%%d", ctx, cur)
		}
	}
	if cur != rp.elem {
		return fmt.Errorf("%s result pointer pointee mismatch", ctx)
	}
	v.valueType[a[1]] = a[0]
	root := v.pointerRoot[a[2]]
	if root == 0 {
		root = a[2]
	}
	v.pointerRoot[a[1]] = root
	return nil
}

func (v *validation) validateArrayLength(in Instruction) error {
	a := in.Operands
	ctx := fmt.Sprintf("word %d OpArrayLength", in.Offset)
	rt, err := v.requireType(a[0], ctx)
	if err != nil {
		return err
	}
	if rt.kind != typeInt || rt.signed || rt.width != 32 {
		return fmt.Errorf("%s result must be uint32", ctx)
	}
	ptid, err := v.requireValue(a[2], ctx)
	if err != nil {
		return err
	}
	pt := v.types[ptid]
	if pt == nil || pt.kind != typePointer {
		return fmt.Errorf("%s structure operand is not pointer", ctx)
	}
	st := v.types[pt.elem]
	if st == nil || st.kind != typeStruct {
		return fmt.Errorf("%s pointer does not point to struct", ctx)
	}
	member := a[3]
	if int(member) >= len(st.members) {
		return fmt.Errorf("%s member %d out of range", ctx, member)
	}
	at := v.types[st.members[member]]
	if at == nil || at.kind != typeRuntimeArray {
		return fmt.Errorf("%s member %d is not runtime array", ctx, member)
	}
	v.valueType[a[1]] = a[0]
	return nil
}

func (v *validation) constantU32(id uint32, ctx string) (uint32, error) {
	tid, err := v.requireValue(id, ctx)
	if err != nil {
		return 0, err
	}
	t := v.types[tid]
	if t == nil || t.kind != typeInt || t.signed || t.width != 32 {
		return 0, fmt.Errorf("%s requires uint32 constant operand %%%d", ctx, id)
	}
	x, ok := v.constants[id]
	if !ok {
		return 0, fmt.Errorf("%s requires constant operand %%%d", ctx, id)
	}
	return uint32(x), nil
}
