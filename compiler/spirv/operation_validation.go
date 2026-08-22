package spirv

import (
	"fmt"
)

func baseScalar(t *typeInfo, all map[uint32]*typeInfo) *typeInfo {
	if t != nil && t.kind == typeVector {
		return all[t.elem]
	}
	return t
}
func sameShape(a, b *typeInfo) bool {
	if a == nil || b == nil {
		return false
	}
	if a.kind == typeVector || b.kind == typeVector {
		return a.kind == b.kind && a.lanes == b.lanes
	}
	return true
}

func (v *validation) validateExtInst(in Instruction) error {
	a := in.Operands
	ctx := fmt.Sprintf("word %d OpExtInst", in.Offset)
	rt, err := v.requireType(a[0], ctx)
	if err != nil {
		return err
	}
	if v.extImports[a[2]] != "GLSL.std.450" {
		if err := v.requireID(a[2], ctx+" instruction set"); err != nil {
			return err
		}
		return fmt.Errorf("%s instruction set %%%d is not GLSL.std.450", ctx, a[2])
	}
	argIDs := a[4:]
	argTypes := make([]uint32, len(argIDs))
	for i, id := range argIDs {
		tid, err := v.requireValue(id, ctx)
		if err != nil {
			return err
		}
		argTypes[i] = tid
	}
	need := func(n int) error {
		if len(argIDs) != n {
			return fmt.Errorf("%s GLSL.std.450 instruction %d has %d args, want %d", ctx, a[3], len(argIDs), n)
		}
		return nil
	}
	allResultType := func() bool {
		for _, tid := range argTypes {
			if tid != a[0] {
				return false
			}
		}
		return true
	}
	base := baseScalar(rt, v.types)
	if base == nil {
		return fmt.Errorf("%s has invalid result type", ctx)
	}

	switch a[3] {
	case GLSL450FAbs:
		if err := need(1); err != nil {
			return err
		}
		if !allResultType() || base.kind != typeFloat {
			return fmt.Errorf("%s FAbs requires matching floating scalar/vector operand and result", ctx)
		}
	case GLSL450SAbs:
		if err := need(1); err != nil {
			return err
		}
		if !allResultType() || base.kind != typeInt || !base.signed {
			return fmt.Errorf("%s SAbs requires matching signed int32 scalar/vector operand and result", ctx)
		}
	case GLSL450Trunc, GLSL450Floor, GLSL450Ceil, GLSL450Sin, GLSL450Cos, GLSL450Tan,
		GLSL450Exp, GLSL450Log, GLSL450Exp2, GLSL450Log2, GLSL450Sqrt, GLSL450InverseSqrt:
		if err := need(1); err != nil {
			return err
		}
		if !allResultType() || base.kind != typeFloat {
			return fmt.Errorf("%s floating unary intrinsic requires matching floating scalar/vector operand and result", ctx)
		}
	case GLSL450Pow:
		if err := need(2); err != nil {
			return err
		}
		if !allResultType() || base.kind != typeFloat {
			return fmt.Errorf("%s Pow requires matching floating scalar/vector operands and result", ctx)
		}
	case GLSL450Fma:
		if err := need(3); err != nil {
			return err
		}
		if !allResultType() || base.kind != typeFloat {
			return fmt.Errorf("%s Fma requires matching floating scalar/vector operands and result", ctx)
		}
	case GLSL450Length:
		if err := need(1); err != nil {
			return err
		}
		at := v.types[argTypes[0]]
		if rt.kind != typeFloat || at == nil || at.kind != typeVector || at.elem != a[0] || baseScalar(at, v.types).kind != typeFloat {
			return fmt.Errorf("%s Length requires a floating vector input and matching component result", ctx)
		}
	case GLSL450Distance:
		if err := need(2); err != nil {
			return err
		}
		at := v.types[argTypes[0]]
		if argTypes[0] != argTypes[1] || rt.kind != typeFloat || at == nil || at.kind != typeVector || at.elem != a[0] || baseScalar(at, v.types).kind != typeFloat {
			return fmt.Errorf("%s Distance requires matching floating vectors and component result", ctx)
		}
	case GLSL450Cross:
		if err := need(2); err != nil {
			return err
		}
		if !allResultType() || rt.kind != typeVector || rt.lanes != 3 || base.kind != typeFloat {
			return fmt.Errorf("%s Cross requires matching three-lane floating operands and result", ctx)
		}
	case GLSL450Normalize:
		if err := need(1); err != nil {
			return err
		}
		if !allResultType() || rt.kind != typeVector || base.kind != typeFloat {
			return fmt.Errorf("%s Normalize requires matching floating vector operand and result", ctx)
		}
	default:
		return fmt.Errorf("%s GLSL.std.450 instruction %d is outside Tach's profile", ctx, a[3])
	}
	v.valueType[a[1]] = a[0]
	return nil
}

func (v *validation) validateDot(in Instruction) error {
	a := in.Operands
	ctx := fmt.Sprintf("word %d OpDot", in.Offset)
	rt, err := v.requireType(a[0], ctx)
	if err != nil {
		return err
	}
	ltID, err := v.requireValue(a[2], ctx)
	if err != nil {
		return err
	}
	rtID, err := v.requireValue(a[3], ctx)
	if err != nil {
		return err
	}
	lt := v.types[ltID]
	if rt.kind != typeFloat || ltID != rtID || lt == nil || lt.kind != typeVector || lt.elem != a[0] || baseScalar(lt, v.types).kind != typeFloat {
		return fmt.Errorf("%s requires matching floating vectors and returns their component type", ctx)
	}
	v.valueType[a[1]] = a[0]
	return nil
}

func (v *validation) validateConversion(in Instruction) error {
	a := in.Operands
	ctx := fmt.Sprintf("word %d %s", in.Offset, opName(in.Op))
	dst, err := v.requireType(a[0], ctx)
	if err != nil {
		return err
	}
	srcID, err := v.requireValue(a[2], ctx)
	if err != nil {
		return err
	}
	src := v.types[srcID]
	ds := baseScalar(dst, v.types)
	ss := baseScalar(src, v.types)
	if !sameShape(dst, src) {
		return fmt.Errorf("%s shape mismatch", ctx)
	}
	ok := false
	switch in.Op {
	case OpConvertFToU:
		ok = ss.kind == typeFloat && ds.kind == typeInt && !ds.signed
	case OpConvertFToS:
		ok = ss.kind == typeFloat && ds.kind == typeInt && ds.signed
	case OpConvertSToF:
		ok = ss.kind == typeInt && ss.signed && ds.kind == typeFloat
	case OpConvertUToF:
		ok = ss.kind == typeInt && !ss.signed && ds.kind == typeFloat
	case OpFConvert:
		ok = ss.kind == typeFloat && ds.kind == typeFloat && ss.width != ds.width
	case OpBitcast:
		ok = ss.kind == typeInt && ds.kind == typeInt && ss.width == ds.width
	}
	if !ok {
		return fmt.Errorf("%s source/result types are incompatible", ctx)
	}
	v.valueType[a[1]] = a[0]
	return nil
}

func (v *validation) validateUnary(in Instruction) error {
	a := in.Operands
	ctx := fmt.Sprintf("word %d %s", in.Offset, opName(in.Op))
	rt, err := v.requireType(a[0], ctx)
	if err != nil {
		return err
	}
	xt, err := v.requireValue(a[2], ctx)
	if err != nil {
		return err
	}
	if xt != a[0] {
		return fmt.Errorf("%s operand/result type mismatch", ctx)
	}
	base := baseScalar(rt, v.types)
	ok := false
	switch in.Op {
	case OpSNegate:
		ok = base.kind == typeInt && base.signed
	case OpFNegate:
		ok = base.kind == typeFloat
	case OpLogicalNot:
		ok = base.kind == typeBool
	case OpNot:
		ok = base.kind == typeInt
	}
	if !ok {
		return fmt.Errorf("%s invalid operand type", ctx)
	}
	v.valueType[a[1]] = a[0]
	return nil
}

func (v *validation) validateMaskReduction(in Instruction) error {
	a := in.Operands
	ctx := fmt.Sprintf("word %d %s", in.Offset, opName(in.Op))
	result, err := v.requireType(a[0], ctx)
	if err != nil {
		return err
	}
	operandType, err := v.requireValue(a[2], ctx)
	if err != nil {
		return err
	}
	operand := v.types[operandType]
	if result.kind != typeBool || operand == nil || operand.kind != typeVector || baseScalar(operand, v.types).kind != typeBool {
		return fmt.Errorf("%s requires a boolean vector and returns bool", ctx)
	}
	v.valueType[a[1]] = a[0]
	return nil
}

func (v *validation) validateBinary(in Instruction) error {
	a := in.Operands
	ctx := fmt.Sprintf("word %d %s", in.Offset, opName(in.Op))
	rt, err := v.requireType(a[0], ctx)
	if err != nil {
		return err
	}
	lt, err := v.requireValue(a[2], ctx)
	if err != nil {
		return err
	}
	rr, err := v.requireValue(a[3], ctx)
	if err != nil {
		return err
	}
	l := v.types[lt]
	r := v.types[rr]
	if in.Op == OpVectorTimesScalar {
		if rt.kind != typeVector || lt != a[0] || r == nil || r.kind != typeInt && r.kind != typeFloat || rt.elem != rr {
			return fmt.Errorf("%s requires vector and matching scalar", ctx)
		}
		v.valueType[a[1]] = a[0]
		return nil
	}
	if in.Op == OpBitwiseAnd || in.Op == OpBitwiseOr || in.Op == OpBitwiseXor {
		base := baseScalar(rt, v.types)
		if lt != a[0] || rr != a[0] || base == nil || base.kind != typeInt {
			return fmt.Errorf("%s requires matching integer scalar/vector operands and result", ctx)
		}
		v.valueType[a[1]] = a[0]
		return nil
	}
	if in.Op == OpShiftRightLogical || in.Op == OpShiftRightArithmetic || in.Op == OpShiftLeftLogical {
		leftBase := baseScalar(rt, v.types)
		rightBase := baseScalar(r, v.types)
		if lt != a[0] || leftBase == nil || leftBase.kind != typeInt || rightBase == nil || rightBase.kind != typeInt || rightBase.signed || !sameShape(rt, r) {
			return fmt.Errorf("%s requires integer value shifted by shape-matching unsigned count", ctx)
		}
		if in.Op == OpShiftRightArithmetic && !leftBase.signed {
			return fmt.Errorf("%s requires signed shifted value in Tach's profile", ctx)
		}
		if in.Op == OpShiftRightLogical && leftBase.signed {
			return fmt.Errorf("%s requires unsigned shifted value in Tach's profile", ctx)
		}
		v.valueType[a[1]] = a[0]
		return nil
	}
	comparison := in.Op >= OpLogicalEqual && in.Op <= OpFOrdGreaterThanEqual
	if comparison {
		if baseScalar(rt, v.types).kind != typeBool || !sameShape(rt, l) {
			return fmt.Errorf("%s comparison result must be bool", ctx)
		}
		if lt != rr {
			return fmt.Errorf("%s comparison operand types differ", ctx)
		}
		base := baseScalar(l, v.types)
		ok := false
		switch in.Op {
		case OpLogicalEqual, OpLogicalNotEqual, OpLogicalOr, OpLogicalAnd:
			ok = base.kind == typeBool
		case OpIEqual, OpINotEqual:
			ok = base.kind == typeInt
		case OpUGreaterThan, OpUGreaterThanEqual, OpULessThan, OpULessThanEqual:
			ok = base.kind == typeInt && !base.signed
		case OpSGreaterThan, OpSGreaterThanEqual, OpSLessThan, OpSLessThanEqual:
			ok = base.kind == typeInt && base.signed
		default:
			ok = base.kind == typeFloat
		}
		if !ok {
			return fmt.Errorf("%s comparison opcode/type mismatch", ctx)
		}
	} else {
		if lt != a[0] || rr != a[0] {
			return fmt.Errorf("%s arithmetic operands/result must share a type", ctx)
		}
		base := baseScalar(rt, v.types)
		ok := false
		switch in.Op {
		case OpIAdd, OpISub, OpIMul:
			ok = base.kind == typeInt
		case OpFAdd, OpFSub, OpFMul, OpFDiv, OpFRem:
			ok = base.kind == typeFloat
		case OpUDiv, OpUMod:
			ok = base.kind == typeInt && !base.signed
		case OpSDiv, OpSRem:
			ok = base.kind == typeInt && base.signed
		}
		if !ok {
			return fmt.Errorf("%s arithmetic opcode/type mismatch", ctx)
		}
	}
	v.valueType[a[1]] = a[0]
	return nil
}

func (v *validation) validateSelect(in Instruction) error {
	a := in.Operands
	ctx := fmt.Sprintf("word %d OpSelect", in.Offset)
	result, err := v.requireType(a[0], ctx)
	if err != nil {
		return err
	}
	conditionType, err := v.requireValue(a[2], ctx)
	if err != nil {
		return err
	}
	condition := v.types[conditionType]
	if baseScalar(condition, v.types).kind != typeBool || !sameShape(result, condition) {
		return fmt.Errorf("%s condition must be a shape-matching bool", ctx)
	}
	for _, operand := range a[3:] {
		operandType, err := v.requireValue(operand, ctx)
		if err != nil {
			return err
		}
		if operandType != a[0] {
			return fmt.Errorf("%s operands must match the result type", ctx)
		}
	}
	v.valueType[a[1]] = a[0]
	return nil
}
