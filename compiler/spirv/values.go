package spirv

import (
	"fmt"
	"tach/foundation"
	"tach/ir"
)

func (s *fnEmitter) emitUnary(x *ir.Unary) error {
	v, err := s.value(x.X)
	if err != nil {
		return err
	}
	tid, _ := s.b.typeID(x.Type, typeLogical)
	id := s.b.id()
	var op Op
	switch x.Op {
	case "!":
		op = OpLogicalNot
	case "-":
		if foundation.IsFloatLike(x.Type) {
			op = OpFNegate
		} else {
			op = OpSNegate
		}
	case "~":
		op = OpNot
	default:
		return fmt.Errorf("unsupported unary operator %s", x.Op)
	}
	emit(&s.b.functions, op, tid, id, v)
	s.def(x.Result, id, x.Type)
	return nil
}

func (s *fnEmitter) splatVector(vector *foundation.Type, scalar uint32) (uint32, error) {
	tid, err := s.b.typeID(vector, typeLogical)
	if err != nil {
		return 0, err
	}
	id := s.b.id()
	ops := []uint32{tid, id}
	for i := 0; i < vector.Lanes; i++ {
		ops = append(ops, scalar)
	}
	emit(&s.b.functions, OpCompositeConstruct, ops...)
	return id, nil
}

func (s *fnEmitter) emitBinary(x *ir.Binary) error {
	lv, err := s.value(x.Left)
	if err != nil {
		return err
	}
	rv, err := s.value(x.Right)
	if err != nil {
		return err
	}
	lt := s.vtypes[x.Left]
	rt := s.vtypes[x.Right]
	tid, _ := s.b.typeID(x.Type, typeLogical)
	id := s.b.id()
	kind := scalarKind(lt)
	var op Op

	isVectorScalar := lt != nil && lt.Kind == foundation.VectorKind && rt != nil && foundation.Equal(lt.Elem, rt)
	isScalarVector := rt != nil && rt.Kind == foundation.VectorKind && lt != nil && foundation.Equal(rt.Elem, lt)
	if x.Op == "*" && (isVectorScalar || isScalarVector) {
		if isScalarVector {
			lv, rv = rv, lv
		}
		emit(&s.b.functions, OpVectorTimesScalar, tid, id, lv, rv)
		s.def(x.Result, id, x.Type)
		return nil
	}
	if x.Op == "/" && isVectorScalar {
		rv, err = s.splatVector(lt, rv)
		if err != nil {
			return err
		}
	}

	switch x.Op {
	case "+":
		if kind == foundation.Float16Kind || kind == foundation.Float32Kind {
			op = OpFAdd
		} else {
			op = OpIAdd
		}
	case "-":
		if kind == foundation.Float16Kind || kind == foundation.Float32Kind {
			op = OpFSub
		} else {
			op = OpISub
		}
	case "*":
		if kind == foundation.Float16Kind || kind == foundation.Float32Kind {
			op = OpFMul
		} else {
			op = OpIMul
		}
	case "/":
		switch kind {
		case foundation.Float16Kind, foundation.Float32Kind:
			op = OpFDiv
		case foundation.Uint32Kind:
			op = OpUDiv
		case foundation.Int32Kind:
			op = OpSDiv
		}
	case "%":
		switch kind {
		case foundation.Float16Kind, foundation.Float32Kind:
			op = OpFRem
		case foundation.Uint32Kind:
			op = OpUMod
		case foundation.Int32Kind:
			op = OpSRem
		}
	case "&&":
		op = OpLogicalAnd
	case "||":
		op = OpLogicalOr
	case "&":
		if kind == foundation.BoolKind {
			op = OpLogicalAnd
		} else {
			op = OpBitwiseAnd
		}
	case "|":
		if kind == foundation.BoolKind {
			op = OpLogicalOr
		} else {
			op = OpBitwiseOr
		}
	case "^":
		if kind == foundation.BoolKind {
			op = OpLogicalNotEqual
		} else {
			op = OpBitwiseXor
		}
	case "<<":
		op = OpShiftLeftLogical
	case ">>":
		if kind == foundation.Int32Kind {
			op = OpShiftRightArithmetic
		} else {
			op = OpShiftRightLogical
		}
	case "==":
		switch kind {
		case foundation.Float16Kind, foundation.Float32Kind:
			op = OpFOrdEqual
		case foundation.Int32Kind, foundation.Uint32Kind:
			op = OpIEqual
		case foundation.BoolKind:
			op = OpLogicalEqual
		}
	case "!=":
		switch kind {
		case foundation.Float16Kind, foundation.Float32Kind:
			op = OpFOrdNotEqual
		case foundation.Int32Kind, foundation.Uint32Kind:
			op = OpINotEqual
		case foundation.BoolKind:
			op = OpLogicalNotEqual
		}
	case "<":
		switch kind {
		case foundation.Float16Kind, foundation.Float32Kind:
			op = OpFOrdLessThan
		case foundation.Uint32Kind:
			op = OpULessThan
		case foundation.Int32Kind:
			op = OpSLessThan
		}
	case "<=":
		switch kind {
		case foundation.Float16Kind, foundation.Float32Kind:
			op = OpFOrdLessThanEqual
		case foundation.Uint32Kind:
			op = OpULessThanEqual
		case foundation.Int32Kind:
			op = OpSLessThanEqual
		}
	case ">":
		switch kind {
		case foundation.Float16Kind, foundation.Float32Kind:
			op = OpFOrdGreaterThan
		case foundation.Uint32Kind:
			op = OpUGreaterThan
		case foundation.Int32Kind:
			op = OpSGreaterThan
		}
	case ">=":
		switch kind {
		case foundation.Float16Kind, foundation.Float32Kind:
			op = OpFOrdGreaterThanEqual
		case foundation.Uint32Kind:
			op = OpUGreaterThanEqual
		case foundation.Int32Kind:
			op = OpSGreaterThanEqual
		}
	}
	if op == 0 {
		return fmt.Errorf("cannot lower binary %s for %s and %s", x.Op, lt, rt)
	}
	emit(&s.b.functions, op, tid, id, lv, rv)
	s.def(x.Result, id, x.Type)
	return nil
}

func (s *fnEmitter) emitConvert(x *ir.Convert) error {
	v, err := s.value(x.X)
	if err != nil {
		return err
	}
	tid, _ := s.b.typeID(x.Type, typeLogical)
	id := s.b.id()
	var op Op
	switch {
	case foundation.IsFloatLike(x.From) && x.Type.Kind == foundation.Uint32Kind:
		op = OpConvertFToU
	case foundation.IsFloatLike(x.From) && x.Type.Kind == foundation.Int32Kind:
		op = OpConvertFToS
	case x.From.Kind == foundation.Int32Kind && foundation.IsFloatLike(x.Type):
		op = OpConvertSToF
	case x.From.Kind == foundation.Uint32Kind && foundation.IsFloatLike(x.Type):
		op = OpConvertUToF
	case foundation.IsFloatLike(x.From) && foundation.IsFloatLike(x.Type):
		op = OpFConvert
	case (x.From.Kind == foundation.Int32Kind && x.Type.Kind == foundation.Uint32Kind) || (x.From.Kind == foundation.Uint32Kind && x.Type.Kind == foundation.Int32Kind):
		op = OpBitcast
	default:
		return fmt.Errorf("unsupported conversion %s -> %s", x.From, x.Type)
	}
	emit(&s.b.functions, op, tid, id, v)
	s.def(x.Result, id, x.Type)
	return nil
}

func (s *fnEmitter) emitIf(x *ir.If) error {
	cond, err := s.value(x.Cond)
	if err != nil {
		return err
	}
	thenLabel, elseLabel, mergeLabel := s.b.id(), s.b.id(), s.b.id()
	emit(&s.b.functions, OpSelectionMerge, mergeLabel, SelectionControlNone)
	emit(&s.b.functions, OpBranchConditional, cond, thenLabel, elseLabel)
	s.terminated = true

	incomingByResult := make([][]phiIncoming, len(x.Results))

	emit(&s.b.functions, OpLabel, thenLabel)
	s.currentLabel, s.terminated = thenLabel, false
	te, err := s.emitBlockExit(x.Then, blockYield)
	if err != nil {
		return err
	}
	if te.falls {
		if len(te.vals) != len(x.Results) {
			return fmt.Errorf("then yield count mismatch")
		}
		for i, v := range te.vals {
			incomingByResult[i] = append(incomingByResult[i], phiIncoming{v, te.pred})
		}
		emit(&s.b.functions, OpBranch, mergeLabel)
		s.terminated = true
	}

	emit(&s.b.functions, OpLabel, elseLabel)
	s.currentLabel, s.terminated = elseLabel, false
	ee, err := s.emitBlockExit(x.Else, blockYield)
	if err != nil {
		return err
	}
	if ee.falls {
		if len(ee.vals) != len(x.Results) {
			return fmt.Errorf("else yield count mismatch")
		}
		for i, v := range ee.vals {
			incomingByResult[i] = append(incomingByResult[i], phiIncoming{v, ee.pred})
		}
		emit(&s.b.functions, OpBranch, mergeLabel)
		s.terminated = true
	}

	emit(&s.b.functions, OpLabel, mergeLabel)
	s.currentLabel, s.terminated = mergeLabel, false
	for i, r := range x.Results {
		incs := incomingByResult[i]
		if len(incs) == 0 {
			return fmt.Errorf("selection result %%%d has no incoming value", r.ID)
		}
		if len(incs) == 1 {
			s.def(r.ID, incs[0].value, r.Type)
			continue
		}
		tid, _ := s.b.typeID(r.Type, typeLogical)
		id := s.b.id()
		ops := []uint32{tid, id}
		for _, in := range incs {
			ops = append(ops, in.value, in.label)
		}
		emit(&s.b.functions, OpPhi, ops...)
		s.def(r.ID, id, r.Type)
	}
	return nil
}

func (s *fnEmitter) emitLoop(x *ir.Loop) error {
	if len(x.Params) != len(x.Results) {
		return fmt.Errorf("loop param/result mismatch")
	}
	preheader := s.currentLabel
	header, condEntry, body, cont, merge := s.b.id(), s.b.id(), s.b.id(), s.b.id(), s.b.id()
	emit(&s.b.functions, OpBranch, header)
	s.terminated = true
	emit(&s.b.functions, OpLabel, header)
	s.currentLabel, s.terminated = header, false

	patches := make([]int, len(x.Params))
	params := make([]uint32, len(x.Params))
	for i, p := range x.Params {
		init, err := s.value(p.Init)
		if err != nil {
			return err
		}
		tid, _ := s.b.typeID(p.Type, typeLogical)
		phi := s.b.id()
		start := emit(&s.b.functions, OpPhi, tid, phi, init, preheader, 0, cont)
		patches[i] = start + 5 // first word + operands: type,result,init,pre,back,cont
		params[i] = phi
		s.def(p.ID, phi, p.Type)
	}

	// Keep the SPIR-V loop header minimal and structurally stable. Tach's loop
	// condition is a region and can itself contain structured short-circuit
	// selections, so it lives in the loop construct after the header.
	emit(&s.b.functions, OpLoopMerge, merge, cont, LoopControlNone)
	emit(&s.b.functions, OpBranch, condEntry)
	s.terminated = true

	emit(&s.b.functions, OpLabel, condEntry)
	s.currentLabel, s.terminated = condEntry, false
	ce, err := s.emitBlockExit(x.Cond, blockYield)
	if err != nil {
		return err
	}
	if !ce.falls || len(ce.vals) != 1 {
		return fmt.Errorf("loop condition must yield one bool")
	}
	conditionExit := ce.pred
	emit(&s.b.functions, OpBranchConditional, ce.vals[0], body, merge)
	s.terminated = true

	emit(&s.b.functions, OpLabel, body)
	s.currentLabel, s.terminated = body, false
	s.loops = append(s.loops, loopTarget{
		merge:      merge,
		continuing: cont,
		breaks:     make([][]phiIncoming, len(x.Params)),
		continues:  make([][]phiIncoming, len(x.Params)),
	})
	_, err = s.emitBlockExit(x.Body, blockNormal)
	if err != nil {
		return err
	}
	loop := s.loops[len(s.loops)-1]
	s.loops = s.loops[:len(s.loops)-1]

	emit(&s.b.functions, OpLabel, cont)
	s.currentLabel, s.terminated = cont, false
	for i, incoming := range loop.continues {
		back := uint32(0)
		switch len(incoming) {
		case 0:
			back, err = s.value(x.Params[i].Init)
		case 1:
			back = incoming[0].value
		default:
			tid, _ := s.b.typeID(x.Params[i].Type, typeLogical)
			back = s.b.id()
			ops := []uint32{tid, back}
			for _, value := range incoming {
				ops = append(ops, value.value, value.label)
			}
			emit(&s.b.functions, OpPhi, ops...)
		}
		if err != nil {
			return err
		}
		s.b.functions[patches[i]] = back
	}
	emit(&s.b.functions, OpBranch, header)
	s.terminated = true

	emit(&s.b.functions, OpLabel, merge)
	s.currentLabel, s.terminated = merge, false
	for i, result := range x.Results {
		incoming := append([]phiIncoming{{params[i], conditionExit}}, loop.breaks[i]...)
		if len(incoming) == 1 {
			s.def(result.ID, incoming[0].value, result.Type)
			continue
		}
		tid, _ := s.b.typeID(result.Type, typeLogical)
		id := s.b.id()
		ops := []uint32{tid, id}
		for _, value := range incoming {
			ops = append(ops, value.value, value.label)
		}
		emit(&s.b.functions, OpPhi, ops...)
		s.def(result.ID, id, result.Type)
	}
	return nil
}
