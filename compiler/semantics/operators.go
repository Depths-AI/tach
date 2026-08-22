package semantics

import (
	"tach/foundation"
	"tach/ir"
	"tach/parser"
)

func (c *analyzer) lowerShortCircuit(b *fnBuilder, e env, x *parser.BinaryExpr) (ir.ValueID, *foundation.Type, error) {
	left, lt, err := c.lowerExpr(b, e, x.Left, foundation.BoolType)
	if err != nil {
		return 0, nil, err
	}
	if !foundation.Equal(lt, foundation.BoolType) {
		return 0, nil, boolDiag(x.Left.GetSpan(), "logical operand", lt, "use &, |, or ^ for lane-wise mask logic, or all/any to reduce a mask")
	}
	then := &ir.Block{}
	tb := b.child(then)
	els := &ir.Block{}
	eb := b.child(els)
	if x.Op == "&&" {
		rv, rt, err := c.lowerExpr(tb, e.clone(), x.Right, foundation.BoolType)
		if err != nil {
			return 0, nil, err
		}
		if !foundation.Equal(rt, foundation.BoolType) {
			return 0, nil, boolDiag(x.Right.GetSpan(), "logical operand", rt, "use &, |, or ^ for lane-wise mask logic, or all/any to reduce a mask")
		}
		then.Term = &ir.Yield{Values: []ir.ValueID{rv}}
		id, _, err := c.lowerExpr(eb, e.clone(), &parser.BoolExpr{Value: false, Span: x.Span}, foundation.BoolType)
		if err != nil {
			return 0, nil, err
		}
		els.Term = &ir.Yield{Values: []ir.ValueID{id}}
	} else {
		id, _, err := c.lowerExpr(tb, e.clone(), &parser.BoolExpr{Value: true, Span: x.Span}, foundation.BoolType)
		if err != nil {
			return 0, nil, err
		}
		then.Term = &ir.Yield{Values: []ir.ValueID{id}}
		rv, rt, err := c.lowerExpr(eb, e.clone(), x.Right, foundation.BoolType)
		if err != nil {
			return 0, nil, err
		}
		if !foundation.Equal(rt, foundation.BoolType) {
			return 0, nil, boolDiag(x.Right.GetSpan(), "logical operand", rt, "use &, |, or ^ for lane-wise mask logic, or all/any to reduce a mask")
		}
		els.Term = &ir.Yield{Values: []ir.ValueID{rv}}
	}
	r := b.value()
	b.emit(&ir.If{Results: []ir.Result{{ID: r, Type: foundation.BoolType}}, Cond: left, Then: then, Else: els, Span: x.Span})
	return r, foundation.BoolType, nil
}
func (c *analyzer) lowerBinaryExpr(b *fnBuilder, e env, x *parser.BinaryExpr, expected *foundation.Type) (ir.ValueID, *foundation.Type, error) {
	// Tach shifts use an unsigned count with the shifted value's vector width.
	// Resolve that directly instead of relying on ordinary binary contextual typing.
	if x.Op == "<<" || x.Op == ">>" {
		l, lt, err := c.lowerExpr(b, e, x.Left, expected)
		if err != nil {
			return 0, nil, err
		}
		return c.lowerShift(b, e, x.Op, l, lt, x.Right, x.Span)
	}
	expressions := []parser.Expr{x.Left, x.Right}
	arguments := make([]detachedExpr, 2)
	if x.Op == "==" || x.Op == "!=" || x.Op == "&" || x.Op == "|" || x.Op == "^" {
		for i, expression := range expressions {
			argument, err := c.lowerDetached(b, e, expression, nil)
			if err != nil {
				return 0, nil, err
			}
			arguments[i] = argument
		}
		if foundation.IsBoolean(arguments[0].type_) || foundation.IsBoolean(arguments[1].type_) {
			return c.emitBooleanBinary(b, arguments[0], arguments[1], x.Op, x.Span)
		}
	} else {
		var err error
		arguments, err = c.lowerNumericArguments(b, e, x.Op, expressions)
		if err != nil {
			return 0, nil, err
		}
	}

	return c.emitResolvedBinary(b, e, x.Op, arguments, expected, x.Span)
}

func (c *analyzer) emitResolvedBinary(b *fnBuilder, e env, op string, arguments []detachedExpr, expected *foundation.Type, span foundation.Span) (ir.ValueID, *foundation.Type, error) {
	var element *foundation.Type
	if expectedElement, _ := numericElement(expected); expectedElement != nil && op != "==" && op != "!=" && op != "<" && op != "<=" && op != ">" && op != ">=" {
		element = expectedElement
	}
	element, lanes, err := resolveNumericOperands(op, ir.NumericAny, arguments, element, 0)
	if err != nil {
		return 0, nil, diag(span, "%v", err)
	}
	vector := foundation.VectorOf(element, lanes)
	values := make([]ir.ValueID, 2)
	valueTypes := make([]*foundation.Type, 2)
	for i, argument := range arguments {
		_, argumentLanes := numericElement(argument.type_)
		want := element
		if argumentLanes > 0 {
			want = vector
		}
		values[i], valueTypes[i], err = c.commitArgument(b, e, argument, want)
		if err != nil {
			return 0, nil, err
		}
	}
	return c.emitBinary(b, op, values[0], valueTypes[0], values[1], valueTypes[1], span)
}

func vectorScalarOperator(op string) bool {
	switch op {
	case "+", "-", "*", "/", "%", "&", "|", "^", "==", "!=", "<", "<=", ">", ">=":
		return true
	}
	return false
}

func (c *analyzer) emitBooleanBinary(b *fnBuilder, left, right detachedExpr, op string, span foundation.Span) (ir.ValueID, *foundation.Type, error) {
	if op != "==" && op != "!=" && op != "&" && op != "|" && op != "^" {
		return 0, nil, diag(span, "%s is not defined for boolean values", op)
	}
	lt, rt := left.type_, right.type_
	if !foundation.IsBoolean(lt) || !foundation.IsBoolean(rt) {
		return 0, nil, diag(span, "%s requires both operands to be bool or vec<bool, N>; got %s and %s", op, lt, rt)
	}
	lanes := 0
	if lt.Kind == foundation.VectorKind {
		lanes = lt.Lanes
	}
	if rt.Kind == foundation.VectorKind {
		if lanes != 0 && lanes != rt.Lanes {
			return 0, nil, diag(span, "%s operands use conflicting vector widths %d and %d", op, lanes, rt.Lanes)
		}
		lanes = rt.Lanes
	}
	b.block.Instrs = append(b.block.Instrs, left.block.Instrs...)
	b.block.Instrs = append(b.block.Instrs, right.block.Instrs...)
	if lanes > 0 {
		vector := foundation.VectorOf(foundation.BoolType, lanes)
		if lt.Kind == foundation.BoolKind {
			left.value, lt = c.splat(b, left.value, vector, span), vector
		}
		if rt.Kind == foundation.BoolKind {
			right.value, rt = c.splat(b, right.value, vector, span), vector
		}
	}
	return c.emitBinary(b, op, left.value, lt, right.value, rt, span)
}

func (c *analyzer) lowerCompound(b *fnBuilder, e env, op string, left ir.ValueID, leftType *foundation.Type, right parser.Expr, span foundation.Span) (ir.ValueID, *foundation.Type, error) {
	if op == "<<" || op == ">>" {
		return c.lowerShift(b, e, op, left, leftType, right, span)
	}
	if foundation.IsBoolean(leftType) {
		operand, err := c.lowerDetached(b, e, right, nil)
		if err != nil {
			return 0, nil, err
		}
		return c.emitBooleanBinary(b, detachedExpr{block: &ir.Block{}, value: left, type_: leftType}, operand, op, span)
	}
	rightArguments, err := c.lowerNumericArguments(b, e, op, []parser.Expr{right})
	if err != nil {
		return 0, nil, err
	}
	arguments := append([]detachedExpr{{block: &ir.Block{}, value: left, type_: leftType}}, rightArguments...)
	return c.emitResolvedBinary(b, e, op, arguments, leftType, span)
}

func (c *analyzer) lowerShift(b *fnBuilder, e env, op string, left ir.ValueID, leftType *foundation.Type, right parser.Expr, span foundation.Span) (ir.ValueID, *foundation.Type, error) {
	countType := foundation.ShiftCountType(leftType)
	if countType == nil {
		return 0, nil, diag(span, "%s requires an int32/uint32 scalar or integer vector on the left, got %s", op, leftType)
	}
	want := countType
	if countType.Kind == foundation.VectorKind && !contextualNumeric(right) {
		want = nil
	}
	r, rt, err := c.lowerExpr(b, e, right, want)
	if err != nil {
		return 0, nil, err
	}
	r, rt, err = c.prepareShiftCount(b, r, rt, leftType, right.GetSpan())
	if err != nil {
		return 0, nil, err
	}
	return c.emitBinary(b, op, left, leftType, r, rt, span)
}

func (c *analyzer) prepareShiftCount(b *fnBuilder, value ir.ValueID, got, shifted *foundation.Type, span foundation.Span) (ir.ValueID, *foundation.Type, error) {
	want := foundation.ShiftCountType(shifted)
	if want == nil {
		return 0, nil, diag(span, "shift requires an int32/uint32 scalar or integer vector")
	}
	if want.Kind == foundation.VectorKind && foundation.Equal(got, foundation.Uint32Type) {
		value = c.splat(b, value, want, span)
		got = want
	}
	if !foundation.Equal(got, want) {
		return 0, nil, diag(span, "shift count is %s, want uint32 or %s", got, want)
	}
	value, got = c.normalizeShiftCount(b, value, got, span)
	return value, got, nil
}

// normalizeShiftCount gives Tach one backend-independent shift meaning: every
// 32-bit shift uses the low five bits of its count.
func (c *analyzer) normalizeShiftCount(b *fnBuilder, value ir.ValueID, t *foundation.Type, span foundation.Span) (ir.ValueID, *foundation.Type) {
	maskScalar := b.value()
	b.emit(&ir.Const{Result: maskScalar, Type: foundation.Uint32Type, Raw: "31", Span: span})
	mask := maskScalar
	if t.Kind == foundation.VectorKind {
		values := make([]ir.ValueID, t.Lanes)
		for i := range values {
			values[i] = maskScalar
		}
		mask = b.value()
		b.emit(&ir.Composite{Result: mask, Type: t, Values: values, Span: span})
	}
	result := b.value()
	b.emit(&ir.Binary{Result: result, Type: t, Op: "&", Left: value, Right: mask, Span: span})
	return result, t
}

func (c *analyzer) emitBinary(b *fnBuilder, op string, l ir.ValueID, lt *foundation.Type, r ir.ValueID, rt *foundation.Type, span foundation.Span) (ir.ValueID, *foundation.Type, error) {
	if vectorScalarOperator(op) {
		if lt.Kind == foundation.VectorKind && foundation.Equal(rt, lt.Elem) && op != "*" && op != "/" {
			r, rt = c.splat(b, r, lt, span), lt
		} else if rt.Kind == foundation.VectorKind && foundation.Equal(lt, rt.Elem) && op != "*" {
			l, lt = c.splat(b, l, rt, span), rt
		}
	}
	var out *foundation.Type
	switch op {
	case "==", "!=", "<", "<=", ">", ">=":
		if !foundation.Equal(lt, rt) || !foundation.IsNumeric(lt) && !((op == "==" || op == "!=") && foundation.IsBoolean(lt)) {
			return 0, nil, diag(span, "comparison %s requires matching numeric operands; got %s and %s", op, lt, rt)
		}
		out = foundation.BoolShape(lt)
	case "+", "-":
		if !foundation.Equal(lt, rt) || !foundation.IsNumeric(lt) {
			return 0, nil, diag(span, "%s requires matching numeric operands; got %s and %s", op, lt, rt)
		}
		out = lt
	case "*":
		if foundation.Equal(lt, rt) && foundation.IsNumeric(lt) {
			out = lt
		} else if lt.Kind == foundation.VectorKind && foundation.Equal(lt.Elem, rt) {
			out = lt
		} else if rt.Kind == foundation.VectorKind && foundation.Equal(rt.Elem, lt) {
			out = rt
		} else {
			return 0, nil, diag(span, "cannot multiply %s by %s", lt, rt)
		}
	case "/":
		if foundation.Equal(lt, rt) && foundation.IsNumeric(lt) {
			out = lt
		} else if lt.Kind == foundation.VectorKind && foundation.Equal(lt.Elem, rt) {
			out = lt
		} else {
			return 0, nil, diag(span, "cannot divide %s by %s", lt, rt)
		}
	case "%":
		if !foundation.Equal(lt, rt) || !foundation.IsNumericScalar(lt) {
			return 0, nil, diag(span, "%% requires matching scalar numeric operands; got %s and %s", lt, rt)
		}
		out = lt
	case "&", "|", "^":
		if !foundation.Equal(lt, rt) || !foundation.IsIntegerLike(lt) && !foundation.IsBoolean(lt) {
			return 0, nil, diag(span, "%s requires matching integer or boolean operands; got %s and %s", op, lt, rt)
		}
		out = lt
	case "<<", ">>":
		if !foundation.IsIntegerLike(lt) || !foundation.Equal(rt, foundation.ShiftCountType(lt)) {
			return 0, nil, diag(span, "%s requires %s shifted by %s; got %s and %s", op, lt, foundation.ShiftCountType(lt), lt, rt)
		}
		out = lt
	default:
		return 0, nil, diag(span, "unsupported binary operator %s", op)
	}
	id := b.value()
	b.emit(&ir.Binary{Result: id, Type: out, Op: op, Left: l, Right: r, Span: span})
	return id, out, nil
}
func intrinsicBuiltin(name string) (ir.IntrinsicKind, bool) {
	switch name {
	case "abs":
		return ir.IntrinsicAbs, true
	case "floor":
		return ir.IntrinsicFloor, true
	case "ceil":
		return ir.IntrinsicCeil, true
	case "trunc":
		return ir.IntrinsicTrunc, true
	case "sin":
		return ir.IntrinsicSin, true
	case "cos":
		return ir.IntrinsicCos, true
	case "tan":
		return ir.IntrinsicTan, true
	case "exp":
		return ir.IntrinsicExp, true
	case "exp2":
		return ir.IntrinsicExp2, true
	case "log":
		return ir.IntrinsicLog, true
	case "log2":
		return ir.IntrinsicLog2, true
	case "sqrt":
		return ir.IntrinsicSqrt, true
	case "rsqrt":
		return ir.IntrinsicRSqrt, true
	case "pow":
		return ir.IntrinsicPow, true
	case "min":
		return ir.IntrinsicMin, true
	case "max":
		return ir.IntrinsicMax, true
	case "clamp":
		return ir.IntrinsicClamp, true
	case "fma":
		return ir.IntrinsicFma, true
	case "dot":
		return ir.IntrinsicDot, true
	case "length":
		return ir.IntrinsicLength, true
	case "distance":
		return ir.IntrinsicDistance, true
	case "cross":
		return ir.IntrinsicCross, true
	case "normalize":
		return ir.IntrinsicNormalize, true
	case "all":
		return ir.IntrinsicAll, true
	case "any":
		return ir.IntrinsicAny, true
	case "select":
		return ir.IntrinsicSelect, true
	default:
		return 0, false
	}
}

func ReservedName(name string) bool {
	if _, ok := intrinsicBuiltin(name); ok {
		return true
	}
	if _, ok := atomicBuiltin(name); ok {
		return true
	}
	if name == "break" || name == "continue" || name == "vec" || name == "workgroupBarrier" || name == "bufferBarrier" || name == "run" || name == "over" || name == "transient" || name == "ceilDiv" || name == "view" || name == "srgb8" {
		return true
	}
	return foundation.ParseBuiltin(name) != nil
}

func viewType(expression parser.TypeExpr) (ir.ViewFormat, bool) {
	generic, ok := expression.(*parser.GenericType)
	if !ok || generic.Name != "view" || len(generic.Args) != 1 {
		return 0, false
	}
	format, ok := generic.Args[0].(*parser.NamedType)
	return ir.SRGB8ViewFormat, ok && format.Name == "srgb8"
}
