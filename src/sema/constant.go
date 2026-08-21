package sema

import (
	"errors"
	"fmt"
	"math"
	"strconv"
	"strings"

	"tach/src/ast"
	"tach/src/ir"
	"tach/src/source"
	"tach/src/types"
)

type runtimeConstantDependency struct{ error }

func (e *runtimeConstantDependency) Unwrap() error { return e.error }

func (c *Checker) tryConstant(expression ast.Expr, expected *types.Type, environment env) (*types.Value, bool, error) {
	value, err := c.evaluateConstant(expression, expected, environment)
	var runtime *runtimeConstantDependency
	if errors.As(err, &runtime) {
		return nil, false, nil
	}
	return value, err == nil, err
}

func (c *Checker) collectConstants() error {
	for _, declaration := range c.ast.Decls {
		item, ok := declaration.(*ast.ConstDecl)
		if !ok {
			continue
		}
		if ReservedName(item.Name) {
			return diag(item.Span, "constant name %q is reserved by Tach", item.Name)
		}
		if c.types[item.Name] != nil || c.consts[item.Name] != nil {
			return diag(item.Span, "declaration %q is already defined", item.Name)
		}
		c.consts[item.Name] = &constantDef{decl: item}
	}
	var diagnostics source.Diagnostics
	reported := map[string]bool{}
	for _, declaration := range c.ast.Decls {
		item, ok := declaration.(*ast.ConstDecl)
		if !ok || c.consts[item.Name].state == 3 {
			continue
		}
		if _, err := c.resolveConstant(item.Name, item.Span); err != nil {
			key := err.Error()
			if !reported[key] {
				diagnostics, reported[key] = appendError(diagnostics, err), true
			}
		}
	}
	if len(diagnostics) > 0 {
		return diagnostics
	}
	return nil
}

func (c *Checker) resolveConstant(name string, reference source.Span) (*types.Value, error) {
	definition := c.consts[name]
	if definition == nil || !c.visible(name, reference.File) {
		return nil, diag(reference, "unknown constant %q", name)
	}
	if definition.state == 2 {
		return definition.value, nil
	}
	if definition.state == 3 {
		return nil, definition.err
	}
	if definition.state == 1 {
		start := 0
		for index, item := range c.constantStack {
			if item == name {
				start = index
				break
			}
		}
		chain := append(append([]string(nil), c.constantStack[start:]...), name)
		diagnostic := &source.Diagnostic{Span: reference, Message: fmt.Sprintf("compile-time constant cycle: %s", strings.Join(chain, " -> "))}
		for _, item := range chain[:len(chain)-1] {
			constant := c.consts[item]
			diagnostic.Related = append(diagnostic.Related, source.Related{Span: constant.decl.Span, Message: fmt.Sprintf("constant %q participates in this cycle", item)})
			constant.state, constant.err = 3, diagnostic
		}
		return nil, diagnostic
	}
	definition.state = 1
	c.constantStack = append(c.constantStack, name)
	defer func() { c.constantStack = c.constantStack[:len(c.constantStack)-1] }()
	value, err := c.evaluateConstantBinding(definition.decl.Type, definition.decl.Value, newEnv())
	if err != nil {
		if definition.state != 3 {
			definition.state, definition.err = 3, err
		}
		return nil, err
	}
	definition.value = value
	definition.state = 2
	return definition.value, nil
}

func (c *Checker) evaluateConstantBinding(typeExpression ast.TypeExpr, expression ast.Expr, environment env) (*types.Value, error) {
	var expected *types.Type
	var err error
	if typeExpression != nil {
		expected, err = c.resolveTypeIn(typeExpression, &environment)
		if err != nil {
			return nil, err
		}
		if !types.IsConstantType(expected) {
			return nil, diag(typeExpression.GetSpan(), "compile-time constant type must be a scalar or vector, got %s", expected)
		}
	}
	return c.evaluateConstant(expression, expected, environment)
}

func (c *Checker) evaluateConstant(expression ast.Expr, expected *types.Type, environment env) (*types.Value, error) {
	block := &ir.Block{}
	builder := &fnBuilder{
		fn:       &ir.Function{Kind: ir.Helper, Return: types.TVoid, Body: block},
		ids:      &idAllocator{},
		block:    block,
		comptime: true,
	}
	result, resultType, err := c.lowerExpr(builder, environment, expression, expected)
	if err != nil {
		return nil, err
	}
	if !types.IsConstantType(resultType) {
		return nil, diag(expression.GetSpan(), "compile-time expression produces %s; constants must be scalar or vector values", resultType)
	}
	if expected != nil && !types.Equal(resultType, expected) {
		return nil, diag(expression.GetSpan(), "compile-time expression is %s, want %s", resultType, expected)
	}
	values := map[ir.ValueID]*types.Value{}
	if _, err := evaluateConstantBlock(block, values); err != nil {
		return nil, diag(expression.GetSpan(), "invalid compile-time expression: %v", err)
	}
	value := values[result]
	if value == nil {
		return nil, diag(expression.GetSpan(), "compile-time expression has no value")
	}
	return value, nil
}

func evaluateConstantBlock(block *ir.Block, values map[ir.ValueID]*types.Value) ([]*types.Value, error) {
	for _, instruction := range block.Instrs {
		var value *types.Value
		var err error
		switch item := instruction.(type) {
		case *ir.Const:
			value, err = parseConstant(item.Type, item.Raw)
		case *ir.Unary:
			value, err = evaluateUnary(item.Op, values[item.X], item.Type)
		case *ir.Binary:
			value, err = evaluateBinary(item.Op, values[item.Left], values[item.Right], item.Type)
		case *ir.Convert:
			value, err = convertConstant(values[item.X], item.Type)
		case *ir.Composite:
			value, err = composeConstant(item.Type, item.Values, values)
		case *ir.Extract:
			base := values[item.Base]
			if base == nil || item.Index < 0 || item.Index >= len(base.Bits) {
				err = fmt.Errorf("constant vector component is unavailable")
			} else {
				value = &types.Value{Type: item.Type, Bits: []uint32{base.Bits[item.Index]}}
			}
		case *ir.VectorIndex:
			base, index := values[item.Base], values[item.Index]
			if base == nil || index == nil || len(index.Bits) != 1 || index.Bits[0] >= uint32(len(base.Bits)) {
				err = fmt.Errorf("constant vector index is outside its lanes")
			} else {
				value = &types.Value{Type: item.Type, Bits: []uint32{base.Bits[index.Bits[0]]}}
			}
		case *ir.Intrinsic:
			arguments := make([]*types.Value, len(item.Args))
			for index, id := range item.Args {
				arguments[index] = values[id]
			}
			value, err = evaluateIntrinsic(item.Kind, item.Type, arguments)
		case *ir.If:
			condition := values[item.Cond]
			if condition == nil || !types.Equal(condition.Type, types.TBool) || len(condition.Bits) != 1 {
				err = fmt.Errorf("constant condition is unavailable")
				break
			}
			branch := item.Else
			if condition.Bits[0] != 0 {
				branch = item.Then
			}
			var yielded []*types.Value
			yielded, err = evaluateConstantBlock(branch, values)
			if err == nil && len(yielded) != len(item.Results) {
				err = fmt.Errorf("constant branch yielded %d values, want %d", len(yielded), len(item.Results))
			}
			if err == nil {
				for index, result := range item.Results {
					values[result.ID] = yielded[index]
				}
			}
			continue
		default:
			err = fmt.Errorf("%T is a runtime operation", instruction)
		}
		if err != nil {
			return nil, err
		}
		if definition, ok := instruction.(ir.ValueDef); ok {
			values[definition.ResultValue()] = value
		}
	}
	if block.Term == nil {
		return nil, nil
	}
	yield, ok := block.Term.(*ir.Yield)
	if !ok {
		return nil, fmt.Errorf("constant expression contains runtime control flow")
	}
	out := make([]*types.Value, len(yield.Values))
	for index, id := range yield.Values {
		out[index] = values[id]
		if out[index] == nil {
			return nil, fmt.Errorf("constant branch value is unavailable")
		}
	}
	return out, nil
}

func parseConstant(t *types.Type, raw string) (*types.Value, error) {
	var bits uint32
	switch t.Kind {
	case types.Bool:
		if raw != "false" {
			if raw != "true" {
				return nil, fmt.Errorf("invalid bool constant %q", raw)
			}
			bits = 1
		}
	case types.I32:
		number, err := strconv.ParseInt(raw, 10, 32)
		if err != nil {
			return nil, err
		}
		bits = uint32(int32(number))
	case types.U32:
		number, err := strconv.ParseUint(raw, 10, 32)
		if err != nil {
			return nil, err
		}
		bits = uint32(number)
	case types.F16, types.F32:
		number, err := strconv.ParseFloat(raw, 64)
		if err != nil {
			return nil, err
		}
		bits, err = floatBits(t, number)
		if err != nil {
			return nil, err
		}
	default:
		return nil, fmt.Errorf("%s is not a scalar constant type", t)
	}
	return &types.Value{Type: t, Bits: []uint32{bits}}, nil
}

func composeConstant(t *types.Type, ids []ir.ValueID, values map[ir.ValueID]*types.Value) (*types.Value, error) {
	if t.Kind != types.Vector || len(ids) != t.Lanes {
		return nil, fmt.Errorf("invalid constant vector composition")
	}
	out := &types.Value{Type: t, Bits: make([]uint32, len(ids))}
	for index, id := range ids {
		value := values[id]
		if value == nil || !types.Equal(value.Type, t.Elem) || len(value.Bits) != 1 {
			return nil, fmt.Errorf("constant vector lane %d is unavailable", index)
		}
		out.Bits[index] = value.Bits[0]
	}
	return out, nil
}

func evaluateUnary(operator string, operand *types.Value, resultType *types.Type) (*types.Value, error) {
	if operand == nil {
		return nil, fmt.Errorf("constant operand is unavailable")
	}
	out := &types.Value{Type: resultType, Bits: make([]uint32, len(operand.Bits))}
	element := resultType
	if resultType.Kind == types.Vector {
		element = resultType.Elem
	}
	for index, bits := range operand.Bits {
		switch operator {
		case "!":
			if element.Kind != types.Bool {
				return nil, fmt.Errorf("! requires bool")
			}
			out.Bits[index] = 1 - bits
		case "~":
			out.Bits[index] = ^bits
		case "-":
			switch element.Kind {
			case types.I32:
				out.Bits[index] = uint32(-int32(bits))
			case types.F16, types.F32:
				value, err := floatBits(element, -floatValue(element, bits))
				if err != nil {
					return nil, err
				}
				out.Bits[index] = value
			default:
				return nil, fmt.Errorf("unary - requires a signed numeric value")
			}
		default:
			return nil, fmt.Errorf("unsupported constant unary operator %s", operator)
		}
	}
	return out, nil
}

func evaluateBinary(operator string, left, right *types.Value, resultType *types.Type) (*types.Value, error) {
	if left == nil || right == nil {
		return nil, fmt.Errorf("constant operand is unavailable")
	}
	lanes := max(len(left.Bits), len(right.Bits))
	if len(left.Bits) != 1 && len(left.Bits) != lanes || len(right.Bits) != 1 && len(right.Bits) != lanes {
		return nil, fmt.Errorf("constant vector widths differ")
	}
	out := &types.Value{Type: resultType, Bits: make([]uint32, lanes)}
	element := left.Type
	if element.Kind == types.Vector {
		element = element.Elem
	}
	for lane := range lanes {
		l, r := left.Bits[min(lane, len(left.Bits)-1)], right.Bits[min(lane, len(right.Bits)-1)]
		var err error
		out.Bits[lane], err = binaryLane(operator, element, l, r)
		if err != nil {
			return nil, err
		}
	}
	if resultType.Kind == types.Bool {
		out.Bits = out.Bits[:1]
	}
	return out, nil
}

func binaryLane(operator string, t *types.Type, left, right uint32) (uint32, error) {
	if operator == "==" || operator == "!=" || operator == "<" || operator == "<=" || operator == ">" || operator == ">=" {
		var less, equal bool
		switch t.Kind {
		case types.I32:
			less, equal = int32(left) < int32(right), left == right
		case types.U32:
			less, equal = left < right, left == right
		case types.F16, types.F32:
			l, r := floatValue(t, left), floatValue(t, right)
			less, equal = l < r, l == r
		default:
			return 0, fmt.Errorf("comparison requires numeric values")
		}
		truth := operator == "==" && equal || operator == "!=" && !equal || operator == "<" && less ||
			operator == "<=" && (less || equal) || operator == ">" && !less && !equal || operator == ">=" && !less
		if truth {
			return 1, nil
		}
		return 0, nil
	}
	if t.Kind == types.F16 || t.Kind == types.F32 {
		l, r := floatValue(t, left), floatValue(t, right)
		var value float64
		switch operator {
		case "+":
			value = l + r
		case "-":
			value = l - r
		case "*":
			value = l * r
		case "/":
			if r == 0 {
				return 0, fmt.Errorf("division by zero")
			}
			value = l / r
		case "%":
			if r == 0 {
				return 0, fmt.Errorf("remainder by zero")
			}
			value = math.Mod(l, r)
		default:
			return 0, fmt.Errorf("operator %s is invalid for %s", operator, t)
		}
		return floatBits(t, value)
	}
	if t.Kind != types.I32 && t.Kind != types.U32 {
		return 0, fmt.Errorf("operator %s requires numeric values", operator)
	}
	switch operator {
	case "+":
		return left + right, nil
	case "-":
		return left - right, nil
	case "*":
		return left * right, nil
	case "/", "%":
		if right == 0 {
			if operator == "%" {
				return 0, fmt.Errorf("remainder by zero")
			}
			return 0, fmt.Errorf("division by zero")
		}
		if t.Kind == types.I32 {
			l, r := int32(left), int32(right)
			if l == math.MinInt32 && r == -1 {
				return 0, fmt.Errorf("signed division overflows int32")
			}
			if operator == "/" {
				return uint32(l / r), nil
			}
			return uint32(l % r), nil
		}
		if operator == "/" {
			return left / right, nil
		}
		return left % right, nil
	case "&":
		return left & right, nil
	case "|":
		return left | right, nil
	case "^":
		return left ^ right, nil
	case "<<":
		return left << (right & 31), nil
	case ">>":
		if t.Kind == types.I32 {
			return uint32(int32(left) >> (right & 31)), nil
		}
		return left >> (right & 31), nil
	default:
		return 0, fmt.Errorf("unsupported constant binary operator %s", operator)
	}
}

func convertConstant(value *types.Value, target *types.Type) (*types.Value, error) {
	if value == nil || len(value.Bits) != 1 || !types.IsNumericScalar(value.Type) || !types.IsNumericScalar(target) {
		return nil, fmt.Errorf("constant scalar conversion is invalid")
	}
	bits := value.Bits[0]
	if (value.Type.Kind == types.I32 || value.Type.Kind == types.U32) && (target.Kind == types.I32 || target.Kind == types.U32) {
		return &types.Value{Type: target, Bits: []uint32{bits}}, nil
	}
	var number float64
	switch value.Type.Kind {
	case types.I32:
		number = float64(int32(bits))
	case types.U32:
		number = float64(bits)
	case types.F16, types.F32:
		number = floatValue(value.Type, bits)
	}
	if target.Kind == types.F16 || target.Kind == types.F32 {
		converted, err := floatBits(target, number)
		return &types.Value{Type: target, Bits: []uint32{converted}}, err
	}
	if value.Type.Kind == types.F16 || value.Type.Kind == types.F32 {
		number = math.Trunc(number)
	}
	if target.Kind == types.I32 {
		if number < math.MinInt32 || number > math.MaxInt32 {
			return nil, fmt.Errorf("constant conversion is outside int32")
		}
		return &types.Value{Type: target, Bits: []uint32{uint32(int32(number))}}, nil
	}
	if number < 0 || number > math.MaxUint32 {
		return nil, fmt.Errorf("constant conversion is outside uint32")
	}
	return &types.Value{Type: target, Bits: []uint32{uint32(number)}}, nil
}

func evaluateIntrinsic(kind ir.IntrinsicKind, resultType *types.Type, arguments []*types.Value) (*types.Value, error) {
	for _, argument := range arguments {
		if argument == nil {
			return nil, fmt.Errorf("constant intrinsic argument is unavailable")
		}
	}
	resultLanes := 1
	resultElement := resultType
	if resultType.Kind == types.Vector {
		resultLanes, resultElement = resultType.Lanes, resultType.Elem
	}
	if kind == ir.IntrinsicDot || kind == ir.IntrinsicLength || kind == ir.IntrinsicDistance {
		return evaluateGeometricScalar(kind, resultElement, arguments)
	}
	if kind == ir.IntrinsicCross {
		out := &types.Value{Type: resultType, Bits: make([]uint32, 3)}
		left, right := arguments[0], arguments[1]
		for lane, pair := range [][2]int{{1, 2}, {2, 0}, {0, 1}} {
			a := floatValue(resultElement, left.Bits[pair[0]]) * floatValue(resultElement, right.Bits[pair[1]])
			b := floatValue(resultElement, left.Bits[pair[1]]) * floatValue(resultElement, right.Bits[pair[0]])
			bits, err := floatBits(resultElement, a-b)
			if err != nil {
				return nil, err
			}
			out.Bits[lane] = bits
		}
		return out, nil
	}
	if kind == ir.IntrinsicNormalize {
		length, err := evaluateGeometricScalar(ir.IntrinsicLength, resultElement, arguments)
		if err != nil {
			return nil, err
		}
		denominator := floatValue(resultElement, length.Bits[0])
		if denominator == 0 {
			return nil, fmt.Errorf("normalize of a zero vector")
		}
		out := &types.Value{Type: resultType, Bits: make([]uint32, resultLanes)}
		for lane := range resultLanes {
			bits, err := floatBits(resultElement, floatValue(resultElement, arguments[0].Bits[lane])/denominator)
			if err != nil {
				return nil, err
			}
			out.Bits[lane] = bits
		}
		return out, nil
	}
	out := &types.Value{Type: resultType, Bits: make([]uint32, resultLanes)}
	for lane := range resultLanes {
		laneArguments := make([]uint32, len(arguments))
		for index, argument := range arguments {
			laneArguments[index] = argument.Bits[min(lane, len(argument.Bits)-1)]
		}
		bits, err := intrinsicLane(kind, resultElement, laneArguments)
		if err != nil {
			return nil, err
		}
		out.Bits[lane] = bits
	}
	return out, nil
}

func evaluateGeometricScalar(kind ir.IntrinsicKind, t *types.Type, arguments []*types.Value) (*types.Value, error) {
	left := arguments[0]
	sum := 0.0
	for lane := range len(left.Bits) {
		value := floatValue(t, left.Bits[lane])
		switch kind {
		case ir.IntrinsicLength:
			sum += value * value
		case ir.IntrinsicDot:
			sum += value * floatValue(t, arguments[1].Bits[lane])
		case ir.IntrinsicDistance:
			difference := value - floatValue(t, arguments[1].Bits[lane])
			sum += difference * difference
		}
	}
	if kind != ir.IntrinsicDot {
		sum = math.Sqrt(sum)
	}
	bits, err := floatBits(t, sum)
	return &types.Value{Type: t, Bits: []uint32{bits}}, err
}

func intrinsicLane(kind ir.IntrinsicKind, t *types.Type, arguments []uint32) (uint32, error) {
	if t.Kind == types.I32 || t.Kind == types.U32 {
		left := arguments[0]
		switch kind {
		case ir.IntrinsicAbs:
			if t.Kind == types.I32 && int32(left) < 0 {
				left = uint32(-int32(left))
			}
			return left, nil
		case ir.IntrinsicMin, ir.IntrinsicMax:
			right := arguments[1]
			first, second := left, right
			if kind == ir.IntrinsicMin {
				first, second = right, left
			}
			less := first < second
			if t.Kind == types.I32 {
				less = int32(first) < int32(second)
			}
			if less {
				return right, nil
			}
			return left, nil
		case ir.IntrinsicClamp:
			bounded, err := intrinsicLane(ir.IntrinsicMax, t, arguments[:2])
			if err != nil {
				return 0, err
			}
			return intrinsicLane(ir.IntrinsicMin, t, []uint32{bounded, arguments[2]})
		}
		return 0, fmt.Errorf("intrinsic %s is invalid for %s constants", kind, t)
	}
	x := floatValue(t, arguments[0])
	var result float64
	switch kind {
	case ir.IntrinsicAbs:
		result = math.Abs(x)
	case ir.IntrinsicFloor:
		result = math.Floor(x)
	case ir.IntrinsicCeil:
		result = math.Ceil(x)
	case ir.IntrinsicTrunc:
		result = math.Trunc(x)
	case ir.IntrinsicSin:
		result = math.Sin(x)
	case ir.IntrinsicCos:
		result = math.Cos(x)
	case ir.IntrinsicTan:
		result = math.Tan(x)
	case ir.IntrinsicExp:
		result = math.Exp(x)
	case ir.IntrinsicExp2:
		result = math.Exp2(x)
	case ir.IntrinsicLog:
		result = math.Log(x)
	case ir.IntrinsicLog2:
		result = math.Log2(x)
	case ir.IntrinsicSqrt:
		result = math.Sqrt(x)
	case ir.IntrinsicRSqrt:
		result = 1 / math.Sqrt(x)
	case ir.IntrinsicPow:
		result = math.Pow(x, floatValue(t, arguments[1]))
	case ir.IntrinsicMin, ir.IntrinsicMax:
		y := floatValue(t, arguments[1])
		first, second := x, y
		if kind == ir.IntrinsicMin {
			first, second = y, x
		}
		if first < second {
			return arguments[1], nil
		}
		return arguments[0], nil
	case ir.IntrinsicClamp:
		bounded, err := intrinsicLane(ir.IntrinsicMax, t, arguments[:2])
		if err != nil {
			return 0, err
		}
		return intrinsicLane(ir.IntrinsicMin, t, []uint32{bounded, arguments[2]})
	case ir.IntrinsicFma:
		result = math.FMA(x, floatValue(t, arguments[1]), floatValue(t, arguments[2]))
	default:
		return 0, fmt.Errorf("intrinsic %s is not constant-evaluable", kind)
	}
	return floatBits(t, result)
}

func floatValue(t *types.Type, bits uint32) float64 {
	if t.Kind == types.F16 {
		return types.Float16frombits(uint16(bits))
	}
	return float64(math.Float32frombits(bits))
}

func floatBits(t *types.Type, value float64) (uint32, error) {
	if math.IsNaN(value) || math.IsInf(value, 0) {
		return 0, fmt.Errorf("constant expression produced a non-finite %s", t)
	}
	if t.Kind == types.F16 {
		bits, ok := types.Float16bits(value)
		if !ok {
			return 0, fmt.Errorf("constant expression is outside float16")
		}
		return uint32(bits), nil
	}
	converted := float32(value)
	if math.IsInf(float64(converted), 0) {
		return 0, fmt.Errorf("constant expression is outside float32")
	}
	return math.Float32bits(converted), nil
}

func materializeConstant(builder *fnBuilder, value *types.Value, span source.Span) (ir.ValueID, *types.Type) {
	result, instructions := ir.MaterializeConstant(value, span, builder.value)
	builder.block.Instrs = append(builder.block.Instrs, instructions...)
	return result, value.Type
}
