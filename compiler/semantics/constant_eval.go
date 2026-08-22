package semantics

import (
	"fmt"
	"math"
	"strconv"
	"tach/foundation"
	"tach/ir"
)

func evaluateConstantBlock(block *ir.Block, values map[ir.ValueID]*foundation.ConstantValue) ([]*foundation.ConstantValue, error) {
	for _, instruction := range block.Instrs {
		var value *foundation.ConstantValue
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
				value = &foundation.ConstantValue{Type: item.Type, Bits: []uint32{base.Bits[item.Index]}}
			}
		case *ir.VectorIndex:
			base, index := values[item.Base], values[item.Index]
			if base == nil || index == nil || len(index.Bits) != 1 || index.Bits[0] >= uint32(len(base.Bits)) {
				err = fmt.Errorf("constant vector index is outside its lanes")
			} else {
				value = &foundation.ConstantValue{Type: item.Type, Bits: []uint32{base.Bits[index.Bits[0]]}}
			}
		case *ir.Intrinsic:
			arguments := make([]*foundation.ConstantValue, len(item.Args))
			for index, id := range item.Args {
				arguments[index] = values[id]
			}
			value, err = evaluateIntrinsic(item.Kind, item.Type, arguments)
		case *ir.If:
			condition := values[item.Cond]
			if condition == nil || !foundation.Equal(condition.Type, foundation.BoolType) || len(condition.Bits) != 1 {
				err = fmt.Errorf("constant condition is unavailable")
				break
			}
			branch := item.Else
			if condition.Bits[0] != 0 {
				branch = item.Then
			}
			var yielded []*foundation.ConstantValue
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
	out := make([]*foundation.ConstantValue, len(yield.Values))
	for index, id := range yield.Values {
		out[index] = values[id]
		if out[index] == nil {
			return nil, fmt.Errorf("constant branch value is unavailable")
		}
	}
	return out, nil
}

func parseConstant(t *foundation.Type, raw string) (*foundation.ConstantValue, error) {
	var bits uint32
	switch t.Kind {
	case foundation.BoolKind:
		if raw != "false" {
			if raw != "true" {
				return nil, fmt.Errorf("invalid bool constant %q", raw)
			}
			bits = 1
		}
	case foundation.Int32Kind:
		number, err := strconv.ParseInt(raw, 10, 32)
		if err != nil {
			return nil, err
		}
		bits = uint32(int32(number))
	case foundation.Uint32Kind:
		number, err := strconv.ParseUint(raw, 10, 32)
		if err != nil {
			return nil, err
		}
		bits = uint32(number)
	case foundation.Float16Kind, foundation.Float32Kind:
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
	return &foundation.ConstantValue{Type: t, Bits: []uint32{bits}}, nil
}

func composeConstant(t *foundation.Type, ids []ir.ValueID, values map[ir.ValueID]*foundation.ConstantValue) (*foundation.ConstantValue, error) {
	if t.Kind != foundation.VectorKind || len(ids) != t.Lanes {
		return nil, fmt.Errorf("invalid constant vector composition")
	}
	out := &foundation.ConstantValue{Type: t, Bits: make([]uint32, len(ids))}
	for index, id := range ids {
		value := values[id]
		if value == nil || !foundation.Equal(value.Type, t.Elem) || len(value.Bits) != 1 {
			return nil, fmt.Errorf("constant vector lane %d is unavailable", index)
		}
		out.Bits[index] = value.Bits[0]
	}
	return out, nil
}

func evaluateUnary(operator string, operand *foundation.ConstantValue, resultType *foundation.Type) (*foundation.ConstantValue, error) {
	if operand == nil {
		return nil, fmt.Errorf("constant operand is unavailable")
	}
	out := &foundation.ConstantValue{Type: resultType, Bits: make([]uint32, len(operand.Bits))}
	element := resultType
	if resultType.Kind == foundation.VectorKind {
		element = resultType.Elem
	}
	for index, bits := range operand.Bits {
		switch operator {
		case "!":
			if element.Kind != foundation.BoolKind {
				return nil, fmt.Errorf("! requires bool")
			}
			out.Bits[index] = 1 - bits
		case "~":
			out.Bits[index] = ^bits
		case "-":
			switch element.Kind {
			case foundation.Int32Kind:
				out.Bits[index] = uint32(-int32(bits))
			case foundation.Float16Kind, foundation.Float32Kind:
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

func evaluateBinary(operator string, left, right *foundation.ConstantValue, resultType *foundation.Type) (*foundation.ConstantValue, error) {
	if left == nil || right == nil {
		return nil, fmt.Errorf("constant operand is unavailable")
	}
	lanes := max(len(left.Bits), len(right.Bits))
	if len(left.Bits) != 1 && len(left.Bits) != lanes || len(right.Bits) != 1 && len(right.Bits) != lanes {
		return nil, fmt.Errorf("constant vector widths differ")
	}
	out := &foundation.ConstantValue{Type: resultType, Bits: make([]uint32, lanes)}
	element := left.Type
	if element.Kind == foundation.VectorKind {
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
	return out, nil
}

func binaryLane(operator string, t *foundation.Type, left, right uint32) (uint32, error) {
	if operator == "==" || operator == "!=" || operator == "<" || operator == "<=" || operator == ">" || operator == ">=" {
		var less, equal bool
		switch t.Kind {
		case foundation.BoolKind:
			if operator != "==" && operator != "!=" {
				return 0, fmt.Errorf("ordered comparison requires numeric values")
			}
			equal = left == right
		case foundation.Int32Kind:
			less, equal = int32(left) < int32(right), left == right
		case foundation.Uint32Kind:
			less, equal = left < right, left == right
		case foundation.Float16Kind, foundation.Float32Kind:
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
	if t.Kind == foundation.BoolKind {
		switch operator {
		case "&":
			return left & right, nil
		case "|":
			return left | right, nil
		case "^":
			return left ^ right, nil
		default:
			return 0, fmt.Errorf("operator %s is invalid for bool", operator)
		}
	}
	if t.Kind == foundation.Float16Kind || t.Kind == foundation.Float32Kind {
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
	if t.Kind != foundation.Int32Kind && t.Kind != foundation.Uint32Kind {
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
		if t.Kind == foundation.Int32Kind {
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
		if t.Kind == foundation.Int32Kind {
			return uint32(int32(left) >> (right & 31)), nil
		}
		return left >> (right & 31), nil
	default:
		return 0, fmt.Errorf("unsupported constant binary operator %s", operator)
	}
}

func convertConstant(value *foundation.ConstantValue, target *foundation.Type) (*foundation.ConstantValue, error) {
	if value == nil || len(value.Bits) != 1 || !foundation.IsNumericScalar(value.Type) || !foundation.IsNumericScalar(target) {
		return nil, fmt.Errorf("constant scalar conversion is invalid")
	}
	bits := value.Bits[0]
	if (value.Type.Kind == foundation.Int32Kind || value.Type.Kind == foundation.Uint32Kind) && (target.Kind == foundation.Int32Kind || target.Kind == foundation.Uint32Kind) {
		return &foundation.ConstantValue{Type: target, Bits: []uint32{bits}}, nil
	}
	var number float64
	switch value.Type.Kind {
	case foundation.Int32Kind:
		number = float64(int32(bits))
	case foundation.Uint32Kind:
		number = float64(bits)
	case foundation.Float16Kind, foundation.Float32Kind:
		number = floatValue(value.Type, bits)
	}
	if target.Kind == foundation.Float16Kind || target.Kind == foundation.Float32Kind {
		converted, err := floatBits(target, number)
		return &foundation.ConstantValue{Type: target, Bits: []uint32{converted}}, err
	}
	if value.Type.Kind == foundation.Float16Kind || value.Type.Kind == foundation.Float32Kind {
		number = math.Trunc(number)
	}
	if target.Kind == foundation.Int32Kind {
		if number < math.MinInt32 || number > math.MaxInt32 {
			return nil, fmt.Errorf("constant conversion is outside int32")
		}
		return &foundation.ConstantValue{Type: target, Bits: []uint32{uint32(int32(number))}}, nil
	}
	if number < 0 || number > math.MaxUint32 {
		return nil, fmt.Errorf("constant conversion is outside uint32")
	}
	return &foundation.ConstantValue{Type: target, Bits: []uint32{uint32(number)}}, nil
}
