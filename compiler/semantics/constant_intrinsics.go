package semantics

import (
	"fmt"
	"math"
	"tach/foundation"
	"tach/ir"
)

func evaluateIntrinsic(kind ir.IntrinsicKind, resultType *foundation.Type, arguments []*foundation.ConstantValue) (*foundation.ConstantValue, error) {
	for _, argument := range arguments {
		if argument == nil {
			return nil, fmt.Errorf("constant intrinsic argument is unavailable")
		}
	}
	if kind == ir.IntrinsicAll || kind == ir.IntrinsicAny {
		truth := kind == ir.IntrinsicAll
		for _, bit := range arguments[0].Bits {
			if kind == ir.IntrinsicAll {
				truth = truth && bit != 0
			} else {
				truth = truth || bit != 0
			}
		}
		return &foundation.ConstantValue{Type: foundation.BoolType, Bits: []uint32{boolBit(truth)}}, nil
	}
	if kind == ir.IntrinsicSelect {
		out := &foundation.ConstantValue{Type: resultType, Bits: make([]uint32, resultType.Lanes)}
		for lane, bit := range arguments[0].Bits {
			arm := arguments[2]
			if bit != 0 {
				arm = arguments[1]
			}
			out.Bits[lane] = arm.Bits[lane]
		}
		return out, nil
	}
	resultLanes := 1
	resultElement := resultType
	if resultType.Kind == foundation.VectorKind {
		resultLanes, resultElement = resultType.Lanes, resultType.Elem
	}
	if kind == ir.IntrinsicDot || kind == ir.IntrinsicLength || kind == ir.IntrinsicDistance {
		return evaluateGeometricScalar(kind, resultElement, arguments)
	}
	if kind == ir.IntrinsicCross {
		out := &foundation.ConstantValue{Type: resultType, Bits: make([]uint32, 3)}
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
		out := &foundation.ConstantValue{Type: resultType, Bits: make([]uint32, resultLanes)}
		for lane := range resultLanes {
			bits, err := floatBits(resultElement, floatValue(resultElement, arguments[0].Bits[lane])/denominator)
			if err != nil {
				return nil, err
			}
			out.Bits[lane] = bits
		}
		return out, nil
	}
	out := &foundation.ConstantValue{Type: resultType, Bits: make([]uint32, resultLanes)}
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

func boolBit(value bool) uint32 {
	if value {
		return 1
	}
	return 0
}

func evaluateGeometricScalar(kind ir.IntrinsicKind, t *foundation.Type, arguments []*foundation.ConstantValue) (*foundation.ConstantValue, error) {
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
	return &foundation.ConstantValue{Type: t, Bits: []uint32{bits}}, err
}

func intrinsicLane(kind ir.IntrinsicKind, t *foundation.Type, arguments []uint32) (uint32, error) {
	if t.Kind == foundation.Int32Kind || t.Kind == foundation.Uint32Kind {
		left := arguments[0]
		switch kind {
		case ir.IntrinsicAbs:
			if t.Kind == foundation.Int32Kind && int32(left) < 0 {
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
			if t.Kind == foundation.Int32Kind {
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

func floatValue(t *foundation.Type, bits uint32) float64 {
	if t.Kind == foundation.Float16Kind {
		return foundation.Float16FromBits(uint16(bits))
	}
	return float64(math.Float32frombits(bits))
}

func floatBits(t *foundation.Type, value float64) (uint32, error) {
	if math.IsNaN(value) || math.IsInf(value, 0) {
		return 0, fmt.Errorf("constant expression produced a non-finite %s", t)
	}
	if t.Kind == foundation.Float16Kind {
		bits, ok := foundation.Float16Bits(value)
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

func materializeConstant(builder *fnBuilder, value *foundation.ConstantValue, span foundation.Span) (ir.ValueID, *foundation.Type) {
	result, instructions := ir.MaterializeConstant(value, span, builder.value)
	builder.block.Instrs = append(builder.block.Instrs, instructions...)
	return result, value.Type
}
