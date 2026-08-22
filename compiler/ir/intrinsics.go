package ir

import "tach/foundation"

type IntrinsicKind uint8

const (
	IntrinsicAbs IntrinsicKind = iota + 1
	IntrinsicFloor
	IntrinsicCeil
	IntrinsicTrunc
	IntrinsicSin
	IntrinsicCos
	IntrinsicTan
	IntrinsicExp
	IntrinsicExp2
	IntrinsicLog
	IntrinsicLog2
	IntrinsicSqrt
	IntrinsicRSqrt
	IntrinsicPow
	IntrinsicMin
	IntrinsicMax
	IntrinsicClamp
	IntrinsicFma
	IntrinsicDot
	IntrinsicLength
	IntrinsicDistance
	IntrinsicCross
	IntrinsicNormalize
	IntrinsicAll
	IntrinsicAny
	IntrinsicSelect
)

type NumericDomain uint8

const (
	NumericAny NumericDomain = iota
	NumericSigned
	NumericFloat
)

type IntrinsicRule struct {
	Arity         int
	Domain        NumericDomain
	Broadcast     uint8
	VectorOnly    bool
	ResultElement bool
	Lanes         int
}

func (k IntrinsicKind) Rule() IntrinsicRule {
	switch k {
	case IntrinsicAbs:
		return IntrinsicRule{Arity: 1, Domain: NumericSigned}
	case IntrinsicFloor, IntrinsicCeil, IntrinsicTrunc, IntrinsicSin, IntrinsicCos, IntrinsicTan, IntrinsicExp, IntrinsicExp2, IntrinsicLog, IntrinsicLog2, IntrinsicSqrt, IntrinsicRSqrt:
		return IntrinsicRule{Arity: 1, Domain: NumericFloat}
	case IntrinsicPow:
		return IntrinsicRule{Arity: 2, Domain: NumericFloat, Broadcast: 1 << 1}
	case IntrinsicMin, IntrinsicMax:
		return IntrinsicRule{Arity: 2, Domain: NumericAny, Broadcast: 0b11}
	case IntrinsicClamp:
		return IntrinsicRule{Arity: 3, Domain: NumericAny, Broadcast: 0b111}
	case IntrinsicFma:
		return IntrinsicRule{Arity: 3, Domain: NumericFloat, Broadcast: 0b111}
	case IntrinsicDot, IntrinsicDistance:
		return IntrinsicRule{Arity: 2, Domain: NumericFloat, VectorOnly: true, ResultElement: true}
	case IntrinsicLength:
		return IntrinsicRule{Arity: 1, Domain: NumericFloat, VectorOnly: true, ResultElement: true}
	case IntrinsicCross:
		return IntrinsicRule{Arity: 2, Domain: NumericFloat, VectorOnly: true, Lanes: 3}
	case IntrinsicNormalize:
		return IntrinsicRule{Arity: 1, Domain: NumericFloat, VectorOnly: true}
	default:
		return IntrinsicRule{}
	}
}

func (d NumericDomain) Accepts(t *foundation.Type) bool {
	switch d {
	case NumericAny:
		return foundation.IsNumericScalar(t)
	case NumericSigned:
		return t != nil && (t.Kind == foundation.Int32Kind || t.Kind == foundation.Float16Kind || t.Kind == foundation.Float32Kind)
	case NumericFloat:
		return t != nil && (t.Kind == foundation.Float16Kind || t.Kind == foundation.Float32Kind)
	default:
		return false
	}
}

func (d NumericDomain) String() string {
	switch d {
	case NumericAny:
		return "numeric"
	case NumericSigned:
		return "signed numeric"
	case NumericFloat:
		return "floating-point"
	default:
		return "invalid"
	}
}
