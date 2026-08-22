package ir

import (
	"fmt"
	"tach/foundation"
)

func verifyIntrinsic(x *Intrinsic, args []*foundation.Type) error {
	if x.Kind == IntrinsicAll || x.Kind == IntrinsicAny {
		if len(args) != 1 || args[0] == nil || args[0].Kind != foundation.VectorKind || args[0].Elem.Kind != foundation.BoolKind || !foundation.Equal(x.Type, foundation.BoolType) {
			return fmt.Errorf("intrinsic %s requires vec<bool, N> and returns bool", x.Kind)
		}
		return nil
	}
	if x.Kind == IntrinsicSelect {
		if len(args) != 3 || args[0] == nil || args[0].Kind != foundation.VectorKind || args[0].Elem.Kind != foundation.BoolKind || !foundation.Equal(args[1], args[2]) || !foundation.Equal(x.Type, args[1]) || x.Type.Kind != foundation.VectorKind || x.Type.Lanes != args[0].Lanes {
			return fmt.Errorf("intrinsic select requires a boolean-vector mask and matching vector arms")
		}
		return nil
	}
	rule := x.Kind.Rule()
	if rule.Arity == 0 {
		return fmt.Errorf("unknown intrinsic %d", x.Kind)
	}
	if len(args) != rule.Arity {
		return fmt.Errorf("intrinsic %s has %d args, want %d", x.Kind, len(args), rule.Arity)
	}
	t := args[0]
	element := t
	lanes := 0
	if t != nil && t.Kind == foundation.VectorKind {
		element, lanes = t.Elem, t.Lanes
	}
	if !rule.Domain.Accepts(element) || rule.VectorOnly && lanes == 0 || rule.Lanes != 0 && lanes != rule.Lanes {
		return fmt.Errorf("intrinsic %s does not accept %s", x.Kind, t)
	}
	for _, argument := range args[1:] {
		if !foundation.Equal(argument, t) {
			return fmt.Errorf("intrinsic %s requires matching operands", x.Kind)
		}
	}
	out := t
	if rule.ResultElement {
		out = element
	}
	if !foundation.Equal(x.Type, out) {
		return fmt.Errorf("intrinsic %s returns %s, got %s", x.Kind, out, x.Type)
	}
	return nil
}

func verifyBinary(x *Binary, l, r *foundation.Type) error {
	switch x.Op {
	case "+", "-":
		if !foundation.Equal(l, r) || !foundation.Equal(x.Type, l) || !foundation.IsNumeric(l) {
			return fmt.Errorf("%s requires matching numeric operands; got %s and %s -> %s", x.Op, l, r, x.Type)
		}
	case "*":
		if foundation.Equal(l, r) && foundation.Equal(x.Type, l) && foundation.IsNumeric(l) {
			return nil
		}
		if l.Kind == foundation.VectorKind && foundation.Equal(r, l.Elem) && foundation.Equal(x.Type, l) {
			return nil
		}
		if r.Kind == foundation.VectorKind && foundation.Equal(l, r.Elem) && foundation.Equal(x.Type, r) {
			return nil
		}
		return fmt.Errorf("* invalid for %s and %s -> %s", l, r, x.Type)
	case "/", "%":
		if foundation.Equal(l, r) && foundation.Equal(x.Type, l) && foundation.IsNumeric(l) && (x.Op == "/" || foundation.IsNumericScalar(l)) {
			return nil
		}
		if x.Op == "/" && l.Kind == foundation.VectorKind && foundation.Equal(r, l.Elem) && foundation.Equal(x.Type, l) {
			return nil
		}
		return fmt.Errorf("%s invalid for %s and %s", x.Op, l, r)
	case "==", "!=", "<", "<=", ">", ">=":
		valid := foundation.IsNumeric(l) || (x.Op == "==" || x.Op == "!=") && foundation.IsBoolean(l)
		if !foundation.Equal(l, r) || !valid || !foundation.Equal(x.Type, foundation.BoolShape(l)) {
			return fmt.Errorf("comparison invalid for %s and %s", l, r)
		}
	case "&&", "||":
		if !foundation.Equal(l, foundation.BoolType) || !foundation.Equal(r, foundation.BoolType) || !foundation.Equal(x.Type, foundation.BoolType) {
			return fmt.Errorf("logical op requires bool operands")
		}
	case "&", "|", "^":
		if !foundation.Equal(l, r) || !foundation.Equal(x.Type, l) || !foundation.IsIntegerLike(l) && !foundation.IsBoolean(l) {
			return fmt.Errorf("%s requires matching integer or boolean operands; got %s and %s -> %s", x.Op, l, r, x.Type)
		}
	case "<<", ">>":
		want := foundation.ShiftCountType(l)
		if want == nil || !foundation.Equal(r, want) || !foundation.Equal(x.Type, l) {
			return fmt.Errorf("%s requires integer value %s shifted by %s; got %s and %s -> %s", x.Op, l, want, l, r, x.Type)
		}
	default:
		return fmt.Errorf("unknown binary op %q", x.Op)
	}
	return nil
}
