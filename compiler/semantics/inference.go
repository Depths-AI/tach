package semantics

import (
	"fmt"
	"tach/foundation"
	"tach/ir"
	"tach/parser"
)

type detachedExpr struct {
	block      *ir.Block
	value      ir.ValueID
	type_      *foundation.Type
	contextual bool
	source     parser.Expr
}

func (c *analyzer) lowerDetached(b *fnBuilder, e env, expression parser.Expr, expected *foundation.Type) (detachedExpr, error) {
	block := &ir.Block{}
	value, type_, err := c.lowerExpr(b.child(block), e, expression, expected)
	return detachedExpr{block: block, value: value, type_: type_, contextual: contextualNumeric(expression), source: expression}, err
}

func contextualNumeric(expression parser.Expr) bool {
	switch x := expression.(type) {
	case *parser.NumberExpr:
		return true
	case *parser.UnaryExpr:
		return (x.Op == "-" || x.Op == "~") && contextualNumeric(x.X)
	case *parser.BinaryExpr:
		return contextualNumeric(x.Left) && contextualNumeric(x.Right)
	case *parser.ConditionalExpr:
		return contextualNumeric(x.Then) && contextualNumeric(x.Else)
	case *parser.CallExpr:
		callee, ok := x.Callee.(*parser.IdentExpr)
		if !ok {
			return false
		}
		if callee.Name != "vec" {
			if _, ok = intrinsicBuiltin(callee.Name); !ok {
				return false
			}
		}
		for _, argument := range x.Args {
			if !contextualNumeric(argument) {
				return false
			}
		}
		return true
	}
	return false
}

func numericElement(t *foundation.Type) (*foundation.Type, int) {
	if foundation.IsNumericScalar(t) {
		return t, 0
	}
	if t != nil && t.Kind == foundation.VectorKind && foundation.IsNumericScalar(t.Elem) {
		return t.Elem, t.Lanes
	}
	return nil, 0
}

func scalarElement(t *foundation.Type) (*foundation.Type, int) {
	if foundation.IsScalar(t) {
		return t, 0
	}
	if t != nil && t.Kind == foundation.VectorKind && foundation.IsScalar(t.Elem) {
		return t.Elem, t.Lanes
	}
	return nil, 0
}

func defaultNumericElement(domain ir.NumericDomain, arguments []detachedExpr) *foundation.Type {
	if domain == ir.NumericFloat {
		return foundation.Float32Type
	}
	for _, argument := range arguments {
		element, _ := numericElement(argument.type_)
		if element != nil && element.Kind == foundation.Float32Kind {
			return foundation.Float32Type
		}
	}
	if domain == ir.NumericSigned {
		return foundation.Int32Type
	}
	for _, argument := range arguments {
		element, _ := numericElement(argument.type_)
		if element != nil && element.Kind == foundation.Int32Kind {
			return foundation.Int32Type
		}
	}
	return foundation.Uint32Type
}

func (c *analyzer) lowerNumericArguments(b *fnBuilder, e env, operation string, expressions []parser.Expr) ([]detachedExpr, error) {
	arguments := make([]detachedExpr, len(expressions))
	for i, expression := range expressions {
		argument, err := c.lowerDetached(b, e, expression, nil)
		if err != nil {
			return nil, err
		}
		if element, _ := numericElement(argument.type_); element == nil {
			return nil, diag(expression.GetSpan(), "%s requires numeric values, got %s", operation, argument.type_)
		}
		arguments[i] = argument
	}
	return arguments, nil
}

func resolveNumericOperands(operation string, domain ir.NumericDomain, arguments []detachedExpr, element *foundation.Type, lanes int) (*foundation.Type, int, error) {
	for _, argument := range arguments {
		argumentElement, argumentLanes := numericElement(argument.type_)
		if argumentLanes > 0 {
			if lanes != 0 && lanes != argumentLanes {
				return nil, 0, fmt.Errorf("%s operands use conflicting vector widths %d and %d", operation, lanes, argumentLanes)
			}
			lanes = argumentLanes
		}
		if !argument.contextual {
			if !domain.Accepts(argumentElement) {
				return nil, 0, fmt.Errorf("%s requires %s operands, got %s", operation, domain, argument.type_)
			}
			if element != nil && !foundation.Equal(element, argumentElement) {
				return nil, 0, fmt.Errorf("%s requires matching numeric operands; got %s and %s", operation, element, argumentElement)
			}
			element = argumentElement
		}
	}
	if element == nil {
		element = defaultNumericElement(domain, arguments)
	}
	if !domain.Accepts(element) {
		return nil, 0, fmt.Errorf("%s requires %s operands", operation, domain)
	}
	return element, lanes, nil
}

func (c *analyzer) commitArgument(b *fnBuilder, e env, argument detachedExpr, want *foundation.Type) (ir.ValueID, *foundation.Type, error) {
	if argument.contextual {
		var err error
		argument, err = c.lowerDetached(b, e, argument.source, want)
		if err != nil {
			return 0, nil, err
		}
	}
	b.block.Instrs = append(b.block.Instrs, argument.block.Instrs...)
	return argument.value, argument.type_, nil
}

func (c *analyzer) lowerIntrinsic(b *fnBuilder, e env, x *parser.CallExpr, kind ir.IntrinsicKind, expected *foundation.Type) (ir.ValueID, *foundation.Type, error) {
	rule := kind.Rule()
	if rule.Arity == 0 {
		return 0, nil, diag(x.Span, "unsupported intrinsic %s", kind)
	}
	if len(x.Args) != rule.Arity {
		return 0, nil, diag(x.Span, "%s expects %d argument(s), got %d", kind, rule.Arity, len(x.Args))
	}
	arguments, err := c.lowerNumericArguments(b, e, kind.String(), x.Args)
	if err != nil {
		return 0, nil, err
	}

	var element *foundation.Type
	lanes := 0
	if expected != nil {
		var expectedLanes int
		element, expectedLanes = numericElement(expected)
		if element == nil || !rule.Domain.Accepts(element) || rule.ResultElement && expectedLanes != 0 || !rule.ResultElement && rule.VectorOnly && expectedLanes == 0 {
			return 0, nil, diag(x.Span, "%s result cannot satisfy %s context", kind, expected)
		}
		if !rule.ResultElement {
			lanes = expectedLanes
		}
	}
	element, lanes, err = resolveNumericOperands(kind.String(), rule.Domain, arguments, element, lanes)
	if err != nil {
		return 0, nil, diag(x.Span, "%v", err)
	}
	if rule.VectorOnly && lanes == 0 {
		return 0, nil, diag(x.Span, "%s requires floating-point vectors", kind)
	}
	if rule.Lanes != 0 && lanes != rule.Lanes {
		return 0, nil, diag(x.Span, "%s requires %d-lane vectors", kind, rule.Lanes)
	}
	vector := foundation.VectorOf(element, lanes)
	args := make([]ir.ValueID, len(arguments))
	for i, argument := range arguments {
		_, argumentLanes := numericElement(argument.type_)
		want := element
		if argumentLanes > 0 {
			want = vector
		}
		value, type_, err := c.commitArgument(b, e, argument, want)
		if err != nil {
			return 0, nil, err
		}
		if foundation.Equal(type_, vector) {
			args[i] = value
			continue
		}
		if foundation.Equal(type_, element) && lanes > 0 && rule.Broadcast&(1<<i) != 0 {
			args[i] = c.splat(b, value, vector, x.Args[i].GetSpan())
			continue
		}
		if lanes == 0 && foundation.Equal(type_, element) {
			args[i] = value
			continue
		}
		return 0, nil, diag(x.Args[i].GetSpan(), "%s argument is %s, want %s", kind, type_, vector)
	}
	out := element
	if !rule.ResultElement && lanes > 0 {
		out = vector
	}
	if expected != nil && !foundation.Equal(out, expected) {
		return 0, nil, diag(x.Span, "%s returns %s, context requires %s", kind, out, expected)
	}
	result := b.value()
	b.emit(&ir.Intrinsic{Result: result, Type: out, Kind: kind, Args: args, Span: x.Span})
	return result, out, nil
}

func (c *analyzer) lowerMaskIntrinsic(b *fnBuilder, e env, x *parser.CallExpr, kind ir.IntrinsicKind, expected *foundation.Type) (ir.ValueID, *foundation.Type, error) {
	if kind == ir.IntrinsicAll || kind == ir.IntrinsicAny {
		if len(x.Args) != 1 {
			return 0, nil, diag(x.Span, "%s expects one argument, got %d", kind, len(x.Args))
		}
		mask, maskType, err := c.lowerExpr(b, e, x.Args[0], nil)
		if err != nil {
			return 0, nil, err
		}
		if maskType.Kind != foundation.VectorKind || maskType.Elem.Kind != foundation.BoolKind {
			return 0, nil, diag(x.Args[0].GetSpan(), "%s requires vec<bool, N>, got %s", kind, maskType)
		}
		if expected != nil && !foundation.Equal(expected, foundation.BoolType) {
			return 0, nil, diag(x.Span, "%s returns bool, context requires %s", kind, expected)
		}
		result := b.value()
		b.emit(&ir.Intrinsic{Result: result, Type: foundation.BoolType, Kind: kind, Args: []ir.ValueID{mask}, Span: x.Span})
		return result, foundation.BoolType, nil
	}
	if len(x.Args) != 3 {
		return 0, nil, diag(x.Span, "select expects mask, whenTrue, and whenFalse arguments; got %d", len(x.Args))
	}
	mask, maskType, err := c.lowerExpr(b, e, x.Args[0], nil)
	if err != nil {
		return 0, nil, err
	}
	if maskType.Kind != foundation.VectorKind || maskType.Elem.Kind != foundation.BoolKind {
		if foundation.Equal(maskType, foundation.BoolType) {
			return 0, nil, diagHelp(x.Args[0].GetSpan(), "use condition ? whenTrue : whenFalse for scalar choice; select is lane-wise", "select mask must be vec<bool, N>, got %s", maskType)
		}
		return 0, nil, diag(x.Args[0].GetSpan(), "select mask must be vec<bool, N>, got %s", maskType)
	}
	arms := make([]detachedExpr, 2)
	for i := range arms {
		arms[i], err = c.lowerDetached(b, e, x.Args[i+1], nil)
		if err != nil {
			return 0, nil, err
		}
	}
	var out *foundation.Type
	if foundation.IsBoolean(arms[0].type_) || foundation.IsBoolean(arms[1].type_) {
		for i, arm := range arms {
			if !foundation.IsBoolean(arm.type_) || arm.type_.Kind == foundation.VectorKind && arm.type_.Lanes != maskType.Lanes {
				return 0, nil, diag(x.Args[i+1].GetSpan(), "select boolean arm is %s, want bool or %s", arm.type_, maskType)
			}
		}
		out = maskType
	} else {
		var element *foundation.Type
		if expected != nil {
			if expected.Kind != foundation.VectorKind || expected.Lanes != maskType.Lanes || !foundation.IsNumericScalar(expected.Elem) {
				return 0, nil, diag(x.Span, "select produces a %d-lane vector, context requires %s", maskType.Lanes, expected)
			}
			element = expected.Elem
		}
		element, _, err = resolveNumericOperands("select", ir.NumericAny, arms, element, maskType.Lanes)
		if err != nil {
			return 0, nil, diag(x.Span, "%v", err)
		}
		out = foundation.VectorOf(element, maskType.Lanes)
	}
	if expected != nil && !foundation.Equal(expected, out) {
		return 0, nil, diag(x.Span, "select returns %s, context requires %s", out, expected)
	}
	args := []ir.ValueID{mask, 0, 0}
	for i, arm := range arms {
		want := out
		if arm.type_.Kind != foundation.VectorKind {
			want = out.Elem
		}
		value, armType, err := c.commitArgument(b, e, arm, want)
		if err != nil {
			return 0, nil, err
		}
		if foundation.Equal(armType, out.Elem) {
			value, armType = c.splat(b, value, out, x.Args[i+1].GetSpan()), out
		}
		if !foundation.Equal(armType, out) {
			return 0, nil, diag(x.Args[i+1].GetSpan(), "select arm is %s, want %s or %s", armType, out.Elem, out)
		}
		args[i+1] = value
	}
	result := b.value()
	b.emit(&ir.Intrinsic{Result: result, Type: out, Kind: kind, Args: args, Span: x.Span})
	return result, out, nil
}

func (c *analyzer) lowerVectorInference(b *fnBuilder, e env, x *parser.CallExpr, expected *foundation.Type) (ir.ValueID, *foundation.Type, error) {
	if len(x.Args) == 0 {
		return 0, nil, diag(x.Span, "vec requires components")
	}
	arguments := make([]detachedExpr, len(x.Args))
	for i, expression := range x.Args {
		argument, err := c.lowerDetached(b, e, expression, nil)
		if err != nil {
			return 0, nil, err
		}
		if element, _ := scalarElement(argument.type_); element == nil {
			return 0, nil, diag(expression.GetSpan(), "vec requires scalar or vector components, got %s", argument.type_)
		}
		arguments[i] = argument
	}
	lanes := 0
	for _, argument := range arguments {
		if _, width := scalarElement(argument.type_); width == 0 {
			lanes++
		} else {
			lanes += width
		}
	}
	if lanes < 2 || lanes > 4 {
		return 0, nil, diag(x.Span, "vec received %d lanes, want 2, 3, or 4", lanes)
	}

	var element *foundation.Type
	if expected != nil {
		if expected.Kind != foundation.VectorKind || expected.Lanes != lanes || !foundation.IsScalar(expected.Elem) {
			return 0, nil, diag(x.Span, "vec produces a %d-lane vector, context requires %s", lanes, expected)
		}
		element = expected.Elem
	}
	for i, argument := range arguments {
		argumentElement, _ := scalarElement(argument.type_)
		if argument.contextual {
			continue
		}
		if element != nil && !foundation.Equal(element, argumentElement) {
			return 0, nil, diag(x.Args[i].GetSpan(), "vec components use %s and %s; convert explicitly", element, argumentElement)
		}
		element = argumentElement
	}
	if element == nil {
		element = defaultNumericElement(ir.NumericAny, arguments)
	}
	vector := foundation.VectorOf(element, lanes)
	values := make([]ir.ValueID, 0, lanes)
	for i, argument := range arguments {
		_, width := scalarElement(argument.type_)
		want := element
		if width > 0 {
			want = foundation.VectorOf(element, width)
		}
		base, type_, err := c.commitArgument(b, e, argument, want)
		if err != nil {
			return 0, nil, err
		}
		if foundation.Equal(type_, element) {
			values = append(values, base)
			continue
		}
		if type_.Kind != foundation.VectorKind || !foundation.Equal(type_.Elem, element) || type_.Lanes != width {
			return 0, nil, diag(x.Args[i].GetSpan(), "vec component is %s, want %s", type_, want)
		}
		for lane := range width {
			component := b.value()
			b.emit(&ir.Extract{Result: component, Type: element, Base: base, Index: lane, Span: x.Args[i].GetSpan()})
			values = append(values, component)
		}
	}
	result := b.value()
	b.emit(&ir.Composite{Result: result, Type: vector, Values: values, Span: x.Span})
	return result, vector, nil
}
