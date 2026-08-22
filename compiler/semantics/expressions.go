package semantics

import (
	"fmt"
	"math"
	"strconv"
	"strings"
	"tach/foundation"
	"tach/ir"
	"tach/parser"
)

func (c *analyzer) lowerExpr(b *fnBuilder, e env, x parser.Expr, expected *foundation.Type) (ir.ValueID, *foundation.Type, error) {
	switch v := x.(type) {
	case *parser.NumberExpr:
		return c.lowerNumber(b, v, expected)
	case *parser.BoolExpr:
		if expected != nil && !foundation.Equal(expected, foundation.BoolType) {
			return 0, nil, diag(v.Span, "bool literal cannot be used as %s", expected)
		}
		id := b.value()
		raw := "false"
		if v.Value {
			raw = "true"
		}
		b.emit(&ir.Const{Result: id, Type: foundation.BoolType, Raw: raw, Span: v.Span})
		return id, foundation.BoolType, nil
	case *parser.IdentExpr:
		if s, ok := e.syms[v.Name]; ok {
			if s.constant != nil {
				value, valueType := materializeConstant(b, s.constant, v.Span)
				return value, valueType, nil
			}
			if b.comptime {
				return 0, nil, &runtimeConstantDependency{diag(v.Span, "compile-time expression depends on runtime value %q; use let for the binding", v.Name)}
			}
			if s.buffer >= 0 {
				if !foundation.IsConstructible(s.ty) {
					return 0, nil, diag(v.Span, "runtime-sized resource %s must be accessed through its fixed fields or indexed tail", v.Name)
				}
				p := b.place()
				b.emit(&ir.PlaceRoot{Result: p, Type: s.ty, Buffer: s.buffer, Span: v.Span})
				id := b.value()
				b.emit(&ir.Load{Result: id, Type: s.ty, Place: p, Span: v.Span})
				return id, s.ty, nil
			}
			if s.workgroup >= 0 {
				if !foundation.IsConstructible(s.ty) {
					return 0, nil, diag(v.Span, "workgroup place %s of type %s must be indexed, field-accessed, or used by an atomic operation", v.Name, s.ty)
				}
				p := b.place()
				b.emit(&ir.PlaceWorkgroup{Result: p, Type: s.ty, Workgroup: s.workgroup, Span: v.Span})
				id := b.value()
				b.emit(&ir.Load{Result: id, Type: s.ty, Place: p, Span: v.Span})
				return id, s.ty, nil
			}
			return s.value, s.ty, nil
		}
		if definition := c.consts[v.Name]; definition != nil && c.visible(v.Name, v.Span.File) {
			value, err := c.resolveConstant(v.Name, v.Span)
			if err != nil {
				return 0, nil, err
			}
			result, resultType := materializeConstant(b, value, v.Span)
			return result, resultType, nil
		}
		return 0, nil, diag(v.Span, "unknown identifier %q", v.Name)
	case *parser.MemberExpr:
		if v.Name == "length" {
			if p, pt, err := c.lowerPlace(b, e, v.Base); err == nil && pt.Kind == foundation.RuntimeArrayKind {
				id := b.value()
				b.emit(&ir.ArrayLength{Result: id, Type: foundation.Uint32Type, Place: p, Span: v.Span})
				return id, foundation.Uint32Type, nil
			}
		}
		if p, pt, err := c.lowerPlace(b, e, v); err == nil {
			id := b.value()
			b.emit(&ir.Load{Result: id, Type: pt, Place: p, Span: v.Span})
			return id, pt, nil
		}
		base, bt, err := c.lowerExpr(b, e, v.Base, nil)
		if err != nil {
			return 0, nil, err
		}
		if bt.Kind == foundation.StructKind {
			idx := foundation.FieldIndex(bt, v.Name)
			if idx < 0 {
				return 0, nil, diag(v.Span, "type %s has no field %s", bt, v.Name)
			}
			ft := bt.Fields[idx].Type
			id := b.value()
			b.emit(&ir.Extract{Result: id, Type: ft, Base: base, Index: idx, Span: v.Span})
			return id, ft, nil
		}
		if bt.Kind == foundation.VectorKind {
			components, ok := vectorComponents(v.Name, bt.Lanes)
			if !ok {
				return 0, nil, diag(v.Span, "vector %s has no component %s", bt, v.Name)
			}
			values := make([]ir.ValueID, len(components))
			for index, component := range components {
				values[index] = b.value()
				b.emit(&ir.Extract{Result: values[index], Type: bt.Elem, Base: base, Index: component, Span: v.Span})
			}
			if len(values) == 1 {
				return values[0], bt.Elem, nil
			}
			resultType := foundation.VectorOf(bt.Elem, len(values))
			result := b.value()
			b.emit(&ir.Composite{Result: result, Type: resultType, Values: values, Span: v.Span})
			return result, resultType, nil
		}
		return 0, nil, diag(v.Span, "member access on %s", bt)
	case *parser.IndexExpr:
		if p, pt, err := c.lowerPlace(b, e, v); err == nil {
			id := b.value()
			b.emit(&ir.Load{Result: id, Type: pt, Place: p, Span: v.Span})
			return id, pt, nil
		}
		base, bt, err := c.lowerExpr(b, e, v.Base, nil)
		if err != nil {
			return 0, nil, err
		}
		if bt.Kind != foundation.VectorKind {
			return 0, nil, diag(v.Span, "indexing a value requires a vector, got %s", bt)
		}
		index, it, err := c.lowerExpr(b, e, v.Index, foundation.Uint32Type)
		if err != nil {
			return 0, nil, err
		}
		if !foundation.IsInteger(it) {
			return 0, nil, diag(v.Index.GetSpan(), "vector index must be int32 or uint32, got %s", it)
		}
		result := b.value()
		b.emit(&ir.VectorIndex{Result: result, Type: bt.Elem, Base: base, Index: index, Span: v.Span})
		return result, bt.Elem, nil
	case *parser.StructLiteralExpr:
		if expected == nil || expected.Kind != foundation.StructKind {
			return 0, nil, diag(v.Span, "struct literal requires a contextual struct type")
		}
		seen := map[string]parser.Expr{}
		for _, f := range v.Fields {
			if _, ok := seen[f.Name]; ok {
				return 0, nil, diag(f.Span, "duplicate struct literal field %s", f.Name)
			}
			seen[f.Name] = f.Value
		}
		vals := make([]ir.ValueID, len(expected.Fields))
		for i, f := range expected.Fields {
			ex := seen[f.Name]
			if ex == nil {
				return 0, nil, diag(v.Span, "missing field %s in %s literal", f.Name, expected)
			}
			id, t, err := c.lowerExpr(b, e, ex, f.Type)
			if err != nil {
				return 0, nil, err
			}
			if !foundation.Equal(t, f.Type) {
				return 0, nil, diag(ex.GetSpan(), "field %s is %s, want %s", f.Name, t, f.Type)
			}
			vals[i] = id
			delete(seen, f.Name)
		}
		for name, ex := range seen {
			return 0, nil, diag(ex.GetSpan(), "unknown field %s in %s literal", name, expected)
		}
		id := b.value()
		b.emit(&ir.Composite{Result: id, Type: expected, Values: vals, Span: v.Span})
		return id, expected, nil
	case *parser.UnaryExpr:
		want := expected
		if v.Op == "-" && want == nil {
			if n, ok := v.X.(*parser.NumberExpr); ok {
				want = foundation.Int32Type
				raw, basePrefixed := splitNumberLiteral(n.Raw)
				if !basePrefixed && strings.ContainsAny(raw, ".eE") {
					want = foundation.Float32Type
				}
			}
		}
		id, t, err := c.lowerExpr(b, e, v.X, want)
		if err != nil {
			return 0, nil, err
		}
		if v.Op == "!" && !foundation.IsBoolean(t) {
			return 0, nil, diag(v.Span, "! requires bool or vec<bool, N>")
		}
		if v.Op == "-" && !foundation.IsSignedNumeric(t) {
			return 0, nil, diag(v.Span, "unary - requires a signed numeric scalar or vector")
		}
		if v.Op == "~" && !foundation.IsIntegerLike(t) {
			return 0, nil, diag(v.Span, "unary ~ requires int32/uint32 or an integer vector")
		}
		r := b.value()
		b.emit(&ir.Unary{Result: r, Type: t, Op: v.Op, X: id, Span: v.Span})
		return r, t, nil
	case *parser.BinaryExpr:
		if v.Op == "&&" || v.Op == "||" {
			return c.lowerShortCircuit(b, e, v)
		}
		return c.lowerBinaryExpr(b, e, v, expected)
	case *parser.ConditionalExpr:
		cond, ct, err := c.lowerExpr(b, e, v.Cond, foundation.BoolType)
		if err != nil {
			return 0, nil, err
		}
		if !foundation.Equal(ct, foundation.BoolType) {
			return 0, nil, diag(v.Cond.GetSpan(), "conditional expression requires bool condition, got %s", ct)
		}
		expressions := []parser.Expr{v.Then, v.Else}
		arguments := make([]detachedExpr, len(expressions))
		expectedElement, expectedLanes := numericElement(expected)
		if expected == nil || expectedElement != nil {
			for i, expression := range expressions {
				arguments[i], err = c.lowerDetached(b, e.clone(), expression, nil)
				if err != nil {
					return 0, nil, err
				}
			}
			thenElement, _ := numericElement(arguments[0].type_)
			elseElement, _ := numericElement(arguments[1].type_)
			if thenElement != nil && elseElement != nil {
				expectedElement, expectedLanes, err = resolveNumericOperands("conditional", ir.NumericAny, arguments, expectedElement, expectedLanes)
				if err != nil {
					return 0, nil, diag(v.Span, "%v", err)
				}
				for i, argument := range arguments {
					_, lanes := numericElement(argument.type_)
					want := expectedElement
					if lanes > 0 {
						want = foundation.VectorOf(expectedElement, expectedLanes)
					}
					if argument.contextual {
						arguments[i], err = c.lowerDetached(b, e.clone(), expressions[i], want)
						if err != nil {
							return 0, nil, err
						}
					}
				}
			} else if expectedElement != nil {
				return 0, nil, diag(v.Span, "conditional branches cannot satisfy %s context", expected)
			}
		} else {
			for i, expression := range expressions {
				arguments[i], err = c.lowerDetached(b, e.clone(), expression, expected)
				if err != nil {
					return 0, nil, err
				}
			}
		}
		if !foundation.Equal(arguments[0].type_, arguments[1].type_) {
			return 0, nil, diag(v.Span, "conditional branches have types %s and %s", arguments[0].type_, arguments[1].type_)
		}
		if expected != nil && !foundation.Equal(arguments[0].type_, expected) {
			return 0, nil, diag(v.Span, "conditional result is %s, context requires %s", arguments[0].type_, expected)
		}
		thenBlock, elseBlock := arguments[0].block, arguments[1].block
		thenBlock.Term = &ir.Yield{Values: []ir.ValueID{arguments[0].value}}
		elseBlock.Term = &ir.Yield{Values: []ir.ValueID{arguments[1].value}}
		r := b.value()
		b.emit(&ir.If{Results: []ir.Result{{ID: r, Type: arguments[0].type_}}, Cond: cond, Then: thenBlock, Else: elseBlock, Span: v.Span})
		return r, arguments[0].type_, nil
	case *parser.CallExpr:
		return c.lowerCall(b, e, v, expected)
	case *parser.TransientExpr:
		return 0, nil, diag(v.Span, "transient allocation is only available as a public program let binding")
	default:
		return 0, nil, fmt.Errorf("unknown expression %T", x)
	}
}
func (c *analyzer) lowerNumber(b *fnBuilder, n *parser.NumberExpr, expected *foundation.Type) (ir.ValueID, *foundation.Type, error) {
	raw, basePrefixed := splitNumberLiteral(n.Raw)
	isFloatSpelling := !basePrefixed && strings.ContainsAny(raw, ".eE")

	var t *foundation.Type
	if expected != nil && foundation.IsNumericScalar(expected) {
		t = expected
	} else if isFloatSpelling {
		t = foundation.Float32Type
	} else {
		// Whole-number literals are naturally non-negative. Unary minus selects
		// int32 when no surrounding context supplies a type.
		t = foundation.Uint32Type
	}

	// Canonicalize literals in Core IR. Backends never need to understand Tach
	// digit separators or base prefixes.
	canonical := raw
	switch t.Kind {
	case foundation.Uint32Kind:
		if isFloatSpelling {
			return 0, nil, diag(n.Span, "uint32 literal must be an integer")
		}
		v, err := strconv.ParseUint(raw, 0, 32)
		if err != nil {
			return 0, nil, diag(n.Span, "uint32 literal out of range")
		}
		canonical = strconv.FormatUint(v, 10)
	case foundation.Int32Kind:
		if isFloatSpelling {
			return 0, nil, diag(n.Span, "int32 literal must be an integer")
		}
		// Base-prefixed literals are still value literals, not bit-pattern casts.
		// They therefore obey the positive int32 range exactly like decimal literals.
		v, err := strconv.ParseUint(raw, 0, 31)
		if err != nil {
			return 0, nil, diag(n.Span, "int32 literal out of range")
		}
		canonical = strconv.FormatUint(v, 10)
	case foundation.Float16Kind, foundation.Float32Kind:
		if basePrefixed {
			return 0, nil, diag(n.Span, "base-prefixed integer literal requires an integer context or explicit conversion")
		}
		bits := 32
		name := "float32"
		if t.Kind == foundation.Float16Kind {
			bits, name = 64, "float16"
		}
		f, err := strconv.ParseFloat(raw, bits)
		if err != nil || math.IsInf(f, 0) || math.IsNaN(f) {
			return 0, nil, diag(n.Span, "invalid %s literal", name)
		}
		if t.Kind == foundation.Float16Kind {
			if _, ok := foundation.Float16Bits(f); !ok {
				return 0, nil, diag(n.Span, "invalid float16 literal")
			}
		} else {
			f = float64(float32(f))
		}
		canonical = strconv.FormatFloat(f, 'g', -1, bits)
		if !strings.ContainsAny(canonical, ".eE") {
			canonical += ".0"
		}
	}
	id := b.value()
	b.emit(&ir.Const{Result: id, Type: t, Raw: canonical, Span: n.Span})
	return id, t, nil
}
