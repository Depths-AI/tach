package semantics

import (
	"tach/foundation"
	"tach/ir"
	"tach/parser"
)

func (c *analyzer) lowerCall(b *fnBuilder, e env, x *parser.CallExpr, expected *foundation.Type) (ir.ValueID, *foundation.Type, error) {
	id, ok := x.Callee.(*parser.IdentExpr)
	if !ok {
		return 0, nil, diag(x.Callee.GetSpan(), "call target must be a function or type name")
	}
	if target := foundation.ParseBuiltin(id.Name); target != nil && target.Kind != foundation.VoidKind && target.Kind != foundation.BoolKind {
		return c.lowerConstructor(b, e, x, target)
	}
	if id.Name == "vec" {
		return c.lowerVectorInference(b, e, x, expected)
	}
	if kind, ok := intrinsicBuiltin(id.Name); ok {
		if kind == ir.IntrinsicAll || kind == ir.IntrinsicAny || kind == ir.IntrinsicSelect {
			return c.lowerMaskIntrinsic(b, e, x, kind, expected)
		}
		return c.lowerIntrinsic(b, e, x, kind, expected)
	}
	if b.comptime {
		var err error = diagHelp(x.Span, "use let inside a function or program when the binding is computed at runtime", "call to %q is not available in compile-time expressions", id.Name)
		if id.Name == "ceilDiv" {
			err = &runtimeConstantDependency{err}
		}
		return 0, nil, err
	}
	if id.Name == "workgroupBarrier" || id.Name == "bufferBarrier" {
		if len(x.Args) != 0 {
			return 0, nil, diag(x.Span, "%s expects no arguments", id.Name)
		}
		if b.fn.Kind != ir.Stage {
			return 0, nil, diag(x.Span, "%s is only valid inside kernels", id.Name)
		}
		kind := ir.BarrierWorkgroup
		if id.Name == "bufferBarrier" {
			kind = ir.BarrierBuffer
		}
		b.emit(&ir.Barrier{Kind: kind, Span: x.Span})
		return 0, foundation.VoidType, nil
	}
	if op, atomic := atomicBuiltin(id.Name); atomic {
		want := 1
		if op == ir.AtomicCompareExchange {
			want = 3
		} else if op != ir.AtomicLoad {
			want = 2
		}
		if len(x.Args) != want {
			return 0, nil, diag(x.Span, "%s expects %d argument(s), got %d", id.Name, want, len(x.Args))
		}
		p, pt, err := c.lowerPlace(b, e, x.Args[0])
		if err != nil {
			return 0, nil, err
		}
		if pt.Kind != foundation.AtomicKind || (pt.Elem.Kind != foundation.Int32Kind && pt.Elem.Kind != foundation.Uint32Kind) {
			return 0, nil, diag(x.Args[0].GetSpan(), "%s requires an atomic<int32> or atomic<uint32> place, got %s", id.Name, pt)
		}
		var value, expected ir.ValueID
		for argument := 1; argument < want; argument++ {
			v, vt, err := c.lowerExpr(b, e, x.Args[argument], pt.Elem)
			if err != nil {
				return 0, nil, err
			}
			if !foundation.Equal(vt, pt.Elem) {
				return 0, nil, diag(x.Args[argument].GetSpan(), "%s value is %s, want %s", id.Name, vt, pt.Elem)
			}
			if op == ir.AtomicCompareExchange && argument == 1 {
				expected = v
			} else {
				value = v
			}
		}
		if op == ir.AtomicStore {
			b.emit(&ir.Atomic{Type: pt.Elem, Op: op, Place: p, Value: value, Span: x.Span})
			return 0, foundation.VoidType, nil
		}
		r := b.value()
		b.emit(&ir.Atomic{Result: r, Type: pt.Elem, Op: op, Place: p, Value: value, Expected: expected, Span: x.Span})
		return r, pt.Elem, nil
	}
	sig := c.funcs[id.Name]
	if sig != nil && !c.visible(id.Name, id.Span.File) {
		sig = nil
	}
	if sig == nil {
		return 0, nil, diag(id.Span, "unknown callable function %q", id.Name)
	}
	if sig.indexed {
		return 0, nil, diag(id.Span, "indexed stage %q cannot be called; use run from a public program", id.Name)
	}
	if sig.exported {
		return 0, nil, diag(id.Span, "public program %q cannot be called", id.Name)
	}
	if len(x.Args) != len(sig.params) {
		return 0, nil, diag(x.Span, "%s expects %d arguments, got %d", id.Name, len(sig.params), len(x.Args))
	}
	args := make([]ir.ValueID, len(x.Args))
	for i, a := range x.Args {
		v, t, err := c.lowerExpr(b, e, a, sig.params[i].ty)
		if err != nil {
			return 0, nil, err
		}
		if !foundation.Equal(t, sig.params[i].ty) {
			return 0, nil, diag(a.GetSpan(), "argument %d to %s is %s, want %s", i+1, id.Name, t, sig.params[i].ty)
		}
		args[i] = v
	}
	r := ir.ValueID(0)
	if sig.ret.Kind != foundation.VoidKind {
		r = b.value()
	}
	b.emit(&ir.Call{Result: r, Type: sig.ret, Function: id.Name, Args: args, Span: x.Span})
	return r, sig.ret, nil
}

func atomicBuiltin(name string) (ir.AtomicKind, bool) {
	switch name {
	case "atomicLoad":
		return ir.AtomicLoad, true
	case "atomicStore":
		return ir.AtomicStore, true
	case "atomicAdd":
		return ir.AtomicAdd, true
	case "atomicSub":
		return ir.AtomicSub, true
	case "atomicMin":
		return ir.AtomicMin, true
	case "atomicMax":
		return ir.AtomicMax, true
	case "atomicAnd":
		return ir.AtomicAnd, true
	case "atomicOr":
		return ir.AtomicOr, true
	case "atomicXor":
		return ir.AtomicXor, true
	case "atomicExchange":
		return ir.AtomicExchange, true
	case "atomicCompareExchange":
		return ir.AtomicCompareExchange, true
	default:
		return 0, false
	}
}
func (c *analyzer) lowerConstructor(b *fnBuilder, e env, x *parser.CallExpr, target *foundation.Type) (ir.ValueID, *foundation.Type, error) {
	if len(x.Args) != 1 {
		return 0, nil, diag(x.Span, "%s constructor expects one argument", target)
	}
	v, t, err := c.lowerExpr(b, e, x.Args[0], target)
	if err == nil && foundation.Equal(t, target) {
		return v, t, nil
	}
	if err != nil { // contextual literal failure may be meaningful; retry without context only for non-literals
		if _, ok := x.Args[0].(*parser.NumberExpr); ok {
			return 0, nil, err
		}
		v, t, err = c.lowerExpr(b, e, x.Args[0], nil)
		if err != nil {
			return 0, nil, err
		}
	}
	if !foundation.IsNumericScalar(t) {
		return 0, nil, diag(x.Args[0].GetSpan(), "cannot convert %s to %s", t, target)
	}
	r := b.value()
	b.emit(&ir.Convert{Result: r, Type: target, X: v, From: t, Span: x.Span})
	return r, target, nil
}

func (c *analyzer) splat(b *fnBuilder, value ir.ValueID, vector *foundation.Type, span foundation.Span) ir.ValueID {
	values := make([]ir.ValueID, vector.Lanes)
	for index := range values {
		values[index] = value
	}
	result := b.value()
	b.emit(&ir.Composite{Result: result, Type: vector, Values: values, Span: span})
	return result
}
