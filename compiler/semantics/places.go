package semantics

import (
	"fmt"
	"sort"
	"strings"
	"tach/foundation"
	"tach/ir"
	"tach/parser"
)

func (c *analyzer) lowerPlace(b *fnBuilder, e env, x parser.Expr) (ir.PlaceID, *foundation.Type, error) {
	switch v := x.(type) {
	case *parser.IdentExpr:
		s, ok := e.syms[v.Name]
		if !ok || (s.buffer < 0 && s.workgroup < 0) {
			return 0, nil, diag(v.Span, "%s is not an addressable GPU place", v.Name)
		}
		p := b.place()
		if s.buffer >= 0 {
			b.emit(&ir.PlaceRoot{Result: p, Type: s.ty, Buffer: s.buffer, Span: v.Span})
		} else {
			b.emit(&ir.PlaceWorkgroup{Result: p, Type: s.ty, Workgroup: s.workgroup, Span: v.Span})
		}
		return p, s.ty, nil
	case *parser.MemberExpr:
		bp, bt, err := c.lowerPlace(b, e, v.Base)
		if err != nil {
			return 0, nil, err
		}
		if bt.Kind == foundation.VectorKind {
			components, ok := vectorComponents(v.Name, bt.Lanes)
			if !ok || len(components) != 1 {
				return 0, nil, diag(v.Span, "vector %s has no component %s", bt, v.Name)
			}
			iv := b.value()
			b.emit(&ir.Const{Result: iv, Type: foundation.Uint32Type, Raw: fmt.Sprintf("%d", components[0]), Span: v.Span})
			p := b.place()
			b.emit(&ir.PlaceIndex{Result: p, Type: bt.Elem, Base: bp, Index: iv, Span: v.Span})
			return p, bt.Elem, nil
		}
		if bt.Kind != foundation.StructKind {
			return 0, nil, diag(v.Span, "field access requires struct/vector place, got %s", bt)
		}
		idx := foundation.FieldIndex(bt, v.Name)
		if idx < 0 {
			return 0, nil, diag(v.Span, "type %s has no field %s", bt, v.Name)
		}
		ft := bt.Fields[idx].Type
		p := b.place()
		b.emit(&ir.PlaceField{Result: p, Type: ft, Base: bp, Field: idx, Span: v.Span})
		return p, ft, nil
	case *parser.IndexExpr:
		bp, bt, err := c.lowerPlace(b, e, v.Base)
		if err != nil {
			return 0, nil, err
		}
		if bt.Kind != foundation.RuntimeArrayKind && bt.Kind != foundation.FixedArrayKind && bt.Kind != foundation.VectorKind {
			return 0, nil, diag(v.Span, "indexing requires an array place, got %s", bt)
		}
		iv, it, err := c.lowerExpr(b, e, v.Index, foundation.Uint32Type)
		if err != nil {
			return 0, nil, err
		}
		if !foundation.Equal(it, foundation.Uint32Type) && !foundation.Equal(it, foundation.Int32Type) {
			return 0, nil, diag(v.Index.GetSpan(), "array index must be int32 or uint32, got %s", it)
		}
		p := b.place()
		b.emit(&ir.PlaceIndex{Result: p, Type: bt.Elem, Base: bp, Index: iv, Span: v.Span})
		return p, bt.Elem, nil
	default:
		return 0, nil, diag(x.GetSpan(), "expression is not an addressable GPU place")
	}
}
func vectorComponents(name string, lanes int) ([]int, bool) {
	if len(name) < 1 || len(name) > 4 {
		return nil, false
	}
	out := make([]int, len(name))
	for index, component := range []byte(name) {
		switch component {
		case 'x':
			out[index] = 0
		case 'y':
			out[index] = 1
		case 'z':
			out[index] = 2
		case 'w':
			out[index] = 3
		default:
			return nil, false
		}
		if out[index] >= lanes {
			return nil, false
		}
	}
	return out, true
}
func diag(span foundation.Span, f string, a ...any) error {
	return &foundation.Diagnostic{Span: span, Message: fmt.Sprintf(f, a...)}
}

func diagHelp(span foundation.Span, help, f string, a ...any) error {
	return &foundation.Diagnostic{Span: span, Message: fmt.Sprintf(f, a...), Help: help}
}

func boolDiag(span foundation.Span, subject string, t *foundation.Type, maskHelp string) error {
	if t.Kind != foundation.VectorKind || t.Elem.Kind != foundation.BoolKind {
		maskHelp = ""
	}
	return diagHelp(span, maskHelp, "%s is %s, want bool", subject, t)
}

func checkRecursion(m *ir.KernelModule) error {
	graph := map[string][]string{}
	for _, f := range m.Functions {
		var walk func(*ir.Block)
		walk = func(b *ir.Block) {
			for _, in := range b.Instrs {
				switch x := in.(type) {
				case *ir.Call:
					graph[f.Name] = append(graph[f.Name], x.Function)
				case *ir.If:
					walk(x.Then)
					walk(x.Else)
				case *ir.Loop:
					walk(x.Cond)
					walk(x.Body)
				}
			}
		}
		walk(f.Body)
	}
	state := map[string]uint8{}
	stack := []string{}
	var visit func(string) error
	visit = func(n string) error {
		if state[n] == 1 {
			return diag(m.Function(n).Span, "recursive function cycle is not allowed: %s -> %s", strings.Join(stack, " -> "), n)
		}
		if state[n] == 2 {
			return nil
		}
		state[n] = 1
		stack = append(stack, n)
		for _, d := range graph[n] {
			if err := visit(d); err != nil {
				return err
			}
		}
		stack = stack[:len(stack)-1]
		state[n] = 2
		return nil
	}
	nodes := make([]string, 0, len(graph))
	for n := range graph {
		nodes = append(nodes, n)
		sort.Strings(graph[n])
	}
	sort.Strings(nodes)
	for _, n := range nodes {
		if err := visit(n); err != nil {
			return err
		}
	}
	return nil
}
