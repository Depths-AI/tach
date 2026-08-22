package semantics

import (
	"fmt"
	"sort"
	"strings"
	"tach/foundation"
	"tach/ir"
	"tach/parser"
)

func (c *analyzer) lowerBlock(b *fnBuilder, src *parser.BlockStmt, e env) error {
	for _, s := range src.Stmts {
		if b.block.Term != nil {
			return diag(s.GetSpan(), "unreachable statement")
		}
		if err := c.lowerStmt(b, e, s); err != nil {
			return err
		}
	}
	return nil
}
func (c *analyzer) lowerStmt(b *fnBuilder, e env, s parser.Stmt) error {
	switch x := s.(type) {
	case *parser.WorkgroupStmt:
		if b.fn.Kind != ir.Stage {
			return diag(x.Span, "shared variables are only valid inside kernels")
		}
		if !b.top {
			return diag(x.Span, "shared variables must be declared in the kernel body, not a nested block")
		}
		if _, ok := e.syms[x.Name]; ok {
			return diag(x.Span, "%q is already defined in this scope", x.Name)
		}
		ty, err := c.resolveTypeIn(x.Type, &e)
		if err != nil {
			return err
		}
		if !foundation.IsWorkgroupStorable(ty) {
			if foundation.Contains(ty, foundation.BoolKind) {
				return diagHelp(x.Span, "store uint32 flags and derive boolean masks after loading them", "shared variable %s has invalid type %s", x.Name, ty)
			}
			return diag(x.Span, "shared variable %s has invalid type %s", x.Name, ty)
		}
		idx := len(b.fn.WorkgroupVars)
		b.fn.WorkgroupVars = append(b.fn.WorkgroupVars, ir.WorkgroupVar{Name: x.Name, Type: ty, Span: x.Span})
		e.syms[x.Name] = symbol{ty: ty, buffer: -1, workgroup: idx}
		return nil
	case *parser.ConstStmt:
		if _, ok := e.syms[x.Name]; ok {
			return diag(x.Span, "%q is already defined in this scope", x.Name)
		}
		value, err := c.evaluateConstantBinding(x.Type, x.Value, e)
		if err != nil {
			return err
		}
		e.syms[x.Name] = symbol{ty: value.Type, constant: value, buffer: -1, workgroup: -1}
		return nil
	case *parser.VarStmt:
		if _, ok := e.syms[x.Name]; ok {
			return diag(x.Span, "%q is already defined in this scope", x.Name)
		}
		var expected *foundation.Type
		var err error
		if x.Type != nil {
			expected, err = c.resolveTypeIn(x.Type, &e)
			if err != nil {
				return err
			}
			if !foundation.IsConstructible(expected) {
				return diag(x.Span, "local %s has invalid type %s", x.Name, expected)
			}
		}
		id, t, err := c.lowerExpr(b, e, x.Value, expected)
		if err != nil {
			return err
		}
		if expected != nil && !foundation.Equal(t, expected) {
			return diag(x.Value.GetSpan(), "local value is %s, want %s", t, expected)
		}
		if t.Kind == foundation.VoidKind {
			return diag(x.Value.GetSpan(), "cannot bind a void expression")
		}
		e.syms[x.Name] = symbol{ty: t, value: id, mutable: true, buffer: -1, workgroup: -1}
		return nil
	case *parser.AssignStmt:
		return c.lowerAssign(b, e, x.Target, x.Op, x.Value, x.Span)
	case *parser.IncStmt:
		return c.lowerInc(b, e, x)
	case *parser.ExprStmt:
		_, _, err := c.lowerExpr(b, e, x.Expr, nil)
		return err
	case *parser.ReturnStmt:
		if b.fn.Return.Kind == foundation.VoidKind {
			if x.Value != nil {
				return diag(x.Span, "void function cannot return a value")
			}
			b.block.Term = &ir.Return{}
			return nil
		}
		if x.Value == nil {
			return diag(x.Span, "function must return %s", b.fn.Return)
		}
		v, t, err := c.lowerExpr(b, e, x.Value, b.fn.Return)
		if err != nil {
			return err
		}
		if !foundation.Equal(t, b.fn.Return) {
			return diag(x.Span, "return value is %s, want %s", t, b.fn.Return)
		}
		b.block.Term = &ir.Return{Value: v, HasValue: true}
		return nil
	case *parser.IfStmt:
		return c.lowerIf(b, e, x)
	case *parser.WhileStmt:
		return c.lowerWhile(b, e, x)
	case *parser.ForStmt:
		return c.lowerFor(b, e, x)
	case *parser.BreakStmt:
		if b.loop == nil {
			return diag(x.Span, "break is only valid inside a loop")
		}
		b.block.Term = &ir.Break{Values: loopValues(b.loop.names, e)}
		return nil
	case *parser.ContinueStmt:
		if b.loop == nil {
			return diag(x.Span, "continue is only valid inside a loop")
		}
		return c.continueLoop(b, e)
	default:
		return fmt.Errorf("unknown statement %T", s)
	}
}
func (c *analyzer) lowerAssign(b *fnBuilder, e env, target parser.Expr, op string, rhs parser.Expr, span foundation.Span) error {
	if id, ok := target.(*parser.IdentExpr); ok {
		sym, exists := e.syms[id.Name]
		if exists && sym.buffer < 0 && sym.workgroup < 0 {
			if !sym.mutable {
				if sym.constant != nil {
					return diag(target.GetSpan(), "cannot assign to compile-time constant %s", id.Name)
				}
				return diag(target.GetSpan(), "cannot assign to immutable value %s", id.Name)
			}
			var nv ir.ValueID
			var nt *foundation.Type
			var err error
			if op == "=" {
				nv, nt, err = c.lowerExpr(b, e, rhs, sym.ty)
			} else {
				nv, nt, err = c.lowerCompound(b, e, strings.TrimSuffix(op, "="), sym.value, sym.ty, rhs, span)
			}
			if err != nil {
				return err
			}
			if !foundation.Equal(nt, sym.ty) {
				return diag(rhs.GetSpan(), "assignment value is %s, want %s", nt, sym.ty)
			}
			sym.value = nv
			e.syms[id.Name] = sym
			return nil
		}
		if c.consts[id.Name] != nil && c.visible(id.Name, id.Span.File) {
			return diag(target.GetSpan(), "cannot assign to compile-time constant %s", id.Name)
		}
	}
	p, pt, err := c.lowerPlace(b, e, target)
	if err != nil {
		return err
	}
	var v ir.ValueID
	var vt *foundation.Type
	if op == "=" {
		v, vt, err = c.lowerExpr(b, e, rhs, pt)
	} else {
		old := b.value()
		b.emit(&ir.Load{Result: old, Type: pt, Place: p, Span: target.GetSpan()})
		v, vt, err = c.lowerCompound(b, e, strings.TrimSuffix(op, "="), old, pt, rhs, span)
	}
	if err != nil {
		return err
	}
	if !foundation.Equal(vt, pt) {
		return diag(rhs.GetSpan(), "store value is %s, want %s", vt, pt)
	}
	b.emit(&ir.Store{Place: p, Value: v, Span: span})
	return nil
}
func (c *analyzer) lowerInc(b *fnBuilder, e env, x *parser.IncStmt) error {
	raw := "1"
	op := "+"
	if x.Delta < 0 {
		op = "-"
	}
	one := &parser.NumberExpr{Raw: raw, Span: x.Span}
	return c.lowerAssign(b, e, x.Target, op+"=", one, x.Span)
}

func assigned(block *parser.BlockStmt, out map[string]bool) {
	for _, s := range block.Stmts {
		switch x := s.(type) {
		case *parser.AssignStmt:
			if id, ok := x.Target.(*parser.IdentExpr); ok {
				out[id.Name] = true
			}
		case *parser.IncStmt:
			if id, ok := x.Target.(*parser.IdentExpr); ok {
				out[id.Name] = true
			}
		case *parser.IfStmt:
			assigned(x.Then, out)
			if x.Else != nil {
				assigned(x.Else, out)
			}
		case *parser.WhileStmt:
			assigned(x.Body, out)
		case *parser.ForStmt:
			assigned(x.Body, out)
			assigned(&parser.BlockStmt{Stmts: []parser.Stmt{x.Post}}, out)
		}
	}
}
func carriedNames(blocks []*parser.BlockStmt, e env) []string {
	set := map[string]bool{}
	for _, bl := range blocks {
		if bl != nil {
			assigned(bl, set)
		}
	}
	var out []string
	for name, s := range e.syms {
		if set[name] && s.buffer < 0 && s.workgroup < 0 && s.mutable {
			out = append(out, name)
		}
	}
	sort.Strings(out)
	return out
}

func (c *analyzer) lowerIf(b *fnBuilder, e env, x *parser.IfStmt) error {
	cond, ct, err := c.lowerExpr(b, e, x.Cond, foundation.BoolType)
	if err != nil {
		return err
	}
	if !foundation.Equal(ct, foundation.BoolType) {
		return boolDiag(x.Cond.GetSpan(), "if condition", ct, "reduce the mask with all(mask) or any(mask)")
	}
	names := carriedNames([]*parser.BlockStmt{x.Then, x.Else}, e)
	thenBlock := &ir.Block{}
	tb := b.child(thenBlock)
	te := e.clone()
	if err := c.lowerBlock(tb, x.Then, te); err != nil {
		return err
	}
	elseBlock := &ir.Block{}
	eb := b.child(elseBlock)
	ee := e.clone()
	if x.Else != nil {
		if err := c.lowerBlock(eb, x.Else, ee); err != nil {
			return err
		}
	}
	thenFalls := thenBlock.Term == nil
	elseFalls := elseBlock.Term == nil
	results := make([]ir.Result, 0, len(names))
	for _, name := range names {
		sym := e.syms[name]
		id := b.value()
		results = append(results, ir.Result{ID: id, Type: sym.ty})
	}
	if thenFalls {
		vals := make([]ir.ValueID, len(names))
		for i, n := range names {
			vals[i] = te.syms[n].value
		}
		thenBlock.Term = &ir.Yield{Values: vals}
	}
	if elseFalls {
		vals := make([]ir.ValueID, len(names))
		for i, n := range names {
			vals[i] = ee.syms[n].value
		}
		elseBlock.Term = &ir.Yield{Values: vals}
	}
	if !thenFalls && !elseFalls {
		results = nil
	}
	b.emit(&ir.If{Results: results, Cond: cond, Then: thenBlock, Else: elseBlock, Span: x.Span})
	if !thenFalls && !elseFalls {
		b.block.Term = &ir.Unreachable{}
		return nil
	}
	for i, n := range names {
		sym := e.syms[n]
		sym.value = results[i].ID
		e.syms[n] = sym
	}
	return nil
}
func loopValues(names []string, e env) []ir.ValueID {
	values := make([]ir.ValueID, len(names))
	for i, name := range names {
		values[i] = e.syms[name].value
	}
	return values
}

func (c *analyzer) continueLoop(b *fnBuilder, e env) error {
	loop := b.loop
	if loop.post != nil {
		postEnv := loop.base.clone()
		for _, name := range loop.names {
			symbol := postEnv.syms[name]
			symbol.value = e.syms[name].value
			postEnv.syms[name] = symbol
		}
		if err := c.lowerStmt(b, postEnv, loop.post); err != nil {
			return err
		}
		e = postEnv
	}
	b.block.Term = &ir.Continue{Values: loopValues(loop.names, e)}
	return nil
}

func (c *analyzer) lowerLoop(b *fnBuilder, e env, cond parser.Expr, body *parser.BlockStmt, post parser.Stmt, span foundation.Span) error {
	blocks := []*parser.BlockStmt{body}
	if post != nil {
		blocks = append(blocks, &parser.BlockStmt{Stmts: []parser.Stmt{post}})
	}
	names := carriedNames(blocks, e)
	params := make([]ir.LoopParam, len(names))
	results := make([]ir.Result, len(names))
	loopEnv := e.clone()
	for i, n := range names {
		sym := e.syms[n]
		pid := b.value()
		rid := b.value()
		params[i] = ir.LoopParam{ID: pid, Type: sym.ty, Init: sym.value}
		results[i] = ir.Result{ID: rid, Type: sym.ty}
		sym.value = pid
		loopEnv.syms[n] = sym
	}
	condBlock := &ir.Block{}
	cb := b.child(condBlock)
	cv, ct, err := c.lowerExpr(cb, loopEnv, cond, foundation.BoolType)
	if err != nil {
		return err
	}
	if !foundation.Equal(ct, foundation.BoolType) {
		return boolDiag(cond.GetSpan(), "loop condition", ct, "reduce the mask with all(mask) or any(mask)")
	}
	condBlock.Term = &ir.Yield{Values: []ir.ValueID{cv}}
	bodyBlock := &ir.Block{}
	bb := b.child(bodyBlock)
	bb.loop = &loopContext{names: names, base: loopEnv, post: post}
	bodyEnv := loopEnv.clone()
	if err := c.lowerBlock(bb, body, bodyEnv); err != nil {
		return err
	}
	if bodyBlock.Term == nil {
		if err := c.continueLoop(bb, bodyEnv); err != nil {
			return err
		}
	}
	b.emit(&ir.Loop{Results: results, Params: params, Cond: condBlock, Body: bodyBlock, Span: span})
	for i, n := range names {
		sym := e.syms[n]
		sym.value = results[i].ID
		e.syms[n] = sym
	}
	return nil
}

func (c *analyzer) lowerWhile(b *fnBuilder, e env, x *parser.WhileStmt) error {
	return c.lowerLoop(b, e, x.Cond, x.Body, nil, x.Span)
}

func (c *analyzer) lowerFor(b *fnBuilder, e env, x *parser.ForStmt) error {
	// A Tach for-loop is a source-level convenience only. It lowers into the
	// exact same structured, loop-carried SSA form as while.
	loopEnv := e.clone()
	if err := c.lowerStmt(b, loopEnv, x.Init); err != nil {
		return err
	}
	if err := c.lowerLoop(b, loopEnv, x.Cond, x.Body, x.Post, x.Span); err != nil {
		return err
	}
	// The initializer is lexically scoped to the for-loop. Mutations to symbols
	// that existed before the loop remain visible after it through loop results.
	for name, outer := range e.syms {
		inner, ok := loopEnv.syms[name]
		if !ok || outer.buffer >= 0 || outer.workgroup >= 0 {
			continue
		}
		outer.value = inner.value
		e.syms[name] = outer
	}
	return nil
}
