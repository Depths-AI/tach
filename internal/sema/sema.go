package sema

import (
	"fmt"
	"math"
	"strconv"
	"strings"

	"pine/internal/ast"
	"pine/internal/ir"
	"pine/internal/layout"
	"pine/internal/source"
	"pine/internal/types"
)

type Checker struct {
	ast       *ast.Module
	mod       *ir.Module
	types     map[string]*types.Type
	funcs     map[string]*funcSig
	resources map[[2]uint32]int
}

type funcSig struct {
	name    string
	params  []namedType
	ret     *types.Type
	decl    *ast.FuncDecl
	compute *ast.ComputeDecl
}
type namedType struct {
	name string
	ty   *types.Type
}

type symbol struct {
	name      string
	ty        *types.Type
	value     ir.ValueID
	mutable   bool
	resource  int // -1 unless this is a module resource place
	workgroup int // -1 unless this is a function workgroup place
}
type env struct{ syms map[string]symbol }

func newEnv() env { return env{syms: map[string]symbol{}} }
func (e env) clone() env {
	m := make(map[string]symbol, len(e.syms))
	for k, v := range e.syms {
		m[k] = v
	}
	return env{m}
}

type idAllocator struct {
	nextValue ir.ValueID
	nextPlace ir.PlaceID
}

type fnBuilder struct {
	c     *Checker
	fn    *ir.Function
	ids   *idAllocator
	block *ir.Block
}

func (b *fnBuilder) value() ir.ValueID {
	b.ids.nextValue++
	return b.ids.nextValue
}
func (b *fnBuilder) place() ir.PlaceID {
	b.ids.nextPlace++
	return b.ids.nextPlace
}
func (b *fnBuilder) emit(i ir.Instr) { b.block.Instrs = append(b.block.Instrs, i) }
func (b *fnBuilder) child(block *ir.Block) *fnBuilder {
	// Structured regions share one allocator. SSA/place IDs are function-global
	// identities even when definitions have region-scoped visibility.
	return &fnBuilder{c: b.c, fn: b.fn, ids: b.ids, block: block}
}
func (b *fnBuilder) sync(_ *fnBuilder) {}

func CheckAndLower(m *ast.Module) (*ir.Module, error) {
	c := &Checker{ast: m, mod: &ir.Module{}, types: map[string]*types.Type{}, funcs: map[string]*funcSig{}, resources: map[[2]uint32]int{}}
	for _, n := range []string{"void", "bool", "i32", "u32", "f32", "vec2f", "vec3f", "vec4f", "vec2u", "vec3u", "vec4u", "vec2i", "vec3i", "vec4i"} {
		c.types[n] = types.ParseBuiltin(n)
	}
	if err := c.collectTypes(); err != nil {
		return nil, err
	}
	if err := c.resolveTypeFields(); err != nil {
		return nil, err
	}
	if err := c.checkRuntimeArrayPlacement(); err != nil {
		return nil, err
	}
	if err := c.checkTypeCycles(); err != nil {
		return nil, err
	}
	if err := c.collectFunctions(); err != nil {
		return nil, err
	}
	if err := c.lowerFunctions(); err != nil {
		return nil, err
	}
	if err := checkRecursion(c.mod); err != nil {
		return nil, err
	}
	if err := ir.Verify(c.mod); err != nil {
		return nil, fmt.Errorf("internal IR verification failed: %w", err)
	}
	return c.mod, nil
}

func (c *Checker) collectTypes() error {
	for _, d := range c.ast.Decls {
		td, ok := d.(*ast.TypeDecl)
		if !ok {
			continue
		}
		if _, exists := c.types[td.Name]; exists {
			return diag(td.Span, "type %q is already defined", td.Name)
		}
		t := &types.Type{Kind: types.Struct, Name: td.Name}
		c.types[td.Name] = t
		c.mod.Structs = append(c.mod.Structs, t)
	}
	return nil
}
func (c *Checker) resolveTypeFields() error {
	for _, d := range c.ast.Decls {
		td, ok := d.(*ast.TypeDecl)
		if !ok {
			continue
		}
		t := c.types[td.Name]
		seen := map[string]bool{}
		for i, f := range td.Fields {
			if seen[f.Name] {
				return diag(f.Span, "duplicate field %q in %s", f.Name, td.Name)
			}
			seen[f.Name] = true
			ft, err := c.resolveType(f.Type, false)
			if err != nil {
				return err
			}
			if ft.Kind == types.Void {
				return diag(f.Span, "field %s.%s cannot be void", td.Name, f.Name)
			}
			if ft.Kind == types.RuntimeArray && i != len(td.Fields)-1 {
				return diag(f.Span, "runtime array must be the final field of a struct")
			}
			t.Fields = append(t.Fields, types.Field{Name: f.Name, Type: ft})
		}
	}
	return nil
}

func (c *Checker) checkRuntimeArrayPlacement() error {
	for _, t := range c.mod.Structs {
		for i, f := range t.Fields {
			if f.Type.Kind == types.RuntimeArray {
				if i != len(t.Fields)-1 {
					return fmt.Errorf("runtime array in %s must be the final member", t.Name)
				}
				if types.ContainsRuntimeArray(f.Type.Elem) {
					return fmt.Errorf("runtime array element %s in %s must have a fixed footprint", f.Type.Elem, t.Name)
				}
				continue
			}
			if types.ContainsRuntimeArray(f.Type) {
				return fmt.Errorf("%s.%s nests a runtime-sized structure; Pine follows WGSL's single trailing runtime-array rule", t.Name, f.Name)
			}
		}
	}
	return nil
}
func (c *Checker) checkTypeCycles() error {
	state := map[string]uint8{}
	var visit func(*types.Type) error
	visit = func(t *types.Type) error {
		if t.Kind == types.RuntimeArray {
			return visit(t.Elem)
		}
		if t.Kind != types.Struct {
			return nil
		}
		if state[t.Name] == 1 {
			return fmt.Errorf("recursive value type %s is not supported", t.Name)
		}
		if state[t.Name] == 2 {
			return nil
		}
		state[t.Name] = 1
		for _, f := range t.Fields {
			if err := visit(f.Type); err != nil {
				return err
			}
		}
		state[t.Name] = 2
		return nil
	}
	for _, t := range c.mod.Structs {
		if err := visit(t); err != nil {
			return err
		}
	}
	return nil
}
func (c *Checker) collectFunctions() error {
	for _, d := range c.ast.Decls {
		switch x := d.(type) {
		case *ast.FuncDecl:
			if isReservedCallable(x.Name) {
				return diag(x.Span, "function name %q is reserved by Pine", x.Name)
			}
			if _, ok := c.funcs[x.Name]; ok {
				return diag(x.Span, "function %q is already defined", x.Name)
			}
			sig := &funcSig{name: x.Name, decl: x, ret: types.TVoid}
			seen := map[string]bool{}
			for _, p := range x.Params {
				if seen[p.Name] {
					return diag(p.Span, "duplicate parameter %q", p.Name)
				}
				seen[p.Name] = true
				t, err := c.resolveType(p.Type, false)
				if err != nil {
					return err
				}
				if t.Kind == types.Void || !types.IsConstructible(t) {
					return diag(p.Span, "helper parameter %s has invalid value type %s", p.Name, t)
				}
				sig.params = append(sig.params, namedType{p.Name, t})
			}
			if x.Return != nil {
				r, err := c.resolveType(x.Return, false)
				if err != nil {
					return err
				}
				if r.Kind != types.Void && !types.IsConstructible(r) {
					return diag(x.Return.GetSpan(), "function cannot return non-constructible type %s", r)
				}
				sig.ret = r
			}
			c.funcs[x.Name] = sig
		case *ast.ComputeDecl:
			if isReservedCallable(x.Name) {
				return diag(x.Span, "compute name %q is reserved by Pine", x.Name)
			}
			if _, ok := c.funcs[x.Name]; ok {
				return diag(x.Span, "function %q is already defined", x.Name)
			}
			c.funcs[x.Name] = &funcSig{name: x.Name, compute: x, ret: types.TVoid}
		}
	}
	return nil
}
func (c *Checker) resolveType(te ast.TypeExpr, resource bool) (*types.Type, error) {
	switch t := te.(type) {
	case *ast.NamedType:
		x := c.types[t.Name]
		if x == nil {
			return nil, diag(t.Span, "unknown type %q", t.Name)
		}
		return x, nil
	case *ast.RuntimeArrayType:
		e, err := c.resolveType(t.Elem, false)
		if err != nil {
			return nil, err
		}
		if e.Kind == types.Void || e.Kind == types.RuntimeArray {
			return nil, diag(t.Span, "invalid runtime array element type %s", e)
		}
		return types.Runtime(e), nil
	case *ast.FixedArrayType:
		e, err := c.resolveType(t.Elem, false)
		if err != nil {
			return nil, err
		}
		if !types.IsWorkgroupStorable(e) {
			return nil, diag(t.Span, "invalid fixed array element type %s", e)
		}
		raw, suffix, _ := splitNumberLiteral(t.Count)
		if suffix == 'f' {
			return nil, diag(t.Span, "fixed array length must be a positive u32 constant")
		}
		n, err := strconv.ParseUint(raw, 0, 32)
		if err != nil || n == 0 {
			return nil, diag(t.Span, "fixed array length must be a positive u32 constant")
		}
		return types.Array(e, uint32(n)), nil
	case *ast.GenericType:
		if resource && (t.Name == "uniform" || t.Name == "storage") {
			return nil, diag(t.Span, "resource wrapper %s is only valid in compute parameter position", t.Name)
		}
		if t.Name == "atomic" {
			if len(t.Args) != 1 || t.Args[0].IsName {
				return nil, diag(t.Span, "atomic<T> takes exactly one type argument")
			}
			e, err := c.resolveType(t.Args[0].Type, false)
			if err != nil {
				return nil, err
			}
			if e.Kind != types.I32 && e.Kind != types.U32 {
				return nil, diag(t.Span, "atomic element type must be i32 or u32, got %s", e)
			}
			return types.AtomicOf(e), nil
		}
		return nil, diag(t.Span, "unknown generic type %q", t.Name)
	default:
		return nil, fmt.Errorf("unknown type expression %T", te)
	}
}

func (c *Checker) lowerFunctions() error {
	for _, d := range c.ast.Decls {
		switch x := d.(type) {
		case *ast.FuncDecl:
			if err := c.lowerHelper(x); err != nil {
				return err
			}
		case *ast.ComputeDecl:
			if err := c.lowerCompute(x); err != nil {
				return err
			}
		}
	}
	return nil
}
func (c *Checker) lowerHelper(d *ast.FuncDecl) error {
	sig := c.funcs[d.Name]
	f := &ir.Function{Name: d.Name, Return: sig.ret, Body: &ir.Block{}, Span: d.Span}
	e := newEnv()
	b := &fnBuilder{c: c, fn: f, ids: &idAllocator{}, block: f.Body}
	for _, p := range sig.params {
		id := b.value()
		f.Params = append(f.Params, ir.Param{Name: p.name, ID: id, Type: p.ty})
		e.syms[p.name] = symbol{name: p.name, ty: p.ty, value: id, resource: -1, workgroup: -1}
	}
	if err := c.lowerBlock(b, d.Body, e, "function"); err != nil {
		return err
	}
	if f.Body.Term == nil {
		if f.Return.Kind != types.Void {
			return diag(d.Body.Span, "function %s can reach the end without returning %s", d.Name, f.Return)
		}
		f.Body.Term = &ir.Return{}
	}
	c.mod.Functions = append(c.mod.Functions, f)
	return nil
}
func (c *Checker) lowerCompute(d *ast.ComputeDecl) error {
	wg, err := workgroup(d.Attrs)
	if err != nil {
		return err
	}
	f := &ir.Function{Name: d.Name, Return: types.TVoid, Body: &ir.Block{}, Compute: true, Workgroup: wg, Span: d.Span}
	e := newEnv()
	b := &fnBuilder{c: c, fn: f, ids: &idAllocator{}, block: f.Body}
	nextBinding := uint32(0)
	localBindings := map[[2]uint32]bool{}
	for _, p := range d.Params {
		kind, ty, access, err := c.resourceType(p.Type)
		if err != nil {
			return err
		}
		group, binding, explicit, err := bindingAttrs(p.Attrs)
		if err != nil {
			return err
		}
		if !explicit {
			group = 0
			// Automatic bindings are module-global. This keeps independently typed
			// resources from different entry points from accidentally acquiring the
			// same shader ABI address. Explicit compatible bindings may still be
			// shared intentionally across entry points.
			for {
				k := [2]uint32{group, nextBinding}
				_, globallyUsed := c.resources[k]
				if !localBindings[k] && !globallyUsed {
					binding = nextBinding
					nextBinding++
					break
				}
				nextBinding++
			}
		}
		key := [2]uint32{group, binding}
		if localBindings[key] {
			return diag(p.Span, "duplicate compute binding @group(%d) @binding(%d)", group, binding)
		}
		if _, used := e.syms[p.Name]; used {
			return diag(p.Span, "duplicate parameter %q", p.Name)
		}
		localBindings[key] = true
		idx, ok := c.resources[key]
		if ok {
			r := c.mod.Resources[idx]
			if r.Kind != kind || r.Access != access || !types.Equal(r.Type, ty) {
				return diag(p.Span, "binding @group(%d) @binding(%d) conflicts with resource %s from another entry point", group, binding, r.Name)
			}
		} else {
			name := p.Name
			idx = len(c.mod.Resources)
			c.resources[key] = idx
			c.mod.Resources = append(c.mod.Resources, ir.Resource{Name: name, Kind: kind, Type: ty, Access: access, Group: group, Binding: binding, Span: p.Span})
		}
		e.syms[p.Name] = symbol{name: p.Name, ty: ty, resource: idx, workgroup: -1}
		f.ResourceParams = append(f.ResourceParams, ir.ResourceParam{Name: p.Name, Resource: idx})
	}
	if err := c.lowerBlock(b, d.Body, e, "function"); err != nil {
		return err
	}
	if f.Body.Term == nil {
		f.Body.Term = &ir.Return{}
	}
	c.mod.Functions = append(c.mod.Functions, f)
	return nil
}
func (c *Checker) resourceType(te ast.TypeExpr) (ir.ResourceKind, *types.Type, ir.Access, error) {
	g, ok := te.(*ast.GenericType)
	if !ok || (g.Name != "uniform" && g.Name != "storage") {
		return 0, nil, 0, diag(te.GetSpan(), "compute parameters must be uniform<T> or storage<T, access>")
	}
	if len(g.Args) < 1 || len(g.Args) > 2 || g.Args[0].IsName {
		return 0, nil, 0, diag(g.Span, "invalid %s resource type", g.Name)
	}
	t, err := c.resolveType(g.Args[0].Type, false)
	if err != nil {
		return 0, nil, 0, err
	}
	if !types.IsHostShareable(t) {
		return 0, nil, 0, diag(g.Span, "resource type %s is not host-shareable", t)
	}
	if _, err := layout.Of(t); err != nil {
		return 0, nil, 0, diag(g.Span, "resource layout: %v", err)
	}
	if g.Name == "uniform" {
		if len(g.Args) != 1 {
			return 0, nil, 0, diag(g.Span, "uniform<T> takes exactly one type argument")
		}
		if t.Kind == types.RuntimeArray {
			return 0, nil, 0, diag(g.Span, "uniform resources cannot be runtime-sized")
		}
		if types.ContainsAtomic(t) {
			return 0, nil, 0, diag(g.Span, "uniform resources cannot contain atomic values")
		}
		return ir.Uniform, t, ir.Read, nil
	}
	acc := ir.Read
	if len(g.Args) == 2 {
		if !g.Args[1].IsName {
			return 0, nil, 0, diag(g.Args[1].Span, "storage access must be read or read_write")
		}
		switch g.Args[1].Name {
		case "read":
			acc = ir.Read
		case "read_write":
			acc = ir.ReadWrite
		default:
			return 0, nil, 0, diag(g.Args[1].Span, "unknown storage access %q", g.Args[1].Name)
		}
	}
	if types.ContainsAtomic(t) && acc != ir.ReadWrite {
		return 0, nil, 0, diag(g.Span, "storage resources containing atomic values require read_write access")
	}
	return ir.Storage, t, acc, nil
}

func workgroup(attrs []ast.Attribute) ([3]uint32, error) {
	var out = [3]uint32{1, 1, 1}
	found := false
	for _, a := range attrs {
		if a.Name != "workgroupSize" {
			return out, diag(a.Span, "unknown compute attribute @%s", a.Name)
		}
		if found {
			return out, diag(a.Span, "duplicate @workgroupSize")
		}
		found = true
		if len(a.Args) < 1 || len(a.Args) > 3 {
			return out, diag(a.Span, "@workgroupSize expects 1 to 3 integer arguments")
		}
		for i, e := range a.Args {
			v, err := constU32(e)
			if err != nil {
				return out, err
			}
			if v == 0 {
				return out, diag(e.GetSpan(), "workgroup dimension must be positive")
			}
			out[i] = v
		}
	}
	if !found {
		return out, fmt.Errorf("compute entry point requires @workgroupSize(...)")
	}
	return out, nil
}
func bindingAttrs(attrs []ast.Attribute) (group, binding uint32, explicit bool, err error) {
	gset, bset := false, false
	for _, a := range attrs {
		if a.Name != "group" && a.Name != "binding" {
			return 0, 0, false, diag(a.Span, "unknown resource attribute @%s", a.Name)
		}
		if len(a.Args) != 1 {
			return 0, 0, false, diag(a.Span, "@%s expects one integer argument", a.Name)
		}
		v, e := constU32(a.Args[0])
		if e != nil {
			return 0, 0, false, e
		}
		if a.Name == "group" {
			if gset {
				return 0, 0, false, diag(a.Span, "duplicate @group")
			}
			gset = true
			group = v
		} else {
			if bset {
				return 0, 0, false, diag(a.Span, "duplicate @binding")
			}
			bset = true
			binding = v
		}
	}
	if gset != bset {
		return 0, 0, false, fmt.Errorf("@group and @binding must be specified together")
	}
	return group, binding, gset, nil
}
func constU32(e ast.Expr) (uint32, error) {
	n, ok := e.(*ast.NumberExpr)
	if !ok {
		return 0, diag(e.GetSpan(), "expected integer literal")
	}
	s, suffix, basePrefixed := splitNumberLiteral(n.Raw)
	if suffix == 'f' || !basePrefixed && strings.ContainsAny(s, ".eE") {
		return 0, diag(e.GetSpan(), "expected integer literal")
	}
	v, err := strconv.ParseUint(s, 0, 32)
	if err != nil {
		return 0, diag(e.GetSpan(), "integer literal out of u32 range")
	}
	return uint32(v), nil
}

func splitNumberLiteral(raw string) (body string, suffix byte, basePrefixed bool) {
	body = strings.ReplaceAll(raw, "_", "")
	if len(body) > 0 {
		last := body[len(body)-1]
		if last == 'u' || last == 'i' || last == 'f' {
			suffix = last
			body = body[:len(body)-1]
		}
	}
	basePrefixed = strings.HasPrefix(body, "0x") || strings.HasPrefix(body, "0X") || strings.HasPrefix(body, "0b") || strings.HasPrefix(body, "0B")
	return body, suffix, basePrefixed
}

func (c *Checker) lowerBlock(b *fnBuilder, src *ast.BlockStmt, e env, kind string) error {
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
func (c *Checker) lowerStmt(b *fnBuilder, e env, s ast.Stmt) error {
	switch x := s.(type) {
	case *ast.WorkgroupStmt:
		if !b.fn.Compute {
			return diag(x.Span, "workgroup variables are only valid inside compute functions")
		}
		if _, ok := e.syms[x.Name]; ok {
			return diag(x.Span, "%q is already defined in this scope", x.Name)
		}
		ty, err := c.resolveType(x.Type, false)
		if err != nil {
			return err
		}
		if !types.IsWorkgroupStorable(ty) {
			return diag(x.Span, "workgroup variable %s has invalid type %s", x.Name, ty)
		}
		idx := len(b.fn.WorkgroupVars)
		b.fn.WorkgroupVars = append(b.fn.WorkgroupVars, ir.WorkgroupVar{Name: x.Name, Type: ty, Span: x.Span})
		e.syms[x.Name] = symbol{name: x.Name, ty: ty, resource: -1, workgroup: idx}
		return nil
	case *ast.VarStmt:
		if _, ok := e.syms[x.Name]; ok {
			return diag(x.Span, "%q is already defined in this scope", x.Name)
		}
		var expected *types.Type
		var err error
		if x.Type != nil {
			expected, err = c.resolveType(x.Type, false)
			if err != nil {
				return err
			}
			if !types.IsConstructible(expected) {
				return diag(x.Span, "local %s has invalid type %s", x.Name, expected)
			}
		}
		id, t, err := c.lowerExpr(b, e, x.Value, expected)
		if err != nil {
			return err
		}
		if t.Kind == types.Void {
			return diag(x.Value.GetSpan(), "cannot bind a void expression")
		}
		e.syms[x.Name] = symbol{name: x.Name, ty: t, value: id, mutable: x.Mutable, resource: -1, workgroup: -1}
		return nil
	case *ast.AssignStmt:
		return c.lowerAssign(b, e, x.Target, x.Op, x.Value, x.Span)
	case *ast.IncStmt:
		return c.lowerInc(b, e, x)
	case *ast.ExprStmt:
		_, _, err := c.lowerExpr(b, e, x.Expr, nil)
		return err
	case *ast.ReturnStmt:
		if b.fn.Return.Kind == types.Void {
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
		if !types.Equal(t, b.fn.Return) {
			return diag(x.Span, "return value is %s, want %s", t, b.fn.Return)
		}
		b.block.Term = &ir.Return{Value: v, HasValue: true}
		return nil
	case *ast.IfStmt:
		return c.lowerIf(b, e, x)
	case *ast.WhileStmt:
		return c.lowerWhile(b, e, x)
	case *ast.ForStmt:
		return c.lowerFor(b, e, x)
	default:
		return fmt.Errorf("unknown statement %T", s)
	}
}
func (c *Checker) lowerAssign(b *fnBuilder, e env, target ast.Expr, op string, rhs ast.Expr, span source.Span) error {
	if id, ok := target.(*ast.IdentExpr); ok {
		sym, exists := e.syms[id.Name]
		if exists && sym.resource < 0 && sym.workgroup < 0 {
			if !sym.mutable {
				return diag(target.GetSpan(), "cannot assign to const %s", id.Name)
			}
			var nv ir.ValueID
			var nt *types.Type
			var err error
			if op == "=" {
				nv, nt, err = c.lowerExpr(b, e, rhs, sym.ty)
			} else {
				binOp := strings.TrimSuffix(op, "=")
				want := sym.ty
				if binOp == "<<" || binOp == ">>" {
					want = types.ShiftCountType(sym.ty)
					if want == nil {
						return diag(target.GetSpan(), "%s requires an i32/u32 scalar or integer vector", binOp)
					}
				}
				rv, rt, e2 := c.lowerExpr(b, e, rhs, want)
				if e2 != nil {
					return e2
				}
				if !types.Equal(rt, want) {
					return diag(rhs.GetSpan(), "assignment value is %s, want %s", rt, want)
				}
				if binOp == "<<" || binOp == ">>" {
					rv, rt = c.normalizeShiftCount(b, rv, rt, rhs.GetSpan())
				}
				nv, nt, err = c.emitBinary(b, binOp, sym.value, sym.ty, rv, rt, span)
			}
			if err != nil {
				return err
			}
			if !types.Equal(nt, sym.ty) {
				return diag(rhs.GetSpan(), "assignment value is %s, want %s", nt, sym.ty)
			}
			sym.value = nv
			e.syms[id.Name] = sym
			return nil
		}
	}
	p, pt, err := c.lowerPlace(b, e, target)
	if err != nil {
		return err
	}
	var v ir.ValueID
	var vt *types.Type
	if op == "=" {
		v, vt, err = c.lowerExpr(b, e, rhs, pt)
	} else {
		old := b.value()
		b.emit(&ir.Load{Result: old, Type: pt, Place: p, Span: target.GetSpan()})
		binOp := strings.TrimSuffix(op, "=")
		want := pt
		if binOp == "<<" || binOp == ">>" {
			want = types.ShiftCountType(pt)
			if want == nil {
				return diag(target.GetSpan(), "%s requires an i32/u32 scalar or integer vector", binOp)
			}
		}
		rv, rt, e2 := c.lowerExpr(b, e, rhs, want)
		if e2 != nil {
			return e2
		}
		if binOp == "<<" || binOp == ">>" {
			rv, rt = c.normalizeShiftCount(b, rv, rt, rhs.GetSpan())
		}
		v, vt, err = c.emitBinary(b, binOp, old, pt, rv, rt, span)
	}
	if err != nil {
		return err
	}
	if !types.Equal(vt, pt) {
		return diag(rhs.GetSpan(), "store value is %s, want %s", vt, pt)
	}
	b.emit(&ir.Store{Place: p, Value: v, Span: span})
	return nil
}
func (c *Checker) lowerInc(b *fnBuilder, e env, x *ast.IncStmt) error {
	raw := "1"
	op := "+"
	if x.Delta < 0 {
		op = "-"
	}
	one := &ast.NumberExpr{Raw: raw, Span: x.Span}
	return c.lowerAssign(b, e, x.Target, op+"=", one, x.Span)
}

func assigned(block *ast.BlockStmt, out map[string]bool) {
	for _, s := range block.Stmts {
		switch x := s.(type) {
		case *ast.AssignStmt:
			if id, ok := x.Target.(*ast.IdentExpr); ok {
				out[id.Name] = true
			}
		case *ast.IncStmt:
			if id, ok := x.Target.(*ast.IdentExpr); ok {
				out[id.Name] = true
			}
		case *ast.IfStmt:
			assigned(x.Then, out)
			if x.Else != nil {
				assigned(x.Else, out)
			}
		case *ast.WhileStmt:
			assigned(x.Body, out)
		case *ast.ForStmt:
			assigned(x.Body, out)
			assigned(&ast.BlockStmt{Stmts: []ast.Stmt{x.Post}}, out)
		}
	}
}
func carriedNames(blocks []*ast.BlockStmt, e env) []string {
	set := map[string]bool{}
	for _, bl := range blocks {
		if bl != nil {
			assigned(bl, set)
		}
	}
	var out []string
	for name, s := range e.syms {
		if set[name] && s.resource < 0 && s.workgroup < 0 && s.mutable {
			out = append(out, name)
		}
	} // deterministic source-independent order by name
	for i := 0; i < len(out); i++ {
		for j := i + 1; j < len(out); j++ {
			if out[j] < out[i] {
				out[i], out[j] = out[j], out[i]
			}
		}
	}
	return out
}

func (c *Checker) lowerIf(b *fnBuilder, e env, x *ast.IfStmt) error {
	cond, ct, err := c.lowerExpr(b, e, x.Cond, types.TBool)
	if err != nil {
		return err
	}
	if !types.Equal(ct, types.TBool) {
		return diag(x.Cond.GetSpan(), "if condition is %s, want bool", ct)
	}
	names := carriedNames([]*ast.BlockStmt{x.Then, x.Else}, e)
	thenBlock := &ir.Block{}
	tb := b.child(thenBlock)
	te := e.clone()
	if err := c.lowerBlock(tb, x.Then, te, "branch"); err != nil {
		return err
	}
	b.sync(tb)
	elseBlock := &ir.Block{}
	eb := b.child(elseBlock)
	ee := e.clone()
	if x.Else != nil {
		if err := c.lowerBlock(eb, x.Else, ee, "branch"); err != nil {
			return err
		}
	}
	b.sync(eb)
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
func (c *Checker) lowerWhile(b *fnBuilder, e env, x *ast.WhileStmt) error {
	names := carriedNames([]*ast.BlockStmt{x.Body}, e)
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
	cv, ct, err := c.lowerExpr(cb, loopEnv, x.Cond, types.TBool)
	if err != nil {
		return err
	}
	if !types.Equal(ct, types.TBool) {
		return diag(x.Cond.GetSpan(), "while condition is %s, want bool", ct)
	}
	condBlock.Term = &ir.Yield{Values: []ir.ValueID{cv}}
	b.sync(cb)
	bodyBlock := &ir.Block{}
	bb := b.child(bodyBlock)
	bodyEnv := loopEnv.clone()
	if err := c.lowerBlock(bb, x.Body, bodyEnv, "loop"); err != nil {
		return err
	}
	b.sync(bb)
	if bodyBlock.Term == nil {
		vals := make([]ir.ValueID, len(names))
		for i, n := range names {
			vals[i] = bodyEnv.syms[n].value
		}
		bodyBlock.Term = &ir.Continue{Values: vals}
	} else {
		if _, ok := bodyBlock.Term.(*ir.Return); ok {
			return diag(x.Body.Span, "a while body that unconditionally returns is better expressed as if; Pine loops currently require a continuing path")
		}
		if _, ok := bodyBlock.Term.(*ir.Unreachable); ok {
			return diag(x.Body.Span, "while body has no continuing path")
		}
	}
	b.emit(&ir.Loop{Results: results, Params: params, Cond: condBlock, Body: bodyBlock, Span: x.Span})
	for i, n := range names {
		sym := e.syms[n]
		sym.value = results[i].ID
		e.syms[n] = sym
	}
	return nil
}

func (c *Checker) lowerFor(b *fnBuilder, e env, x *ast.ForStmt) error {
	// A Pine for-loop is a source-level convenience only. It lowers into the
	// exact same structured, loop-carried SSA form as while.
	loopEnv := e.clone()
	if err := c.lowerStmt(b, loopEnv, x.Init); err != nil {
		return err
	}
	body := &ast.BlockStmt{Span: x.Body.Span}
	body.Stmts = append(body.Stmts, x.Body.Stmts...)
	body.Stmts = append(body.Stmts, x.Post)
	if err := c.lowerWhile(b, loopEnv, &ast.WhileStmt{Cond: x.Cond, Body: body, Span: x.Span}); err != nil {
		return err
	}
	// The initializer is lexically scoped to the for-loop. Mutations to symbols
	// that existed before the loop remain visible after it through loop results.
	for name, outer := range e.syms {
		inner, ok := loopEnv.syms[name]
		if !ok || outer.resource >= 0 || outer.workgroup >= 0 {
			continue
		}
		outer.value = inner.value
		e.syms[name] = outer
	}
	return nil
}

func (c *Checker) lowerExpr(b *fnBuilder, e env, x ast.Expr, expected *types.Type) (ir.ValueID, *types.Type, error) {
	switch v := x.(type) {
	case *ast.NumberExpr:
		return c.lowerNumber(b, v, expected)
	case *ast.BoolExpr:
		if expected != nil && !types.Equal(expected, types.TBool) {
			return 0, nil, diag(v.Span, "bool literal cannot be used as %s", expected)
		}
		id := b.value()
		raw := "false"
		if v.Value {
			raw = "true"
		}
		b.emit(&ir.Const{Result: id, Type: types.TBool, Raw: raw, Span: v.Span})
		return id, types.TBool, nil
	case *ast.IdentExpr:
		if s, ok := e.syms[v.Name]; ok {
			if s.resource >= 0 {
				if !types.IsConstructible(s.ty) {
					return 0, nil, diag(v.Span, "runtime-sized resource %s must be accessed through its fixed fields or indexed tail", v.Name)
				}
				p := b.place()
				b.emit(&ir.PlaceRoot{Result: p, Type: s.ty, Resource: s.resource, Span: v.Span})
				id := b.value()
				b.emit(&ir.Load{Result: id, Type: s.ty, Place: p, Span: v.Span})
				return id, s.ty, nil
			}
			if s.workgroup >= 0 {
				if !types.IsConstructible(s.ty) {
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
		if k, t, ok := builtin(v.Name); ok {
			id := b.value()
			b.emit(&ir.Builtin{Result: id, Type: t, Kind: k, Span: v.Span})
			return id, t, nil
		}
		return 0, nil, diag(v.Span, "unknown identifier %q", v.Name)
	case *ast.MemberExpr:
		if v.Name == "length" {
			if p, pt, err := c.lowerPlace(b, e, v.Base); err == nil && pt.Kind == types.RuntimeArray {
				id := b.value()
				b.emit(&ir.ArrayLength{Result: id, Type: types.TU32, Place: p, Span: v.Span})
				return id, types.TU32, nil
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
		if bt.Kind == types.Struct {
			idx := types.FieldIndex(bt, v.Name)
			if idx < 0 {
				return 0, nil, diag(v.Span, "type %s has no field %s", bt, v.Name)
			}
			ft := bt.Fields[idx].Type
			id := b.value()
			b.emit(&ir.Extract{Result: id, Type: ft, Base: base, Index: idx, Span: v.Span})
			return id, ft, nil
		}
		if bt.Kind == types.Vector {
			idx := component(v.Name)
			if idx < 0 || idx >= bt.Lanes {
				return 0, nil, diag(v.Span, "vector %s has no component %s", bt, v.Name)
			}
			id := b.value()
			b.emit(&ir.Extract{Result: id, Type: bt.Elem, Base: base, Index: idx, Span: v.Span})
			return id, bt.Elem, nil
		}
		return 0, nil, diag(v.Span, "member access on %s", bt)
	case *ast.IndexExpr:
		p, pt, err := c.lowerPlace(b, e, v)
		if err != nil {
			return 0, nil, err
		}
		id := b.value()
		b.emit(&ir.Load{Result: id, Type: pt, Place: p, Span: v.Span})
		return id, pt, nil
	case *ast.StructLiteralExpr:
		if expected == nil || expected.Kind != types.Struct {
			return 0, nil, diag(v.Span, "struct literal requires a contextual struct type")
		}
		seen := map[string]ast.Expr{}
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
			if !types.Equal(t, f.Type) {
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
	case *ast.UnaryExpr:
		var want *types.Type
		if v.Op == "!" {
			want = types.TBool
		} else {
			want = expected
		}
		id, t, err := c.lowerExpr(b, e, v.X, want)
		if err != nil {
			return 0, nil, err
		}
		if v.Op == "!" && !types.Equal(t, types.TBool) {
			return 0, nil, diag(v.Span, "! requires bool")
		}
		if v.Op == "-" && !types.IsSignedNumeric(t) {
			return 0, nil, diag(v.Span, "unary - requires i32/f32 or a vector of them")
		}
		if v.Op == "~" && !types.IsIntegerLike(t) {
			return 0, nil, diag(v.Span, "unary ~ requires i32/u32 or an integer vector")
		}
		r := b.value()
		b.emit(&ir.Unary{Result: r, Type: t, Op: v.Op, X: id, Span: v.Span})
		return r, t, nil
	case *ast.BinaryExpr:
		if v.Op == "&&" || v.Op == "||" {
			return c.lowerShortCircuit(b, e, v)
		}
		return c.lowerBinaryExpr(b, e, v, expected)
	case *ast.ConditionalExpr:
		cond, ct, err := c.lowerExpr(b, e, v.Cond, types.TBool)
		if err != nil {
			return 0, nil, err
		}
		if !types.Equal(ct, types.TBool) {
			return 0, nil, diag(v.Cond.GetSpan(), "conditional expression requires bool condition, got %s", ct)
		}
		thenBlock := &ir.Block{}
		tb := b.child(thenBlock)
		tv, tt, err := c.lowerExpr(tb, e.clone(), v.Then, expected)
		if err != nil {
			return 0, nil, err
		}
		thenBlock.Term = &ir.Yield{Values: []ir.ValueID{tv}}
		b.sync(tb)
		elseBlock := &ir.Block{}
		eb := b.child(elseBlock)
		elseExpected := expected
		if elseExpected == nil {
			elseExpected = tt
		}
		ev, et, err := c.lowerExpr(eb, e.clone(), v.Else, elseExpected)
		if err != nil {
			return 0, nil, err
		}
		if !types.Equal(tt, et) {
			return 0, nil, diag(v.Span, "conditional branches have types %s and %s", tt, et)
		}
		elseBlock.Term = &ir.Yield{Values: []ir.ValueID{ev}}
		b.sync(eb)
		r := b.value()
		b.emit(&ir.If{Results: []ir.Result{{ID: r, Type: tt}}, Cond: cond, Then: thenBlock, Else: elseBlock, Span: v.Span})
		return r, tt, nil
	case *ast.CallExpr:
		return c.lowerCall(b, e, v, expected)
	default:
		return 0, nil, fmt.Errorf("unknown expression %T", x)
	}
}
func (c *Checker) lowerNumber(b *fnBuilder, n *ast.NumberExpr, expected *types.Type) (ir.ValueID, *types.Type, error) {
	raw, suffix, basePrefixed := splitNumberLiteral(n.Raw)
	isFloatSpelling := !basePrefixed && strings.ContainsAny(raw, ".eE")

	var t *types.Type
	switch suffix {
	case 'u':
		t = types.TU32
	case 'i':
		t = types.TI32
	case 'f':
		t = types.TF32
	default:
		if expected != nil && types.IsNumericScalar(expected) {
			t = expected
		} else if isFloatSpelling {
			t = types.TF32
		} else {
			t = types.TI32
		}
	}
	if expected != nil && types.IsNumericScalar(expected) && !types.Equal(t, expected) && suffix != 0 {
		return 0, nil, diag(n.Span, "literal suffix gives %s, context requires %s", t, expected)
	}

	// Canonicalize literals in Core IR. Backends never need to understand Pine
	// digit separators, base prefixes, or source suffixes.
	canonical := raw
	switch t.Kind {
	case types.U32:
		if suffix == 'f' || isFloatSpelling {
			return 0, nil, diag(n.Span, "u32 literal must be an integer")
		}
		v, err := strconv.ParseUint(raw, 0, 32)
		if err != nil {
			return 0, nil, diag(n.Span, "u32 literal out of range")
		}
		canonical = strconv.FormatUint(v, 10)
	case types.I32:
		if suffix == 'f' || isFloatSpelling {
			return 0, nil, diag(n.Span, "i32 literal must be an integer")
		}
		// Base-prefixed literals are still value literals, not bit-pattern casts.
		// They therefore obey the positive i32 range exactly like decimal literals.
		v, err := strconv.ParseUint(raw, 0, 31)
		if err != nil {
			return 0, nil, diag(n.Span, "i32 literal out of range")
		}
		canonical = strconv.FormatUint(v, 10)
	case types.F32:
		if basePrefixed {
			return 0, nil, diag(n.Span, "base-prefixed integer literal requires an integer context or explicit conversion")
		}
		f, err := strconv.ParseFloat(raw, 32)
		if err != nil || math.IsInf(f, 0) || math.IsNaN(f) {
			return 0, nil, diag(n.Span, "invalid f32 literal")
		}
		canonical = strconv.FormatFloat(f, 'g', -1, 32)
		if !strings.ContainsAny(canonical, ".eE") {
			canonical += ".0"
		}
	}
	id := b.value()
	b.emit(&ir.Const{Result: id, Type: t, Raw: canonical, Span: n.Span})
	return id, t, nil
}

func (c *Checker) lowerShortCircuit(b *fnBuilder, e env, x *ast.BinaryExpr) (ir.ValueID, *types.Type, error) {
	left, lt, err := c.lowerExpr(b, e, x.Left, types.TBool)
	if err != nil {
		return 0, nil, err
	}
	if !types.Equal(lt, types.TBool) {
		return 0, nil, diag(x.Left.GetSpan(), "logical operand is %s, want bool", lt)
	}
	then := &ir.Block{}
	tb := b.child(then)
	els := &ir.Block{}
	eb := b.child(els)
	if x.Op == "&&" {
		rv, rt, err := c.lowerExpr(tb, e.clone(), x.Right, types.TBool)
		if err != nil {
			return 0, nil, err
		}
		if !types.Equal(rt, types.TBool) {
			return 0, nil, diag(x.Right.GetSpan(), "logical operand is %s, want bool", rt)
		}
		then.Term = &ir.Yield{Values: []ir.ValueID{rv}}
		b.sync(tb)
		id, _, err := c.lowerExpr(eb, e.clone(), &ast.BoolExpr{Value: false, Span: x.Span}, types.TBool)
		if err != nil {
			return 0, nil, err
		}
		els.Term = &ir.Yield{Values: []ir.ValueID{id}}
		b.sync(eb)
	} else {
		id, _, err := c.lowerExpr(tb, e.clone(), &ast.BoolExpr{Value: true, Span: x.Span}, types.TBool)
		if err != nil {
			return 0, nil, err
		}
		then.Term = &ir.Yield{Values: []ir.ValueID{id}}
		b.sync(tb)
		rv, rt, err := c.lowerExpr(eb, e.clone(), x.Right, types.TBool)
		if err != nil {
			return 0, nil, err
		}
		if !types.Equal(rt, types.TBool) {
			return 0, nil, diag(x.Right.GetSpan(), "logical operand is %s, want bool", rt)
		}
		els.Term = &ir.Yield{Values: []ir.ValueID{rv}}
		b.sync(eb)
	}
	r := b.value()
	b.emit(&ir.If{Results: []ir.Result{{ID: r, Type: types.TBool}}, Cond: left, Then: then, Else: els, Span: x.Span})
	return r, types.TBool, nil
}
func (c *Checker) lowerBinaryExpr(b *fnBuilder, e env, x *ast.BinaryExpr, expected *types.Type) (ir.ValueID, *types.Type, error) {
	// WGSL/SPIR-V shifts have a deliberately asymmetric type rule: the shifted
	// value is signed or unsigned integer, while the count is always u32 with
	// matching vector width. Resolve that directly instead of relying on ordinary
	// binary contextual typing.
	if x.Op == "<<" || x.Op == ">>" {
		l, lt, err := c.lowerExpr(b, e, x.Left, expected)
		if err != nil {
			return 0, nil, err
		}
		countType := types.ShiftCountType(lt)
		if countType == nil {
			return 0, nil, diag(x.Left.GetSpan(), "%s requires an i32/u32 scalar or integer vector on the left, got %s", x.Op, lt)
		}
		r, rt, err := c.lowerExpr(b, e, x.Right, countType)
		if err != nil {
			return 0, nil, err
		}
		r, rt = c.normalizeShiftCount(b, r, rt, x.Right.GetSpan())
		return c.emitBinary(b, x.Op, l, lt, r, rt, x.Span)
	}

	var l ir.ValueID
	var lt *types.Type
	var r ir.ValueID
	var rt *types.Type
	var err error
	_, ln := x.Left.(*ast.NumberExpr)
	_, rn := x.Right.(*ast.NumberExpr)
	if ln && !rn {
		r, rt, err = c.lowerExpr(b, e, x.Right, nil)
		if err != nil {
			return 0, nil, err
		}
		want := rt
		if x.Op == "*" || x.Op == "/" {
			if rt.Kind == types.Vector {
				want = rt.Elem
			}
		}
		l, lt, err = c.lowerExpr(b, e, x.Left, want)
	} else {
		l, lt, err = c.lowerExpr(b, e, x.Left, expected)
		if err == nil {
			want := lt
			if (x.Op == "*" || x.Op == "/") && lt.Kind == types.Vector {
				want = lt.Elem
			}
			r, rt, err = c.lowerExpr(b, e, x.Right, want)
		}
	}
	if err != nil {
		return 0, nil, err
	}
	return c.emitBinary(b, x.Op, l, lt, r, rt, x.Span)
}

// normalizeShiftCount gives Pine one backend-independent shift meaning. Pine's
// 32-bit shifts use the low five bits of each count, matching WGSL semantics
// while ensuring SPIR-V never observes an out-of-range shift count.
func (c *Checker) normalizeShiftCount(b *fnBuilder, value ir.ValueID, t *types.Type, span source.Span) (ir.ValueID, *types.Type) {
	maskScalar := b.value()
	b.emit(&ir.Const{Result: maskScalar, Type: types.TU32, Raw: "31", Span: span})
	mask := maskScalar
	if t.Kind == types.Vector {
		values := make([]ir.ValueID, t.Lanes)
		for i := range values {
			values[i] = maskScalar
		}
		mask = b.value()
		b.emit(&ir.Composite{Result: mask, Type: t, Values: values, Span: span})
	}
	result := b.value()
	b.emit(&ir.Binary{Result: result, Type: t, Op: "&", Left: value, Right: mask, Span: span})
	return result, t
}

func (c *Checker) emitBinary(b *fnBuilder, op string, l ir.ValueID, lt *types.Type, r ir.ValueID, rt *types.Type, span source.Span) (ir.ValueID, *types.Type, error) {
	var out *types.Type
	switch op {
	case "==", "!=", "<", "<=", ">", ">=":
		if !types.Equal(lt, rt) || !types.IsNumericScalar(lt) {
			return 0, nil, diag(span, "comparison %s requires matching scalar numeric operands; got %s and %s", op, lt, rt)
		}
		out = types.TBool
	case "+", "-":
		if !types.Equal(lt, rt) || !types.IsNumeric(lt) {
			return 0, nil, diag(span, "%s requires matching numeric operands; got %s and %s", op, lt, rt)
		}
		out = lt
	case "*":
		if types.Equal(lt, rt) && types.IsNumeric(lt) {
			out = lt
		} else if lt.Kind == types.Vector && types.Equal(lt.Elem, rt) {
			out = lt
		} else if rt.Kind == types.Vector && types.Equal(rt.Elem, lt) {
			out = rt
		} else {
			return 0, nil, diag(span, "cannot multiply %s by %s", lt, rt)
		}
	case "/":
		if types.Equal(lt, rt) && types.IsNumeric(lt) {
			out = lt
		} else if lt.Kind == types.Vector && types.Equal(lt.Elem, rt) {
			out = lt
		} else {
			return 0, nil, diag(span, "cannot divide %s by %s", lt, rt)
		}
	case "%":
		if !types.Equal(lt, rt) || !types.IsNumericScalar(lt) {
			return 0, nil, diag(span, "%% requires matching scalar numeric operands; got %s and %s", lt, rt)
		}
		out = lt
	case "&", "|", "^":
		if !types.Equal(lt, rt) || !types.IsIntegerLike(lt) {
			return 0, nil, diag(span, "%s requires matching i32/u32 scalar or integer-vector operands; got %s and %s", op, lt, rt)
		}
		out = lt
	case "<<", ">>":
		if !types.IsIntegerLike(lt) || !types.Equal(rt, types.ShiftCountType(lt)) {
			return 0, nil, diag(span, "%s requires %s shifted by %s; got %s and %s", op, lt, types.ShiftCountType(lt), lt, rt)
		}
		out = lt
	default:
		return 0, nil, diag(span, "unsupported binary operator %s", op)
	}
	id := b.value()
	b.emit(&ir.Binary{Result: id, Type: out, Op: op, Left: l, Right: r, Span: span})
	return id, out, nil
}
func intrinsicBuiltin(name string) (ir.IntrinsicKind, bool) {
	switch name {
	case "abs":
		return ir.IntrinsicAbs, true
	case "floor":
		return ir.IntrinsicFloor, true
	case "ceil":
		return ir.IntrinsicCeil, true
	case "trunc":
		return ir.IntrinsicTrunc, true
	case "sin":
		return ir.IntrinsicSin, true
	case "cos":
		return ir.IntrinsicCos, true
	case "tan":
		return ir.IntrinsicTan, true
	case "exp":
		return ir.IntrinsicExp, true
	case "exp2":
		return ir.IntrinsicExp2, true
	case "log":
		return ir.IntrinsicLog, true
	case "log2":
		return ir.IntrinsicLog2, true
	case "sqrt":
		return ir.IntrinsicSqrt, true
	case "inverseSqrt":
		return ir.IntrinsicInverseSqrt, true
	case "pow":
		return ir.IntrinsicPow, true
	case "min":
		return ir.IntrinsicMin, true
	case "max":
		return ir.IntrinsicMax, true
	case "clamp":
		return ir.IntrinsicClamp, true
	case "dot":
		return ir.IntrinsicDot, true
	case "length":
		return ir.IntrinsicLength, true
	case "distance":
		return ir.IntrinsicDistance, true
	case "cross":
		return ir.IntrinsicCross, true
	case "normalize":
		return ir.IntrinsicNormalize, true
	default:
		return 0, false
	}
}

func isReservedCallable(name string) bool {
	if _, ok := intrinsicBuiltin(name); ok {
		return true
	}
	if _, ok := atomicBuiltin(name); ok {
		return true
	}
	if name == "workgroupBarrier" || name == "storageBarrier" {
		return true
	}
	return types.ParseBuiltin(name) != nil
}

func intrinsicArity(kind ir.IntrinsicKind) int {
	switch kind {
	case ir.IntrinsicPow, ir.IntrinsicMin, ir.IntrinsicMax, ir.IntrinsicDot, ir.IntrinsicDistance, ir.IntrinsicCross:
		return 2
	case ir.IntrinsicClamp:
		return 3
	default:
		return 1
	}
}

func (c *Checker) lowerIntrinsic(b *fnBuilder, e env, x *ast.CallExpr, kind ir.IntrinsicKind, expected *types.Type) (ir.ValueID, *types.Type, error) {
	wantN := intrinsicArity(kind)
	if len(x.Args) != wantN {
		return 0, nil, diag(x.Span, "%s expects %d argument(s), got %d", kind, wantN, len(x.Args))
	}
	args := make([]ir.ValueID, wantN)
	argTypes := make([]*types.Type, wantN)

	firstExpected := expected
	if kind == ir.IntrinsicDot || kind == ir.IntrinsicLength || kind == ir.IntrinsicDistance {
		firstExpected = nil // result f32 does not determine vector width
	}
	v, t, err := c.lowerExpr(b, e, x.Args[0], firstExpected)
	if err != nil {
		return 0, nil, err
	}
	args[0], argTypes[0] = v, t
	for i := 1; i < wantN; i++ {
		v, t, err := c.lowerExpr(b, e, x.Args[i], argTypes[0])
		if err != nil {
			return 0, nil, err
		}
		args[i], argTypes[i] = v, t
	}

	floatVec := func(t *types.Type) bool {
		return t != nil && t.Kind == types.Vector && t.Elem != nil && t.Elem.Kind == types.F32
	}
	same := func() bool {
		for i := 1; i < len(argTypes); i++ {
			if !types.Equal(argTypes[0], argTypes[i]) {
				return false
			}
		}
		return true
	}
	out := argTypes[0]
	switch kind {
	case ir.IntrinsicAbs:
		t := argTypes[0]
		ok := t.Kind == types.I32 || t.Kind == types.F32 || t.Kind == types.Vector && (t.Elem.Kind == types.I32 || t.Elem.Kind == types.F32)
		if !ok {
			return 0, nil, diag(x.Span, "abs requires i32/f32 scalar or vector, got %s", t)
		}
	case ir.IntrinsicFloor, ir.IntrinsicCeil, ir.IntrinsicTrunc, ir.IntrinsicSin, ir.IntrinsicCos, ir.IntrinsicTan, ir.IntrinsicExp, ir.IntrinsicExp2, ir.IntrinsicLog, ir.IntrinsicLog2, ir.IntrinsicSqrt, ir.IntrinsicInverseSqrt:
		if !types.IsFloatLike(argTypes[0]) {
			return 0, nil, diag(x.Span, "%s requires f32 scalar/vector, got %s", kind, argTypes[0])
		}
	case ir.IntrinsicPow:
		if !same() || !types.IsFloatLike(argTypes[0]) {
			return 0, nil, diag(x.Span, "pow requires matching f32 scalar/vector operands")
		}
	case ir.IntrinsicMin, ir.IntrinsicMax:
		if !same() || !types.IsIntegerLike(argTypes[0]) {
			return 0, nil, diag(x.Span, "%s requires matching integer scalar/vector operands", kind)
		}
	case ir.IntrinsicClamp:
		if !same() || !types.IsIntegerLike(argTypes[0]) {
			return 0, nil, diag(x.Span, "clamp requires three matching integer scalar/vector operands")
		}
	case ir.IntrinsicDot:
		if !same() || !floatVec(argTypes[0]) {
			return 0, nil, diag(x.Span, "dot requires matching f32 vectors")
		}
		out = types.TF32
	case ir.IntrinsicLength:
		if !floatVec(argTypes[0]) {
			return 0, nil, diag(x.Span, "length requires an f32 vector")
		}
		out = types.TF32
	case ir.IntrinsicDistance:
		if !same() || !floatVec(argTypes[0]) {
			return 0, nil, diag(x.Span, "distance requires matching f32 vectors")
		}
		out = types.TF32
	case ir.IntrinsicCross:
		if !same() || !floatVec(argTypes[0]) || argTypes[0].Lanes != 3 {
			return 0, nil, diag(x.Span, "cross requires two vec3f operands")
		}
	case ir.IntrinsicNormalize:
		if !floatVec(argTypes[0]) {
			return 0, nil, diag(x.Span, "normalize requires an f32 vector")
		}
	default:
		return 0, nil, diag(x.Span, "unsupported intrinsic %s", kind)
	}
	if expected != nil && !types.Equal(out, expected) {
		return 0, nil, diag(x.Span, "%s returns %s, context requires %s", kind, out, expected)
	}
	r := b.value()
	b.emit(&ir.Intrinsic{Result: r, Type: out, Kind: kind, Args: args, Span: x.Span})
	return r, out, nil
}

func (c *Checker) lowerCall(b *fnBuilder, e env, x *ast.CallExpr, expected *types.Type) (ir.ValueID, *types.Type, error) {
	id, ok := x.Callee.(*ast.IdentExpr)
	if !ok {
		return 0, nil, diag(x.Callee.GetSpan(), "call target must be a function or type name")
	}
	if target := types.ParseBuiltin(id.Name); target != nil && target.Kind != types.Void && target.Kind != types.Bool {
		return c.lowerConstructor(b, e, x, target)
	}
	if kind, ok := intrinsicBuiltin(id.Name); ok {
		return c.lowerIntrinsic(b, e, x, kind, expected)
	}
	if id.Name == "workgroupBarrier" || id.Name == "storageBarrier" {
		if len(x.Args) != 0 {
			return 0, nil, diag(x.Span, "%s expects no arguments", id.Name)
		}
		if !b.fn.Compute {
			return 0, nil, diag(x.Span, "%s is only valid in compute functions", id.Name)
		}
		kind := ir.BarrierWorkgroup
		if id.Name == "storageBarrier" {
			kind = ir.BarrierStorage
		}
		b.emit(&ir.Barrier{Kind: kind, Span: x.Span})
		return 0, types.TVoid, nil
	}
	if op, atomic := atomicBuiltin(id.Name); atomic {
		want := 1
		if op != ir.AtomicLoad {
			want = 2
		}
		if len(x.Args) != want {
			return 0, nil, diag(x.Span, "%s expects %d argument(s), got %d", id.Name, want, len(x.Args))
		}
		p, pt, err := c.lowerPlace(b, e, x.Args[0])
		if err != nil {
			return 0, nil, err
		}
		if pt.Kind != types.Atomic || (pt.Elem.Kind != types.I32 && pt.Elem.Kind != types.U32) {
			return 0, nil, diag(x.Args[0].GetSpan(), "%s requires an atomic<i32> or atomic<u32> place, got %s", id.Name, pt)
		}
		var value ir.ValueID
		if op != ir.AtomicLoad {
			v, vt, err := c.lowerExpr(b, e, x.Args[1], pt.Elem)
			if err != nil {
				return 0, nil, err
			}
			if !types.Equal(vt, pt.Elem) {
				return 0, nil, diag(x.Args[1].GetSpan(), "%s value is %s, want %s", id.Name, vt, pt.Elem)
			}
			value = v
		}
		if op == ir.AtomicStore {
			b.emit(&ir.Atomic{Type: pt.Elem, Op: op, Place: p, Value: value, Span: x.Span})
			return 0, types.TVoid, nil
		}
		r := b.value()
		b.emit(&ir.Atomic{Result: r, Type: pt.Elem, Op: op, Place: p, Value: value, Span: x.Span})
		return r, pt.Elem, nil
	}
	sig := c.funcs[id.Name]
	if sig == nil || sig.compute != nil {
		return 0, nil, diag(id.Span, "unknown callable function %q", id.Name)
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
		if !types.Equal(t, sig.params[i].ty) {
			return 0, nil, diag(a.GetSpan(), "argument %d to %s is %s, want %s", i+1, id.Name, t, sig.params[i].ty)
		}
		args[i] = v
	}
	r := ir.ValueID(0)
	if sig.ret.Kind != types.Void {
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
	default:
		return 0, false
	}
}
func (c *Checker) lowerConstructor(b *fnBuilder, e env, x *ast.CallExpr, target *types.Type) (ir.ValueID, *types.Type, error) {
	if target.Kind == types.I32 || target.Kind == types.U32 || target.Kind == types.F32 {
		if len(x.Args) != 1 {
			return 0, nil, diag(x.Span, "%s constructor expects one argument", target)
		}
		v, t, err := c.lowerExpr(b, e, x.Args[0], target)
		if err == nil && types.Equal(t, target) {
			return v, t, nil
		}
		if err != nil { // contextual literal failure may be meaningful; retry without context only for non-literals
			if _, ok := x.Args[0].(*ast.NumberExpr); ok {
				return 0, nil, err
			}
			v, t, err = c.lowerExpr(b, e, x.Args[0], nil)
			if err != nil {
				return 0, nil, err
			}
		}
		if !types.IsNumericScalar(t) {
			return 0, nil, diag(x.Args[0].GetSpan(), "cannot convert %s to %s", t, target)
		}
		r := b.value()
		b.emit(&ir.Convert{Result: r, Type: target, X: v, From: t, Span: x.Span})
		return r, target, nil
	}
	if target.Kind == types.Vector {
		if len(x.Args) != 1 && len(x.Args) != target.Lanes {
			return 0, nil, diag(x.Span, "%s constructor expects 1 or %d scalar arguments", types.Friendly(target), target.Lanes)
		}
		vals := make([]ir.ValueID, target.Lanes)
		if len(x.Args) == 1 {
			v, t, err := c.lowerExpr(b, e, x.Args[0], target.Elem)
			if err != nil {
				return 0, nil, err
			}
			if !types.Equal(t, target.Elem) {
				return 0, nil, diag(x.Args[0].GetSpan(), "vector splat expects %s, got %s", target.Elem, t)
			}
			for i := range vals {
				vals[i] = v
			}
		} else {
			for i, a := range x.Args {
				v, t, err := c.lowerExpr(b, e, a, target.Elem)
				if err != nil {
					return 0, nil, err
				}
				if !types.Equal(t, target.Elem) {
					return 0, nil, diag(a.GetSpan(), "vector component is %s, want %s", t, target.Elem)
				}
				vals[i] = v
			}
		}
		r := b.value()
		b.emit(&ir.Composite{Result: r, Type: target, Values: vals, Span: x.Span})
		return r, target, nil
	}
	return 0, nil, diag(x.Span, "type %s is not directly constructible", target)
}
func (c *Checker) lowerPlace(b *fnBuilder, e env, x ast.Expr) (ir.PlaceID, *types.Type, error) {
	switch v := x.(type) {
	case *ast.IdentExpr:
		s, ok := e.syms[v.Name]
		if !ok || (s.resource < 0 && s.workgroup < 0) {
			return 0, nil, diag(v.Span, "%s is not an addressable GPU place", v.Name)
		}
		p := b.place()
		if s.resource >= 0 {
			b.emit(&ir.PlaceRoot{Result: p, Type: s.ty, Resource: s.resource, Span: v.Span})
		} else {
			b.emit(&ir.PlaceWorkgroup{Result: p, Type: s.ty, Workgroup: s.workgroup, Span: v.Span})
		}
		return p, s.ty, nil
	case *ast.MemberExpr:
		bp, bt, err := c.lowerPlace(b, e, v.Base)
		if err != nil {
			return 0, nil, err
		}
		if bt.Kind == types.Vector {
			idx := component(v.Name)
			if idx < 0 || idx >= bt.Lanes {
				return 0, nil, diag(v.Span, "vector %s has no component %s", bt, v.Name)
			}
			iv := b.value()
			b.emit(&ir.Const{Result: iv, Type: types.TU32, Raw: fmt.Sprintf("%d", idx), Span: v.Span})
			p := b.place()
			b.emit(&ir.PlaceIndex{Result: p, Type: bt.Elem, Base: bp, Index: iv, Span: v.Span})
			return p, bt.Elem, nil
		}
		if bt.Kind != types.Struct {
			return 0, nil, diag(v.Span, "field access requires struct/vector place, got %s", bt)
		}
		idx := types.FieldIndex(bt, v.Name)
		if idx < 0 {
			return 0, nil, diag(v.Span, "type %s has no field %s", bt, v.Name)
		}
		ft := bt.Fields[idx].Type
		p := b.place()
		b.emit(&ir.PlaceField{Result: p, Type: ft, Base: bp, Field: idx, Span: v.Span})
		return p, ft, nil
	case *ast.IndexExpr:
		bp, bt, err := c.lowerPlace(b, e, v.Base)
		if err != nil {
			return 0, nil, err
		}
		if bt.Kind != types.RuntimeArray && bt.Kind != types.FixedArray && bt.Kind != types.Vector {
			return 0, nil, diag(v.Span, "indexing requires an array place, got %s", bt)
		}
		iv, it, err := c.lowerExpr(b, e, v.Index, types.TU32)
		if err != nil {
			return 0, nil, err
		}
		if !types.Equal(it, types.TU32) && !types.Equal(it, types.TI32) {
			return 0, nil, diag(v.Index.GetSpan(), "array index must be i32 or u32, got %s", it)
		}
		p := b.place()
		b.emit(&ir.PlaceIndex{Result: p, Type: bt.Elem, Base: bp, Index: iv, Span: v.Span})
		return p, bt.Elem, nil
	default:
		return 0, nil, diag(x.GetSpan(), "expression is not an addressable GPU place")
	}
}
func builtin(name string) (ir.BuiltinKind, *types.Type, bool) {
	switch name {
	case "globalId":
		return ir.GlobalID, types.Vec(types.TU32, 3), true
	case "localId":
		return ir.LocalID, types.Vec(types.TU32, 3), true
	case "localIndex":
		return ir.LocalIndex, types.TU32, true
	case "workgroupId":
		return ir.WorkgroupID, types.Vec(types.TU32, 3), true
	case "numWorkgroups":
		return ir.NumWorkgroups, types.Vec(types.TU32, 3), true
	}
	return 0, nil, false
}
func component(name string) int {
	switch name {
	case "x":
		return 0
	case "y":
		return 1
	case "z":
		return 2
	case "w":
		return 3
	}
	return -1
}
func diag(span source.Span, f string, a ...any) error {
	return &source.Error{Span: span, Message: fmt.Sprintf(f, a...)}
}

func checkRecursion(m *ir.Module) error {
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
			return fmt.Errorf("recursive function cycle is not allowed: %s -> %s", strings.Join(stack, " -> "), n)
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
	for n := range graph {
		if err := visit(n); err != nil {
			return err
		}
	}
	return nil
}
