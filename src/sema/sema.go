package sema

import (
	"fmt"
	"math"
	"runtime"
	"sort"
	"strconv"
	"strings"
	"sync"

	"tach/src/ast"
	"tach/src/flow"
	"tach/src/ir"
	"tach/src/layout"
	"tach/src/source"
	"tach/src/types"
)

type Checker struct {
	ast     *ast.Module
	mod     *ir.Module
	flow    *flow.Module
	types   map[string]*types.Type
	funcs   map[string]*funcSig
	owners  map[string]string
	imports map[string]map[string]bool
	workers int
}

type funcSig struct {
	name     string
	params   []namedType
	ret      *types.Type
	decl     *ast.FunctionDecl
	indexed  bool
	exported bool
	view     flow.ViewFormat
}
type namedType struct {
	name   string
	ty     *types.Type
	buffer bool
}

type symbol struct {
	name      string
	ty        *types.Type
	value     ir.ValueID
	mutable   bool
	buffer    int // -1 unless this is a stage buffer place
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
	top   bool
	loop  *loopContext
}

type loopContext struct {
	names []string
	base  env
	post  ast.Stmt
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
	return &fnBuilder{c: b.c, fn: b.fn, ids: b.ids, block: block, loop: b.loop}
}

func CheckAndLower(m *ast.Module) (*flow.Module, error) {
	module, _, err := CheckAndLowerProject([]*ast.Module{m})
	return module, err
}

func CheckAndLowerProject(modules []*ast.Module, requestedWorkers ...int) (*flow.Module, []flow.Documentation, error) {
	workers := runtime.GOMAXPROCS(0)
	if len(requestedWorkers) > 0 && requestedWorkers[0] > 0 && requestedWorkers[0] < workers {
		workers = requestedWorkers[0]
	}
	merged := &ast.Module{}
	documentation := flow.Documentation{Types: map[string]flow.TypeDocumentation{}, Functions: map[string]flow.FunctionDocumentation{}}
	files := make([]flow.Documentation, 0, len(modules))
	var documentationDiagnostics source.Diagnostics
	for _, module := range modules {
		docs, err := checkDocumentation(module)
		if err != nil {
			documentationDiagnostics = appendError(documentationDiagnostics, err)
		}
		files = append(files, docs)
		if len(modules) == 1 {
			documentation.Title, documentation.Summary = docs.Title, docs.Summary
		}
		for name, item := range docs.Types {
			documentation.Types[name] = item
		}
		for name, item := range docs.Functions {
			documentation.Functions[name] = item
		}
		merged.Decls = append(merged.Decls, module.Decls...)
	}
	kernel := &ir.Module{}
	c := &Checker{ast: merged, mod: kernel, flow: &flow.Module{Kernel: kernel, Documentation: documentation}, types: map[string]*types.Type{}, funcs: map[string]*funcSig{}, owners: map[string]string{}, imports: map[string]map[string]bool{}, workers: workers}
	for _, module := range modules {
		file := strings.TrimSuffix(module.File, ".tach")
		visible := map[string]bool{file: true}
		for _, item := range module.Imports {
			visible[item.Target] = true
		}
		c.imports[file] = visible
		for _, declaration := range module.Decls {
			switch item := declaration.(type) {
			case *ast.TypeDecl:
				c.owners[item.Name] = file
			case *ast.FunctionDecl:
				c.owners[item.Name] = file
			}
		}
	}
	for _, n := range []string{"void", "bool", "int32", "uint32", "float16", "float32", "float16x2", "float16x3", "float16x4", "float32x2", "float32x3", "float32x4", "uint32x2", "uint32x3", "uint32x4", "int32x2", "int32x3", "int32x4"} {
		c.types[n] = types.ParseBuiltin(n)
	}
	var interfaceDiagnostics source.Diagnostics
	for _, check := range []func() error{c.collectTypes, c.resolveTypeFields, c.checkRuntimeArrayPlacement, c.checkTypeCycles, c.collectFunctions} {
		if err := check(); err != nil {
			interfaceDiagnostics = appendError(interfaceDiagnostics, err)
		}
	}
	if len(interfaceDiagnostics) > 0 {
		return nil, nil, append(documentationDiagnostics, interfaceDiagnostics...).Sorted()
	}
	if err := c.lowerFunctions(); err != nil {
		return nil, nil, appendError(documentationDiagnostics, err).Sorted()
	}
	inferBufferAccess(c.mod)
	if err := checkRecursion(c.mod); err != nil {
		return nil, nil, appendError(documentationDiagnostics, err).Sorted()
	}
	if err := ir.Verify(c.mod); err != nil {
		return nil, nil, fmt.Errorf("internal IR verification failed: %w", err)
	}
	if err := c.lowerPrograms(); err != nil {
		return nil, nil, appendError(documentationDiagnostics, err).Sorted()
	}
	if err := flow.Verify(c.flow); err != nil {
		return nil, nil, fmt.Errorf("internal Flow IR verification failed: %w", err)
	}
	if len(documentationDiagnostics) > 0 {
		return nil, nil, documentationDiagnostics.Sorted()
	}
	return c.flow, files, nil
}

func appendError(diagnostics source.Diagnostics, err error) source.Diagnostics {
	if list, ok := err.(source.Diagnostics); ok {
		return append(diagnostics, list...)
	}
	if diagnostic, ok := err.(*source.Diagnostic); ok {
		diagnostic.Kind = "semantic"
		return append(diagnostics, *diagnostic)
	}
	return append(diagnostics, source.Diagnostic{Kind: "semantic", Message: err.Error()})
}

func parallel(workers, count int, work func(int) error) []error {
	errors := make([]error, count)
	jobs := make(chan int)
	var wait sync.WaitGroup
	workers = min(workers, count)
	for range workers {
		wait.Add(1)
		go func() {
			defer wait.Done()
			for index := range jobs {
				errors[index] = work(index)
			}
		}()
	}
	for index := range count {
		jobs <- index
	}
	close(jobs)
	wait.Wait()
	return errors
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
	var diagnostics source.Diagnostics
	var declarations []*ast.TypeDecl
	for _, d := range c.ast.Decls {
		td, ok := d.(*ast.TypeDecl)
		if ok {
			declarations = append(declarations, td)
		}
	}
	errors := parallel(c.workers, len(declarations), func(index int) error {
		td := declarations[index]
		if len(td.Fields) == 0 {
			return diag(td.Span, "type %s requires at least one field", td.Name)
		}
		t := c.types[td.Name]
		seen := map[string]bool{}
		for i, f := range td.Fields {
			if seen[f.Name] {
				return diag(f.Span, "duplicate field %q in %s", f.Name, td.Name)
			}
			seen[f.Name] = true
			ft, err := c.resolveType(f.Type)
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
		return nil
	})
	for _, err := range errors {
		if err != nil {
			diagnostics = appendError(diagnostics, err)
		}
	}
	if len(diagnostics) > 0 {
		return diagnostics
	}
	return nil
}

func (c *Checker) checkRuntimeArrayPlacement() error {
	for _, t := range c.mod.Structs {
		for i, f := range t.Fields {
			if f.Type.Kind == types.RuntimeArray {
				if i != len(t.Fields)-1 {
					return diag(c.fieldSpan(t.Name, f.Name), "runtime array in %s must be the final member", t.Name)
				}
				if types.ContainsRuntimeArray(f.Type.Elem) {
					return diag(c.fieldSpan(t.Name, f.Name), "runtime array element %s in %s must have a fixed footprint", f.Type.Elem, t.Name)
				}
				continue
			}
			if types.ContainsRuntimeArray(f.Type) {
				return diag(c.fieldSpan(t.Name, f.Name), "%s.%s nests a runtime-sized structure; Tach permits one trailing runtime array", t.Name, f.Name)
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
			return diag(c.declarationSpan(t.Name), "recursive value type %s is not supported", t.Name)
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

func (c *Checker) declarationSpan(name string) source.Span {
	for _, declaration := range c.ast.Decls {
		if item, ok := declaration.(*ast.TypeDecl); ok && item.Name == name {
			return item.Span
		}
	}
	return source.Span{}
}

func (c *Checker) fieldSpan(typeName, fieldName string) source.Span {
	for _, declaration := range c.ast.Decls {
		if item, ok := declaration.(*ast.TypeDecl); ok && item.Name == typeName {
			for _, field := range item.Fields {
				if field.Name == fieldName {
					return field.Span
				}
			}
			return item.Span
		}
	}
	return source.Span{}
}

func (c *Checker) collectFunctions() error {
	var diagnostics source.Diagnostics
	var declarations []*ast.FunctionDecl
	for _, d := range c.ast.Decls {
		if function, ok := d.(*ast.FunctionDecl); ok {
			declarations = append(declarations, function)
		}
	}
	signatures := make([]*funcSig, len(declarations))
	errors := parallel(c.workers, len(declarations), func(index int) error {
		x := declarations[index]
		if ReservedName(x.Name) {
			return diag(x.Span, "function name %q is reserved by Tach", x.Name)
		}
		sig := &funcSig{name: x.Name, decl: x, ret: types.TVoid, indexed: len(x.Indices) > 0, exported: x.Exported}
		seen := map[string]bool{}
		for _, p := range x.Params {
			if seen[p.Name] {
				return diag(p.Span, "duplicate parameter %q", p.Name)
			}
			seen[p.Name] = true
			t, buffer, err := c.parameterType(p.Type, sig.indexed || x.Exported)
			if err != nil {
				return err
			}
			if buffer && !sig.indexed && !x.Exported {
				return diag(p.Span, "helper parameter %s cannot be a buffer", p.Name)
			}
			sig.params = append(sig.params, namedType{p.Name, t, buffer})
		}
		if sig.indexed && x.Return != nil {
			return diag(x.Return.GetSpan(), "indexed stage %s cannot declare a return type", x.Name)
		}
		if x.Return != nil {
			if format, ok := viewType(x.Return); ok {
				if !x.Exported {
					return diag(x.Return.GetSpan(), "view return is only valid on an exported program")
				}
				sig.view = format
				signatures[index] = sig
				return nil
			}
			if x.Exported {
				return diag(x.Return.GetSpan(), "public program %s can only return view<srgb8>", x.Name)
			}
			r, err := c.resolveType(x.Return)
			if err != nil {
				return err
			}
			if r.Kind != types.Void && !types.IsConstructible(r) {
				return diag(x.Return.GetSpan(), "function cannot return non-constructible type %s", r)
			}
			sig.ret = r
		}
		signatures[index] = sig
		return nil
	})
	for index, err := range errors {
		if err != nil {
			diagnostics = appendError(diagnostics, err)
			continue
		}
		sig := signatures[index]
		if _, exists := c.funcs[sig.name]; exists {
			diagnostics = appendError(diagnostics, diag(sig.decl.Span, "function %q is already defined", sig.name))
			continue
		}
		c.funcs[sig.name] = sig
	}
	if len(diagnostics) > 0 {
		return diagnostics
	}
	return nil
}
func (c *Checker) resolveType(te ast.TypeExpr) (*types.Type, error) {
	switch t := te.(type) {
	case *ast.NamedType:
		x := c.types[t.Name]
		if x != nil && !c.visible(t.Name, t.Span.File) {
			x = nil
		}
		if x == nil {
			return nil, diag(t.Span, "unknown type %q", t.Name)
		}
		return x, nil
	case *ast.RuntimeArrayType:
		e, err := c.resolveType(t.Elem)
		if err != nil {
			return nil, err
		}
		if e.Kind == types.Void || e.Kind == types.RuntimeArray {
			return nil, diag(t.Span, "invalid runtime array element type %s", e)
		}
		return types.Runtime(e), nil
	case *ast.FixedArrayType:
		e, err := c.resolveType(t.Elem)
		if err != nil {
			return nil, err
		}
		if !types.IsWorkgroupStorable(e) {
			return nil, diag(t.Span, "invalid fixed array element type %s", e)
		}
		raw, _ := splitNumberLiteral(t.Count)
		n, err := strconv.ParseUint(raw, 0, 32)
		if err != nil || n == 0 {
			return nil, diag(t.Span, "fixed array length must be a positive uint32 constant")
		}
		return types.Array(e, uint32(n)), nil
	case *ast.GenericType:
		if t.Name == "buffer" {
			return nil, diag(t.Span, "buffer<T> is only valid in a kernel parameter")
		}
		if t.Name == "shared" {
			return nil, diag(t.Span, "shared<T> is only valid in an uninitialized kernel-body let declaration")
		}
		if t.Name == "atomic" {
			if len(t.Args) != 1 {
				return nil, diag(t.Span, "atomic<T> takes exactly one type argument")
			}
			e, err := c.resolveType(t.Args[0])
			if err != nil {
				return nil, err
			}
			if e.Kind != types.I32 && e.Kind != types.U32 {
				return nil, diag(t.Span, "atomic element type must be int32 or uint32, got %s", e)
			}
			return types.AtomicOf(e), nil
		}
		return nil, diag(t.Span, "unknown generic type %q", t.Name)
	default:
		return nil, fmt.Errorf("unknown type expression %T", te)
	}
}

func (c *Checker) visible(name, file string) bool {
	owner := c.owners[name]
	return owner == "" || c.imports[strings.TrimSuffix(file, ".tach")][owner]
}

func (c *Checker) lowerFunctions() error {
	var diagnostics source.Diagnostics
	var declarations []*ast.FunctionDecl
	for _, d := range c.ast.Decls {
		if function, ok := d.(*ast.FunctionDecl); ok && (len(function.Indices) > 0 || !function.Exported) {
			declarations = append(declarations, function)
		}
	}
	functions := make([]*ir.Function, len(declarations))
	errors := parallel(c.workers, len(declarations), func(index int) error {
		local := *c
		local.mod = &ir.Module{}
		declaration := declarations[index]
		var err error
		if len(declaration.Indices) > 0 {
			err = local.lowerStage(declaration)
		} else {
			err = local.lowerHelper(declaration)
		}
		if err == nil {
			functions[index] = local.mod.Functions[0]
		}
		return err
	})
	for index, err := range errors {
		if err != nil {
			diagnostics = appendError(diagnostics, err)
		} else {
			c.mod.Functions = append(c.mod.Functions, functions[index])
		}
	}
	if len(diagnostics) > 0 {
		return diagnostics
	}
	return nil
}

func inferBufferAccess(m *ir.Module) {
	for _, f := range m.Functions {
		for i := range f.BufferParams {
			f.BufferParams[i].Access = ir.Read
		}
		roots := map[ir.PlaceID]int{}
		var walk func(*ir.Block)
		walk = func(block *ir.Block) {
			for _, in := range block.Instrs {
				switch x := in.(type) {
				case *ir.PlaceRoot:
					roots[x.Result] = x.Buffer
				case *ir.PlaceField:
					if resource, ok := roots[x.Base]; ok {
						roots[x.Result] = resource
					}
				case *ir.PlaceIndex:
					if resource, ok := roots[x.Base]; ok {
						roots[x.Result] = resource
					}
				case *ir.Store:
					markBufferWritable(f, roots, x.Place)
				case *ir.Atomic:
					if x.Op != ir.AtomicLoad {
						markBufferWritable(f, roots, x.Place)
					}
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
}

func markBufferWritable(f *ir.Function, roots map[ir.PlaceID]int, place ir.PlaceID) {
	buffer, ok := roots[place]
	if ok && buffer >= 0 && buffer < len(f.BufferParams) {
		f.BufferParams[buffer].Access = ir.Mutable
	}
}

func (c *Checker) lowerHelper(d *ast.FunctionDecl) error {
	if len(d.Attrs) > 0 {
		return diag(d.Span, "attributes are invalid on helper %s", d.Name)
	}
	sig := c.funcs[d.Name]
	f := &ir.Function{Name: d.Name, Kind: ir.Helper, Return: sig.ret, Body: &ir.Block{}, Span: d.Span}
	e := newEnv()
	b := &fnBuilder{c: c, fn: f, ids: &idAllocator{}, block: f.Body, top: true}
	for _, p := range sig.params {
		id := b.value()
		f.Params = append(f.Params, ir.Param{Name: p.name, ID: id, Type: p.ty})
		e.syms[p.name] = symbol{name: p.name, ty: p.ty, value: id, buffer: -1, workgroup: -1}
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
func (c *Checker) lowerStage(d *ast.FunctionDecl) error {
	if len(d.Indices) < 1 || len(d.Indices) > 3 {
		return diag(d.Span, "kernel %s requires 1 to 3 logical indices", d.Name)
	}
	wg, err := workgroup(d.Attrs, len(d.Indices))
	if err != nil {
		return err
	}
	f := &ir.Function{Name: d.Name, Kind: ir.Stage, Return: types.TVoid, Body: &ir.Block{}, Workgroup: wg, Span: d.Span}
	e := newEnv()
	b := &fnBuilder{c: c, fn: f, ids: &idAllocator{}, block: f.Body, top: true}
	for _, index := range d.Indices {
		if _, used := e.syms[index.Name]; used {
			return diag(index.Span, "duplicate logical index %q", index.Name)
		}
		id := b.value()
		f.Indices = append(f.Indices, ir.Param{Name: index.Name, ID: id, Type: types.TU32})
		e.syms[index.Name] = symbol{name: index.Name, ty: types.TU32, value: id, buffer: -1, workgroup: -1}
	}
	hasBuffer := false
	for _, p := range d.Params {
		if _, used := e.syms[p.Name]; used {
			return diag(p.Span, "duplicate parameter %q", p.Name)
		}
		ty, buffer, err := c.parameterType(p.Type, true)
		if err != nil {
			return err
		}
		if buffer {
			hasBuffer = true
			idx := len(f.BufferParams)
			f.BufferParams = append(f.BufferParams, ir.BufferParam{Name: p.Name, Type: ty, Access: ir.Read, Span: p.Span})
			e.syms[p.Name] = symbol{name: p.Name, ty: ty, buffer: idx, workgroup: -1}
			f.SourceParams = append(f.SourceParams, ir.SourceParam{Name: p.Name, Kind: ir.SourceBuffer, Buffer: idx})
			continue
		}
		id := b.value()
		f.Params = append(f.Params, ir.Param{Name: p.Name, ID: id, Type: ty})
		e.syms[p.Name] = symbol{name: p.Name, ty: ty, value: id, buffer: -1, workgroup: -1}
		f.SourceParams = append(f.SourceParams, ir.SourceParam{Name: p.Name, Kind: ir.SourceValue, Value: id, Buffer: -1})
	}
	if !hasBuffer {
		return diag(d.Span, "kernel %s requires at least one buffer parameter", d.Name)
	}
	if err := c.lowerBlock(b, d.Body, e, "function"); err != nil {
		return err
	}
	if f.Body.Term == nil {
		f.Body.Term = &ir.Return{}
	}
	if !f.Workgroup.Explicit && (len(f.WorkgroupVars) > 0 || blockHasBarrier(f.Body)) {
		return diag(d.Span, "stage %s uses workgroup-scoped state or barriers and requires explicit @workgroup", d.Name)
	}
	c.mod.Functions = append(c.mod.Functions, f)
	return nil
}
func (c *Checker) parameterType(te ast.TypeExpr, allowBuffer bool) (*types.Type, bool, error) {
	g, ok := te.(*ast.GenericType)
	if !ok || g.Name != "buffer" {
		t, err := c.resolveType(te)
		if err != nil {
			return nil, false, err
		}
		if t.Kind == types.Void || !types.IsConstructible(t) {
			return nil, false, diag(te.GetSpan(), "kernel value parameter has invalid type %s", t)
		}
		return t, false, nil
	}
	if !allowBuffer {
		return nil, false, diag(te.GetSpan(), "buffer<T> is not valid here")
	}
	if len(g.Args) != 1 {
		return nil, false, diag(g.Span, "buffer<T> takes exactly one type argument")
	}
	t, err := c.resolveType(g.Args[0])
	if err != nil {
		return nil, false, err
	}
	if !types.IsHostShareable(t) {
		return nil, false, diag(g.Span, "buffer type %s is not host-shareable", t)
	}
	if _, err := layout.Of(t); err != nil {
		return nil, false, diag(g.Span, "buffer layout: %v", err)
	}
	return t, true, nil
}

func workgroup(attrs []ast.Attribute, dimensions int) (ir.WorkgroupConstraint, error) {
	out := ir.WorkgroupConstraint{}
	found := false
	for _, a := range attrs {
		if a.Name != "workgroup" {
			return out, diag(a.Span, "unknown kernel attribute @%s", a.Name)
		}
		if found {
			return out, diag(a.Span, "duplicate @workgroup")
		}
		found = true
		if len(a.Args) < 1 || len(a.Args) > dimensions {
			return out, diag(a.Span, "@workgroup expects 1 to %d integer arguments for this kernel", dimensions)
		}
		out = ir.WorkgroupConstraint{Explicit: true, Size: [3]uint32{1, 1, 1}}
		for i, e := range a.Args {
			v, err := constU32(e)
			if err != nil {
				return out, err
			}
			if v == 0 {
				return out, diag(e.GetSpan(), "workgroup dimension must be positive")
			}
			out.Size[i] = v
		}
		limits := [3]uint32{256, 256, 64}
		invocations := uint64(1)
		for i, dimension := range out.Size {
			if dimension > limits[i] {
				return out, diag(a.Span, "@workgroup dimension %d exceeds Tach's portable limit %d", i, limits[i])
			}
			invocations *= uint64(dimension)
		}
		if invocations > 256 {
			return out, diag(a.Span, "@workgroup contains %d invocations; Tach's portable limit is 256", invocations)
		}
	}
	return out, nil
}

func blockHasBarrier(block *ir.Block) bool {
	for _, instruction := range block.Instrs {
		switch x := instruction.(type) {
		case *ir.Barrier:
			return true
		case *ir.If:
			if blockHasBarrier(x.Then) || blockHasBarrier(x.Else) {
				return true
			}
		case *ir.Loop:
			if blockHasBarrier(x.Cond) || blockHasBarrier(x.Body) {
				return true
			}
		case *ir.Scope:
			if blockHasBarrier(x.Body) {
				return true
			}
		}
	}
	return false
}
func constU32(e ast.Expr) (uint32, error) {
	n, ok := e.(*ast.NumberExpr)
	if !ok {
		return 0, diag(e.GetSpan(), "expected integer literal")
	}
	s, basePrefixed := splitNumberLiteral(n.Raw)
	if !basePrefixed && strings.ContainsAny(s, ".eE") {
		return 0, diag(e.GetSpan(), "expected integer literal")
	}
	v, err := strconv.ParseUint(s, 0, 32)
	if err != nil {
		return 0, diag(e.GetSpan(), "integer literal out of uint32 range")
	}
	return uint32(v), nil
}

func splitNumberLiteral(raw string) (body string, basePrefixed bool) {
	body = strings.ReplaceAll(raw, "_", "")
	basePrefixed = strings.HasPrefix(body, "0x") || strings.HasPrefix(body, "0X") || strings.HasPrefix(body, "0b") || strings.HasPrefix(body, "0B")
	return body, basePrefixed
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
		if b.fn.Kind != ir.Stage {
			return diag(x.Span, "shared variables are only valid inside kernels")
		}
		if !b.top {
			return diag(x.Span, "shared variables must be declared in the kernel body, not a nested block")
		}
		if _, ok := e.syms[x.Name]; ok {
			return diag(x.Span, "%q is already defined in this scope", x.Name)
		}
		ty, err := c.resolveType(x.Type)
		if err != nil {
			return err
		}
		if !types.IsWorkgroupStorable(ty) {
			return diag(x.Span, "shared variable %s has invalid type %s", x.Name, ty)
		}
		idx := len(b.fn.WorkgroupVars)
		b.fn.WorkgroupVars = append(b.fn.WorkgroupVars, ir.WorkgroupVar{Name: x.Name, Type: ty, Span: x.Span})
		e.syms[x.Name] = symbol{name: x.Name, ty: ty, buffer: -1, workgroup: idx}
		return nil
	case *ast.VarStmt:
		if _, ok := e.syms[x.Name]; ok {
			return diag(x.Span, "%q is already defined in this scope", x.Name)
		}
		var expected *types.Type
		var err error
		if x.Type != nil {
			expected, err = c.resolveType(x.Type)
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
		e.syms[x.Name] = symbol{name: x.Name, ty: t, value: id, mutable: x.Mutable, buffer: -1, workgroup: -1}
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
	case *ast.BreakStmt:
		if b.loop == nil {
			return diag(x.Span, "break is only valid inside a loop")
		}
		b.block.Term = &ir.Break{Values: loopValues(b.loop.names, e)}
		return nil
	case *ast.ContinueStmt:
		if b.loop == nil {
			return diag(x.Span, "continue is only valid inside a loop")
		}
		return c.continueLoop(b, e)
	default:
		return fmt.Errorf("unknown statement %T", s)
	}
}
func (c *Checker) lowerAssign(b *fnBuilder, e env, target ast.Expr, op string, rhs ast.Expr, span source.Span) error {
	if id, ok := target.(*ast.IdentExpr); ok {
		sym, exists := e.syms[id.Name]
		if exists && sym.buffer < 0 && sym.workgroup < 0 {
			if !sym.mutable {
				return diag(target.GetSpan(), "cannot assign to immutable value %s", id.Name)
			}
			var nv ir.ValueID
			var nt *types.Type
			var err error
			if op == "=" {
				nv, nt, err = c.lowerExpr(b, e, rhs, sym.ty)
			} else {
				binOp := strings.TrimSuffix(op, "=")
				want := binaryRHSExpected(sym.ty, rhs, binOp)
				rv, rt, e2 := c.lowerExpr(b, e, rhs, want)
				if e2 != nil {
					return e2
				}
				if binOp == "<<" || binOp == ">>" {
					rv, rt, e2 = c.prepareShiftCount(b, rv, rt, sym.ty, rhs.GetSpan())
					if e2 != nil {
						return e2
					}
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
		want := binaryRHSExpected(pt, rhs, binOp)
		rv, rt, e2 := c.lowerExpr(b, e, rhs, want)
		if e2 != nil {
			return e2
		}
		if binOp == "<<" || binOp == ">>" {
			rv, rt, e2 = c.prepareShiftCount(b, rv, rt, pt, rhs.GetSpan())
			if e2 != nil {
				return e2
			}
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
		if set[name] && s.buffer < 0 && s.workgroup < 0 && s.mutable {
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
	elseBlock := &ir.Block{}
	eb := b.child(elseBlock)
	ee := e.clone()
	if x.Else != nil {
		if err := c.lowerBlock(eb, x.Else, ee, "branch"); err != nil {
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

func (c *Checker) continueLoop(b *fnBuilder, e env) error {
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

func (c *Checker) lowerLoop(b *fnBuilder, e env, cond ast.Expr, body *ast.BlockStmt, post ast.Stmt, span source.Span) error {
	blocks := []*ast.BlockStmt{body}
	if post != nil {
		blocks = append(blocks, &ast.BlockStmt{Stmts: []ast.Stmt{post}})
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
	cv, ct, err := c.lowerExpr(cb, loopEnv, cond, types.TBool)
	if err != nil {
		return err
	}
	if !types.Equal(ct, types.TBool) {
		return diag(cond.GetSpan(), "loop condition is %s, want bool", ct)
	}
	condBlock.Term = &ir.Yield{Values: []ir.ValueID{cv}}
	bodyBlock := &ir.Block{}
	bb := b.child(bodyBlock)
	bb.loop = &loopContext{names: names, base: loopEnv, post: post}
	bodyEnv := loopEnv.clone()
	if err := c.lowerBlock(bb, body, bodyEnv, "loop"); err != nil {
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

func (c *Checker) lowerWhile(b *fnBuilder, e env, x *ast.WhileStmt) error {
	return c.lowerLoop(b, e, x.Cond, x.Body, nil, x.Span)
}

func (c *Checker) lowerFor(b *fnBuilder, e env, x *ast.ForStmt) error {
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
			if s.buffer >= 0 {
				if !types.IsConstructible(s.ty) {
					return 0, nil, diag(v.Span, "runtime-sized resource %s must be accessed through its fixed fields or indexed tail", v.Name)
				}
				p := b.place()
				b.emit(&ir.PlaceRoot{Result: p, Type: s.ty, Buffer: s.buffer, Span: v.Span})
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
			resultType := types.Vec(bt.Elem, len(values))
			result := b.value()
			b.emit(&ir.Composite{Result: result, Type: resultType, Values: values, Span: v.Span})
			return result, resultType, nil
		}
		return 0, nil, diag(v.Span, "member access on %s", bt)
	case *ast.IndexExpr:
		if p, pt, err := c.lowerPlace(b, e, v); err == nil {
			id := b.value()
			b.emit(&ir.Load{Result: id, Type: pt, Place: p, Span: v.Span})
			return id, pt, nil
		}
		base, bt, err := c.lowerExpr(b, e, v.Base, nil)
		if err != nil {
			return 0, nil, err
		}
		if bt.Kind != types.Vector {
			return 0, nil, diag(v.Span, "indexing a value requires a vector, got %s", bt)
		}
		index, it, err := c.lowerExpr(b, e, v.Index, types.TU32)
		if err != nil {
			return 0, nil, err
		}
		if !types.IsInteger(it) {
			return 0, nil, diag(v.Index.GetSpan(), "vector index must be int32 or uint32, got %s", it)
		}
		result := b.value()
		b.emit(&ir.VectorIndex{Result: result, Type: bt.Elem, Base: base, Index: index, Span: v.Span})
		return result, bt.Elem, nil
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
		if v.Op == "-" && want == nil {
			if n, ok := v.X.(*ast.NumberExpr); ok {
				want = types.TI32
				raw, basePrefixed := splitNumberLiteral(n.Raw)
				if !basePrefixed && strings.ContainsAny(raw, ".eE") {
					want = types.TF32
				}
			}
		}
		id, t, err := c.lowerExpr(b, e, v.X, want)
		if err != nil {
			return 0, nil, err
		}
		if v.Op == "!" && !types.Equal(t, types.TBool) {
			return 0, nil, diag(v.Span, "! requires bool")
		}
		if v.Op == "-" && !types.IsSignedNumeric(t) {
			return 0, nil, diag(v.Span, "unary - requires a signed numeric scalar or vector")
		}
		if v.Op == "~" && !types.IsIntegerLike(t) {
			return 0, nil, diag(v.Span, "unary ~ requires int32/uint32 or an integer vector")
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
		elseBlock := &ir.Block{}
		eb := b.child(elseBlock)
		thenNumber, thenIsNumber := v.Then.(*ast.NumberExpr)
		elseNumber, elseIsNumber := v.Else.(*ast.NumberExpr)
		branchType := numberPairExpected(thenNumber, elseNumber, expected)
		var tv, ev ir.ValueID
		var tt, et *types.Type
		if expected == nil && thenIsNumber && !elseIsNumber && !floatNumber(thenNumber) {
			ev, et, err = c.lowerExpr(eb, e.clone(), v.Else, nil)
			if err == nil {
				tv, tt, err = c.lowerExpr(tb, e.clone(), v.Then, et)
			}
		} else {
			tv, tt, err = c.lowerExpr(tb, e.clone(), v.Then, branchType)
			if err == nil {
				branchType = expected
				if branchType == nil {
					branchType = tt
				}
				ev, et, err = c.lowerExpr(eb, e.clone(), v.Else, branchType)
			}
		}
		if err != nil {
			return 0, nil, err
		}
		thenBlock.Term = &ir.Yield{Values: []ir.ValueID{tv}}
		elseBlock.Term = &ir.Yield{Values: []ir.ValueID{ev}}
		if !types.Equal(tt, et) {
			return 0, nil, diag(v.Span, "conditional branches have types %s and %s", tt, et)
		}
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
	raw, basePrefixed := splitNumberLiteral(n.Raw)
	isFloatSpelling := !basePrefixed && strings.ContainsAny(raw, ".eE")

	var t *types.Type
	if expected != nil && types.IsNumericScalar(expected) {
		t = expected
	} else if isFloatSpelling {
		t = types.TF32
	} else {
		// Whole-number literals are naturally non-negative. Unary minus selects
		// int32 when no surrounding context supplies a type.
		t = types.TU32
	}

	// Canonicalize literals in Core IR. Backends never need to understand Tach
	// digit separators or base prefixes.
	canonical := raw
	switch t.Kind {
	case types.U32:
		if isFloatSpelling {
			return 0, nil, diag(n.Span, "uint32 literal must be an integer")
		}
		v, err := strconv.ParseUint(raw, 0, 32)
		if err != nil {
			return 0, nil, diag(n.Span, "uint32 literal out of range")
		}
		canonical = strconv.FormatUint(v, 10)
	case types.I32:
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
	case types.F16, types.F32:
		if basePrefixed {
			return 0, nil, diag(n.Span, "base-prefixed integer literal requires an integer context or explicit conversion")
		}
		bits := 32
		name := "float32"
		if t.Kind == types.F16 {
			bits, name = 64, "float16"
		}
		f, err := strconv.ParseFloat(raw, bits)
		if err != nil || math.IsInf(f, 0) || math.IsNaN(f) {
			return 0, nil, diag(n.Span, "invalid %s literal", name)
		}
		if t.Kind == types.F16 {
			if _, ok := types.Float16bits(f); !ok {
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
		id, _, err := c.lowerExpr(eb, e.clone(), &ast.BoolExpr{Value: false, Span: x.Span}, types.TBool)
		if err != nil {
			return 0, nil, err
		}
		els.Term = &ir.Yield{Values: []ir.ValueID{id}}
	} else {
		id, _, err := c.lowerExpr(tb, e.clone(), &ast.BoolExpr{Value: true, Span: x.Span}, types.TBool)
		if err != nil {
			return 0, nil, err
		}
		then.Term = &ir.Yield{Values: []ir.ValueID{id}}
		rv, rt, err := c.lowerExpr(eb, e.clone(), x.Right, types.TBool)
		if err != nil {
			return 0, nil, err
		}
		if !types.Equal(rt, types.TBool) {
			return 0, nil, diag(x.Right.GetSpan(), "logical operand is %s, want bool", rt)
		}
		els.Term = &ir.Yield{Values: []ir.ValueID{rv}}
	}
	r := b.value()
	b.emit(&ir.If{Results: []ir.Result{{ID: r, Type: types.TBool}}, Cond: left, Then: then, Else: els, Span: x.Span})
	return r, types.TBool, nil
}
func (c *Checker) lowerBinaryExpr(b *fnBuilder, e env, x *ast.BinaryExpr, expected *types.Type) (ir.ValueID, *types.Type, error) {
	// Tach shifts use an unsigned count with the shifted value's vector width.
	// Resolve that directly instead of relying on ordinary binary contextual typing.
	if x.Op == "<<" || x.Op == ">>" {
		l, lt, err := c.lowerExpr(b, e, x.Left, expected)
		if err != nil {
			return 0, nil, err
		}
		countType := types.ShiftCountType(lt)
		if countType == nil {
			return 0, nil, diag(x.Left.GetSpan(), "%s requires an int32/uint32 scalar or integer vector on the left, got %s", x.Op, lt)
		}
		want := countType
		if countType.Kind == types.Vector {
			want = binaryRHSExpected(lt, x.Right, x.Op)
		}
		r, rt, err := c.lowerExpr(b, e, x.Right, want)
		if err != nil {
			return 0, nil, err
		}
		r, rt, err = c.prepareShiftCount(b, r, rt, lt, x.Right.GetSpan())
		if err != nil {
			return 0, nil, err
		}
		return c.emitBinary(b, x.Op, l, lt, r, rt, x.Span)
	}

	var l ir.ValueID
	var lt *types.Type
	var r ir.ValueID
	var rt *types.Type
	var err error
	leftNumber, ln := x.Left.(*ast.NumberExpr)
	rightNumber, rn := x.Right.(*ast.NumberExpr)
	expected = numberPairExpected(leftNumber, rightNumber, expected)
	if ln && !rn && expected == nil && !floatNumber(leftNumber) {
		r, rt, err = c.lowerExpr(b, e, x.Right, nil)
		if err != nil {
			return 0, nil, err
		}
		want := rt
		if rt.Kind == types.Vector && vectorScalarOperator(x.Op) {
			want = rt.Elem
		}
		l, lt, err = c.lowerExpr(b, e, x.Left, want)
	} else {
		l, lt, err = c.lowerExpr(b, e, x.Left, expected)
		if err == nil {
			want := binaryRHSExpected(lt, x.Right, x.Op)
			r, rt, err = c.lowerExpr(b, e, x.Right, want)
		}
	}
	if err != nil {
		return 0, nil, err
	}
	return c.emitBinary(b, x.Op, l, lt, r, rt, x.Span)
}

func numberPairExpected(left, right *ast.NumberExpr, expected *types.Type) *types.Type {
	if expected != nil || left == nil || right == nil {
		return expected
	}
	if floatNumber(left) || floatNumber(right) {
		return types.TF32
	}
	return nil
}

func floatNumber(number *ast.NumberExpr) bool {
	raw, basePrefixed := splitNumberLiteral(number.Raw)
	return !basePrefixed && strings.ContainsAny(raw, ".eE")
}

func vectorScalarOperator(op string) bool {
	switch op {
	case "+", "-", "*", "/", "%", "&", "|", "^":
		return true
	}
	return false
}

func binaryRHSExpected(left *types.Type, right ast.Expr, op string) *types.Type {
	if left == nil || left.Kind != types.Vector || (!vectorScalarOperator(op) && op != "<<" && op != ">>") {
		if op == "<<" || op == ">>" {
			return types.ShiftCountType(left)
		}
		return left
	}
	if _, literal := right.(*ast.NumberExpr); literal {
		if op == "<<" || op == ">>" {
			return types.TU32
		}
		return left.Elem
	}
	return nil
}

func (c *Checker) prepareShiftCount(b *fnBuilder, value ir.ValueID, got, shifted *types.Type, span source.Span) (ir.ValueID, *types.Type, error) {
	want := types.ShiftCountType(shifted)
	if want == nil {
		return 0, nil, diag(span, "shift requires an int32/uint32 scalar or integer vector")
	}
	if want.Kind == types.Vector && types.Equal(got, types.TU32) {
		value = c.splat(b, value, want, span)
		got = want
	}
	if !types.Equal(got, want) {
		return 0, nil, diag(span, "shift count is %s, want uint32 or %s", got, want)
	}
	value, got = c.normalizeShiftCount(b, value, got, span)
	return value, got, nil
}

// normalizeShiftCount gives Tach one backend-independent shift meaning: every
// 32-bit shift uses the low five bits of its count.
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
	if vectorScalarOperator(op) {
		if lt.Kind == types.Vector && types.Equal(rt, lt.Elem) && op != "*" && op != "/" {
			r, rt = c.splat(b, r, lt, span), lt
		} else if rt.Kind == types.Vector && types.Equal(lt, rt.Elem) && op != "*" {
			l, lt = c.splat(b, l, rt, span), rt
		}
	}
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
			return 0, nil, diag(span, "%s requires matching int32/uint32 scalar or integer-vector operands; got %s and %s", op, lt, rt)
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
	case "rsqrt":
		return ir.IntrinsicRSqrt, true
	case "pow":
		return ir.IntrinsicPow, true
	case "min":
		return ir.IntrinsicMin, true
	case "max":
		return ir.IntrinsicMax, true
	case "clamp":
		return ir.IntrinsicClamp, true
	case "fma":
		return ir.IntrinsicFma, true
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

func ReservedName(name string) bool {
	if _, ok := intrinsicBuiltin(name); ok {
		return true
	}
	if _, ok := atomicBuiltin(name); ok {
		return true
	}
	if name == "break" || name == "continue" || name == "workgroupBarrier" || name == "bufferBarrier" || name == "run" || name == "over" || name == "transient" || name == "ceilDiv" || name == "view" || name == "srgb8" {
		return true
	}
	return types.ParseBuiltin(name) != nil
}

func viewType(expression ast.TypeExpr) (flow.ViewFormat, bool) {
	generic, ok := expression.(*ast.GenericType)
	if !ok || generic.Name != "view" || len(generic.Args) != 1 {
		return 0, false
	}
	format, ok := generic.Args[0].(*ast.NamedType)
	return flow.SRGB8, ok && format.Name == "srgb8"
}

func intrinsicArity(kind ir.IntrinsicKind) int {
	switch kind {
	case ir.IntrinsicPow, ir.IntrinsicMin, ir.IntrinsicMax, ir.IntrinsicDot, ir.IntrinsicDistance, ir.IntrinsicCross:
		return 2
	case ir.IntrinsicClamp, ir.IntrinsicFma:
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
		firstExpected = nil // a component result does not determine vector width
	}
	v, t, err := c.lowerExpr(b, e, x.Args[0], firstExpected)
	if err != nil {
		return 0, nil, err
	}
	args[0], argTypes[0] = v, t
	for i := 1; i < wantN; i++ {
		want := argTypes[0]
		broadcast := intrinsicBroadcastsScalars(kind) && argTypes[0].Kind == types.Vector
		if broadcast {
			want = nil
			if _, literal := x.Args[i].(*ast.NumberExpr); literal {
				want = argTypes[0].Elem
			}
		}
		v, t, err := c.lowerExpr(b, e, x.Args[i], want)
		if err != nil {
			return 0, nil, err
		}
		if broadcast && types.Equal(t, argTypes[0].Elem) {
			v, t = c.splat(b, v, argTypes[0], x.Args[i].GetSpan()), argTypes[0]
		}
		args[i], argTypes[i] = v, t
	}

	floatVec := func(t *types.Type) bool {
		return t != nil && t.Kind == types.Vector && types.IsFloatLike(t.Elem)
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
		ok := types.IsSignedNumeric(t)
		if !ok {
			return 0, nil, diag(x.Span, "abs requires a signed numeric scalar or vector, got %s", t)
		}
	case ir.IntrinsicFloor, ir.IntrinsicCeil, ir.IntrinsicTrunc, ir.IntrinsicSin, ir.IntrinsicCos, ir.IntrinsicTan, ir.IntrinsicExp, ir.IntrinsicExp2, ir.IntrinsicLog, ir.IntrinsicLog2, ir.IntrinsicSqrt, ir.IntrinsicRSqrt:
		if !types.IsFloatLike(argTypes[0]) {
			return 0, nil, diag(x.Span, "%s requires a floating-point scalar/vector, got %s", kind, argTypes[0])
		}
	case ir.IntrinsicPow:
		if !same() || !types.IsFloatLike(argTypes[0]) {
			return 0, nil, diag(x.Span, "pow requires matching floating-point scalar/vector operands")
		}
	case ir.IntrinsicMin, ir.IntrinsicMax:
		// DECISION: Float bounds stay unavailable until Tach defines NaN and
		// signed-zero behavior; integer-only semantics are identical everywhere.
		if !same() || !types.IsIntegerLike(argTypes[0]) {
			return 0, nil, diag(x.Span, "%s requires matching integer scalar/vector operands", kind)
		}
	case ir.IntrinsicClamp:
		// See the float-bound decision above; clamp inherits the same ceiling.
		if !same() || !types.IsIntegerLike(argTypes[0]) {
			return 0, nil, diag(x.Span, "clamp requires three matching integer scalar/vector operands")
		}
	case ir.IntrinsicFma:
		if !same() || !types.IsFloatLike(argTypes[0]) {
			return 0, nil, diag(x.Span, "fma requires three matching floating-point scalar/vector operands")
		}
	case ir.IntrinsicDot:
		if !same() || !floatVec(argTypes[0]) {
			return 0, nil, diag(x.Span, "dot requires matching floating-point vectors")
		}
		out = argTypes[0].Elem
	case ir.IntrinsicLength:
		if !floatVec(argTypes[0]) {
			return 0, nil, diag(x.Span, "length requires a floating-point vector")
		}
		out = argTypes[0].Elem
	case ir.IntrinsicDistance:
		if !same() || !floatVec(argTypes[0]) {
			return 0, nil, diag(x.Span, "distance requires matching floating-point vectors")
		}
		out = argTypes[0].Elem
	case ir.IntrinsicCross:
		if !same() || !floatVec(argTypes[0]) || argTypes[0].Lanes != 3 {
			return 0, nil, diag(x.Span, "cross requires two matching three-lane floating-point vectors")
		}
	case ir.IntrinsicNormalize:
		if !floatVec(argTypes[0]) {
			return 0, nil, diag(x.Span, "normalize requires a floating-point vector")
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

func intrinsicBroadcastsScalars(kind ir.IntrinsicKind) bool {
	switch kind {
	case ir.IntrinsicPow, ir.IntrinsicMin, ir.IntrinsicMax, ir.IntrinsicClamp:
		return true
	}
	return false
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
			return 0, nil, diag(x.Args[0].GetSpan(), "%s requires an atomic<int32> or atomic<uint32> place, got %s", id.Name, pt)
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
	if types.IsNumericScalar(target) {
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
		if len(x.Args) == 0 {
			return 0, nil, diag(x.Span, "%s constructor requires components", target)
		}
		vals := make([]ir.ValueID, 0, target.Lanes)
		for _, argument := range x.Args {
			v, t, err := c.lowerExpr(b, e, argument, nil)
			if err != nil {
				return 0, nil, err
			}
			if types.IsNumericScalar(t) {
				v, err = c.convertScalar(b, v, t, target.Elem, argument.GetSpan())
				if err != nil {
					return 0, nil, err
				}
				if len(x.Args) == 1 {
					for range target.Lanes {
						vals = append(vals, v)
					}
				} else {
					vals = append(vals, v)
				}
				continue
			}
			if t.Kind != types.Vector || !types.IsNumeric(t) {
				return 0, nil, diag(argument.GetSpan(), "%s constructor component is not numeric", target)
			}
			if len(x.Args) == 1 && types.Equal(t, target) {
				return v, target, nil
			}
			for lane := 0; lane < t.Lanes; lane++ {
				extracted := b.value()
				b.emit(&ir.Extract{Result: extracted, Type: t.Elem, Base: v, Index: lane, Span: argument.GetSpan()})
				extracted, err = c.convertScalar(b, extracted, t.Elem, target.Elem, argument.GetSpan())
				if err != nil {
					return 0, nil, err
				}
				vals = append(vals, extracted)
			}
		}
		if len(vals) != target.Lanes {
			return 0, nil, diag(x.Span, "%s constructor received %d lanes, want %d", target, len(vals), target.Lanes)
		}
		r := b.value()
		b.emit(&ir.Composite{Result: r, Type: target, Values: vals, Span: x.Span})
		return r, target, nil
	}
	return 0, nil, diag(x.Span, "type %s is not directly constructible", target)
}

func (c *Checker) convertScalar(b *fnBuilder, value ir.ValueID, from, to *types.Type, span source.Span) (ir.ValueID, error) {
	if types.Equal(from, to) {
		return value, nil
	}
	if !types.IsNumericScalar(from) || !types.IsNumericScalar(to) {
		return 0, diag(span, "cannot convert %s to %s", from, to)
	}
	result := b.value()
	b.emit(&ir.Convert{Result: result, Type: to, X: value, From: from, Span: span})
	return result, nil
}

func (c *Checker) splat(b *fnBuilder, value ir.ValueID, vector *types.Type, span source.Span) ir.ValueID {
	values := make([]ir.ValueID, vector.Lanes)
	for index := range values {
		values[index] = value
	}
	result := b.value()
	b.emit(&ir.Composite{Result: result, Type: vector, Values: values, Span: span})
	return result
}

func (c *Checker) lowerPlace(b *fnBuilder, e env, x ast.Expr) (ir.PlaceID, *types.Type, error) {
	switch v := x.(type) {
	case *ast.IdentExpr:
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
	case *ast.MemberExpr:
		bp, bt, err := c.lowerPlace(b, e, v.Base)
		if err != nil {
			return 0, nil, err
		}
		if bt.Kind == types.Vector {
			components, ok := vectorComponents(v.Name, bt.Lanes)
			if !ok || len(components) != 1 {
				return 0, nil, diag(v.Span, "vector %s has no component %s", bt, v.Name)
			}
			iv := b.value()
			b.emit(&ir.Const{Result: iv, Type: types.TU32, Raw: fmt.Sprintf("%d", components[0]), Span: v.Span})
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
func diag(span source.Span, f string, a ...any) error {
	return &source.Diagnostic{Span: span, Message: fmt.Sprintf(f, a...)}
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
