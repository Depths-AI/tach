package sema

import (
	"errors"
	"fmt"
	"maps"
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
	ast           *ast.Module
	mod           *ir.Module
	flow          *flow.Module
	types         map[string]*types.Type
	consts        map[string]*constantDef
	funcs         map[string]*funcSig
	owners        map[string]string
	imports       map[string]map[string]bool
	workers       int
	constantStack []string
}

type constantDef struct {
	decl  *ast.ConstDecl
	value *types.Value
	err   error
	state uint8
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
	ty        *types.Type
	value     ir.ValueID
	constant  *types.Value
	mutable   bool
	buffer    int // -1 unless this is a stage buffer place
	workgroup int // -1 unless this is a function workgroup place
}
type env struct{ syms map[string]symbol }

func newEnv() env        { return env{syms: map[string]symbol{}} }
func (e env) clone() env { return env{maps.Clone(e.syms)} }

type idAllocator struct {
	nextValue ir.ValueID
	nextPlace ir.PlaceID
}

type fnBuilder struct {
	fn       *ir.Function
	ids      *idAllocator
	block    *ir.Block
	top      bool
	loop     *loopContext
	comptime bool
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
	return &fnBuilder{fn: b.fn, ids: b.ids, block: block, loop: b.loop, comptime: b.comptime}
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
	c := &Checker{ast: merged, mod: kernel, flow: &flow.Module{Kernel: kernel, Documentation: documentation}, types: map[string]*types.Type{}, consts: map[string]*constantDef{}, funcs: map[string]*funcSig{}, owners: map[string]string{}, imports: map[string]map[string]bool{}, workers: workers}
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
			case *ast.ConstDecl:
				c.owners[item.Name] = file
			case *ast.FunctionDecl:
				c.owners[item.Name] = file
			}
		}
	}
	for _, n := range []string{"void", "bool", "int32", "uint32", "float16", "float32"} {
		c.types[n] = types.ParseBuiltin(n)
	}
	var interfaceDiagnostics source.Diagnostics
	for _, check := range []func() error{c.collectTypes, c.collectConstants, c.resolveTypeFields, c.checkRuntimeArrayPlacement, c.checkTypeCycles, c.collectFunctions} {
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
		var diagnostic *source.Diagnostic
		if errors.As(err, &diagnostic) {
			return nil, nil, appendError(documentationDiagnostics, diagnostic).Sorted()
		}
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
	var list source.Diagnostics
	if errors.As(err, &list) {
		return append(diagnostics, list...)
	}
	var diagnostic *source.Diagnostic
	if errors.As(err, &diagnostic) {
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
			if x.Exported && !buffer && !types.IsHostParameter(t) {
				return diag(p.Type.GetSpan(), "public value parameter type %s cannot cross the host parameter ABI", t)
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
		if _, exists := c.funcs[sig.name]; exists || c.consts[sig.name] != nil || c.types[sig.name] != nil {
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
	return c.resolveTypeIn(te, nil)
}

func (c *Checker) resolveTypeIn(te ast.TypeExpr, environment *env) (*types.Type, error) {
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
		e, err := c.resolveTypeIn(t.Elem, environment)
		if err != nil {
			return nil, err
		}
		if e.Kind == types.Void || e.Kind == types.RuntimeArray {
			return nil, diag(t.Span, "invalid runtime array element type %s", e)
		}
		return types.Runtime(e), nil
	case *ast.FixedArrayType:
		e, err := c.resolveTypeIn(t.Elem, environment)
		if err != nil {
			return nil, err
		}
		if !types.IsWorkgroupStorable(e) {
			return nil, diag(t.Span, "invalid fixed array element type %s", e)
		}
		scope := newEnv()
		if environment != nil {
			scope = *environment
		}
		value, err := c.evaluateConstant(t.Count, types.TU32, scope)
		if err != nil {
			return nil, err
		}
		if len(value.Bits) != 1 || value.Bits[0] == 0 {
			return nil, diag(t.Span, "fixed array length must be a positive uint32 constant")
		}
		return types.Array(e, value.Bits[0]), nil
	case *ast.VectorType:
		e, err := c.resolveTypeIn(t.Elem, environment)
		if err != nil {
			return nil, err
		}
		if !types.IsScalar(e) {
			return nil, diag(t.Span, "vec element type must be scalar, got %s", e)
		}
		lanes, err := strconv.Atoi(t.Lanes)
		if err != nil || lanes < 2 || lanes > 4 {
			return nil, diag(t.Span, "vec lane count must be 2, 3, or 4")
		}
		return types.Vec(e, lanes), nil
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
			e, err := c.resolveTypeIn(t.Args[0], environment)
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
	b := &fnBuilder{fn: f, ids: &idAllocator{}, block: f.Body, top: true}
	for _, p := range sig.params {
		id := b.value()
		f.Params = append(f.Params, ir.Param{Name: p.name, ID: id, Type: p.ty})
		e.syms[p.name] = symbol{ty: p.ty, value: id, buffer: -1, workgroup: -1}
	}
	if err := c.lowerBlock(b, d.Body, e); err != nil {
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
	wg, err := c.workgroup(d.Attrs, len(d.Indices))
	if err != nil {
		return err
	}
	f := &ir.Function{Name: d.Name, Kind: ir.Stage, Return: types.TVoid, Body: &ir.Block{}, Workgroup: wg, Span: d.Span}
	e := newEnv()
	b := &fnBuilder{fn: f, ids: &idAllocator{}, block: f.Body, top: true}
	for _, index := range d.Indices {
		if _, used := e.syms[index.Name]; used {
			return diag(index.Span, "duplicate logical index %q", index.Name)
		}
		id := b.value()
		f.Indices = append(f.Indices, ir.Param{Name: index.Name, ID: id, Type: types.TU32})
		e.syms[index.Name] = symbol{ty: types.TU32, value: id, buffer: -1, workgroup: -1}
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
			e.syms[p.Name] = symbol{ty: ty, buffer: idx, workgroup: -1}
			f.SourceParams = append(f.SourceParams, ir.SourceParam{Name: p.Name, Kind: ir.SourceBuffer, Buffer: idx})
			continue
		}
		id := b.value()
		f.Params = append(f.Params, ir.Param{Name: p.Name, ID: id, Type: ty})
		e.syms[p.Name] = symbol{ty: ty, value: id, buffer: -1, workgroup: -1}
		f.SourceParams = append(f.SourceParams, ir.SourceParam{Name: p.Name, Kind: ir.SourceValue, Value: id, Buffer: -1})
	}
	if !hasBuffer {
		return diag(d.Span, "kernel %s requires at least one buffer parameter", d.Name)
	}
	if err := c.lowerBlock(b, d.Body, e); err != nil {
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

func (c *Checker) workgroup(attrs []ast.Attribute, dimensions int) (ir.WorkgroupConstraint, error) {
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
			value, err := c.evaluateConstant(e, types.TU32, newEnv())
			if err != nil {
				return out, err
			}
			v := value.Bits[0]
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
func splitNumberLiteral(raw string) (body string, basePrefixed bool) {
	body = strings.ReplaceAll(raw, "_", "")
	basePrefixed = strings.HasPrefix(body, "0x") || strings.HasPrefix(body, "0X") || strings.HasPrefix(body, "0b") || strings.HasPrefix(body, "0B")
	return body, basePrefixed
}

func (c *Checker) lowerBlock(b *fnBuilder, src *ast.BlockStmt, e env) error {
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
		ty, err := c.resolveTypeIn(x.Type, &e)
		if err != nil {
			return err
		}
		if !types.IsWorkgroupStorable(ty) {
			return diag(x.Span, "shared variable %s has invalid type %s", x.Name, ty)
		}
		idx := len(b.fn.WorkgroupVars)
		b.fn.WorkgroupVars = append(b.fn.WorkgroupVars, ir.WorkgroupVar{Name: x.Name, Type: ty, Span: x.Span})
		e.syms[x.Name] = symbol{ty: ty, buffer: -1, workgroup: idx}
		return nil
	case *ast.ConstStmt:
		if _, ok := e.syms[x.Name]; ok {
			return diag(x.Span, "%q is already defined in this scope", x.Name)
		}
		value, err := c.evaluateConstantBinding(x.Type, x.Value, e)
		if err != nil {
			return err
		}
		e.syms[x.Name] = symbol{ty: value.Type, constant: value, buffer: -1, workgroup: -1}
		return nil
	case *ast.VarStmt:
		if _, ok := e.syms[x.Name]; ok {
			return diag(x.Span, "%q is already defined in this scope", x.Name)
		}
		var expected *types.Type
		var err error
		if x.Type != nil {
			expected, err = c.resolveTypeIn(x.Type, &e)
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
		if expected != nil && !types.Equal(t, expected) {
			return diag(x.Value.GetSpan(), "local value is %s, want %s", t, expected)
		}
		if t.Kind == types.Void {
			return diag(x.Value.GetSpan(), "cannot bind a void expression")
		}
		e.syms[x.Name] = symbol{ty: t, value: id, mutable: true, buffer: -1, workgroup: -1}
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
				if sym.constant != nil {
					return diag(target.GetSpan(), "cannot assign to compile-time constant %s", id.Name)
				}
				return diag(target.GetSpan(), "cannot assign to immutable value %s", id.Name)
			}
			var nv ir.ValueID
			var nt *types.Type
			var err error
			if op == "=" {
				nv, nt, err = c.lowerExpr(b, e, rhs, sym.ty)
			} else {
				nv, nt, err = c.lowerCompound(b, e, strings.TrimSuffix(op, "="), sym.value, sym.ty, rhs, span)
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
		if c.consts[id.Name] != nil && c.visible(id.Name, id.Span.File) {
			return diag(target.GetSpan(), "cannot assign to compile-time constant %s", id.Name)
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
		v, vt, err = c.lowerCompound(b, e, strings.TrimSuffix(op, "="), old, pt, rhs, span)
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
	}
	sort.Strings(out)
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
			if s.constant != nil {
				value, valueType := materializeConstant(b, s.constant, v.Span)
				return value, valueType, nil
			}
			if b.comptime {
				return 0, nil, &runtimeConstantDependency{diag(v.Span, "compile-time expression depends on runtime value %q; use let for the binding", v.Name)}
			}
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
		if definition := c.consts[v.Name]; definition != nil && c.visible(v.Name, v.Span.File) {
			value, err := c.resolveConstant(v.Name, v.Span)
			if err != nil {
				return 0, nil, err
			}
			result, resultType := materializeConstant(b, value, v.Span)
			return result, resultType, nil
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
		want := expected
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
		if v.Op == "!" && !types.IsBoolean(t) {
			return 0, nil, diag(v.Span, "! requires bool or vec<bool, N>")
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
		expressions := []ast.Expr{v.Then, v.Else}
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
						want = types.Vec(expectedElement, expectedLanes)
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
		if !types.Equal(arguments[0].type_, arguments[1].type_) {
			return 0, nil, diag(v.Span, "conditional branches have types %s and %s", arguments[0].type_, arguments[1].type_)
		}
		if expected != nil && !types.Equal(arguments[0].type_, expected) {
			return 0, nil, diag(v.Span, "conditional result is %s, context requires %s", arguments[0].type_, expected)
		}
		thenBlock, elseBlock := arguments[0].block, arguments[1].block
		thenBlock.Term = &ir.Yield{Values: []ir.ValueID{arguments[0].value}}
		elseBlock.Term = &ir.Yield{Values: []ir.ValueID{arguments[1].value}}
		r := b.value()
		b.emit(&ir.If{Results: []ir.Result{{ID: r, Type: arguments[0].type_}}, Cond: cond, Then: thenBlock, Else: elseBlock, Span: v.Span})
		return r, arguments[0].type_, nil
	case *ast.CallExpr:
		return c.lowerCall(b, e, v, expected)
	case *ast.TransientExpr:
		return 0, nil, diag(v.Span, "transient allocation is only available as a public program let binding")
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
		return c.lowerShift(b, e, x.Op, l, lt, x.Right, x.Span)
	}
	expressions := []ast.Expr{x.Left, x.Right}
	arguments := make([]detachedExpr, 2)
	if x.Op == "==" || x.Op == "!=" || x.Op == "&" || x.Op == "|" || x.Op == "^" {
		for i, expression := range expressions {
			argument, err := c.lowerDetached(b, e, expression, nil)
			if err != nil {
				return 0, nil, err
			}
			arguments[i] = argument
		}
		if types.IsBoolean(arguments[0].type_) || types.IsBoolean(arguments[1].type_) {
			return c.emitBooleanBinary(b, arguments[0], arguments[1], x.Op, x.Span)
		}
	} else {
		var err error
		arguments, err = c.lowerNumericArguments(b, e, x.Op, expressions)
		if err != nil {
			return 0, nil, err
		}
	}

	return c.emitResolvedBinary(b, e, x.Op, arguments, expected, x.Span)
}

func (c *Checker) emitResolvedBinary(b *fnBuilder, e env, op string, arguments []detachedExpr, expected *types.Type, span source.Span) (ir.ValueID, *types.Type, error) {
	var element *types.Type
	if expectedElement, _ := numericElement(expected); expectedElement != nil && op != "==" && op != "!=" && op != "<" && op != "<=" && op != ">" && op != ">=" {
		element = expectedElement
	}
	element, lanes, err := resolveNumericOperands(op, ir.NumericAny, arguments, element, 0)
	if err != nil {
		return 0, nil, diag(span, "%v", err)
	}
	vector := types.Vec(element, lanes)
	values := make([]ir.ValueID, 2)
	valueTypes := make([]*types.Type, 2)
	for i, argument := range arguments {
		_, argumentLanes := numericElement(argument.type_)
		want := element
		if argumentLanes > 0 {
			want = vector
		}
		values[i], valueTypes[i], err = c.commitArgument(b, e, argument, want)
		if err != nil {
			return 0, nil, err
		}
	}
	return c.emitBinary(b, op, values[0], valueTypes[0], values[1], valueTypes[1], span)
}

func vectorScalarOperator(op string) bool {
	switch op {
	case "+", "-", "*", "/", "%", "&", "|", "^", "==", "!=", "<", "<=", ">", ">=":
		return true
	}
	return false
}

func (c *Checker) emitBooleanBinary(b *fnBuilder, left, right detachedExpr, op string, span source.Span) (ir.ValueID, *types.Type, error) {
	if op != "==" && op != "!=" && op != "&" && op != "|" && op != "^" {
		return 0, nil, diag(span, "%s is not defined for boolean values", op)
	}
	lt, rt := left.type_, right.type_
	if !types.IsBoolean(lt) || !types.IsBoolean(rt) {
		return 0, nil, diag(span, "%s requires both operands to be bool or vec<bool, N>; got %s and %s", op, lt, rt)
	}
	lanes := 0
	if lt.Kind == types.Vector {
		lanes = lt.Lanes
	}
	if rt.Kind == types.Vector {
		if lanes != 0 && lanes != rt.Lanes {
			return 0, nil, diag(span, "%s operands use conflicting vector widths %d and %d", op, lanes, rt.Lanes)
		}
		lanes = rt.Lanes
	}
	b.block.Instrs = append(b.block.Instrs, left.block.Instrs...)
	b.block.Instrs = append(b.block.Instrs, right.block.Instrs...)
	if lanes > 0 {
		vector := types.Vec(types.TBool, lanes)
		if lt.Kind == types.Bool {
			left.value, lt = c.splat(b, left.value, vector, span), vector
		}
		if rt.Kind == types.Bool {
			right.value, rt = c.splat(b, right.value, vector, span), vector
		}
	}
	return c.emitBinary(b, op, left.value, lt, right.value, rt, span)
}

func (c *Checker) lowerCompound(b *fnBuilder, e env, op string, left ir.ValueID, leftType *types.Type, right ast.Expr, span source.Span) (ir.ValueID, *types.Type, error) {
	if op == "<<" || op == ">>" {
		return c.lowerShift(b, e, op, left, leftType, right, span)
	}
	if types.IsBoolean(leftType) {
		operand, err := c.lowerDetached(b, e, right, nil)
		if err != nil {
			return 0, nil, err
		}
		return c.emitBooleanBinary(b, detachedExpr{block: &ir.Block{}, value: left, type_: leftType}, operand, op, span)
	}
	rightArguments, err := c.lowerNumericArguments(b, e, op, []ast.Expr{right})
	if err != nil {
		return 0, nil, err
	}
	arguments := append([]detachedExpr{{block: &ir.Block{}, value: left, type_: leftType}}, rightArguments...)
	return c.emitResolvedBinary(b, e, op, arguments, leftType, span)
}

func (c *Checker) lowerShift(b *fnBuilder, e env, op string, left ir.ValueID, leftType *types.Type, right ast.Expr, span source.Span) (ir.ValueID, *types.Type, error) {
	countType := types.ShiftCountType(leftType)
	if countType == nil {
		return 0, nil, diag(span, "%s requires an int32/uint32 scalar or integer vector on the left, got %s", op, leftType)
	}
	want := countType
	if countType.Kind == types.Vector && !contextualNumeric(right) {
		want = nil
	}
	r, rt, err := c.lowerExpr(b, e, right, want)
	if err != nil {
		return 0, nil, err
	}
	r, rt, err = c.prepareShiftCount(b, r, rt, leftType, right.GetSpan())
	if err != nil {
		return 0, nil, err
	}
	return c.emitBinary(b, op, left, leftType, r, rt, span)
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
		if !types.Equal(lt, rt) || !types.IsNumeric(lt) && !((op == "==" || op == "!=") && types.IsBoolean(lt)) {
			return 0, nil, diag(span, "comparison %s requires matching numeric operands; got %s and %s", op, lt, rt)
		}
		out = types.BoolShape(lt)
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
		if !types.Equal(lt, rt) || !types.IsIntegerLike(lt) && !types.IsBoolean(lt) {
			return 0, nil, diag(span, "%s requires matching integer or boolean operands; got %s and %s", op, lt, rt)
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
	case "all":
		return ir.IntrinsicAll, true
	case "any":
		return ir.IntrinsicAny, true
	case "select":
		return ir.IntrinsicSelect, true
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
	if name == "break" || name == "continue" || name == "vec" || name == "workgroupBarrier" || name == "bufferBarrier" || name == "run" || name == "over" || name == "transient" || name == "ceilDiv" || name == "view" || name == "srgb8" {
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

type detachedExpr struct {
	block      *ir.Block
	value      ir.ValueID
	type_      *types.Type
	contextual bool
	source     ast.Expr
}

func (c *Checker) lowerDetached(b *fnBuilder, e env, expression ast.Expr, expected *types.Type) (detachedExpr, error) {
	block := &ir.Block{}
	value, type_, err := c.lowerExpr(b.child(block), e, expression, expected)
	return detachedExpr{block: block, value: value, type_: type_, contextual: contextualNumeric(expression), source: expression}, err
}

func contextualNumeric(expression ast.Expr) bool {
	switch x := expression.(type) {
	case *ast.NumberExpr:
		return true
	case *ast.UnaryExpr:
		return (x.Op == "-" || x.Op == "~") && contextualNumeric(x.X)
	case *ast.BinaryExpr:
		return contextualNumeric(x.Left) && contextualNumeric(x.Right)
	case *ast.ConditionalExpr:
		return contextualNumeric(x.Then) && contextualNumeric(x.Else)
	case *ast.CallExpr:
		callee, ok := x.Callee.(*ast.IdentExpr)
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

func numericElement(t *types.Type) (*types.Type, int) {
	if types.IsNumericScalar(t) {
		return t, 0
	}
	if t != nil && t.Kind == types.Vector && types.IsNumericScalar(t.Elem) {
		return t.Elem, t.Lanes
	}
	return nil, 0
}

func scalarElement(t *types.Type) (*types.Type, int) {
	if types.IsScalar(t) {
		return t, 0
	}
	if t != nil && t.Kind == types.Vector && types.IsScalar(t.Elem) {
		return t.Elem, t.Lanes
	}
	return nil, 0
}

func defaultNumericElement(domain ir.NumericDomain, arguments []detachedExpr) *types.Type {
	if domain == ir.NumericFloat {
		return types.TF32
	}
	for _, argument := range arguments {
		element, _ := numericElement(argument.type_)
		if element != nil && element.Kind == types.F32 {
			return types.TF32
		}
	}
	if domain == ir.NumericSigned {
		return types.TI32
	}
	for _, argument := range arguments {
		element, _ := numericElement(argument.type_)
		if element != nil && element.Kind == types.I32 {
			return types.TI32
		}
	}
	return types.TU32
}

func (c *Checker) lowerNumericArguments(b *fnBuilder, e env, operation string, expressions []ast.Expr) ([]detachedExpr, error) {
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

func resolveNumericOperands(operation string, domain ir.NumericDomain, arguments []detachedExpr, element *types.Type, lanes int) (*types.Type, int, error) {
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
			if element != nil && !types.Equal(element, argumentElement) {
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

func (c *Checker) commitArgument(b *fnBuilder, e env, argument detachedExpr, want *types.Type) (ir.ValueID, *types.Type, error) {
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

func (c *Checker) lowerIntrinsic(b *fnBuilder, e env, x *ast.CallExpr, kind ir.IntrinsicKind, expected *types.Type) (ir.ValueID, *types.Type, error) {
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

	var element *types.Type
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
	vector := types.Vec(element, lanes)
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
		if types.Equal(type_, vector) {
			args[i] = value
			continue
		}
		if types.Equal(type_, element) && lanes > 0 && rule.Broadcast&(1<<i) != 0 {
			args[i] = c.splat(b, value, vector, x.Args[i].GetSpan())
			continue
		}
		if lanes == 0 && types.Equal(type_, element) {
			args[i] = value
			continue
		}
		return 0, nil, diag(x.Args[i].GetSpan(), "%s argument is %s, want %s", kind, type_, vector)
	}
	out := element
	if !rule.ResultElement && lanes > 0 {
		out = vector
	}
	if expected != nil && !types.Equal(out, expected) {
		return 0, nil, diag(x.Span, "%s returns %s, context requires %s", kind, out, expected)
	}
	result := b.value()
	b.emit(&ir.Intrinsic{Result: result, Type: out, Kind: kind, Args: args, Span: x.Span})
	return result, out, nil
}

func (c *Checker) lowerMaskIntrinsic(b *fnBuilder, e env, x *ast.CallExpr, kind ir.IntrinsicKind, expected *types.Type) (ir.ValueID, *types.Type, error) {
	if kind == ir.IntrinsicAll || kind == ir.IntrinsicAny {
		if len(x.Args) != 1 {
			return 0, nil, diag(x.Span, "%s expects one argument, got %d", kind, len(x.Args))
		}
		mask, maskType, err := c.lowerExpr(b, e, x.Args[0], nil)
		if err != nil {
			return 0, nil, err
		}
		if maskType.Kind != types.Vector || maskType.Elem.Kind != types.Bool {
			return 0, nil, diag(x.Args[0].GetSpan(), "%s requires vec<bool, N>, got %s", kind, maskType)
		}
		if expected != nil && !types.Equal(expected, types.TBool) {
			return 0, nil, diag(x.Span, "%s returns bool, context requires %s", kind, expected)
		}
		result := b.value()
		b.emit(&ir.Intrinsic{Result: result, Type: types.TBool, Kind: kind, Args: []ir.ValueID{mask}, Span: x.Span})
		return result, types.TBool, nil
	}
	if len(x.Args) != 3 {
		return 0, nil, diag(x.Span, "select expects mask, whenTrue, and whenFalse arguments; got %d", len(x.Args))
	}
	mask, maskType, err := c.lowerExpr(b, e, x.Args[0], nil)
	if err != nil {
		return 0, nil, err
	}
	if maskType.Kind != types.Vector || maskType.Elem.Kind != types.Bool {
		return 0, nil, diag(x.Args[0].GetSpan(), "select mask must be vec<bool, N>, got %s", maskType)
	}
	arms := make([]detachedExpr, 2)
	for i := range arms {
		arms[i], err = c.lowerDetached(b, e, x.Args[i+1], nil)
		if err != nil {
			return 0, nil, err
		}
	}
	var out *types.Type
	if types.IsBoolean(arms[0].type_) || types.IsBoolean(arms[1].type_) {
		for i, arm := range arms {
			if !types.IsBoolean(arm.type_) || arm.type_.Kind == types.Vector && arm.type_.Lanes != maskType.Lanes {
				return 0, nil, diag(x.Args[i+1].GetSpan(), "select boolean arm is %s, want bool or %s", arm.type_, maskType)
			}
		}
		out = maskType
	} else {
		var element *types.Type
		if expected != nil {
			if expected.Kind != types.Vector || expected.Lanes != maskType.Lanes || !types.IsNumericScalar(expected.Elem) {
				return 0, nil, diag(x.Span, "select produces a %d-lane vector, context requires %s", maskType.Lanes, expected)
			}
			element = expected.Elem
		}
		element, _, err = resolveNumericOperands("select", ir.NumericAny, arms, element, maskType.Lanes)
		if err != nil {
			return 0, nil, diag(x.Span, "%v", err)
		}
		out = types.Vec(element, maskType.Lanes)
	}
	if expected != nil && !types.Equal(expected, out) {
		return 0, nil, diag(x.Span, "select returns %s, context requires %s", out, expected)
	}
	args := []ir.ValueID{mask, 0, 0}
	for i, arm := range arms {
		want := out
		if arm.type_.Kind != types.Vector {
			want = out.Elem
		}
		value, armType, err := c.commitArgument(b, e, arm, want)
		if err != nil {
			return 0, nil, err
		}
		if types.Equal(armType, out.Elem) {
			value, armType = c.splat(b, value, out, x.Args[i+1].GetSpan()), out
		}
		if !types.Equal(armType, out) {
			return 0, nil, diag(x.Args[i+1].GetSpan(), "select arm is %s, want %s or %s", armType, out.Elem, out)
		}
		args[i+1] = value
	}
	result := b.value()
	b.emit(&ir.Intrinsic{Result: result, Type: out, Kind: kind, Args: args, Span: x.Span})
	return result, out, nil
}

func (c *Checker) lowerVectorInference(b *fnBuilder, e env, x *ast.CallExpr, expected *types.Type) (ir.ValueID, *types.Type, error) {
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

	var element *types.Type
	if expected != nil {
		if expected.Kind != types.Vector || expected.Lanes != lanes || !types.IsScalar(expected.Elem) {
			return 0, nil, diag(x.Span, "vec produces a %d-lane vector, context requires %s", lanes, expected)
		}
		element = expected.Elem
	}
	for i, argument := range arguments {
		argumentElement, _ := scalarElement(argument.type_)
		if argument.contextual {
			continue
		}
		if element != nil && !types.Equal(element, argumentElement) {
			return 0, nil, diag(x.Args[i].GetSpan(), "vec components use %s and %s; convert explicitly", element, argumentElement)
		}
		element = argumentElement
	}
	if element == nil {
		element = defaultNumericElement(ir.NumericAny, arguments)
	}
	vector := types.Vec(element, lanes)
	values := make([]ir.ValueID, 0, lanes)
	for i, argument := range arguments {
		_, width := scalarElement(argument.type_)
		want := element
		if width > 0 {
			want = types.Vec(element, width)
		}
		base, type_, err := c.commitArgument(b, e, argument, want)
		if err != nil {
			return 0, nil, err
		}
		if types.Equal(type_, element) {
			values = append(values, base)
			continue
		}
		if type_.Kind != types.Vector || !types.Equal(type_.Elem, element) || type_.Lanes != width {
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

func (c *Checker) lowerCall(b *fnBuilder, e env, x *ast.CallExpr, expected *types.Type) (ir.ValueID, *types.Type, error) {
	id, ok := x.Callee.(*ast.IdentExpr)
	if !ok {
		return 0, nil, diag(x.Callee.GetSpan(), "call target must be a function or type name")
	}
	if target := types.ParseBuiltin(id.Name); target != nil && target.Kind != types.Void && target.Kind != types.Bool {
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
		var err error = diag(x.Span, "call to %q is not available in compile-time expressions", id.Name)
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
		return 0, types.TVoid, nil
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
		if pt.Kind != types.Atomic || (pt.Elem.Kind != types.I32 && pt.Elem.Kind != types.U32) {
			return 0, nil, diag(x.Args[0].GetSpan(), "%s requires an atomic<int32> or atomic<uint32> place, got %s", id.Name, pt)
		}
		var value, expected ir.ValueID
		for argument := 1; argument < want; argument++ {
			v, vt, err := c.lowerExpr(b, e, x.Args[argument], pt.Elem)
			if err != nil {
				return 0, nil, err
			}
			if !types.Equal(vt, pt.Elem) {
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
			return 0, types.TVoid, nil
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
	case "atomicCompareExchange":
		return ir.AtomicCompareExchange, true
	default:
		return 0, false
	}
}
func (c *Checker) lowerConstructor(b *fnBuilder, e env, x *ast.CallExpr, target *types.Type) (ir.ValueID, *types.Type, error) {
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
