package semantics

import (
	"strings"
	"tach/foundation"
	"tach/ir"
	"tach/parser"
)

func (c *analyzer) lowerFunctions() error {
	var diagnostics foundation.Diagnostics
	var declarations []*parser.FunctionDecl
	for _, d := range c.syntax.Decls {
		if function, ok := d.(*parser.FunctionDecl); ok && (len(function.Indices) > 0 || !function.Exported) {
			declarations = append(declarations, function)
		}
	}
	functions := make([]*ir.Function, len(declarations))
	errors := parallel(c.workers, len(declarations), func(index int) error {
		local := *c
		local.kernel = &ir.KernelModule{}
		declaration := declarations[index]
		var err error
		if len(declaration.Indices) > 0 {
			err = local.lowerStage(declaration)
		} else {
			err = local.lowerHelper(declaration)
		}
		if err == nil {
			functions[index] = local.kernel.Functions[0]
		}
		return err
	})
	for index, err := range errors {
		if err != nil {
			diagnostics = appendError(diagnostics, err)
		} else {
			c.kernel.Functions = append(c.kernel.Functions, functions[index])
		}
	}
	if len(diagnostics) > 0 {
		return diagnostics
	}
	return nil
}

func inferBufferAccess(m *ir.KernelModule) {
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

func (c *analyzer) lowerHelper(d *parser.FunctionDecl) error {
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
		if f.Return.Kind != foundation.VoidKind {
			return diag(d.Body.Span, "function %s can reach the end without returning %s", d.Name, f.Return)
		}
		f.Body.Term = &ir.Return{}
	}
	c.kernel.Functions = append(c.kernel.Functions, f)
	return nil
}
func (c *analyzer) lowerStage(d *parser.FunctionDecl) error {
	if len(d.Indices) < 1 || len(d.Indices) > 3 {
		return diag(d.Span, "kernel %s requires 1 to 3 logical indices", d.Name)
	}
	wg, err := c.workgroup(d.Attrs, len(d.Indices))
	if err != nil {
		return err
	}
	f := &ir.Function{Name: d.Name, Kind: ir.Stage, Return: foundation.VoidType, Body: &ir.Block{}, Workgroup: wg, Span: d.Span}
	e := newEnv()
	b := &fnBuilder{fn: f, ids: &idAllocator{}, block: f.Body, top: true}
	for _, index := range d.Indices {
		if _, used := e.syms[index.Name]; used {
			return diag(index.Span, "duplicate logical index %q", index.Name)
		}
		id := b.value()
		f.Indices = append(f.Indices, ir.Param{Name: index.Name, ID: id, Type: foundation.Uint32Type})
		e.syms[index.Name] = symbol{ty: foundation.Uint32Type, value: id, buffer: -1, workgroup: -1}
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
	c.kernel.Functions = append(c.kernel.Functions, f)
	return nil
}
func (c *analyzer) parameterType(te parser.TypeExpr, allowBuffer bool) (*foundation.Type, bool, error) {
	g, ok := te.(*parser.GenericType)
	if !ok || g.Name != "buffer" {
		t, err := c.resolveType(te)
		if err != nil {
			return nil, false, err
		}
		if t.Kind == foundation.VoidKind || !foundation.IsConstructible(t) {
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
	if !foundation.IsHostShareable(t) {
		if foundation.Contains(t, foundation.BoolKind) {
			return nil, false, diagHelp(g.Span, "store uint32 flags and derive boolean masks after loading them", "buffer type %s is not host-shareable", t)
		}
		return nil, false, diag(g.Span, "buffer type %s is not host-shareable", t)
	}
	if _, err := foundation.LayoutOf(t); err != nil {
		return nil, false, diag(g.Span, "buffer layout: %v", err)
	}
	return t, true, nil
}

func (c *analyzer) workgroup(attrs []parser.Attribute, dimensions int) (ir.WorkgroupConstraint, error) {
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
			value, err := c.evaluateConstant(e, foundation.Uint32Type, newEnv())
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
