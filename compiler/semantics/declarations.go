package semantics

import (
	"fmt"
	"strconv"
	"strings"
	"tach/foundation"
	"tach/parser"
)

func (c *analyzer) collectTypes() error {
	for _, d := range c.syntax.Decls {
		td, ok := d.(*parser.TypeDecl)
		if !ok {
			continue
		}
		if _, exists := c.types[td.Name]; exists {
			return diag(td.Span, "type %q is already defined", td.Name)
		}
		t := &foundation.Type{Kind: foundation.StructKind, Name: td.Name}
		c.types[td.Name] = t
		c.kernel.Structs = append(c.kernel.Structs, t)
	}
	return nil
}
func (c *analyzer) resolveTypeFields() error {
	var diagnostics foundation.Diagnostics
	var declarations []*parser.TypeDecl
	for _, d := range c.syntax.Decls {
		td, ok := d.(*parser.TypeDecl)
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
			if ft.Kind == foundation.VoidKind {
				return diag(f.Span, "field %s.%s cannot be void", td.Name, f.Name)
			}
			if ft.Kind == foundation.RuntimeArrayKind && i != len(td.Fields)-1 {
				return diag(f.Span, "runtime array must be the final field of a struct")
			}
			t.Fields = append(t.Fields, foundation.TypeField{Name: f.Name, Type: ft})
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

func (c *analyzer) checkRuntimeArrayPlacement() error {
	for _, t := range c.kernel.Structs {
		for i, f := range t.Fields {
			if f.Type.Kind == foundation.RuntimeArrayKind {
				if i != len(t.Fields)-1 {
					return diag(c.fieldSpan(t.Name, f.Name), "runtime array in %s must be the final member", t.Name)
				}
				if foundation.ContainsRuntimeArray(f.Type.Elem) {
					return diag(c.fieldSpan(t.Name, f.Name), "runtime array element %s in %s must have a fixed footprint", f.Type.Elem, t.Name)
				}
				continue
			}
			if foundation.ContainsRuntimeArray(f.Type) {
				return diag(c.fieldSpan(t.Name, f.Name), "%s.%s nests a runtime-sized structure; Tach permits one trailing runtime array", t.Name, f.Name)
			}
		}
	}
	return nil
}
func (c *analyzer) checkTypeCycles() error {
	state := map[string]uint8{}
	var visit func(*foundation.Type) error
	visit = func(t *foundation.Type) error {
		if t.Kind == foundation.RuntimeArrayKind {
			return visit(t.Elem)
		}
		if t.Kind != foundation.StructKind {
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
	for _, t := range c.kernel.Structs {
		if err := visit(t); err != nil {
			return err
		}
	}
	return nil
}

func (c *analyzer) declarationSpan(name string) foundation.Span {
	for _, declaration := range c.syntax.Decls {
		if item, ok := declaration.(*parser.TypeDecl); ok && item.Name == name {
			return item.Span
		}
	}
	return foundation.Span{}
}

func (c *analyzer) fieldSpan(typeName, fieldName string) foundation.Span {
	for _, declaration := range c.syntax.Decls {
		if item, ok := declaration.(*parser.TypeDecl); ok && item.Name == typeName {
			for _, field := range item.Fields {
				if field.Name == fieldName {
					return field.Span
				}
			}
			return item.Span
		}
	}
	return foundation.Span{}
}

func (c *analyzer) collectFunctions() error {
	var diagnostics foundation.Diagnostics
	var declarations []*parser.FunctionDecl
	for _, d := range c.syntax.Decls {
		if function, ok := d.(*parser.FunctionDecl); ok {
			declarations = append(declarations, function)
		}
	}
	signatures := make([]*funcSig, len(declarations))
	errors := parallel(c.workers, len(declarations), func(index int) error {
		x := declarations[index]
		if ReservedName(x.Name) {
			return diag(x.Span, "function name %q is reserved by Tach", x.Name)
		}
		sig := &funcSig{name: x.Name, decl: x, ret: foundation.VoidType, indexed: len(x.Indices) > 0, exported: x.Exported}
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
			if x.Exported && !buffer && !foundation.IsHostParameter(t) {
				return diagHelp(p.Type.GetSpan(), "pass uint32 flags and derive boolean masks after loading them", "public parameter type %s has no host representation", t)
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
			if r.Kind != foundation.VoidKind && !foundation.IsConstructible(r) {
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
func (c *analyzer) resolveType(te parser.TypeExpr) (*foundation.Type, error) {
	return c.resolveTypeIn(te, nil)
}

func (c *analyzer) resolveTypeIn(te parser.TypeExpr, environment *env) (*foundation.Type, error) {
	switch t := te.(type) {
	case *parser.NamedType:
		x := c.types[t.Name]
		if x != nil && !c.visible(t.Name, t.Span.File) {
			x = nil
		}
		if x == nil {
			return nil, diag(t.Span, "unknown type %q", t.Name)
		}
		return x, nil
	case *parser.RuntimeArrayType:
		e, err := c.resolveTypeIn(t.Elem, environment)
		if err != nil {
			return nil, err
		}
		if e.Kind == foundation.VoidKind || e.Kind == foundation.RuntimeArrayKind {
			return nil, diag(t.Span, "invalid runtime array element type %s", e)
		}
		return foundation.RuntimeArrayOf(e), nil
	case *parser.FixedArrayType:
		e, err := c.resolveTypeIn(t.Elem, environment)
		if err != nil {
			return nil, err
		}
		if !foundation.IsWorkgroupStorable(e) {
			if foundation.Contains(e, foundation.BoolKind) {
				return nil, diagHelp(t.Span, "store uint32 flags and derive boolean masks after loading them", "invalid fixed array element type %s", e)
			}
			return nil, diag(t.Span, "invalid fixed array element type %s", e)
		}
		scope := newEnv()
		if environment != nil {
			scope = *environment
		}
		value, err := c.evaluateConstant(t.Count, foundation.Uint32Type, scope)
		if err != nil {
			return nil, err
		}
		if len(value.Bits) != 1 || value.Bits[0] == 0 {
			return nil, diag(t.Span, "fixed array length must be a positive uint32 constant")
		}
		return foundation.FixedArrayOf(e, value.Bits[0]), nil
	case *parser.VectorType:
		e, err := c.resolveTypeIn(t.Elem, environment)
		if err != nil {
			return nil, err
		}
		if !foundation.IsScalar(e) {
			return nil, diag(t.Span, "vec element type must be scalar, got %s", e)
		}
		lanes, err := strconv.Atoi(t.Lanes)
		if err != nil || lanes < 2 || lanes > 4 {
			return nil, diag(t.Span, "vec lane count must be 2, 3, or 4")
		}
		return foundation.VectorOf(e, lanes), nil
	case *parser.GenericType:
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
			if e.Kind != foundation.Int32Kind && e.Kind != foundation.Uint32Kind {
				return nil, diag(t.Span, "atomic element type must be int32 or uint32, got %s", e)
			}
			return foundation.AtomicOf(e), nil
		}
		return nil, diag(t.Span, "unknown generic type %q", t.Name)
	default:
		return nil, fmt.Errorf("unknown type expression %T", te)
	}
}

func (c *analyzer) visible(name, file string) bool {
	owner := c.owners[name]
	return owner == "" || c.imports[strings.TrimSuffix(file, ".tach")][owner]
}
