package sema

import (
	"strings"

	"tach/src/ast"
	"tach/src/foundation"
	"tach/src/ir"
)

func checkDocumentation(module *ast.Module) (ir.Documentation, error) {
	docs := ir.Documentation{Types: map[string]ir.TypeDocumentation{}, Functions: map[string]ir.FunctionDocumentation{}}
	var diagnostics foundation.Diagnostics
	moduleDocs, rest, err := takeDocs(module.Attrs)
	if err != nil {
		diagnostics = appendError(diagnostics, err)
	}
	if len(rest) > 0 {
		diagnostics = appendError(diagnostics, diag(rest[0].Span, "only @docs is valid at kernel scope"))
	}
	if moduleDocs != nil {
		if err := readModuleDocs(*moduleDocs, &docs); err != nil {
			diagnostics = appendError(diagnostics, err)
		}
	}
	for _, declaration := range module.Decls {
		switch d := declaration.(type) {
		case *ast.TypeDecl:
			a, remaining, err := takeDocs(d.Attrs)
			if err != nil {
				diagnostics = appendError(diagnostics, err)
			}
			if len(remaining) > 0 {
				diagnostics = appendError(diagnostics, diag(remaining[0].Span, "only @docs is valid on type %s", d.Name))
			}
			d.Attrs = nil
			if a != nil {
				doc, err := readTypeDocs(*a, d)
				if err != nil {
					diagnostics = appendError(diagnostics, err)
					continue
				}
				docs.Types[d.Name] = doc
			}
		case *ast.FunctionDecl:
			a, remaining, err := takeDocs(d.Attrs)
			if err != nil {
				diagnostics = appendError(diagnostics, err)
			}
			d.Attrs = remaining
			if a != nil {
				doc, err := readFunctionDocs(*a, d)
				if err != nil {
					diagnostics = appendError(diagnostics, err)
					continue
				}
				docs.Functions[d.Name] = doc
			}
		}
	}
	if len(diagnostics) > 0 {
		return docs, diagnostics
	}
	return docs, nil
}

func takeDocs(attrs []ast.Attribute) (*ast.Attribute, []ast.Attribute, error) {
	var docs *ast.Attribute
	var duplicate error
	rest := make([]ast.Attribute, 0, len(attrs))
	for i := range attrs {
		if attrs[i].Name != "docs" {
			rest = append(rest, attrs[i])
			continue
		}
		if docs != nil {
			if duplicate == nil {
				duplicate = diag(attrs[i].Span, "duplicate @docs")
			}
			continue
		}
		docs = &attrs[i]
	}
	return docs, rest, duplicate
}

func readModuleDocs(attribute ast.Attribute, out *ir.Documentation) error {
	seen := map[string]bool{}
	for _, expression := range attribute.Args {
		name, args, err := docClause(expression)
		if err != nil {
			return err
		}
		if name != "title" && name != "summary" {
			return diag(expression.GetSpan(), "@docs clause %s is invalid for a module", name)
		}
		if seen[name] {
			return diag(expression.GetSpan(), "duplicate %s in @docs", name)
		}
		seen[name] = true
		value, err := docText(name, args, expression.GetSpan())
		if err != nil {
			return err
		}
		if name == "title" {
			out.Title = value
		} else {
			out.Summary = value
		}
	}
	return requireSummary(attribute, seen)
}

func readTypeDocs(attribute ast.Attribute, declaration *ast.TypeDecl) (ir.TypeDocumentation, error) {
	out := ir.TypeDocumentation{Fields: map[string]string{}}
	fields := map[string]bool{}
	for _, field := range declaration.Fields {
		fields[field.Name] = true
	}
	seen := map[string]bool{}
	for _, expression := range attribute.Args {
		name, args, err := docClause(expression)
		if err != nil {
			return out, err
		}
		switch name {
		case "summary":
			if seen[name] {
				return out, diag(expression.GetSpan(), "duplicate summary in @docs")
			}
			seen[name] = true
			out.Summary, err = docText(name, args, expression.GetSpan())
		case "field":
			field, text, e := namedDoc(name, args, expression.GetSpan())
			if e != nil {
				return out, e
			}
			if !fields[field] {
				return out, diag(expression.GetSpan(), "@docs references unknown field %s.%s", declaration.Name, field)
			}
			if _, duplicate := out.Fields[field]; duplicate {
				return out, diag(expression.GetSpan(), "duplicate documentation for field %s.%s", declaration.Name, field)
			}
			out.Fields[field] = text
		default:
			return out, diag(expression.GetSpan(), "@docs clause %s is invalid for a type", name)
		}
		if err != nil {
			return out, err
		}
	}
	return out, requireSummary(attribute, seen)
}

func readFunctionDocs(attribute ast.Attribute, declaration *ast.FunctionDecl) (ir.FunctionDocumentation, error) {
	out := ir.FunctionDocumentation{Parameters: map[string]string{}, Coordinates: map[string]string{}}
	parameters, coordinates := map[string]bool{}, map[string]bool{}
	for _, parameter := range declaration.Params {
		parameters[parameter.Name] = true
	}
	for _, coordinate := range declaration.Indices {
		coordinates[coordinate.Name] = true
	}
	seen := map[string]bool{}
	for _, expression := range attribute.Args {
		name, args, err := docClause(expression)
		if err != nil {
			return out, err
		}
		switch name {
		case "summary":
			if seen[name] {
				return out, diag(expression.GetSpan(), "duplicate summary in @docs")
			}
			seen[name] = true
			out.Summary, err = docText(name, args, expression.GetSpan())
		case "param":
			param, text, e := namedDoc(name, args, expression.GetSpan())
			if e != nil {
				return out, e
			}
			if !parameters[param] {
				return out, diag(expression.GetSpan(), "@docs references unknown parameter %s.%s", declaration.Name, param)
			}
			if _, duplicate := out.Parameters[param]; duplicate {
				return out, diag(expression.GetSpan(), "duplicate documentation for parameter %s.%s", declaration.Name, param)
			}
			out.Parameters[param] = text
		case "coordinate":
			coordinate, text, e := namedDoc(name, args, expression.GetSpan())
			if e != nil {
				return out, e
			}
			if !coordinates[coordinate] {
				return out, diag(expression.GetSpan(), "@docs references unknown coordinate %s.%s", declaration.Name, coordinate)
			}
			if _, duplicate := out.Coordinates[coordinate]; duplicate {
				return out, diag(expression.GetSpan(), "duplicate documentation for coordinate %s.%s", declaration.Name, coordinate)
			}
			out.Coordinates[coordinate] = text
		case "returns":
			if declaration.Return == nil || isVoidType(declaration.Return) {
				return out, diag(expression.GetSpan(), "void function %s cannot document a return value", declaration.Name)
			}
			if seen[name] {
				return out, diag(expression.GetSpan(), "duplicate returns in @docs")
			}
			seen[name] = true
			out.Returns, err = docText(name, args, expression.GetSpan())
		default:
			return out, diag(expression.GetSpan(), "@docs clause %s is invalid for a function", name)
		}
		if err != nil {
			return out, err
		}
	}
	return out, requireSummary(attribute, seen)
}

func docClause(expression ast.Expr) (string, []ast.Expr, error) {
	call, ok := expression.(*ast.CallExpr)
	if !ok {
		return "", nil, diag(expression.GetSpan(), "@docs entries must be clauses such as summary(\"...\")")
	}
	name, ok := call.Callee.(*ast.IdentExpr)
	if !ok {
		return "", nil, diag(call.Callee.GetSpan(), "@docs clause must have a simple name")
	}
	return name.Name, call.Args, nil
}

func docText(clause string, args []ast.Expr, span foundation.Span) (string, error) {
	if len(args) != 1 {
		return "", diag(argsSpan(args, span), "%s expects one string", clause)
	}
	text, ok := args[0].(*ast.StringExpr)
	if !ok || strings.TrimSpace(text.Value) == "" {
		return "", diag(args[0].GetSpan(), "%s expects a non-empty string", clause)
	}
	return strings.TrimSpace(text.Value), nil
}

func namedDoc(clause string, args []ast.Expr, span foundation.Span) (string, string, error) {
	if len(args) != 2 {
		return "", "", diag(argsSpan(args, span), "%s expects a name and a string", clause)
	}
	name, ok := args[0].(*ast.IdentExpr)
	if !ok {
		return "", "", diag(args[0].GetSpan(), "%s expects an unquoted declaration name", clause)
	}
	text, err := docText(clause, args[1:], span)
	return name.Name, text, err
}

func requireSummary(attribute ast.Attribute, seen map[string]bool) error {
	if !seen["summary"] {
		return diag(attribute.Span, "@docs requires summary(\"...\")")
	}
	return nil
}

func argsSpan(args []ast.Expr, fallback foundation.Span) foundation.Span {
	if len(args) > 0 {
		return args[0].GetSpan()
	}
	return fallback
}

func isVoidType(expression ast.TypeExpr) bool {
	named, ok := expression.(*ast.NamedType)
	return ok && named.Name == "void"
}
