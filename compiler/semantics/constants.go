package semantics

import (
	"errors"
	"fmt"
	"strings"
	"tach/foundation"
	"tach/ir"
	"tach/parser"
)

type runtimeConstantDependency struct{ error }

func (e *runtimeConstantDependency) Unwrap() error { return e.error }

func (c *analyzer) tryConstant(expression parser.Expr, expected *foundation.Type, environment env) (*foundation.ConstantValue, bool, error) {
	value, err := c.evaluateConstant(expression, expected, environment)
	var runtime *runtimeConstantDependency
	if errors.As(err, &runtime) {
		return nil, false, nil
	}
	return value, err == nil, err
}

func (c *analyzer) collectConstants() error {
	for _, declaration := range c.syntax.Decls {
		item, ok := declaration.(*parser.ConstDecl)
		if !ok {
			continue
		}
		if ReservedName(item.Name) {
			return diag(item.Span, "constant name %q is reserved by Tach", item.Name)
		}
		if c.types[item.Name] != nil || c.consts[item.Name] != nil {
			return diag(item.Span, "declaration %q is already defined", item.Name)
		}
		c.consts[item.Name] = &constantDef{decl: item}
	}
	var diagnostics foundation.Diagnostics
	reported := map[string]bool{}
	for _, declaration := range c.syntax.Decls {
		item, ok := declaration.(*parser.ConstDecl)
		if !ok || c.consts[item.Name].state == 3 {
			continue
		}
		if _, err := c.resolveConstant(item.Name, item.Span); err != nil {
			key := err.Error()
			if !reported[key] {
				diagnostics, reported[key] = appendError(diagnostics, err), true
			}
		}
	}
	if len(diagnostics) > 0 {
		return diagnostics
	}
	return nil
}

func (c *analyzer) resolveConstant(name string, reference foundation.Span) (*foundation.ConstantValue, error) {
	definition := c.consts[name]
	if definition == nil || !c.visible(name, reference.File) {
		return nil, diag(reference, "unknown constant %q", name)
	}
	if definition.state == 2 {
		return definition.value, nil
	}
	if definition.state == 3 {
		return nil, definition.err
	}
	if definition.state == 1 {
		start := 0
		for index, item := range c.constantStack {
			if item == name {
				start = index
				break
			}
		}
		chain := append(append([]string(nil), c.constantStack[start:]...), name)
		diagnostic := &foundation.Diagnostic{Span: reference, Message: fmt.Sprintf("compile-time constant cycle: %s", strings.Join(chain, " -> "))}
		for _, item := range chain[:len(chain)-1] {
			constant := c.consts[item]
			diagnostic.Related = append(diagnostic.Related, foundation.RelatedDiagnostic{Span: constant.decl.Span, Message: fmt.Sprintf("constant %q participates in this cycle", item)})
			constant.state, constant.err = 3, diagnostic
		}
		return nil, diagnostic
	}
	definition.state = 1
	c.constantStack = append(c.constantStack, name)
	defer func() { c.constantStack = c.constantStack[:len(c.constantStack)-1] }()
	value, err := c.evaluateConstantBinding(definition.decl.Type, definition.decl.Value, newEnv())
	if err != nil {
		if definition.state != 3 {
			definition.state, definition.err = 3, err
		}
		return nil, err
	}
	definition.value = value
	definition.state = 2
	return definition.value, nil
}

func (c *analyzer) evaluateConstantBinding(typeExpression parser.TypeExpr, expression parser.Expr, environment env) (*foundation.ConstantValue, error) {
	var expected *foundation.Type
	var err error
	if typeExpression != nil {
		expected, err = c.resolveTypeIn(typeExpression, &environment)
		if err != nil {
			return nil, err
		}
		if !foundation.IsConstantType(expected) {
			return nil, diag(typeExpression.GetSpan(), "compile-time constant type must be a scalar or vector, got %s", expected)
		}
	}
	return c.evaluateConstant(expression, expected, environment)
}

func (c *analyzer) evaluateConstant(expression parser.Expr, expected *foundation.Type, environment env) (*foundation.ConstantValue, error) {
	block := &ir.Block{}
	builder := &fnBuilder{
		fn:       &ir.Function{Kind: ir.Helper, Return: foundation.VoidType, Body: block},
		ids:      &idAllocator{},
		block:    block,
		comptime: true,
	}
	result, resultType, err := c.lowerExpr(builder, environment, expression, expected)
	if err != nil {
		return nil, err
	}
	if !foundation.IsConstantType(resultType) {
		return nil, diag(expression.GetSpan(), "compile-time expression produces %s; constants must be scalar or vector values", resultType)
	}
	if expected != nil && !foundation.Equal(resultType, expected) {
		return nil, diag(expression.GetSpan(), "compile-time expression is %s, want %s", resultType, expected)
	}
	values := map[ir.ValueID]*foundation.ConstantValue{}
	if _, err := evaluateConstantBlock(block, values); err != nil {
		return nil, diag(expression.GetSpan(), "invalid compile-time expression: %v", err)
	}
	value := values[result]
	if value == nil {
		return nil, diag(expression.GetSpan(), "compile-time expression has no value")
	}
	return value, nil
}
