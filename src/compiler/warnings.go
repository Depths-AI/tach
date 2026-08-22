package compiler

import (
	"fmt"
	"maps"
	"strconv"
	"strings"

	"tach/src/ast"
	"tach/src/foundation"
	"tach/src/ir"
)

func warnings(project *project, module *ir.Module) foundation.Diagnostics {
	owners := map[string]string{}
	functions := map[string]*ast.FunctionDecl{}
	constants := map[string]*ast.ConstDecl{}
	for i := range project.Kernels {
		for _, declaration := range project.Kernels[i].AST.Decls {
			switch item := declaration.(type) {
			case *ast.TypeDecl:
				owners[item.Name] = project.Kernels[i].Identity
			case *ast.ConstDecl:
				owners[item.Name], constants[item.Name] = project.Kernels[i].Identity, item
			case *ast.FunctionDecl:
				owners[item.Name], functions[item.Name] = project.Kernels[i].Identity, item
			}
		}
	}

	refs := map[string]map[string]bool{}
	var typeRoots []map[string]bool
	var diagnostics foundation.Diagnostics
	for i := range project.Kernels {
		kernel := &project.Kernels[i]
		used := map[string]bool{}
		visibleConstants := map[string]bool{}
		visibleOwners := map[string]bool{kernel.Identity: true}
		for _, item := range kernel.AST.Imports {
			visibleOwners[item.Target] = true
		}
		for name := range constants {
			if visibleOwners[owners[name]] {
				visibleConstants[name] = true
			}
		}
		for _, declaration := range kernel.AST.Decls {
			analysis := &analysis{refs: map[string]bool{}, reads: map[string]bool{}, globals: visibleConstants, locals: map[string]bool{}}
			switch item := declaration.(type) {
			case *ast.TypeDecl:
				for _, field := range item.Fields {
					analysis.typeExpression(field.Type)
				}
				typeRoots = append(typeRoots, analysis.refs)
			case *ast.ConstDecl:
				if item.Type != nil {
					analysis.typeExpression(item.Type)
				}
				analysis.expression(item.Value)
				refs[item.Name] = analysis.refs
			case *ast.FunctionDecl:
				analysis.function(item)
				refs[item.Name] = analysis.refs
				diagnostics = append(diagnostics, analysis.diagnostics...)
			}
			names := analysis.refs
			for name := range names {
				used[owners[name]] = true
			}
		}
		for _, item := range kernel.AST.Imports {
			if !used[item.Target] {
				diagnostics = append(diagnostics, warning("unused-import", item.Span, fmt.Sprintf("import %q is never used", item.Target), "remove the import"))
			}
		}
	}

	reachable := map[string]bool{}
	var visit func(string)
	visit = func(name string) {
		if reachable[name] {
			return
		}
		reachable[name] = true
		for dependency := range refs[name] {
			if functions[dependency] != nil || constants[dependency] != nil {
				visit(dependency)
			}
		}
	}
	for _, roots := range typeRoots {
		for name := range roots {
			visit(name)
		}
	}
	for name, function := range functions {
		if function.Exported {
			visit(name)
		}
	}
	for name, constant := range constants {
		if !reachable[name] {
			diagnostics = append(diagnostics, warning("unused-constant", constant.Span, fmt.Sprintf("compile-time constant %q is never used", name), "remove it or use it from a type, constant, or exported function"))
		}
	}
	for name, function := range functions {
		if !function.Exported && !reachable[name] {
			diagnostics = append(diagnostics, warning("unreachable-function", function.Span, fmt.Sprintf("private function %q is unreachable from every export", name), "remove it or call it from an exported program's dependency graph"))
		}
	}

	for _, function := range module.Kernel.Functions {
		if function.Kind != ir.Stage || !reachable[function.Name] {
			continue
		}
		summary := ir.AnalyzeAccess(function)
		if !summary.Effects.Memory {
			diagnostics = append(diagnostics, warning("no-effect-kernel", function.Span, fmt.Sprintf("kernel %q has no externally observable memory effect", function.Name), "write a buffer or remove the dispatch"))
		}
		if len(function.Indices) == 1 {
			unconditionalWrites := map[int]bool{}
			strided := map[string][]ir.MemoryAccess{}
			for _, instruction := range function.Body.Instrs {
				if store, ok := instruction.(*ir.Store); ok {
					unconditionalWrites[store.Span.Start.Offset] = true
				}
			}
			for bufferIndex, buffer := range summary.Buffers {
				for _, access := range buffer.Accesses {
					if len(access.Indices) != 1 || !access.Indices[0].Exact {
						continue
					}
					index := access.Indices[0]
					if access.Kind == ir.MemoryWrite && index.Coefficient == [3]int64{} && unconditionalWrites[access.Span.Start.Offset] {
						diagnostics = append(diagnostics, warning("constant-write-index", access.Span, "an unconditional non-atomic write address does not depend on the invocation index", "for launches above one, guard one invocation, use an invocation-dependent index, or use an atomic operation"))
					} else if stride := index.Coefficient[0]; stride > 1 || stride < -1 {
						key := fmt.Sprintf("%d:%d", bufferIndex, stride)
						strided[key] = append(strided[key], access)
					}
				}
			}
			for key, accesses := range strided {
				_, stride, _ := strings.Cut(key, ":")
				diagnostic := warning("strided-access", accesses[0].Span, fmt.Sprintf("adjacent invocations access one buffer with stride %s", stride), "prefer adjacent elements when the algorithm permits contiguous GPU memory access")
				for _, access := range accesses[1:] {
					diagnostic.Related = append(diagnostic.Related, foundation.RelatedDiagnostic{Span: access.Span, Message: "same strided access pattern"})
				}
				diagnostics = append(diagnostics, diagnostic)
			}
		}
	}
	return enrichDiagnostics(diagnostics, project.sources, "warning")
}

func warning(code string, span foundation.Span, message, help string) foundation.Diagnostic {
	return foundation.Diagnostic{Severity: "warning", Kind: code, Span: span, Message: message, Help: help}
}

type binding struct {
	kind, name string
	span       foundation.Span
}

type analysis struct {
	refs, reads map[string]bool
	globals     map[string]bool
	locals      map[string]bool
	bindings    []binding
	diagnostics foundation.Diagnostics
}

func (a *analysis) function(function *ast.FunctionDecl) {
	for _, attribute := range function.Attrs {
		if attribute.Name == "workgroup" {
			for _, argument := range attribute.Args {
				a.expression(argument)
			}
		}
	}
	for _, index := range function.Indices {
		a.bindings = append(a.bindings, binding{"index", index.Name, index.Span})
		a.locals[index.Name] = true
	}
	for _, parameter := range function.Params {
		a.bindings = append(a.bindings, binding{"parameter", parameter.Name, parameter.Span})
		a.locals[parameter.Name] = true
		a.typeExpression(parameter.Type)
	}
	if function.Return != nil {
		a.typeExpression(function.Return)
	}
	a.block(function.Body)
	for _, item := range a.bindings {
		if !a.reads[item.name] {
			a.diagnostics = append(a.diagnostics, warning("unused-binding", item.span, fmt.Sprintf("%s %q is never used", item.kind, item.name), "remove the binding or use its value"))
		}
	}
}

func (a *analysis) typeExpression(expression ast.TypeExpr) {
	switch item := expression.(type) {
	case *ast.NamedType:
		a.refs[item.Name] = true
	case *ast.RuntimeArrayType:
		a.typeExpression(item.Elem)
	case *ast.FixedArrayType:
		a.typeExpression(item.Elem)
		a.expression(item.Count)
	case *ast.VectorType:
		a.typeExpression(item.Elem)
	case *ast.GenericType:
		for _, argument := range item.Args {
			a.typeExpression(argument)
		}
	}
}

func (a *analysis) block(block *ast.BlockStmt) {
	outer := maps.Clone(a.locals)
	defer func() { a.locals = outer }()
	for _, statement := range block.Stmts {
		switch item := statement.(type) {
		case *ast.VarStmt:
			a.bindings = append(a.bindings, binding{"local", item.Name, item.Span})
			if item.Type != nil {
				a.typeExpression(item.Type)
			}
			a.expression(item.Value)
			a.locals[item.Name] = true
		case *ast.ConstStmt:
			a.bindings = append(a.bindings, binding{"constant", item.Name, item.Span})
			if item.Type != nil {
				a.typeExpression(item.Type)
			}
			a.expression(item.Value)
			a.locals[item.Name] = true
		case *ast.WorkgroupStmt:
			a.bindings = append(a.bindings, binding{"shared variable", item.Name, item.Span})
			a.typeExpression(item.Type)
			a.locals[item.Name] = true
		case *ast.AssignStmt:
			a.expression(item.Target)
			a.expression(item.Value)
		case *ast.IncStmt:
			a.expression(item.Target)
		case *ast.ExprStmt:
			a.expression(item.Expr)
			if !sideEffecting(item.Expr) {
				a.diagnostics = append(a.diagnostics, warning("discarded-value", item.Span, "pure expression result is discarded", "remove the statement or bind and use its result"))
			}
		case *ast.IfStmt:
			a.expression(item.Cond)
			a.warnBool(item.Cond, "if")
			a.block(item.Then)
			if item.Else != nil {
				a.block(item.Else)
			}
		case *ast.WhileStmt:
			a.expression(item.Cond)
			if value, ok := item.Cond.(*ast.BoolExpr); ok && !value.Value {
				a.diagnostics = append(a.diagnostics, warning("constant-condition", value.Span, "while condition is always false", "remove the loop or use a non-constant condition"))
			}
			a.block(item.Body)
		case *ast.ForStmt:
			a.bindings = append(a.bindings, binding{"local", item.Init.Name, item.Init.Span})
			a.expression(item.Init.Value)
			a.locals[item.Init.Name] = true
			a.expression(item.Cond)
			if value, ok := item.Cond.(*ast.BoolExpr); ok && !value.Value {
				a.diagnostics = append(a.diagnostics, warning("constant-condition", value.Span, "for condition is always false", "remove the loop or use a non-constant condition"))
			}
			a.block(item.Body)
			a.block(&ast.BlockStmt{Stmts: []ast.Stmt{item.Post}})
			delete(a.locals, item.Init.Name)
		case *ast.ReturnStmt:
			if item.Value != nil {
				a.expression(item.Value)
			}
		case *ast.RunStmt:
			a.refs[item.Stage] = true
			for _, argument := range item.Args {
				a.expression(argument)
			}
			for _, axis := range item.Domain.Axes {
				a.expression(axis)
				if number, ok := axis.(*ast.NumberExpr); ok && zero(number.Raw) {
					a.diagnostics = append(a.diagnostics, warning("zero-dispatch", number.Span, "dispatch axis is always zero, so the kernel cannot run", "use a positive launch extent or remove the dispatch"))
				}
			}
		}
	}
}

func (a *analysis) expression(expression ast.Expr) {
	switch item := expression.(type) {
	case *ast.IdentExpr:
		a.reads[item.Name] = true
		if !a.locals[item.Name] && a.globals[item.Name] {
			a.refs[item.Name] = true
		}
	case *ast.UnaryExpr:
		a.expression(item.X)
	case *ast.BinaryExpr:
		a.expression(item.Left)
		a.expression(item.Right)
	case *ast.ConditionalExpr:
		a.expression(item.Cond)
		a.warnBool(item.Cond, "conditional")
		a.expression(item.Then)
		a.expression(item.Else)
	case *ast.CallExpr:
		if callee, ok := item.Callee.(*ast.IdentExpr); ok {
			a.refs[callee.Name] = true
		} else {
			a.expression(item.Callee)
		}
		for _, argument := range item.Args {
			a.expression(argument)
		}
	case *ast.MemberExpr:
		a.expression(item.Base)
	case *ast.IndexExpr:
		a.expression(item.Base)
		a.expression(item.Index)
	case *ast.StructLiteralExpr:
		for _, field := range item.Fields {
			a.expression(field.Value)
		}
	case *ast.TransientExpr:
		a.typeExpression(item.Elem)
		a.expression(item.Count)
	}
}

func (a *analysis) warnBool(expression ast.Expr, context string) {
	if value, ok := expression.(*ast.BoolExpr); ok {
		a.diagnostics = append(a.diagnostics, warning("constant-condition", value.Span, fmt.Sprintf("%s condition is always %t", context, value.Value), "remove the branch or use a non-constant condition"))
	}
}

func sideEffecting(expression ast.Expr) bool {
	call, ok := expression.(*ast.CallExpr)
	if !ok {
		return false
	}
	callee, ok := call.Callee.(*ast.IdentExpr)
	return ok && (callee.Name == "workgroupBarrier" || callee.Name == "bufferBarrier" || strings.HasPrefix(callee.Name, "atomic"))
}

func zero(raw string) bool {
	value, err := strconv.ParseUint(strings.ReplaceAll(raw, "_", ""), 0, 32)
	return err == nil && value == 0
}
