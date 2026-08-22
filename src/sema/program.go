package sema

import (
	"tach/src/ast"
	"tach/src/flow"
	"tach/src/foundation"
	"tach/src/ir"
)

type programSymbol struct {
	resource  flow.ResourceID
	shape     flow.ShapeID
	type_     *foundation.Type
	constant  *foundation.ConstantValue
	parameter int
}

func (c *Checker) lowerPrograms() error {
	var diagnostics foundation.Diagnostics
	var declarations []*ast.FunctionDecl
	for _, declaration := range c.ast.Decls {
		function, ok := declaration.(*ast.FunctionDecl)
		if ok && function.Exported {
			declarations = append(declarations, function)
		}
	}
	programs := make([]*flow.Program, len(declarations))
	errors := parallel(c.workers, len(declarations), func(index int) error {
		function := declarations[index]
		var err error
		if len(function.Indices) > 0 {
			programs[index], err = c.lowerIndexedProgram(function)
		} else {
			programs[index], err = c.lowerProgram(function)
		}
		return err
	})
	for index, err := range errors {
		if err != nil {
			diagnostics = appendError(diagnostics, err)
			continue
		}
		c.flow.Programs = append(c.flow.Programs, programs[index])
	}
	if len(diagnostics) > 0 {
		return diagnostics
	}
	return nil
}

func (c *Checker) lowerIndexedProgram(declaration *ast.FunctionDecl) (*flow.Program, error) {
	stage := c.mod.Function(declaration.Name)
	program := &flow.Program{Name: declaration.Name, Span: declaration.Span, Indexed: true, Rank: len(stage.Indices)}
	symbols, current, err := c.addProgramParameters(program, declaration)
	if err != nil {
		return nil, err
	}
	if len(current) == 0 {
		return nil, diag(declaration.Span, "public program %s requires at least one buffer parameter", declaration.Name)
	}
	dispatch := flow.Dispatch{Stage: stage.Name, Span: declaration.Span}
	for axis := range stage.Indices {
		dispatch.Domain = append(dispatch.Domain, program.AddShape(flow.Shape{Op: flow.ShapeLaunchAxis, Axis: uint8(axis), Span: declaration.Span}))
	}
	if err := c.bindStageArguments(program, &dispatch, stage, declaration.Params, symbols, current); err != nil {
		return nil, err
	}
	id := program.AddDispatch(dispatch)
	if err := c.finishDispatch(program, &program.Dispatches[len(program.Dispatches)-1], id, current); err != nil {
		return nil, err
	}
	finishResources(program, current)
	return program, nil
}

func (c *Checker) lowerProgram(declaration *ast.FunctionDecl) (*flow.Program, error) {
	if len(declaration.Attrs) > 0 {
		return nil, diag(declaration.Span, "attributes are invalid on public program %s", declaration.Name)
	}
	program := &flow.Program{Name: declaration.Name, Span: declaration.Span}
	symbols, current, err := c.addProgramParameters(program, declaration)
	if err != nil {
		return nil, err
	}
	if len(current) == 0 && c.funcs[declaration.Name].view == 0 {
		return nil, diag(declaration.Span, "public program %s requires at least one buffer parameter", declaration.Name)
	}
	for index, statement := range declaration.Body.Stmts {
		switch x := statement.(type) {
		case *ast.ConstStmt:
			if _, exists := symbols[x.Name]; exists {
				return nil, diag(x.Span, "%q is already defined", x.Name)
			}
			value, err := c.evaluateConstantBinding(x.Type, x.Value, programEnvironment(symbols))
			if err != nil {
				return nil, err
			}
			symbols[x.Name] = programSymbol{type_: value.Type, constant: value, parameter: -1}
		case *ast.VarStmt:
			if _, exists := symbols[x.Name]; exists {
				return nil, diag(x.Span, "%q is already defined", x.Name)
			}
			if transient, ok := x.Value.(*ast.TransientExpr); ok {
				elem, err := c.resolveType(transient.Elem)
				if err != nil {
					return nil, err
				}
				if !foundation.IsTransientElement(elem) {
					return nil, diag(transient.Span, "transient element %s must have a fixed host-shareable non-atomic footprint", elem)
				}
				transientType := foundation.RuntimeArrayOf(elem)
				if x.Type != nil {
					declared, err := c.resolveType(x.Type)
					if err != nil {
						return nil, err
					}
					if !foundation.Equal(declared, transientType) {
						return nil, diag(x.Type.GetSpan(), "program transient is declared as %s, but its initializer produces %s", declared, transientType)
					}
				}
				length, err := c.lowerShape(program, transient.Count, symbols)
				if err != nil {
					return nil, err
				}
				resource := program.AddResource(flow.Resource{Name: x.Name, Kind: flow.Transient, Type: transientType, Length: length, Parameter: -1, Span: x.Span})
				initial := program.AddVersion(flow.Version{Resource: resource, Defined: false})
				program.Resource(resource).Initial = initial
				symbols[x.Name] = programSymbol{resource: resource, type_: transientType, parameter: -1}
				current[resource] = initial
			} else {
				shape, err := c.lowerShape(program, x.Value, symbols)
				if err != nil {
					return nil, err
				}
				if x.Type != nil {
					declared, err := c.resolveType(x.Type)
					if err != nil {
						return nil, err
					}
					if !foundation.Equal(declared, foundation.Uint32Type) {
						return nil, diag(x.Type.GetSpan(), "program shape binding must be uint32, got %s", declared)
					}
				}
				symbols[x.Name] = programSymbol{shape: shape, type_: foundation.Uint32Type, parameter: -1}
			}
		case *ast.RunStmt:
			sig := c.funcs[x.Stage]
			if sig != nil && !c.visible(x.Stage, x.Span.File) {
				sig = nil
			}
			if sig != nil && sig.exported && !sig.indexed {
				return nil, diag(x.Span, "public program %q cannot be a run target", x.Stage)
			}
			stage := c.mod.Function(x.Stage)
			if stage != nil && !c.visible(x.Stage, x.Span.File) {
				stage = nil
			}
			if stage == nil || stage.Kind != ir.Stage {
				return nil, diag(x.Span, "run target %q is not an indexed stage", x.Stage)
			}
			if len(x.Domain.Axes) != len(stage.Indices) {
				return nil, diag(x.Domain.Span, "run domain has rank %d, stage %s has rank %d", len(x.Domain.Axes), stage.Name, len(stage.Indices))
			}
			dispatch := flow.Dispatch{Stage: stage.Name, Span: x.Span}
			for _, axis := range x.Domain.Axes {
				shape, err := c.lowerShape(program, axis, symbols)
				if err != nil {
					return nil, err
				}
				dispatch.Domain = append(dispatch.Domain, shape)
			}
			if len(x.Args) != len(stage.SourceParams) {
				return nil, diag(x.Span, "stage %s expects %d arguments, got %d", stage.Name, len(stage.SourceParams), len(x.Args))
			}
			if err := c.bindRunArguments(program, &dispatch, stage, x.Args, symbols, current); err != nil {
				return nil, err
			}
			id := program.AddDispatch(dispatch)
			if err := c.finishDispatch(program, &program.Dispatches[len(program.Dispatches)-1], id, current); err != nil {
				return nil, err
			}
		case *ast.ReturnStmt:
			if c.funcs[declaration.Name].view == 0 {
				return nil, diag(x.Span, "public program %s does not return a view", declaration.Name)
			}
			if index != len(declaration.Body.Stmts)-1 {
				return nil, diag(x.Span, "view return must be the final program statement")
			}
			view, err := c.lowerView(program, x.Value, symbols, current)
			if err != nil {
				return nil, err
			}
			view.Format = c.funcs[declaration.Name].view
			program.View = view
		default:
			return nil, diag(statement.GetSpan(), "public program bodies permit only const and let declarations, run statements, and a final view return")
		}
	}
	if len(program.Dispatches) == 0 {
		return nil, diag(declaration.Body.Span, "public program %s requires at least one run", declaration.Name)
	}
	if c.funcs[declaration.Name].view != 0 && program.View == nil {
		return nil, diag(declaration.Body.Span, "public program %s must return its view", declaration.Name)
	}
	finishResources(program, current)
	return program, nil
}

func programEnvironment(symbols map[string]programSymbol) env {
	environment := newEnv()
	for name, item := range symbols {
		environment.syms[name] = symbol{ty: item.type_, constant: item.constant, buffer: -1, workgroup: -1}
	}
	return environment
}

func (c *Checker) lowerView(program *flow.Program, expression ast.Expr, symbols map[string]programSymbol, current map[flow.ResourceID]flow.VersionID) (*flow.View, error) {
	call, ok := expression.(*ast.CallExpr)
	name, named := callName(call)
	if !ok || !named || name != "view" || len(call.Args) != 3 {
		return nil, diag(expression.GetSpan(), "view return must be view(pixels, width, height)")
	}
	identifier, ok := call.Args[0].(*ast.IdentExpr)
	symbol, exists := symbols[identifierName(identifier)]
	pixel := foundation.VectorOf(foundation.Float32Type, 4)
	if !ok || !exists || symbol.resource == 0 || symbol.type_.Kind != foundation.RuntimeArrayKind || !foundation.Equal(symbol.type_.Elem, pixel) {
		return nil, diag(call.Args[0].GetSpan(), "view pixels must be a vec<float32, 4> buffer or transient")
	}
	if version := program.Version(current[symbol.resource]); version == nil || !version.Defined {
		return nil, diag(call.Args[0].GetSpan(), "view pixels must be fully defined before presentation")
	}
	width, err := c.lowerShape(program, call.Args[1], symbols)
	if err != nil {
		return nil, err
	}
	height, err := c.lowerShape(program, call.Args[2], symbols)
	if err != nil {
		return nil, err
	}
	return &flow.View{Source: symbol.resource, Input: current[symbol.resource], Width: width, Height: height, Span: expression.GetSpan()}, nil
}

func callName(call *ast.CallExpr) (string, bool) {
	if call == nil {
		return "", false
	}
	identifier, ok := call.Callee.(*ast.IdentExpr)
	return identifierName(identifier), ok
}

func (c *Checker) addProgramParameters(program *flow.Program, declaration *ast.FunctionDecl) (map[string]programSymbol, map[flow.ResourceID]flow.VersionID, error) {
	symbols := map[string]programSymbol{}
	current := map[flow.ResourceID]flow.VersionID{}
	for position, parameter := range declaration.Params {
		sig := c.funcs[declaration.Name].params[position]
		if sig.buffer {
			resource := program.AddResource(flow.Resource{Name: parameter.Name, Kind: flow.External, Type: sig.ty, Parameter: position, Span: parameter.Span})
			initial := program.AddVersion(flow.Version{Resource: resource, Defined: true})
			program.Resource(resource).Initial = initial
			program.Parameters = append(program.Parameters, flow.Parameter{Name: parameter.Name, Kind: flow.BufferParameter, Type: sig.ty, Resource: resource, Span: parameter.Span})
			symbols[parameter.Name] = programSymbol{resource: resource, type_: sig.ty, parameter: position}
			current[resource] = initial
		} else {
			program.Parameters = append(program.Parameters, flow.Parameter{Name: parameter.Name, Kind: flow.ValueParameter, Type: sig.ty, Span: parameter.Span})
			symbols[parameter.Name] = programSymbol{type_: sig.ty, parameter: position}
		}
	}
	return symbols, current, nil
}

func (c *Checker) bindStageArguments(program *flow.Program, dispatch *flow.Dispatch, stage *ir.Function, params []ast.Param, symbols map[string]programSymbol, current map[flow.ResourceID]flow.VersionID) error {
	args := make([]ast.Expr, len(params))
	for i, parameter := range params {
		args[i] = &ast.IdentExpr{Name: parameter.Name, Span: parameter.Span}
	}
	return c.bindRunArguments(program, dispatch, stage, args, symbols, current)
}

func (c *Checker) bindRunArguments(program *flow.Program, dispatch *flow.Dispatch, stage *ir.Function, args []ast.Expr, symbols map[string]programSymbol, current map[flow.ResourceID]flow.VersionID) error {
	seen := map[flow.ResourceID]bool{}
	for sourcePosition, formal := range stage.SourceParams {
		argument := args[sourcePosition]
		if formal.Kind == ir.SourceBuffer {
			identifier, ok := argument.(*ast.IdentExpr)
			symbol, exists := symbols[identifierName(identifier)]
			if !ok || !exists || symbol.resource == 0 {
				return diag(argument.GetSpan(), "buffer argument %s must name an external buffer or transient", formal.Name)
			}
			if seen[symbol.resource] {
				return diag(argument.GetSpan(), "stage %s receives resource %s in multiple buffer formals", stage.Name, identifier.Name)
			}
			seen[symbol.resource] = true
			if !foundation.Equal(stage.BufferParams[formal.Buffer].Type, symbol.type_) {
				return diag(argument.GetSpan(), "buffer argument for %s has type %s, want %s", formal.Name, symbol.type_, stage.BufferParams[formal.Buffer].Type)
			}
			dispatch.Buffers = append(dispatch.Buffers, flow.BufferArgument{Formal: formal.Buffer, Resource: symbol.resource, Input: current[symbol.resource]})
			continue
		}
		value, err := c.lowerProgramValue(program, argument, stage.Params[len(dispatch.Values)].Type, symbols)
		if err != nil {
			return err
		}
		value.Formal = len(dispatch.Values)
		dispatch.Values = append(dispatch.Values, value)
	}
	return nil
}

func identifierName(identifier *ast.IdentExpr) string {
	if identifier == nil {
		return ""
	}
	return identifier.Name
}

func (c *Checker) finishDispatch(program *flow.Program, dispatch *flow.Dispatch, id flow.DispatchID, current map[flow.ResourceID]flow.VersionID) error {
	stage := c.mod.Function(dispatch.Stage)
	summary := ir.AnalyzeAccess(stage)
	for i := range dispatch.Buffers {
		argument := &dispatch.Buffers[i]
		input := program.Version(argument.Input)
		access := summary.Buffers[argument.Formal]
		if (input == nil || !input.Defined) && access.Read {
			return diag(dispatch.Span, "stage %s reads resource %s before every element has been defined", stage.Name, program.Resource(argument.Resource).Name)
		}
		if stage.BufferParams[argument.Formal].Access == ir.Mutable {
			defined := input != nil && input.Defined || program.DispatchDefines(dispatch, *argument, access)
			output := program.AddVersion(flow.Version{Resource: argument.Resource, Previous: argument.Input, Producer: id, Defined: defined})
			argument.Output = output
			current[argument.Resource] = output
		}
	}
	return nil
}

func finishResources(program *flow.Program, current map[flow.ResourceID]flow.VersionID) {
	for i := range program.Resources {
		program.Resources[i].Final = current[program.Resources[i].ID]
	}
}

func (c *Checker) lowerProgramValue(program *flow.Program, expression ast.Expr, want *foundation.Type, symbols map[string]programSymbol) (flow.ValueArgument, error) {
	if identifier, ok := expression.(*ast.IdentExpr); ok {
		symbol, exists := symbols[identifier.Name]
		if exists && symbol.resource == 0 && symbol.parameter >= 0 && foundation.Equal(symbol.type_, want) {
			return flow.ValueArgument{Kind: flow.ValueParameterRef, Parameter: symbol.parameter}, nil
		}
		if exists && symbol.shape != 0 && foundation.Equal(want, foundation.Uint32Type) {
			return flow.ValueArgument{Kind: flow.ValueShape, Shape: symbol.shape}, nil
		}
	}
	if symbol, path, got, ok := programPath(expression, symbols); ok && symbol.resource == 0 && symbol.parameter >= 0 && foundation.Equal(got, want) {
		return flow.ValueArgument{Kind: flow.ValueParameterRef, Parameter: symbol.parameter, Path: path}, nil
	}
	if value, constant, err := c.tryConstant(expression, want, programEnvironment(symbols)); err != nil {
		return flow.ValueArgument{}, err
	} else if constant {
		return flow.ValueArgument{Kind: flow.ValueConstant, Constant: value}, nil
	}
	if want.Kind == foundation.Uint32Kind {
		shape, err := c.lowerShape(program, expression, symbols)
		if err == nil {
			return flow.ValueArgument{Kind: flow.ValueShape, Shape: shape}, nil
		}
	}
	return flow.ValueArgument{}, diag(expression.GetSpan(), "program argument is not a supported %s value source", want)
}

func (c *Checker) lowerShape(program *flow.Program, expression ast.Expr, symbols map[string]programSymbol) (flow.ShapeID, error) {
	if value, constant, err := c.tryConstant(expression, foundation.Uint32Type, programEnvironment(symbols)); err != nil {
		return 0, err
	} else if constant {
		return program.AddShape(flow.Shape{Op: flow.ShapeConstant, Value: value.Bits[0], Span: expression.GetSpan()}), nil
	}
	switch x := expression.(type) {
	case *ast.IdentExpr:
		symbol, ok := symbols[x.Name]
		if !ok {
			return 0, diag(x.Span, "unknown shape symbol %s", x.Name)
		}
		if symbol.shape != 0 {
			return symbol.shape, nil
		}
		if symbol.parameter >= 0 && symbol.resource == 0 && foundation.Equal(symbol.type_, foundation.Uint32Type) {
			return program.AddShape(flow.Shape{Op: flow.ShapeParameter, Parameter: symbol.parameter, Span: x.Span}), nil
		}
	case *ast.MemberExpr:
		if x.Name == "length" {
			if symbol, path, final, ok := programPath(x.Base, symbols); ok && symbol.resource != 0 && final.Kind == foundation.RuntimeArrayKind {
				if resource := program.Resource(symbol.resource); resource != nil && resource.Kind == flow.Transient && len(path) == 0 {
					return resource.Length, nil
				}
				return program.AddShape(flow.Shape{Op: flow.ShapeResourceLength, Resource: symbol.resource, Path: path, Span: x.Span}), nil
			}
		}
		if symbol, path, final, ok := programPath(x, symbols); ok && symbol.resource == 0 && symbol.parameter >= 0 && foundation.Equal(final, foundation.Uint32Type) {
			return program.AddShape(flow.Shape{Op: flow.ShapeParameter, Parameter: symbol.parameter, Path: path, Span: x.Span}), nil
		}
	case *ast.BinaryExpr:
		op := map[string]flow.ShapeOp{"+": flow.ShapeAdd, "-": flow.ShapeSub, "*": flow.ShapeMul, "/": flow.ShapeDiv, "%": flow.ShapeRem}[x.Op]
		if op == 0 {
			break
		}
		left, err := c.lowerShape(program, x.Left, symbols)
		if err != nil {
			return 0, err
		}
		right, err := c.lowerShape(program, x.Right, symbols)
		if err != nil {
			return 0, err
		}
		return program.AddShape(flow.Shape{Op: op, Left: left, Right: right, Span: x.Span}), nil
	case *ast.CallExpr:
		identifier, ok := x.Callee.(*ast.IdentExpr)
		if !ok || len(x.Args) != 2 {
			break
		}
		op := map[string]flow.ShapeOp{"min": flow.ShapeMin, "max": flow.ShapeMax, "ceilDiv": flow.ShapeCeilDiv}[identifier.Name]
		if op == 0 {
			break
		}
		left, err := c.lowerShape(program, x.Args[0], symbols)
		if err != nil {
			return 0, err
		}
		right, err := c.lowerShape(program, x.Args[1], symbols)
		if err != nil {
			return 0, err
		}
		return program.AddShape(flow.Shape{Op: op, Left: left, Right: right, Span: x.Span}), nil
	}
	return 0, diag(expression.GetSpan(), "expression is not a checked uint32 shape expression")
}

func programPath(expression ast.Expr, symbols map[string]programSymbol) (programSymbol, []string, *foundation.Type, bool) {
	switch x := expression.(type) {
	case *ast.IdentExpr:
		symbol, ok := symbols[x.Name]
		return symbol, nil, symbol.type_, ok
	case *ast.MemberExpr:
		symbol, path, parent, ok := programPath(x.Base, symbols)
		if !ok || parent == nil || parent.Kind != foundation.StructKind {
			return programSymbol{}, nil, nil, false
		}
		field := foundation.FieldIndex(parent, x.Name)
		if field < 0 {
			return programSymbol{}, nil, nil, false
		}
		return symbol, append(path, x.Name), parent.Fields[field].Type, true
	default:
		return programSymbol{}, nil, nil, false
	}
}
