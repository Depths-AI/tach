package ir

import (
	"fmt"
	"tach/foundation"
)

func SpecializeParameters(function *Function, values []ValueArgument) ([]ValueArgument, error) {
	uses, _, err := UseCounts(function)
	if err != nil {
		return nil, err
	}
	next := MaxValueID(function)
	fresh := func() ValueID { next++; return next }
	replacements := map[ValueID]ValueID{}
	var definitions []Instr
	parameters := make([]Param, 0, len(function.Params))
	kept := make([]ValueArgument, 0, len(values))
	for index, parameter := range function.Params {
		if uses[parameter.ID] == 0 {
			continue
		}
		value := values[index]
		if value.Kind != ValueFromConstant {
			value.Formal = len(kept)
			parameters, kept = append(parameters, parameter), append(kept, value)
			continue
		}
		replacement, added := MaterializeConstant(value.Constant, function.Span, fresh)
		definitions, replacements[parameter.ID] = append(definitions, added...), replacement
	}
	if len(replacements) > 0 {
		ReplaceValueUses(function, replacements)
		function.Body.Instrs = append(definitions, function.Body.Instrs...)
	}
	function.Params = parameters
	function.SourceParams = function.SourceParams[:0]
	for index, parameter := range function.BufferParams {
		function.SourceParams = append(function.SourceParams, SourceParam{Name: parameter.Name, Kind: SourceBuffer, Buffer: index})
	}
	for _, parameter := range function.Params {
		function.SourceParams = append(function.SourceParams, SourceParam{Name: parameter.Name, Kind: SourceValue, Value: parameter.ID, Buffer: -1})
	}
	return kept, nil
}

func AppendLogicalLengths(function *Function, values *[]ValueArgument, program *Program, dispatch *Dispatch) map[int]ValueID {
	lengths := map[int]ValueID{}
	next := MaxValueID(function) + 1
	for buffer, parameter := range function.BufferParams {
		path, ok := f16RuntimePath(parameter.Type)
		if !ok || !usesBufferLength(function.Body, buffer, map[PlaceID]bool{}) {
			continue
		}
		for _, argument := range dispatch.Buffers {
			if argument.Formal != buffer {
				continue
			}
			shape := program.AddShape(Shape{Op: ShapeResourceLength, Resource: argument.Resource, Path: path, Span: dispatch.Span})
			formal := len(function.Params)
			function.Params = append(function.Params, Param{Name: fmt.Sprintf("__tach_length_%d", buffer), ID: next, Type: foundation.Uint32Type})
			function.SourceParams = append(function.SourceParams, SourceParam{Name: function.Params[formal].Name, Kind: SourceValue, Value: next, Buffer: -1})
			*values = append(*values, ValueArgument{Formal: formal, Kind: ValueFromShape, Shape: shape})
			lengths[buffer], next = next, next+1
			break
		}
	}
	return lengths
}

func f16RuntimePath(t *foundation.Type) ([]string, bool) {
	if t.Kind == foundation.RuntimeArrayKind {
		return nil, t.Elem.Kind == foundation.Float16Kind
	}
	if t.Kind == foundation.StructKind && len(t.Fields) > 0 {
		tail := t.Fields[len(t.Fields)-1]
		if tail.Type.Kind == foundation.RuntimeArrayKind && tail.Type.Elem.Kind == foundation.Float16Kind {
			return []string{tail.Name}, true
		}
	}
	return nil, false
}

func usesBufferLength(block *Block, buffer int, places map[PlaceID]bool) bool {
	for _, instruction := range block.Instrs {
		switch item := instruction.(type) {
		case *PlaceRoot:
			places[item.Result] = item.Buffer == buffer
		case *PlaceField:
			places[item.Result] = places[item.Base]
		case *PlaceIndex:
			places[item.Result] = places[item.Base]
		case *ArrayLength:
			if places[item.Place] {
				return true
			}
		case *If:
			if usesBufferLength(item.Then, buffer, places) || usesBufferLength(item.Else, buffer, places) {
				return true
			}
		case *Loop:
			if usesBufferLength(item.Cond, buffer, places) || usesBufferLength(item.Body, buffer, places) {
				return true
			}
		case *Scope:
			if usesBufferLength(item.Body, buffer, places) {
				return true
			}
		}
	}
	return false
}

func CanInternalizeRepeat(function *Function) bool {
	if function == nil || containsLoop(function.Body) {
		return false
	}
	summary := AnalyzeAccess(function)
	if summary.Effects.Atomic || summary.Effects.Workgroup || summary.Effects.Barrier {
		return false
	}
	for _, buffer := range summary.Buffers {
		for _, access := range buffer.Accesses {
			if len(access.Indices) != 1 || !access.Indices[0].Exact || access.Indices[0].Constant != 0 || access.Indices[0].Coefficient != [3]int64{1, 0, 0} {
				return false
			}
		}
	}
	return true
}

func containsLoop(block *Block) bool {
	if block == nil {
		return false
	}
	for _, instruction := range block.Instrs {
		switch x := instruction.(type) {
		case *Loop:
			return true
		case *If:
			if containsLoop(x.Then) || containsLoop(x.Else) {
				return true
			}
		case *Scope:
			if containsLoop(x.Body) {
				return true
			}
		}
	}
	return false
}

func InternalizeRepeat(function *Function) error {
	next := MaxValueID(function)
	next++
	repeat := Param{Name: "__tach_repeat", ID: next, Type: foundation.Uint32Type}
	function.Params = append(function.Params, repeat)
	function.SourceParams = append(function.SourceParams, SourceParam{Name: repeat.Name, Kind: SourceValue, Value: repeat.ID, Buffer: -1})
	if !rewriteReturns(function.Body) {
		return fmt.Errorf("stage %s has a value return", function.Name)
	}
	next++
	zero := next
	next++
	result := next
	next++
	parameter := next
	next++
	condition := next
	next++
	one := next
	next++
	incremented := next
	original := function.Body
	function.Body = &Block{
		Instrs: []Instr{
			&Const{Result: zero, Type: foundation.Uint32Type, Raw: "0", Span: function.Span},
			&Loop{
				Results: []Result{{ID: result, Type: foundation.Uint32Type}},
				Params:  []LoopParam{{ID: parameter, Type: foundation.Uint32Type, Init: zero}},
				Cond: &Block{Instrs: []Instr{
					&Binary{Result: condition, Type: foundation.BoolType, Op: "<", Left: parameter, Right: repeat.ID, Span: function.Span},
				}, Term: &Yield{Values: []ValueID{condition}}},
				Body: &Block{Instrs: []Instr{
					&Scope{Body: original, Span: function.Span},
					&Const{Result: one, Type: foundation.Uint32Type, Raw: "1", Span: function.Span},
					&Binary{Result: incremented, Type: foundation.Uint32Type, Op: "+", Left: parameter, Right: one, Span: function.Span},
				}, Term: &Continue{Values: []ValueID{incremented}}},
				Span: function.Span,
			},
		},
		Term: &Return{},
	}
	return nil
}

func rewriteReturns(block *Block) bool {
	if block == nil {
		return true
	}
	for _, instruction := range block.Instrs {
		switch x := instruction.(type) {
		case *If:
			if !rewriteReturns(x.Then) || !rewriteReturns(x.Else) {
				return false
			}
		case *Scope:
			if !rewriteReturns(x.Body) {
				return false
			}
		case *Loop:
			return false
		}
	}
	if ret, ok := block.Term.(*Return); ok {
		if ret.HasValue {
			return false
		}
		block.Term = &ExitScope{}
	}
	return true
}

func CloneFunction(function *Function) *Function {
	module := CloneKernel(&KernelModule{Functions: []*Function{function}})
	return module.Functions[0]
}
