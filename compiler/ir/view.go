package ir

import (
	"fmt"
	"tach/foundation"
)

func FusibleView(module *Module, program *Program) (int, int) {
	if program.View == nil || len(program.Dispatches) == 0 {
		return -1, -1
	}
	resource := program.Resource(program.View.Source)
	version := program.Version(program.View.Input)
	last := len(program.Dispatches) - 1
	if resource == nil || resource.Kind != TransientResourceKind || version == nil || version.Producer != program.Dispatches[last].ID || !shapeProduct(program, resource.Length, program.View.Width, program.View.Height) {
		return -1, -1
	}
	for _, dispatch := range program.Dispatches[:last] {
		for _, argument := range dispatch.Buffers {
			if argument.Resource == resource.ID {
				return -1, -1
			}
		}
	}
	dispatch := program.Dispatches[last]
	stage := module.Kernel.Function(dispatch.Stage)
	if stage == nil {
		return -1, -1
	}
	summary := AnalyzeAccess(stage)
	for _, argument := range dispatch.Buffers {
		if argument.Resource != resource.ID || argument.Output != program.View.Input {
			continue
		}
		access := summary.Buffers[argument.Formal]
		if program.DispatchDefines(&dispatch, argument, access) && len(access.Accesses[0].FieldPath) == 0 {
			return last, argument.Formal
		}
	}
	return -1, -1
}

func shapeProduct(program *Program, product, left, right ShapeID) bool {
	shape := program.Shape(product)
	if shape != nil && shape.Op == ShapeMul && (shape.Left == left && shape.Right == right || shape.Left == right && shape.Right == left) {
		return true
	}
	a, b := program.Shape(left), program.Shape(right)
	if a == nil || b == nil {
		return false
	}
	if (product == left && b.Op == ShapeConstant && b.Value == 1) || (product == right && a.Op == ShapeConstant && a.Value == 1) {
		return true
	}
	return shape != nil && shape.Op == ShapeConstant && a.Op == ShapeConstant && b.Op == ShapeConstant && uint64(a.Value)*uint64(b.Value) == uint64(shape.Value)
}

func AppendViewExtent(function *Function, values *[]ValueArgument, view *View) (ValueID, ValueID) {
	next := MaxValueID(function) + 1
	width, height := next, next+1
	for _, parameter := range []Param{{Name: "__tach_view_width", ID: width, Type: foundation.Uint32Type}, {Name: "__tach_view_height", ID: height, Type: foundation.Uint32Type}} {
		function.Params = append(function.Params, parameter)
		function.SourceParams = append(function.SourceParams, SourceParam{Name: parameter.Name, Kind: SourceValue, Value: parameter.ID, Buffer: -1})
	}
	formal := len(*values)
	*values = append(*values,
		ValueArgument{Formal: formal, Kind: ValueFromShape, Shape: view.Width},
		ValueArgument{Formal: formal + 1, Kind: ValueFromShape, Shape: view.Height},
	)
	return width, height
}

func FuseView(function *Function, binding int) error {
	output := foundation.RuntimeArrayOf(foundation.Uint32Type)
	function.BufferParams[binding].Type = output
	places := map[PlaceID]bool{}
	next, stores := MaxValueID(function)+1, 0
	var rewrite func(*Block) error
	rewrite = func(block *Block) error {
		var instructions []Instr
		for _, instruction := range block.Instrs {
			switch x := instruction.(type) {
			case *PlaceRoot:
				if x.Buffer == binding {
					x.Type, places[x.Result] = output, true
				}
			case *PlaceField:
				if places[x.Base] {
					return fmt.Errorf("fused view output contains a field access")
				}
			case *PlaceIndex:
				if places[x.Base] {
					x.Type, places[x.Result] = foundation.Uint32Type, true
				}
			case *Store:
				if places[x.Place] {
					packed, value := packViewRGBA(x.Value, &next)
					instructions = append(instructions, packed...)
					x.Value, stores = value, stores+1
				}
			case *If:
				if err := rewrite(x.Then); err != nil {
					return err
				}
				if err := rewrite(x.Else); err != nil {
					return err
				}
			case *Loop:
				if err := rewrite(x.Cond); err != nil {
					return err
				}
				if err := rewrite(x.Body); err != nil {
					return err
				}
			case *Scope:
				if err := rewrite(x.Body); err != nil {
					return err
				}
			}
			instructions = append(instructions, instruction)
		}
		block.Instrs = instructions
		return nil
	}
	if err := rewrite(function.Body); err != nil {
		return err
	}
	if stores != 1 {
		return fmt.Errorf("fused view stage has %d output stores", stores)
	}
	return nil
}

const (
	viewUnitHelper = "$tach_unit"
	viewSRGBHelper = "$tach_srgb"
)

func ViewHelpers() []*Function { return []*Function{viewUnitFunction(), viewSRGBFunction()} }

func ViewProjectionFunction() *Function {
	pixel := foundation.VectorOf(foundation.Float32Type, 4)
	function := &Function{
		Kind:         Stage,
		Indices:      []Param{{Name: "x", ID: 1, Type: foundation.Uint32Type}, {Name: "y", ID: 2, Type: foundation.Uint32Type}},
		Params:       []Param{{Name: "width", ID: 3, Type: foundation.Uint32Type}, {Name: "height", ID: 4, Type: foundation.Uint32Type}},
		BufferParams: []BufferParam{{Name: "pixels", Type: foundation.RuntimeArrayOf(pixel), Access: Read}, {Name: "output", Type: foundation.RuntimeArrayOf(foundation.Uint32Type), Access: Mutable}},
		SourceParams: []SourceParam{{Name: "pixels", Kind: SourceBuffer, Buffer: 0}, {Name: "output", Kind: SourceBuffer, Buffer: 1}, {Name: "width", Kind: SourceValue, Value: 3, Buffer: -1}, {Name: "height", Kind: SourceValue, Value: 4, Buffer: -1}},
		Return:       foundation.VoidType, Workgroup: WorkgroupConstraint{Explicit: true, Size: [3]uint32{16, 16, 1}},
	}
	function.Body = viewProjectionBody(pixel)
	return function
}

func viewSRGBFunction() *Function {
	return &Function{
		Name:   viewSRGBHelper,
		Kind:   Helper,
		Params: []Param{{Name: "value", ID: 1, Type: foundation.Float32Type}},
		Return: foundation.Float32Type,
		Body: &Block{
			Instrs: []Instr{
				&Call{Result: 2, Type: foundation.Float32Type, Function: viewUnitHelper, Args: []ValueID{1}},
				&Const{Result: 3, Type: foundation.Float32Type, Raw: "0.0031308"},
				&Binary{Result: 4, Type: foundation.BoolType, Op: "<=", Left: 2, Right: 3},
				&Const{Result: 5, Type: foundation.Float32Type, Raw: "12.92"},
				&Binary{Result: 6, Type: foundation.Float32Type, Op: "*", Left: 2, Right: 5},
				&Const{Result: 7, Type: foundation.Float32Type, Raw: "0.416666667"},
				&Intrinsic{Result: 8, Type: foundation.Float32Type, Kind: IntrinsicPow, Args: []ValueID{2, 7}},
				&Const{Result: 9, Type: foundation.Float32Type, Raw: "1.055"},
				&Binary{Result: 10, Type: foundation.Float32Type, Op: "*", Left: 8, Right: 9},
				&Const{Result: 11, Type: foundation.Float32Type, Raw: "0.055"},
				&Binary{Result: 12, Type: foundation.Float32Type, Op: "-", Left: 10, Right: 11},
				&If{Results: []Result{{ID: 13, Type: foundation.Float32Type}}, Cond: 4, Then: &Block{Term: &Yield{Values: []ValueID{6}}}, Else: &Block{Term: &Yield{Values: []ValueID{12}}}},
			},
			Term: &Return{Value: 13, HasValue: true},
		},
	}
}

func viewUnitFunction() *Function {
	return &Function{
		Name: viewUnitHelper, Kind: Helper, Params: []Param{{Name: "value", ID: 1, Type: foundation.Float32Type}}, Return: foundation.Float32Type,
		Body: &Block{
			Instrs: []Instr{
				&Const{Result: 2, Type: foundation.Float32Type, Raw: "0.0"},
				&Const{Result: 3, Type: foundation.Float32Type, Raw: "1.0"},
				&Binary{Result: 4, Type: foundation.BoolType, Op: ">", Left: 1, Right: 2},
				&Binary{Result: 5, Type: foundation.BoolType, Op: "<", Left: 1, Right: 3},
				&If{Results: []Result{{ID: 6, Type: foundation.Float32Type}}, Cond: 5, Then: &Block{Term: &Yield{Values: []ValueID{1}}}, Else: &Block{Term: &Yield{Values: []ValueID{3}}}},
				&If{Results: []Result{{ID: 7, Type: foundation.Float32Type}}, Cond: 4, Then: &Block{Term: &Yield{Values: []ValueID{6}}}, Else: &Block{Term: &Yield{Values: []ValueID{2}}}},
			},
			Term: &Return{Value: 7, HasValue: true},
		},
	}
}

func viewProjectionBody(pixel *foundation.Type) *Block {
	then := &Block{Instrs: []Instr{
		&Binary{Result: 8, Type: foundation.Uint32Type, Op: "*", Left: 2, Right: 3},
		&Binary{Result: 9, Type: foundation.Uint32Type, Op: "+", Left: 8, Right: 1},
		&PlaceRoot{Result: 1, Type: foundation.RuntimeArrayOf(pixel), Buffer: 0},
		&PlaceIndex{Result: 2, Type: pixel, Base: 1, Index: 9},
		&Load{Result: 10, Type: pixel, Place: 2},
	}}
	next := ValueID(11)
	packed, merged := packViewRGBA(10, &next)
	then.Instrs = append(then.Instrs, packed...)
	then.Instrs = append(then.Instrs,
		&PlaceRoot{Result: 3, Type: foundation.RuntimeArrayOf(foundation.Uint32Type), Buffer: 1},
		&PlaceIndex{Result: 4, Type: foundation.Uint32Type, Base: 3, Index: 9},
		&Store{Place: 4, Value: merged},
	)
	then.Term = &Yield{}
	return &Block{
		Instrs: []Instr{
			&Binary{Result: 5, Type: foundation.BoolType, Op: "<", Left: 1, Right: 3},
			&Binary{Result: 6, Type: foundation.BoolType, Op: "<", Left: 2, Right: 4},
			&Binary{Result: 7, Type: foundation.BoolType, Op: "&&", Left: 5, Right: 6},
			&If{Cond: 7, Then: then, Else: &Block{Term: &Yield{}}},
		},
		Term: &Return{},
	}
}

func packViewRGBA(value ValueID, next *ValueID) ([]Instr, ValueID) {
	newValue := func() ValueID {
		id := *next
		*next = id + 1
		return id
	}
	var instructions []Instr
	channels := make([]ValueID, 4)
	for index := range channels {
		channels[index] = newValue()
		instructions = append(instructions, &Extract{Result: channels[index], Type: foundation.Float32Type, Base: value, Index: index})
	}
	for index := range 3 {
		encoded := newValue()
		instructions = append(instructions, &Call{Result: encoded, Type: foundation.Float32Type, Function: viewSRGBHelper, Args: []ValueID{channels[index]}})
		channels[index] = encoded
	}
	alpha := newValue()
	instructions = append(instructions, &Call{Result: alpha, Type: foundation.Float32Type, Function: viewUnitHelper, Args: []ValueID{channels[3]}})
	channels[3] = alpha
	scale, half := newValue(), newValue()
	instructions = append(instructions,
		&Const{Result: scale, Type: foundation.Float32Type, Raw: "255.0"},
		&Const{Result: half, Type: foundation.Float32Type, Raw: "0.5"},
	)
	packed := make([]ValueID, 4)
	for index, channel := range channels {
		multiply, round, convert := newValue(), newValue(), newValue()
		instructions = append(instructions,
			&Binary{Result: multiply, Type: foundation.Float32Type, Op: "*", Left: channel, Right: scale},
			&Binary{Result: round, Type: foundation.Float32Type, Op: "+", Left: multiply, Right: half},
			&Convert{Result: convert, Type: foundation.Uint32Type, X: round, From: foundation.Float32Type},
		)
		packed[index] = convert
	}
	for index := 1; index < 4; index++ {
		shift, shifted := newValue(), newValue()
		instructions = append(instructions,
			&Const{Result: shift, Type: foundation.Uint32Type, Raw: fmt.Sprintf("%d", index*8)},
			&Binary{Result: shifted, Type: foundation.Uint32Type, Op: "<<", Left: packed[index], Right: shift},
		)
		packed[index] = shifted
	}
	merged := packed[0]
	for index := 1; index < 4; index++ {
		result := newValue()
		instructions = append(instructions, &Binary{Result: result, Type: foundation.Uint32Type, Op: "|", Left: merged, Right: packed[index]})
		merged = result
	}
	return instructions, merged
}
