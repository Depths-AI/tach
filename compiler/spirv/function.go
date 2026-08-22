package spirv

import (
	"fmt"
	"tach/foundation"
	"tach/ir"
)

type spvPlace struct {
	ptr         uint32
	ty          *foundation.Type
	storage     uint32
	resource    int
	arrayBase   uint32 // pointer to struct containing a runtime array
	arrayMember uint32
	hasArrayLen bool
}

type fnEmitter struct {
	b *builder
	f *ir.Function

	values map[ir.ValueID]uint32
	vtypes map[ir.ValueID]*foundation.Type
	places map[ir.PlaceID]spvPlace
	inputs map[inputKind]uint32

	currentLabel uint32
	terminated   bool
	scopeMerges  []uint32
	loops        []loopTarget
}

func (b *builder) emitFunction(f *ir.Function) error {
	retType, err := b.typeID(f.Return, typeLogical)
	if err != nil {
		return err
	}
	parameters := f.Params
	if f.Kind == ir.Stage {
		parameters = nil
	}
	fnType, err := b.functionTypeID(f.Return, parameters)
	if err != nil {
		return err
	}
	fid := b.funcIDs[f.Name]
	control := FunctionControlNone
	if f.Kind == ir.Helper {
		control = FunctionControlInline | FunctionControlConst
	}
	emit(&b.functions, OpFunction, retType, fid, control, fnType)

	s := &fnEmitter{b: b, f: f, values: map[ir.ValueID]uint32{}, vtypes: map[ir.ValueID]*foundation.Type{}, places: map[ir.PlaceID]spvPlace{}, inputs: map[inputKind]uint32{}}
	for _, p := range parameters {
		pt, err := b.typeID(p.Type, typeLogical)
		if err != nil {
			return err
		}
		id := b.id()
		emit(&b.functions, OpFunctionParameter, pt, id)
		s.values[p.ID] = id
		s.vtypes[p.ID] = p.Type
	}
	label := b.id()
	emit(&b.functions, OpLabel, label)
	s.currentLabel = label
	if err := s.emitParameters(); err != nil {
		return err
	}
	if err := s.emitCoordinates(); err != nil {
		return err
	}
	if err := s.emitBlock(f.Body, blockNormal); err != nil {
		return err
	}
	if !s.terminated {
		return fmt.Errorf("function ended without a terminator")
	}
	emit(&b.functions, OpFunctionEnd)
	return nil
}

func (s *fnEmitter) useGlobal(id uint32) {
	used := s.b.globalUses[s.f.Name]
	if used == nil {
		used = map[uint32]bool{}
		s.b.globalUses[s.f.Name] = used
	}
	used[id] = true
}

func (s *fnEmitter) emitParameters() error {
	if s.f.Kind != ir.Stage || len(s.f.Params) == 0 {
		return nil
	}
	block := s.b.parameterBlocks[s.f]
	if block == nil || s.b.parameterIDs[s.f] == 0 {
		return fmt.Errorf("kernel %s parameter block was not declared", s.f.Name)
	}
	cursor := 0
	for parameter, value := range s.f.Params {
		id, err := s.emitParameterValue(block, parameter, value.Type, &cursor)
		if err != nil {
			return err
		}
		s.def(value.ID, id, value.Type)
	}
	if cursor != len(block.Fields) {
		return fmt.Errorf("kernel %s parameter block consumed %d of %d fields", s.f.Name, cursor, len(block.Fields))
	}
	return nil
}

func (s *fnEmitter) emitParameterValue(block *ir.HostParameterBlock, parameter int, logical *foundation.Type, cursor *int) (uint32, error) {
	if logical.Kind == foundation.StructKind {
		values := make([]uint32, len(logical.Fields))
		for index, field := range logical.Fields {
			value, err := s.emitParameterValue(block, parameter, field.Type, cursor)
			if err != nil {
				return 0, err
			}
			values[index] = value
		}
		typeID, err := s.b.typeID(logical, typeLogical)
		if err != nil {
			return 0, err
		}
		result := s.b.id()
		emit(&s.b.functions, OpCompositeConstruct, append([]uint32{typeID, result}, values...)...)
		return result, nil
	}
	if block == nil || *cursor >= len(block.Fields) {
		return 0, fmt.Errorf("kernel parameter %d has no physical field", parameter)
	}
	fieldIndex := *cursor
	field := block.Fields[fieldIndex]
	*cursor = fieldIndex + 1
	if field.Parameter != parameter || !foundation.Equal(field.Logical, logical) {
		return 0, fmt.Errorf("kernel parameter %d field %d does not match %s", parameter, fieldIndex, logical)
	}
	pointer, err := s.b.pointerID(StorageUniform, field.Physical)
	if err != nil {
		return 0, err
	}
	index, err := s.b.u32Constant(uint32(fieldIndex))
	if err != nil {
		return 0, err
	}
	address := s.b.id()
	variable := s.b.parameterIDs[s.f]
	if variable == 0 {
		return 0, fmt.Errorf("kernel %s parameter block was not declared", s.f.Name)
	}
	s.useGlobal(variable)
	emit(&s.b.functions, OpAccessChain, pointer, address, variable, index)
	typeID, err := s.b.typeID(field.Physical, typeLogical)
	if err != nil {
		return 0, err
	}
	loaded := s.b.id()
	if err := s.emitLoad(typeID, loaded, address, StorageUniform, field.Physical); err != nil {
		return 0, err
	}
	if logical.Kind != foundation.BoolKind {
		return loaded, nil
	}
	zero, err := s.b.u32Constant(0)
	if err != nil {
		return 0, err
	}
	boolType, err := s.b.typeID(foundation.BoolType, typeLogical)
	if err != nil {
		return 0, err
	}
	result := s.b.id()
	emit(&s.b.functions, OpINotEqual, boolType, result, loaded, zero)
	return result, nil
}

func (s *fnEmitter) emitCoordinates() error {
	lowered := s.b.p.functions[s.f]
	active := false
	for _, value := range lowered.Order {
		active = active || lowered.Uses[value] > 0
	}
	if !active {
		return nil
	}
	u32Type, err := s.b.typeID(foundation.Uint32Type, typeLogical)
	if err != nil {
		return err
	}
	for _, value := range lowered.Order {
		if lowered.Uses[value] == 0 {
			continue
		}
		input, dimension := coordinate(lowered, value)
		loaded := s.inputs[input]
		if loaded == 0 {
			variable := s.b.inputIDs[input]
			if variable == 0 {
				return fmt.Errorf("coordinate input %d was not declared", input)
			}
			s.useGlobal(variable)
			t := foundation.Uint32Type
			if input != inputLocalLinear {
				t = foundation.VectorOf(foundation.Uint32Type, 3)
			}
			typeID, err := s.b.typeID(t, typeLogical)
			if err != nil {
				return err
			}
			loaded = s.b.id()
			if err := s.emitLoad(typeID, loaded, variable, StorageInput, t); err != nil {
				return err
			}
			s.inputs[input] = loaded
		}
		if input == inputLocalLinear {
			s.def(value, loaded, foundation.Uint32Type)
			continue
		}
		id := s.b.id()
		emit(&s.b.functions, OpCompositeExtract, u32Type, id, loaded, dimension)
		s.def(value, id, foundation.Uint32Type)
	}
	return nil
}

type blockMode uint8

const (
	blockNormal blockMode = iota
	blockYield
	blockScope
)

type blockExit struct {
	kind  blockMode
	vals  []uint32
	pred  uint32
	falls bool
}

type phiIncoming struct{ value, label uint32 }

type loopTarget struct {
	merge, continuing uint32
	breaks, continues [][]phiIncoming
}

func (s *fnEmitter) emitBlock(bl *ir.Block, mode blockMode) error {
	_, err := s.emitBlockExit(bl, mode)
	return err
}

func (s *fnEmitter) emitLoopTransfer(values []ir.ValueID, pred uint32, breaking bool) (blockExit, error) {
	if len(s.loops) == 0 {
		return blockExit{}, fmt.Errorf("loop transfer outside loop")
	}
	loop := &s.loops[len(s.loops)-1]
	incoming, target, name := loop.continues, loop.continuing, "continue"
	if breaking {
		incoming, target, name = loop.breaks, loop.merge, "break"
	}
	if len(values) != len(incoming) {
		return blockExit{}, fmt.Errorf("%s carries %d values, want %d", name, len(values), len(incoming))
	}
	for i, id := range values {
		value, err := s.value(id)
		if err != nil {
			return blockExit{}, err
		}
		incoming[i] = append(incoming[i], phiIncoming{value, pred})
	}
	emit(&s.b.functions, OpBranch, target)
	s.terminated = true
	return blockExit{pred: pred}, nil
}

func (s *fnEmitter) emitBlockExit(bl *ir.Block, mode blockMode) (blockExit, error) {
	for _, in := range bl.Instrs {
		if s.terminated {
			return blockExit{}, fmt.Errorf("instruction %T emitted after block termination", in)
		}
		if err := s.emitInstr(in); err != nil {
			return blockExit{}, err
		}
	}
	pred := s.currentLabel
	switch t := bl.Term.(type) {
	case *ir.Return:
		if t.HasValue {
			v, err := s.value(t.Value)
			if err != nil {
				return blockExit{}, err
			}
			emit(&s.b.functions, OpReturnValue, v)
		} else {
			emit(&s.b.functions, OpReturn)
		}
		s.terminated = true
		return blockExit{pred: pred}, nil
	case *ir.Unreachable:
		emit(&s.b.functions, OpUnreachable)
		s.terminated = true
		return blockExit{pred: pred}, nil
	case *ir.Yield:
		if mode != blockYield {
			return blockExit{}, fmt.Errorf("yield outside structured selection/condition")
		}
		vals := make([]uint32, len(t.Values))
		for i, id := range t.Values {
			v, err := s.value(id)
			if err != nil {
				return blockExit{}, err
			}
			vals[i] = v
		}
		return blockExit{kind: blockYield, vals: vals, pred: pred, falls: true}, nil
	case *ir.Continue:
		return s.emitLoopTransfer(t.Values, pred, false)
	case *ir.Break:
		return s.emitLoopTransfer(t.Values, pred, true)
	case *ir.ExitScope:
		if len(s.scopeMerges) == 0 {
			return blockExit{}, fmt.Errorf("exit_scope outside scope")
		}
		emit(&s.b.functions, OpBranch, s.scopeMerges[len(s.scopeMerges)-1])
		s.terminated = true
		return blockExit{kind: blockScope, pred: pred}, nil
	default:
		return blockExit{}, fmt.Errorf("unknown block terminator %T", bl.Term)
	}
}

func (s *fnEmitter) emitScope(scope *ir.Scope) error {
	header, body, cont, merge := s.b.id(), s.b.id(), s.b.id(), s.b.id()
	again, err := s.b.constant(foundation.BoolType, "false")
	if err != nil {
		return err
	}
	emit(&s.b.functions, OpBranch, header)
	emit(&s.b.functions, OpLabel, header)
	emit(&s.b.functions, OpLoopMerge, merge, cont, LoopControlNone)
	emit(&s.b.functions, OpBranch, body)
	s.terminated = true
	emit(&s.b.functions, OpLabel, body)
	s.currentLabel, s.terminated = body, false
	s.scopeMerges = append(s.scopeMerges, merge)
	exit, err := s.emitBlockExit(scope.Body, blockScope)
	s.scopeMerges = s.scopeMerges[:len(s.scopeMerges)-1]
	if err != nil {
		return err
	}
	if exit.falls {
		emit(&s.b.functions, OpBranch, cont)
		s.terminated = true
	}
	emit(&s.b.functions, OpLabel, cont)
	emit(&s.b.functions, OpBranchConditional, again, header, merge)
	emit(&s.b.functions, OpLabel, merge)
	s.currentLabel, s.terminated = merge, false
	return nil
}
