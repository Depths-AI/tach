package spirv

import (
	"encoding/binary"
	"fmt"
	"math"
	"strconv"
	"strings"

	"tach/src/abi"
	"tach/src/ir"
	"tach/src/layout"
	"tach/src/types"
)

// Emit lowers verified Tach IR to a SPIR-V 1.3 compute module and immediately
// parses and validates the produced binary with Tach's own SPIR-V validator.
func Emit(m *ir.Module) ([]byte, error) {
	if err := ir.Verify(m); err != nil {
		return nil, fmt.Errorf("IR verification: %w", err)
	}
	b := newBuilder(m)
	if err := b.build(); err != nil {
		return nil, err
	}
	words := b.words()
	out := make([]byte, 4*len(words))
	for i, w := range words {
		binary.LittleEndian.PutUint32(out[i*4:], w)
	}
	if err := Validate(out); err != nil {
		return nil, fmt.Errorf("Tach SPIR-V self-validation failed: %w", err)
	}
	return out, nil
}

type builder struct {
	m *ir.Module

	nextID uint32

	capabilities []uint32
	extImports   []uint32
	memoryModel  []uint32
	entryPoints  []uint32
	execModes    []uint32
	debug        []uint32
	annotations  []uint32
	typesGlobals []uint32
	functions    []uint32

	types        map[string]uint32
	pointers     map[string]uint32
	fnTypes      map[string]uint32
	constants    map[string]uint32
	funcIDs      map[string]uint32
	resourceIDs  []uint32
	builtinIDs   map[ir.BuiltinKind]uint32
	workgroupIDs map[string][]uint32
	glsl450      uint32
}

func newBuilder(m *ir.Module) *builder {
	return &builder{
		m:            m,
		nextID:       1,
		types:        map[string]uint32{},
		pointers:     map[string]uint32{},
		fnTypes:      map[string]uint32{},
		constants:    map[string]uint32{},
		funcIDs:      map[string]uint32{},
		builtinIDs:   map[ir.BuiltinKind]uint32{},
		workgroupIDs: map[string][]uint32{},
	}
}

func (b *builder) id() uint32 {
	id := b.nextID
	b.nextID++
	return id
}

func emit(dst *[]uint32, op Op, operands ...uint32) int {
	start := len(*dst)
	wc := uint32(len(operands) + 1)
	*dst = append(*dst, wc<<16|uint32(op))
	*dst = append(*dst, operands...)
	return start
}

func encodeString(s string) []uint32 {
	buf := append([]byte(s), 0)
	for len(buf)%4 != 0 {
		buf = append(buf, 0)
	}
	words := make([]uint32, len(buf)/4)
	for i := range words {
		words[i] = binary.LittleEndian.Uint32(buf[i*4:])
	}
	return words
}

func (b *builder) build() error {
	emit(&b.capabilities, OpCapability, CapabilityShader)
	emit(&b.memoryModel, OpMemoryModel, AddressingLogical, MemoryGLSL450)

	// Function IDs must exist before entry-point declarations and forward calls.
	for _, f := range b.m.Functions {
		b.funcIDs[f.Name] = b.id()
		emit(&b.debug, OpName, append([]uint32{b.funcIDs[f.Name]}, encodeString(f.Name)...)...)
	}

	if err := b.emitStructDebugAndLayouts(); err != nil {
		return err
	}
	if err := b.emitResources(); err != nil {
		return err
	}
	if err := b.emitBuiltins(); err != nil {
		return err
	}
	if err := b.emitWorkgroups(); err != nil {
		return err
	}
	b.emitEntryPoints()

	for _, f := range b.m.Functions {
		if err := b.emitFunction(f); err != nil {
			return fmt.Errorf("SPIR-V lower %s: %w", f.Name, err)
		}
	}
	return nil
}

func (b *builder) words() []uint32 {
	out := []uint32{Magic, Version, 0, b.nextID, 0}
	for _, s := range [][]uint32{b.capabilities, b.extImports, b.memoryModel, b.entryPoints, b.execModes, b.debug, b.annotations, b.typesGlobals, b.functions} {
		out = append(out, s...)
	}
	return out
}

func (b *builder) ensureGLSL450() uint32 {
	if b.glsl450 != 0 {
		return b.glsl450
	}
	b.glsl450 = b.id()
	ops := append([]uint32{b.glsl450}, encodeString("GLSL.std.450")...)
	emit(&b.extImports, OpExtInstImport, ops...)
	return b.glsl450
}

func typeKey(t *types.Type) string {
	if t == nil {
		return "<nil>"
	}
	switch t.Kind {
	case types.Vector:
		return fmt.Sprintf("v%d:%s", t.Lanes, typeKey(t.Elem))
	case types.RuntimeArray:
		return "ra:" + typeKey(t.Elem)
	case types.Struct:
		return "s:" + t.Name
	default:
		return t.String()
	}
}

func (b *builder) typeID(t *types.Type) (uint32, error) {
	key := typeKey(t)
	if id := b.types[key]; id != 0 {
		return id, nil
	}
	if t.Kind == types.Atomic {
		// SPIR-V atomicity is an operation property; the pointed-to object keeps
		// the underlying integer type. Resolve it before allocating an ID so the
		// module ID space remains dense and deterministic.
		elem, err := b.typeID(t.Elem)
		if err != nil {
			return 0, err
		}
		b.types[key] = elem
		return elem, nil
	}
	id := b.id()
	b.types[key] = id
	switch t.Kind {
	case types.Void:
		emit(&b.typesGlobals, OpTypeVoid, id)
	case types.Bool:
		emit(&b.typesGlobals, OpTypeBool, id)
	case types.I32:
		emit(&b.typesGlobals, OpTypeInt, id, 32, 1)
	case types.U32:
		emit(&b.typesGlobals, OpTypeInt, id, 32, 0)
	case types.F32:
		emit(&b.typesGlobals, OpTypeFloat, id, 32)
	case types.Vector:
		elem, err := b.typeID(t.Elem)
		if err != nil {
			return 0, err
		}
		emit(&b.typesGlobals, OpTypeVector, id, elem, uint32(t.Lanes))
	case types.FixedArray:
		elem, err := b.typeID(t.Elem)
		if err != nil {
			return 0, err
		}
		length, err := b.u32Constant(t.Count)
		if err != nil {
			return 0, err
		}
		l, err := layout.Of(t)
		if err != nil {
			return 0, err
		}
		emit(&b.typesGlobals, OpTypeArray, id, elem, length)
		emit(&b.annotations, OpDecorate, id, DecorationArrayStride, l.Stride)
	case types.RuntimeArray:
		elem, err := b.typeID(t.Elem)
		if err != nil {
			return 0, err
		}
		l, err := layout.Of(t)
		if err != nil {
			return 0, err
		}
		emit(&b.typesGlobals, OpTypeRuntimeArray, id, elem)
		emit(&b.annotations, OpDecorate, id, DecorationArrayStride, l.Stride)
	case types.Struct:
		members := []uint32{id}
		for _, f := range t.Fields {
			mid, err := b.typeID(f.Type)
			if err != nil {
				return 0, err
			}
			members = append(members, mid)
		}
		emit(&b.typesGlobals, OpTypeStruct, members...)
		l, err := layout.Of(t)
		if err != nil {
			return 0, err
		}
		for i, fl := range l.Fields {
			emit(&b.annotations, OpMemberDecorate, id, uint32(i), DecorationOffset, fl.Offset)
		}
	default:
		return 0, fmt.Errorf("unsupported SPIR-V type %s", t)
	}
	return id, nil
}

func (b *builder) pointerID(storage uint32, t *types.Type) (uint32, error) {
	pointee, err := b.typeID(t)
	if err != nil {
		return 0, err
	}
	key := fmt.Sprintf("%d:%d", storage, pointee)
	if id := b.pointers[key]; id != 0 {
		return id, nil
	}
	id := b.id()
	b.pointers[key] = id
	emit(&b.typesGlobals, OpTypePointer, id, storage, pointee)
	return id, nil
}

func (b *builder) functionTypeID(ret *types.Type, params []ir.Param) (uint32, error) {
	rid, err := b.typeID(ret)
	if err != nil {
		return 0, err
	}
	ids := make([]uint32, len(params))
	var sb strings.Builder
	fmt.Fprintf(&sb, "%d", rid)
	for i, p := range params {
		ids[i], err = b.typeID(p.Type)
		if err != nil {
			return 0, err
		}
		fmt.Fprintf(&sb, ":%d", ids[i])
	}
	key := sb.String()
	if id := b.fnTypes[key]; id != 0 {
		return id, nil
	}
	id := b.id()
	b.fnTypes[key] = id
	ops := []uint32{id, rid}
	ops = append(ops, ids...)
	emit(&b.typesGlobals, OpTypeFunction, ops...)
	return id, nil
}

func (b *builder) emitStructDebugAndLayouts() error {
	for _, t := range b.m.Structs {
		id, err := b.typeID(t)
		if err != nil {
			return err
		}
		emit(&b.debug, OpName, append([]uint32{id}, encodeString(t.Name)...)...)
		for i, f := range t.Fields {
			ops := []uint32{id, uint32(i)}
			ops = append(ops, encodeString(f.Name)...)
			emit(&b.debug, OpMemberName, ops...)
		}
	}
	return nil
}

func (b *builder) storageClass(r ir.Resource) uint32 {
	if r.Kind == ir.Uniform {
		return StorageUniform
	}
	return StorageStorageBuffer
}

func (b *builder) emitResources() error {
	b.resourceIDs = make([]uint32, len(b.m.Resources))
	for i, r := range b.m.Resources {
		logical, err := b.typeID(r.Type)
		if err != nil {
			return fmt.Errorf("resource %s type: %w", r.Name, err)
		}
		wrapper := b.id()
		emit(&b.typesGlobals, OpTypeStruct, wrapper, logical)
		emit(&b.annotations, OpDecorate, wrapper, DecorationBlock)
		emit(&b.annotations, OpMemberDecorate, wrapper, 0, DecorationOffset, 0)
		emit(&b.debug, OpName, append([]uint32{wrapper}, encodeString("__tach_resource_"+strconv.Itoa(i))...)...)
		emit(&b.debug, OpMemberName, append([]uint32{wrapper, 0}, encodeString("data")...)...)

		storage := b.storageClass(r)
		ptr := b.id()
		emit(&b.typesGlobals, OpTypePointer, ptr, storage, wrapper)
		varID := b.id()
		b.resourceIDs[i] = varID
		emit(&b.typesGlobals, OpVariable, ptr, varID, storage)
		emit(&b.annotations, OpDecorate, varID, DecorationDescriptorSet, r.Group)
		emit(&b.annotations, OpDecorate, varID, DecorationBinding, r.Binding)
		if r.Kind == ir.Storage && r.Access == ir.Read {
			emit(&b.annotations, OpDecorate, varID, DecorationNonWritable)
		}
		emit(&b.debug, OpName, append([]uint32{varID}, encodeString(r.Name)...)...)
	}
	return nil
}

func scanBuiltins(block *ir.Block, out map[ir.BuiltinKind]bool) {
	for _, in := range block.Instrs {
		switch x := in.(type) {
		case *ir.Builtin:
			out[x.Kind] = true
		case *ir.If:
			scanBuiltins(x.Then, out)
			scanBuiltins(x.Else, out)
		case *ir.Loop:
			scanBuiltins(x.Cond, out)
			scanBuiltins(x.Body, out)
		}
	}
}

func builtinInfo(k ir.BuiltinKind) (*types.Type, uint32, string) {
	vec3u := types.Vec(types.TU32, 3)
	switch k {
	case ir.GlobalID:
		return vec3u, BuiltInGlobalInvocationID, "globalId"
	case ir.LocalID:
		return vec3u, BuiltInLocalInvocationID, "localId"
	case ir.LocalIndex:
		return types.TU32, BuiltInLocalInvocationIndex, "localIndex"
	case ir.WorkgroupID:
		return vec3u, BuiltInWorkgroupID, "workgroupId"
	case ir.NumWorkgroups:
		return vec3u, BuiltInNumWorkgroups, "numWorkgroups"
	default:
		return nil, 0, ""
	}
}

func (b *builder) emitBuiltins() error {
	used := map[ir.BuiltinKind]bool{}
	for _, f := range b.m.Functions {
		if f.Compute {
			scanBuiltins(f.Body, used)
		}
	}
	order := []ir.BuiltinKind{ir.GlobalID, ir.LocalID, ir.LocalIndex, ir.WorkgroupID, ir.NumWorkgroups}
	for _, k := range order {
		if !used[k] {
			continue
		}
		t, decoration, name := builtinInfo(k)
		ptr, err := b.pointerID(StorageInput, t)
		if err != nil {
			return err
		}
		id := b.id()
		b.builtinIDs[k] = id
		emit(&b.typesGlobals, OpVariable, ptr, id, StorageInput)
		emit(&b.annotations, OpDecorate, id, DecorationBuiltIn, decoration)
		emit(&b.debug, OpName, append([]uint32{id}, encodeString("__tach_"+name)...)...)
	}
	return nil
}

func (b *builder) emitWorkgroups() error {
	for _, f := range b.m.Functions {
		if !f.Compute || len(f.WorkgroupVars) == 0 {
			continue
		}
		ids := make([]uint32, len(f.WorkgroupVars))
		for i, w := range f.WorkgroupVars {
			ptr, err := b.pointerID(StorageWorkgroup, w.Type)
			if err != nil {
				return fmt.Errorf("workgroup %s.%s type: %w", f.Name, w.Name, err)
			}
			id := b.id()
			ids[i] = id
			emit(&b.typesGlobals, OpVariable, ptr, id, StorageWorkgroup)
			emit(&b.debug, OpName, append([]uint32{id}, encodeString("__tach_w_"+f.Name+"_"+strconv.Itoa(i))...)...)
		}
		b.workgroupIDs[f.Name] = ids
	}
	return nil
}

func (b *builder) emitEntryPoints() {
	order := []ir.BuiltinKind{ir.GlobalID, ir.LocalID, ir.LocalIndex, ir.WorkgroupID, ir.NumWorkgroups}
	for _, f := range b.m.Functions {
		if !f.Compute {
			continue
		}
		used := map[ir.BuiltinKind]bool{}
		scanBuiltins(f.Body, used)
		ops := []uint32{ExecutionModelGLCompute, b.funcIDs[f.Name]}
		ops = append(ops, encodeString(abi.KernelEntry(f.Name))...)
		// SPIR-V 1.3 entry-point interfaces contain Input/Output variables only.
		for _, k := range order {
			if used[k] {
				ops = append(ops, b.builtinIDs[k])
			}
		}
		emit(&b.entryPoints, OpEntryPoint, ops...)
		emit(&b.execModes, OpExecutionMode, b.funcIDs[f.Name], ExecutionModeLocalSize, f.Workgroup[0], f.Workgroup[1], f.Workgroup[2])
	}
}

func (b *builder) constant(t *types.Type, raw string) (uint32, error) {
	tid, err := b.typeID(t)
	if err != nil {
		return 0, err
	}
	key := fmt.Sprintf("%d:%s", tid, raw)
	if id := b.constants[key]; id != 0 {
		return id, nil
	}
	id := b.id()
	switch t.Kind {
	case types.Bool:
		if raw == "true" {
			emit(&b.typesGlobals, OpConstantTrue, tid, id)
		} else if raw == "false" {
			emit(&b.typesGlobals, OpConstantFalse, tid, id)
		} else {
			return 0, fmt.Errorf("invalid bool constant %q", raw)
		}
	case types.I32:
		v, err := strconv.ParseInt(strings.TrimSuffix(raw, "i"), 10, 32)
		if err != nil {
			return 0, err
		}
		emit(&b.typesGlobals, OpConstant, tid, id, uint32(int32(v)))
	case types.U32:
		v, err := strconv.ParseUint(strings.TrimSuffix(raw, "u"), 10, 32)
		if err != nil {
			return 0, err
		}
		emit(&b.typesGlobals, OpConstant, tid, id, uint32(v))
	case types.F32:
		v, err := strconv.ParseFloat(strings.TrimSuffix(raw, "f"), 32)
		if err != nil {
			return 0, err
		}
		emit(&b.typesGlobals, OpConstant, tid, id, math.Float32bits(float32(v)))
	default:
		return 0, fmt.Errorf("constant type %s is not scalar", t)
	}
	b.constants[key] = id
	return id, nil
}

func (b *builder) u32Constant(v uint32) (uint32, error) {
	return b.constant(types.TU32, strconv.FormatUint(uint64(v), 10))
}

type spvPlace struct {
	ptr         uint32
	ty          *types.Type
	storage     uint32
	arrayBase   uint32 // pointer to struct containing a runtime array
	arrayMember uint32
	hasArrayLen bool
}

type fnEmitter struct {
	b *builder
	f *ir.Function

	values map[ir.ValueID]uint32
	vtypes map[ir.ValueID]*types.Type
	places map[ir.PlaceID]spvPlace

	currentLabel uint32
	terminated   bool
}

func (b *builder) emitFunction(f *ir.Function) error {
	retType, err := b.typeID(f.Return)
	if err != nil {
		return err
	}
	fnType, err := b.functionTypeID(f.Return, f.Params)
	if err != nil {
		return err
	}
	fid := b.funcIDs[f.Name]
	emit(&b.functions, OpFunction, retType, fid, FunctionControlNone, fnType)

	s := &fnEmitter{b: b, f: f, values: map[ir.ValueID]uint32{}, vtypes: map[ir.ValueID]*types.Type{}, places: map[ir.PlaceID]spvPlace{}}
	for _, p := range f.Params {
		pt, err := b.typeID(p.Type)
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
	if err := s.emitBlock(f.Body, blockNormal); err != nil {
		return err
	}
	if !s.terminated {
		return fmt.Errorf("function ended without a terminator")
	}
	emit(&b.functions, OpFunctionEnd)
	return nil
}

type blockMode uint8

const (
	blockNormal blockMode = iota
	blockYield
	blockContinue
)

type blockExit struct {
	kind  blockMode
	vals  []uint32
	pred  uint32
	falls bool
}

func (s *fnEmitter) emitBlock(bl *ir.Block, mode blockMode) error {
	_, err := s.emitBlockExit(bl, mode)
	return err
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
		if mode != blockContinue {
			return blockExit{}, fmt.Errorf("continue outside loop")
		}
		vals := make([]uint32, len(t.Values))
		for i, id := range t.Values {
			v, err := s.value(id)
			if err != nil {
				return blockExit{}, err
			}
			vals[i] = v
		}
		return blockExit{kind: blockContinue, vals: vals, pred: pred, falls: true}, nil
	default:
		return blockExit{}, fmt.Errorf("unknown block terminator %T", bl.Term)
	}
}

func (s *fnEmitter) value(id ir.ValueID) (uint32, error) {
	v := s.values[id]
	if v == 0 {
		return 0, fmt.Errorf("undefined IR value %%%d", id)
	}
	return v, nil
}

func (s *fnEmitter) place(id ir.PlaceID) (spvPlace, error) {
	p, ok := s.places[id]
	if !ok {
		return spvPlace{}, fmt.Errorf("undefined IR place &p%d", id)
	}
	return p, nil
}

func (s *fnEmitter) def(irID ir.ValueID, spvID uint32, t *types.Type) {
	if irID != 0 {
		s.values[irID] = spvID
		s.vtypes[irID] = t
	}
}

func (s *fnEmitter) emitInstr(in ir.Instr) error {
	switch x := in.(type) {
	case *ir.Const:
		id, err := s.b.constant(x.Type, x.Raw)
		if err != nil {
			return err
		}
		s.def(x.Result, id, x.Type)
	case *ir.Builtin:
		varID := s.b.builtinIDs[x.Kind]
		if varID == 0 {
			return fmt.Errorf("builtin %d was not declared", x.Kind)
		}
		tid, _ := s.b.typeID(x.Type)
		id := s.b.id()
		emit(&s.b.functions, OpLoad, tid, id, varID)
		s.def(x.Result, id, x.Type)
	case *ir.Unary:
		return s.emitUnary(x)
	case *ir.Binary:
		return s.emitBinary(x)
	case *ir.Intrinsic:
		return s.emitIntrinsic(x)
	case *ir.Convert:
		return s.emitConvert(x)
	case *ir.Composite:
		tid, _ := s.b.typeID(x.Type)
		id := s.b.id()
		ops := []uint32{tid, id}
		for _, v := range x.Values {
			sv, err := s.value(v)
			if err != nil {
				return err
			}
			ops = append(ops, sv)
		}
		emit(&s.b.functions, OpCompositeConstruct, ops...)
		s.def(x.Result, id, x.Type)
	case *ir.Extract:
		base, err := s.value(x.Base)
		if err != nil {
			return err
		}
		tid, _ := s.b.typeID(x.Type)
		id := s.b.id()
		emit(&s.b.functions, OpCompositeExtract, tid, id, base, uint32(x.Index))
		s.def(x.Result, id, x.Type)
	case *ir.Call:
		fid := s.b.funcIDs[x.Function]
		if fid == 0 {
			return fmt.Errorf("unknown callee %s", x.Function)
		}
		tid, _ := s.b.typeID(x.Type)
		result := s.b.id() // OpFunctionCall always carries a Result <id>, including void calls.
		ops := []uint32{tid, result, fid}
		for _, a := range x.Args {
			v, err := s.value(a)
			if err != nil {
				return err
			}
			ops = append(ops, v)
		}
		emit(&s.b.functions, OpFunctionCall, ops...)
		s.def(x.Result, result, x.Type)
	case *ir.PlaceRoot:
		return s.emitPlaceRoot(x)
	case *ir.PlaceWorkgroup:
		ids := s.b.workgroupIDs[s.f.Name]
		if x.Workgroup < 0 || x.Workgroup >= len(ids) {
			return fmt.Errorf("workgroup index %d out of bounds", x.Workgroup)
		}
		s.places[x.Result] = spvPlace{ptr: ids[x.Workgroup], ty: x.Type, storage: StorageWorkgroup}
	case *ir.PlaceField:
		return s.emitPlaceField(x)
	case *ir.PlaceIndex:
		return s.emitPlaceIndex(x)
	case *ir.Load:
		p, err := s.place(x.Place)
		if err != nil {
			return err
		}
		tid, _ := s.b.typeID(x.Type)
		id := s.b.id()
		emit(&s.b.functions, OpLoad, tid, id, p.ptr)
		s.def(x.Result, id, x.Type)
	case *ir.Store:
		p, err := s.place(x.Place)
		if err != nil {
			return err
		}
		v, err := s.value(x.Value)
		if err != nil {
			return err
		}
		emit(&s.b.functions, OpStore, p.ptr, v)
	case *ir.Atomic:
		return s.emitAtomic(x)
	case *ir.Barrier:
		return s.emitBarrier(x)
	case *ir.ArrayLength:
		p, err := s.place(x.Place)
		if err != nil {
			return err
		}
		if !p.hasArrayLen {
			return fmt.Errorf("runtime-array place lacks OpArrayLength base")
		}
		tid, _ := s.b.typeID(types.TU32)
		id := s.b.id()
		emit(&s.b.functions, OpArrayLength, tid, id, p.arrayBase, p.arrayMember)
		s.def(x.Result, id, types.TU32)
	case *ir.If:
		return s.emitIf(x)
	case *ir.Loop:
		return s.emitLoop(x)
	default:
		return fmt.Errorf("unsupported IR instruction %T", in)
	}
	return nil
}

func (s *fnEmitter) emitIntrinsic(x *ir.Intrinsic) error {
	tid, err := s.b.typeID(x.Type)
	if err != nil {
		return err
	}
	args := make([]uint32, len(x.Args))
	for i, a := range x.Args {
		v, err := s.value(a)
		if err != nil {
			return err
		}
		args[i] = v
	}
	id := s.b.id()
	if x.Kind == ir.IntrinsicDot {
		emit(&s.b.functions, OpDot, tid, id, args[0], args[1])
		s.def(x.Result, id, x.Type)
		return nil
	}

	var inst uint32
	switch x.Kind {
	case ir.IntrinsicAbs:
		if scalarKind(x.Type) == types.F32 {
			inst = GLSL450FAbs
		} else {
			inst = GLSL450SAbs
		}
	case ir.IntrinsicFloor:
		inst = GLSL450Floor
	case ir.IntrinsicCeil:
		inst = GLSL450Ceil
	case ir.IntrinsicTrunc:
		inst = GLSL450Trunc
	case ir.IntrinsicSin:
		inst = GLSL450Sin
	case ir.IntrinsicCos:
		inst = GLSL450Cos
	case ir.IntrinsicTan:
		inst = GLSL450Tan
	case ir.IntrinsicExp:
		inst = GLSL450Exp
	case ir.IntrinsicExp2:
		inst = GLSL450Exp2
	case ir.IntrinsicLog:
		inst = GLSL450Log
	case ir.IntrinsicLog2:
		inst = GLSL450Log2
	case ir.IntrinsicSqrt:
		inst = GLSL450Sqrt
	case ir.IntrinsicInverseSqrt:
		inst = GLSL450InverseSqrt
	case ir.IntrinsicPow:
		inst = GLSL450Pow
	case ir.IntrinsicMin:
		if scalarKind(x.Type) == types.I32 {
			inst = GLSL450SMin
		} else {
			inst = GLSL450UMin
		}
	case ir.IntrinsicMax:
		if scalarKind(x.Type) == types.I32 {
			inst = GLSL450SMax
		} else {
			inst = GLSL450UMax
		}
	case ir.IntrinsicClamp:
		if scalarKind(x.Type) == types.I32 {
			inst = GLSL450SClamp
		} else {
			inst = GLSL450UClamp
		}
	case ir.IntrinsicLength:
		inst = GLSL450Length
	case ir.IntrinsicDistance:
		inst = GLSL450Distance
	case ir.IntrinsicCross:
		inst = GLSL450Cross
	case ir.IntrinsicNormalize:
		inst = GLSL450Normalize
	default:
		return fmt.Errorf("unsupported intrinsic %s", x.Kind)
	}
	ops := []uint32{tid, id, s.b.ensureGLSL450(), inst}
	ops = append(ops, args...)
	emit(&s.b.functions, OpExtInst, ops...)
	s.def(x.Result, id, x.Type)
	return nil
}

func (s *fnEmitter) emitAtomic(x *ir.Atomic) error {
	p, err := s.place(x.Place)
	if err != nil {
		return err
	}
	if p.storage != StorageWorkgroup && p.storage != StorageStorageBuffer {
		return fmt.Errorf("atomic place uses invalid storage class %d", p.storage)
	}
	scopeValue := ScopeDevice
	if p.storage == StorageWorkgroup {
		scopeValue = ScopeWorkgroup
	}
	scope, err := s.b.u32Constant(scopeValue)
	if err != nil {
		return err
	}
	semantics, err := s.b.u32Constant(MemorySemanticsRelaxed)
	if err != nil {
		return err
	}
	tid, err := s.b.typeID(x.Type)
	if err != nil {
		return err
	}

	switch x.Op {
	case ir.AtomicLoad:
		id := s.b.id()
		emit(&s.b.functions, OpAtomicLoad, tid, id, p.ptr, scope, semantics)
		s.def(x.Result, id, x.Type)
		return nil
	case ir.AtomicStore:
		value, err := s.value(x.Value)
		if err != nil {
			return err
		}
		emit(&s.b.functions, OpAtomicStore, p.ptr, scope, semantics, value)
		return nil
	}

	value, err := s.value(x.Value)
	if err != nil {
		return err
	}
	var op Op
	switch x.Op {
	case ir.AtomicExchange:
		op = OpAtomicExchange
	case ir.AtomicAdd:
		op = OpAtomicIAdd
	case ir.AtomicSub:
		op = OpAtomicISub
	case ir.AtomicMin:
		if x.Type.Kind == types.I32 {
			op = OpAtomicSMin
		} else {
			op = OpAtomicUMin
		}
	case ir.AtomicMax:
		if x.Type.Kind == types.I32 {
			op = OpAtomicSMax
		} else {
			op = OpAtomicUMax
		}
	case ir.AtomicAnd:
		op = OpAtomicAnd
	case ir.AtomicOr:
		op = OpAtomicOr
	case ir.AtomicXor:
		op = OpAtomicXor
	default:
		return fmt.Errorf("unsupported atomic operation %d", x.Op)
	}
	id := s.b.id()
	emit(&s.b.functions, op, tid, id, p.ptr, scope, semantics, value)
	s.def(x.Result, id, x.Type)
	return nil
}

func (s *fnEmitter) emitBarrier(x *ir.Barrier) error {
	execScope, err := s.b.u32Constant(ScopeWorkgroup)
	if err != nil {
		return err
	}
	memoryScope := execScope
	sem := MemorySemanticsAcquireRelease
	switch x.Kind {
	case ir.BarrierWorkgroup:
		sem |= MemorySemanticsWorkgroupMemory
	case ir.BarrierStorage:
		sem |= MemorySemanticsUniformMemory
	default:
		return fmt.Errorf("unsupported barrier kind %d", x.Kind)
	}
	semantics, err := s.b.u32Constant(sem)
	if err != nil {
		return err
	}
	emit(&s.b.functions, OpControlBarrier, execScope, memoryScope, semantics)
	return nil
}

func (s *fnEmitter) emitPlaceRoot(x *ir.PlaceRoot) error {
	if x.Resource < 0 || x.Resource >= len(s.b.m.Resources) {
		return fmt.Errorf("resource index %d out of bounds", x.Resource)
	}
	r := s.b.m.Resources[x.Resource]
	storage := s.b.storageClass(r)
	ptrType, err := s.b.pointerID(storage, x.Type)
	if err != nil {
		return err
	}
	zero, err := s.b.u32Constant(0)
	if err != nil {
		return err
	}
	id := s.b.id()
	emit(&s.b.functions, OpAccessChain, ptrType, id, s.b.resourceIDs[x.Resource], zero)
	p := spvPlace{ptr: id, ty: x.Type, storage: storage}
	if x.Type.Kind == types.RuntimeArray {
		p.arrayBase = s.b.resourceIDs[x.Resource]
		p.arrayMember = 0
		p.hasArrayLen = true
	}
	s.places[x.Result] = p
	return nil
}

func (s *fnEmitter) emitPlaceField(x *ir.PlaceField) error {
	base, err := s.place(x.Base)
	if err != nil {
		return err
	}
	ptrType, err := s.b.pointerID(base.storage, x.Type)
	if err != nil {
		return err
	}
	idx, err := s.b.u32Constant(uint32(x.Field))
	if err != nil {
		return err
	}
	id := s.b.id()
	emit(&s.b.functions, OpAccessChain, ptrType, id, base.ptr, idx)
	p := spvPlace{ptr: id, ty: x.Type, storage: base.storage}
	if x.Type.Kind == types.RuntimeArray {
		p.arrayBase = base.ptr
		p.arrayMember = uint32(x.Field)
		p.hasArrayLen = true
	}
	s.places[x.Result] = p
	return nil
}

func (s *fnEmitter) emitPlaceIndex(x *ir.PlaceIndex) error {
	base, err := s.place(x.Base)
	if err != nil {
		return err
	}
	idx, err := s.value(x.Index)
	if err != nil {
		return err
	}
	ptrType, err := s.b.pointerID(base.storage, x.Type)
	if err != nil {
		return err
	}
	id := s.b.id()
	emit(&s.b.functions, OpAccessChain, ptrType, id, base.ptr, idx)
	s.places[x.Result] = spvPlace{ptr: id, ty: x.Type, storage: base.storage}
	return nil
}

func scalarKind(t *types.Type) types.Kind {
	if t != nil && t.Kind == types.Vector {
		return t.Elem.Kind
	}
	if t == nil {
		return types.Invalid
	}
	return t.Kind
}

func (s *fnEmitter) emitUnary(x *ir.Unary) error {
	v, err := s.value(x.X)
	if err != nil {
		return err
	}
	tid, _ := s.b.typeID(x.Type)
	id := s.b.id()
	var op Op
	switch x.Op {
	case "!":
		op = OpLogicalNot
	case "-":
		if scalarKind(x.Type) == types.F32 {
			op = OpFNegate
		} else {
			op = OpSNegate
		}
	case "~":
		op = OpNot
	default:
		return fmt.Errorf("unsupported unary operator %s", x.Op)
	}
	emit(&s.b.functions, op, tid, id, v)
	s.def(x.Result, id, x.Type)
	return nil
}

func (s *fnEmitter) splatVector(vector *types.Type, scalar uint32) (uint32, error) {
	tid, err := s.b.typeID(vector)
	if err != nil {
		return 0, err
	}
	id := s.b.id()
	ops := []uint32{tid, id}
	for i := 0; i < vector.Lanes; i++ {
		ops = append(ops, scalar)
	}
	emit(&s.b.functions, OpCompositeConstruct, ops...)
	return id, nil
}

func (s *fnEmitter) emitBinary(x *ir.Binary) error {
	lv, err := s.value(x.Left)
	if err != nil {
		return err
	}
	rv, err := s.value(x.Right)
	if err != nil {
		return err
	}
	lt := s.vtypes[x.Left]
	rt := s.vtypes[x.Right]
	tid, _ := s.b.typeID(x.Type)
	id := s.b.id()
	kind := scalarKind(lt)
	var op Op

	isVectorScalar := lt != nil && lt.Kind == types.Vector && rt != nil && types.Equal(lt.Elem, rt)
	isScalarVector := rt != nil && rt.Kind == types.Vector && lt != nil && types.Equal(rt.Elem, lt)
	if x.Op == "*" && (isVectorScalar || isScalarVector) {
		if isScalarVector {
			lv, rv = rv, lv
		}
		emit(&s.b.functions, OpVectorTimesScalar, tid, id, lv, rv)
		s.def(x.Result, id, x.Type)
		return nil
	}
	if x.Op == "/" && isVectorScalar {
		rv, err = s.splatVector(lt, rv)
		if err != nil {
			return err
		}
	}

	switch x.Op {
	case "+":
		if kind == types.F32 {
			op = OpFAdd
		} else {
			op = OpIAdd
		}
	case "-":
		if kind == types.F32 {
			op = OpFSub
		} else {
			op = OpISub
		}
	case "*":
		if kind == types.F32 {
			op = OpFMul
		} else {
			op = OpIMul
		}
	case "/":
		switch kind {
		case types.F32:
			op = OpFDiv
		case types.U32:
			op = OpUDiv
		case types.I32:
			op = OpSDiv
		}
	case "%":
		switch kind {
		case types.F32:
			op = OpFRem
		case types.U32:
			op = OpUMod
		case types.I32:
			op = OpSRem
		}
	case "&&":
		op = OpLogicalAnd
	case "||":
		op = OpLogicalOr
	case "&":
		op = OpBitwiseAnd
	case "|":
		op = OpBitwiseOr
	case "^":
		op = OpBitwiseXor
	case "<<":
		op = OpShiftLeftLogical
	case ">>":
		if kind == types.I32 {
			op = OpShiftRightArithmetic
		} else {
			op = OpShiftRightLogical
		}
	case "==":
		switch kind {
		case types.F32:
			op = OpFOrdEqual
		case types.I32, types.U32:
			op = OpIEqual
		case types.Bool:
			op = OpLogicalEqual
		}
	case "!=":
		switch kind {
		case types.F32:
			op = OpFOrdNotEqual
		case types.I32, types.U32:
			op = OpINotEqual
		case types.Bool:
			op = OpLogicalNotEqual
		}
	case "<":
		switch kind {
		case types.F32:
			op = OpFOrdLessThan
		case types.U32:
			op = OpULessThan
		case types.I32:
			op = OpSLessThan
		}
	case "<=":
		switch kind {
		case types.F32:
			op = OpFOrdLessThanEqual
		case types.U32:
			op = OpULessThanEqual
		case types.I32:
			op = OpSLessThanEqual
		}
	case ">":
		switch kind {
		case types.F32:
			op = OpFOrdGreaterThan
		case types.U32:
			op = OpUGreaterThan
		case types.I32:
			op = OpSGreaterThan
		}
	case ">=":
		switch kind {
		case types.F32:
			op = OpFOrdGreaterThanEqual
		case types.U32:
			op = OpUGreaterThanEqual
		case types.I32:
			op = OpSGreaterThanEqual
		}
	}
	if op == 0 {
		return fmt.Errorf("cannot lower binary %s for %s and %s", x.Op, lt, rt)
	}
	emit(&s.b.functions, op, tid, id, lv, rv)
	s.def(x.Result, id, x.Type)
	return nil
}

func (s *fnEmitter) emitConvert(x *ir.Convert) error {
	v, err := s.value(x.X)
	if err != nil {
		return err
	}
	tid, _ := s.b.typeID(x.Type)
	id := s.b.id()
	var op Op
	switch {
	case x.From.Kind == types.F32 && x.Type.Kind == types.U32:
		op = OpConvertFToU
	case x.From.Kind == types.F32 && x.Type.Kind == types.I32:
		op = OpConvertFToS
	case x.From.Kind == types.I32 && x.Type.Kind == types.F32:
		op = OpConvertSToF
	case x.From.Kind == types.U32 && x.Type.Kind == types.F32:
		op = OpConvertUToF
	case (x.From.Kind == types.I32 && x.Type.Kind == types.U32) || (x.From.Kind == types.U32 && x.Type.Kind == types.I32):
		op = OpBitcast
	default:
		return fmt.Errorf("unsupported conversion %s -> %s", x.From, x.Type)
	}
	emit(&s.b.functions, op, tid, id, v)
	s.def(x.Result, id, x.Type)
	return nil
}

func (s *fnEmitter) emitIf(x *ir.If) error {
	cond, err := s.value(x.Cond)
	if err != nil {
		return err
	}
	thenLabel, elseLabel, mergeLabel := s.b.id(), s.b.id(), s.b.id()
	emit(&s.b.functions, OpSelectionMerge, mergeLabel, SelectionControlNone)
	emit(&s.b.functions, OpBranchConditional, cond, thenLabel, elseLabel)
	s.terminated = true

	type incoming struct{ val, label uint32 }
	incomingByResult := make([][]incoming, len(x.Results))

	emit(&s.b.functions, OpLabel, thenLabel)
	s.currentLabel, s.terminated = thenLabel, false
	te, err := s.emitBlockExit(x.Then, blockYield)
	if err != nil {
		return err
	}
	if te.falls {
		if len(te.vals) != len(x.Results) {
			return fmt.Errorf("then yield count mismatch")
		}
		for i, v := range te.vals {
			incomingByResult[i] = append(incomingByResult[i], incoming{v, te.pred})
		}
		emit(&s.b.functions, OpBranch, mergeLabel)
		s.terminated = true
	}

	emit(&s.b.functions, OpLabel, elseLabel)
	s.currentLabel, s.terminated = elseLabel, false
	ee, err := s.emitBlockExit(x.Else, blockYield)
	if err != nil {
		return err
	}
	if ee.falls {
		if len(ee.vals) != len(x.Results) {
			return fmt.Errorf("else yield count mismatch")
		}
		for i, v := range ee.vals {
			incomingByResult[i] = append(incomingByResult[i], incoming{v, ee.pred})
		}
		emit(&s.b.functions, OpBranch, mergeLabel)
		s.terminated = true
	}

	emit(&s.b.functions, OpLabel, mergeLabel)
	s.currentLabel, s.terminated = mergeLabel, false
	for i, r := range x.Results {
		incs := incomingByResult[i]
		if len(incs) == 0 {
			return fmt.Errorf("selection result %%%d has no incoming value", r.ID)
		}
		if len(incs) == 1 {
			s.def(r.ID, incs[0].val, r.Type)
			continue
		}
		tid, _ := s.b.typeID(r.Type)
		id := s.b.id()
		ops := []uint32{tid, id}
		for _, in := range incs {
			ops = append(ops, in.val, in.label)
		}
		emit(&s.b.functions, OpPhi, ops...)
		s.def(r.ID, id, r.Type)
	}
	return nil
}

func (s *fnEmitter) emitLoop(x *ir.Loop) error {
	if len(x.Params) != len(x.Results) {
		return fmt.Errorf("loop param/result mismatch")
	}
	preheader := s.currentLabel
	header, condEntry, body, cont, merge := s.b.id(), s.b.id(), s.b.id(), s.b.id(), s.b.id()
	emit(&s.b.functions, OpBranch, header)
	s.terminated = true
	emit(&s.b.functions, OpLabel, header)
	s.currentLabel, s.terminated = header, false

	patches := make([]int, len(x.Params))
	for i, p := range x.Params {
		init, err := s.value(p.Init)
		if err != nil {
			return err
		}
		tid, _ := s.b.typeID(p.Type)
		phi := s.b.id()
		start := emit(&s.b.functions, OpPhi, tid, phi, init, preheader, 0, cont)
		patches[i] = start + 5 // first word + operands: type,result,init,pre,back,cont
		s.def(p.ID, phi, p.Type)
		s.def(x.Results[i].ID, phi, x.Results[i].Type)
	}

	// Keep the SPIR-V loop header minimal and structurally stable. Tach's loop
	// condition is a region and can itself contain structured short-circuit
	// selections, so it lives in the loop construct after the header.
	emit(&s.b.functions, OpLoopMerge, merge, cont, LoopControlNone)
	emit(&s.b.functions, OpBranch, condEntry)
	s.terminated = true

	emit(&s.b.functions, OpLabel, condEntry)
	s.currentLabel, s.terminated = condEntry, false
	ce, err := s.emitBlockExit(x.Cond, blockYield)
	if err != nil {
		return err
	}
	if !ce.falls || len(ce.vals) != 1 {
		return fmt.Errorf("loop condition must yield one bool")
	}
	emit(&s.b.functions, OpBranchConditional, ce.vals[0], body, merge)
	s.terminated = true

	emit(&s.b.functions, OpLabel, body)
	s.currentLabel, s.terminated = body, false
	be, err := s.emitBlockExit(x.Body, blockContinue)
	if err != nil {
		return err
	}
	if !be.falls || len(be.vals) != len(x.Params) {
		return fmt.Errorf("loop body must continue with %d carried values", len(x.Params))
	}
	emit(&s.b.functions, OpBranch, cont)
	s.terminated = true

	emit(&s.b.functions, OpLabel, cont)
	s.currentLabel, s.terminated = cont, false
	for i, back := range be.vals {
		s.b.functions[patches[i]] = back
	}
	emit(&s.b.functions, OpBranch, header)
	s.terminated = true

	emit(&s.b.functions, OpLabel, merge)
	s.currentLabel, s.terminated = merge, false
	return nil
}
