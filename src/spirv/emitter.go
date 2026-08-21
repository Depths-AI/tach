package spirv

import (
	"encoding/binary"
	"fmt"
	"math"
	"sort"
	"strconv"
	"strings"

	"tach/src/abi"
	"tach/src/backend"
	"tach/src/flow"
	"tach/src/ir"
	"tach/src/layout"
	"tach/src/types"
)

type inputKind uint8

const (
	inputGlobalIndex inputKind = iota + 1
	inputLocalIndex
	inputLocalLinear
)

type program struct {
	executable *backend.Executable
	source     *ir.Module
	functions  map[*ir.Function]*backend.Coordinates
	kernels    map[*ir.Function]*backend.PhysicalKernel
}

func Lower(logical *flow.Module) (*backend.Executable, error) {
	return backend.Lower(logical, backend.SPIRVProfile)
}

func lower(executable *backend.Executable) (*program, error) {
	functions, kernels, err := executable.IndexFunctions()
	if err != nil {
		return nil, err
	}
	return &program{executable: executable, source: executable.KernelModule, functions: functions, kernels: kernels}, nil
}

func inputs(_ *ir.Function, coordinates *backend.Coordinates) map[inputKind]bool {
	used := map[inputKind]bool{}
	for id, coordinate := range coordinates.Values {
		if coordinates.Uses[id] == 0 {
			continue
		}
		switch coordinate.Space {
		case backend.Global:
			used[inputGlobalIndex] = true
		case backend.Local:
			used[inputLocalIndex] = true
		case backend.LocalLinear:
			used[inputLocalLinear] = true
		}
	}
	return used
}

func coordinate(f *backend.Coordinates, id ir.ValueID) (inputKind, uint32) {
	coordinate := f.Values[id]
	switch coordinate.Space {
	case backend.Global:
		return inputGlobalIndex, uint32(coordinate.Dimension)
	case backend.Local:
		return inputLocalIndex, uint32(coordinate.Dimension)
	case backend.LocalLinear:
		return inputLocalLinear, 0
	}
	panic("unknown lowered coordinate space")
}

// Emit lowers verified Tach IR to a SPIR-V 1.6 compute module and immediately
// parses and validates the produced binary with Tach's own SPIR-V validator.
func Emit(executable *backend.Executable) ([]byte, error) {
	if err := backend.Verify(executable); err != nil {
		return nil, fmt.Errorf("executable verification: %w", err)
	}
	p, err := lower(executable)
	if err != nil {
		return nil, err
	}
	b := newBuilder(p)
	if err := b.build(); err != nil {
		return nil, err
	}
	words := b.words()
	out := make([]byte, 4*len(words))
	for i, w := range words {
		binary.LittleEndian.PutUint32(out[i*4:], w)
	}
	if err := Validate(out); err != nil {
		return nil, fmt.Errorf("tach SPIR-V self-validation failed: %w", err)
	}
	return out, nil
}

type builder struct {
	p *program
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

	types           map[string]uint32
	pointers        map[string]uint32
	fnTypes         map[string]uint32
	constants       map[string]uint32
	funcIDs         map[string]uint32
	resourceIDs     map[*ir.Function][]uint32
	parameterBlocks map[*ir.Function]*abi.ParameterBlock
	parameterIDs    map[*ir.Function]uint32
	inputIDs        map[inputKind]uint32
	workgroupIDs    map[string][]uint32
	globalUses      map[string]map[uint32]bool
	calls           map[string]map[string]bool
	glsl450         uint32
}

type typeRole uint8

const (
	typeLogical typeRole = iota
	typeHostABI
)

func newBuilder(p *program) *builder {
	return &builder{
		p:               p,
		m:               p.source,
		nextID:          1,
		types:           map[string]uint32{},
		pointers:        map[string]uint32{},
		fnTypes:         map[string]uint32{},
		constants:       map[string]uint32{},
		funcIDs:         map[string]uint32{},
		resourceIDs:     map[*ir.Function][]uint32{},
		parameterBlocks: map[*ir.Function]*abi.ParameterBlock{},
		parameterIDs:    map[*ir.Function]uint32{},
		inputIDs:        map[inputKind]uint32{},
		workgroupIDs:    map[string][]uint32{},
		globalUses:      map[string]map[uint32]bool{},
		calls:           map[string]map[string]bool{},
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
	if ir.UsesKind(b.m, types.F16) {
		emit(&b.capabilities, OpCapability, CapabilityFloat16)
		features := backend.RequiredFeatures(b.p.executable)
		for _, feature := range features {
			switch feature {
			case backend.StorageBuffer16BitAccess:
				emit(&b.capabilities, OpCapability, CapabilityStorageBuffer16BitAccess)
			case backend.UniformAndStorage16BitAccess:
				emit(&b.capabilities, OpCapability, CapabilityUniformAndStorage16BitAccess)
			}
		}
	}
	emit(&b.capabilities, OpCapability, CapabilityVulkanMemoryModel)
	emit(&b.memoryModel, OpMemoryModel, AddressingLogical, MemoryVulkan)

	// Function IDs must exist before entry-point declarations and forward calls.
	for _, f := range b.m.Functions {
		b.funcIDs[f.Name] = b.id()
		emit(&b.debug, OpName, append([]uint32{b.funcIDs[f.Name]}, encodeString(f.Name)...)...)
	}

	if err := b.emitStructDebugTypes(); err != nil {
		return err
	}
	if err := b.emitResources(); err != nil {
		return err
	}
	if err := b.emitParameterBlocks(); err != nil {
		return err
	}
	if err := b.emitInputs(); err != nil {
		return err
	}
	if err := b.emitWorkgroups(); err != nil {
		return err
	}
	for _, f := range b.m.Functions {
		if err := b.emitFunction(f); err != nil {
			return fmt.Errorf("SPIR-V lower %s: %w", f.Name, err)
		}
	}
	return b.emitEntryPoints()
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

func normalizedTypeRole(t *types.Type, role typeRole) typeRole {
	if role == typeHostABI && t != nil {
		switch t.Kind {
		case types.FixedArray, types.RuntimeArray, types.Struct:
			return typeHostABI
		}
	}
	return typeLogical
}

// typeID is the single SPIR-V type-lowering path. Logical types are used by
// SSA values and Workgroup memory. Host-ABI types exist only behind Uniform or
// StorageBuffer pointers and carry Tach's compiler-owned layout decorations.
func (b *builder) typeID(t *types.Type, role typeRole) (uint32, error) {
	role = normalizedTypeRole(t, role)
	key := fmt.Sprintf("%d:%s", role, typeKey(t))
	if id := b.types[key]; id != 0 {
		return id, nil
	}
	if t.Kind == types.Atomic {
		// SPIR-V atomicity is an operation property; the pointed-to object keeps
		// the underlying integer type. Resolve it before allocating an ID so the
		// module ID space remains dense and deterministic.
		elem, err := b.typeID(t.Elem, typeLogical)
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
	case types.F16:
		emit(&b.typesGlobals, OpTypeFloat, id, 16)
	case types.F32:
		emit(&b.typesGlobals, OpTypeFloat, id, 32)
	case types.Vector:
		elem, err := b.typeID(t.Elem, typeLogical)
		if err != nil {
			return 0, err
		}
		emit(&b.typesGlobals, OpTypeVector, id, elem, uint32(t.Lanes))
	case types.FixedArray:
		elem, err := b.typeID(t.Elem, role)
		if err != nil {
			return 0, err
		}
		length, err := b.u32Constant(t.Count)
		if err != nil {
			return 0, err
		}
		emit(&b.typesGlobals, OpTypeArray, id, elem, length)
		if role == typeHostABI {
			l, err := layout.Of(t)
			if err != nil {
				return 0, err
			}
			emit(&b.annotations, OpDecorate, id, DecorationArrayStride, l.Stride)
		}
	case types.RuntimeArray:
		elem, err := b.typeID(t.Elem, role)
		if err != nil {
			return 0, err
		}
		emit(&b.typesGlobals, OpTypeRuntimeArray, id, elem)
		if role == typeHostABI {
			l, err := layout.Of(t)
			if err != nil {
				return 0, err
			}
			emit(&b.annotations, OpDecorate, id, DecorationArrayStride, l.Stride)
		}
	case types.Struct:
		members := []uint32{id}
		for _, f := range t.Fields {
			mid, err := b.typeID(f.Type, role)
			if err != nil {
				return 0, err
			}
			members = append(members, mid)
		}
		emit(&b.typesGlobals, OpTypeStruct, members...)
		if role == typeHostABI {
			l, err := layout.Of(t)
			if err != nil {
				return 0, err
			}
			for i, fl := range l.Fields {
				emit(&b.annotations, OpMemberDecorate, id, uint32(i), DecorationOffset, fl.Offset)
			}
		}
	default:
		return 0, fmt.Errorf("unsupported SPIR-V type %s", t)
	}
	return id, nil
}

func typeRoleForStorage(storage uint32) typeRole {
	if storage == StorageUniform || storage == StorageStorageBuffer {
		return typeHostABI
	}
	return typeLogical
}

func (b *builder) pointerID(storage uint32, t *types.Type) (uint32, error) {
	pointee, err := b.typeID(t, typeRoleForStorage(storage))
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
	rid, err := b.typeID(ret, typeLogical)
	if err != nil {
		return 0, err
	}
	ids := make([]uint32, len(params))
	var sb strings.Builder
	fmt.Fprintf(&sb, "%d", rid)
	for i, p := range params {
		ids[i], err = b.typeID(p.Type, typeLogical)
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

func (b *builder) emitStructDebugTypes() error {
	for _, t := range b.m.Structs {
		if types.ContainsRuntimeArray(t) {
			continue
		}
		id, err := b.typeID(t, typeLogical)
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

func (b *builder) emitResources() error {
	for kernelIndex := range b.p.executable.PhysicalKernels {
		kernel := &b.p.executable.PhysicalKernels[kernelIndex]
		b.resourceIDs[kernel.Function] = make([]uint32, len(kernel.Function.BufferParams))
		for i, r := range kernel.Function.BufferParams {
			physical, err := b.typeID(r.Type, typeHostABI)
			if err != nil {
				return fmt.Errorf("resource %s type: %w", r.Name, err)
			}
			root := physical
			if r.Type.Kind == types.Struct && types.ContainsRuntimeArray(r.Type) {
				emit(&b.annotations, OpDecorate, root, DecorationBlock)
			} else {
				root = b.id()
				emit(&b.typesGlobals, OpTypeStruct, root, physical)
				emit(&b.annotations, OpDecorate, root, DecorationBlock)
				emit(&b.annotations, OpMemberDecorate, root, 0, DecorationOffset, 0)
				emit(&b.debug, OpName, append([]uint32{root}, encodeString(fmt.Sprintf("__tach_resource_%d_%d", kernelIndex, i))...)...)
				emit(&b.debug, OpMemberName, append([]uint32{root, 0}, encodeString("data")...)...)
			}

			storage := uint32(StorageStorageBuffer)
			ptr := b.id()
			emit(&b.typesGlobals, OpTypePointer, ptr, storage, root)
			varID := b.id()
			b.resourceIDs[kernel.Function][i] = varID
			emit(&b.typesGlobals, OpVariable, ptr, varID, storage)
			emit(&b.annotations, OpDecorate, varID, DecorationDescriptorSet, 0)
			emit(&b.annotations, OpDecorate, varID, DecorationBinding, uint32(i))
			if r.Access == ir.Read && !types.ContainsAtomic(r.Type) {
				emit(&b.annotations, OpDecorate, varID, DecorationNonWritable)
			}
			emit(&b.debug, OpName, append([]uint32{varID}, encodeString(r.Name)...)...)
		}
	}
	return nil
}

func (b *builder) emitParameterBlocks() error {
	for i := range b.p.executable.PhysicalKernels {
		block := b.p.executable.PhysicalKernels[i].Parameters
		if block == nil {
			continue
		}
		b.parameterBlocks[block.Function] = block
		typeID, err := b.typeID(block.Type, typeHostABI)
		if err != nil {
			return fmt.Errorf("kernel %s parameter block type: %w", block.Function.Name, err)
		}
		emit(&b.annotations, OpDecorate, typeID, DecorationBlock)
		emit(&b.debug, OpName, append([]uint32{typeID}, encodeString(block.Type.Name)...)...)
		for index, field := range block.Fields {
			emit(&b.debug, OpMemberName, append([]uint32{typeID, uint32(index)}, encodeString(field.Name)...)...)
		}
		pointer, err := b.pointerID(StorageUniform, block.Type)
		if err != nil {
			return err
		}
		variable := b.id()
		b.parameterIDs[block.Function] = variable
		emit(&b.typesGlobals, OpVariable, pointer, variable, StorageUniform)
		emit(&b.annotations, OpDecorate, variable, DecorationDescriptorSet, 0)
		emit(&b.annotations, OpDecorate, variable, DecorationBinding, block.Binding)
		emit(&b.debug, OpName, append([]uint32{variable}, encodeString("__tach_parameters_"+block.Function.Name)...)...)
	}
	return nil
}

func inputInfo(k inputKind) (*types.Type, uint32, string) {
	vec3u := types.Vec(types.TU32, 3)
	switch k {
	case inputGlobalIndex:
		return vec3u, BuiltInGlobalInvocationID, "globalIndex"
	case inputLocalIndex:
		return vec3u, BuiltInLocalInvocationID, "localIndex"
	case inputLocalLinear:
		return types.TU32, BuiltInLocalInvocationIndex, "localLinear"
	default:
		return nil, 0, ""
	}
}

func (b *builder) emitInputs() error {
	used := map[inputKind]bool{}
	for _, f := range b.m.Functions {
		for k := range inputs(f, b.p.functions[f]) {
			used[k] = true
		}
	}
	order := []inputKind{inputGlobalIndex, inputLocalIndex, inputLocalLinear}
	for _, k := range order {
		if !used[k] {
			continue
		}
		t, decoration, name := inputInfo(k)
		ptr, err := b.pointerID(StorageInput, t)
		if err != nil {
			return err
		}
		id := b.id()
		b.inputIDs[k] = id
		emit(&b.typesGlobals, OpVariable, ptr, id, StorageInput)
		emit(&b.annotations, OpDecorate, id, DecorationBuiltIn, decoration)
		emit(&b.debug, OpName, append([]uint32{id}, encodeString("__tach_"+name)...)...)
	}
	return nil
}

func (b *builder) emitWorkgroups() error {
	for _, f := range b.m.Functions {
		if f.Kind != ir.Stage || len(f.WorkgroupVars) == 0 {
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
			zero, err := b.nullConstant(w.Type)
			if err != nil {
				return fmt.Errorf("workgroup %s.%s initializer: %w", f.Name, w.Name, err)
			}
			emit(&b.typesGlobals, OpVariable, ptr, id, StorageWorkgroup, zero)
			emit(&b.debug, OpName, append([]uint32{id}, encodeString("__tach_w_"+f.Name+"_"+strconv.Itoa(i))...)...)
		}
		b.workgroupIDs[f.Name] = ids
	}
	return nil
}

func (b *builder) emitEntryPoints() error {
	for _, f := range b.m.Functions {
		if f.Kind != ir.Stage {
			continue
		}
		ops := []uint32{ExecutionModelGLCompute, b.funcIDs[f.Name]}
		ops = append(ops, encodeString(f.Name)...)
		globals, err := b.entryGlobals(f.Name, map[string]bool{}, map[string]bool{})
		if err != nil {
			return err
		}
		ops = append(ops, globals...)
		emit(&b.entryPoints, OpEntryPoint, ops...)
		workgroup := b.p.kernels[f].Workgroup
		emit(&b.execModes, OpExecutionMode, b.funcIDs[f.Name], ExecutionModeLocalSize, workgroup[0], workgroup[1], workgroup[2])
	}
	return nil
}

func (b *builder) entryGlobals(name string, visiting, visited map[string]bool) ([]uint32, error) {
	used := map[uint32]bool{}
	var walk func(string) error
	walk = func(function string) error {
		if visiting[function] {
			return fmt.Errorf("recursive static call graph at %s", function)
		}
		if visited[function] {
			return nil
		}
		if b.funcIDs[function] == 0 {
			return fmt.Errorf("static call graph references unknown function %s", function)
		}
		visiting[function] = true
		for id := range b.globalUses[function] {
			used[id] = true
		}
		for callee := range b.calls[function] {
			if err := walk(callee); err != nil {
				return err
			}
		}
		delete(visiting, function)
		visited[function] = true
		return nil
	}
	if err := walk(name); err != nil {
		return nil, err
	}
	ids := make([]uint32, 0, len(used))
	for id := range used {
		ids = append(ids, id)
	}
	sort.Slice(ids, func(i, j int) bool { return ids[i] < ids[j] })
	return ids, nil
}

func (b *builder) constant(t *types.Type, raw string) (uint32, error) {
	tid, err := b.typeID(t, typeLogical)
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
		v, err := strconv.ParseInt(raw, 10, 32)
		if err != nil {
			return 0, err
		}
		emit(&b.typesGlobals, OpConstant, tid, id, uint32(int32(v)))
	case types.U32:
		v, err := strconv.ParseUint(raw, 10, 32)
		if err != nil {
			return 0, err
		}
		emit(&b.typesGlobals, OpConstant, tid, id, uint32(v))
	case types.F32:
		v, err := strconv.ParseFloat(raw, 32)
		if err != nil {
			return 0, err
		}
		emit(&b.typesGlobals, OpConstant, tid, id, math.Float32bits(float32(v)))
	case types.F16:
		v, err := strconv.ParseFloat(raw, 64)
		bits, ok := types.Float16bits(v)
		if err != nil || !ok {
			return 0, fmt.Errorf("invalid float16 constant %q", raw)
		}
		emit(&b.typesGlobals, OpConstant, tid, id, uint32(bits))
	default:
		return 0, fmt.Errorf("constant type %s is not scalar", t)
	}
	b.constants[key] = id
	return id, nil
}

func (b *builder) u32Constant(v uint32) (uint32, error) {
	return b.constant(types.TU32, strconv.FormatUint(uint64(v), 10))
}

func (b *builder) nullConstant(t *types.Type) (uint32, error) {
	tid, err := b.typeID(t, typeLogical)
	if err != nil {
		return 0, err
	}
	key := fmt.Sprintf("%d:null", tid)
	if id := b.constants[key]; id != 0 {
		return id, nil
	}
	id := b.id()
	b.constants[key] = id
	emit(&b.typesGlobals, OpConstantNull, tid, id)
	return id, nil
}

type spvPlace struct {
	ptr         uint32
	ty          *types.Type
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
	vtypes map[ir.ValueID]*types.Type
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

	s := &fnEmitter{b: b, f: f, values: map[ir.ValueID]uint32{}, vtypes: map[ir.ValueID]*types.Type{}, places: map[ir.PlaceID]spvPlace{}, inputs: map[inputKind]uint32{}}
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

func (s *fnEmitter) emitParameterValue(block *abi.ParameterBlock, parameter int, logical *types.Type, cursor *int) (uint32, error) {
	if logical.Kind == types.Struct {
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
	if field.Parameter != parameter || !types.Equal(field.Logical, logical) {
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
	if logical.Kind != types.Bool {
		return loaded, nil
	}
	zero, err := s.b.u32Constant(0)
	if err != nil {
		return 0, err
	}
	boolType, err := s.b.typeID(types.TBool, typeLogical)
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
	u32Type, err := s.b.typeID(types.TU32, typeLogical)
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
			t := types.TU32
			if input != inputLocalLinear {
				t = types.Vec(types.TU32, 3)
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
			s.def(value, loaded, types.TU32)
			continue
		}
		id := s.b.id()
		emit(&s.b.functions, OpCompositeExtract, u32Type, id, loaded, dimension)
		s.def(value, id, types.TU32)
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

func (s *fnEmitter) accessField(base spvPlace, field int, t *types.Type) (spvPlace, error) {
	if base.ty == nil || base.ty.Kind != types.Struct {
		return spvPlace{}, fmt.Errorf("field access base is not a struct place")
	}
	if field < 0 || field >= len(base.ty.Fields) || !types.Equal(base.ty.Fields[field].Type, t) {
		return spvPlace{}, fmt.Errorf("field %d does not match place type %s", field, base.ty)
	}
	ptrType, err := s.b.pointerID(base.storage, t)
	if err != nil {
		return spvPlace{}, err
	}
	idx, err := s.b.u32Constant(uint32(field))
	if err != nil {
		return spvPlace{}, err
	}
	id := s.b.id()
	emit(&s.b.functions, OpAccessChain, ptrType, id, base.ptr, idx)
	p := spvPlace{ptr: id, ty: t, storage: base.storage, resource: base.resource}
	if t.Kind == types.RuntimeArray {
		p.arrayBase = base.ptr
		p.arrayMember = uint32(field)
		p.hasArrayLen = true
	}
	return p, nil
}

// loadPlace keeps physical host-layout aggregates behind descriptor pointers.
// A constructible resource struct is loaded field-by-field into its one logical
// SSA type; Workgroup and ordinary values use that logical type directly.
func (s *fnEmitter) loadPlace(p spvPlace, t *types.Type) (uint32, error) {
	if !types.IsConstructible(t) {
		return 0, fmt.Errorf("cannot load non-constructible place type %s", t)
	}
	role := typeRoleForStorage(p.storage)
	if role == typeHostABI && t.Kind == types.Struct {
		tid, err := s.b.typeID(t, typeLogical)
		if err != nil {
			return 0, err
		}
		ops := []uint32{tid, s.b.id()}
		for i, f := range t.Fields {
			fp, err := s.accessField(p, i, f.Type)
			if err != nil {
				return 0, err
			}
			v, err := s.loadPlace(fp, f.Type)
			if err != nil {
				return 0, err
			}
			ops = append(ops, v)
		}
		emit(&s.b.functions, OpCompositeConstruct, ops...)
		return ops[1], nil
	}

	physical, err := s.b.typeID(t, role)
	if err != nil {
		return 0, err
	}
	logical, err := s.b.typeID(t, typeLogical)
	if err != nil {
		return 0, err
	}
	if physical != logical {
		return 0, fmt.Errorf("place type %s requires structural loading", t)
	}
	id := s.b.id()
	if err := s.emitLoad(logical, id, p.ptr, p.storage, t); err != nil {
		return 0, err
	}
	return id, nil
}

// storePlace is the exact inverse of loadPlace: logical resource structs are
// decomposed into host-layout fields, while logical Workgroup values are stored
// directly. No physical aggregate is admitted into the SSA value domain.
func (s *fnEmitter) storePlace(p spvPlace, value uint32) error {
	t := p.ty
	if !types.IsConstructible(t) {
		return fmt.Errorf("cannot store non-constructible place type %s", t)
	}
	role := typeRoleForStorage(p.storage)
	if role == typeHostABI && t.Kind == types.Struct {
		for i, f := range t.Fields {
			fieldType, err := s.b.typeID(f.Type, typeLogical)
			if err != nil {
				return err
			}
			fieldValue := s.b.id()
			emit(&s.b.functions, OpCompositeExtract, fieldType, fieldValue, value, uint32(i))
			fp, err := s.accessField(p, i, f.Type)
			if err != nil {
				return err
			}
			if err := s.storePlace(fp, fieldValue); err != nil {
				return err
			}
		}
		return nil
	}

	physical, err := s.b.typeID(t, role)
	if err != nil {
		return err
	}
	logical, err := s.b.typeID(t, typeLogical)
	if err != nil {
		return err
	}
	if physical != logical {
		return fmt.Errorf("place type %s requires structural storage", t)
	}
	return s.emitStore(p.ptr, value, p.storage, t)
}

func (s *fnEmitter) emitInstr(in ir.Instr) error {
	if definition, ok := in.(ir.ValueDef); ok {
		lowered := s.b.p.functions[s.f]
		if lowered.Replaced[definition.ResultValue()] {
			return nil
		}
	}
	switch x := in.(type) {
	case *ir.Const:
		if s.b.p.functions[s.f].Uses[x.Result] == 0 {
			return nil
		}
		id, err := s.b.constant(x.Type, x.Raw)
		if err != nil {
			return err
		}
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
		tid, _ := s.b.typeID(x.Type, typeLogical)
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
		tid, _ := s.b.typeID(x.Type, typeLogical)
		id := s.b.id()
		emit(&s.b.functions, OpCompositeExtract, tid, id, base, uint32(x.Index))
		s.def(x.Result, id, x.Type)
	case *ir.VectorIndex:
		base, err := s.value(x.Base)
		if err != nil {
			return err
		}
		index, err := s.value(x.Index)
		if err != nil {
			return err
		}
		tid, _ := s.b.typeID(x.Type, typeLogical)
		id := s.b.id()
		emit(&s.b.functions, OpVectorExtractDynamic, tid, id, base, index)
		s.def(x.Result, id, x.Type)
	case *ir.Call:
		fid := s.b.funcIDs[x.Function]
		if fid == 0 {
			return fmt.Errorf("unknown callee %s", x.Function)
		}
		tid, _ := s.b.typeID(x.Type, typeLogical)
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
		calls := s.b.calls[s.f.Name]
		if calls == nil {
			calls = map[string]bool{}
			s.b.calls[s.f.Name] = calls
		}
		calls[x.Function] = true
		s.def(x.Result, result, x.Type)
	case *ir.PlaceRoot:
		return s.emitPlaceRoot(x)
	case *ir.PlaceWorkgroup:
		ids := s.b.workgroupIDs[s.f.Name]
		if x.Workgroup < 0 || x.Workgroup >= len(ids) {
			return fmt.Errorf("workgroup index %d out of bounds", x.Workgroup)
		}
		s.useGlobal(ids[x.Workgroup])
		s.places[x.Result] = spvPlace{ptr: ids[x.Workgroup], ty: x.Type, storage: StorageWorkgroup, resource: -1}
	case *ir.PlaceField:
		return s.emitPlaceField(x)
	case *ir.PlaceIndex:
		return s.emitPlaceIndex(x)
	case *ir.Load:
		p, err := s.place(x.Place)
		if err != nil {
			return err
		}
		id, err := s.loadPlace(p, x.Type)
		if err != nil {
			return err
		}
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
		if err := s.storePlace(p, v); err != nil {
			return err
		}
	case *ir.Atomic:
		return s.emitAtomic(x)
	case *ir.Barrier:
		return s.emitBarrier(x)
	case *ir.ArrayLength:
		p, err := s.place(x.Place)
		if err != nil {
			return err
		}
		if length, ok := s.b.p.kernels[s.f].LogicalLengths[p.resource]; ok {
			id, err := s.value(length)
			if err != nil {
				return err
			}
			s.def(x.Result, id, types.TU32)
			return nil
		}
		if !p.hasArrayLen {
			return fmt.Errorf("runtime-array place lacks OpArrayLength base")
		}
		tid, _ := s.b.typeID(types.TU32, typeLogical)
		id := s.b.id()
		emit(&s.b.functions, OpArrayLength, tid, id, p.arrayBase, p.arrayMember)
		s.def(x.Result, id, types.TU32)
	case *ir.If:
		return s.emitIf(x)
	case *ir.Loop:
		return s.emitLoop(x)
	case *ir.Scope:
		return s.emitScope(x)
	default:
		return fmt.Errorf("unsupported IR instruction %T", in)
	}
	return nil
}

func (s *fnEmitter) emitScope(scope *ir.Scope) error {
	header, body, cont, merge := s.b.id(), s.b.id(), s.b.id(), s.b.id()
	again, err := s.b.constant(types.TBool, "false")
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

func (s *fnEmitter) emitIntrinsic(x *ir.Intrinsic) error {
	tid, err := s.b.typeID(x.Type, typeLogical)
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
	if x.Kind == ir.IntrinsicAll || x.Kind == ir.IntrinsicAny {
		op := OpAll
		if x.Kind == ir.IntrinsicAny {
			op = OpAny
		}
		id := s.b.id()
		emit(&s.b.functions, op, tid, id, args[0])
		s.def(x.Result, id, x.Type)
		return nil
	}
	if x.Kind == ir.IntrinsicSelect {
		id := s.b.id()
		emit(&s.b.functions, OpSelect, tid, id, args[0], args[1], args[2])
		s.def(x.Result, id, x.Type)
		return nil
	}
	if x.Kind == ir.IntrinsicDot {
		id := s.b.id()
		emit(&s.b.functions, OpDot, tid, id, args[0], args[1])
		s.def(x.Result, id, x.Type)
		return nil
	}
	if x.Kind == ir.IntrinsicMin || x.Kind == ir.IntrinsicMax || x.Kind == ir.IntrinsicClamp {
		conditionType := types.TBool
		if x.Type.Kind == types.Vector {
			conditionType = types.Vec(types.TBool, x.Type.Lanes)
		}
		conditionTypeID, err := s.b.typeID(conditionType, typeLogical)
		if err != nil {
			return err
		}
		kind := scalarKind(x.Type)
		less := OpFOrdLessThan
		if kind == types.I32 {
			less = OpSLessThan
		} else if kind == types.U32 {
			less = OpULessThan
		}
		bound := func(minimum bool, left, right uint32) uint32 {
			condition, result := s.b.id(), s.b.id()
			first, second := left, right
			if minimum {
				first, second = right, left
			}
			emit(&s.b.functions, less, conditionTypeID, condition, first, second)
			emit(&s.b.functions, OpSelect, tid, result, condition, right, left)
			return result
		}
		result := bound(x.Kind == ir.IntrinsicMin, args[0], args[1])
		if x.Kind == ir.IntrinsicClamp {
			result = bound(true, result, args[2])
		}
		s.def(x.Result, result, x.Type)
		return nil
	}

	id := s.b.id()
	var inst uint32
	switch x.Kind {
	case ir.IntrinsicAbs:
		if types.IsFloatLike(x.Type) {
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
	case ir.IntrinsicRSqrt:
		inst = GLSL450InverseSqrt
	case ir.IntrinsicPow:
		inst = GLSL450Pow
	case ir.IntrinsicFma:
		inst = GLSL450Fma
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
	scopeValue := ScopeQueueFamily // QueueFamily avoids vulkanMemoryModelDeviceScope.
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
	tid, err := s.b.typeID(x.Type, typeLogical)
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
	if x.Op == ir.AtomicCompareExchange {
		expected, err := s.value(x.Expected)
		if err != nil {
			return err
		}
		value, err := s.value(x.Value)
		if err != nil {
			return err
		}
		id := s.b.id()
		emit(&s.b.functions, OpAtomicCompareExchange, tid, id, p.ptr, scope, semantics, semantics, value, expected)
		s.def(x.Result, id, x.Type)
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
	sem := MemorySemanticsAcquireRelease | MemorySemanticsMakeAvailable | MemorySemanticsMakeVisible
	switch x.Kind {
	case ir.BarrierWorkgroup:
		sem |= MemorySemanticsWorkgroupMemory
	case ir.BarrierBuffer:
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
	resources := s.b.resourceIDs[s.f]
	if x.Buffer < 0 || x.Buffer >= len(resources) {
		return fmt.Errorf("buffer index %d out of bounds", x.Buffer)
	}
	s.useGlobal(resources[x.Buffer])
	storage := uint32(StorageStorageBuffer)
	if x.Type.Kind == types.Struct && types.ContainsRuntimeArray(x.Type) {
		s.places[x.Result] = spvPlace{ptr: resources[x.Buffer], ty: x.Type, storage: storage, resource: x.Buffer}
		return nil
	}
	ptrType, err := s.b.pointerID(storage, x.Type)
	if err != nil {
		return err
	}
	zero, err := s.b.u32Constant(0)
	if err != nil {
		return err
	}
	id := s.b.id()
	emit(&s.b.functions, OpAccessChain, ptrType, id, resources[x.Buffer], zero)
	p := spvPlace{ptr: id, ty: x.Type, storage: storage, resource: x.Buffer}
	if x.Type.Kind == types.RuntimeArray {
		p.arrayBase = resources[x.Buffer]
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
	p, err := s.accessField(base, x.Field, x.Type)
	if err != nil {
		return err
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
	s.places[x.Result] = spvPlace{ptr: id, ty: x.Type, storage: base.storage, resource: base.resource}
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

func memoryAlignment(storage uint32, t *types.Type) (uint32, error) {
	if storage == StorageUniform || storage == StorageStorageBuffer {
		l, err := layout.Of(t)
		if err != nil {
			return 0, err
		}
		return l.Align, nil
	}
	return logicalAlignment(t)
}

func logicalAlignment(t *types.Type) (uint32, error) {
	if t == nil {
		return 0, fmt.Errorf("nil type has no memory alignment")
	}
	switch t.Kind {
	case types.F16:
		return 2, nil
	case types.I32, types.U32, types.F32, types.Atomic:
		return 4, nil
	case types.Vector:
		element, err := logicalAlignment(t.Elem)
		if err != nil {
			return 0, err
		}
		if t.Lanes == 2 {
			return element * 2, nil
		}
		return element * 4, nil
	case types.FixedArray:
		return logicalAlignment(t.Elem)
	case types.Struct:
		var align uint32
		for _, field := range t.Fields {
			fieldAlign, err := logicalAlignment(field.Type)
			if err != nil {
				return 0, err
			}
			if fieldAlign > align {
				align = fieldAlign
			}
		}
		if align == 0 {
			return 0, fmt.Errorf("struct %s has no aligned members", t)
		}
		return align, nil
	default:
		return 0, fmt.Errorf("type %s has no memory alignment", t)
	}
}

func (s *fnEmitter) memoryAccess(storage uint32, t *types.Type) (mask, align uint32, err error) {
	align, err = memoryAlignment(storage, t)
	if err != nil {
		return 0, 0, err
	}
	mask = MemoryAccessAligned
	if storage == StorageStorageBuffer || storage == StorageUniform || storage == StorageWorkgroup {
		mask |= MemoryAccessNonPrivatePointer
	}
	return mask, align, nil
}

func (s *fnEmitter) emitLoad(resultType, result, ptr, storage uint32, t *types.Type) error {
	mask, align, err := s.memoryAccess(storage, t)
	if err != nil {
		return err
	}
	emit(&s.b.functions, OpLoad, resultType, result, ptr, mask, align)
	return nil
}

func (s *fnEmitter) emitStore(ptr, value, storage uint32, t *types.Type) error {
	mask, align, err := s.memoryAccess(storage, t)
	if err != nil {
		return err
	}
	emit(&s.b.functions, OpStore, ptr, value, mask, align)
	return nil
}

func (s *fnEmitter) emitUnary(x *ir.Unary) error {
	v, err := s.value(x.X)
	if err != nil {
		return err
	}
	tid, _ := s.b.typeID(x.Type, typeLogical)
	id := s.b.id()
	var op Op
	switch x.Op {
	case "!":
		op = OpLogicalNot
	case "-":
		if types.IsFloatLike(x.Type) {
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
	tid, err := s.b.typeID(vector, typeLogical)
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
	tid, _ := s.b.typeID(x.Type, typeLogical)
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
		if kind == types.F16 || kind == types.F32 {
			op = OpFAdd
		} else {
			op = OpIAdd
		}
	case "-":
		if kind == types.F16 || kind == types.F32 {
			op = OpFSub
		} else {
			op = OpISub
		}
	case "*":
		if kind == types.F16 || kind == types.F32 {
			op = OpFMul
		} else {
			op = OpIMul
		}
	case "/":
		switch kind {
		case types.F16, types.F32:
			op = OpFDiv
		case types.U32:
			op = OpUDiv
		case types.I32:
			op = OpSDiv
		}
	case "%":
		switch kind {
		case types.F16, types.F32:
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
		if kind == types.Bool {
			op = OpLogicalAnd
		} else {
			op = OpBitwiseAnd
		}
	case "|":
		if kind == types.Bool {
			op = OpLogicalOr
		} else {
			op = OpBitwiseOr
		}
	case "^":
		if kind == types.Bool {
			op = OpLogicalNotEqual
		} else {
			op = OpBitwiseXor
		}
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
		case types.F16, types.F32:
			op = OpFOrdEqual
		case types.I32, types.U32:
			op = OpIEqual
		case types.Bool:
			op = OpLogicalEqual
		}
	case "!=":
		switch kind {
		case types.F16, types.F32:
			op = OpFOrdNotEqual
		case types.I32, types.U32:
			op = OpINotEqual
		case types.Bool:
			op = OpLogicalNotEqual
		}
	case "<":
		switch kind {
		case types.F16, types.F32:
			op = OpFOrdLessThan
		case types.U32:
			op = OpULessThan
		case types.I32:
			op = OpSLessThan
		}
	case "<=":
		switch kind {
		case types.F16, types.F32:
			op = OpFOrdLessThanEqual
		case types.U32:
			op = OpULessThanEqual
		case types.I32:
			op = OpSLessThanEqual
		}
	case ">":
		switch kind {
		case types.F16, types.F32:
			op = OpFOrdGreaterThan
		case types.U32:
			op = OpUGreaterThan
		case types.I32:
			op = OpSGreaterThan
		}
	case ">=":
		switch kind {
		case types.F16, types.F32:
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
	tid, _ := s.b.typeID(x.Type, typeLogical)
	id := s.b.id()
	var op Op
	switch {
	case types.IsFloatLike(x.From) && x.Type.Kind == types.U32:
		op = OpConvertFToU
	case types.IsFloatLike(x.From) && x.Type.Kind == types.I32:
		op = OpConvertFToS
	case x.From.Kind == types.I32 && types.IsFloatLike(x.Type):
		op = OpConvertSToF
	case x.From.Kind == types.U32 && types.IsFloatLike(x.Type):
		op = OpConvertUToF
	case types.IsFloatLike(x.From) && types.IsFloatLike(x.Type):
		op = OpFConvert
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

	incomingByResult := make([][]phiIncoming, len(x.Results))

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
			incomingByResult[i] = append(incomingByResult[i], phiIncoming{v, te.pred})
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
			incomingByResult[i] = append(incomingByResult[i], phiIncoming{v, ee.pred})
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
			s.def(r.ID, incs[0].value, r.Type)
			continue
		}
		tid, _ := s.b.typeID(r.Type, typeLogical)
		id := s.b.id()
		ops := []uint32{tid, id}
		for _, in := range incs {
			ops = append(ops, in.value, in.label)
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
	params := make([]uint32, len(x.Params))
	for i, p := range x.Params {
		init, err := s.value(p.Init)
		if err != nil {
			return err
		}
		tid, _ := s.b.typeID(p.Type, typeLogical)
		phi := s.b.id()
		start := emit(&s.b.functions, OpPhi, tid, phi, init, preheader, 0, cont)
		patches[i] = start + 5 // first word + operands: type,result,init,pre,back,cont
		params[i] = phi
		s.def(p.ID, phi, p.Type)
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
	conditionExit := ce.pred
	emit(&s.b.functions, OpBranchConditional, ce.vals[0], body, merge)
	s.terminated = true

	emit(&s.b.functions, OpLabel, body)
	s.currentLabel, s.terminated = body, false
	s.loops = append(s.loops, loopTarget{
		merge:      merge,
		continuing: cont,
		breaks:     make([][]phiIncoming, len(x.Params)),
		continues:  make([][]phiIncoming, len(x.Params)),
	})
	_, err = s.emitBlockExit(x.Body, blockNormal)
	if err != nil {
		return err
	}
	loop := s.loops[len(s.loops)-1]
	s.loops = s.loops[:len(s.loops)-1]

	emit(&s.b.functions, OpLabel, cont)
	s.currentLabel, s.terminated = cont, false
	for i, incoming := range loop.continues {
		back := uint32(0)
		switch len(incoming) {
		case 0:
			back, err = s.value(x.Params[i].Init)
		case 1:
			back = incoming[0].value
		default:
			tid, _ := s.b.typeID(x.Params[i].Type, typeLogical)
			back = s.b.id()
			ops := []uint32{tid, back}
			for _, value := range incoming {
				ops = append(ops, value.value, value.label)
			}
			emit(&s.b.functions, OpPhi, ops...)
		}
		if err != nil {
			return err
		}
		s.b.functions[patches[i]] = back
	}
	emit(&s.b.functions, OpBranch, header)
	s.terminated = true

	emit(&s.b.functions, OpLabel, merge)
	s.currentLabel, s.terminated = merge, false
	for i, result := range x.Results {
		incoming := append([]phiIncoming{{params[i], conditionExit}}, loop.breaks[i]...)
		if len(incoming) == 1 {
			s.def(result.ID, incoming[0].value, result.Type)
			continue
		}
		tid, _ := s.b.typeID(result.Type, typeLogical)
		id := s.b.id()
		ops := []uint32{tid, id}
		for _, value := range incoming {
			ops = append(ops, value.value, value.label)
		}
		emit(&s.b.functions, OpPhi, ops...)
		s.def(result.ID, id, result.Type)
	}
	return nil
}
