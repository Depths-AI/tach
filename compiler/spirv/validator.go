package spirv

import (
	"fmt"
)

type typeKind uint8

const (
	typeVoid typeKind = iota + 1
	typeBool
	typeInt
	typeFloat
	typeVector
	typeArray
	typeRuntimeArray
	typeStruct
	typePointer
	typeFunction
)

type typeInfo struct {
	kind    typeKind
	width   uint32
	signed  bool
	elem    uint32
	lanes   uint32
	length  uint32
	members []uint32
	storage uint32
	ret     uint32
	params  []uint32
}

type decorationInfo struct {
	block       bool
	nonWritable bool
	arrayStride *uint32
	builtin     *uint32
	binding     *uint32
	set         *uint32
	offsets     map[uint32]uint32
}

type blockInfo struct {
	label uint32
	insts []int // indexes in Module.Instructions
	succ  []uint32
	pred  []uint32
	phis  []int
}

type functionInfo struct {
	id     uint32
	ret    uint32
	fnType uint32
	params []uint32
	blocks map[uint32]*blockInfo
	order  []uint32
}

type validation struct {
	m            *Module
	defs         map[uint32]int
	types        map[uint32]*typeInfo
	valueType    map[uint32]uint32
	constants    map[uint32]uint64
	decor        map[uint32]*decorationInfo
	functions    map[uint32]*functionInfo
	globalVars   map[uint32]uint32 // id -> storage
	pointerRoot  map[uint32]uint32
	entryPoints  map[uint32]string
	localSize    map[uint32][3]uint32
	extImports   map[uint32]string
	capabilities map[uint32]int
}

// Validate owns the complete validity contract for Tach-generated SPIR-V. It
// validates binary framing, logical layout, IDs, types, memory accesses,
// descriptor ABI, CFG/SSA shape, structured merge form, and entry-point ABI.
func Validate(data []byte) error {
	m, err := Decode(data)
	if err != nil {
		return err
	}
	v := &validation{
		m: m, defs: map[uint32]int{}, types: map[uint32]*typeInfo{}, valueType: map[uint32]uint32{},
		constants: map[uint32]uint64{}, decor: map[uint32]*decorationInfo{}, functions: map[uint32]*functionInfo{},
		globalVars: map[uint32]uint32{}, pointerRoot: map[uint32]uint32{}, entryPoints: map[uint32]string{}, localSize: map[uint32][3]uint32{},
		extImports: map[uint32]string{}, capabilities: map[uint32]int{},
	}
	if err := v.validateLayoutAndDefinitions(); err != nil {
		return err
	}
	if err := v.validateReferencesAndTypes(); err != nil {
		return err
	}
	if err := v.validateDecorationsAndABI(); err != nil {
		return err
	}
	if err := v.validateFunctions(); err != nil {
		return err
	}
	if err := v.validateEntryPoints(); err != nil {
		return err
	}
	return nil
}

func section(op Op) int {
	switch op {
	case OpCapability:
		return 1
	case OpExtInstImport:
		return 2
	case OpMemoryModel:
		return 4
	case OpEntryPoint:
		return 5
	case OpExecutionMode:
		return 6
	case OpName, OpMemberName:
		return 7
	case OpDecorate, OpMemberDecorate:
		return 8
	case OpTypeVoid, OpTypeBool, OpTypeInt, OpTypeFloat, OpTypeVector, OpTypeArray, OpTypeRuntimeArray, OpTypeStruct,
		OpTypePointer, OpTypeFunction, OpConstantTrue, OpConstantFalse, OpConstant, OpConstantComposite, OpConstantNull, OpVariable:
		return 9
	default:
		return 10
	}
}

func resultID(in Instruction) uint32 {
	a := in.Operands
	switch in.Op {
	case OpExtInstImport:
		return a[0]
	case OpTypeVoid, OpTypeBool, OpTypeInt, OpTypeFloat, OpTypeVector, OpTypeArray, OpTypeRuntimeArray, OpTypeStruct, OpTypePointer, OpTypeFunction, OpLabel:
		return a[0]
	}
	if hasResultType(in.Op) {
		return a[1]
	}
	return 0
}

func resultTypeID(in Instruction) uint32 {
	if hasResultType(in.Op) {
		return in.Operands[0]
	}
	return 0
}

func hasResultType(op Op) bool {
	switch op {
	case OpConstantTrue, OpConstantFalse, OpConstant, OpConstantComposite, OpConstantNull, OpFunction, OpFunctionParameter,
		OpFunctionCall, OpVariable, OpLoad, OpAccessChain, OpArrayLength, OpCompositeConstruct, OpVectorExtractDynamic, OpCompositeExtract,
		OpConvertFToU, OpConvertFToS, OpConvertSToF, OpConvertUToF, OpFConvert, OpBitcast, OpSNegate, OpFNegate,
		OpIAdd, OpFAdd, OpISub, OpFSub, OpIMul, OpFMul, OpUDiv, OpSDiv, OpFDiv, OpUMod, OpSRem, OpFRem,
		OpVectorTimesScalar, OpAny, OpAll, OpLogicalEqual, OpLogicalNotEqual, OpLogicalOr, OpLogicalAnd, OpLogicalNot, OpNot,
		OpShiftRightLogical, OpShiftRightArithmetic, OpShiftLeftLogical, OpBitwiseOr, OpBitwiseXor, OpBitwiseAnd, OpSelect,
		OpIEqual, OpINotEqual, OpUGreaterThan, OpSGreaterThan, OpUGreaterThanEqual, OpSGreaterThanEqual,
		OpULessThan, OpSLessThan, OpULessThanEqual, OpSLessThanEqual, OpFOrdEqual, OpFOrdNotEqual,
		OpFOrdLessThan, OpFOrdGreaterThan, OpFOrdLessThanEqual, OpFOrdGreaterThanEqual,
		OpAtomicLoad, OpAtomicExchange, OpAtomicCompareExchange, OpAtomicIAdd, OpAtomicISub, OpAtomicSMin, OpAtomicUMin,
		OpAtomicSMax, OpAtomicUMax, OpAtomicAnd, OpAtomicOr, OpAtomicXor, OpPhi, OpExtInst, OpDot:
		return true
	}
	return false
}
func (v *validation) def(id uint32, idx int, what string) error {
	if id == 0 || id >= v.m.Bound {
		return fmt.Errorf("%s defines invalid id %%%d (bound %d)", what, id, v.m.Bound)
	}
	if prev, ok := v.defs[id]; ok {
		return fmt.Errorf("id %%%d is defined twice (instructions %d and %d)", id, prev, idx)
	}
	v.defs[id] = idx
	return nil
}

func (v *validation) decoration(id uint32) *decorationInfo {
	d := v.decor[id]
	if d == nil {
		d = &decorationInfo{offsets: map[uint32]uint32{}}
		v.decor[id] = d
	}
	return d
}

func setOnce(dst **uint32, val uint32, name string, id uint32) error {
	if *dst != nil {
		return fmt.Errorf("%%%d has duplicate %s decoration", id, name)
	}
	x := val
	*dst = &x
	return nil
}

func (v *validation) validateLayoutAndDefinitions() error {
	lastSec := 0
	memory := 0
	inFunc := false
	var cur *functionInfo
	paramsOpen := false
	for i, in := range v.m.Instructions {
		sec := section(in.Op)
		if sec < lastSec {
			return fmt.Errorf("word %d %s appears out of SPIR-V logical layout order", in.Offset, opName(in.Op))
		}
		lastSec = sec
		if id := resultID(in); id != 0 {
			if err := v.def(id, i, opName(in.Op)); err != nil {
				return err
			}
			if tid := resultTypeID(in); tid != 0 {
				v.valueType[id] = tid
			}
		}
		a := in.Operands
		switch in.Op {
		case OpCapability:
			switch a[0] {
			case CapabilityShader, CapabilityFloat16, CapabilityStorageBuffer16BitAccess, CapabilityUniformAndStorage16BitAccess, CapabilityVulkanMemoryModel:
				v.capabilities[a[0]]++
			default:
				return fmt.Errorf("capability %d is outside Tach profile", a[0])
			}
		case OpExtInstImport:
			name, next, err := literalString(a, 1)
			if err != nil || next != len(a) {
				return fmt.Errorf("invalid extended instruction import")
			}
			if name != "GLSL.std.450" {
				return fmt.Errorf("tach supports only GLSL.std.450 extended instructions, got %q", name)
			}
			for id, existing := range v.extImports {
				if existing == name && id != a[0] {
					return fmt.Errorf("GLSL.std.450 imported more than once")
				}
			}
			v.extImports[a[0]] = name
		case OpMemoryModel:
			memory++
			if a[0] != AddressingLogical || a[1] != MemoryVulkan {
				return fmt.Errorf("tach requires Logical + Vulkan memory model")
			}
		case OpEntryPoint:
			if a[0] != ExecutionModelGLCompute {
				return fmt.Errorf("tach supports GLCompute entry points only")
			}
			name, _, _ := literalString(a, 2)
			if _, exists := v.entryPoints[a[1]]; exists {
				return fmt.Errorf("function %%%d declared as entry point more than once", a[1])
			}
			v.entryPoints[a[1]] = name
		case OpExecutionMode:
			if a[1] != ExecutionModeLocalSize {
				return fmt.Errorf("tach supports LocalSize execution mode only")
			}
			if a[2] == 0 || a[3] == 0 || a[4] == 0 {
				return fmt.Errorf("LocalSize components must be positive")
			}
			if _, exists := v.localSize[a[0]]; exists {
				return fmt.Errorf("function %%%d has duplicate LocalSize", a[0])
			}
			v.localSize[a[0]] = [3]uint32{a[2], a[3], a[4]}
		case OpDecorate:
			d := v.decoration(a[0])
			switch a[1] {
			case DecorationBlock:
				if d.block {
					return fmt.Errorf("%%%d has duplicate Block", a[0])
				}
				d.block = true
			case DecorationNonWritable:
				if d.nonWritable {
					return fmt.Errorf("%%%d has duplicate NonWritable", a[0])
				}
				d.nonWritable = true
			case DecorationArrayStride:
				if err := setOnce(&d.arrayStride, a[2], "ArrayStride", a[0]); err != nil {
					return err
				}
			case DecorationBuiltIn:
				if err := setOnce(&d.builtin, a[2], "BuiltIn", a[0]); err != nil {
					return err
				}
			case DecorationBinding:
				if err := setOnce(&d.binding, a[2], "Binding", a[0]); err != nil {
					return err
				}
			case DecorationDescriptorSet:
				if err := setOnce(&d.set, a[2], "DescriptorSet", a[0]); err != nil {
					return err
				}
			}
		case OpMemberDecorate:
			d := v.decoration(a[0])
			if _, exists := d.offsets[a[1]]; exists {
				return fmt.Errorf("%%%d member %d has duplicate Offset", a[0], a[1])
			}
			d.offsets[a[1]] = a[3]
		case OpTypeVoid:
			v.types[a[0]] = &typeInfo{kind: typeVoid}
		case OpTypeBool:
			v.types[a[0]] = &typeInfo{kind: typeBool}
		case OpTypeInt:
			if a[1] != 32 || (a[2] != 0 && a[2] != 1) {
				return fmt.Errorf("tach integer type must be 32-bit signed/unsigned")
			}
			v.types[a[0]] = &typeInfo{kind: typeInt, width: a[1], signed: a[2] == 1}
		case OpTypeFloat:
			if a[1] != 16 && a[1] != 32 {
				return fmt.Errorf("tach float type must be float16 or float32")
			}
			v.types[a[0]] = &typeInfo{kind: typeFloat, width: a[1]}
		case OpTypeVector:
			if a[2] < 2 || a[2] > 4 {
				return fmt.Errorf("tach vector width must be 2..4")
			}
			v.types[a[0]] = &typeInfo{kind: typeVector, elem: a[1], lanes: a[2]}
		case OpTypeArray:
			v.types[a[0]] = &typeInfo{kind: typeArray, elem: a[1], length: a[2]}
		case OpTypeRuntimeArray:
			v.types[a[0]] = &typeInfo{kind: typeRuntimeArray, elem: a[1]}
		case OpTypeStruct:
			v.types[a[0]] = &typeInfo{kind: typeStruct, members: append([]uint32(nil), a[1:]...)}
		case OpTypePointer:
			if a[1] != StorageInput && a[1] != StorageUniform && a[1] != StorageWorkgroup && a[1] != StorageStorageBuffer {
				return fmt.Errorf("pointer %%%d uses storage class %d outside Tach profile", a[0], a[1])
			}
			v.types[a[0]] = &typeInfo{kind: typePointer, storage: a[1], elem: a[2]}
		case OpTypeFunction:
			v.types[a[0]] = &typeInfo{kind: typeFunction, ret: a[1], params: append([]uint32(nil), a[2:]...)}
		case OpConstantTrue:
			v.valueType[a[1]] = a[0]
			v.constants[a[1]] = 1
		case OpConstantFalse:
			v.valueType[a[1]] = a[0]
			v.constants[a[1]] = 0
		case OpConstant:
			v.valueType[a[1]] = a[0]
			v.constants[a[1]] = uint64(a[2])
		case OpConstantComposite:
			v.valueType[a[1]] = a[0]
		case OpConstantNull:
			v.valueType[a[1]] = a[0]
		case OpVariable:
			v.valueType[a[1]] = a[0]
			v.globalVars[a[1]] = a[2]
			v.pointerRoot[a[1]] = a[1]
		case OpFunction:
			if inFunc {
				return fmt.Errorf("nested OpFunction")
			}
			inFunc = true
			paramsOpen = true
			cur = &functionInfo{id: a[1], ret: a[0], fnType: a[3], blocks: map[uint32]*blockInfo{}}
			v.functions[cur.id] = cur
			v.valueType[cur.id] = a[0]
		case OpFunctionParameter:
			if !inFunc || !paramsOpen {
				return fmt.Errorf("OpFunctionParameter outside parameter list")
			}
			cur.params = append(cur.params, a[1])
			v.valueType[a[1]] = a[0]
		case OpLabel:
			if !inFunc {
				return fmt.Errorf("OpLabel outside function")
			}
			paramsOpen = false
			if _, exists := cur.blocks[a[0]]; exists {
				return fmt.Errorf("duplicate label %%%d", a[0])
			}
			cur.blocks[a[0]] = &blockInfo{label: a[0]}
			cur.order = append(cur.order, a[0])
		case OpFunctionEnd:
			if !inFunc {
				return fmt.Errorf("OpFunctionEnd outside function")
			}
			if len(cur.order) == 0 {
				return fmt.Errorf("function %%%d has no basic block", cur.id)
			}
			inFunc = false
			cur = nil
			paramsOpen = false
		default:
			if sec == 10 && !inFunc {
				return fmt.Errorf("%s appears outside a function", opName(in.Op))
			}
			if inFunc && in.Op != OpFunctionEnd {
				if len(cur.order) == 0 {
					return fmt.Errorf("%s appears before first OpLabel", opName(in.Op))
				}
				bl := cur.blocks[cur.order[len(cur.order)-1]]
				bl.insts = append(bl.insts, i)
				if in.Op == OpPhi {
					bl.phis = append(bl.phis, i)
				}
			}
		}
	}
	if inFunc {
		return fmt.Errorf("unterminated OpFunction")
	}
	if v.capabilities[CapabilityShader] != 1 || v.capabilities[CapabilityVulkanMemoryModel] != 1 {
		return fmt.Errorf("tach module must declare Shader and vulkanMemoryModel exactly once each")
	}
	for capability, count := range v.capabilities {
		if count != 1 {
			return fmt.Errorf("tach capability %d must be declared at most once", capability)
		}
	}
	float16 := false
	for _, t := range v.types {
		float16 = float16 || t.kind == typeFloat && t.width == 16
	}
	if float16 != (v.capabilities[CapabilityFloat16] == 1) {
		return fmt.Errorf("Float16 capability must exactly match use of float16 types")
	}
	if memory != 1 {
		return fmt.Errorf("tach module must declare one memory model")
	}
	return nil
}

func (v *validation) requireID(id uint32, ctx string) error {
	if id == 0 || id >= v.m.Bound {
		return fmt.Errorf("%s references invalid id %%%d", ctx, id)
	}
	if _, ok := v.defs[id]; !ok {
		return fmt.Errorf("%s references undefined id %%%d", ctx, id)
	}
	return nil
}
func (v *validation) requireType(id uint32, ctx string) (*typeInfo, error) {
	if err := v.requireID(id, ctx); err != nil {
		return nil, err
	}
	t := v.types[id]
	if t == nil {
		return nil, fmt.Errorf("%s references %%%d which is not a type", ctx, id)
	}
	return t, nil
}
func (v *validation) requireValue(id uint32, ctx string) (uint32, error) {
	if err := v.requireID(id, ctx); err != nil {
		return 0, err
	}
	t := v.valueType[id]
	if t == 0 {
		return 0, fmt.Errorf("%s references %%%d which is not a value", ctx, id)
	}
	return t, nil
}
