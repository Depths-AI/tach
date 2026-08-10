package spirv

import (
	"encoding/binary"
	"fmt"
	"sort"
	"strings"
)

// Instruction is Tach's decoded representation of one SPIR-V instruction.
type Instruction struct {
	Offset   int // word offset in the module
	Op       Op
	Operands []uint32
}

// Module is the binary-level representation consumed by Tach's SPIR-V
// validator and disassembler.
type Module struct {
	Version      uint32
	Generator    uint32
	Bound        uint32
	Instructions []Instruction
}

// Decode performs strict binary framing validation and rejects every opcode
// outside Tach's owned SPIR-V profile.
func Decode(data []byte) (*Module, error) {
	if len(data) < 20 || len(data)%4 != 0 {
		return nil, fmt.Errorf("SPIR-V binary size %d is not a valid word stream", len(data))
	}
	words := make([]uint32, len(data)/4)
	for i := range words {
		words[i] = binary.LittleEndian.Uint32(data[i*4:])
	}
	if words[0] != Magic {
		return nil, fmt.Errorf("bad SPIR-V magic 0x%08x", words[0])
	}
	if words[1] != Version {
		return nil, fmt.Errorf("Tach SPIR-V profile requires version 1.3, got word 0x%08x", words[1])
	}
	if words[3] == 0 {
		return nil, fmt.Errorf("SPIR-V id bound must be non-zero")
	}
	if words[4] != 0 {
		return nil, fmt.Errorf("SPIR-V reserved schema word must be zero")
	}
	m := &Module{Version: words[1], Generator: words[2], Bound: words[3]}
	for off := 5; off < len(words); {
		first := words[off]
		wc := int(first >> 16)
		op := Op(first & 0xffff)
		if wc == 0 {
			return nil, fmt.Errorf("word %d: instruction has zero word count", off)
		}
		if off+wc > len(words) {
			return nil, fmt.Errorf("word %d: %s overruns module", off, opName(op))
		}
		if _, ok := opNames[op]; !ok {
			return nil, fmt.Errorf("word %d: opcode %d is outside Tach's SPIR-V profile", off, op)
		}
		ops := append([]uint32(nil), words[off+1:off+wc]...)
		if err := validateArity(op, ops); err != nil {
			return nil, fmt.Errorf("word %d %s: %w", off, opName(op), err)
		}
		m.Instructions = append(m.Instructions, Instruction{Offset: off, Op: op, Operands: ops})
		off += wc
	}
	return m, nil
}

func exact(op []uint32, n int) error {
	if len(op) != n {
		return fmt.Errorf("has %d operands, want %d", len(op), n)
	}
	return nil
}
func atLeast(op []uint32, n int) error {
	if len(op) < n {
		return fmt.Errorf("has %d operands, want at least %d", len(op), n)
	}
	return nil
}

func validateArity(op Op, a []uint32) error {
	switch op {
	case OpName:
		if err := atLeast(a, 2); err != nil {
			return err
		}
		_, _, err := literalString(a, 1)
		return err
	case OpMemberName:
		if err := atLeast(a, 3); err != nil {
			return err
		}
		_, _, err := literalString(a, 2)
		return err
	case OpExtInstImport:
		if err := atLeast(a, 2); err != nil {
			return err
		}
		_, next, err := literalString(a, 1)
		if err != nil {
			return err
		}
		if next != len(a) {
			return fmt.Errorf("has trailing operands after import name")
		}
		return nil
	case OpExtInst:
		return atLeast(a, 5)
	case OpMemoryModel:
		return exact(a, 2)
	case OpEntryPoint:
		if err := atLeast(a, 3); err != nil {
			return err
		}
		_, _, err := literalString(a, 2)
		return err
	case OpExecutionMode:
		return exact(a, 5)
	case OpCapability:
		return exact(a, 1)
	case OpTypeVoid, OpTypeBool:
		return exact(a, 1)
	case OpTypeInt:
		return exact(a, 3)
	case OpTypeFloat:
		return exact(a, 2)
	case OpTypeVector:
		return exact(a, 3)
	case OpTypeArray:
		return exact(a, 3)
	case OpTypeRuntimeArray:
		return exact(a, 2)
	case OpTypeStruct:
		return atLeast(a, 1)
	case OpTypePointer:
		return exact(a, 3)
	case OpTypeFunction:
		return atLeast(a, 2)
	case OpConstantTrue, OpConstantFalse:
		return exact(a, 2)
	case OpConstant:
		return exact(a, 3)
	case OpConstantComposite:
		return atLeast(a, 2)
	case OpFunction:
		return exact(a, 4)
	case OpFunctionParameter:
		return exact(a, 2)
	case OpFunctionEnd:
		return exact(a, 0)
	case OpFunctionCall:
		return atLeast(a, 3)
	case OpVariable:
		return exact(a, 3)
	case OpLoad:
		return exact(a, 3)
	case OpStore:
		return exact(a, 2)
	case OpAccessChain:
		return atLeast(a, 4)
	case OpArrayLength:
		return exact(a, 4)
	case OpDecorate:
		if err := atLeast(a, 2); err != nil {
			return err
		}
		switch a[1] {
		case DecorationBlock, DecorationNonWritable:
			return exact(a, 2)
		case DecorationArrayStride, DecorationBuiltIn, DecorationBinding, DecorationDescriptorSet:
			return exact(a, 3)
		default:
			return fmt.Errorf("decoration %d is outside Tach's profile", a[1])
		}
	case OpMemberDecorate:
		if err := atLeast(a, 3); err != nil {
			return err
		}
		if a[2] != DecorationOffset {
			return fmt.Errorf("member decoration %d is outside Tach's profile", a[2])
		}
		return exact(a, 4)
	case OpCompositeConstruct:
		return atLeast(a, 2)
	case OpCompositeExtract:
		return atLeast(a, 4)
	case OpConvertFToU, OpConvertFToS, OpConvertSToF, OpConvertUToF, OpBitcast,
		OpSNegate, OpFNegate, OpLogicalNot, OpNot:
		return exact(a, 3)
	case OpIAdd, OpFAdd, OpISub, OpFSub, OpIMul, OpFMul, OpUDiv, OpSDiv, OpFDiv,
		OpUMod, OpSRem, OpFRem, OpVectorTimesScalar, OpLogicalEqual, OpLogicalNotEqual,
		OpLogicalOr, OpLogicalAnd, OpIEqual, OpINotEqual, OpUGreaterThan, OpSGreaterThan,
		OpUGreaterThanEqual, OpSGreaterThanEqual, OpULessThan, OpSLessThan,
		OpULessThanEqual, OpSLessThanEqual, OpFOrdEqual, OpFOrdNotEqual, OpFOrdLessThan,
		OpFOrdGreaterThan, OpFOrdLessThanEqual, OpFOrdGreaterThanEqual,
		OpShiftRightLogical, OpShiftRightArithmetic, OpShiftLeftLogical, OpBitwiseOr, OpBitwiseXor, OpBitwiseAnd:
		return exact(a, 4)
	case OpDot:
		return exact(a, 4)
	case OpControlBarrier:
		return exact(a, 3)
	case OpAtomicLoad:
		return exact(a, 5)
	case OpAtomicStore:
		return exact(a, 4)
	case OpAtomicExchange, OpAtomicIAdd, OpAtomicISub, OpAtomicSMin, OpAtomicUMin,
		OpAtomicSMax, OpAtomicUMax, OpAtomicAnd, OpAtomicOr, OpAtomicXor:
		return exact(a, 6)
	case OpPhi:
		if len(a) < 6 || (len(a)-2)%2 != 0 {
			return fmt.Errorf("phi must contain at least two value/label pairs")
		}
		return nil
	case OpLoopMerge:
		return exact(a, 3)
	case OpSelectionMerge:
		return exact(a, 2)
	case OpLabel:
		return exact(a, 1)
	case OpBranch:
		return exact(a, 1)
	case OpBranchConditional:
		return exact(a, 3)
	case OpReturn:
		return exact(a, 0)
	case OpReturnValue:
		return exact(a, 1)
	case OpUnreachable:
		return exact(a, 0)
	default:
		return fmt.Errorf("unsupported opcode")
	}
}

// literalString decodes a SPIR-V null-terminated UTF-8 byte string. Tach names
// are ASCII/UTF-8 and require zero padding after the first terminator.
func literalString(words []uint32, start int) (string, int, error) {
	if start >= len(words) {
		return "", start, fmt.Errorf("missing literal string")
	}
	var buf []byte
	for wi := start; wi < len(words); wi++ {
		w := words[wi]
		for bi := 0; bi < 4; bi++ {
			c := byte(w >> (8 * bi))
			if c == 0 {
				// Remaining bytes in the terminating word are padding and must be zero.
				for bj := bi + 1; bj < 4; bj++ {
					if byte(w>>(8*bj)) != 0 {
						return "", start, fmt.Errorf("literal string has non-zero padding")
					}
				}
				return string(buf), wi + 1, nil
			}
			buf = append(buf, c)
		}
	}
	return "", start, fmt.Errorf("literal string is not null-terminated")
}

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
	m           *Module
	defs        map[uint32]int
	types       map[uint32]*typeInfo
	valueType   map[uint32]uint32
	constants   map[uint32]uint64
	decor       map[uint32]*decorationInfo
	functions   map[uint32]*functionInfo
	globalVars  map[uint32]uint32 // id -> storage
	pointerRoot map[uint32]uint32
	entryPoints map[uint32]string
	localSize   map[uint32][3]uint32
	extImports  map[uint32]string
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
		extImports: map[uint32]string{},
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
		OpTypePointer, OpTypeFunction, OpConstantTrue, OpConstantFalse, OpConstant, OpConstantComposite, OpVariable:
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
	case OpConstantTrue, OpConstantFalse, OpConstant, OpConstantComposite, OpFunction, OpFunctionParameter,
		OpFunctionCall, OpVariable, OpLoad, OpAccessChain, OpArrayLength, OpCompositeConstruct, OpCompositeExtract,
		OpConvertFToU, OpConvertFToS, OpConvertSToF, OpConvertUToF, OpBitcast, OpSNegate, OpFNegate,
		OpIAdd, OpFAdd, OpISub, OpFSub, OpIMul, OpFMul, OpUDiv, OpSDiv, OpFDiv, OpUMod, OpSRem, OpFRem,
		OpVectorTimesScalar, OpLogicalEqual, OpLogicalNotEqual, OpLogicalOr, OpLogicalAnd, OpLogicalNot, OpNot,
		OpShiftRightLogical, OpShiftRightArithmetic, OpShiftLeftLogical, OpBitwiseOr, OpBitwiseXor, OpBitwiseAnd,
		OpIEqual, OpINotEqual, OpUGreaterThan, OpSGreaterThan, OpUGreaterThanEqual, OpSGreaterThanEqual,
		OpULessThan, OpSLessThan, OpULessThanEqual, OpSLessThanEqual, OpFOrdEqual, OpFOrdNotEqual,
		OpFOrdLessThan, OpFOrdGreaterThan, OpFOrdLessThanEqual, OpFOrdGreaterThanEqual,
		OpAtomicLoad, OpAtomicExchange, OpAtomicIAdd, OpAtomicISub, OpAtomicSMin, OpAtomicUMin,
		OpAtomicSMax, OpAtomicUMax, OpAtomicAnd, OpAtomicOr, OpAtomicXor, OpPhi, OpExtInst, OpDot:
		return a[1]
	}
	return 0
}

func resultTypeID(in Instruction) uint32 {
	a := in.Operands
	switch in.Op {
	case OpConstantTrue, OpConstantFalse, OpConstant, OpConstantComposite, OpFunction, OpFunctionParameter,
		OpFunctionCall, OpVariable, OpLoad, OpAccessChain, OpArrayLength, OpCompositeConstruct, OpCompositeExtract,
		OpConvertFToU, OpConvertFToS, OpConvertSToF, OpConvertUToF, OpBitcast, OpSNegate, OpFNegate,
		OpIAdd, OpFAdd, OpISub, OpFSub, OpIMul, OpFMul, OpUDiv, OpSDiv, OpFDiv, OpUMod, OpSRem, OpFRem,
		OpVectorTimesScalar, OpLogicalEqual, OpLogicalNotEqual, OpLogicalOr, OpLogicalAnd, OpLogicalNot, OpNot,
		OpShiftRightLogical, OpShiftRightArithmetic, OpShiftLeftLogical, OpBitwiseOr, OpBitwiseXor, OpBitwiseAnd,
		OpIEqual, OpINotEqual, OpUGreaterThan, OpSGreaterThan, OpUGreaterThanEqual, OpSGreaterThanEqual,
		OpULessThan, OpSLessThan, OpULessThanEqual, OpSLessThanEqual, OpFOrdEqual, OpFOrdNotEqual,
		OpFOrdLessThan, OpFOrdGreaterThan, OpFOrdLessThanEqual, OpFOrdGreaterThanEqual,
		OpAtomicLoad, OpAtomicExchange, OpAtomicIAdd, OpAtomicISub, OpAtomicSMin, OpAtomicUMin,
		OpAtomicSMax, OpAtomicUMax, OpAtomicAnd, OpAtomicOr, OpAtomicXor, OpPhi, OpExtInst, OpDot:
		return a[0]
	}
	return 0
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
	caps := 0
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
			caps++
			if a[0] != CapabilityShader {
				return fmt.Errorf("only Shader capability is valid in Tach profile")
			}
		case OpExtInstImport:
			name, next, err := literalString(a, 1)
			if err != nil || next != len(a) {
				return fmt.Errorf("invalid extended instruction import")
			}
			if name != "GLSL.std.450" {
				return fmt.Errorf("Tach supports only GLSL.std.450 extended instructions, got %q", name)
			}
			for id, existing := range v.extImports {
				if existing == name && id != a[0] {
					return fmt.Errorf("GLSL.std.450 imported more than once")
				}
			}
			v.extImports[a[0]] = name
		case OpMemoryModel:
			memory++
			if a[0] != AddressingLogical || a[1] != MemoryGLSL450 {
				return fmt.Errorf("Tach requires Logical + GLSL450 memory model")
			}
		case OpEntryPoint:
			if a[0] != ExecutionModelGLCompute {
				return fmt.Errorf("Tach supports GLCompute entry points only")
			}
			name, _, _ := literalString(a, 2)
			if _, exists := v.entryPoints[a[1]]; exists {
				return fmt.Errorf("function %%%d declared as entry point more than once", a[1])
			}
			v.entryPoints[a[1]] = name
		case OpExecutionMode:
			if a[1] != ExecutionModeLocalSize {
				return fmt.Errorf("Tach supports LocalSize execution mode only")
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
				return fmt.Errorf("Tach integer type must be 32-bit signed/unsigned")
			}
			v.types[a[0]] = &typeInfo{kind: typeInt, width: a[1], signed: a[2] == 1}
		case OpTypeFloat:
			if a[1] != 32 {
				return fmt.Errorf("Tach float type must be f32")
			}
			v.types[a[0]] = &typeInfo{kind: typeFloat, width: a[1]}
		case OpTypeVector:
			if a[2] < 2 || a[2] > 4 {
				return fmt.Errorf("Tach vector width must be 2..4")
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
	if caps != 1 {
		return fmt.Errorf("Tach module must declare Shader capability exactly once")
	}
	if memory != 1 {
		return fmt.Errorf("Tach module must declare one memory model")
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
func same(a, b uint32) bool { return a == b }

func (v *validation) validateReferencesAndTypes() error {
	for _, in := range v.m.Instructions {
		a := in.Operands
		ctx := fmt.Sprintf("word %d %s", in.Offset, opName(in.Op))
		switch in.Op {
		case OpName:
			if err := v.requireID(a[0], ctx); err != nil {
				return err
			}
		case OpMemberName:
			t, err := v.requireType(a[0], ctx)
			if err != nil {
				return err
			}
			if t.kind != typeStruct || int(a[1]) >= len(t.members) {
				return fmt.Errorf("%s names invalid struct member %d", ctx, a[1])
			}
		case OpEntryPoint:
			if err := v.requireID(a[1], ctx); err != nil {
				return err
			}
			_, next, _ := literalString(a, 2)
			for _, id := range a[next:] {
				if err := v.requireID(id, ctx+" interface"); err != nil {
					return err
				}
			}
		case OpExecutionMode:
			if err := v.requireID(a[0], ctx); err != nil {
				return err
			}
		case OpDecorate:
			if err := v.requireID(a[0], ctx); err != nil {
				return err
			}
		case OpMemberDecorate:
			t, err := v.requireType(a[0], ctx)
			if err != nil {
				return err
			}
			if t.kind != typeStruct || int(a[1]) >= len(t.members) {
				return fmt.Errorf("%s decorates invalid member %d", ctx, a[1])
			}
		case OpTypeVector:
			t, err := v.requireType(a[1], ctx)
			if err != nil {
				return err
			}
			if t.kind != typeInt && t.kind != typeFloat {
				return fmt.Errorf("%s vector element must be integer/float", ctx)
			}
		case OpTypeArray:
			if _, err := v.requireType(a[1], ctx); err != nil {
				return err
			}
			lt, err := v.requireValue(a[2], ctx)
			if err != nil {
				return err
			}
			li := v.types[lt]
			if li == nil || li.kind != typeInt || li.signed || li.width != 32 {
				return fmt.Errorf("%s array length must be a u32 constant", ctx)
			}
			if _, ok := v.constants[a[2]]; !ok || v.constants[a[2]] == 0 {
				return fmt.Errorf("%s array length must be a positive constant", ctx)
			}
		case OpTypeRuntimeArray:
			if _, err := v.requireType(a[1], ctx); err != nil {
				return err
			}
		case OpTypeStruct:
			for _, id := range a[1:] {
				if _, err := v.requireType(id, ctx); err != nil {
					return err
				}
			}
		case OpTypePointer:
			if _, err := v.requireType(a[2], ctx); err != nil {
				return err
			}
		case OpTypeFunction:
			for _, id := range a[1:] {
				if _, err := v.requireType(id, ctx); err != nil {
					return err
				}
			}
		case OpConstantTrue, OpConstantFalse:
			t, err := v.requireType(a[0], ctx)
			if err != nil {
				return err
			}
			if t.kind != typeBool {
				return fmt.Errorf("%s requires bool result type", ctx)
			}
		case OpConstant:
			t, err := v.requireType(a[0], ctx)
			if err != nil {
				return err
			}
			if t.kind != typeInt && t.kind != typeFloat {
				return fmt.Errorf("%s scalar constant requires int/float type", ctx)
			}
		case OpConstantComposite:
			t, err := v.requireType(a[0], ctx)
			if err != nil {
				return err
			}
			if t.kind != typeVector && t.kind != typeStruct {
				return fmt.Errorf("%s composite constant has invalid type", ctx)
			}
			for _, id := range a[2:] {
				if _, err := v.requireValue(id, ctx); err != nil {
					return err
				}
			}
		case OpVariable:
			pt, err := v.requireType(a[0], ctx)
			if err != nil {
				return err
			}
			if pt.kind != typePointer || pt.storage != a[2] {
				return fmt.Errorf("%s variable result type/storage mismatch", ctx)
			}
		case OpFunction:
			if _, err := v.requireType(a[0], ctx); err != nil {
				return err
			}
			ft, err := v.requireType(a[3], ctx)
			if err != nil {
				return err
			}
			if ft.kind != typeFunction || ft.ret != a[0] {
				return fmt.Errorf("%s function type return mismatch", ctx)
			}
		case OpFunctionParameter:
			if _, err := v.requireType(a[0], ctx); err != nil {
				return err
			}
		case OpFunctionCall:
			rt, err := v.requireType(a[0], ctx)
			if err != nil {
				return err
			}
			_ = rt
			callee := v.functions[a[2]]
			if callee == nil {
				return fmt.Errorf("%s callee %%%d is not a function", ctx, a[2])
			}
			if callee.ret != a[0] {
				return fmt.Errorf("%s call result type does not match callee", ctx)
			}
			ft := v.types[callee.fnType]
			if ft == nil || ft.kind != typeFunction {
				return fmt.Errorf("%s callee function type missing", ctx)
			}
			if len(a[3:]) != len(ft.params) {
				return fmt.Errorf("%s call has %d args, want %d", ctx, len(a[3:]), len(ft.params))
			}
			for i, id := range a[3:] {
				vt, err := v.requireValue(id, ctx)
				if err != nil {
					return err
				}
				if vt != ft.params[i] {
					return fmt.Errorf("%s arg %d type mismatch", ctx, i)
				}
			}
			v.valueType[a[1]] = a[0]
		case OpExtInst:
			if err := v.validateExtInst(in); err != nil {
				return err
			}
		case OpDot:
			if err := v.validateDot(in); err != nil {
				return err
			}
		case OpLoad:
			rt, err := v.requireType(a[0], ctx)
			if err != nil {
				return err
			}
			_ = rt
			ptid, err := v.requireValue(a[2], ctx)
			if err != nil {
				return err
			}
			pt := v.types[ptid]
			if pt == nil || pt.kind != typePointer || pt.elem != a[0] {
				return fmt.Errorf("%s load pointer/result mismatch", ctx)
			}
			v.valueType[a[1]] = a[0]
		case OpStore:
			ptid, err := v.requireValue(a[0], ctx)
			if err != nil {
				return err
			}
			pt := v.types[ptid]
			if pt == nil || pt.kind != typePointer {
				return fmt.Errorf("%s store target is not pointer", ctx)
			}
			vt, err := v.requireValue(a[1], ctx)
			if err != nil {
				return err
			}
			if pt.elem != vt {
				return fmt.Errorf("%s store value type mismatch", ctx)
			}
			if pt.storage == StorageUniform {
				return fmt.Errorf("%s attempts to store through Uniform pointer", ctx)
			}
			root := v.pointerRoot[a[0]]
			if root != 0 && v.decoration(root).nonWritable {
				return fmt.Errorf("%s stores through NonWritable resource %%%d", ctx, root)
			}
		case OpAtomicLoad, OpAtomicStore, OpAtomicExchange, OpAtomicIAdd, OpAtomicISub,
			OpAtomicSMin, OpAtomicUMin, OpAtomicSMax, OpAtomicUMax, OpAtomicAnd, OpAtomicOr, OpAtomicXor:
			if err := v.validateAtomic(in); err != nil {
				return err
			}
		case OpControlBarrier:
			if err := v.validateBarrier(in); err != nil {
				return err
			}
		case OpAccessChain:
			if err := v.validateAccessChain(in); err != nil {
				return err
			}
		case OpArrayLength:
			if err := v.validateArrayLength(in); err != nil {
				return err
			}
		case OpCompositeConstruct:
			if err := v.validateCompositeConstruct(in); err != nil {
				return err
			}
		case OpCompositeExtract:
			if err := v.validateCompositeExtract(in); err != nil {
				return err
			}
		case OpConvertFToU, OpConvertFToS, OpConvertSToF, OpConvertUToF, OpBitcast:
			if err := v.validateConversion(in); err != nil {
				return err
			}
		case OpSNegate, OpFNegate, OpLogicalNot, OpNot:
			if err := v.validateUnary(in); err != nil {
				return err
			}
		case OpIAdd, OpFAdd, OpISub, OpFSub, OpIMul, OpFMul, OpUDiv, OpSDiv, OpFDiv, OpUMod, OpSRem, OpFRem,
			OpVectorTimesScalar, OpLogicalEqual, OpLogicalNotEqual, OpLogicalOr, OpLogicalAnd, OpIEqual, OpINotEqual,
			OpUGreaterThan, OpSGreaterThan, OpUGreaterThanEqual, OpSGreaterThanEqual, OpULessThan, OpSLessThan,
			OpULessThanEqual, OpSLessThanEqual, OpFOrdEqual, OpFOrdNotEqual, OpFOrdLessThan, OpFOrdGreaterThan,
			OpFOrdLessThanEqual, OpFOrdGreaterThanEqual, OpShiftRightLogical, OpShiftRightArithmetic,
			OpShiftLeftLogical, OpBitwiseOr, OpBitwiseXor, OpBitwiseAnd:
			if err := v.validateBinary(in); err != nil {
				return err
			}
		case OpPhi:
			if _, err := v.requireType(a[0], ctx); err != nil {
				return err
			}
			v.valueType[a[1]] = a[0]
			for i := 2; i < len(a); i += 2 {
				vt, err := v.requireValue(a[i], ctx)
				if err != nil {
					return err
				}
				if vt != a[0] {
					return fmt.Errorf("%s phi incoming type mismatch", ctx)
				}
				if err := v.requireID(a[i+1], ctx); err != nil {
					return err
				}
			}
		case OpLoopMerge:
			if err := v.requireID(a[0], ctx); err != nil {
				return err
			}
			if err := v.requireID(a[1], ctx); err != nil {
				return err
			}
			if a[2] != LoopControlNone {
				return fmt.Errorf("%s loop control outside Tach profile", ctx)
			}
		case OpSelectionMerge:
			if err := v.requireID(a[0], ctx); err != nil {
				return err
			}
			if a[1] != SelectionControlNone {
				return fmt.Errorf("%s selection control outside Tach profile", ctx)
			}
		case OpBranch:
			if err := v.requireID(a[0], ctx); err != nil {
				return err
			}
		case OpBranchConditional:
			ct, err := v.requireValue(a[0], ctx)
			if err != nil {
				return err
			}
			if v.types[ct] == nil || v.types[ct].kind != typeBool {
				return fmt.Errorf("%s condition is not bool", ctx)
			}
			if err := v.requireID(a[1], ctx); err != nil {
				return err
			}
			if err := v.requireID(a[2], ctx); err != nil {
				return err
			}
		case OpReturnValue:
			if _, err := v.requireValue(a[0], ctx); err != nil {
				return err
			}
		}
	}
	return nil
}

func (v *validation) validateAccessChain(in Instruction) error {
	a := in.Operands
	ctx := fmt.Sprintf("word %d OpAccessChain", in.Offset)
	rp, err := v.requireType(a[0], ctx)
	if err != nil {
		return err
	}
	if rp.kind != typePointer {
		return fmt.Errorf("%s result type is not pointer", ctx)
	}
	btid, err := v.requireValue(a[2], ctx)
	if err != nil {
		return err
	}
	bp := v.types[btid]
	if bp == nil || bp.kind != typePointer {
		return fmt.Errorf("%s base is not pointer", ctx)
	}
	if bp.storage != rp.storage {
		return fmt.Errorf("%s changes storage class", ctx)
	}
	cur := bp.elem
	for _, idxID := range a[3:] {
		it, err := v.requireValue(idxID, ctx)
		if err != nil {
			return err
		}
		itp := v.types[it]
		if itp == nil || itp.kind != typeInt {
			return fmt.Errorf("%s index is not integer", ctx)
		}
		t := v.types[cur]
		if t == nil {
			return fmt.Errorf("%s indexes non-type %%%d", ctx, cur)
		}
		switch t.kind {
		case typeStruct:
			c, ok := v.constants[idxID]
			if !ok {
				return fmt.Errorf("%s struct index must be constant", ctx)
			}
			idx := uint32(c)
			if int(idx) >= len(t.members) {
				return fmt.Errorf("%s struct index %d out of bounds", ctx, idx)
			}
			cur = t.members[idx]
		case typeArray, typeRuntimeArray:
			cur = t.elem
		case typeVector:
			cur = t.elem
		default:
			return fmt.Errorf("%s cannot index type %%%d", ctx, cur)
		}
	}
	if cur != rp.elem {
		return fmt.Errorf("%s result pointer pointee mismatch", ctx)
	}
	v.valueType[a[1]] = a[0]
	root := v.pointerRoot[a[2]]
	if root == 0 {
		root = a[2]
	}
	v.pointerRoot[a[1]] = root
	return nil
}

func (v *validation) validateArrayLength(in Instruction) error {
	a := in.Operands
	ctx := fmt.Sprintf("word %d OpArrayLength", in.Offset)
	rt, err := v.requireType(a[0], ctx)
	if err != nil {
		return err
	}
	if rt.kind != typeInt || rt.signed || rt.width != 32 {
		return fmt.Errorf("%s result must be u32", ctx)
	}
	ptid, err := v.requireValue(a[2], ctx)
	if err != nil {
		return err
	}
	pt := v.types[ptid]
	if pt == nil || pt.kind != typePointer {
		return fmt.Errorf("%s structure operand is not pointer", ctx)
	}
	st := v.types[pt.elem]
	if st == nil || st.kind != typeStruct {
		return fmt.Errorf("%s pointer does not point to struct", ctx)
	}
	member := a[3]
	if int(member) >= len(st.members) {
		return fmt.Errorf("%s member %d out of range", ctx, member)
	}
	at := v.types[st.members[member]]
	if at == nil || at.kind != typeRuntimeArray {
		return fmt.Errorf("%s member %d is not runtime array", ctx, member)
	}
	v.valueType[a[1]] = a[0]
	return nil
}

func (v *validation) constantU32(id uint32, ctx string) (uint32, error) {
	tid, err := v.requireValue(id, ctx)
	if err != nil {
		return 0, err
	}
	t := v.types[tid]
	if t == nil || t.kind != typeInt || t.signed || t.width != 32 {
		return 0, fmt.Errorf("%s requires u32 constant operand %%%d", ctx, id)
	}
	x, ok := v.constants[id]
	if !ok {
		return 0, fmt.Errorf("%s requires constant operand %%%d", ctx, id)
	}
	return uint32(x), nil
}

func (v *validation) validateAtomic(in Instruction) error {
	a := in.Operands
	ctx := fmt.Sprintf("word %d %s", in.Offset, opName(in.Op))
	var resultType, resultID, ptrID, scopeID, semanticsID, valueID uint32
	switch in.Op {
	case OpAtomicLoad:
		resultType, resultID, ptrID, scopeID, semanticsID = a[0], a[1], a[2], a[3], a[4]
	case OpAtomicStore:
		ptrID, scopeID, semanticsID, valueID = a[0], a[1], a[2], a[3]
	default:
		resultType, resultID, ptrID, scopeID, semanticsID, valueID = a[0], a[1], a[2], a[3], a[4], a[5]
	}
	ptid, err := v.requireValue(ptrID, ctx)
	if err != nil {
		return err
	}
	pt := v.types[ptid]
	if pt == nil || pt.kind != typePointer || (pt.storage != StorageWorkgroup && pt.storage != StorageStorageBuffer) {
		return fmt.Errorf("%s pointer must be Workgroup or StorageBuffer", ctx)
	}
	pointee := v.types[pt.elem]
	if pointee == nil || pointee.kind != typeInt || pointee.width != 32 {
		return fmt.Errorf("%s pointer must point to a 32-bit integer", ctx)
	}
	if in.Op != OpAtomicStore {
		rt, err := v.requireType(resultType, ctx)
		if err != nil {
			return err
		}
		if rt.kind != typeInt || resultType != pt.elem {
			return fmt.Errorf("%s result type must equal the pointed-to integer type", ctx)
		}
		v.valueType[resultID] = resultType
	}
	if valueID != 0 {
		vt, err := v.requireValue(valueID, ctx)
		if err != nil {
			return err
		}
		if vt != pt.elem {
			return fmt.Errorf("%s value type does not match pointer", ctx)
		}
	}
	scope, err := v.constantU32(scopeID, ctx+" scope")
	if err != nil {
		return err
	}
	wantScope := ScopeDevice
	if pt.storage == StorageWorkgroup {
		wantScope = ScopeWorkgroup
	}
	if scope != wantScope {
		return fmt.Errorf("%s scope=%d, Tach requires %d for storage class %d", ctx, scope, wantScope, pt.storage)
	}
	sem, err := v.constantU32(semanticsID, ctx+" semantics")
	if err != nil {
		return err
	}
	if sem != MemorySemanticsRelaxed {
		return fmt.Errorf("%s memory semantics=0x%x, Tach atomics require Relaxed", ctx, sem)
	}
	root := v.pointerRoot[ptrID]
	if root != 0 && v.decoration(root).nonWritable {
		return fmt.Errorf("%s accesses NonWritable resource %%%d", ctx, root)
	}
	if (in.Op == OpAtomicSMin || in.Op == OpAtomicSMax) && !pointee.signed {
		return fmt.Errorf("%s requires signed integer type", ctx)
	}
	if (in.Op == OpAtomicUMin || in.Op == OpAtomicUMax) && pointee.signed {
		return fmt.Errorf("%s requires unsigned integer type", ctx)
	}
	return nil
}

func (v *validation) validateBarrier(in Instruction) error {
	a := in.Operands
	ctx := fmt.Sprintf("word %d OpControlBarrier", in.Offset)
	exec, err := v.constantU32(a[0], ctx+" execution scope")
	if err != nil {
		return err
	}
	mem, err := v.constantU32(a[1], ctx+" memory scope")
	if err != nil {
		return err
	}
	sem, err := v.constantU32(a[2], ctx+" semantics")
	if err != nil {
		return err
	}
	if exec != ScopeWorkgroup || mem != ScopeWorkgroup {
		return fmt.Errorf("%s requires Workgroup execution and memory scopes", ctx)
	}
	wg := MemorySemanticsAcquireRelease | MemorySemanticsWorkgroupMemory
	storage := MemorySemanticsAcquireRelease | MemorySemanticsUniformMemory
	if sem != wg && sem != storage {
		return fmt.Errorf("%s semantics=0x%x outside Tach barrier profile", ctx, sem)
	}
	return nil
}

func (v *validation) validateCompositeConstruct(in Instruction) error {
	a := in.Operands
	ctx := fmt.Sprintf("word %d OpCompositeConstruct", in.Offset)
	t, err := v.requireType(a[0], ctx)
	if err != nil {
		return err
	}
	want := []uint32{}
	switch t.kind {
	case typeVector:
		for i := uint32(0); i < t.lanes; i++ {
			want = append(want, t.elem)
		}
	case typeStruct:
		want = t.members
	default:
		return fmt.Errorf("%s result type is not constructible composite", ctx)
	}
	if len(a[2:]) != len(want) {
		return fmt.Errorf("%s has %d constituents, want %d", ctx, len(a[2:]), len(want))
	}
	for i, id := range a[2:] {
		vt, err := v.requireValue(id, ctx)
		if err != nil {
			return err
		}
		if vt != want[i] {
			return fmt.Errorf("%s constituent %d type mismatch", ctx, i)
		}
	}
	v.valueType[a[1]] = a[0]
	return nil
}

func (v *validation) validateCompositeExtract(in Instruction) error {
	a := in.Operands
	ctx := fmt.Sprintf("word %d OpCompositeExtract", in.Offset)
	if _, err := v.requireType(a[0], ctx); err != nil {
		return err
	}
	bt, err := v.requireValue(a[2], ctx)
	if err != nil {
		return err
	}
	cur := bt
	for _, idx := range a[3:] {
		t := v.types[cur]
		if t == nil {
			return fmt.Errorf("%s indexes invalid type", ctx)
		}
		switch t.kind {
		case typeVector:
			if idx >= t.lanes {
				return fmt.Errorf("%s vector index out of range", ctx)
			}
			cur = t.elem
		case typeStruct:
			if int(idx) >= len(t.members) {
				return fmt.Errorf("%s struct index out of range", ctx)
			}
			cur = t.members[idx]
		default:
			return fmt.Errorf("%s cannot extract from this type", ctx)
		}
	}
	if cur != a[0] {
		return fmt.Errorf("%s result type mismatch", ctx)
	}
	v.valueType[a[1]] = a[0]
	return nil
}

func baseScalar(t *typeInfo, all map[uint32]*typeInfo) *typeInfo {
	if t != nil && t.kind == typeVector {
		return all[t.elem]
	}
	return t
}
func sameShape(a, b *typeInfo) bool {
	if a == nil || b == nil {
		return false
	}
	if a.kind == typeVector || b.kind == typeVector {
		return a.kind == b.kind && a.lanes == b.lanes
	}
	return true
}

func (v *validation) validateExtInst(in Instruction) error {
	a := in.Operands
	ctx := fmt.Sprintf("word %d OpExtInst", in.Offset)
	rt, err := v.requireType(a[0], ctx)
	if err != nil {
		return err
	}
	if v.extImports[a[2]] != "GLSL.std.450" {
		if err := v.requireID(a[2], ctx+" instruction set"); err != nil {
			return err
		}
		return fmt.Errorf("%s instruction set %%%d is not GLSL.std.450", ctx, a[2])
	}
	argIDs := a[4:]
	argTypes := make([]uint32, len(argIDs))
	for i, id := range argIDs {
		tid, err := v.requireValue(id, ctx)
		if err != nil {
			return err
		}
		argTypes[i] = tid
	}
	need := func(n int) error {
		if len(argIDs) != n {
			return fmt.Errorf("%s GLSL.std.450 instruction %d has %d args, want %d", ctx, a[3], len(argIDs), n)
		}
		return nil
	}
	allResultType := func() bool {
		for _, tid := range argTypes {
			if tid != a[0] {
				return false
			}
		}
		return true
	}
	base := baseScalar(rt, v.types)
	if base == nil {
		return fmt.Errorf("%s has invalid result type", ctx)
	}

	switch a[3] {
	case GLSL450FAbs:
		if err := need(1); err != nil {
			return err
		}
		if !allResultType() || base.kind != typeFloat {
			return fmt.Errorf("%s FAbs requires matching f32 scalar/vector operand and result", ctx)
		}
	case GLSL450SAbs:
		if err := need(1); err != nil {
			return err
		}
		if !allResultType() || base.kind != typeInt || !base.signed {
			return fmt.Errorf("%s SAbs requires matching signed i32 scalar/vector operand and result", ctx)
		}
	case GLSL450Trunc, GLSL450Floor, GLSL450Ceil, GLSL450Sin, GLSL450Cos, GLSL450Tan,
		GLSL450Exp, GLSL450Log, GLSL450Exp2, GLSL450Log2, GLSL450Sqrt, GLSL450InverseSqrt:
		if err := need(1); err != nil {
			return err
		}
		if !allResultType() || base.kind != typeFloat {
			return fmt.Errorf("%s floating unary intrinsic requires matching f32 scalar/vector operand and result", ctx)
		}
	case GLSL450Pow:
		if err := need(2); err != nil {
			return err
		}
		if !allResultType() || base.kind != typeFloat {
			return fmt.Errorf("%s Pow requires matching f32 scalar/vector operands and result", ctx)
		}
	case GLSL450UMin, GLSL450UMax:
		if err := need(2); err != nil {
			return err
		}
		if !allResultType() || base.kind != typeInt || base.signed {
			return fmt.Errorf("%s unsigned min/max requires matching u32 scalar/vector operands and result", ctx)
		}
	case GLSL450SMin, GLSL450SMax:
		if err := need(2); err != nil {
			return err
		}
		if !allResultType() || base.kind != typeInt || !base.signed {
			return fmt.Errorf("%s signed min/max requires matching i32 scalar/vector operands and result", ctx)
		}
	case GLSL450UClamp:
		if err := need(3); err != nil {
			return err
		}
		if !allResultType() || base.kind != typeInt || base.signed {
			return fmt.Errorf("%s UClamp requires matching u32 scalar/vector operands and result", ctx)
		}
	case GLSL450SClamp:
		if err := need(3); err != nil {
			return err
		}
		if !allResultType() || base.kind != typeInt || !base.signed {
			return fmt.Errorf("%s SClamp requires matching i32 scalar/vector operands and result", ctx)
		}
	case GLSL450Length:
		if err := need(1); err != nil {
			return err
		}
		at := v.types[argTypes[0]]
		if rt.kind != typeFloat || at == nil || at.kind != typeVector || at.elem != a[0] || baseScalar(at, v.types).kind != typeFloat {
			return fmt.Errorf("%s Length requires f32 vector input and f32 component result", ctx)
		}
	case GLSL450Distance:
		if err := need(2); err != nil {
			return err
		}
		at := v.types[argTypes[0]]
		if argTypes[0] != argTypes[1] || rt.kind != typeFloat || at == nil || at.kind != typeVector || at.elem != a[0] || baseScalar(at, v.types).kind != typeFloat {
			return fmt.Errorf("%s Distance requires matching f32 vectors and f32 component result", ctx)
		}
	case GLSL450Cross:
		if err := need(2); err != nil {
			return err
		}
		if !allResultType() || rt.kind != typeVector || rt.lanes != 3 || base.kind != typeFloat {
			return fmt.Errorf("%s Cross requires matching vec3f operands and result", ctx)
		}
	case GLSL450Normalize:
		if err := need(1); err != nil {
			return err
		}
		if !allResultType() || rt.kind != typeVector || base.kind != typeFloat {
			return fmt.Errorf("%s Normalize requires matching f32 vector operand and result", ctx)
		}
	default:
		return fmt.Errorf("%s GLSL.std.450 instruction %d is outside Tach's profile", ctx, a[3])
	}
	v.valueType[a[1]] = a[0]
	return nil
}

func (v *validation) validateDot(in Instruction) error {
	a := in.Operands
	ctx := fmt.Sprintf("word %d OpDot", in.Offset)
	rt, err := v.requireType(a[0], ctx)
	if err != nil {
		return err
	}
	ltID, err := v.requireValue(a[2], ctx)
	if err != nil {
		return err
	}
	rtID, err := v.requireValue(a[3], ctx)
	if err != nil {
		return err
	}
	lt := v.types[ltID]
	if rt.kind != typeFloat || ltID != rtID || lt == nil || lt.kind != typeVector || lt.elem != a[0] || baseScalar(lt, v.types).kind != typeFloat {
		return fmt.Errorf("%s requires matching f32 vectors and returns their f32 component type", ctx)
	}
	v.valueType[a[1]] = a[0]
	return nil
}

func (v *validation) validateConversion(in Instruction) error {
	a := in.Operands
	ctx := fmt.Sprintf("word %d %s", in.Offset, opName(in.Op))
	dst, err := v.requireType(a[0], ctx)
	if err != nil {
		return err
	}
	srcID, err := v.requireValue(a[2], ctx)
	if err != nil {
		return err
	}
	src := v.types[srcID]
	ds := baseScalar(dst, v.types)
	ss := baseScalar(src, v.types)
	if !sameShape(dst, src) {
		return fmt.Errorf("%s shape mismatch", ctx)
	}
	ok := false
	switch in.Op {
	case OpConvertFToU:
		ok = ss.kind == typeFloat && ds.kind == typeInt && !ds.signed
	case OpConvertFToS:
		ok = ss.kind == typeFloat && ds.kind == typeInt && ds.signed
	case OpConvertSToF:
		ok = ss.kind == typeInt && ss.signed && ds.kind == typeFloat
	case OpConvertUToF:
		ok = ss.kind == typeInt && !ss.signed && ds.kind == typeFloat
	case OpBitcast:
		ok = ss.kind == typeInt && ds.kind == typeInt && ss.width == ds.width
	}
	if !ok {
		return fmt.Errorf("%s source/result types are incompatible", ctx)
	}
	v.valueType[a[1]] = a[0]
	return nil
}

func (v *validation) validateUnary(in Instruction) error {
	a := in.Operands
	ctx := fmt.Sprintf("word %d %s", in.Offset, opName(in.Op))
	rt, err := v.requireType(a[0], ctx)
	if err != nil {
		return err
	}
	xt, err := v.requireValue(a[2], ctx)
	if err != nil {
		return err
	}
	if xt != a[0] {
		return fmt.Errorf("%s operand/result type mismatch", ctx)
	}
	base := baseScalar(rt, v.types)
	ok := false
	switch in.Op {
	case OpSNegate:
		ok = base.kind == typeInt && base.signed
	case OpFNegate:
		ok = base.kind == typeFloat
	case OpLogicalNot:
		ok = base.kind == typeBool
	case OpNot:
		ok = base.kind == typeInt
	}
	if !ok {
		return fmt.Errorf("%s invalid operand type", ctx)
	}
	v.valueType[a[1]] = a[0]
	return nil
}

func (v *validation) validateBinary(in Instruction) error {
	a := in.Operands
	ctx := fmt.Sprintf("word %d %s", in.Offset, opName(in.Op))
	rt, err := v.requireType(a[0], ctx)
	if err != nil {
		return err
	}
	lt, err := v.requireValue(a[2], ctx)
	if err != nil {
		return err
	}
	rr, err := v.requireValue(a[3], ctx)
	if err != nil {
		return err
	}
	l := v.types[lt]
	r := v.types[rr]
	if in.Op == OpVectorTimesScalar {
		if rt.kind != typeVector || lt != a[0] || r == nil || r.kind != typeInt && r.kind != typeFloat || rt.elem != rr {
			return fmt.Errorf("%s requires vector and matching scalar", ctx)
		}
		v.valueType[a[1]] = a[0]
		return nil
	}
	if in.Op == OpBitwiseAnd || in.Op == OpBitwiseOr || in.Op == OpBitwiseXor {
		base := baseScalar(rt, v.types)
		if lt != a[0] || rr != a[0] || base == nil || base.kind != typeInt {
			return fmt.Errorf("%s requires matching integer scalar/vector operands and result", ctx)
		}
		v.valueType[a[1]] = a[0]
		return nil
	}
	if in.Op == OpShiftRightLogical || in.Op == OpShiftRightArithmetic || in.Op == OpShiftLeftLogical {
		leftBase := baseScalar(rt, v.types)
		rightBase := baseScalar(r, v.types)
		if lt != a[0] || leftBase == nil || leftBase.kind != typeInt || rightBase == nil || rightBase.kind != typeInt || rightBase.signed || !sameShape(rt, r) {
			return fmt.Errorf("%s requires integer value shifted by shape-matching unsigned count", ctx)
		}
		if in.Op == OpShiftRightArithmetic && !leftBase.signed {
			return fmt.Errorf("%s requires signed shifted value in Tach's profile", ctx)
		}
		if in.Op == OpShiftRightLogical && leftBase.signed {
			return fmt.Errorf("%s requires unsigned shifted value in Tach's profile", ctx)
		}
		v.valueType[a[1]] = a[0]
		return nil
	}
	comparison := in.Op >= OpLogicalEqual && in.Op <= OpFOrdGreaterThanEqual
	if comparison {
		if rt.kind != typeBool {
			return fmt.Errorf("%s comparison result must be bool", ctx)
		}
		if lt != rr {
			return fmt.Errorf("%s comparison operand types differ", ctx)
		}
		base := baseScalar(l, v.types)
		ok := false
		switch in.Op {
		case OpLogicalEqual, OpLogicalNotEqual, OpLogicalOr, OpLogicalAnd:
			ok = base.kind == typeBool
		case OpIEqual, OpINotEqual:
			ok = base.kind == typeInt
		case OpUGreaterThan, OpUGreaterThanEqual, OpULessThan, OpULessThanEqual:
			ok = base.kind == typeInt && !base.signed
		case OpSGreaterThan, OpSGreaterThanEqual, OpSLessThan, OpSLessThanEqual:
			ok = base.kind == typeInt && base.signed
		default:
			ok = base.kind == typeFloat
		}
		if !ok {
			return fmt.Errorf("%s comparison opcode/type mismatch", ctx)
		}
	} else {
		if lt != a[0] || rr != a[0] {
			return fmt.Errorf("%s arithmetic operands/result must share a type", ctx)
		}
		base := baseScalar(rt, v.types)
		ok := false
		switch in.Op {
		case OpIAdd, OpISub, OpIMul:
			ok = base.kind == typeInt
		case OpFAdd, OpFSub, OpFMul, OpFDiv, OpFRem:
			ok = base.kind == typeFloat
		case OpUDiv, OpUMod:
			ok = base.kind == typeInt && !base.signed
		case OpSDiv, OpSRem:
			ok = base.kind == typeInt && base.signed
		}
		if !ok {
			return fmt.Errorf("%s arithmetic opcode/type mismatch", ctx)
		}
	}
	v.valueType[a[1]] = a[0]
	return nil
}

func (v *validation) validateDecorationsAndABI() error {
	// Type-level decorations and Tach's host ABI layout.
	memo := map[uint32]abiLayout{}
	visiting := map[uint32]bool{}
	for id, t := range v.types {
		if t.kind == typeArray || t.kind == typeRuntimeArray {
			d := v.decoration(id)
			if d.arrayStride == nil {
				return fmt.Errorf("array type %%%d lacks ArrayStride", id)
			}
			el, err := v.abiOf(t.elem, memo, visiting)
			if err != nil {
				return err
			}
			want := roundUp(el.align, el.size)
			if *d.arrayStride != want {
				return fmt.Errorf("array %%%d ArrayStride=%d, Tach ABI requires %d", id, *d.arrayStride, want)
			}
		}
		if t.kind == typeStruct {
			l, err := v.abiOf(id, memo, visiting)
			if err != nil {
				return err
			}
			_ = l
		}
	}

	pairs := map[[2]uint32]uint32{}
	for id, storage := range v.globalVars {
		d := v.decoration(id)
		vt := v.types[v.valueType[id]]
		if vt == nil || vt.kind != typePointer {
			return fmt.Errorf("global variable %%%d lacks pointer type", id)
		}
		switch storage {
		case StorageInput:
			if d.builtin == nil {
				return fmt.Errorf("Input variable %%%d lacks BuiltIn decoration", id)
			}
			if d.binding != nil || d.set != nil {
				return fmt.Errorf("Input builtin %%%d cannot have descriptor decorations", id)
			}
			switch *d.builtin {
			case BuiltInNumWorkgroups, BuiltInWorkgroupID, BuiltInLocalInvocationID, BuiltInGlobalInvocationID, BuiltInLocalInvocationIndex:
			default:
				return fmt.Errorf("Input %%%d uses builtin %d outside Tach profile", id, *d.builtin)
			}
		case StorageUniform, StorageStorageBuffer:
			if d.binding == nil || d.set == nil {
				return fmt.Errorf("descriptor variable %%%d requires DescriptorSet and Binding", id)
			}
			pair := [2]uint32{*d.set, *d.binding}
			if prev, ok := pairs[pair]; ok {
				return fmt.Errorf("descriptor variables %%%d and %%%d share set=%d binding=%d", prev, id, pair[0], pair[1])
			}
			pairs[pair] = id
			st := v.types[vt.elem]
			if st == nil || st.kind != typeStruct || !v.decoration(vt.elem).block {
				return fmt.Errorf("descriptor variable %%%d must point to Block struct", id)
			}
			if storage == StorageUniform && containsRuntime(vt.elem, v.types, map[uint32]bool{}) {
				return fmt.Errorf("uniform descriptor %%%d contains runtime array", id)
			}
		case StorageWorkgroup:
			if d.builtin != nil || d.binding != nil || d.set != nil || d.nonWritable {
				return fmt.Errorf("Workgroup variable %%%d has invalid interface/descriptor decoration", id)
			}
		default:
			return fmt.Errorf("global variable %%%d storage class %d outside Tach profile", id, storage)
		}
	}
	return nil
}

type abiLayout struct {
	size, align, stride uint32
	runtime             bool
}

func roundUp(a, v uint32) uint32 {
	if a == 0 {
		return v
	}
	return (v + a - 1) &^ (a - 1)
}
func max32(a, b uint32) uint32 {
	if a > b {
		return a
	}
	return b
}
func (v *validation) abiOf(id uint32, memo map[uint32]abiLayout, vis map[uint32]bool) (abiLayout, error) {
	if l, ok := memo[id]; ok {
		return l, nil
	}
	if vis[id] {
		return abiLayout{}, fmt.Errorf("recursive SPIR-V type %%%d", id)
	}
	vis[id] = true
	defer delete(vis, id)
	t := v.types[id]
	if t == nil {
		return abiLayout{}, fmt.Errorf("ABI references unknown type %%%d", id)
	}
	var l abiLayout
	switch t.kind {
	case typeInt, typeFloat:
		l = abiLayout{size: 4, align: 4}
	case typeVector:
		e := v.types[t.elem]
		if e == nil || (e.kind != typeInt && e.kind != typeFloat) {
			return l, fmt.Errorf("vector %%%d has non-host element", id)
		}
		switch t.lanes {
		case 2:
			l = abiLayout{size: 8, align: 8}
		case 3:
			l = abiLayout{size: 12, align: 16}
		case 4:
			l = abiLayout{size: 16, align: 16}
		default:
			return l, fmt.Errorf("invalid vector width")
		}
	case typeRuntimeArray:
		e, err := v.abiOf(t.elem, memo, vis)
		if err != nil {
			return l, err
		}
		l = abiLayout{align: e.align, stride: roundUp(e.align, e.size), runtime: true}
	case typeArray:
		e, err := v.abiOf(t.elem, memo, vis)
		if err != nil {
			return l, err
		}
		count, ok := v.constants[t.length]
		if !ok || count == 0 {
			return l, fmt.Errorf("array %%%d has invalid length constant", id)
		}
		stride := roundUp(e.align, e.size)
		l = abiLayout{align: e.align, stride: stride, size: stride * uint32(count)}
	case typeStruct:
		d := v.decoration(id)
		align := uint32(16)
		off := uint32(0)
		runtime := false
		if len(d.offsets) != len(t.members) {
			return l, fmt.Errorf("struct %%%d has %d Offset decorations for %d members", id, len(d.offsets), len(t.members))
		}
		for i, mid := range t.members {
			ml, err := v.abiOf(mid, memo, vis)
			if err != nil {
				return l, err
			}
			if runtime {
				return l, fmt.Errorf("runtime array in struct %%%d is not final", id)
			}
			req := ml.align
			mt := v.types[mid]
			if mt.kind == typeStruct {
				req = max32(req, 16)
			}
			want := roundUp(req, off)
			got, ok := d.offsets[uint32(i)]
			if !ok || got != want {
				return l, fmt.Errorf("struct %%%d member %d Offset=%d, Tach ABI requires %d", id, i, got, want)
			}
			if ml.runtime {
				runtime = true
			} else {
				sz := ml.size
				if mt.kind == typeStruct {
					sz = roundUp(16, sz)
				}
				off = want + sz
			}
			align = max32(align, req)
		}
		l = abiLayout{align: align, runtime: runtime}
		if runtime {
			l.size = off
		} else {
			l.size = roundUp(align, off)
		}
	default:
		return l, fmt.Errorf("type %%%d is not in Tach host ABI", id)
	}
	memo[id] = l
	return l, nil
}
func containsRuntime(id uint32, ts map[uint32]*typeInfo, seen map[uint32]bool) bool {
	if seen[id] {
		return false
	}
	seen[id] = true
	t := ts[id]
	if t == nil {
		return false
	}
	if t.kind == typeRuntimeArray {
		return true
	}
	if t.kind == typeArray {
		return containsRuntime(t.elem, ts, seen)
	}
	if t.kind == typeStruct {
		for _, m := range t.members {
			if containsRuntime(m, ts, seen) {
				return true
			}
		}
	}
	return false
}

func isTerminator(op Op) bool {
	switch op {
	case OpBranch, OpBranchConditional, OpReturn, OpReturnValue, OpUnreachable:
		return true
	}
	return false
}

func (v *validation) validateFunctions() error {
	// Reconstruct block instruction lists including labels' body operations, CFG,
	// phi predecessor sets, merge placement, function signatures, and SSA dominance.
	funcOfInst := map[int]*functionInfo{}
	blockOfInst := map[int]uint32{}
	var cur *functionInfo
	var block uint32
	for i, in := range v.m.Instructions {
		switch in.Op {
		case OpFunction:
			cur = v.functions[in.Operands[1]]
			block = 0
		case OpLabel:
			block = in.Operands[0]
			if cur != nil {
				funcOfInst[i] = cur
				blockOfInst[i] = block
			}
		case OpFunctionEnd:
			cur = nil
			block = 0
		default:
			if cur != nil {
				funcOfInst[i] = cur
				blockOfInst[i] = block
			}
		}
	}
	for _, f := range v.functions {
		ft := v.types[f.fnType]
		if ft == nil || ft.kind != typeFunction {
			return fmt.Errorf("function %%%d has invalid function type", f.id)
		}
		if ft.ret != f.ret || len(ft.params) != len(f.params) {
			return fmt.Errorf("function %%%d signature mismatches OpTypeFunction", f.id)
		}
		for i, p := range f.params {
			if v.valueType[p] != ft.params[i] {
				return fmt.Errorf("function %%%d parameter %d type mismatch", f.id, i)
			}
		}
		for _, label := range f.order {
			bl := f.blocks[label]
			if len(bl.insts) == 0 {
				return fmt.Errorf("function %%%d block %%%d is empty", f.id, label)
			}
			seenNonPhi := false
			terminated := false
			for pos, idx := range bl.insts {
				in := v.m.Instructions[idx]
				if terminated {
					return fmt.Errorf("block %%%d has instruction after terminator", label)
				}
				if in.Op == OpPhi {
					if seenNonPhi {
						return fmt.Errorf("block %%%d has OpPhi after non-phi instruction", label)
					}
				} else {
					seenNonPhi = true
				}
				if in.Op == OpSelectionMerge || in.Op == OpLoopMerge {
					if pos+1 >= len(bl.insts) {
						return fmt.Errorf("block %%%d merge instruction lacks following branch", label)
					}
					next := v.m.Instructions[bl.insts[pos+1]].Op
					if next != OpBranch && next != OpBranchConditional {
						return fmt.Errorf("block %%%d merge must immediately precede branch", label)
					}
				}
				if isTerminator(in.Op) {
					terminated = true
					switch in.Op {
					case OpBranch:
						bl.succ = []uint32{in.Operands[0]}
					case OpBranchConditional:
						bl.succ = []uint32{in.Operands[1], in.Operands[2]}
					}
				}
			}
			if !terminated {
				return fmt.Errorf("function %%%d block %%%d lacks terminator", f.id, label)
			}
		}
		for _, label := range f.order {
			for _, dst := range f.blocks[label].succ {
				db := f.blocks[dst]
				if db == nil {
					return fmt.Errorf("function %%%d branches to label %%%d outside function", f.id, dst)
				}
				db.pred = append(db.pred, label)
			}
		}
		for _, label := range f.order {
			bl := f.blocks[label]
			sort.Slice(bl.pred, func(i, j int) bool { return bl.pred[i] < bl.pred[j] })
			for _, idx := range bl.phis {
				a := v.m.Instructions[idx].Operands
				var incoming []uint32
				for i := 3; i < len(a); i += 2 {
					incoming = append(incoming, a[i])
				}
				sort.Slice(incoming, func(i, j int) bool { return incoming[i] < incoming[j] })
				if !equalU32(incoming, bl.pred) {
					return fmt.Errorf("function %%%d block %%%d phi predecessors %v do not match CFG predecessors %v", f.id, label, incoming, bl.pred)
				}
			}
		}
		if err := v.validateReturnTypes(f); err != nil {
			return err
		}
		if err := v.validateDominance(f, funcOfInst, blockOfInst); err != nil {
			return err
		}
	}
	return nil
}
func equalU32(a, b []uint32) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func (v *validation) validateReturnTypes(f *functionInfo) error {
	rt := v.types[f.ret]
	for _, label := range f.order {
		for _, idx := range f.blocks[label].insts {
			in := v.m.Instructions[idx]
			if in.Op == OpReturn && rt.kind != typeVoid {
				return fmt.Errorf("function %%%d returns void but return type is non-void", f.id)
			}
			if in.Op == OpReturnValue {
				if rt.kind == typeVoid {
					return fmt.Errorf("void function %%%d uses OpReturnValue", f.id)
				}
				vt := v.valueType[in.Operands[0]]
				if vt != f.ret {
					return fmt.Errorf("function %%%d return value type mismatch", f.id)
				}
			}
		}
	}
	return nil
}

func (v *validation) validateDominance(f *functionInfo, funcOfInst map[int]*functionInfo, blockOfInst map[int]uint32) error {
	if len(f.order) == 0 {
		return nil
	}
	entry := f.order[0]
	reachable := map[uint32]bool{}
	var walk func(uint32)
	walk = func(x uint32) {
		if reachable[x] {
			return
		}
		reachable[x] = true
		for _, y := range f.blocks[x].succ {
			walk(y)
		}
	}
	walk(entry)
	// Standard iterative dominator sets.
	dom := map[uint32]map[uint32]bool{}
	for _, l := range f.order {
		dom[l] = map[uint32]bool{}
		if l == entry {
			dom[l][l] = true
		} else if reachable[l] {
			for _, x := range f.order {
				if reachable[x] {
					dom[l][x] = true
				}
			}
		}
	}
	changed := true
	for changed {
		changed = false
		for _, l := range f.order {
			if l == entry || !reachable[l] {
				continue
			}
			preds := f.blocks[l].pred
			var next map[uint32]bool
			for _, p := range preds {
				if !reachable[p] {
					continue
				}
				if next == nil {
					next = cloneSet(dom[p])
				} else {
					for x := range next {
						if !dom[p][x] {
							delete(next, x)
						}
					}
				}
			}
			if next == nil {
				next = map[uint32]bool{}
			}
			next[l] = true
			if !setEq(next, dom[l]) {
				dom[l] = next
				changed = true
			}
		}
	}
	// Local definition location/order. Function parameters use block 0 and globals
	// are absent from this map, both dominating every function block.
	type loc struct {
		block uint32
		pos   int
	}
	defs := map[uint32]loc{}
	for _, p := range f.params {
		defs[p] = loc{}
	}
	for _, l := range f.order {
		for pos, idx := range f.blocks[l].insts {
			if id := resultID(v.m.Instructions[idx]); id != 0 {
				defs[id] = loc{l, pos}
			}
		}
	}
	// Labels and function id are not ordinary values.
	for _, l := range f.order {
		for pos, idx := range f.blocks[l].insts {
			in := v.m.Instructions[idx]
			for _, use := range valueUses(in) {
				d, local := defs[use]
				if !local {
					continue
				}
				if d.block == 0 {
					continue
				}
				if in.Op == OpPhi {
					continue
				}
				if d.block == l {
					if d.pos >= pos {
						return fmt.Errorf("function %%%d value %%%d does not precede its use in block %%%d", f.id, use, l)
					}
				} else if !dom[l][d.block] {
					return fmt.Errorf("function %%%d value %%%d defined in block %%%d does not dominate use in %%%d", f.id, use, d.block, l)
				}
			}
			if in.Op == OpPhi {
				a := in.Operands
				for j := 2; j < len(a); j += 2 {
					use, pred := a[j], a[j+1]
					d, local := defs[use]
					if !local || d.block == 0 {
						continue
					}
					if d.block == pred {
						continue
					}
					if !dom[pred][d.block] {
						return fmt.Errorf("function %%%d phi value %%%d does not dominate predecessor %%%d", f.id, use, pred)
					}
				}
			}
		}
	}
	return nil
}
func cloneSet(s map[uint32]bool) map[uint32]bool {
	r := map[uint32]bool{}
	for x := range s {
		r[x] = true
	}
	return r
}
func setEq(a, b map[uint32]bool) bool {
	if len(a) != len(b) {
		return false
	}
	for x := range a {
		if !b[x] {
			return false
		}
	}
	return true
}

// valueUses returns only ordinary SSA/object value operands; type ids, function
// ids, labels, and decoration targets are intentionally excluded.
func valueUses(in Instruction) []uint32 {
	a := in.Operands
	switch in.Op {
	case OpFunctionCall:
		return append([]uint32(nil), a[3:]...)
	case OpExtInst:
		return append([]uint32(nil), a[4:]...)
	case OpDot:
		return []uint32{a[2], a[3]}
	case OpLoad:
		return []uint32{a[2]}
	case OpStore:
		return []uint32{a[0], a[1]}
	case OpAtomicLoad:
		return []uint32{a[2], a[3], a[4]}
	case OpAtomicStore:
		return []uint32{a[0], a[1], a[2], a[3]}
	case OpAtomicExchange, OpAtomicIAdd, OpAtomicISub, OpAtomicSMin, OpAtomicUMin, OpAtomicSMax, OpAtomicUMax, OpAtomicAnd, OpAtomicOr, OpAtomicXor:
		return []uint32{a[2], a[3], a[4], a[5]}
	case OpControlBarrier:
		return []uint32{a[0], a[1], a[2]}
	case OpAccessChain:
		return append([]uint32{a[2]}, a[3:]...)
	case OpArrayLength:
		return []uint32{a[2]}
	case OpCompositeConstruct:
		return append([]uint32(nil), a[2:]...)
	case OpCompositeExtract:
		return []uint32{a[2]}
	case OpConvertFToU, OpConvertFToS, OpConvertSToF, OpConvertUToF, OpBitcast, OpSNegate, OpFNegate, OpLogicalNot, OpNot:
		return []uint32{a[2]}
	case OpIAdd, OpFAdd, OpISub, OpFSub, OpIMul, OpFMul, OpUDiv, OpSDiv, OpFDiv, OpUMod, OpSRem, OpFRem, OpVectorTimesScalar, OpLogicalEqual, OpLogicalNotEqual, OpLogicalOr, OpLogicalAnd, OpIEqual, OpINotEqual, OpUGreaterThan, OpSGreaterThan, OpUGreaterThanEqual, OpSGreaterThanEqual, OpULessThan, OpSLessThan, OpULessThanEqual, OpSLessThanEqual, OpFOrdEqual, OpFOrdNotEqual, OpFOrdLessThan, OpFOrdGreaterThan, OpFOrdLessThanEqual, OpFOrdGreaterThanEqual, OpShiftRightLogical, OpShiftRightArithmetic, OpShiftLeftLogical, OpBitwiseOr, OpBitwiseXor, OpBitwiseAnd:
		return []uint32{a[2], a[3]}
	case OpBranchConditional:
		return []uint32{a[0]}
	case OpReturnValue:
		return []uint32{a[0]}
	}
	return nil
}

func (v *validation) validateEntryPoints() error {
	for fid, name := range v.entryPoints {
		f := v.functions[fid]
		if f == nil {
			return fmt.Errorf("entry point %q references non-function %%%d", name, fid)
		}
		rt := v.types[f.ret]
		if rt == nil || rt.kind != typeVoid {
			return fmt.Errorf("entry point %q must return void", name)
		}
		if len(f.params) != 0 {
			return fmt.Errorf("entry point %q cannot have function parameters", name)
		}
		if _, ok := v.localSize[fid]; !ok {
			return fmt.Errorf("entry point %q lacks LocalSize", name)
		}
		// Parse declared interface set.
		decl := map[uint32]bool{}
		for _, in := range v.m.Instructions {
			if in.Op == OpEntryPoint && in.Operands[1] == fid {
				_, next, _ := literalString(in.Operands, 2)
				for _, id := range in.Operands[next:] {
					decl[id] = true
				}
			}
		}
		used := map[uint32]bool{}
		for _, l := range f.order {
			for _, idx := range f.blocks[l].insts {
				in := v.m.Instructions[idx]
				for _, id := range valueUses(in) {
					if v.globalVars[id] == StorageInput {
						used[id] = true
					}
				}
			}
		}
		if !setEq(decl, used) {
			return fmt.Errorf("entry point %q interface %v does not exactly match statically used Input globals %v", name, keys(decl), keys(used))
		}
	}
	for fid := range v.localSize {
		if _, ok := v.entryPoints[fid]; !ok {
			return fmt.Errorf("LocalSize applied to non-entry function %%%d", fid)
		}
	}
	return nil
}
func keys(m map[uint32]bool) []uint32 {
	r := make([]uint32, 0, len(m))
	for x := range m {
		r = append(r, x)
	}
	sort.Slice(r, func(i, j int) bool { return r[i] < r[j] })
	return r
}

// Summary returns a compact deterministic validation summary useful in CLI and
// tests without exposing validator internals.
func Summary(data []byte) (string, error) {
	m, err := Decode(data)
	if err != nil {
		return "", err
	}
	if err := Validate(data); err != nil {
		return "", err
	}
	entries := []string{}
	for _, in := range m.Instructions {
		if in.Op == OpEntryPoint {
			name, _, _ := literalString(in.Operands, 2)
			entries = append(entries, name)
		}
	}
	sort.Strings(entries)
	return fmt.Sprintf("SPIR-V 1.3: %d words, bound %d, entries [%s]", len(data)/4, m.Bound, strings.Join(entries, ", ")), nil
}
