package spirv

import (
	"encoding/binary"
	"fmt"
)

type Instruction struct {
	Offset   int // word offset in the module
	Op       Op
	Operands []uint32
}

// Module is the binary-level representation consumed by Tach's SPIR-V
// validator and diagnostic summary.
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
		return nil, fmt.Errorf("tach SPIR-V profile requires version 1.6, got word 0x%08x", words[1])
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
	case OpConstantTrue, OpConstantFalse, OpConstantNull:
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
		if len(a) != 3 && len(a) != 4 {
			return fmt.Errorf("has %d operands, want 3 or 4", len(a))
		}
		return nil
	case OpLoad:
		return exact(a, 5)
	case OpStore:
		return exact(a, 4)
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
	case OpVectorExtractDynamic:
		return exact(a, 4)
	case OpCompositeExtract:
		return atLeast(a, 4)
	case OpConvertFToU, OpConvertFToS, OpConvertSToF, OpConvertUToF, OpFConvert, OpBitcast,
		OpSNegate, OpFNegate, OpLogicalNot, OpNot, OpAny, OpAll:
		return exact(a, 3)
	case OpIAdd, OpFAdd, OpISub, OpFSub, OpIMul, OpFMul, OpUDiv, OpSDiv, OpFDiv,
		OpUMod, OpSRem, OpFRem, OpVectorTimesScalar, OpLogicalEqual, OpLogicalNotEqual,
		OpLogicalOr, OpLogicalAnd, OpIEqual, OpINotEqual, OpUGreaterThan, OpSGreaterThan,
		OpUGreaterThanEqual, OpSGreaterThanEqual, OpULessThan, OpSLessThan,
		OpULessThanEqual, OpSLessThanEqual, OpFOrdEqual, OpFOrdNotEqual, OpFOrdLessThan,
		OpFOrdGreaterThan, OpFOrdLessThanEqual, OpFOrdGreaterThanEqual,
		OpShiftRightLogical, OpShiftRightArithmetic, OpShiftLeftLogical, OpBitwiseOr, OpBitwiseXor, OpBitwiseAnd:
		return exact(a, 4)
	case OpSelect:
		return exact(a, 5)
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
	case OpAtomicCompareExchange:
		return exact(a, 8)
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
