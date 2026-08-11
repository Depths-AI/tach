package spirv

// Tach emits a deliberately small, fully-owned SPIR-V 1.3 compute profile.
// Values below are fixed by the SPIR-V unified specification.
const (
	Magic   uint32 = 0x07230203
	Version uint32 = 0x00010300
)

type Op uint16

const (
	OpName                 Op = 5
	OpMemberName           Op = 6
	OpExtInstImport        Op = 11
	OpExtInst              Op = 12
	OpMemoryModel          Op = 14
	OpEntryPoint           Op = 15
	OpExecutionMode        Op = 16
	OpCapability           Op = 17
	OpTypeVoid             Op = 19
	OpTypeBool             Op = 20
	OpTypeInt              Op = 21
	OpTypeFloat            Op = 22
	OpTypeVector           Op = 23
	OpTypeArray            Op = 28
	OpTypeRuntimeArray     Op = 29
	OpTypeStruct           Op = 30
	OpTypePointer          Op = 32
	OpTypeFunction         Op = 33
	OpConstantTrue         Op = 41
	OpConstantFalse        Op = 42
	OpConstant             Op = 43
	OpConstantComposite    Op = 44
	OpConstantNull         Op = 46
	OpFunction             Op = 54
	OpFunctionParameter    Op = 55
	OpFunctionEnd          Op = 56
	OpFunctionCall         Op = 57
	OpVariable             Op = 59
	OpLoad                 Op = 61
	OpStore                Op = 62
	OpAccessChain          Op = 65
	OpArrayLength          Op = 68
	OpDecorate             Op = 71
	OpMemberDecorate       Op = 72
	OpVectorExtractDynamic Op = 77
	OpCompositeConstruct   Op = 80
	OpCompositeExtract     Op = 81
	OpConvertFToU          Op = 109
	OpConvertFToS          Op = 110
	OpConvertSToF          Op = 111
	OpConvertUToF          Op = 112
	OpBitcast              Op = 124
	OpSNegate              Op = 126
	OpFNegate              Op = 127
	OpIAdd                 Op = 128
	OpFAdd                 Op = 129
	OpISub                 Op = 130
	OpFSub                 Op = 131
	OpIMul                 Op = 132
	OpFMul                 Op = 133
	OpUDiv                 Op = 134
	OpSDiv                 Op = 135
	OpFDiv                 Op = 136
	OpUMod                 Op = 137
	OpSRem                 Op = 138
	OpFRem                 Op = 140
	OpVectorTimesScalar    Op = 142
	OpDot                  Op = 148
	OpLogicalEqual         Op = 164
	OpLogicalNotEqual      Op = 165
	OpLogicalOr            Op = 166
	OpLogicalAnd           Op = 167
	OpLogicalNot           Op = 168
	OpIEqual               Op = 170
	OpINotEqual            Op = 171
	OpUGreaterThan         Op = 172
	OpSGreaterThan         Op = 173
	OpUGreaterThanEqual    Op = 174
	OpSGreaterThanEqual    Op = 175
	OpULessThan            Op = 176
	OpSLessThan            Op = 177
	OpULessThanEqual       Op = 178
	OpSLessThanEqual       Op = 179
	OpFOrdEqual            Op = 180
	OpFOrdNotEqual         Op = 182
	OpFOrdLessThan         Op = 184
	OpFOrdGreaterThan      Op = 186
	OpFOrdLessThanEqual    Op = 188
	OpFOrdGreaterThanEqual Op = 190
	OpShiftRightLogical    Op = 194
	OpShiftRightArithmetic Op = 195
	OpShiftLeftLogical     Op = 196
	OpBitwiseOr            Op = 197
	OpBitwiseXor           Op = 198
	OpBitwiseAnd           Op = 199
	OpNot                  Op = 200
	OpControlBarrier       Op = 224
	OpAtomicLoad           Op = 227
	OpAtomicStore          Op = 228
	OpAtomicExchange       Op = 229
	OpAtomicIAdd           Op = 234
	OpAtomicISub           Op = 235
	OpAtomicSMin           Op = 236
	OpAtomicUMin           Op = 237
	OpAtomicSMax           Op = 238
	OpAtomicUMax           Op = 239
	OpAtomicAnd            Op = 240
	OpAtomicOr             Op = 241
	OpAtomicXor            Op = 242
	OpPhi                  Op = 245
	OpLoopMerge            Op = 246
	OpSelectionMerge       Op = 247
	OpLabel                Op = 248
	OpBranch               Op = 249
	OpBranchConditional    Op = 250
	OpReturn               Op = 253
	OpReturnValue          Op = 254
	OpUnreachable          Op = 255
)

// GLSL.std.450 extended instruction numbers used by Tach's portable math
// profile. These values are fixed by the Khronos extended instruction set.
const (
	GLSL450Trunc       uint32 = 3
	GLSL450FAbs        uint32 = 4
	GLSL450SAbs        uint32 = 5
	GLSL450Floor       uint32 = 8
	GLSL450Ceil        uint32 = 9
	GLSL450Sin         uint32 = 13
	GLSL450Cos         uint32 = 14
	GLSL450Tan         uint32 = 15
	GLSL450Pow         uint32 = 26
	GLSL450Exp         uint32 = 27
	GLSL450Log         uint32 = 28
	GLSL450Exp2        uint32 = 29
	GLSL450Log2        uint32 = 30
	GLSL450Sqrt        uint32 = 31
	GLSL450InverseSqrt uint32 = 32
	GLSL450UMin        uint32 = 38
	GLSL450SMin        uint32 = 39
	GLSL450UMax        uint32 = 41
	GLSL450SMax        uint32 = 42
	GLSL450UClamp      uint32 = 44
	GLSL450SClamp      uint32 = 45
	GLSL450Length      uint32 = 66
	GLSL450Distance    uint32 = 67
	GLSL450Cross       uint32 = 68
	GLSL450Normalize   uint32 = 69
)

const (
	CapabilityShader uint32 = 1
)

const (
	AddressingLogical uint32 = 0
	MemoryGLSL450     uint32 = 1
)

const (
	ExecutionModelGLCompute uint32 = 5
	ExecutionModeLocalSize  uint32 = 17
)

const (
	StorageInput         uint32 = 1
	StorageUniform       uint32 = 2
	StorageWorkgroup     uint32 = 4
	StorageStorageBuffer uint32 = 12
)

const (
	ScopeDevice    uint32 = 1
	ScopeWorkgroup uint32 = 2
)

const (
	MemorySemanticsRelaxed         uint32 = 0
	MemorySemanticsAcquireRelease  uint32 = 0x8
	MemorySemanticsUniformMemory   uint32 = 0x40
	MemorySemanticsWorkgroupMemory uint32 = 0x100
)

const (
	DecorationBlock         uint32 = 2
	DecorationArrayStride   uint32 = 6
	DecorationBuiltIn       uint32 = 11
	DecorationNonWritable   uint32 = 24
	DecorationBinding       uint32 = 33
	DecorationDescriptorSet uint32 = 34
	DecorationOffset        uint32 = 35
)

const (
	BuiltInNumWorkgroups        uint32 = 24
	BuiltInWorkgroupID          uint32 = 26
	BuiltInLocalInvocationID    uint32 = 27
	BuiltInGlobalInvocationID   uint32 = 28
	BuiltInLocalInvocationIndex uint32 = 29
)

const (
	FunctionControlNone   uint32 = 0
	FunctionControlInline uint32 = 0x1
	FunctionControlConst  uint32 = 0x8
	SelectionControlNone  uint32 = 0
	LoopControlNone       uint32 = 0
)

var opNames = map[Op]string{
	OpName: "OpName", OpMemberName: "OpMemberName", OpMemoryModel: "OpMemoryModel",
	OpExtInstImport: "OpExtInstImport", OpExtInst: "OpExtInst",
	OpEntryPoint: "OpEntryPoint", OpExecutionMode: "OpExecutionMode", OpCapability: "OpCapability",
	OpTypeVoid: "OpTypeVoid", OpTypeBool: "OpTypeBool", OpTypeInt: "OpTypeInt", OpTypeFloat: "OpTypeFloat",
	OpTypeVector: "OpTypeVector", OpTypeArray: "OpTypeArray", OpTypeRuntimeArray: "OpTypeRuntimeArray", OpTypeStruct: "OpTypeStruct",
	OpTypePointer: "OpTypePointer", OpTypeFunction: "OpTypeFunction", OpConstantTrue: "OpConstantTrue",
	OpConstantFalse: "OpConstantFalse", OpConstant: "OpConstant", OpConstantComposite: "OpConstantComposite",
	OpConstantNull: "OpConstantNull",
	OpFunction:     "OpFunction", OpFunctionParameter: "OpFunctionParameter", OpFunctionEnd: "OpFunctionEnd",
	OpFunctionCall: "OpFunctionCall", OpVariable: "OpVariable", OpLoad: "OpLoad", OpStore: "OpStore",
	OpAccessChain: "OpAccessChain", OpArrayLength: "OpArrayLength", OpDecorate: "OpDecorate",
	OpMemberDecorate: "OpMemberDecorate", OpCompositeConstruct: "OpCompositeConstruct",
	OpVectorExtractDynamic: "OpVectorExtractDynamic", OpCompositeExtract: "OpCompositeExtract",
	OpConvertFToU: "OpConvertFToU", OpConvertFToS: "OpConvertFToS",
	OpConvertSToF: "OpConvertSToF", OpConvertUToF: "OpConvertUToF", OpBitcast: "OpBitcast",
	OpSNegate: "OpSNegate", OpFNegate: "OpFNegate", OpIAdd: "OpIAdd", OpFAdd: "OpFAdd",
	OpISub: "OpISub", OpFSub: "OpFSub", OpIMul: "OpIMul", OpFMul: "OpFMul", OpUDiv: "OpUDiv",
	OpSDiv: "OpSDiv", OpFDiv: "OpFDiv", OpUMod: "OpUMod", OpSRem: "OpSRem", OpFRem: "OpFRem",
	OpVectorTimesScalar: "OpVectorTimesScalar", OpDot: "OpDot", OpLogicalEqual: "OpLogicalEqual",
	OpLogicalNotEqual: "OpLogicalNotEqual", OpLogicalOr: "OpLogicalOr", OpLogicalAnd: "OpLogicalAnd",
	OpLogicalNot: "OpLogicalNot", OpIEqual: "OpIEqual", OpINotEqual: "OpINotEqual",
	OpUGreaterThan: "OpUGreaterThan", OpSGreaterThan: "OpSGreaterThan", OpUGreaterThanEqual: "OpUGreaterThanEqual",
	OpSGreaterThanEqual: "OpSGreaterThanEqual", OpULessThan: "OpULessThan", OpSLessThan: "OpSLessThan",
	OpULessThanEqual: "OpULessThanEqual", OpSLessThanEqual: "OpSLessThanEqual", OpFOrdEqual: "OpFOrdEqual",
	OpFOrdNotEqual: "OpFOrdNotEqual", OpFOrdLessThan: "OpFOrdLessThan", OpFOrdGreaterThan: "OpFOrdGreaterThan",
	OpFOrdLessThanEqual: "OpFOrdLessThanEqual", OpFOrdGreaterThanEqual: "OpFOrdGreaterThanEqual",
	OpShiftRightLogical: "OpShiftRightLogical", OpShiftRightArithmetic: "OpShiftRightArithmetic",
	OpShiftLeftLogical: "OpShiftLeftLogical", OpBitwiseOr: "OpBitwiseOr", OpBitwiseXor: "OpBitwiseXor",
	OpBitwiseAnd: "OpBitwiseAnd", OpNot: "OpNot",
	OpControlBarrier: "OpControlBarrier", OpAtomicLoad: "OpAtomicLoad", OpAtomicStore: "OpAtomicStore",
	OpAtomicExchange: "OpAtomicExchange", OpAtomicIAdd: "OpAtomicIAdd", OpAtomicISub: "OpAtomicISub",
	OpAtomicSMin: "OpAtomicSMin", OpAtomicUMin: "OpAtomicUMin", OpAtomicSMax: "OpAtomicSMax", OpAtomicUMax: "OpAtomicUMax",
	OpAtomicAnd: "OpAtomicAnd", OpAtomicOr: "OpAtomicOr", OpAtomicXor: "OpAtomicXor",
	OpPhi: "OpPhi", OpLoopMerge: "OpLoopMerge", OpSelectionMerge: "OpSelectionMerge", OpLabel: "OpLabel",
	OpBranch: "OpBranch", OpBranchConditional: "OpBranchConditional", OpReturn: "OpReturn",
	OpReturnValue: "OpReturnValue", OpUnreachable: "OpUnreachable",
}

func opName(op Op) string {
	if s := opNames[op]; s != "" {
		return s
	}
	return "OpUnknown"
}
