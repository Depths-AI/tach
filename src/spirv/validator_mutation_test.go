package spirv

import (
	"encoding/binary"
	"strings"
	"testing"

	"tach/src/opt"
	"tach/src/parser"
	"tach/src/sema"
)

func compileSourceForMutation(t *testing.T, source string) []byte {
	t.Helper()
	a, err := parser.Parse("mutation.tach", source)
	if err != nil {
		t.Fatal(err)
	}
	m, err := sema.CheckAndLower(a)
	if err != nil {
		t.Fatal(err)
	}
	opt.OptimizeLogical(m)
	executable, err := Lower(m)
	if err != nil {
		t.Fatal(err)
	}
	bin, err := Emit(executable)
	if err != nil {
		t.Fatal(err)
	}
	return bin
}

func compileSPVForMutation(t *testing.T) []byte {
	t.Helper()
	return compileSourceForMutation(t, `
@workgroup(1)
export function math[i](out: buffer<float32[]>) {
  if (i < out.length) { out[i] = sin(float32(i)); }
}`)
}

func insertBeforeTypes(t *testing.T, bin []byte, op Op, operands ...uint32) []byte {
	t.Helper()
	m, err := Decode(bin)
	if err != nil {
		t.Fatal(err)
	}
	insertWord := -1
	for _, in := range m.Instructions {
		if section(in.Op) >= 9 {
			insertWord = in.Offset
			break
		}
	}
	if insertWord < 0 {
		t.Fatal("module has no type/global section")
	}
	words := append([]uint32{uint32(len(operands)+1)<<16 | uint32(op)}, operands...)
	encoded := make([]byte, len(words)*4)
	for i, word := range words {
		binary.LittleEndian.PutUint32(encoded[i*4:], word)
	}
	insertByte := insertWord * 4
	out := make([]byte, len(bin)+len(encoded))
	copy(out, bin[:insertByte])
	copy(out[insertByte:], encoded)
	copy(out[insertByte+len(encoded):], bin[insertByte:])
	return out
}

func removeInstruction(bin []byte, in Instruction) []byte {
	start := in.Offset * 4
	end := start + (len(in.Operands)+1)*4
	out := make([]byte, 0, len(bin)-(end-start))
	out = append(out, bin[:start]...)
	out = append(out, bin[end:]...)
	return out
}

func workgroupRootType(t *testing.T, bin []byte) uint32 {
	t.Helper()
	m, err := Decode(bin)
	if err != nil {
		t.Fatal(err)
	}
	pointee := map[uint32]uint32{}
	for _, in := range m.Instructions {
		if in.Op == OpTypePointer {
			pointee[in.Operands[0]] = in.Operands[2]
		}
	}
	var root uint32
	for _, in := range m.Instructions {
		if in.Op != OpVariable || in.Operands[2] != StorageWorkgroup {
			continue
		}
		if root != 0 {
			t.Fatal("test module has multiple Workgroup variables")
		}
		root = pointee[in.Operands[0]]
	}
	if root == 0 {
		t.Fatal("test module has no Workgroup variable")
	}
	return root
}

func TestValidatorRejectsUnsupportedGLSL450Instruction(t *testing.T) {
	bin := append([]byte(nil), compileSPVForMutation(t)...)
	m, err := Decode(bin)
	if err != nil {
		t.Fatal(err)
	}
	found := false
	for _, in := range m.Instructions {
		if in.Op != OpExtInst {
			continue
		}
		// OpExtInst operands are ResultType, Result, Set, Instruction, ...
		binary.LittleEndian.PutUint32(bin[(in.Offset+4)*4:], 0xffff)
		found = true
		break
	}
	if !found {
		t.Fatal("test module emitted no OpExtInst")
	}
	err = Validate(bin)
	if err == nil || !strings.Contains(err.Error(), "outside Tach's profile") {
		t.Fatalf("Validate error = %v, want rejected GLSL.std.450 instruction", err)
	}
}

func TestValidatorRejectsNonScalarDynamicVectorIndex(t *testing.T) {
	bin := compileSourceForMutation(t, `
function lane(value: uint32x4, index: uint32): uint32 { return value[index]; }
@workgroup(1)
export function readLane[i](out: buffer<uint32>) { out = lane(uint32x4(1, 2, 3, 4), i); }
`)
	m, err := Decode(bin)
	if err != nil {
		t.Fatal(err)
	}
	for _, in := range m.Instructions {
		if in.Op != OpVectorExtractDynamic {
			continue
		}
		binary.LittleEndian.PutUint32(bin[(in.Offset+4)*4:], in.Operands[2])
		err = Validate(bin)
		if err == nil || !strings.Contains(err.Error(), "index is not an integer scalar") {
			t.Fatalf("Validate error = %v, want dynamic-vector-index type rejection", err)
		}
		return
	}
	t.Fatal("test module emitted no OpVectorExtractDynamic")
}

func TestValidatorRejectsFunctionControlOutsideEmittedProfile(t *testing.T) {
	bin := compileSourceForMutation(t, `
function twice(x: float32): float32 { return x + x; }
@workgroup(1)
export function useHelper[i](out: buffer<float32>) { out = twice(2.0); }
`)
	m, err := Decode(bin)
	if err != nil {
		t.Fatal(err)
	}
	for _, in := range m.Instructions {
		if in.Op == OpFunction && in.Operands[2] != FunctionControlNone {
			binary.LittleEndian.PutUint32(bin[(in.Offset+3)*4:], FunctionControlInline)
			err = Validate(bin)
			if err == nil || !strings.Contains(err.Error(), "function control outside Tach profile") {
				t.Fatalf("Validate error = %v, want function-control rejection", err)
			}
			return
		}
	}
	t.Fatal("test module emitted no marked helper function")
}

func TestValidatorRejectsDuplicateSSAResult(t *testing.T) {
	bin := append([]byte(nil), compileSPVForMutation(t)...)
	m, err := Decode(bin)
	if err != nil {
		t.Fatal(err)
	}
	var firstResult uint32
	for _, in := range m.Instructions {
		id := resultID(in)
		if id == 0 || in.Op == OpExtInstImport {
			continue
		}
		if firstResult == 0 {
			firstResult = id
			continue
		}
		// Replace a later result id with an already-defined id.
		switch in.Op {
		case OpTypeVoid, OpTypeBool, OpTypeInt, OpTypeFloat, OpTypeVector, OpTypeArray, OpTypeRuntimeArray, OpTypeStruct, OpTypePointer, OpTypeFunction, OpLabel:
			binary.LittleEndian.PutUint32(bin[(in.Offset+1)*4:], firstResult)
		default:
			binary.LittleEndian.PutUint32(bin[(in.Offset+2)*4:], firstResult)
		}
		err = Validate(bin)
		if err == nil || !strings.Contains(err.Error(), "defined twice") {
			t.Fatalf("Validate error = %v, want duplicate-id rejection", err)
		}
		return
	}
	t.Fatal("could not find two result-producing instructions")
}

func TestValidatorRequiresDescriptorArrayStride(t *testing.T) {
	bin := compileSPVForMutation(t)
	m, err := Decode(bin)
	if err != nil {
		t.Fatal(err)
	}
	for _, in := range m.Instructions {
		if in.Op != OpDecorate || in.Operands[1] != DecorationArrayStride {
			continue
		}
		err = Validate(removeInstruction(bin, in))
		if err == nil || !strings.Contains(err.Error(), "lacks ArrayStride") {
			t.Fatalf("Validate error = %v, want missing descriptor ArrayStride rejection", err)
		}
		return
	}
	t.Fatal("test module emitted no ArrayStride")
}

func TestValidatorRejectsWorkgroupArrayStride(t *testing.T) {
	bin := compileSourceForMutation(t, `
@workgroup(1)
export function arrayMemory[i](out: buffer<uint32>) {
  let scratch: shared<uint32[4]>;
  scratch[0] = 7;
  out = scratch[0];
}`)
	root := workgroupRootType(t, bin)
	bin = insertBeforeTypes(t, bin, OpDecorate, root, DecorationArrayStride, 4)
	err := Validate(bin)
	if err == nil || !strings.Contains(err.Error(), "ArrayStride explicit layout") {
		t.Fatalf("Validate error = %v, want Workgroup ArrayStride rejection", err)
	}
}

func TestValidatorRejectsWorkgroupMemberOffsets(t *testing.T) {
	bin := compileSourceForMutation(t, `
type Pair = { x: uint32, y: uint32 };
@workgroup(1)
export function structMemory[i](out: buffer<uint32>) {
  let pair: shared<Pair>;
  pair = { x: 7, y: 9 };
  const copy: Pair = pair;
  out = copy.x + copy.y;
}`)
	root := workgroupRootType(t, bin)
	bin = insertBeforeTypes(t, bin, OpMemberDecorate, root, 0, DecorationOffset, 0)
	bin = insertBeforeTypes(t, bin, OpMemberDecorate, root, 1, DecorationOffset, 4)
	err := Validate(bin)
	if err == nil || !strings.Contains(err.Error(), "Offset explicit layout") {
		t.Fatalf("Validate error = %v, want Workgroup Offset rejection", err)
	}
}
