package spirv

import (
	"encoding/binary"
	"strings"
	"testing"

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
	bin, err := Emit(m)
	if err != nil {
		t.Fatal(err)
	}
	return bin
}

func compileSPVForMutation(t *testing.T) []byte {
	t.Helper()
	return compileSourceForMutation(t, `
@workgroupSize(1)
export compute math(out: storage<f32[], read_write>) {
  if (globalId.x < out.length) { out[globalId.x] = sin(f32(globalId.x)); }
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
@workgroupSize(1)
export compute arrayMemory(out: storage<u32, read_write>) {
  workgroup scratch: u32[4];
  scratch[0u] = 7u;
  out = scratch[0u];
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
type Pair = { x: u32, y: u32 };
@workgroupSize(1)
export compute structMemory(out: storage<u32, read_write>) {
  workgroup pair: Pair;
  pair = { x: 7u, y: 9u };
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
