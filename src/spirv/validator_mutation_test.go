package spirv

import (
	"encoding/binary"
	"strings"
	"testing"

	"tach/src/parser"
	"tach/src/sema"
)

func compileSPVForMutation(t *testing.T) []byte {
	t.Helper()
	a, err := parser.Parse("mutation.tach", `
@workgroupSize(1)
export compute math(out: storage<f32[], read_write>) {
  if (globalId.x < out.length) { out[globalId.x] = sin(f32(globalId.x)); }
}`)
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
