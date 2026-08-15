package spirv_test

import (
	"fmt"
	"os"
	"reflect"
	"strings"
	"testing"

	"tach/src/ast"
	"tach/src/flow"
	"tach/src/opt"
	"tach/src/parser"
	"tach/src/sema"
	"tach/src/spirv"
)

func emitSource(t *testing.T, name, source string) []byte {
	t.Helper()
	a, err := parser.Parse(name, source)
	if err != nil {
		t.Fatal(err)
	}
	m, err := sema.CheckAndLower(a)
	if err != nil {
		t.Fatal(err)
	}
	opt.OptimizeLogical(m)
	executable, err := spirv.Lower(m)
	if err != nil {
		t.Fatal(err)
	}
	bin, err := spirv.Emit(executable)
	if err != nil {
		t.Fatal(err)
	}
	return bin
}

func particleModule(t *testing.T) *flow.Module {
	t.Helper()
	var modules []*ast.Module
	for _, name := range []string{"types", "particles"} {
		source, err := os.ReadFile("../../examples/simulation/" + name + ".tach")
		if err != nil {
			t.Fatal(err)
		}
		module, err := parser.Parse("simulation/"+name+".tach", string(source))
		if err != nil {
			t.Fatal(err)
		}
		module.File = "simulation/" + name
		modules = append(modules, module)
	}
	module, _, err := sema.CheckAndLowerProject(modules)
	if err != nil {
		t.Fatal(err)
	}
	return module
}

func TestParticlesSPIRV(t *testing.T) {
	m := particleModule(t)
	opt.OptimizeLogical(m)
	executable, err := spirv.Lower(m)
	if err != nil {
		t.Fatal(err)
	}
	bin, err := spirv.Emit(executable)
	if err != nil {
		t.Fatal(err)
	}
	if len(bin) < 20 {
		t.Fatalf("short SPIR-V binary: %d", len(bin))
	}
	s, err := spirv.Summary(bin)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(s, "entries [_tach_k0]") {
		t.Fatalf("unexpected summary: %s", s)
	}
}

func TestLogicalIndicesAreOptimizedAfterSPIRVBackendLowering(t *testing.T) {
	bin := emitSource(t, "coordinates.tach", `
@workgroup(16, 8)
export function coordinates[x, y](out: buffer<uint32[]>) {
  const localX = x % 16;
  const localY = y % 8;
  const local = localY * 16 + localX;
  out[local] = local + x + y;
}`)
	m, err := spirv.Decode(bin)
	if err != nil {
		t.Fatal(err)
	}
	var global, local, linear bool
	for _, in := range m.Instructions {
		if in.Op == spirv.OpUMod || in.Op == spirv.OpIMul {
			t.Fatal("SPIR-V backend left local-coordinate arithmetic in emitted code")
		}
		if in.Op != spirv.OpDecorate || len(in.Operands) < 3 || in.Operands[1] != spirv.DecorationBuiltIn {
			continue
		}
		global = global || in.Operands[2] == spirv.BuiltInGlobalInvocationID
		local = local || in.Operands[2] == spirv.BuiltInLocalInvocationID
		linear = linear || in.Operands[2] == spirv.BuiltInLocalInvocationIndex
	}
	if !global || local || !linear {
		t.Fatalf("SPIR-V coordinate inputs: global=%v local=%v linear=%v, want true false true", global, local, linear)
	}
}

func TestWorkgroupAggregatesHaveNoExplicitLayout(t *testing.T) {
	tests := []struct {
		name   string
		source string
	}{
		{
			name: "array",
			source: `
@workgroup(1)
export function arrayMemory[i](out: buffer<uint32>) {
  let scratch: shared<uint32[4]>;
  scratch[0] = 7;
  out = scratch[0];
}`,
		},
		{
			name: "struct",
			source: `
type Pair = { x: uint32, y: uint32 };
@workgroup(1)
export function structMemory[i](out: buffer<uint32>) {
  let pair: shared<Pair>;
  pair = { x: 7, y: 9 };
  workgroupBarrier();
  const copy: Pair = pair;
  out = copy.x + copy.y;
}`,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			bin := emitSource(t, tc.name+".tach", tc.source)
			m, err := spirv.Decode(bin)
			if err != nil {
				t.Fatal(err)
			}

			pointerElem := map[uint32]uint32{}
			children := map[uint32][]uint32{}
			explicit := map[uint32]string{}
			var workgroupPointers []uint32
			for _, in := range m.Instructions {
				a := in.Operands
				switch in.Op {
				case spirv.OpTypePointer:
					pointerElem[a[0]] = a[2]
				case spirv.OpTypeVector, spirv.OpTypeArray, spirv.OpTypeRuntimeArray:
					children[a[0]] = []uint32{a[1]}
				case spirv.OpTypeStruct:
					children[a[0]] = append([]uint32(nil), a[1:]...)
				case spirv.OpVariable:
					if a[2] == spirv.StorageWorkgroup {
						workgroupPointers = append(workgroupPointers, a[0])
					}
				case spirv.OpDecorate:
					switch a[1] {
					case spirv.DecorationBlock:
						explicit[a[0]] = "Block"
					case spirv.DecorationArrayStride:
						explicit[a[0]] = "ArrayStride"
					}
				case spirv.OpMemberDecorate:
					if a[2] == spirv.DecorationOffset {
						explicit[a[0]] = "Offset"
					}
				}
			}
			if len(workgroupPointers) != 1 {
				t.Fatalf("found %d Workgroup variables, want 1", len(workgroupPointers))
			}

			seen := map[uint32]bool{}
			var walk func(uint32)
			walk = func(id uint32) {
				if seen[id] {
					return
				}
				seen[id] = true
				if decoration := explicit[id]; decoration != "" {
					t.Errorf("Workgroup-reachable type %%%d carries %s", id, decoration)
				}
				for _, child := range children[id] {
					walk(child)
				}
			}
			walk(pointerElem[workgroupPointers[0]])
		})
	}
}

func TestHostResourceAggregatesKeepExplicitLayout(t *testing.T) {
	logical := particleModule(t)
	if err := opt.OptimizeLogical(logical); err != nil {
		t.Fatal(err)
	}
	executable, err := spirv.Lower(logical)
	if err != nil {
		t.Fatal(err)
	}
	bin, err := spirv.Emit(executable)
	if err != nil {
		t.Fatal(err)
	}
	m, err := spirv.Decode(bin)
	if err != nil {
		t.Fatal(err)
	}
	var block, offset, stride, workgroupPrologue bool
	for _, in := range m.Instructions {
		a := in.Operands
		switch in.Op {
		case spirv.OpDecorate:
			block = block || a[1] == spirv.DecorationBlock
			stride = stride || a[1] == spirv.DecorationArrayStride
			workgroupPrologue = workgroupPrologue || a[1] == spirv.DecorationBuiltIn && a[2] == spirv.BuiltInLocalInvocationIndex
		case spirv.OpMemberDecorate:
			offset = offset || a[2] == spirv.DecorationOffset
		case spirv.OpConstantNull:
			workgroupPrologue = true
		}
	}
	if !block || !offset || !stride {
		t.Fatalf("host resource layout decorations: Block=%v Offset=%v ArrayStride=%v", block, offset, stride)
	}
	if workgroupPrologue {
		t.Fatal("kernel without Workgroup variables received an initialization prologue")
	}
}

func TestSameStructCrossesHostAndWorkgroupRepresentations(t *testing.T) {
	bin := emitSource(t, "shared-struct.tach", `
type Pair = { x: uint32, y: uint32 };
@workgroup(1)
export function sharedStruct[i](io: buffer<Pair>) {
  let pair: shared<Pair>;
  pair = io;
  io = pair;
}`)
	m, err := spirv.Decode(bin)
	if err != nil {
		t.Fatal(err)
	}

	pointerElem := map[uint32]uint32{}
	structMembers := map[uint32][]uint32{}
	offsetCount := map[uint32]int{}
	var workgroupType, hostType uint32
	var hasLogicalConstruct, hasLogicalExtract bool
	for _, in := range m.Instructions {
		a := in.Operands
		switch in.Op {
		case spirv.OpTypePointer:
			pointerElem[a[0]] = a[2]
		case spirv.OpTypeStruct:
			structMembers[a[0]] = append([]uint32(nil), a[1:]...)
		case spirv.OpMemberDecorate:
			if a[2] == spirv.DecorationOffset {
				offsetCount[a[0]]++
			}
		case spirv.OpVariable:
			switch a[2] {
			case spirv.StorageWorkgroup:
				workgroupType = pointerElem[a[0]]
			case spirv.StorageStorageBuffer:
				wrapper := pointerElem[a[0]]
				members := structMembers[wrapper]
				if len(members) == 1 {
					hostType = members[0]
				}
			}
		}
	}
	if workgroupType == 0 || hostType == 0 {
		t.Fatalf("representation roots: Workgroup=%%%d host=%%%d", workgroupType, hostType)
	}
	if workgroupType == hostType {
		t.Fatalf("host and Workgroup aggregates reused decorated type %%%d", hostType)
	}
	if offsetCount[workgroupType] != 0 || offsetCount[hostType] != 2 {
		t.Fatalf("member offsets: Workgroup=%d host=%d, want 0 and 2", offsetCount[workgroupType], offsetCount[hostType])
	}
	for _, in := range m.Instructions {
		switch in.Op {
		case spirv.OpCompositeConstruct:
			hasLogicalConstruct = hasLogicalConstruct || in.Operands[0] == workgroupType
		case spirv.OpCompositeExtract:
			hasLogicalExtract = hasLogicalExtract || in.Operands[0] == structMembers[workgroupType][0]
		}
	}
	if !hasLogicalConstruct || !hasLogicalExtract {
		t.Fatalf("structural host boundary: construct=%v extract=%v", hasLogicalConstruct, hasLogicalExtract)
	}
}

func TestWorkgroupMemoryIsZeroInitialized(t *testing.T) {
	bin := emitSource(t, "zero-workgroup.tach", `
@workgroup(1)
export function zeroWorkgroup[i](out: buffer<uint32>) {
  let scratch: shared<uint32>;
  out = scratch;
}`)
	m, err := spirv.Decode(bin)
	if err != nil {
		t.Fatal(err)
	}
	workgroupVars := map[uint32]bool{}
	nulls := map[uint32]bool{}
	hasBarrier := false
	hasLocalIndex := false
	hasInitializer := false
	for _, in := range m.Instructions {
		a := in.Operands
		switch in.Op {
		case spirv.OpVariable:
			if a[2] == spirv.StorageWorkgroup {
				workgroupVars[a[1]] = true
				hasInitializer = len(a) == 4 && nulls[a[3]]
			}
		case spirv.OpConstantNull:
			nulls[a[1]] = true
		case spirv.OpControlBarrier:
			hasBarrier = true
		case spirv.OpDecorate:
			hasLocalIndex = hasLocalIndex || a[1] == spirv.DecorationBuiltIn && a[2] == spirv.BuiltInLocalInvocationIndex
		}
	}
	hasWorkgroupStore := false
	for _, in := range m.Instructions {
		if in.Op == spirv.OpStore && workgroupVars[in.Operands[0]] {
			hasWorkgroupStore = true
			break
		}
	}
	if len(workgroupVars) != 1 || !hasInitializer || hasWorkgroupStore || hasBarrier || hasLocalIndex {
		t.Fatalf("native Workgroup initialization: variables=%d initializer=%v store=%v barrier=%v localIndex=%v", len(workgroupVars), hasInitializer, hasWorkgroupStore, hasBarrier, hasLocalIndex)
	}
}

func TestEntryPointInterfaceIncludesEveryUsedGlobalStorageClass(t *testing.T) {
	bin := emitSource(t, "interface.tach", `
@workgroup(1)
export function interfaceGlobals[i](out: buffer<uint32[]>, add: uint32) {
  let scratch: shared<uint32>;
  scratch = add;
  workgroupBarrier();
  if (i < out.length) { out[i] = scratch; }
}`)
	m, err := spirv.Decode(bin)
	if err != nil {
		t.Fatal(err)
	}
	storage := map[uint32]uint32{}
	var declared []uint32
	for _, in := range m.Instructions {
		switch in.Op {
		case spirv.OpVariable:
			storage[in.Operands[1]] = in.Operands[2]
		case spirv.OpEntryPoint:
			_, next, _ := entryName(in.Operands)
			declared = append(declared, in.Operands[next:]...)
		}
	}
	classes := map[uint32]bool{}
	for index, id := range declared {
		if index > 0 && id <= declared[index-1] {
			t.Fatalf("interface is not strictly ascending: %v", declared)
		}
		classes[storage[id]] = true
	}
	for _, class := range []uint32{spirv.StorageInput, spirv.StorageUniform, spirv.StorageWorkgroup, spirv.StorageStorageBuffer} {
		if !classes[class] {
			t.Errorf("entry interface omits storage class %d: ids=%v classes=%v", class, declared, classes)
		}
	}
}

func entryName(operands []uint32) (string, int, error) {
	bytes := make([]byte, 0, 16)
	for index, word := range operands[2:] {
		for shift := range 4 {
			value := byte(word >> (8 * shift))
			if value == 0 {
				return string(bytes), index + 3, nil
			}
			bytes = append(bytes, value)
		}
	}
	return "", 0, fmt.Errorf("unterminated entry name")
}

func TestHelpersRequestConstInlining(t *testing.T) {
	bin := emitSource(t, "inline.tach", `
function twice(x: float32): float32 { return x + x; }
@workgroup(1)
export function useHelper[i](out: buffer<float32>) { out = twice(2.0); }
`)
	m, err := spirv.Decode(bin)
	if err != nil {
		t.Fatal(err)
	}
	var controls []uint32
	for _, in := range m.Instructions {
		if in.Op == spirv.OpFunction {
			controls = append(controls, in.Operands[2])
		}
	}
	want := []uint32{spirv.FunctionControlInline | spirv.FunctionControlConst, spirv.FunctionControlNone}
	if !reflect.DeepEqual(controls, want) {
		t.Fatalf("function controls %v, want %v", controls, want)
	}
}
