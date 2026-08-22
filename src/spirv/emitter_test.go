package spirv_test

import (
	"fmt"
	"os"
	"reflect"
	"strings"
	"testing"

	"tach/src/ir"
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
	opt.Optimize(m)
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

func particleModule(t *testing.T) *ir.Module {
	t.Helper()
	var modules []*parser.File
	for _, name := range []string{"types", "particles"} {
		source, err := os.ReadFile("../../examples/simulation/" + name + ".tach")
		if err != nil {
			t.Fatal(err)
		}
		module, err := parser.Parse("simulation/"+name+".tach", string(source))
		if err != nil {
			t.Fatal(err)
		}
		module.Path = "simulation/" + name
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
	opt.Optimize(m)
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

func TestFloat16SPIRVCapabilitiesTypesConstantsAndConversions(t *testing.T) {
	bin := emitSource(t, "float16.tach", `
@workgroup(1)
export function half[i](values: buffer<float16[]>, factor: float16) {
  if (i < values.length) {
    let one: float16 = 1.0;
    values[i] = float16(float32(values[i] * factor)) + one;
  }
}`)
	m, err := spirv.Decode(bin)
	if err != nil {
		t.Fatal(err)
	}
	caps, halfTypes := map[uint32]int{}, map[uint32]bool{}
	var constants, one, converts, arrayLengths int
	for _, in := range m.Instructions {
		switch in.Op {
		case spirv.OpCapability:
			caps[in.Operands[0]]++
		case spirv.OpTypeFloat:
			if in.Operands[1] == 16 {
				halfTypes[in.Operands[0]] = true
			}
		case spirv.OpConstant:
			if halfTypes[in.Operands[0]] {
				constants++
				if in.Operands[2] == 0x3c00 {
					one++
				}
				if in.Operands[2]&0xffff0000 != 0 {
					t.Fatalf("Float16 constant used high literal bits: %v", in.Operands)
				}
			}
		case spirv.OpFConvert:
			converts++
		case spirv.OpArrayLength:
			arrayLengths++
		}
	}
	for _, capability := range []uint32{spirv.CapabilityShader, spirv.CapabilityFloat16, spirv.CapabilityStorageBuffer16BitAccess, spirv.CapabilityUniformAndStorage16BitAccess, spirv.CapabilityVulkanMemoryModel} {
		if caps[capability] != 1 {
			t.Fatalf("capability %d count = %d, want 1", capability, caps[capability])
		}
	}
	if len(halfTypes) != 1 || constants == 0 || one != 1 || converts != 2 || arrayLengths != 0 {
		t.Fatalf("Float16 SPIR-V types/constants/one/conversions/physical lengths = %d/%d/%d/%d/%d, want 1/>0/1/2/0", len(halfTypes), constants, one, converts, arrayLengths)
	}
}

func TestLoopControlAndFmaSPIRV(t *testing.T) {
	bin := emitSource(t, "control.tach", `
export function control[i](out: buffer<float32[]>, half: buffer<vec<float16, 4>[]>, limit: uint32) {
  let total: float32 = 0;
  for (let step = 0; step < limit; step++) {
    if (step == 2) { continue; }
    total = fma(float32(step), 0.5, total);
    if (total > 10.0) { break; }
  }
  if (i < out.length && i < half.length) {
    out[i] = total;
    half[i] = fma(half[i], float16(2), vec(1, 1, 1, 1));
  }
}`)
	if err := spirv.Validate(bin); err != nil {
		t.Fatal(err)
	}
	m, err := spirv.Decode(bin)
	if err != nil {
		t.Fatal(err)
	}
	var fma32, fma16, phis int
	floatWidth := map[uint32]uint32{}
	for _, instruction := range m.Instructions {
		switch instruction.Op {
		case spirv.OpTypeFloat:
			floatWidth[instruction.Operands[0]] = instruction.Operands[1]
		case spirv.OpTypeVector:
			floatWidth[instruction.Operands[0]] = floatWidth[instruction.Operands[1]]
		case spirv.OpPhi:
			phis++
		}
		if instruction.Op == spirv.OpExtInst && instruction.Operands[3] == spirv.GLSL450Fma {
			switch floatWidth[instruction.Operands[0]] {
			case 16:
				fma16++
			case 32:
				fma32++
			}
		}
	}
	if fma16 != 1 || fma32 != 1 || phis == 0 {
		t.Fatalf("SPIR-V Fma/loop phi counts: float16=%d float32=%d phis=%d, want 1/1/>0", fma16, fma32, phis)
	}
}

func TestFloatBoundsAndStrongCompareExchangeSPIRV(t *testing.T) {
	bin := emitSource(t, "bounds-atomic.tach", `
export function boundsAtomic[i](values: buffer<vec<float32, 2>[]>, half: buffer<float16[]>, state: buffer<atomic<uint32>[]>) {
  if (i < values.length && i < half.length && i < state.length) {
    values[i] = clamp(values[i], min(values[i], -1.0), max(values[i], 1.0));
    half[i] = clamp(half[i], float16(-1), float16(1));
    let observed = atomicCompareExchange(state[i], 0, 1);
    if (observed != 0) { atomicAdd(state[i], 1); }
  }
}`)
	if err := spirv.Validate(bin); err != nil {
		t.Fatal(err)
	}
	m, err := spirv.Decode(bin)
	if err != nil {
		t.Fatal(err)
	}
	var compareExchange, selects, orderedLess, extendedInstructions int
	for _, instruction := range m.Instructions {
		switch instruction.Op {
		case spirv.OpAtomicCompareExchange:
			compareExchange++
		case spirv.OpSelect:
			selects++
		case spirv.OpFOrdLessThan:
			orderedLess++
		case spirv.OpExtInst:
			extendedInstructions++
		}
	}
	if compareExchange != 1 || selects < 6 || orderedLess < 6 || extendedInstructions != 0 {
		t.Fatalf("SPIR-V bounds/CAS: compare-exchange=%d selects=%d ordered-less=%d extended-instructions=%d", compareExchange, selects, orderedLess, extendedInstructions)
	}
}

func TestBooleanVectorsAndMasksSPIRV(t *testing.T) {
	bin := emitSource(t, "masks.tach", `
function choose(value: vec<float32, 4>): vec<float32, 4> {
  let inside = value >= -1.0 & value <= 1.0;
  let identity = select(inside, true, false);
  let changed = select(identity == inside, identity, identity != inside) ^ vec(false, true, false, true);
  let selected = select(changed | value == 0.0, abs(value), -value);
  return all(inside) || any(!changed) ? selected : vec(0.0, 0.0, 0.0, 0.0);
}
export function masks[i](out: buffer<vec<float32, 4>[]>) {
  if (i < out.length) { out[i] = choose(vec(float32(i), -0.5, 0.0, 2.0)); }
}`)
	if err := spirv.Validate(bin); err != nil {
		t.Fatal(err)
	}
	m, err := spirv.Decode(bin)
	if err != nil {
		t.Fatal(err)
	}
	counts := map[spirv.Op]int{}
	for _, instruction := range m.Instructions {
		counts[instruction.Op]++
	}
	for _, op := range []spirv.Op{spirv.OpTypeBool, spirv.OpAny, spirv.OpAll, spirv.OpSelect, spirv.OpLogicalNot, spirv.OpLogicalAnd, spirv.OpLogicalOr, spirv.OpLogicalEqual, spirv.OpLogicalNotEqual, spirv.OpFOrdGreaterThanEqual, spirv.OpFOrdLessThanEqual} {
		if counts[op] == 0 {
			t.Fatalf("mask SPIR-V missing opcode %d", op)
		}
	}
}

func TestViewProjectionIsValidSPIRV16(t *testing.T) {
	bin := emitSource(t, "view.tach", `
function paint[i](pixels: buffer<vec<float32, 4>[]>) {
  if (i < pixels.length) { pixels[i] = vec(0.1, 0.2, 0.3, 1.0); }
}
export function image(width: uint32, height: uint32): view<srgb8> {
  let pixels = transient<vec<float32, 4>>(width * height);
  run paint(pixels) over pixels.length;
  return view(pixels, width, height);
}`)
	summary, err := spirv.Summary(bin)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(summary, "SPIR-V 1.6") || !strings.Contains(summary, "entries [_tach_k0]") {
		t.Fatalf("projection summary = %s", summary)
	}
}

func TestExternalViewUsesStandaloneSPIRVProjection(t *testing.T) {
	bin := emitSource(t, "view.tach", `
function paint[i](pixels: buffer<vec<float32, 4>[]>) { pixels[i] = vec(0.1, 0.2, 0.3, 1.0); }
export function image(pixels: buffer<vec<float32, 4>[]>, width: uint32, height: uint32): view<srgb8> {
  run paint(pixels) over width * height;
  return view(pixels, width, height);
}`)
	summary, err := spirv.Summary(bin)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(summary, "entries [_tach_k0, _tach_k1]") {
		t.Fatalf("projection summary = %s", summary)
	}
}

func TestLogicalIndicesAreOptimizedAfterSPIRVBackendLowering(t *testing.T) {
	bin := emitSource(t, "coordinates.tach", `
@workgroup(16, 8)
export function coordinates[x, y](out: buffer<uint32[]>) {
  let localX = x % 16;
  let localY = y % 8;
  let local = localY * 16 + localX;
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
  let copy: Pair = pair;
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
	if err := opt.Optimize(logical); err != nil {
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
	valueType := map[uint32]uint32{}
	for _, in := range m.Instructions {
		a := in.Operands
		switch in.Op {
		case spirv.OpVariable, spirv.OpAccessChain, spirv.OpLoad:
			if len(a) >= 2 {
				valueType[a[1]] = a[0]
			}
		}
	}
	var workgroupAccesses int
	for _, in := range m.Instructions {
		var ptr, align uint32
		switch in.Op {
		case spirv.OpLoad:
			ptr, align = in.Operands[2], in.Operands[4]
		case spirv.OpStore:
			ptr, align = in.Operands[0], in.Operands[3]
		default:
			continue
		}
		if pointerElem[valueType[ptr]] != workgroupType {
			continue
		}
		workgroupAccesses++
		if align != 4 {
			t.Fatalf("Workgroup Pair access Aligned %d, want logical 4", align)
		}
	}
	if workgroupAccesses == 0 {
		t.Fatal("no Workgroup Pair load/store")
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

func TestVulkan13MemorySemantics(t *testing.T) {
	bin := emitSource(t, "semantics.tach", `
@workgroup(1)
export function semantics[i](out: buffer<uint32[]>, counters: buffer<atomic<uint32>[]>, add: uint32) {
  let scratch: shared<uint32>;
  scratch = add;
  workgroupBarrier();
  if (i < out.length) { out[i] = scratch; }
  if (i < counters.length) { atomicAdd(counters[i], 1); }
}`)
	m, err := spirv.Decode(bin)
	if err != nil {
		t.Fatal(err)
	}
	caps := map[uint32]int{}
	pointerStorage := map[uint32]uint32{}
	valueType := map[uint32]uint32{}
	var memoryModel []uint32
	var loads, stores, atomics, barriers int
	for _, in := range m.Instructions {
		a := in.Operands
		switch in.Op {
		case spirv.OpCapability:
			caps[a[0]]++
		case spirv.OpMemoryModel:
			memoryModel = a
		case spirv.OpTypePointer:
			pointerStorage[a[0]] = a[1]
		case spirv.OpVariable, spirv.OpAccessChain, spirv.OpFunctionParameter:
			if len(a) >= 2 {
				valueType[a[1]] = a[0]
			}
		case spirv.OpLoad:
			valueType[a[1]] = a[0]
			loads++
			storage := pointerStorage[valueType[a[2]]]
			if a[3]&spirv.MemoryAccessAligned == 0 || a[4] == 0 || a[4]&(a[4]-1) != 0 {
				t.Fatalf("load missing Aligned: %v", a)
			}
			if storage == spirv.StorageInput {
				if a[3]&spirv.MemoryAccessNonPrivatePointer != 0 {
					t.Fatalf("input load is NonPrivate: %v", a)
				}
			} else if a[3]&spirv.MemoryAccessNonPrivatePointer == 0 {
				t.Fatalf("shared load missing NonPrivatePointer: %v", a)
			}
		case spirv.OpStore:
			stores++
			if a[2]&spirv.MemoryAccessAligned == 0 || a[2]&spirv.MemoryAccessNonPrivatePointer == 0 {
				t.Fatalf("store missing Aligned|NonPrivatePointer: %v", a)
			}
		case spirv.OpAtomicIAdd:
			atomics++
			scope := uint32(0)
			for _, inst := range m.Instructions {
				if inst.Op == spirv.OpConstant && inst.Operands[1] == a[3] {
					scope = inst.Operands[2]
				}
			}
			storage := pointerStorage[valueType[a[2]]]
			want := spirv.ScopeQueueFamily
			if storage == spirv.StorageWorkgroup {
				want = spirv.ScopeWorkgroup
			}
			if scope != want {
				t.Fatalf("atomic scope=%d storage=%d, want %d", scope, storage, want)
			}
		case spirv.OpControlBarrier:
			barriers++
			sem := uint32(0)
			for _, inst := range m.Instructions {
				if inst.Op == spirv.OpConstant && inst.Operands[1] == a[2] {
					sem = inst.Operands[2]
				}
			}
			need := spirv.MemorySemanticsAcquireRelease | spirv.MemorySemanticsMakeAvailable | spirv.MemorySemanticsMakeVisible
			if sem&need != need {
				t.Fatalf("barrier semantics=0x%x missing availability/visibility", sem)
			}
		}
	}
	if caps[spirv.CapabilityShader] != 1 || caps[spirv.CapabilityVulkanMemoryModel] != 1 || len(caps) != 2 {
		t.Fatalf("capabilities = %v", caps)
	}
	if len(memoryModel) != 2 || memoryModel[0] != spirv.AddressingLogical || memoryModel[1] != spirv.MemoryVulkan {
		t.Fatalf("memory model = %v", memoryModel)
	}
	if loads == 0 || stores == 0 || atomics == 0 || barriers == 0 {
		t.Fatalf("coverage loads=%d stores=%d atomics=%d barriers=%d", loads, stores, atomics, barriers)
	}
}
