package spirv_test

import (
	"os"
	"strings"
	"tach/parser"
	"tach/semantics"
	"tach/spirv"
	"testing"
)

func emitSource(t *testing.T, name, source string) []byte {
	t.Helper()
	a, err := parser.Parse(name, source)
	if err != nil {
		t.Fatal(err)
	}
	result, err := semantics.Build([]*parser.File{a}, 0)
	if err != nil {
		t.Fatal(err)
	}
	lowered, err := spirv.Lower(result.Module)
	if err != nil {
		t.Fatal(err)
	}
	return lowered.Binary
}

func particleProgram(t *testing.T) *semantics.Result {
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
	result, err := semantics.Build(modules, 0)
	if err != nil {
		t.Fatal(err)
	}
	return result
}

func TestParticlesSPIRV(t *testing.T) {
	result := particleProgram(t)
	lowered, err := spirv.Lower(result.Module)
	if err != nil {
		t.Fatal(err)
	}
	if len(lowered.Binary) < 20 {
		t.Fatalf("short SPIR-V binary: %d", len(lowered.Binary))
	}
	s, err := spirv.Summary(lowered.Binary)
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
