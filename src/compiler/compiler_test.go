package compiler

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"
)

func TestCompletePipelineStructuredSSA(t *testing.T) {
	source := `
type Params = { scale: f32, count: u32 };
@workgroupSize(64)
export compute transform(data: storage<f32[], read_write>, params: uniform<Params>) {
  let i = globalId.x;
  let acc = 0.0;
  while (i < params.count && acc < 1000.0) {
    acc += data[i] * params.scale;
    i += 64u;
  }
  if (globalId.x < params.count) { data[globalId.x] = acc; }
}`
	r, err := Compile("control.tach", source)
	if err != nil {
		t.Fatal(err)
	}
	for name, text := range map[string]string{
		"IR": r.IR, "WGSL": r.WGSL, "SPIR-V asm": r.SPIRVAsm,
		"JavaScript": r.JavaScript, "TypeScript": r.TypeScript, "metadata": string(r.Metadata),
	} {
		if strings.TrimSpace(text) == "" {
			t.Fatalf("%s artifact is empty", name)
		}
	}
	if len(r.SPIRV) < 20 {
		t.Fatalf("SPIR-V is only %d bytes", len(r.SPIRV))
	}
	if !strings.Contains(r.IR, "loop params=") {
		t.Fatal("IR did not retain structured loop-carried SSA")
	}
	if !strings.Contains(r.SPIRVAsm, "OpPhi") {
		t.Fatal("SPIR-V did not lower loop carriers to OpPhi")
	}
	if !strings.Contains(r.WGSL, "loop {") {
		t.Fatal("WGSL did not retain structured loop")
	}
}

func TestCompletePipelineAtomicsAndBarriers(t *testing.T) {
	source := `
type Counters = { total: atomic<u32> };
@workgroupSize(64)
export compute accumulate(@group(0) @binding(0) counters: storage<Counters, read_write>) {
  workgroup scratch: atomic<u32>[64];
  if (localIndex == 0u) { atomicStore(scratch[0u], 0u); }
  workgroupBarrier();
  const old = atomicAdd(scratch[localIndex], 1u);
  if (old == 0u) { atomicAdd(counters.total, 1u); }
  storageBarrier();
}`
	r, err := Compile("atomics.tach", source)
	if err != nil {
		t.Fatal(err)
	}
	for _, want := range []string{"place.workgroup", "atomic_store", "atomic_add", "workgroup_barrier", "storage_barrier"} {
		if !strings.Contains(r.IR, want) {
			t.Fatalf("IR missing %q:\n%s", want, r.IR)
		}
	}
	for _, want := range []string{"var<workgroup>", "atomicStore", "atomicAdd", "workgroupBarrier();", "storageBarrier();"} {
		if !strings.Contains(r.WGSL, want) {
			t.Fatalf("WGSL missing %q:\n%s", want, r.WGSL)
		}
	}
	for _, want := range []string{"OpTypeArray", "OpAtomicStore", "OpAtomicIAdd", "OpControlBarrier"} {
		if !strings.Contains(r.SPIRVAsm, want) {
			t.Fatalf("SPIR-V missing %q:\n%s", want, r.SPIRVAsm)
		}
	}
}

func TestBarrierUniformityRejectsDivergentControl(t *testing.T) {
	_, err := Compile("bad.tach", `
@workgroupSize(64)
export compute bad(data: storage<u32[], read_write>) {
  if (localIndex == 0u) { workgroupBarrier(); }
}`)
	if err == nil || !strings.Contains(err.Error(), "non-uniform control flow") {
		t.Fatalf("Compile error = %v, want divergent-barrier error", err)
	}
}

func TestBarrierUniformityAllowsVaryingCarriedDataWithUniformTripCount(t *testing.T) {
	_, err := Compile("uniform-loop.tach", `
@workgroupSize(64)
export compute good(data: storage<u32[], read_write>, params: uniform<u32>) {
  let i = 0u;
  let varying = localIndex;
  while (i < params) {
    varying += 1u;
    workgroupBarrier();
    i += 1u;
  }
  if (localIndex < data.length) { data[localIndex] = varying; }
}`)
	if err != nil {
		t.Fatal(err)
	}
}

func TestAtomicRequiresWritableStorage(t *testing.T) {
	_, err := Compile("readonly.tach", `
type Counters = { total: atomic<u32> };
@workgroupSize(1)
export compute bad(counters: storage<Counters, read>) { atomicLoad(counters.total); }
`)
	if err == nil || !strings.Contains(err.Error(), "require read_write access") {
		t.Fatalf("Compile error = %v, want read_write atomic-resource error", err)
	}
}

func TestNumericLiteralCanonicalization(t *testing.T) {
	r, err := Compile("numbers.tach", `
@workgroupSize(1)
export compute numbers(out: storage<u32[], read_write>) {
  const mask: u32 = 0xff00_ff00u;
  const hex: u32 = 0xff;
  const count: u32 = 0b1010u;
  const scale: f32 = 1.25e-3f;
  if (globalId.x < out.length) { out[globalId.x] = mask + hex + count + u32(scale); }
}`)
	if err != nil {
		t.Fatal(err)
	}
	for _, bad := range []string{"0xff00_ff00", "0b1010"} {
		if strings.Contains(r.IR, bad) {
			t.Fatalf("IR retained source literal spelling %q:\n%s", bad, r.IR)
		}
	}
	if !strings.Contains(r.IR, "4278255360") || !strings.Contains(r.IR, "255") || !strings.Contains(r.IR, "10") {
		t.Fatalf("IR did not canonicalize integer literals:\n%s", r.IR)
	}
}

func TestCompletePipelineBitwiseAndPortableShifts(t *testing.T) {
	source := `
@workgroupSize(32)
export compute bitwise(out: storage<u32[], read_write>) {
  let u: u32 = 0xff00u;
  let s: i32 = -64i;
  const left = u << 40u;
  const logical = u >> 36u;
  const arithmetic = s >> 35u;
  let mixed = (left | logical) ^ ~u;
  mixed &= 0xffffu;
  mixed <<= 33u;
  if (globalId.x < out.length) { out[globalId.x] = mixed | u32(arithmetic); }
}`
	r, err := Compile("bitwise.tach", source)
	if err != nil {
		t.Fatal(err)
	}
	for _, want := range []string{"OpShiftLeftLogical", "OpShiftRightLogical", "OpShiftRightArithmetic", "OpBitwiseAnd", "OpBitwiseOr", "OpBitwiseXor", "OpNot"} {
		if !strings.Contains(r.SPIRVAsm, want) {
			t.Fatalf("SPIR-V missing %q:\n%s", want, r.SPIRVAsm)
		}
	}
	// Every source shift is normalized through an explicit low-five-bit mask in
	// Core IR, so SPIR-V can never observe a count >= 32.
	if strings.Count(r.IR, " & ") < 4 || !strings.Contains(r.IR, "31") {
		t.Fatalf("IR did not normalize shift counts:\n%s", r.IR)
	}
	for _, want := range []string{" << ", " >> ", " & ", " | ", " ^ ", "~"} {
		if !strings.Contains(r.WGSL, want) {
			t.Fatalf("WGSL missing %q:\n%s", want, r.WGSL)
		}
	}
}

func TestCompletePipelineMathIntrinsics(t *testing.T) {
	source := `
@workgroupSize(64)
export compute math(out: storage<vec4f[], read_write>) {
  const i = globalId.x;
  if (i < out.length) {
    const a = vec3f(f32(i) + 1.0, 2.0, 3.0);
    const b = normalize(a);
    const c = cross(a, vec3f(0.0, 1.0, 0.0));
    const len = length(a);
    const dist = distance(a, c);
    const d = dot(a, b);
    const wave = sin(len) + cos(d) + tan(0.25);
    const shaped = sqrt(abs(wave)) + inverseSqrt(len + 1.0);
    const expo = exp2(log2(len + 1.0)) + exp(log(len + 1.0));
    const powered = pow(len + 1.0, 2.0);
    const rounded = floor(powered) + ceil(dist) + trunc(shaped);
    const bounded: u32 = clamp(min(i, 1024u), 1u, max(i, 1u));
    out[i] = vec4f(b.x, b.y, b.z, shaped + expo + rounded + f32(bounded));
  }
}`
	r, err := Compile("math.tach", source)
	if err != nil {
		t.Fatal(err)
	}
	for _, want := range []string{"intrinsic normalize", "intrinsic cross", "intrinsic length", "intrinsic distance", "intrinsic dot", "intrinsic sin", "intrinsic pow", "intrinsic clamp"} {
		if !strings.Contains(r.IR, want) {
			t.Fatalf("IR missing %q:\n%s", want, r.IR)
		}
	}
	for _, want := range []string{"normalize(", "cross(", "length(", "distance(", "dot(", "sin(", "pow(", "clamp("} {
		if !strings.Contains(r.WGSL, want) {
			t.Fatalf("WGSL missing %q:\n%s", want, r.WGSL)
		}
	}
	for _, want := range []string{"OpExtInstImport", `"GLSL.std.450"`, "OpExtInst", "OpDot"} {
		if !strings.Contains(r.SPIRVAsm, want) {
			t.Fatalf("SPIR-V missing %q:\n%s", want, r.SPIRVAsm)
		}
	}
}

func TestFloatMinMaxClampAreRejectedUntilPortableSpecialValueSemanticsExist(t *testing.T) {
	for _, expr := range []string{"min(1.0, 2.0)", "max(1.0, 2.0)", "clamp(1.0, 0.0, 2.0)"} {
		_, err := Compile("float-bounds.tach", `
@workgroupSize(1)
export compute bad(out: storage<f32[], read_write>) {
  if (globalId.x < out.length) { out[globalId.x] = `+expr+`; }
}`)
		if err == nil || !strings.Contains(err.Error(), "integer") {
			t.Fatalf("%s error = %v, want integer-only intrinsic diagnostic", expr, err)
		}
	}
}

func TestCompilationIsByteDeterministic(t *testing.T) {
	source := `
type Params = { scale: f32, count: u32 };
fn shape(x: f32): f32 { return sin(x) * exp2(x); }
@workgroupSize(64)
export compute deterministic(data: storage<f32[], read_write>, params: uniform<Params>) {
  let i = globalId.x;
  if (i < params.count && i < data.length) { data[i] = shape(data[i] * params.scale); }
}`
	a, err := Compile("deterministic.tach", source)
	if err != nil {
		t.Fatal(err)
	}
	b, err := Compile("deterministic.tach", source)
	if err != nil {
		t.Fatal(err)
	}
	if a.IR != b.IR || a.WGSL != b.WGSL || a.SPIRVAsm != b.SPIRVAsm || a.JavaScript != b.JavaScript || a.TypeScript != b.TypeScript || !bytes.Equal(a.SPIRV, b.SPIRV) || !bytes.Equal(a.Metadata, b.Metadata) {
		t.Fatal("compiling identical Tach source twice produced different artifacts")
	}
}

func TestForLoopLowersToSameStructuredSSA(t *testing.T) {
	r, err := Compile("for.tach", `
@workgroupSize(64)
export compute counted(data: storage<u32[], read_write>) {
  let sum = 0u;
  for (let i = 0u; i < 4u; i++) { sum += i; }
  if (globalId.x < data.length) { data[globalId.x] = sum; }
}`)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(r.IR, "loop params=") || !strings.Contains(r.SPIRVAsm, "OpLoopMerge") || !strings.Contains(r.SPIRVAsm, "OpPhi") || !strings.Contains(r.WGSL, "loop {") {
		t.Fatalf("for loop did not use the shared structured-loop lowering:\nIR:\n%s\nSPIR-V:\n%s\nWGSL:\n%s", r.IR, r.SPIRVAsm, r.WGSL)
	}
}

func TestConditionalExpressionUsesStructuredValueIf(t *testing.T) {
	r, err := Compile("conditional.tach", `
@workgroupSize(32)
export compute choose(out: storage<u32[], read_write>) {
  const i = globalId.x;
  if (i < out.length) { out[i] = (i & 1u) == 0u ? i : i + 1u; }
}`)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(r.IR, "if %") || !strings.Contains(r.SPIRVAsm, "OpPhi") || !strings.Contains(r.WGSL, "var _v") {
		t.Fatalf("conditional expression did not lower through structured SSA value selection:\nIR:\n%s\nSPIR-V:\n%s\nWGSL:\n%s", r.IR, r.SPIRVAsm, r.WGSL)
	}
}

func TestVectorLanePlacesAndElseIf(t *testing.T) {
	r, err := Compile("lanes.tach", `
@workgroupSize(32)
export compute lanes(data: storage<vec4u[], read_write>) {
  const i = globalId.x;
  if (i < data.length) {
    if (i == 0u) { data[i].x = 1u; }
    else if (i == 1u) { data[i][1u] = 2u; }
    else { data[i].z = 3u; }
  }
}`)
	if err != nil {
		t.Fatal(err)
	}
	if strings.Count(r.IR, "place.index") < 5 || !strings.Contains(r.SPIRVAsm, "OpAccessChain") {
		t.Fatalf("vector lane stores did not lower through place/access-chain IR:\n%s\n%s", r.IR, r.SPIRVAsm)
	}
}

func TestKernelEntryPointIsOneCrossTargetABIName(t *testing.T) {
	r, err := Compile("entry.tach", `
@workgroupSize(8)
export compute step(data: storage<u32[], read_write>) {
  const i = globalId.x;
  if (i < data.length) { data[i] = i; }
}`)
	if err != nil {
		t.Fatal(err)
	}
	const entry = "step"
	if !strings.Contains(r.WGSL, "fn "+entry+"(") {
		t.Fatalf("WGSL does not export ABI entry %q:\n%s", entry, r.WGSL)
	}
	if !strings.Contains(r.SPIRVAsm, `"`+entry+`"`) {
		t.Fatalf("SPIR-V does not export ABI entry %q:\n%s", entry, r.SPIRVAsm)
	}
	var meta struct {
		Kernels []struct {
			EntryPoint string `json:"entryPoint"`
		} `json:"kernels"`
	}
	if err := json.Unmarshal(r.Metadata, &meta); err != nil {
		t.Fatal(err)
	}
	if len(meta.Kernels) != 1 || meta.Kernels[0].EntryPoint != entry {
		t.Fatalf("metadata entry points = %+v, want %q", meta.Kernels, entry)
	}
}

func TestAutomaticBindingsAreModuleGlobalAcrossEntryPoints(t *testing.T) {
	r, err := Compile("multi.tach", `
@workgroupSize(8)
export compute writeU32(data: storage<u32[], read_write>) {
  const i = globalId.x;
  if (i < data.length) { data[i] = i; }
}
@workgroupSize(8)
export compute writeF32(data: storage<f32[], read_write>) {
  const i = globalId.x;
  if (i < data.length) { data[i] = f32(i); }
}`)
	if err != nil {
		t.Fatal(err)
	}
	var meta struct {
		Resources []struct {
			Group   uint32 `json:"group"`
			Binding uint32 `json:"binding"`
		} `json:"resources"`
	}
	if err := json.Unmarshal(r.Metadata, &meta); err != nil {
		t.Fatal(err)
	}
	if len(meta.Resources) != 2 {
		t.Fatalf("resources = %d, want 2", len(meta.Resources))
	}
	if meta.Resources[0].Group != 0 || meta.Resources[0].Binding != 0 || meta.Resources[1].Group != 0 || meta.Resources[1].Binding != 1 {
		t.Fatalf("auto bindings = %+v, want group 0 bindings 0 and 1", meta.Resources)
	}
}
