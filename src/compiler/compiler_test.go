package compiler

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestBuildTargetsCompileAndWriteOnlyTheirArtifacts(t *testing.T) {
	const source = `export function copy[i](data: buffer<uint32[]>) {
  if (i < data.length) { data[i] = i; }
}`
	directory := t.TempDir()
	tests := []struct {
		target BuildTarget
		files  string
	}{
		{TargetAll, "module.d.ts,module.js,module.spv,module.spvasm,module.tach.json,module.tir,module.wgsl"},
		{TargetWeb, "module.d.ts,module.js,module.wgsl"},
		{TargetSPIRV, "module.spv"},
	}
	for _, test := range tests {
		result, err := CompileTarget("module.tach", source, test.target)
		if err != nil {
			t.Fatal(err)
		}
		if test.target == TargetWeb && (result.IR != "" || len(result.SPIRV) != 0 || result.SPIRVAsm != "") {
			t.Fatal("web target generated diagnostic or SPIR-V artifacts")
		}
		if test.target == TargetSPIRV && (result.IR != "" || result.WGSL != "" || result.JavaScript != "" || result.TypeScript != "" || len(result.Metadata) != 0 || result.SPIRVAsm != "") {
			t.Fatal("spirv target generated non-SPIR-V artifacts")
		}
		if _, err := WriteDirectory(result, directory, "module"); err != nil {
			t.Fatal(err)
		}
		entries, err := os.ReadDir(directory)
		if err != nil {
			t.Fatal(err)
		}
		names := make([]string, len(entries))
		for i, entry := range entries {
			names[i] = entry.Name()
		}
		if got := strings.Join(names, ","); got != test.files {
			t.Fatalf("%s target files = %s, want %s", test.target, got, test.files)
		}
	}
}

func TestUnknownBuildTargetIsRejected(t *testing.T) {
	_, err := CompileTarget("module.tach", "", BuildTarget("metal"))
	if err == nil || !strings.Contains(err.Error(), "web, spirv, or all") {
		t.Fatalf("CompileTarget error = %v, want target diagnostic", err)
	}
}

func TestCompletePipelineStructuredSSA(t *testing.T) {
	source := `
type Params = { scale: float32, count: uint32 };
@workgroup(64)
export function transform[start](data: buffer<float32[]>, params: Params) {
  let i = start;
  let acc = 0.0;
  while (i < params.count && acc < 1000.0) {
    acc += data[i] * params.scale;
    i += 64;
  }
  if (start < params.count) { data[start] = acc; }
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

func TestKernelValuesUseOneCrossTargetParameterBlock(t *testing.T) {
	r, err := Compile("parameters.tach", `
type Settings = { enabled: bool, gain: float32 };
export function apply[i](data: buffer<float32[]>, settings: Settings, bias: float32) {
  if (settings.enabled && i < data.length) { data[i] = data[i] * settings.gain + bias; }
}`)
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(r.IR, "uniform") || !strings.Contains(r.IR, "settings=%2: Settings, bias=%3: float32") {
		t.Fatalf("Core IR did not retain target-neutral value parameters:\n%s", r.IR)
	}
	if strings.Count(r.WGSL, "var<uniform>") != 1 || !strings.Contains(r.WGSL, "!= 0u") {
		t.Fatalf("WGSL did not lower values and bool through one physical block:\n%s", r.WGSL)
	}
	if !strings.Contains(r.SPIRVAsm, `"__tach_parameters_apply"`) || strings.Contains(r.SPIRVAsm, "OpFunctionParameter") {
		t.Fatalf("SPIR-V did not lower entry values through a physical block:\n%s", r.SPIRVAsm)
	}
	if !strings.Contains(r.TypeScript, "export type Settings") || !strings.Contains(r.TypeScript, "readonly enabled: boolean") {
		t.Fatalf("generated TypeScript did not preserve the logical source types:\n%s", r.TypeScript)
	}
	if strings.Contains(r.JavaScript, `"path":null`) || !strings.Contains(r.JavaScript, `"path":[]`) {
		t.Fatalf("generated runtime plan has a non-canonical scalar path:\n%s", r.JavaScript)
	}
	var metadata struct {
		Resources []json.RawMessage `json:"resources"`
		Kernels   []struct {
			Parameters []struct {
				Kind string `json:"kind"`
			} `json:"parameters"`
			ParameterBlock *struct {
				Binding  uint32 `json:"binding"`
				ByteSize uint32 `json:"byteSize"`
				Fields   []struct {
					Parameter int      `json:"parameter"`
					Path      []string `json:"path"`
				} `json:"fields"`
			} `json:"parameterBlock"`
		} `json:"kernels"`
	}
	if err := json.Unmarshal(r.Metadata, &metadata); err != nil {
		t.Fatal(err)
	}
	block := metadata.Kernels[0].ParameterBlock
	if len(metadata.Resources) != 1 || block == nil || block.Binding != 1 || block.ByteSize != 16 || len(block.Fields) != 3 {
		t.Fatalf("metadata parameter ABI = %+v", metadata)
	}
	if got := metadata.Kernels[0].Parameters; len(got) != 3 || got[0].Kind != "buffer" || got[1].Kind != "value" || got[2].Kind != "value" {
		t.Fatalf("logical parameters = %+v", got)
	}
	if block.Fields[2].Parameter != 2 || block.Fields[2].Path == nil || len(block.Fields[2].Path) != 0 {
		t.Fatalf("scalar field path = %#v", block.Fields[2])
	}
}

func TestCompletePipelineAtomicsAndBarriers(t *testing.T) {
	source := `
type Counters = { total: atomic<uint32> };
@workgroup(64)
export function accumulate[i](counters: buffer<Counters>) {
  let scratch: shared<atomic<uint32>[64]>;
  const lane = i % 64;
  if (lane == 0) { atomicStore(scratch[0], 0); }
  workgroupBarrier();
  const old = atomicAdd(scratch[lane], 1);
  if (old == 0) { atomicAdd(counters.total, 1); }
  bufferBarrier();
}`
	r, err := Compile("atomics.tach", source)
	if err != nil {
		t.Fatal(err)
	}
	for _, want := range []string{"place.workgroup", "atomic_store", "atomic_add", "workgroup_barrier", "buffer_barrier"} {
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
@workgroup(64)
export function bad[i](data: buffer<uint32[]>) {
  if (i % 64 == 0) { workgroupBarrier(); }
}`)
	if err == nil || !strings.Contains(err.Error(), "non-uniform control flow") {
		t.Fatalf("Compile error = %v, want divergent-barrier error", err)
	}
}

func TestBarrierUniformityAllowsVaryingCarriedDataWithUniformTripCount(t *testing.T) {
	_, err := Compile("uniform-loop.tach", `
@workgroup(64)
export function good[index](data: buffer<uint32[]>, params: uint32) {
  let i = 0;
  let varying = index % 64;
  while (i < params) {
    varying += 1;
    workgroupBarrier();
    i += 1;
  }
  const lane = index % 64;
  if (lane < data.length) { data[lane] = varying; }
}`)
	if err != nil {
		t.Fatal(err)
	}
}

func TestAtomicLoadNeedsNoSourceAccessQualifier(t *testing.T) {
	r, err := Compile("atomic-load.tach", `
type Counters = { total: atomic<uint32> };
@workgroup(1)
export function bad[i](counters: buffer<Counters>) { atomicLoad(counters.total); }
`)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(r.IR, "access=read") || !strings.Contains(r.WGSL, "var<storage, read_write>") {
		t.Fatalf("atomic-load access was not separated across Tach IR and WGSL:\nIR:\n%s\nWGSL:\n%s", r.IR, r.WGSL)
	}
}

func TestNumericLiteralCanonicalization(t *testing.T) {
	r, err := Compile("numbers.tach", `
@workgroup(1)
export function numbers[i](out: buffer<uint32[]>) {
  const mask: uint32 = 0xff00_ff00;
  const hex: uint32 = 0xff;
  const count: uint32 = 0b1010;
  const scale: float32 = 1.25e-3;
  if (i < out.length) { out[i] = mask + hex + count + uint32(scale); }
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

func TestSuffixlessNumericInferenceIsOperandOrderIndependent(t *testing.T) {
	r, err := Compile("literal-order.tach", `
export function literals[i](out: buffer<float32x4>) {
  const signed = i == 0 ? 1 : -2;
  out = float32x4(1 + 2.0, 2.0 + 1, 1.0 + -2, float32(signed));
}`)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(r.IR, "const int32") || strings.Count(r.IR, "const float32") != 2 {
		t.Fatalf("suffixless binary/conditional literals did not infer order-independent numeric types:\n%s", r.IR)
	}
}

func TestCompletePipelineBitwiseAndPortableShifts(t *testing.T) {
	source := `
@workgroup(32)
export function bitwise[i](out: buffer<uint32[]>) {
  let u: uint32 = 0xff00;
  let s: int32 = -64;
  const left = u << 40;
  const logical = u >> 36;
  const arithmetic = s >> 35;
  let mixed = (left | logical) ^ ~u;
  mixed &= 0xffff;
  mixed <<= 33;
  if (i < out.length) { out[i] = mixed | uint32(arithmetic); }
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
@workgroup(64)
export function math[i](out: buffer<float32x4[]>) {
  if (i < out.length) {
    const a = float32x3(float32(i) + 1.0, 2.0, 3.0);
    const b = normalize(a);
    const c = cross(a, float32x3(0.0, 1.0, 0.0));
    const len = length(a);
    const dist = distance(a, c);
    const d = dot(a, b);
    const wave = sin(len) + cos(d) + tan(0.25);
    const shaped = sqrt(abs(wave)) + rsqrt(len + 1.0);
    const expo = exp2(log2(len + 1.0)) + exp(log(len + 1.0));
    const powered = pow(len + 1.0, 2.0);
    const rounded = floor(powered) + ceil(dist) + trunc(shaped);
    const bounded: uint32 = clamp(min(i, 1024), 1, max(i, 1));
    out[i] = float32x4(b.x, b.y, b.z, shaped + expo + rounded + float32(bounded));
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
@workgroup(1)
export function bad[i](out: buffer<float32[]>) {
  if (i < out.length) { out[i] = `+expr+`; }
}`)
		if err == nil || !strings.Contains(err.Error(), "integer") {
			t.Fatalf("%s error = %v, want integer-only intrinsic diagnostic", expr, err)
		}
	}
}

func TestCompilationIsByteDeterministic(t *testing.T) {
	source := `
type Params = { scale: float32, count: uint32 };
function shape(x: float32): float32 { return sin(x) * exp2(x); }
function enabled(value: bool): bool { return value; }
@workgroup(64)
export function deterministic[i](data: buffer<float32[]>, params: Params) {
  if (enabled(i < params.count && i < data.length)) { data[i] = shape(data[i] * params.scale); }
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
@workgroup(64)
export function counted[index](data: buffer<uint32[]>) {
  const lanes = uint32x4(1, 2, 3, 4);
  let sum = 0;
  for (let i = 0; i < 4; i++) { sum += lanes[i]; }
  if (index < data.length) { data[index] = sum; }
}`)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(r.IR, "loop params=") || !strings.Contains(r.IR, "vector_index") || !strings.Contains(r.SPIRVAsm, "OpLoopMerge") || !strings.Contains(r.SPIRVAsm, "OpPhi") || !strings.Contains(r.SPIRVAsm, "OpVectorExtractDynamic") || !strings.Contains(r.WGSL, "loop {") {
		t.Fatalf("for loop did not use the shared structured-loop lowering:\nIR:\n%s\nSPIR-V:\n%s\nWGSL:\n%s", r.IR, r.SPIRVAsm, r.WGSL)
	}
}

func TestConditionalExpressionUsesStructuredValueIf(t *testing.T) {
	r, err := Compile("conditional.tach", `
@workgroup(32)
export function choose[i](out: buffer<uint32[]>) {
  if (i < out.length) { out[i] = (i & 1) == 0 ? i : i + 1; }
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
@workgroup(32)
export function lanes[i](data: buffer<uint32x4[]>) {
  if (i < data.length) {
    if (i == 0) { data[i].x = 1; }
    else if (i == 1) { data[i][1] = 2; }
    else { data[i].z = 3; }
  }
}`)
	if err != nil {
		t.Fatal(err)
	}
	if strings.Count(r.IR, "place.index") < 5 || !strings.Contains(r.SPIRVAsm, "OpAccessChain") {
		t.Fatalf("vector lane stores did not lower through place/access-chain IR:\n%s\n%s", r.IR, r.SPIRVAsm)
	}
}

func TestTachVectorCompositionAndSwizzlesStayTargetNeutral(t *testing.T) {
	r, err := Compile("vectors.tach", `
@workgroup(32)
export function vectors[i](data: buffer<float32x4[]>) {
  if (i < data.length) {
    const low = float32x2(1, 2);
    const value = float32x4(low, 3.0, 4.0);
    data[i] = float32x4(value.yx, value.zw);
  }
}`)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(r.IR, "float32x4") || !strings.Contains(r.IR, "extract") || strings.Contains(r.IR, "vec4") {
		t.Fatalf("Core IR did not retain Tach-native vector semantics:\n%s", r.IR)
	}
	if !strings.Contains(r.WGSL, "vec4<f32>") {
		t.Fatalf("WGSL backend did not lower Tach float32x4:\n%s", r.WGSL)
	}
}

func TestVectorScalarArithmeticIsNormalizedInsideCore(t *testing.T) {
	r, err := Compile("vector-scalars.tach", `
@workgroup(32)
export function vectorScalars[i](values: buffer<float32x4[]>, bits: buffer<uint32x4[]>) {
  if (i < values.length && i < bits.length) {
    values[i] = pow(values[i] + 1, 2) / 2;
    bits[i] = (bits[i] << 3) | 1;
  }
}`)
	if err != nil {
		t.Fatal(err)
	}
	if strings.Count(r.IR, "composite float32x4") < 2 || strings.Count(r.IR, "composite uint32x4") < 2 {
		t.Fatalf("scalar broadcasts were not made explicit in Core IR:\n%s", r.IR)
	}
	for _, targetSyntax := range []string{"= vec4<f32>(", "= vec4<u32>("} {
		if !strings.Contains(r.WGSL, targetSyntax) {
			t.Fatalf("WGSL backend did not lower internal broadcast %q:\n%s", targetSyntax, r.WGSL)
		}
	}
}

func TestLegacyAndTargetShapedSourceFormsAreRejected(t *testing.T) {
	tests := []struct{ name, source, want string }{
		{"fn keyword", `fn old(x: float32): float32 { return x; }`, "expected type, function, or export function declaration"},
		{"compute keyword", `export compute old[i](data: buffer<uint32[]>) { }`, `expected "function"`},
		{"boolean type", `function old(value: boolean): boolean { return value; }`, `unknown type "boolean"`},
		{"short integer", `function old(value: u32): u32 { return value; }`, `unknown type "u32"`},
		{"short signed integer", `function old(value: i32): i32 { return value; }`, `unknown type "i32"`},
		{"short float", `function old(value: f32): f32 { return value; }`, `unknown type "f32"`},
		{"short vector", `function old(value: f32x4): f32x4 { return value; }`, `unknown type "f32x4"`},
		{"uniform wrapper", `@workgroup(1) export function old[i](data: buffer<uint32[]>, value: uniform<uint32>) { }`, `unknown generic type "uniform"`},
		{"workgroup declaration", `@workgroup(1) export function old[i](data: buffer<uint32[]>) { workgroup scratch: uint32; }`, `expected ;`},
		{"initialized shared memory", `@workgroup(1) export function old[i](data: buffer<uint32[]>) { let scratch: shared<uint32> = 0; }`, `shared<T> is only valid in an uninitialized kernel-body let declaration`},
		{"nested shared memory", `@workgroup(1) export function old[i](data: buffer<uint32[]>) { if (true) { let scratch: shared<uint32>; } }`, `shared variables must be declared in the kernel body`},
		{"empty type", `type Empty = {}; export function old[i](data: buffer<uint32[]>) {}`, `type Empty requires at least one field`},
		{"vector name", `@workgroup(1) export function old[i](data: buffer<vec4f[]>) { }`, `unknown type "vec4f"`},
		{"resource wrapper", `@workgroup(1) export function old[i](data: storage<uint32[]>) { }`, `unknown generic type "storage"`},
		{"resource coordinates", `@workgroup(1) export function old[i](@bind(0, 0) data: buffer<uint32[]>) { }`, "expected identifier"},
		{"intrinsic spelling", `@workgroup(1) export function old[i](data: buffer<float32>) { data = inverseSqrt(data); }`, `unknown callable function "inverseSqrt"`},
		{"ambient invocation ID", `@workgroup(1) export function old[i](data: buffer<uint32[]>) { data[0] = globalId.x; }`, `unknown identifier "globalId"`},
		{"implicit indices", `@workgroup(1) export function old(data: buffer<uint32[]>) { }`, `expected [`},
		{"old workgroup attribute", `@workgroupSize(1) export function old[i](data: buffer<uint32[]>) { }`, `unknown kernel attribute @workgroupSize`},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			_, err := Compile("old.tach", test.source)
			if err == nil || !strings.Contains(err.Error(), test.want) {
				t.Fatalf("Compile error = %v, want %q", err, test.want)
			}
		})
	}
}

func TestLogicalIndicesAndPortableWorkgroupDefaults(t *testing.T) {
	r, err := Compile("indices.tach", `
export function linear[i](one: buffer<uint32[]>) {
  if (i < one.length) { one[i] = i; }
}
export function planar[x, y](two: buffer<uint32[]>) {
  if (x < two.length) { two[x] = x + y; }
}
export function volume[x, y, z](three: buffer<uint32[]>) {
  if (x < three.length) { three[x] = x + y + z; }
}`)
	if err != nil {
		t.Fatal(err)
	}
	for _, want := range []string{
		"compute @linear[i=%1](one=@0) workgroup(256,1,1)",
		"compute @planar[x=%1, y=%2](two=@1) workgroup(16,16,1)",
		"compute @volume[x=%1, y=%2, z=%3](three=@2) workgroup(8,8,4)",
	} {
		if !strings.Contains(r.IR, want) {
			t.Fatalf("Core IR missing %q:\n%s", want, r.IR)
		}
	}
	for _, leak := range []string{"builtin", "global_id", "local_id", "invocation"} {
		if strings.Contains(r.IR, leak) {
			t.Fatalf("Core IR leaked backend term %q:\n%s", leak, r.IR)
		}
	}
	var metadata struct {
		Kernels []struct {
			Dimensions uint32 `json:"dimensions"`
		} `json:"kernels"`
	}
	if err := json.Unmarshal(r.Metadata, &metadata); err != nil {
		t.Fatal(err)
	}
	if len(metadata.Kernels) != 3 || metadata.Kernels[0].Dimensions != 1 || metadata.Kernels[1].Dimensions != 2 || metadata.Kernels[2].Dimensions != 3 {
		t.Fatalf("kernel logical dimensions = %+v", metadata.Kernels)
	}
	if !strings.Contains(r.TypeScript, "DispatchOptions<readonly [x: number, y: number]>") {
		t.Fatalf("generated declarations lost the planar dispatch type:\n%s", r.TypeScript)
	}
	if !strings.Contains(r.TypeScript, "DispatchOptions<readonly [x: number, y: number, z: number]>") {
		t.Fatalf("generated declarations lost the volume dispatch type:\n%s", r.TypeScript)
	}
}

func TestLogicalIndexAndWorkgroupDeclarationsAreValidated(t *testing.T) {
	tests := []struct{ source, want string }{
		{`export function none[](out: buffer<uint32[]>) { }`, "requires 1 to 3 logical indices"},
		{`export function many[a, b, c, d](out: buffer<uint32[]>) { }`, "requires 1 to 3 logical indices"},
		{`export function duplicate[i, i](out: buffer<uint32[]>) { }`, `duplicate logical index "i"`},
		{`export function collision[i](i: buffer<uint32[]>) { }`, `duplicate parameter "i"`},
		{`@workgroup(257) export function wide[i](out: buffer<uint32[]>) { }`, "portable limit 256"},
		{`@workgroup(32, 16) export function crowded[x, y](out: buffer<uint32[]>) { }`, "portable limit is 256"},
		{`@workgroup(16, 2) export function rankMismatch[i](out: buffer<uint32[]>) { }`, "expects 1 to 1 integer arguments"},
	}
	for _, test := range tests {
		_, err := Compile("invalid.tach", test.source)
		if err == nil || !strings.Contains(err.Error(), test.want) {
			t.Fatalf("Compile error = %v, want %q", err, test.want)
		}
	}
}

func TestKernelEntryPointIsOneCrossTargetABIName(t *testing.T) {
	r, err := Compile("entry.tach", `
@workgroup(8)
export function step[i](data: buffer<uint32[]>) {
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

func TestBackendsAssignModuleGlobalResourceBindings(t *testing.T) {
	r, err := Compile("multi.tach", `
@workgroup(8)
export function writeU32[i](data: buffer<uint32[]>) {
  if (i < data.length) { data[i] = i; }
}
@workgroup(8)
export function writeF32[i](data: buffer<float32[]>) {
  if (i < data.length) { data[i] = float32(i); }
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
		t.Fatalf("backend bindings = %+v, want group 0 bindings 0 and 1", meta.Resources)
	}
	for _, backendTerm := range []string{"@group", "@binding", "storage<", "read_write", "space=", "slot="} {
		if strings.Contains(r.IR, backendTerm) {
			t.Fatalf("Core IR leaked backend term %q:\n%s", backendTerm, r.IR)
		}
	}
}

func TestDocumentationTachExamples(t *testing.T) {
	root := filepath.Join("..", "..")
	files := []string{
		"README.md",
		filepath.Join("docs", "language.md"),
		filepath.Join("docs", "ir.md"),
		filepath.Join("docs", "architecture.md"),
		filepath.Join("docs", "abi.md"),
		filepath.Join("tach-ts", "README.md"),
	}
	compiled := 0
	for _, name := range files {
		data, err := os.ReadFile(filepath.Join(root, name))
		if err != nil {
			t.Fatal(err)
		}
		blocks := strings.Split(strings.ReplaceAll(string(data), "\r\n", "\n"), "```tach\n")
		for index, block := range blocks[1:] {
			source, _, closed := strings.Cut(block, "\n```")
			if !closed {
				t.Fatalf("%s Tach block %d is not closed", name, index+1)
			}
			compiled++
			if _, err := Compile(name, source); err != nil {
				t.Errorf("%s Tach block %d: %v", name, index+1, err)
			}
		}
	}
	if compiled == 0 {
		t.Fatal("documentation contains no Tach examples")
	}
}
