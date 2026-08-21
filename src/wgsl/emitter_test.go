package wgsl_test

import (
	"os"
	"strings"
	"testing"

	"tach/src/ast"
	"tach/src/flow"
	"tach/src/opt"
	"tach/src/parser"
	"tach/src/sema"
	"tach/src/wgsl"
)

func emit(m *flow.Module) (string, error) {
	if err := opt.OptimizeLogical(m); err != nil {
		return "", err
	}
	executable, err := wgsl.Lower(m)
	if err != nil {
		return "", err
	}
	return wgsl.Emit(executable)
}

func TestParticlesWGSL(t *testing.T) {
	var modules []*ast.Module
	for _, name := range []string{"types", "particles"} {
		src, err := os.ReadFile("../../examples/simulation/" + name + ".tach")
		if err != nil {
			t.Fatal(err)
		}
		module, err := parser.Parse("simulation/"+name+".tach", string(src))
		if err != nil {
			t.Fatal(err)
		}
		module.File = "simulation/" + name
		modules = append(modules, module)
	}
	m, _, err := sema.CheckAndLowerProject(modules)
	if err != nil {
		t.Fatal(err)
	}
	out, err := emit(m)
	if err != nil {
		t.Fatal(err)
	}
	for _, want := range []string{"@compute", "@workgroup_size(256, 1, 1)", "@builtin(global_invocation_id)", "var<storage, read_write>", "array<", "return;"} {
		if !strings.Contains(out, want) {
			t.Fatalf("WGSL missing %q:\n%s", want, out)
		}
	}
	if err := wgsl.Validate(out); err != nil {
		t.Fatal(err)
	}
}

func TestValidatorRejectsMalformedGeneratedShape(t *testing.T) {
	bad := `@compute @workgroup_size(1, 1, 1) fn bad() { let _v1: u32 = 1u }`
	if err := wgsl.Validate(bad); err == nil {
		t.Fatal("WGSL validator accepted a generated statement without semicolon")
	}
}

func TestValidatorRejectsReservedDoubleUnderscoreIdentifier(t *testing.T) {
	bad := `@compute @workgroup_size(1, 1, 1) fn __bad() { return; }`
	if err := wgsl.Validate(bad); err == nil || !strings.Contains(err.Error(), "reserved identifier") {
		t.Fatalf("WGSL validator error = %v, want reserved identifier", err)
	}
}

func TestRuntimeArrayWrapperKeepsNaturalAlignment(t *testing.T) {
	a, err := parser.Parse("runtime.tach", `
type Series = { offset: float16, values: float16[] };
@workgroup(1)
export function clear[i](data: buffer<uint32[]>) {
  if (i < data.length) { data[i] = 0; }
}
export function clearSeries[i](data: buffer<Series>) {
  if (i < data.values.length) { data.values[i] = data.offset; }
}`)
	if err != nil {
		t.Fatal(err)
	}
	m, err := sema.CheckAndLower(a)
	if err != nil {
		t.Fatal(err)
	}
	out, err := emit(m)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(out, "data: array<u32>,") || strings.Contains(out, "@align(16) data: array<u32>,") {
		t.Fatalf("runtime array wrapper has a synthetic fixed-resource alignment:\n%s", out)
	}
	if !strings.Contains(out, "var<storage, read_write> _tach_r1_0: _tach_t_Series;") || strings.Contains(out, "data: _tach_t_Series") {
		t.Fatalf("runtime-tail struct is not the storage root:\n%s", out)
	}
	if !strings.Contains(out, "struct _tach_t_Series {\n  f0_offset: f16,") {
		t.Fatalf("runtime-tail struct has a synthetic fixed-struct alignment:\n%s", out)
	}
}

func TestFloat16WGSLFeatureAndTypes(t *testing.T) {
	a, err := parser.Parse("float16.tach", `
export function half[i](values: buffer<vec<float16, 2>[]>, factor: float16) {
  if (i < values.length) { values[i] = sin(values[i]) * factor; }
}`)
	if err != nil {
		t.Fatal(err)
	}
	m, err := sema.CheckAndLower(a)
	if err != nil {
		t.Fatal(err)
	}
	out, err := emit(m)
	if err != nil {
		t.Fatal(err)
	}
	for _, want := range []string{"enable f16;", "vec2<f16>", "f0: f16"} {
		if !strings.Contains(out, want) {
			t.Fatalf("Float16 WGSL missing %q:\n%s", want, out)
		}
	}
	if err := wgsl.Validate(out); err != nil {
		t.Fatal(err)
	}
	if err := wgsl.Validate("fn ok() { return; }\nenable f16;"); err == nil {
		t.Fatal("validator accepted enable directive after a declaration")
	}
	if err := wgsl.Validate("enable unknown; fn ok() { return; }"); err == nil {
		t.Fatal("validator accepted an unknown enable directive")
	}
}

func TestLoopControlAndFmaWGSL(t *testing.T) {
	a, err := parser.Parse("control.tach", `
export function control[i](out: buffer<float32[]>, limit: uint32) {
  let total: float32 = 0;
  for (let step = 0; step < limit; step++) {
    if (step == 2) { continue; }
    total = fma(float32(step), 0.5, total);
    if (total > 10.0) { break; }
  }
  if (i < out.length) { out[i] = total; }
}`)
	if err != nil {
		t.Fatal(err)
	}
	m, err := sema.CheckAndLower(a)
	if err != nil {
		t.Fatal(err)
	}
	out, err := emit(m)
	if err != nil {
		t.Fatal(err)
	}
	for _, want := range []string{"fma(", "continue;", "break;"} {
		if !strings.Contains(out, want) {
			t.Fatalf("WGSL missing %q:\n%s", want, out)
		}
	}
	if err := wgsl.Validate(out); err != nil {
		t.Fatal(err)
	}
}

func TestFloatBoundsAndStrongCompareExchangeWGSL(t *testing.T) {
	a, err := parser.Parse("bounds-atomic.tach", `
export function boundsAtomic[i](values: buffer<vec<float32, 2>[]>, state: buffer<atomic<uint32>[]>) {
  if (i < values.length && i < state.length) {
    let lower = min(values[i], -values[i]);
    let upper = max(lower, values[i]);
    values[i] = clamp(upper, 1.0, -1.0);
    let observed = atomicCompareExchange(state[i], 0, 1);
    if (observed != 0) { atomicAdd(state[i], 1); }
  }
}`)
	if err != nil {
		t.Fatal(err)
	}
	m, err := sema.CheckAndLower(a)
	if err != nil {
		t.Fatal(err)
	}
	out, err := emit(m)
	if err != nil {
		t.Fatal(err)
	}
	for _, want := range []string{"_bounded", "select(", "atomicCompareExchangeWeak(", ".exchanged ||", ".old_value !=", "loop {"} {
		if !strings.Contains(out, want) {
			t.Fatalf("bounds/CAS WGSL missing %q:\n%s", want, out)
		}
	}
	for _, incompatible := range []string{"min(", "max(", "clamp("} {
		if strings.Contains(out, incompatible) {
			t.Fatalf("Tach bounds leaked WGSL's incompatible %s contract:\n%s", incompatible, out)
		}
	}
	if err := wgsl.Validate(out); err != nil {
		t.Fatal(err)
	}
}

func TestBooleanVectorsAndMasksWGSL(t *testing.T) {
	a, err := parser.Parse("masks.tach", `
function choose(value: vec<float32, 4>): vec<float32, 4> {
  let inside = value >= -1.0 & value <= 1.0;
  let changed = inside ^ vec(false, true, false, true);
  let selected = select(changed | value == 0.0, abs(value), -value);
  return all(inside) || any(!changed) ? selected : vec(0.0, 0.0, 0.0, 0.0);
}
export function masks[i](out: buffer<vec<float32, 4>[]>) {
  if (i < out.length) { out[i] = choose(vec(float32(i), -0.5, 0.0, 2.0)); }
}`)
	if err != nil {
		t.Fatal(err)
	}
	m, err := sema.CheckAndLower(a)
	if err != nil {
		t.Fatal(err)
	}
	out, err := emit(m)
	if err != nil {
		t.Fatal(err)
	}
	for _, want := range []string{"vec4<bool>", " & ", " | ", " != ", "all(", "any(", "select("} {
		if !strings.Contains(out, want) {
			t.Fatalf("mask WGSL missing %q:\n%s", want, out)
		}
	}
	if strings.Contains(out, " ^ ") {
		t.Fatalf("boolean XOR was not normalized to WGSL inequality:\n%s", out)
	}
	if err := wgsl.Validate(out); err != nil {
		t.Fatal(err)
	}
}

func TestViewProjectsStraightIntoStorageTexture(t *testing.T) {
	a, err := parser.Parse("view.tach", `
function paint[i](pixels: buffer<vec<float32, 4>[]>) {
  if (i < pixels.length) { pixels[i] = vec(0.1, 0.2, 0.3, 1.0); }
}
export function image(width: uint32, height: uint32): view<srgb8> {
  let pixels = transient<vec<float32, 4>>(width * height);
  run paint(pixels) over pixels.length;
  return view(pixels, width, height);
}`)
	if err != nil {
		t.Fatal(err)
	}
	m, err := sema.CheckAndLower(a)
	if err != nil {
		t.Fatal(err)
	}
	out, err := emit(m)
	if err != nil {
		t.Fatal(err)
	}
	for _, want := range []string{"texture_storage_2d<rgba8unorm, write>", "255.0", "0.5", "unpack4x8unorm(", "textureStore(", "@workgroup_size(256, 1, 1)"} {
		if !strings.Contains(out, want) {
			t.Fatalf("view WGSL missing %q:\n%s", want, out)
		}
	}
	if strings.Count(out, "@compute") != 1 || strings.Contains(out, "_tach_view_srgb") || strings.Contains(out, "_tach_view_source_") {
		t.Fatalf("view was not fused through shared pack math:\n%s", out)
	}
	if err := wgsl.Validate(out); err != nil {
		t.Fatal(err)
	}
}

func TestExternalViewUsesStandaloneProjection(t *testing.T) {
	a, err := parser.Parse("view.tach", `
function paint[i](pixels: buffer<vec<float32, 4>[]>) { pixels[i] = vec(0.1, 0.2, 0.3, 1.0); }
export function image(pixels: buffer<vec<float32, 4>[]>, width: uint32, height: uint32): view<srgb8> {
  run paint(pixels) over width * height;
  return view(pixels, width, height);
}`)
	if err != nil {
		t.Fatal(err)
	}
	m, err := sema.CheckAndLower(a)
	if err != nil {
		t.Fatal(err)
	}
	out, err := emit(m)
	if err != nil {
		t.Fatal(err)
	}
	if strings.Count(out, "@compute") != 2 || !strings.Contains(out, "@workgroup_size(16, 16, 1)") || !strings.Contains(out, "unpack4x8unorm(") || !strings.Contains(out, "255.0") || strings.Contains(out, "_tach_view_srgb") {
		t.Fatalf("standalone projection missing shared pack math:\n%s", out)
	}
}

func TestLogicalIndicesAreOptimizedAfterWGSLBackendLowering(t *testing.T) {
	a, err := parser.Parse("coordinates.tach", `
@workgroup(16, 8)
export function coordinates[x, y](out: buffer<uint32[]>) {
  let localX = x % 16;
  let localY = y % 8;
  let local = localY * 16 + localX;
  out[local] = local + x + y;
}`)
	if err != nil {
		t.Fatal(err)
	}
	m, err := sema.CheckAndLower(a)
	if err != nil {
		t.Fatal(err)
	}
	out, err := emit(m)
	if err != nil {
		t.Fatal(err)
	}
	for _, want := range []string{"@builtin(global_invocation_id)", "@builtin(local_invocation_index)", "_tach_local_linear"} {
		if !strings.Contains(out, want) {
			t.Fatalf("WGSL backend lowering missing %q:\n%s", want, out)
		}
	}
	if strings.Contains(out, "@builtin(local_invocation_id)") || strings.Contains(out, " % ") || strings.Contains(out, " * ") {
		t.Fatalf("WGSL backend left local-coordinate arithmetic in emitted code:\n%s", out)
	}
}
