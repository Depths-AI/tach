package sema_test

import (
	"math"
	"os"
	"strings"
	"testing"

	"tach/src/ast"
	"tach/src/flow"
	"tach/src/ir"
	"tach/src/parser"
	"tach/src/sema"
	"tach/src/types"
)

func lower(t *testing.T, name, source string) *flow.Module {
	t.Helper()
	parsed, err := parser.Parse(name, source)
	if err != nil {
		t.Fatal(err)
	}
	module, err := sema.CheckAndLower(parsed)
	if err != nil {
		t.Fatal(err)
	}
	return module
}

func TestParticlesEndToIR(t *testing.T) {
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
	dump := ir.Dump(m.Kernel)
	for _, want := range []string{"stage @integrate[i=%1]", "place.index", "call @integrateParticle"} {
		if !strings.Contains(dump, want) {
			t.Fatalf("IR missing %q:\n%s", want, dump)
		}
	}
	if strings.Contains(dump, "builtin") {
		t.Fatalf("Core IR leaked a backend builtin:\n%s", dump)
	}
}

func TestDuplicateComputeParameterNameRejected(t *testing.T) {
	m, err := parser.Parse("dup-param.tach", `
@workgroup(1)
export function first[i](shared: buffer<uint32[]>) { }
@workgroup(1)
export function second[i](
  shared: buffer<uint32[]>,
  shared: buffer<uint32[]>
) { }
`)
	if err != nil {
		t.Fatal(err)
	}
	_, err = sema.CheckAndLower(m)
	if err == nil || !strings.Contains(err.Error(), "duplicate parameter") {
		t.Fatalf("CheckAndLower error = %v, want duplicate parameter diagnostic", err)
	}
}

func TestComputeKernelRequiresBuffer(t *testing.T) {
	m, err := parser.Parse("value-only.tach", `
@workgroup(1)
export function invisible[i](params: uint32) { }
`)
	if err != nil {
		t.Fatal(err)
	}
	_, err = sema.CheckAndLower(m)
	if err == nil || !strings.Contains(err.Error(), "requires at least one buffer parameter") {
		t.Fatalf("CheckAndLower error = %v, want buffer-parameter diagnostic", err)
	}
}

func TestViewProgramsAreStructuredAndValidated(t *testing.T) {
	valid := `
function paint[i](pixels: buffer<vec<float32, 4>[]>) {
  if (i < pixels.length) { pixels[i] = vec(0.1, 0.2, 0.3, 1.0); }
}
export function image(width: uint32, height: uint32): view<srgb8> {
  let pixels = transient<vec<float32, 4>>(width * height);
  run paint(pixels) over pixels.length;
  return view(pixels, width, height);
}`
	parsed, err := parser.Parse("view.tach", valid)
	if err != nil {
		t.Fatal(err)
	}
	module, err := sema.CheckAndLower(parsed)
	if err != nil {
		t.Fatal(err)
	}
	program := module.Programs[0]
	if program.Name != "image" || len(program.Resources) != 1 || program.Resources[0].Kind != flow.Transient || program.View == nil || program.View.Format != flow.SRGB8 || program.View.Source != program.Resources[0].ID {
		t.Fatalf("view program = %#v", program)
	}
	invalid := []struct {
		source string
		want   string
	}{
		{`function image(): view<srgb8> { return view(missing, 1, 1); }`, "only valid on an exported program"},
		{`export function image[i](pixels: buffer<vec<float32, 4>[]>): view<srgb8> {}`, "indexed stage image cannot declare a return type"},
		{`export function image(data: buffer<uint32[]>): uint32 { return 1; }`, "can only return view<srgb8>"},
		{strings.Replace(valid, "return view(pixels, width, height);", "", 1), "must return its view"},
		{strings.Replace(valid, "view(pixels, width, height)", "view(pixels, width)", 1), "view(pixels, width, height)"},
		{`function paint[i](pixels: buffer<uint32[]>) { pixels[i] = 0; } export function image(): view<srgb8> { let pixels = transient<uint32>(1); run paint(pixels) over 1; return view(pixels, 1, 1); }`, "vec<float32, 4> buffer or transient"},
		{strings.Replace(valid, "view(pixels, width, height)", "view(pixels, 1.0, height)", 1), "uint32 literal must be an integer"},
		{strings.Replace(valid, "view<srgb8>", "view<rgba8>", 1), "can only return view<srgb8>"},
	}
	for _, test := range invalid {
		parsed, parseErr := parser.Parse("invalid-view.tach", test.source)
		if parseErr != nil {
			t.Fatal(parseErr)
		}
		if _, checkErr := sema.CheckAndLower(parsed); checkErr == nil || !strings.Contains(checkErr.Error(), test.want) {
			t.Fatalf("CheckAndLower error = %v, want %q", checkErr, test.want)
		}
	}
}

func TestCompileTimeConstantsAreEvaluatedAndSubstituted(t *testing.T) {
	shared, err := parser.Parse("shared/constants.tach", `
const tile: uint32 = 8;
const tileArea = tile * tile;
const tint: vec<float32, 3> = normalize(vec(3.0, 4.0, 0.0));
`)
	if err != nil {
		t.Fatal(err)
	}
	main, err := parser.Parse("app/main.tach", `
import "shared/constants";

@workgroup(tile)
export function constants[i](out: buffer<vec<float32, 3>[]>) {
  const doubled = tile * 2;
  const reversed = tint.zyx;
  let scratch: shared<uint32[tileArea]>;
  scratch[i] = doubled;
  workgroupBarrier();
  if (i < out.length) {
    out[i] = reversed * float32(scratch[i]);
  }
}`)
	if err != nil {
		t.Fatal(err)
	}
	module, _, err := sema.CheckAndLowerProject([]*ast.Module{shared, main})
	if err != nil {
		t.Fatal(err)
	}
	function := module.Kernel.Function("constants")
	if function == nil || function.Workgroup.Size != [3]uint32{8, 1, 1} || len(function.WorkgroupVars) != 1 || function.WorkgroupVars[0].Type.Count != 64 {
		t.Fatalf("constant structural lowering = %#v", function)
	}
	dump := ir.Dump(module.Kernel)
	for _, want := range []string{"const uint32 16", "const float32 0.8", "const float32 0.6"} {
		if !strings.Contains(dump, want) {
			t.Fatalf("constant IR missing %q:\n%s", want, dump)
		}
	}
}

func TestCompileTimeConstantAlgebraCoversEveryValueKind(t *testing.T) {
	module := lower(t, "algebra.tach", `
const enabled = !false && 3 < 4;
const signedResult: int32 = -7 + 2 * 3;
const maskedShift: uint32 = ((0xff & 0x0f) << 36) | 2;
const unsignedResult = maskedShift + (enabled ? 1 : 0);
const halfResult: float16 = fma(0.5, 2.0, 0.25);
const floatResult: float32 = sqrt(9.0) + pow(2.0, 3.0);
const allBits = uint32(-1);
const unit = normalize(vec(0.0, 3.0, 4.0));
const selected = unit.zyx[1];
const integerMath = clamp(min(abs(-7), max(3, 4)), 1, 6);
const rounded = floor(1.75) + ceil(1.25) + trunc(1.75);
const trigonometry = sin(0.25) + cos(0.25) + tan(0.25);
const exponential = exp(0.25) + exp2(0.25) + log(2.0) + log2(2.0);
const roots = rsqrt(4.0);
const geometry = dot(vec(1.0, 0.0, 0.0), vec(1.0, 0.0, 0.0))
  + length(vec(3.0, 4.0)) + distance(vec(1.0, 1.0), vec(4.0, 5.0))
  + cross(vec(1.0, 0.0, 0.0), vec(0.0, 1.0, 0.0)).z;
const broadcast = fma(vec(1.0, 2.0), 2.0, vec(3.0, 4.0));
const lazyLogic = false && 1 / 0 == 0;
const lazyChoice = true ? 4 : 1 / 0;

export function algebra[i](
  unsignedOut: buffer<uint32[]>,
  signedOut: buffer<int32[]>,
  halfOut: buffer<float16[]>,
  floatOut: buffer<float32[]>,
  vectorOut: buffer<vec<float32, 3>[]>,
) {
  if (i == 0) {
    unsignedOut[i] = unsignedResult + (allBits & 0) + uint32(integerMath)
      + uint32(lazyLogic ? 1 : 0) + lazyChoice;
    signedOut[i] = signedResult;
    halfOut[i] = halfResult;
    floatOut[i] = floatResult + selected + rounded + trigonometry + exponential
      + roots + geometry;
    vectorOut[i] = unit * broadcast.x;
  }
}`)
	dump := ir.Dump(module.Kernel)
	for _, want := range []string{
		"const uint32 243",
		"const uint32 4294967295",
		"const int32 -1",
		"const float16 1.25",
		"const float32 11.0",
		"const float32 0.6",
		"const float32 0.8",
	} {
		if !strings.Contains(dump, want) {
			t.Fatalf("constant algebra missing %q:\n%s", want, dump)
		}
	}
}

func TestCompileTimeConstantErrorsAreSpecific(t *testing.T) {
	tests := []struct {
		name, source, want string
	}{
		{"runtime parameter", `export function bad[i](out: buffer<uint32[]>, value: uint32) { const doubled = value * 2; out[i] = doubled; }`, "depends on runtime value \"value\""},
		{"runtime coordinate", `export function bad[i](out: buffer<uint32[]>) { const lane = i; out[i] = lane; }`, "depends on runtime value \"i\""},
		{"helper call", `function twice(value: uint32): uint32 { return value * 2; } export function bad[i](out: buffer<uint32[]>) { const value = twice(2); out[i] = value; }`, "not available in compile-time expressions"},
		{"assignment", `const value = 2; export function bad[i](out: buffer<uint32[]>) { value = 3; out[i] = value; }`, "compile-time constant value"},
		{"division", `const value = 1 / 0; export function bad[i](out: buffer<uint32[]>) { out[i] = value; }`, "division by zero"},
		{"non-finite", `const value = sqrt(-1.0); export function bad[i](out: buffer<float32[]>) { out[i] = value; }`, "non-finite"},
		{"vector index", `const value = vec(1, 2)[2]; export function bad[i](out: buffer<uint32[]>) { out[i] = value; }`, "outside its lanes"},
		{"invalid type", `type Pair = { x: uint32 }; const value: Pair = { x: 1 }; export function bad[i](out: buffer<uint32[]>) { out[i] = value.x; }`, "must be a scalar or vector"},
		{"array zero", `const count = 0; export function bad[i](out: buffer<uint32[]>) { let scratch: shared<uint32[count]>; out[i] = i; }`, "positive uint32 constant"},
		{"local declaration order", `export function bad[i](out: buffer<uint32[]>) { const value = later; const later = 1; out[i] = value; }`, "unknown identifier \"later\""},
		{"transient", `export function bad[i](out: buffer<uint32[]>) { const value = transient<uint32>(4); out[i] = 0; }`, "only available as a public program let binding"},
		{"program parameter", `function fill[i](out: buffer<uint32[]>, value: uint32) { out[i] = value; } export function bad(out: buffer<uint32[]>, value: uint32) { const fixed = value; run fill(out, fixed) over out.length; }`, "depends on runtime value \"value\""},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			module, parseErr := parser.Parse("constants.tach", test.source)
			if parseErr != nil {
				t.Fatal(parseErr)
			}
			if _, checkErr := sema.CheckAndLower(module); checkErr == nil || !strings.Contains(checkErr.Error(), test.want) {
				t.Fatalf("CheckAndLower error = %v, want %q", checkErr, test.want)
			}
		})
	}

	cycle, err := parser.Parse("cycle.tach", `const first = second + 1; const second = first + 1; export function bad[i](out: buffer<uint32[]>) { out[i] = first; }`)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := sema.CheckAndLower(cycle); err == nil || !strings.Contains(err.Error(), "constant cycle") {
		t.Fatalf("constant cycle error = %v", err)
	}
}

func TestDocumentationIsNormalizedAndValidated(t *testing.T) {
	m, err := parser.Parse("docs.tach", `
@docs(title("Particles"), summary("Simulation kernels."));
@docs(summary("Particle data."), field(position, "World position."))
type Particle = { position: vec<float32, 4> };
@docs(summary("Advance one particle."), param(particle, "Current state."), returns("Updated state."))
function advance(particle: Particle): Particle { return particle; }
@docs(summary("Advance all particles."), coordinate(i, "Particle index."), param(particles, "Mutable state."))
export function step[i](particles: buffer<Particle[]>) { if (i < particles.length) { particles[i] = advance(particles[i]); } }
`)
	if err != nil {
		t.Fatal(err)
	}
	module, err := sema.CheckAndLower(m)
	if err != nil {
		t.Fatal(err)
	}
	if module.Documentation.Title != "Particles" || module.Documentation.Types["Particle"].Fields["position"] != "World position." || module.Documentation.Functions["step"].Coordinates["i"] != "Particle index." {
		t.Fatalf("documentation = %#v", module.Documentation)
	}
	for _, source := range []string{
		`@docs(summary("X"), field(missing, "No.")) type X = { value: uint32 }; export function k[i](out: buffer<uint32[]>) {}`,
		`@docs(summary("K"), coordinate(x, "No.")) export function k[i](out: buffer<uint32[]>) {}`,
		`@docs(returns("No."), summary("K")) export function k[i](out: buffer<uint32[]>) {}`,
		`@docs(param(out, "No.")) export function k[i](out: buffer<uint32[]>) {}`,
		`@docs(summary("One."), summary("Two.")) export function k[i](out: buffer<uint32[]>) {}`,
		`@docs(summary("K"), param("out", "Quoted.")) export function k[i](out: buffer<uint32[]>) {}`,
		`@docs(summary("X"), title("Wrong context.")) type X = { value: uint32 }; export function k[i](out: buffer<uint32[]>) {}`,
	} {
		parsed, parseErr := parser.Parse("invalid-docs.tach", source)
		if parseErr != nil {
			t.Fatal(parseErr)
		}
		if _, checkErr := sema.CheckAndLower(parsed); checkErr == nil {
			t.Fatalf("accepted invalid documentation: %s", source)
		}
	}
}

func TestKernelValueParameterIsImmutable(t *testing.T) {
	m, err := parser.Parse("immutable-parameter.tach", `
export function invalid[i](out: buffer<uint32[]>, value: uint32) {
  value = i;
}
`)
	if err != nil {
		t.Fatal(err)
	}
	_, err = sema.CheckAndLower(m)
	if err == nil || !strings.Contains(err.Error(), "cannot assign to immutable value value") {
		t.Fatalf("CheckAndLower error = %v, want immutable-parameter diagnostic", err)
	}
}

func TestFloat16LanguageSurface(t *testing.T) {
	valid := `
function shape(value: vec<float16, 3>): vec<float16, 3> {
  let direction = normalize(value);
  let magnitude = length(value) + distance(value, direction) + dot(value, direction);
  let crossed = cross(value, direction);
  return crossed + direction * magnitude;
}
export function halfMath[i](values: buffer<vec<float16, 4>[]>, factor: float16) {
  if (i < values.length) {
    let value = values[i];
    let geometry = shape(value.xyz);
    let wave = sin(value.x) + cos(value.y) + tan(value.z);
    let exponential = exp(value.x) + exp2(value.y) + log(value.z) + log2(value.w);
    let rooted = sqrt(abs(value.x)) + rsqrt(value.w);
    let rounded = floor(value.x) + ceil(value.y) + trunc(value.z);
    let converted = float16(float32(value.w));
    values[i] = vec(geometry.x + pow(wave, factor), exponential, rooted + rounded, converted);
  }
}`
	parsed, err := parser.Parse("float16.tach", valid)
	if err != nil {
		t.Fatal(err)
	}
	module, err := sema.CheckAndLower(parsed)
	if err != nil {
		t.Fatal(err)
	}
	dump := ir.Dump(module.Kernel)
	for _, want := range []string{"float16", "vec<float16, 4>", "intrinsic normalize", ": float16 -> float32", ": float32 -> float16"} {
		if !strings.Contains(dump, want) {
			t.Fatalf("Float16 IR missing %q:\n%s", want, dump)
		}
	}

	for _, test := range []struct{ source, want string }{
		{`export function bad[i](out: buffer<float16[]>) { if (i < out.length) { out[i] = float16(70000.0); } }`, "invalid float16 literal"},
		{`export function bad[i](out: buffer<float16[]>) { if (i < out.length) { out[i] = out[i] + float32(out[i]); } }`, "matching numeric operands"},
	} {
		parsed, parseErr := parser.Parse("invalid-float16.tach", test.source)
		if parseErr != nil {
			t.Fatal(parseErr)
		}
		if _, checkErr := sema.CheckAndLower(parsed); checkErr == nil || !strings.Contains(checkErr.Error(), test.want) {
			t.Fatalf("CheckAndLower error = %v, want %q", checkErr, test.want)
		}
	}
}

func TestLoopControlAndFmaLowerToCoreIR(t *testing.T) {
	parsed, err := parser.Parse("control.tach", `
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
	if err != nil {
		t.Fatal(err)
	}
	module, err := sema.CheckAndLower(parsed)
	if err != nil {
		t.Fatal(err)
	}
	dump := ir.Dump(module.Kernel)
	for _, want := range []string{"intrinsic fma", "continue [", "break ["} {
		if !strings.Contains(dump, want) {
			t.Fatalf("control/FMA IR missing %q:\n%s", want, dump)
		}
	}

	for _, test := range []struct{ source, want string }{
		{`export function bad[i](out: buffer<uint32[]>) { break; }`, "break is only valid inside a loop"},
		{`export function bad[i](out: buffer<uint32[]>) { continue; }`, "continue is only valid inside a loop"},
		{`export function bad[i](out: buffer<uint32[]>) { out[i] = fma(1, 2, 3); }`, "cannot satisfy uint32 context"},
		{`export function bad[i](out: buffer<float32[]>) { out[i] = fma(1.0, 2.0); }`, "expects 3 argument"},
		{`export function bad[i](state: buffer<atomic<uint32>[]>) { atomicCompareExchange(state[i], 0); }`, "expects 3 argument"},
		{`export function bad[i](state: buffer<uint32[]>) { atomicCompareExchange(state[i], 0, 1); }`, "requires an atomic"},
		{`export function bad[i](state: buffer<atomic<uint32>[]>) { atomicCompareExchange(state[i], 0, -1); }`, "unary - requires"},
		{`export function bad[i](out: buffer<uint32[]>) { if (min(true, false)) { out[i] = 0; } }`, "requires numeric values"},
	} {
		parsed, parseErr := parser.Parse("invalid-control.tach", test.source)
		if parseErr != nil {
			t.Fatal(parseErr)
		}
		if _, checkErr := sema.CheckAndLower(parsed); checkErr == nil || !strings.Contains(checkErr.Error(), test.want) {
			t.Fatalf("CheckAndLower error = %v, want %q", checkErr, test.want)
		}
	}
}

func TestFloatBoundsAndStrongCompareExchangeLowerToCoreIR(t *testing.T) {
	module := lower(t, "bounds-atomic.tach", `
const bounded: float32 = clamp(0.5, 2.0, 1.0);
export function boundsAtomic[i](values: buffer<vec<float32, 2>[]>, half: buffer<float16[]>, state: buffer<atomic<uint32>[]>) {
  if (i < values.length && i < half.length && i < state.length) {
    values[i] = clamp(values[i], min(values[i], -bounded), max(values[i], bounded));
    half[i] = clamp(half[i], float16(-1), float16(1));
    let observed = atomicCompareExchange(state[i], 0, 1);
    if (observed != 0) { atomicAdd(state[i], 1); }
  }
}`)
	dump := ir.Dump(module.Kernel)
	for _, want := range []string{"const float32 1", "intrinsic min", "intrinsic max", "intrinsic clamp", "atomic_compare_exchange", "vec<float32, 2>", "float16"} {
		if !strings.Contains(dump, want) {
			t.Fatalf("bounds/CAS IR missing %q:\n%s", want, dump)
		}
	}
}

func TestContextualNumericInference(t *testing.T) {
	parsed, err := parser.Parse("inference.tach", `
function inferredFloat(): float32 {
  return fma(2, 3, 4) + sin(1) + pow(2, 3);
}
function inferredHalf(): float16 {
  return fma(2, 3, 4);
}
function inferredSigned(): int32 {
  return abs(1);
}
function inferredVector(scale: float32): vec<float32, 3> {
  let axis = normalize(vec(1, 0, 0));
  return fma(axis, scale, vec(0, 1, 0));
}
function inferredDot(): float16 {
  return dot(vec(1, 2, 3), vec(4, 5, 6));
}
function inferredBinary(value: float16): float16 {
  return -0.745 - value * value;
}
function inferredFill(): vec<float16, 3> {
  return fma(1, 2, 3);
}
function inferredConditional(enabled: bool, value: float16): float16 {
  return enabled ? 1 + 2 : -0.5 - value;
}
function inferredIntegerConditional(enabled: bool): int32 {
  return enabled ? 1 : -2;
}
function inferredCompound(value: vec<float16, 3>): vec<float16, 3> {
  let result = value;
  result += sin(1);
  result *= 1 + 2;
  return result;
}
function inferredPlaceCompound[i](values: buffer<vec<float16, 3>[]>) {
  values[i] += sin(1);
}
export function inferred[i](out: buffer<vec<float32, 4>[]>) {
  if (i < out.length) {
    let direction = inferredVector(2.0);
    let fill = inferredFill();
    out[i] = vec(direction, inferredFloat() + float32(inferredHalf()) + float32(inferredSigned()) + float32(inferredDot()) + float32(inferredBinary(fill.x)) + float32(inferredConditional(true, fill.y)) + float32(inferredIntegerConditional(false)));
  }
}`)
	if err != nil {
		t.Fatal(err)
	}
	module, err := sema.CheckAndLower(parsed)
	if err != nil {
		t.Fatal(err)
	}
	dump := ir.Dump(module.Kernel)
	for _, want := range []string{"intrinsic fma", "intrinsic sin", "intrinsic pow", "intrinsic abs", "intrinsic normalize", "intrinsic dot", "vec<float16, 3>", "vec<float32, 4>"} {
		if !strings.Contains(dump, want) {
			t.Fatalf("inferred IR missing %q:\n%s", want, dump)
		}
	}

	for _, test := range []struct{ source, want string }{
		{`export function bad[i](out: buffer<float32[]>) { out[i] = vec(1); }`, "want 2, 3, or 4"},
		{`export function bad[i](out: buffer<vec<float32, 4>[]>) { out[i] = vec(1, 2, 3, 4, 5); }`, "want 2, 3, or 4"},
		{`export function bad[i](out: buffer<vec<float32, 2>[]>) { out[i] = vec(float16(1), float32(2)); }`, "convert explicitly"},
		{`export function bad[i](out: buffer<vec<float32, 2>[]>) { out[i] = fma(vec(1, 1), vec(2, 2, 2), vec(3, 3)); }`, "conflicting vector widths"},
		{`export function bad[i](out: buffer<float32[]>) { out[i] = dot(1, 2); }`, "requires floating-point vectors"},
		{`export function bad[i](out: buffer<vec<float32, 2>[]>) { out[i] = pow(2, vec(3, 4)); }`, "argument is float32, want vec<float32, 2>"},
		{`export function bad[i](out: buffer<vec<uint32, 2>[]>) { out[i] = vec(true, 1); }`, "components use uint32 and bool"},
		{`export function bad[i](out: buffer<vec<float32, 2>[]>) { out[i] = true ? 1 : vec(2, 3); }`, "conditional branches have types"},
		{`export function bad[i](out: buffer<vec<bool, 2>[]>) { out[i] = vec(1, 2); }`, "not host-shareable"},
		{`export function bad[i](out: buffer<vec<float32, 5>[]>) { out[i] = vec(1, 2, 3, 4); }`, "vec lane count must be 2, 3, or 4"},
		{`function vec(value: uint32): uint32 { return value; } export function bad[i](out: buffer<uint32[]>) { out[i] = i; }`, "reserved"},
	} {
		parsed, parseErr := parser.Parse("invalid-inference.tach", test.source)
		if parseErr != nil {
			t.Fatal(parseErr)
		}
		if _, checkErr := sema.CheckAndLower(parsed); checkErr == nil || !strings.Contains(checkErr.Error(), test.want) {
			t.Fatalf("CheckAndLower error = %v, want %q", checkErr, test.want)
		}
	}
}

func TestBooleanVectorsAndMasksLowerToCoreIR(t *testing.T) {
	module := lower(t, "masks.tach", `
const policy: vec<bool, 4> = vec(true, false, true, true);
const defaults: vec<float32, 4> = select(policy, vec(1, 2, 3, 4), 0);
const valid: bool = all((defaults > 0.0) | !policy);
function choose(value: vec<float32, 4>): vec<float32, 4> {
  let inside = value >= -1.0 & value <= 1.0;
  let changed = inside ^ vec(false, true, false, true);
  let selected = select(changed | value == 0.0, abs(value), -value);
  return (any(changed) || all(inside)) && valid ? selected : defaults;
}
export function masks[i](out: buffer<vec<float32, 4>[]>) {
  if (i < out.length) { out[i] = choose(vec(float32(i) - 2.0, -0.5, 0.0, 2.0)); }
}`)
	dump := ir.Dump(module.Kernel)
	for _, want := range []string{"vec<bool, 4>", "intrinsic all", "intrinsic any", "intrinsic select", " = & ", " = | ", " = ^ "} {
		if !strings.Contains(dump, want) {
			t.Fatalf("mask IR missing %q:\n%s", want, dump)
		}
	}

	for _, test := range []struct{ source, want string }{
		{`export function bad[i](out: buffer<vec<bool, 2>[]>) {}`, "not host-shareable"},
		{`export function bad[i](out: buffer<uint32[]>) { let x = all(true); out[i] = x ? 1 : 0; }`, "requires vec<bool, N>"},
		{`export function bad[i](out: buffer<uint32[]>) { let x = select(true, vec(1, 2), vec(3, 4)); out[i] = x.x; }`, "mask must be vec<bool, N>"},
		{`export function bad[i](out: buffer<uint32[]>) { let x = select(vec(true, false), vec(1, 2, 3), vec(4, 5, 6)); out[i] = x.x; }`, "conflicting vector widths"},
		{`export function bad[i](out: buffer<uint32[]>) { let x = select(vec(true, false), true, 0); out[i] = x.x ? 1 : 0; }`, "boolean arm"},
		{`export function bad[i](out: buffer<uint32[]>) { if (vec(1, 2) < 3) { out[i] = 1; } }`, "want bool"},
		{`export function bad[i](out: buffer<uint32[]>) { let x = vec(true, false) < vec(false, true); out[i] = x.x ? 1 : 0; }`, "requires numeric values"},
	} {
		parsed, parseErr := parser.Parse("invalid-mask.tach", test.source)
		if parseErr != nil {
			t.Fatal(parseErr)
		}
		if _, checkErr := sema.CheckAndLower(parsed); checkErr == nil || !strings.Contains(checkErr.Error(), test.want) {
			t.Fatalf("CheckAndLower error = %v, want %q", checkErr, test.want)
		}
	}
}

func TestLoopControlPreservesBarrierUniformity(t *testing.T) {
	for _, source := range []string{
		`@workgroup(4) export function valid[i](out: buffer<uint32[]>) { for (let step = 0; step < 4; step++) { if (step == 2) { break; } } workgroupBarrier(); out[i] = i; }`,
		`@workgroup(4) export function valid[i](out: buffer<uint32[]>) { for (let step = 0; step < 4; step++) { if (step == 2) { continue; } } workgroupBarrier(); out[i] = i; }`,
	} {
		parsed, err := parser.Parse("uniform-control.tach", source)
		if err != nil {
			t.Fatal(err)
		}
		if _, err := sema.CheckAndLower(parsed); err != nil {
			t.Fatalf("uniform loop transfer rejected: %v", err)
		}
	}
	for _, keyword := range []string{"break", "continue"} {
		parsed, err := parser.Parse("varying-control.tach", `
@workgroup(4)
export function invalid[i](out: buffer<uint32[]>) {
  for (let step = 0; step < 4; step++) {
    if (i == step) { `+keyword+`; }
  }
  workgroupBarrier();
  out[i] = i;
}`)
		if err != nil {
			t.Fatal(err)
		}
		if _, err := sema.CheckAndLower(parsed); err == nil || !strings.Contains(err.Error(), "non-uniform control flow") {
			t.Fatalf("varying %s barrier error = %v", keyword, err)
		}
	}
}

func TestFloat16Encoding(t *testing.T) {
	for _, test := range []struct {
		value float64
		bits  uint16
	}{
		{0, 0x0000}, {math.Copysign(0, -1), 0x8000}, {1, 0x3c00}, {-2, 0xc000}, {65504, 0x7bff},
		{0.00006103515625, 0x0400}, {0.000000059604644775390625, 0x0001},
		{1.00048828125, 0x3c00}, {1.00146484375, 0x3c02},
		{math.Nextafter(1.00048828125, 2), 0x3c01}, {math.Nextafter(1.00146484375, 1), 0x3c01},
	} {
		bits, ok := types.Float16bits(test.value)
		if !ok || bits != test.bits {
			t.Errorf("Float16bits(%g) = %#04x, %v; want %#04x, true", test.value, bits, ok, test.bits)
		}
	}
	for _, value := range []float64{65505, -65505} {
		if _, ok := types.Float16bits(value); ok {
			t.Errorf("Float16bits(%g) accepted an out-of-range value", value)
		}
	}
}

func FuzzSemanticCheckingReturnsSourceDiagnostics(f *testing.F) {
	for _, seed := range []string{
		`function twice(x: float32): float32 { return x * 2.0; } export function scale[i](out: buffer<float32[]>) { if (i < out.length) { out[i] = twice(out[i]); } }`,
		`function copy[i](source: buffer<float32[]>, target: buffer<float32[]>) { target[i] = source[i]; } export function run(source: buffer<float32[]>, target: buffer<float32[]>) { let count = min(source.length, target.length); let scratch = transient<float32>(count); run copy(source, scratch) over count; run copy(scratch, target) over count; }`,
		`export function invalid[i](out: buffer<uint32[]>) { out[true] = missing; }`,
	} {
		f.Add(seed)
	}
	f.Fuzz(func(t *testing.T, input string) {
		// DECISION: bound one fuzz case at 64 KiB; project tests own larger-file pressure, and this cap can rise with semantic limits.
		if len(input) > 64<<10 {
			t.Skip()
		}
		module, err := parser.Parse("fuzz.tach", input)
		if err != nil {
			return
		}
		_, err = sema.CheckAndLower(module)
		if err != nil && strings.Contains(err.Error(), "internal ") {
			t.Fatalf("user source reached an internal verifier diagnostic: %v", err)
		}
	})
}
