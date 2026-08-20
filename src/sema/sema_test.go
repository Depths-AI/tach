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
function paint[i](pixels: buffer<float32x4[]>) {
  if (i < pixels.length) { pixels[i] = float32x4(0.1, 0.2, 0.3, 1.0); }
}
export function image(width: uint32, height: uint32): view<srgb8> {
  const pixels = transient<float32x4>(width * height);
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
		{`export function image[i](pixels: buffer<float32x4[]>): view<srgb8> {}`, "indexed stage image cannot declare a return type"},
		{`export function image(data: buffer<uint32[]>): uint32 { return 1; }`, "can only return view<srgb8>"},
		{strings.Replace(valid, "return view(pixels, width, height);", "", 1), "must return its view"},
		{strings.Replace(valid, "view(pixels, width, height)", "view(pixels, width)", 1), "view(pixels, width, height)"},
		{`function paint[i](pixels: buffer<uint32[]>) { pixels[i] = 0; } export function image(): view<srgb8> { const pixels = transient<uint32>(1); run paint(pixels) over 1; return view(pixels, 1, 1); }`, "float32x4 buffer or transient"},
		{strings.Replace(valid, "view(pixels, width, height)", "view(pixels, 1.0, height)", 1), "shape literal must be uint32"},
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

func TestDocumentationIsNormalizedAndValidated(t *testing.T) {
	m, err := parser.Parse("docs.tach", `
@docs(title("Particles"), summary("Simulation kernels."));
@docs(summary("Particle data."), field(position, "World position."))
type Particle = { position: float32x4 };
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
function shape(value: float16x3): float16x3 {
  const direction = normalize(value);
  const magnitude = length(value) + distance(value, direction) + dot(value, direction);
  const crossed = cross(value, direction);
  return crossed + direction * float16x3(magnitude);
}
export function halfMath[i](values: buffer<float16x4[]>, factor: float16) {
  if (i < values.length) {
    const value = values[i];
    const geometry = shape(value.xyz);
    const wave = sin(value.x) + cos(value.y) + tan(value.z);
    const exponential = exp(value.x) + exp2(value.y) + log(value.z) + log2(value.w);
    const rooted = sqrt(abs(value.x)) + rsqrt(value.w);
    const rounded = floor(value.x) + ceil(value.y) + trunc(value.z);
    const converted = float16(float32(value.w));
    values[i] = float16x4(geometry.x + pow(wave, factor), exponential, rooted + rounded, converted);
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
	for _, want := range []string{"float16", "float16x4", "intrinsic normalize", ": float16 -> float32", ": float32 -> float16"} {
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
export function control[i](out: buffer<float32[]>, half: buffer<float16[]>, limit: uint32) {
  let total: float32 = 0;
  for (let step = 0; step < limit; step++) {
    if (step == 2) { continue; }
    total = fma(float32(step), 0.5, total);
    if (total > 10.0) { break; }
  }
  if (i < out.length && i < half.length) {
    out[i] = total;
    half[i] = fma(half[i], float16(2), float16(1));
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
		{`export function bad[i](out: buffer<uint32[]>) { out[i] = fma(1, 2, 3); }`, "matching floating-point"},
		{`export function bad[i](out: buffer<float32[]>) { out[i] = fma(1.0, 2.0); }`, "expects 3 argument"},
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
		`function copy[i](source: buffer<float32[]>, target: buffer<float32[]>) { target[i] = source[i]; } export function run(source: buffer<float32[]>, target: buffer<float32[]>) { const count = min(source.length, target.length); const scratch = transient<float32>(count); run copy(source, scratch) over count; run copy(scratch, target) over count; }`,
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
