package semantics

import (
	"errors"
	"os"
	"strings"
	"testing"

	"tach/foundation"
	"tach/ir"
	"tach/parser"
)

func analyzeSource(t *testing.T, name, source string) *ir.Module {
	t.Helper()
	parsed, err := parser.Parse(name, source)
	if err != nil {
		t.Fatal(err)
	}
	module, err := analyze(parsed)
	if err != nil {
		t.Fatal(err)
	}
	return module
}

func reject(t *testing.T, name, text, want string) foundation.Diagnostic {
	t.Helper()
	parsed, err := parser.Parse(name, text)
	if err != nil {
		t.Fatal(err)
	}
	_, err = analyze(parsed)
	if err == nil || !strings.Contains(err.Error(), want) {
		t.Fatalf("semantic error = %v, want %q", err, want)
	}
	var diagnostics foundation.Diagnostics
	if !errors.As(err, &diagnostics) || len(diagnostics) != 1 {
		t.Fatalf("semantic error = %#v, want one source diagnostic", err)
	}
	return diagnostics[0]
}

func TestParticlesEndToIR(t *testing.T) {
	var modules []*parser.File
	for _, name := range []string{"types", "particles"} {
		src, err := os.ReadFile("../../examples/simulation/" + name + ".tach")
		if err != nil {
			t.Fatal(err)
		}
		module, err := parser.Parse("simulation/"+name+".tach", string(src))
		if err != nil {
			t.Fatal(err)
		}
		module.Path = "simulation/" + name
		modules = append(modules, module)
	}
	m, _, err := analyzeProject(modules, 0)
	if err != nil {
		t.Fatal(err)
	}
	dump := ir.DumpKernel(m.Kernel)
	for _, want := range []string{"stage @integrate[i=%1]", "place.index", "call @integrateParticle"} {
		if !strings.Contains(dump, want) {
			t.Fatalf("IR missing %q:\n%s", want, dump)
		}
	}
	if strings.Contains(dump, "builtin") {
		t.Fatalf("Core IR leaked a backend builtin:\n%s", dump)
	}
}

func TestBuildProducesOneVerifiedOptimizedLogicalProgram(t *testing.T) {
	parsed, err := parser.Parse("build.tach", `
export function fill[i](out: buffer<uint32[]>) {
  let dead = sin(2.0);
  let first = i + 1;
  let second = i + 1;
  if (i < out.length) { out[i] = first + second; }
}`)
	if err != nil {
		t.Fatal(err)
	}
	result, err := Build([]*parser.File{parsed}, 1)
	if err != nil {
		t.Fatal(err)
	}
	if dump := ir.DumpKernel(result.Module.Kernel); strings.Contains(dump, "intrinsic sin") || strings.Count(dump, "const uint32 1") != 1 {
		t.Fatalf("logical program was not optimized once:\n%s", dump)
	}
	if err := ir.Verify(result.Module); err != nil {
		t.Fatalf("optimized logical program is invalid: %v", err)
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
	_, err = analyze(m)
	if err == nil || !strings.Contains(err.Error(), "duplicate parameter") {
		t.Fatalf("semantic error = %v, want duplicate parameter diagnostic", err)
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
	_, err = analyze(m)
	if err == nil || !strings.Contains(err.Error(), "requires at least one buffer parameter") {
		t.Fatalf("semantic error = %v, want buffer-parameter diagnostic", err)
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
	module, err := analyze(parsed)
	if err != nil {
		t.Fatal(err)
	}
	program := module.Programs[0]
	if program.Name != "image" || len(program.Resources) != 1 || program.Resources[0].Kind != ir.TransientResourceKind || program.View == nil || program.View.Format != ir.SRGB8ViewFormat || program.View.Source != program.Resources[0].ID {
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
		{`function paint[i](pixels: buffer<vec<float32, 4>[]>) { if (i < 1) { pixels[i] = vec(0.0, 0.0, 0.0, 1.0); } } export function image(): view<srgb8> { let pixels = transient<vec<float32, 4>>(4); run paint(pixels) over pixels.length; return view(pixels, 2, 2); }`, "fully defined before presentation"},
		{`function fill[i](data: buffer<uint32[]>) { data[i] = i; } function copy[i](input: buffer<uint32[]>, output: buffer<uint32[]>) { output[i] = input[i]; } export function incomplete(output: buffer<uint32[]>) { let scratch = transient<uint32>(8); run fill(scratch) over 4; run copy(scratch, output) over scratch.length; }`, "before every element has been defined"},
	}
	for _, test := range invalid {
		reject(t, "invalid-view.tach", test.source, test.want)
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
	module, _, err := analyzeProject([]*parser.File{shared, main}, 0)
	if err != nil {
		t.Fatal(err)
	}
	function := module.Kernel.Function("constants")
	if function == nil || function.Workgroup.Size != [3]uint32{8, 1, 1} || len(function.WorkgroupVars) != 1 || function.WorkgroupVars[0].Type.Count != 64 {
		t.Fatalf("constant structural lowering = %#v", function)
	}
	dump := ir.DumpKernel(module.Kernel)
	for _, want := range []string{"const uint32 16", "const float32 0.8", "const float32 0.6"} {
		if !strings.Contains(dump, want) {
			t.Fatalf("constant IR missing %q:\n%s", want, dump)
		}
	}
}

func TestCompileTimeConstantAlgebraCoversEveryValueKind(t *testing.T) {
	module := analyzeSource(t, "algebra.tach", `
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
	dump := ir.DumpKernel(module.Kernel)
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
		{"remainder", `const value = 1 % 0; export function bad[i](out: buffer<uint32[]>) { out[i] = value; }`, "remainder by zero"},
		{"signed division overflow", `const value: int32 = (-2147483647 - 1) / -1; export function bad[i](out: buffer<int32[]>) { out[i] = value; }`, "signed division overflows int32"},
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
			reject(t, "constants.tach", test.source, test.want)
		})
	}

	cycle, err := parser.Parse("cycle.tach", `const first = second + 1; const second = first + 1; export function bad[i](out: buffer<uint32[]>) { out[i] = first; }`)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := analyze(cycle); err == nil || !strings.Contains(err.Error(), "constant cycle") {
		t.Fatalf("constant cycle error = %v", err)
	}
	diagnostic := reject(t, "runtime-shape.tach", `export function bad(out: buffer<uint32[]>, count: uint32) { const blocks = ceilDiv(count, 256); }`, "not available in compile-time expressions")
	if !strings.Contains(diagnostic.Help, "use let") {
		t.Fatalf("runtime const help = %q", diagnostic.Help)
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
	module, err := analyze(m)
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
		if _, checkErr := analyze(parsed); checkErr == nil {
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
	_, err = analyze(m)
	if err == nil || !strings.Contains(err.Error(), "cannot assign to immutable value value") {
		t.Fatalf("semantic error = %v, want immutable-parameter diagnostic", err)
	}
}
