package sema_test

import (
	"os"
	"strings"
	"testing"

	"tach/src/ast"
	"tach/src/ir"
	"tach/src/parser"
	"tach/src/sema"
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
