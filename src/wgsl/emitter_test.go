package wgsl_test

import (
	"os"
	"strings"
	"testing"

	"tach/src/parser"
	"tach/src/sema"
	"tach/src/wgsl"
)

func TestParticlesWGSL(t *testing.T) {
	src, err := os.ReadFile("../../examples/particles.tach")
	if err != nil {
		t.Fatal(err)
	}
	a, err := parser.Parse("particles.tach", string(src))
	if err != nil {
		t.Fatal(err)
	}
	m, err := sema.CheckAndLower(a)
	if err != nil {
		t.Fatal(err)
	}
	out, err := wgsl.Emit(m)
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
	bad := `@compute @workgroup_size(1, 1, 1) fn _tach_k_bad() { let _v1: u32 = 1u }`
	if err := wgsl.Validate(bad); err == nil {
		t.Fatal("WGSL validator accepted a generated statement without semicolon")
	}
}

func TestValidatorRejectsReservedDoubleUnderscoreIdentifier(t *testing.T) {
	bad := `@compute @workgroup_size(1, 1, 1) fn __tach_k_bad() { return; }`
	if err := wgsl.Validate(bad); err == nil || !strings.Contains(err.Error(), "reserved identifier") {
		t.Fatalf("WGSL validator error = %v, want reserved identifier", err)
	}
}

func TestRuntimeArrayWrapperKeepsNaturalAlignment(t *testing.T) {
	a, err := parser.Parse("runtime.tach", `
@workgroupSize(1)
export compute clear(data: storage<u32[], read_write>) {
  if (globalId.x < data.length) { data[globalId.x] = 0u; }
}`)
	if err != nil {
		t.Fatal(err)
	}
	m, err := sema.CheckAndLower(a)
	if err != nil {
		t.Fatal(err)
	}
	out, err := wgsl.Emit(m)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(out, "data: array<u32>,") || strings.Contains(out, "@align(16) data: array<u32>,") {
		t.Fatalf("runtime array wrapper has a synthetic fixed-resource alignment:\n%s", out)
	}
}
