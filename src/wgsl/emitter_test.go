package wgsl_test

import (
	"os"
	"strings"
	"testing"

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
@workgroup(1)
export function clear[i](data: buffer<uint32[]>) {
  if (i < data.length) { data[i] = 0; }
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
}

func TestLogicalIndicesAreOptimizedAfterWGSLBackendLowering(t *testing.T) {
	a, err := parser.Parse("coordinates.tach", `
@workgroup(16, 8)
export function coordinates[x, y](out: buffer<uint32[]>) {
  const localX = x % 16;
  const localY = y % 8;
  const local = localY * 16 + localX;
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
