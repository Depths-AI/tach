package sema_test

import (
	"os"
	"strings"
	"tach/src/ir"
	"tach/src/parser"
	"tach/src/sema"
	"testing"
)

func TestParticlesEndToIR(t *testing.T) {
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
