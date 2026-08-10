package sema_test

import (
	"os"
	"pine/internal/ir"
	"pine/internal/parser"
	"pine/internal/sema"
	"strings"
	"testing"
)

func TestParticlesEndToIR(t *testing.T) {
	src, err := os.ReadFile("../../examples/particles.pine")
	if err != nil {
		t.Fatal(err)
	}
	a, err := parser.Parse("particles.pine", string(src))
	if err != nil {
		t.Fatal(err)
	}
	m, err := sema.CheckAndLower(a)
	if err != nil {
		t.Fatal(err)
	}
	dump := ir.Dump(m)
	for _, want := range []string{"compute @integrate", "place.index", "call @integrateParticle", "builtin"} {
		if !strings.Contains(dump, want) {
			t.Fatalf("IR missing %q:\n%s", want, dump)
		}
	}
}

func TestDuplicateComputeParameterNameRejectedEvenWhenBindingAlreadyExists(t *testing.T) {
	m, err := parser.Parse("dup-param.pine", `
@workgroupSize(1)
export compute first(@group(0) @binding(1) shared: storage<u32[], read_write>) { }
@workgroupSize(1)
export compute second(
  @group(0) @binding(0) shared: storage<u32[], read_write>,
  @group(0) @binding(1) shared: storage<u32[], read_write>
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
