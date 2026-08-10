package spirv_test

import (
	"os"
	"strings"
	"testing"

	"pine/internal/parser"
	"pine/internal/sema"
	"pine/internal/spirv"
)

func TestParticlesSPIRV(t *testing.T) {
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
	bin, err := spirv.Emit(m)
	if err != nil {
		t.Fatal(err)
	}
	if len(bin) < 20 {
		t.Fatalf("short SPIR-V binary: %d", len(bin))
	}
	s, err := spirv.Summary(bin)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(s, "entries [__pine_k_integrate]") {
		t.Fatalf("unexpected summary: %s", s)
	}
}
