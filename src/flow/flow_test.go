package flow_test

import (
	"strings"
	"testing"

	"tach/src/flow"
	"tach/src/parser"
	"tach/src/sema"
)

func logical(t *testing.T) *flow.Module {
	t.Helper()
	a, err := parser.Parse("flow.tach", `export function fill[i](out: buffer<uint32[]>) { if (i < out.length) { out[i] = i; } }`)
	if err != nil {
		t.Fatal(err)
	}
	m, err := sema.CheckAndLower(a)
	if err != nil {
		t.Fatal(err)
	}
	return m
}

func TestCloneIsIndependentAndDumpIsDeterministic(t *testing.T) {
	original := logical(t)
	clone := flow.Clone(original)
	clone.Programs[0].Name = "changed"
	if original.Programs[0].Name != "fill" {
		t.Fatal("clone mutated original")
	}
	if dump := flow.Dump(original); dump != flow.Dump(original) || !strings.Contains(dump, "program @fill") {
		t.Fatalf("dump = %q", dump)
	}
}

func TestVerifierRejectsVersionAndShapeMutations(t *testing.T) {
	tests := []func(*flow.Module){
		func(m *flow.Module) { m.Programs[0].Versions[1].Previous = 99 },
		func(m *flow.Module) { m.Programs[0].Dispatches[0].Domain[0] = 99 },
		func(m *flow.Module) { m.Programs[0].Dispatches[0].Stage = "missing" },
	}
	for _, mutate := range tests {
		m := flow.Clone(logical(t))
		mutate(m)
		if err := flow.Verify(m); err == nil {
			t.Fatal("accepted malformed Flow IR")
		}
	}
}
