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

func viewLogical(t *testing.T) *flow.Module {
	t.Helper()
	a, err := parser.Parse("view.tach", `
function paint[i](pixels: buffer<vec<float32, 4>[]>) {
  if (i < pixels.length) { pixels[i] = vec(0.0, 0.0, 0.0, 1.0); }
}
export function image(width: uint32, height: uint32): view<srgb8> {
  let pixels = transient<vec<float32, 4>>(width * height);
  run paint(pixels) over pixels.length;
  return view(pixels, width, height);
}`)
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

func TestConstantArgumentsCloneAndVerifyCanonically(t *testing.T) {
	a, err := parser.Parse("constant.tach", `
function fill[i](out: buffer<uint32[]>, value: uint32) {
  if (i < out.length) { out[i] = value; }
}
export function filled(out: buffer<uint32[]>) {
  run fill(out, 7) over out.length;
}`)
	if err != nil {
		t.Fatal(err)
	}
	m, err := sema.CheckAndLower(a)
	if err != nil {
		t.Fatal(err)
	}
	constant := m.Programs[0].Dispatches[0].Values[0].Constant
	clone := flow.Clone(m)
	clone.Programs[0].Dispatches[0].Values[0].Constant.Bits[0] = 8
	if constant.Bits[0] != 7 {
		t.Fatal("constant clone aliases the original bits")
	}
	clone.Programs[0].Dispatches[0].Values[0].Constant.Bits = nil
	if err := flow.Verify(clone); err == nil {
		t.Fatal("accepted malformed constant argument")
	}
}

func TestViewCloneDumpAndVerification(t *testing.T) {
	original := viewLogical(t)
	clone := flow.Clone(original)
	clone.Programs[0].View.Width = 99
	if original.Programs[0].View.Width == 99 {
		t.Fatal("view clone mutated original")
	}
	if dump := flow.Dump(original); !strings.Contains(dump, "view srgb8") {
		t.Fatalf("Flow dump omitted view:\n%s", dump)
	}
	mutations := []func(*flow.View){
		func(view *flow.View) { view.Format = 0 },
		func(view *flow.View) { view.Source = 99 },
		func(view *flow.View) { view.Input = 99 },
		func(view *flow.View) { view.Width = 99 },
		func(view *flow.View) { view.Height = 99 },
	}
	for _, mutate := range mutations {
		module := flow.Clone(original)
		mutate(module.Programs[0].View)
		if err := flow.Verify(module); err == nil {
			t.Fatal("accepted malformed Flow view")
		}
	}
}
