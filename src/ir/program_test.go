package ir_test

import (
	"math"
	"strings"
	"testing"

	"tach/src/ir"
	"tach/src/parser"
	"tach/src/sema"
)

func logical(t *testing.T) *ir.Module {
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

func viewLogical(t *testing.T) *ir.Module {
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
	original.Documentation.Types["Value"] = ir.TypeDocumentation{Fields: map[string]string{"x": "original"}}
	clone := ir.Clone(original)
	clone.Programs[0].Name = "changed"
	clone.Kernel.Functions[0].Name = "changed"
	documentation := clone.Documentation.Types["Value"]
	documentation.Fields["x"] = "changed"
	if original.Programs[0].Name != "fill" {
		t.Fatal("program clone mutated original")
	}
	if original.Kernel.Functions[0].Name != "fill" {
		t.Fatal("kernel clone mutated original")
	}
	if original.Documentation.Types["Value"].Fields["x"] != "original" {
		t.Fatal("documentation clone mutated original")
	}
	if dump := ir.Dump(original); dump != ir.Dump(original) || !strings.Contains(dump, "program @fill") {
		t.Fatalf("dump = %q", dump)
	}
}

func TestModuleVerifierIncludesKernelIR(t *testing.T) {
	module := ir.Clone(logical(t))
	module.Kernel.Functions[0] = nil
	if err := ir.Verify(module); err == nil || !strings.Contains(err.Error(), "kernel IR") {
		t.Fatalf("invalid kernel error = %v", err)
	}
}

func TestVerifierRejectsVersionAndShapeMutations(t *testing.T) {
	tests := []func(*ir.Module){
		func(m *ir.Module) { m.Programs[0].Versions[1].Previous = 99 },
		func(m *ir.Module) { m.Programs[0].Dispatches[0].Domain[0] = 99 },
		func(m *ir.Module) { m.Programs[0].Dispatches[0].Stage = "missing" },
	}
	for _, mutate := range tests {
		m := ir.Clone(logical(t))
		mutate(m)
		if err := ir.Verify(m); err == nil {
			t.Fatal("accepted malformed Flow IR")
		}
	}
}

func TestConstantArgumentsCloneAndVerifyCanonically(t *testing.T) {
	a, err := parser.Parse("constant.tach", `
function fill[i](out: buffer<float32[]>, value: float32) {
  if (i < out.length) { out[i] = value; }
}
export function filled(out: buffer<float32[]>) {
  run fill(out, 7.0) over out.length;
}`)
	if err != nil {
		t.Fatal(err)
	}
	m, err := sema.CheckAndLower(a)
	if err != nil {
		t.Fatal(err)
	}
	constant := m.Programs[0].Dispatches[0].Values[0].Constant
	clone := ir.Clone(m)
	clone.Programs[0].Dispatches[0].Values[0].Constant.Bits[0] = math.Float32bits(8)
	if constant.Bits[0] != math.Float32bits(7) {
		t.Fatal("constant clone aliases the original bits")
	}
	clone.Programs[0].Dispatches[0].Values[0].Constant.Bits = nil
	if err := ir.Verify(clone); err == nil {
		t.Fatal("accepted malformed constant argument")
	}
	clone = ir.Clone(m)
	clone.Programs[0].Dispatches[0].Values[0].Constant.Bits[0] = math.Float32bits(float32(math.Inf(1)))
	if err := ir.Verify(clone); err == nil {
		t.Fatal("accepted non-finite constant argument")
	}
}

func TestViewCloneDumpAndVerification(t *testing.T) {
	original := viewLogical(t)
	clone := ir.Clone(original)
	clone.Programs[0].View.Width = 99
	if original.Programs[0].View.Width == 99 {
		t.Fatal("view clone mutated original")
	}
	if dump := ir.Dump(original); !strings.Contains(dump, "view srgb8") {
		t.Fatalf("program dump omitted view:\n%s", dump)
	}
	mutations := []func(*ir.View){
		func(view *ir.View) { view.Format = 0 },
		func(view *ir.View) { view.Source = 99 },
		func(view *ir.View) { view.Input = 99 },
		func(view *ir.View) { view.Width = 99 },
		func(view *ir.View) { view.Height = 99 },
	}
	for _, mutate := range mutations {
		module := ir.Clone(original)
		mutate(module.Programs[0].View)
		if err := ir.Verify(module); err == nil {
			t.Fatal("accepted malformed Flow view")
		}
	}
}
