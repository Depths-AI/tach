package spirv

import (
	"testing"

	"tach/foundation"
	"tach/ir"
	"tach/parser"
	"tach/semantics"
)

func lowerPlan(t *testing.T, source string) *plan {
	t.Helper()
	must := func(err error) {
		if err != nil {
			t.Fatal(err)
		}
	}
	file, err := parser.Parse("planner.tach", source)
	must(err)
	logical, err := semantics.Build([]*parser.File{file}, 0)
	must(err)
	executable, err := planModule(logical.Module)
	must(err)
	return executable
}

func TestPlannerInsertsVulkanDependenciesAndRepeatBarrier(t *testing.T) {
	executable := lowerPlan(t, `
function first[i](data: buffer<uint32[]>) { data[i] += 1; }
function second[i](data: buffer<uint32[]>) { data[i] += data[i + 1]; }
export function graph(data: buffer<uint32[]>) {
  let count = data.length;
  run first(data) over count;
  run second(data) over count;
}`)
	program := executable.Programs[0]
	if program.Repeat != repeatProgram || len(program.Steps) != 3 || program.Steps[1].Kind != barrierStepKind || len(program.RepeatBarrier) == 0 {
		t.Fatalf("Vulkan dependency plan = %#v", program)
	}
}

func TestPlannerUsesPackedBufferForFusedView(t *testing.T) {
	executable := lowerPlan(t, `
function paint[i](pixels: buffer<vec<float32, 4>[]>) {
  if (i < pixels.length) { pixels[i] = vec(0.1, 0.2, 0.3, 1.0); }
}
export function image(width: uint32, height: uint32): view<srgb8> {
  let pixels = transient<vec<float32, 4>>(width * height);
  run paint(pixels) over pixels.length;
  return view(pixels, width, height);
}`)
	view := executable.Programs[0].View
	if view == nil {
		t.Fatal("Vulkan fused view is missing")
	}
	kernel := executable.PhysicalKernels[view.step.Kernel]
	if !view.Fused || view.Output != 0 || len(executable.PhysicalKernels) != 1 || kernel.Bindings[0].Type.Kind != foundation.RuntimeArrayKind || kernel.Bindings[0].Type.Elem != foundation.Uint32Type {
		t.Fatalf("Vulkan fused view = %#v; kernel = %#v", view, kernel)
	}
}

func TestPlannerPreservesLogicalFloat16RuntimeLength(t *testing.T) {
	executable := lowerPlan(t, `
type HalfSeries = { offset: float16, values: float16[] };
export function inspect[i](data: buffer<HalfSeries>, out: buffer<uint32[]>) {
  if (i < out.length) { out[i] = data.values.length; }
}`)
	kernel := executable.PhysicalKernels[0]
	length, ok := kernel.LogicalLengths[0]
	if !ok || len(kernel.Function.Params) == 0 {
		t.Fatalf("logical Float16 length = %#v", kernel.LogicalLengths)
	}
	step := executable.Programs[0].Steps[0]
	found := false
	for _, argument := range step.Parameters {
		found = found || argument.Kind == ir.ValueFromShape && kernel.Function.Params[argument.Formal].ID == length
	}
	if !found {
		t.Fatalf("Float16 length is not sourced from a runtime shape: %#v", step.Parameters)
	}
}
