package backend_test

import (
	"testing"

	"tach/src/backend"
	"tach/src/flow"
	"tach/src/ir"
	"tach/src/opt"
	"tach/src/parser"
	"tach/src/sema"
)

func lower(t *testing.T, source string, profile backend.Profile) *backend.Executable {
	t.Helper()
	a, err := parser.Parse("backend.tach", source)
	if err != nil {
		t.Fatal(err)
	}
	m, err := sema.CheckAndLower(a)
	if err != nil {
		t.Fatal(err)
	}
	if err := opt.OptimizeLogical(m); err != nil {
		t.Fatal(err)
	}
	executable, err := backend.Lower(m, profile)
	if err != nil {
		t.Fatal(err)
	}
	return executable
}

func TestPlannerInternalizesSafeRepeatAndPrunesDeadValues(t *testing.T) {
	executable := lower(t, `export function scale[i](data: buffer<float32[]>, used: float32, unused: float32) { data[i] *= used; }`, backend.WebProfile)
	plan := executable.Programs[0]
	if plan.Repeat != backend.RepeatInvocationLoop || len(plan.Steps) != 1 || plan.Steps[0].Parameters[len(plan.Steps[0].Parameters)-1].Kind != flow.ValueRepeat {
		t.Fatalf("plan = %#v", plan)
	}
	kernel := executable.PhysicalKernels[0]
	if len(kernel.Function.Params) != 2 || len(kernel.Parameters.Fields) != 2 {
		t.Fatalf("params = %#v; block = %#v", kernel.Function.Params, kernel.Parameters)
	}
	if _, ok := kernel.Function.Body.Instrs[1].(*ir.Loop); !ok {
		t.Fatalf("body = %#v", kernel.Function.Body)
	}
}

func TestSPIRVPlanKeepsCrossInvocationRepeatAndBarrier(t *testing.T) {
	executable := lower(t, `
function first[i](data: buffer<uint32[]>) { data[i] += uint32(1); }
function second[i](data: buffer<uint32[]>) { data[i] += data[i + uint32(1)]; }
export function graph(data: buffer<uint32[]>) { const count = data.length; run first(data) over count; run second(data) over count; }`, backend.SPIRVProfile)
	plan := executable.Programs[0]
	if plan.Repeat != backend.RepeatProgram || len(plan.Steps) != 3 || plan.Steps[1].Kind != backend.BarrierStepKind || len(plan.RepeatBarrier) == 0 {
		t.Fatalf("plan = %#v", plan)
	}
}

func TestViewProjectionIsBackendSpecific(t *testing.T) {
	const source = `
function paint[i](pixels: buffer<float32x4[]>) {
  if (i < pixels.length) { pixels[i] = float32x4(0.1, 0.2, 0.3, 1.0); }
}
export function image(width: uint32, height: uint32): view<srgb8> {
  const pixels = transient<float32x4>(width * height);
  run paint(pixels) over pixels.length;
  return view(pixels, width, height);
}`
	web := lower(t, source, backend.WebProfile)
	spirv := lower(t, source, backend.SPIRVProfile)
	for _, executable := range []*backend.Executable{web, spirv} {
		plan := executable.Programs[0]
		if plan.View == nil || !plan.View.Fused || len(executable.PhysicalKernels) != 1 || len(plan.Steps) != 0 || len(plan.Transients) != 0 || plan.View.Step.Kernel != 0 || executable.Logical.Programs[0].Shape(plan.View.Width).Op != flow.ShapeParameter || executable.Logical.Programs[0].Shape(plan.View.Height).Op != flow.ShapeParameter {
			t.Fatalf("view plan = %#v; kernels = %#v", plan.View, executable.PhysicalKernels)
		}
		projection := executable.PhysicalKernels[plan.View.Step.Kernel]
		if !projection.FusedView || projection.Projection || len(projection.Bindings) != 1 || plan.View.Output != 0 {
			t.Fatalf("projection kernel = %#v", projection)
		}
	}
	if !web.PhysicalKernels[0].Bindings[0].Texture || spirv.PhysicalKernels[0].Bindings[0].Texture {
		t.Fatalf("web view = %#v; SPIR-V view = %#v", web.Programs[0].View, spirv.Programs[0].View)
	}
}

func TestExternalViewKeepsGeneralProjector(t *testing.T) {
	executable := lower(t, `
function paint[i](pixels: buffer<float32x4[]>) { pixels[i] = float32x4(0.1, 0.2, 0.3, 1.0); }
export function image(pixels: buffer<float32x4[]>, width: uint32, height: uint32): view<srgb8> {
  run paint(pixels) over width * height;
  return view(pixels, width, height);
}`, backend.WebProfile)
	plan := executable.Programs[0]
	if plan.View == nil || plan.View.Fused || len(plan.Steps) != 1 || len(executable.PhysicalKernels) != 2 || !executable.PhysicalKernels[plan.View.Step.Kernel].Projection || len(plan.View.Step.Resources) != 1 || plan.View.Step.Resources[0].Kind != backend.ExternalSource {
		t.Fatalf("view plan = %#v; kernels = %#v", plan.View, executable.PhysicalKernels)
	}
}

func TestConstantExtentViewStillFuses(t *testing.T) {
	executable := lower(t, `
function paint[i](pixels: buffer<float32x4[]>) { pixels[i] = float32x4(0.1, 0.2, 0.3, 1.0); }
export function image(): view<srgb8> {
  const pixels = transient<float32x4>(4);
  run paint(pixels) over pixels.length;
  return view(pixels, 2, 2);
}`, backend.SPIRVProfile)
	if view := executable.Programs[0].View; view == nil || !view.Fused {
		t.Fatalf("view plan = %#v", view)
	}
}
