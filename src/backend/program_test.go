package backend_test

import (
	"testing"

	"tach/src/backend"
	"tach/src/flow"
	"tach/src/ir"
	"tach/src/opt"
	"tach/src/parser"
	"tach/src/sema"
	"tach/src/types"
)

func packsDisplayPixel(function *ir.Function) bool {
	srgb, quantize, packed := false, false, false
	var visit func(*ir.Block)
	visit = func(block *ir.Block) {
		if block == nil {
			return
		}
		for _, instruction := range block.Instrs {
			switch x := instruction.(type) {
			case *ir.Call:
				if x.Function == "__tach_srgb" {
					srgb = true
				}
			case *ir.Convert:
				if x.Type == types.TU32 && x.From == types.TF32 {
					quantize = true
				}
			case *ir.Binary:
				if x.Op == "<<" || x.Op == "|" {
					packed = true
				}
			case *ir.If:
				visit(x.Then)
				visit(x.Else)
			case *ir.Loop:
				visit(x.Cond)
				visit(x.Body)
			case *ir.Scope:
				visit(x.Body)
			}
		}
	}
	visit(function.Body)
	return srgb && quantize && packed
}

func runtimeU32(t *types.Type) bool {
	return t != nil && t.Kind == types.RuntimeArray && t.Elem == types.TU32
}

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
		kernel := executable.PhysicalKernels[plan.View.Step.Kernel]
		if !kernel.FusedView || kernel.Projection || len(kernel.Bindings) != 1 || plan.View.Output != 0 {
			t.Fatalf("fused view kernel = %#v", kernel)
		}
	}
	if !web.PhysicalKernels[0].Bindings[0].Texture || spirv.PhysicalKernels[0].Bindings[0].Texture {
		t.Fatalf("web view = %#v; SPIR-V view = %#v", web.Programs[0].View, spirv.Programs[0].View)
	}
	if !packsDisplayPixel(web.PhysicalKernels[0].Function) || !packsDisplayPixel(spirv.PhysicalKernels[0].Function) {
		t.Fatalf("fused view skipped shared pack math")
	}
	if !runtimeU32(web.PhysicalKernels[0].Bindings[0].Type) || !runtimeU32(spirv.PhysicalKernels[0].Bindings[0].Type) {
		t.Fatalf("fused view output types = %s, %s", web.PhysicalKernels[0].Bindings[0].Type, spirv.PhysicalKernels[0].Bindings[0].Type)
	}
}

func TestStandaloneProjectionPacksTheSameDisplayWord(t *testing.T) {
	const source = `
function paint[i](pixels: buffer<float32x4[]>) { pixels[i] = float32x4(0.1, 0.2, 0.3, 1.0); }
export function image(pixels: buffer<float32x4[]>, width: uint32, height: uint32): view<srgb8> {
  run paint(pixels) over width * height;
  return view(pixels, width, height);
}`
	for _, profile := range []backend.Profile{backend.WebProfile, backend.SPIRVProfile} {
		executable := lower(t, source, profile)
		plan := executable.Programs[0]
		projection := executable.PhysicalKernels[plan.View.Step.Kernel]
		if plan.View == nil || plan.View.Fused || len(plan.Steps) != 1 || len(executable.PhysicalKernels) != 2 || !projection.Projection || len(plan.View.Step.Resources) != 1 || plan.View.Step.Resources[0].Kind != backend.ExternalSource {
			t.Fatalf("%s view plan = %#v; kernels = %#v", profile.Target, plan.View, executable.PhysicalKernels)
		}
		if !packsDisplayPixel(projection.Function) || !runtimeU32(projection.Bindings[1].Type) || projection.Bindings[1].Texture != (profile.Target == backend.Web) {
			t.Fatalf("%s projection = %#v", profile.Target, projection)
		}
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
