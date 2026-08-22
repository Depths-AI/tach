package wgsl

import (
	"strings"
	"testing"

	"tach/foundation"
	"tach/ir"
	"tach/parser"
	"tach/semantics"
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
				if x.Function == "$tach_srgb" {
					srgb = true
				}
			case *ir.Convert:
				if x.Type == foundation.Uint32Type && x.From == foundation.Float32Type {
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

func runtimeU32(t *foundation.Type) bool {
	return t != nil && t.Kind == foundation.RuntimeArrayKind && t.Elem == foundation.Uint32Type
}

func lowerPlan(t *testing.T, source string) *plan {
	t.Helper()
	a, err := parser.Parse("executable.tach", source)
	if err != nil {
		t.Fatal(err)
	}
	result, err := semantics.Build([]*parser.File{a}, 0)
	if err != nil {
		t.Fatal(err)
	}
	executable, err := planModule(result.Module)
	if err != nil {
		t.Fatal(err)
	}
	return executable
}

func TestPlannerInternalizesSafeRepeatAndPrunesDeadValues(t *testing.T) {
	executable := lowerPlan(t, `export function scale[i](data: buffer<float32[]>, used: float32, unused: float32) { data[i] *= used; }`)
	plan := executable.Programs[0]
	if plan.Repeat != repeatInvocationLoop || len(plan.Steps) != 1 || plan.Steps[0].Parameters[len(plan.Steps[0].Parameters)-1].Kind != ir.ValueFromRepeat {
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

func TestLogicalFloat16LengthFollowsStructTail(t *testing.T) {
	const source = `
type HalfSeries = { offset: float16, values: float16[] };
export function inspect[i](data: buffer<HalfSeries>, out: buffer<uint32[]>) {
  if (i < out.length) { out[i] = data.values.length; }
}`
	web := lowerPlan(t, source)
	kernel := web.PhysicalKernels[0]
	if got := kernel.Bindings[0].MinimumByteSize; got != 4 {
		t.Fatalf("runtime-tail minimum byte size = %d, want 4", got)
	}
	length, ok := kernel.LogicalLengths[0]
	if !ok {
		t.Fatal("Web plan omitted the logical Float16 tail length")
	}
	formal := -1
	for index, parameter := range kernel.Function.Params {
		if parameter.ID == length {
			formal = index
		}
	}
	if formal < 0 || formal >= len(web.Programs[0].Steps[0].Parameters) {
		t.Fatalf("logical length parameter = %d; params = %#v", length, kernel.Function.Params)
	}
	argument := web.Programs[0].Steps[0].Parameters[formal]
	shape := web.Logical.Programs[0].Shape(argument.Shape)
	if argument.Kind != ir.ValueFromShape || shape == nil || len(shape.Path) != 1 || shape.Path[0] != "values" {
		t.Fatalf("logical length source = %#v; shape = %#v", argument, shape)
	}
}

func TestDispatchConstantsSpecializeBeforeExecutablePlanning(t *testing.T) {
	executable := lowerPlan(t, `
const factor: float16 = 0.5;
const adjustment: vec<float16, 2> = vec(0.25, 0.75);
function half[i](values: buffer<float16[]>, scale: float16, bias: vec<float16, 2>, unused: uint32) {
  if (i < values.length) { values[i] *= scale + bias.x - bias.x; }
}
export function halve(values: buffer<float16[]>) {
  const localFactor = factor;
  const localAdjustment = adjustment;
  run half(values, localFactor, localAdjustment, 99) over values.length;
}`)
	kernel := executable.PhysicalKernels[0]
	for _, parameter := range kernel.Function.Params {
		if foundation.Contains(parameter.Type, foundation.Float16Kind) {
			t.Fatalf("compile-time value remained a runtime parameter: %#v", kernel.Function.Params)
		}
	}
	found := false
	for _, instruction := range kernel.Function.Body.Instrs {
		if constant, ok := instruction.(*ir.Const); ok && constant.Type.Kind == foundation.Float16Kind && constant.Raw == "0.5" {
			found = true
		}
	}
	if !found {
		t.Fatalf("specialized Float16 value missing from kernel:\n%s", ir.DumpKernel(executable.KernelModule))
	}
	if dump := ir.DumpKernel(executable.KernelModule); strings.Contains(dump, "const uint32 99") {
		t.Fatalf("unused compile-time argument was materialized:\n%s", dump)
	}
}

func TestViewProjectionIsBackendSpecific(t *testing.T) {
	const source = `
function paint[i](pixels: buffer<vec<float32, 4>[]>) {
  if (i < pixels.length) { pixels[i] = vec(0.1, 0.2, 0.3, 1.0); }
}
export function image(width: uint32, height: uint32): view<srgb8> {
  let pixels = transient<vec<float32, 4>>(width * height);
  run paint(pixels) over pixels.length;
  return view(pixels, width, height);
}`
	web := lowerPlan(t, source)
	plan := web.Programs[0]
	kernel := web.PhysicalKernels[plan.View.step.Kernel]
	if plan.View == nil || !plan.View.Fused || len(web.PhysicalKernels) != 1 || len(plan.Steps) != 0 || len(plan.Transients) != 0 || plan.View.step.Kernel != 0 || web.Logical.Programs[0].Shape(plan.View.Width).Op != ir.ShapeParameter || web.Logical.Programs[0].Shape(plan.View.Height).Op != ir.ShapeParameter || !kernel.FusedView || kernel.Projection || len(kernel.Bindings) != 1 || plan.View.Output != 0 || !kernel.Bindings[0].Texture || !packsDisplayPixel(kernel.Function) || !runtimeU32(kernel.Bindings[0].Type) {
		t.Fatalf("web fused view plan = %#v; kernel = %#v", plan.View, kernel)
	}
}

func TestStandaloneProjectionPacksTheSameDisplayWord(t *testing.T) {
	const source = `
function paint[i](pixels: buffer<vec<float32, 4>[]>) { pixels[i] = vec(0.1, 0.2, 0.3, 1.0); }
export function image(pixels: buffer<vec<float32, 4>[]>, width: uint32, height: uint32): view<srgb8> {
  run paint(pixels) over width * height;
  return view(pixels, width, height);
}`
	executable := lowerPlan(t, source)
	plan := executable.Programs[0]
	projection := executable.PhysicalKernels[plan.View.step.Kernel]
	if plan.View == nil || plan.View.Fused || len(plan.Steps) != 1 || len(executable.PhysicalKernels) != 2 || !projection.Projection || len(plan.View.step.Resources) != 1 || plan.View.step.Resources[0].Kind != externalSource || !packsDisplayPixel(projection.Function) || !runtimeU32(projection.Bindings[1].Type) || !projection.Bindings[1].Texture {
		t.Fatalf("web projection plan = %#v; kernel = %#v", plan.View, projection)
	}
}

func TestConstantExtentViewStillFuses(t *testing.T) {
	executable := lowerPlan(t, `
function paint[i](pixels: buffer<vec<float32, 4>[]>) { pixels[i] = vec(0.1, 0.2, 0.3, 1.0); }
export function image(): view<srgb8> {
  let pixels = transient<vec<float32, 4>>(4);
  run paint(pixels) over pixels.length;
  return view(pixels, 2, 2);
}`)
	if view := executable.Programs[0].View; view == nil || !view.Fused {
		t.Fatalf("view plan = %#v", view)
	}
}
