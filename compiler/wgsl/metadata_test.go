package wgsl

import (
	"encoding/json"
	"testing"

	"tach/ir"
)

func runtimeForTest(t *testing.T, source string) (*runtimeMetadata, *ir.Module) {
	t.Helper()
	executable := lowerPlan(t, source)
	metadata, err := describePlan(executable)
	if err != nil {
		t.Fatal(err)
	}
	return metadata, executable.Logical
}

func cloneRuntime(t *testing.T, metadata *runtimeMetadata) *runtimeMetadata {
	t.Helper()
	encoded, err := json.Marshal(metadata)
	if err != nil {
		t.Fatal(err)
	}
	var clone runtimeMetadata
	if err := json.Unmarshal(encoded, &clone); err != nil {
		t.Fatal(err)
	}
	return &clone
}

func TestRuntimeValidationRejectsCorruptWebPlans(t *testing.T) {
	metadata, logical := runtimeForTest(t, `export function fill[i](out: buffer<uint32[]>) { if (i < out.length) { out[i] = i; } }`)
	mutations := map[string]func(*runtimeMetadata){
		"feature":       func(m *runtimeMetadata) { m.Features = []string{"invalid"} },
		"program count": func(m *runtimeMetadata) { m.Programs = nil },
		"program index": func(m *runtimeMetadata) { m.Programs[0].Program = 1 },
		"repeat":        func(m *runtimeMetadata) { m.Programs[0].Repeat = "invalid" },
		"step kind":     func(m *runtimeMetadata) { m.Programs[0].Steps[0].Kind = "barrier" },
		"kernel link":   func(m *runtimeMetadata) { m.Programs[0].Steps[0].Kernel = 99 },
		"entry point":   func(m *runtimeMetadata) { m.Kernels[0].EntryPoint = "wrong" },
		"workgroup":     func(m *runtimeMetadata) { m.Kernels[0].WorkgroupSize[0] = 0 },
		"binding group": func(m *runtimeMetadata) { m.Kernels[0].Bindings[0].Group = 1 },
		"binding index": func(m *runtimeMetadata) { m.Kernels[0].Bindings[0].Binding = 2 },
	}
	for name, mutate := range mutations {
		t.Run(name, func(t *testing.T) {
			clone := cloneRuntime(t, metadata)
			mutate(clone)
			if err := validateRuntime(clone, logical); err == nil {
				t.Fatal("accepted corrupt Web runtime plan")
			}
		})
	}
}

func TestRuntimeValidationRejectsCorruptWebViews(t *testing.T) {
	metadata, logical := runtimeForTest(t, `
function paint[i](pixels: buffer<vec<float32, 4>[]>) { pixels[i] = vec(0.0, 0.0, 0.0, 1.0); }
export function image(width: uint32, height: uint32): view<srgb8> {
  let pixels = transient<vec<float32, 4>>(width * height);
  run paint(pixels) over pixels.length;
  return view(pixels, width, height);
}`)
	mutations := map[string]func(*runtimeMetadata){
		"public contract": func(m *runtimeMetadata) { m.Programs[0].View = nil },
		"format":          func(m *runtimeMetadata) { m.Programs[0].View.Format = "rgba8" },
		"step kind":       func(m *runtimeMetadata) { m.Programs[0].View.Step.Kind = "barrier" },
		"kernel":          func(m *runtimeMetadata) { m.Programs[0].View.Step.Kernel = 99 },
		"output":          func(m *runtimeMetadata) { m.Programs[0].View.Output = 99 },
		"output input": func(m *runtimeMetadata) {
			view := m.Programs[0].View
			view.Step.Resources = []resourceMetadata{{Binding: view.Output, Kind: "external"}}
		},
		"fused":        func(m *runtimeMetadata) { m.Programs[0].View.Fused = false },
		"output color": func(m *runtimeMetadata) { m.Programs[0].View.OutputColor = -1 },
		"texture": func(m *runtimeMetadata) {
			view := m.Programs[0].View
			m.Kernels[view.Step.Kernel].Bindings[view.Output].Kind = "buffer"
		},
	}
	for name, mutate := range mutations {
		t.Run(name, func(t *testing.T) {
			clone := cloneRuntime(t, metadata)
			mutate(clone)
			if err := validateRuntime(clone, logical); err == nil {
				t.Fatal("accepted corrupt Web view plan")
			}
		})
	}
}
