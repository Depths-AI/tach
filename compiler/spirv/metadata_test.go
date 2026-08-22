package spirv

import (
	"encoding/json"
	"testing"
)

func TestRuntimeValidationRejectsInvalidVulkanContract(t *testing.T) {
	executable := lowerPlan(t, `export function fill[i](out: buffer<uint32[]>) { if (i < out.length) { out[i] = i; } }`)
	original, err := describePlan(executable)
	if err != nil {
		t.Fatal(err)
	}
	for name, mutate := range map[string]func(*runtimeMetadata){
		"Vulkan version": func(m *runtimeMetadata) { m.Vulkan = "1.2" },
		"SPIR-V version": func(m *runtimeMetadata) { m.SPIRV = "1.5" },
		"required features": func(m *runtimeMetadata) {
			m.Features = nil
		},
		"feature order": func(m *runtimeMetadata) {
			m.Features = append(m.Features, storageBuffer16BitAccess)
		},
		"barrier kind": func(m *runtimeMetadata) { m.Programs[0].Steps[0].Kind = "invalid" },
	} {
		t.Run(name, func(t *testing.T) {
			encoded, err := json.Marshal(original)
			if err != nil {
				t.Fatal(err)
			}
			var corrupted runtimeMetadata
			if err := json.Unmarshal(encoded, &corrupted); err != nil {
				t.Fatal(err)
			}
			mutate(&corrupted)
			if err := validateRuntime(&corrupted, executable.Logical); err == nil {
				t.Fatal("accepted invalid Vulkan runtime contract")
			}
		})
	}
}

func TestRuntimeMetadataOwnsBarriersAndPackedViews(t *testing.T) {
	barriers := lowerPlan(t, `
function first[i](data: buffer<uint32[]>) { data[i] += 1; }
function second[i](data: buffer<uint32[]>) { data[i] += data[i + 1]; }
export function graph(data: buffer<uint32[]>) {
  run first(data) over data.length;
  run second(data) over data.length;
}`)
	barrierMetadata, err := describePlan(barriers)
	if err != nil {
		t.Fatal(err)
	}
	if barrierMetadata.Programs[0].RepeatBarrier == nil || barrierMetadata.Programs[0].Steps[1].Kind != "barrier" {
		t.Fatalf("barrier metadata = %#v", barrierMetadata.Programs[0])
	}

	view := lowerPlan(t, `
function paint[i](pixels: buffer<vec<float32, 4>[]>) { pixels[i] = vec(0.0, 0.0, 0.0, 1.0); }
export function image(width: uint32, height: uint32): view<srgb8> {
  let pixels = transient<vec<float32, 4>>(width * height);
  run paint(pixels) over pixels.length;
  return view(pixels, width, height);
}`)
	viewMetadata, err := describePlan(view)
	if err != nil {
		t.Fatal(err)
	}
	projection := viewMetadata.Programs[0].View
	viewMetadata.Kernels[projection.Step.Kernel].Bindings[projection.Output].Kind = "texture"
	if err := validateRuntime(viewMetadata, view.Logical); err == nil {
		t.Fatal("accepted a texture-backed Vulkan view")
	}
}
