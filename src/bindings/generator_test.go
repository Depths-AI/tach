package bindings

import (
	"encoding/json"
	"testing"

	"tach/src/backend"
	"tach/src/opt"
	"tach/src/parser"
	"tach/src/sema"
	"tach/src/spirv"
	"tach/src/wgsl"
)

func generateSource(t *testing.T, source string) (*Artifacts, *Metadata) {
	t.Helper()
	a, err := parser.Parse("bindings.tach", source)
	if err != nil {
		t.Fatal(err)
	}
	logical, err := sema.CheckAndLower(a)
	if err != nil {
		t.Fatal(err)
	}
	if err := opt.OptimizeLogical(logical); err != nil {
		t.Fatal(err)
	}
	web, err := wgsl.Lower(logical)
	if err != nil {
		t.Fatal(err)
	}
	spv, err := spirv.Lower(logical)
	if err != nil {
		t.Fatal(err)
	}
	artifacts, err := Generate(logical, web, spv)
	if err != nil {
		t.Fatal(err)
	}
	var metadata Metadata
	if err := json.Unmarshal(artifacts.MetadataJSON, &metadata); err != nil {
		t.Fatal(err)
	}
	return artifacts, &metadata
}

func corrupted(t *testing.T, metadata *Metadata, mutate func(*Metadata)) *Metadata {
	t.Helper()
	encoded, err := json.Marshal(metadata)
	if err != nil {
		t.Fatal(err)
	}
	var out Metadata
	if err := json.Unmarshal(encoded, &out); err != nil {
		t.Fatal(err)
	}
	mutate(&out)
	return &out
}

func TestGenerateCompleteRuntimeMetadata(t *testing.T) {
	artifacts, metadata := generateSource(t, `export function scale[i](values: buffer<float32[]>, factor: float32) { if (i < values.length) { values[i] *= factor; } }`)
	if artifacts.Metadata == nil || metadata.Schema != 2 || len(metadata.Programs) != 1 || metadata.Programs[0].Name != "scale" {
		t.Fatalf("metadata = %#v", metadata)
	}
	if metadata.Targets.Web == nil || metadata.Targets.SPIRV == nil || len(metadata.Targets.Web.Kernels) != 1 {
		t.Fatalf("targets = %#v", metadata.Targets)
	}
	if got := metadata.Targets.Web.Kernels[0].EntryPoint; got != "_tach_k0" {
		t.Fatalf("entry = %q", got)
	}
	spv := metadata.Targets.SPIRV
	if spv.Vulkan != backend.VulkanVersion || spv.SPIRV != backend.SPIRVVersion || len(spv.Features) != 2 {
		t.Fatalf("SPIR-V profile = %#v", spv)
	}
}

func TestGenerateOrchestrationPlansForBothBackends(t *testing.T) {
	_, metadata := generateSource(t, `
function copy[i](input: buffer<float32[]>, output: buffer<float32[]>) { output[i] = input[i]; }
function twice[i](input: buffer<float32[]>, output: buffer<float32[]>) { output[i] = input[i] * 2.0; }
export function transform(input: buffer<float32[]>, output: buffer<float32[]>) {
  const count = min(input.length, output.length);
  const temporary = transient<float32>(count);
  run copy(input, temporary) over count;
  run twice(temporary, output) over count;
}`)
	for name, plan := range map[string]*TargetPlanMeta{"web": metadata.Targets.Web, "spirv": metadata.Targets.SPIRV} {
		if len(plan.Programs[0].Steps) < 2 || len(plan.Programs[0].Transients) != 1 {
			t.Errorf("%s plan = %#v", name, plan.Programs[0])
		}
	}
}

func TestGenerateViewContractForBothBackends(t *testing.T) {
	_, metadata := generateSource(t, `
function paint[i](pixels: buffer<float32x4[]>) {
  if (i < pixels.length) { pixels[i] = float32x4(0.1, 0.2, 0.3, 1.0); }
}
export function image(width: uint32, height: uint32): view<srgb8> {
  const pixels = transient<float32x4>(width * height);
  run paint(pixels) over pixels.length;
  return view(pixels, width, height);
}`)
	if !metadata.Programs[0].View || len(metadata.Programs[0].Resources) != 0 {
		t.Fatalf("public view metadata = %#v", metadata.Programs[0])
	}
	for name, target := range map[string]*TargetPlanMeta{"web": metadata.Targets.Web, "spirv": metadata.Targets.SPIRV} {
		view := target.Programs[0].View
		if view == nil || view.Format != "srgb8" || !view.Fused || view.Step.Kind != "dispatch" || len(view.Step.Resources) != 0 || view.Width.Op != "parameter" || view.Height.Op != "parameter" {
			t.Fatalf("%s view metadata = %#v", name, view)
		}
		bindings := target.Kernels[view.Step.Kernel].Bindings
		want := "buffer"
		if name == "web" {
			want = "texture"
		}
		if len(bindings) != 1 || bindings[0].Kind != want || view.Output != 0 {
			t.Fatalf("%s projection bindings = %#v", name, bindings)
		}
	}
}

func TestValidateMetadataRejectsCorruptRuntimeSeams(t *testing.T) {
	_, metadata := generateSource(t, `export function fill[i](out: buffer<uint32[]>) { if (i < out.length) { out[i] = i; } }`)
	mutations := map[string]func(*Metadata){
		"schema":         func(m *Metadata) { m.Schema = 0 },
		"no web":         func(m *Metadata) { m.Targets.Web = nil },
		"no spirv":       func(m *Metadata) { m.Targets.SPIRV = nil },
		"vulkan version": func(m *Metadata) { m.Targets.SPIRV.Vulkan = "1.2" },
		"spirv version":  func(m *Metadata) { m.Targets.SPIRV.SPIRV = "1.3" },
		"feature":        func(m *Metadata) { m.Targets.SPIRV.Features = nil },
		"program count":  func(m *Metadata) { m.Targets.Web.Programs = nil },
		"program index":  func(m *Metadata) { m.Targets.Web.Programs[0].Program = 1 },
		"repeat":         func(m *Metadata) { m.Targets.Web.Programs[0].Repeat = "invalid" },
		"step kind":      func(m *Metadata) { m.Targets.Web.Programs[0].Steps[0].Kind = "invalid" },
		"kernel link":    func(m *Metadata) { m.Targets.Web.Programs[0].Steps[0].Kernel = 99 },
		"entry point":    func(m *Metadata) { m.Targets.Web.Kernels[0].EntryPoint = "wrong" },
		"workgroup":      func(m *Metadata) { m.Targets.Web.Kernels[0].WorkgroupSize[0] = 0 },
		"binding group":  func(m *Metadata) { m.Targets.Web.Kernels[0].Bindings[0].Group = 1 },
		"binding index":  func(m *Metadata) { m.Targets.Web.Kernels[0].Bindings[0].Binding = 2 },
	}
	for name, mutate := range mutations {
		t.Run(name, func(t *testing.T) {
			if err := ValidateMetadata(corrupted(t, metadata, mutate)); err == nil {
				t.Fatal("accepted corrupt runtime metadata")
			}
		})
	}
}

func TestValidateMetadataRejectsCorruptViewSeams(t *testing.T) {
	_, metadata := generateSource(t, `
function paint[i](pixels: buffer<float32x4[]>) { pixels[i] = float32x4(0.0, 0.0, 0.0, 1.0); }
export function image(width: uint32, height: uint32): view<srgb8> {
  const pixels = transient<float32x4>(width * height);
  run paint(pixels) over pixels.length;
  return view(pixels, width, height);
}`)
	mutations := map[string]func(*Metadata){
		"public flag": func(m *Metadata) { m.Programs[0].View = false },
		"format":      func(m *Metadata) { m.Targets.Web.Programs[0].View.Format = "rgba8" },
		"step kind":   func(m *Metadata) { m.Targets.Web.Programs[0].View.Step.Kind = "barrier" },
		"kernel":      func(m *Metadata) { m.Targets.Web.Programs[0].View.Step.Kernel = 99 },
		"output":      func(m *Metadata) { m.Targets.Web.Programs[0].View.Output = 99 },
		"output input": func(m *Metadata) {
			view := m.Targets.Web.Programs[0].View
			view.Step.Resources = []ResourceSourceMeta{{Binding: view.Output, Kind: "external", Resource: 0}}
		},
		"fused":        func(m *Metadata) { m.Targets.Web.Programs[0].View.Fused = false },
		"output color": func(m *Metadata) { m.Targets.Web.Programs[0].View.OutputColor = -1 },
		"texture": func(m *Metadata) {
			view := m.Targets.Web.Programs[0].View
			m.Targets.Web.Kernels[view.Step.Kernel].Bindings[view.Output].Kind = "buffer"
		},
	}
	for name, mutate := range mutations {
		t.Run(name, func(t *testing.T) {
			if err := ValidateMetadata(corrupted(t, metadata, mutate)); err == nil {
				t.Fatal("accepted corrupt view metadata")
			}
		})
	}
}
