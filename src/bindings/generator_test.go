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

func TestGenerateCompleteRuntimeMetadata(t *testing.T) {
	artifacts, metadata := generateSource(t, `export function scale[i](values: buffer<float32[]>, factor: float32) { if (i < values.length) { values[i] *= factor; } }`)
	if artifacts.Metadata == nil || metadata.Schema != 1 || len(metadata.Programs) != 1 || metadata.Programs[0].Name != "scale" {
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
			encoded, err := json.Marshal(metadata)
			if err != nil {
				t.Fatal(err)
			}
			var corrupted Metadata
			if err := json.Unmarshal(encoded, &corrupted); err != nil {
				t.Fatal(err)
			}
			mutate(&corrupted)
			if err := ValidateMetadata(&corrupted); err == nil {
				t.Fatal("accepted corrupt runtime metadata")
			}
		})
	}
}
