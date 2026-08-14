package bindings

import (
	"encoding/json"
	"strings"
	"testing"

	"tach/src/backend"
	"tach/src/opt"
	"tach/src/parser"
	"tach/src/sema"
	"tach/src/wgsl"
)

func generateSource(t *testing.T, source string, includeSPIRV bool) (*Artifacts, *Metadata) {
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
	wgslSource, err := wgsl.Emit(web)
	if err != nil {
		t.Fatal(err)
	}
	var spv *backend.Executable
	if includeSPIRV {
		spv, err = backend.Lower(logical, backend.SPIRVProfile)
		if err != nil {
			t.Fatal(err)
		}
	}
	artifacts, err := Generate(logical, web, spv, wgslSource)
	if err != nil {
		t.Fatal(err)
	}
	var metadata Metadata
	if err := json.Unmarshal(artifacts.MetadataJSON, &metadata); err != nil {
		t.Fatal(err)
	}
	return artifacts, &metadata
}

func TestGenerateBaselineProgram(t *testing.T) {
	artifacts, metadata := generateSource(t, `export function scale[i](values: buffer<float32[]>, factor: float32) { if (i < values.length) { values[i] *= factor; } }`, true)
	if metadata.Schema != 1 || len(metadata.Programs) != 1 || metadata.Programs[0].Name != "scale" {
		t.Fatalf("metadata = %#v", metadata)
	}
	if metadata.Targets.Web == nil || metadata.Targets.SPIRV == nil || len(metadata.Targets.Web.Kernels) != 1 {
		t.Fatalf("targets = %#v", metadata.Targets)
	}
	if got := metadata.Targets.Web.Kernels[0].EntryPoint; got != "_tach_k0" {
		t.Fatalf("entry = %q", got)
	}
	if !strings.Contains(artifacts.Declarations, "$LaunchOptions<number>") || !strings.Contains(artifacts.JavaScript, "$tach.command(0") {
		t.Fatalf("generated artifacts:\n%s\n%s", artifacts.Declarations, artifacts.JavaScript)
	}
}

func TestGenerateOrchestrationUsesCommandOptions(t *testing.T) {
	artifacts, metadata := generateSource(t, `
function copy[i](input: buffer<float32[]>, output: buffer<float32[]>) { output[i] = input[i]; }
function twice[i](input: buffer<float32[]>, output: buffer<float32[]>) { output[i] = input[i] * 2.0; }
export function transform(input: buffer<float32[]>, output: buffer<float32[]>) {
  const count = min(input.length, output.length);
  const temporary = transient<float32>(count);
  run copy(input, temporary) over count;
  run twice(temporary, output) over count;
}`, false)
	plan := metadata.Targets.Web.Programs[0]
	if len(plan.Steps) != 2 || len(plan.Transients) != 1 {
		t.Fatalf("plan = %#v", plan)
	}
	if !strings.Contains(artifacts.Declarations, "$options?: $CommandOptions") {
		t.Fatalf("declarations:\n%s", artifacts.Declarations)
	}
	if metadata.Targets.SPIRV != nil {
		t.Fatal("unexpected SPIR-V target")
	}
}

func TestValidateMetadataRejectsCorruptRuntimeSeams(t *testing.T) {
	_, metadata := generateSource(t, `export function fill[i](out: buffer<uint32[]>) { if (i < out.length) { out[i] = i; } }`, false)
	mutations := map[string]func(*Metadata){
		"schema":        func(m *Metadata) { m.Schema = 0 },
		"no target":     func(m *Metadata) { m.Targets.Web = nil },
		"program count": func(m *Metadata) { m.Targets.Web.Programs = nil },
		"program index": func(m *Metadata) { m.Targets.Web.Programs[0].Program = 1 },
		"repeat":        func(m *Metadata) { m.Targets.Web.Programs[0].Repeat = "invalid" },
		"step kind":     func(m *Metadata) { m.Targets.Web.Programs[0].Steps[0].Kind = "invalid" },
		"kernel link":   func(m *Metadata) { m.Targets.Web.Programs[0].Steps[0].Kernel = 99 },
		"entry point":   func(m *Metadata) { m.Targets.Web.Kernels[0].EntryPoint = "wrong" },
		"workgroup":     func(m *Metadata) { m.Targets.Web.Kernels[0].WorkgroupSize[0] = 0 },
		"binding group": func(m *Metadata) { m.Targets.Web.Kernels[0].Bindings[0].Group = 1 },
		"binding index": func(m *Metadata) { m.Targets.Web.Kernels[0].Bindings[0].Binding = 2 },
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

func TestDocumentationReachesTypeScriptDeclarations(t *testing.T) {
	artifacts, _ := generateSource(t, `
@docs(summary("A host-visible value."), field(value, "The stored value."))
type Item = { value: float32 };
@docs(summary("Scales items."), coordinate(i, "Item index."), param(items, "Items to update."))
export function scale[i](items: buffer<Item[]>) { if (i < items.length) { items[i].value *= 2.0; } }
`, false)
	for _, want := range []string{"A host-visible value.", "The stored value.", "Scales items.", "@remarks Coordinate `i`: Item index.", "@param items Read/write GPU buffer. Items to update."} {
		if !strings.Contains(artifacts.Declarations, want) {
			t.Fatalf("declarations missing %q:\n%s", want, artifacts.Declarations)
		}
	}
}
