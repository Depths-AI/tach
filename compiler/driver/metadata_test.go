package driver

import (
	"encoding/json"
	"slices"
	"testing"

	"tach/parser"
	"tach/semantics"
	"tach/spirv"
	"tach/wgsl"
)

type decodedTarget struct {
	Vulkan, SPIRV string
	Features      []string
	Kernels       []struct {
		EntryPoint     string
		Bindings       []struct{ Kind string }
		ParameterBlock *struct {
			Fields []struct{ Layout *HostLayout }
		}
	}
	Programs []struct {
		Transients []json.RawMessage
		Steps      []struct {
			Parameters []struct{ Kind string }
		}
		View *struct {
			Format        string
			Fused         bool
			Output        uint32
			Width, Height struct{ Op string }
			Step          struct {
				Kind      string
				Kernel    int
				Resources []json.RawMessage
			}
		}
	}
}

func generatedMetadata(t *testing.T, source string) (*Metadata, decodedTarget, decodedTarget) {
	t.Helper()
	parsed, err := parser.Parse("metadata.tach", source)
	if err != nil {
		t.Fatal(err)
	}
	result, err := semantics.Build([]*parser.File{parsed}, 0)
	if err != nil {
		t.Fatal(err)
	}
	web, err := wgsl.Lower(result.Module)
	if err != nil {
		t.Fatal(err)
	}
	native, err := spirv.Lower(result.Module)
	if err != nil {
		t.Fatal(err)
	}
	metadata, encoded, err := generateMetadata(result.Module, web.RuntimeJSON, native.RuntimeJSON)
	if err != nil {
		t.Fatal(err)
	}
	var decoded Metadata
	if err := json.Unmarshal(encoded, &decoded); err != nil {
		t.Fatal(err)
	}
	var webTarget, nativeTarget decodedTarget
	if err := json.Unmarshal(decoded.Targets.Web, &webTarget); err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(decoded.Targets.SPIRV, &nativeTarget); err != nil {
		t.Fatal(err)
	}
	return metadata, webTarget, nativeTarget
}

func TestGenerateCompleteRuntimeMetadata(t *testing.T) {
	metadata, web, native := generatedMetadata(t, `export function scale[i](values: buffer<float32[]>, factor: float32) { if (i < values.length) { values[i] *= factor; } }`)
	if metadata.Schema != 2 || len(metadata.Programs) != 1 || metadata.Programs[0].Name != "scale" || len(web.Kernels) != 1 || web.Kernels[0].EntryPoint != "_tach_k0" {
		t.Fatalf("metadata = %#v, web = %#v", metadata, web)
	}
	if native.Vulkan != "1.3" || native.SPIRV != "1.6" || len(native.Features) != 3 {
		t.Fatalf("SPIR-V profile = %#v", native)
	}
}

func TestGenerateFloat16Contract(t *testing.T) {
	metadata, web, native := generatedMetadata(t, `export function half[i](values: buffer<vec<float16, 3>[]>, factor: float16) { if (i < values.length) { values[i] *= factor; } }`)
	if !slices.Equal(web.Features, []string{"shader-f16"}) {
		t.Fatalf("web features = %v", web.Features)
	}
	want := []string{"synchronization2", "shaderZeroInitializeWorkgroupMemory", "vulkanMemoryModel", "shaderFloat16", "storageBuffer16BitAccess", "uniformAndStorageBuffer16BitAccess"}
	if !slices.Equal(native.Features, want) {
		t.Fatalf("SPIR-V features = %v, want %v", native.Features, want)
	}
	resource := metadata.Programs[0].Resources[0]
	if resource.RuntimeStride != 8 || resource.Alignment != 8 || resource.Layout.Elem.Kind != "vector" || resource.Layout.Elem.Size != 6 || resource.Layout.Elem.Elem.Kind != "f16" {
		t.Fatalf("Float16 resource = %#v", resource)
	}
	parameter := web.Kernels[0].ParameterBlock.Fields[0]
	if parameter.Layout.Kind != "f16" || parameter.Layout.Size != 2 {
		t.Fatalf("Float16 parameter = %#v", parameter)
	}
}

func TestFloat16FeaturesMatchInterfacesExactly(t *testing.T) {
	for _, test := range []struct {
		name, source string
		extra        []string
	}{
		{"computation", `export function half[i](out: buffer<uint32[]>) { let value: float16 = 1.0; if (i < out.length) { out[i] = uint32(value); } }`, []string{"shaderFloat16"}},
		{"storage", `export function half[i](out: buffer<float16[]>) { if (i < out.length) { out[i] = 1.0; } }`, []string{"shaderFloat16", "storageBuffer16BitAccess"}},
		{"uniform", `export function half[i](out: buffer<uint32[]>, value: float16) { if (i < out.length) { out[i] = uint32(value); } }`, []string{"shaderFloat16", "uniformAndStorageBuffer16BitAccess"}},
	} {
		t.Run(test.name, func(t *testing.T) {
			_, web, native := generatedMetadata(t, test.source)
			if !slices.Equal(web.Features, []string{"shader-f16"}) || !slices.Equal(native.Features[3:], test.extra) {
				t.Fatalf("features: web=%v SPIR-V=%v", web.Features, native.Features)
			}
		})
	}
}

func TestCompileTimeArgumentNeverReachesRuntimeMetadata(t *testing.T) {
	_, web, native := generatedMetadata(t, `
const factor: float16 = 0.5;
function half[i](values: buffer<float16[]>, factor: float16) { if (i < values.length) { values[i] *= factor; } }
export function halve(values: buffer<float16[]>) { run half(values, factor) over values.length; }`)
	for name, target := range map[string]decodedTarget{"web": web, "spirv": native} {
		for _, source := range target.Programs[0].Steps[0].Parameters {
			if source.Kind != "shape" && source.Kind != "repeat" {
				t.Fatalf("%s exposed compile-time Float16: %#v", name, source)
			}
		}
	}
}

func TestGenerateOrchestrationPlansForBothBackends(t *testing.T) {
	_, web, native := generatedMetadata(t, `
function copy[i](input: buffer<float32[]>, output: buffer<float32[]>) { output[i] = input[i]; }
function twice[i](input: buffer<float32[]>, output: buffer<float32[]>) { output[i] = input[i] * 2.0; }
export function transform(input: buffer<float32[]>, output: buffer<float32[]>) {
  let count = min(input.length, output.length);
  let temporary = transient<float32>(count);
  run copy(input, temporary) over count;
  run twice(temporary, output) over count;
}`)
	for name, target := range map[string]decodedTarget{"web": web, "spirv": native} {
		if len(target.Programs[0].Steps) < 2 || len(target.Programs[0].Transients) != 1 {
			t.Errorf("%s plan = %#v", name, target.Programs[0])
		}
	}
}

func TestGenerateViewContractForBothBackends(t *testing.T) {
	metadata, web, native := generatedMetadata(t, `
function paint[i](pixels: buffer<vec<float32, 4>[]>) { if (i < pixels.length) { pixels[i] = vec(0.1, 0.2, 0.3, 1.0); } }
export function image(width: uint32, height: uint32): view<srgb8> {
  let pixels = transient<vec<float32, 4>>(width * height);
  run paint(pixels) over pixels.length;
  return view(pixels, width, height);
}`)
	if !metadata.Programs[0].View || len(metadata.Programs[0].Resources) != 0 {
		t.Fatalf("public view metadata = %#v", metadata.Programs[0])
	}
	for name, target := range map[string]decodedTarget{"web": web, "spirv": native} {
		view := target.Programs[0].View
		if view == nil || view.Format != "srgb8" || !view.Fused || view.Step.Kind != "dispatch" || len(view.Step.Resources) != 0 || view.Width.Op != "parameter" || view.Height.Op != "parameter" {
			t.Fatalf("%s view metadata = %#v", name, view)
		}
		want := "buffer"
		if name == "web" {
			want = "texture"
		}
		if bindings := target.Kernels[view.Step.Kernel].Bindings; len(bindings) != 1 || bindings[0].Kind != want || view.Output != 0 {
			t.Fatalf("%s projection bindings = %#v", name, bindings)
		}
	}
}

func TestGenerateMetadataRejectsIncompleteArtifacts(t *testing.T) {
	parsed, err := parser.Parse("metadata.tach", `export function fill[i](out: buffer<uint32[]>) { out[i] = i; }`)
	if err != nil {
		t.Fatal(err)
	}
	result, err := semantics.Build([]*parser.File{parsed}, 0)
	if err != nil {
		t.Fatal(err)
	}
	if _, _, err := generateMetadata(result.Module, nil, json.RawMessage(`{}`)); err == nil {
		t.Fatal("accepted missing Web runtime metadata")
	}
}
