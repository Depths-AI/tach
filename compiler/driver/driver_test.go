package driver

import (
	"bytes"
	"compress/gzip"
	"encoding/json"
	"io"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

const baselineSource = `export function scale[i](values: buffer<float32[]>, factor: float32) {
  if (i < values.length) { values[i] *= factor; }
}`

func projectFixture(t *testing.T, files map[string]string) string {
	t.Helper()
	root := t.TempDir()
	manifest := `{"name":"fixture","version":"0.1.0","javascript":{"package":"@test/fixture"},"docs":{"title":"Fixture","summary":"Fixture project."}}`
	if value, ok := files["tach.json"]; ok {
		manifest = value
		delete(files, "tach.json")
	}
	writeFixture(t, root, "tach.json", manifest)
	for name, source := range files {
		writeFixture(t, root, name, source)
	}
	return root
}

func writeFixture(t *testing.T, root, name, content string) {
	t.Helper()
	path := filepath.Join(root, filepath.FromSlash(name))
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}
}

func TestProjectBuildWritesCompleteNativeArtifacts(t *testing.T) {
	root := projectFixture(t, map[string]string{"kernels/scale.tach": baselineSource})
	for _, verbose := range []bool{false, true} {
		result, err := Build(root, 1)
		if err != nil {
			t.Fatal(err)
		}
		output := t.TempDir()
		if err := WriteNativeArtifacts(result, output, verbose); err != nil {
			t.Fatal(err)
		}
		entries, err := os.ReadDir(output)
		if err != nil {
			t.Fatal(err)
		}
		names := make([]string, len(entries))
		for i := range entries {
			names[i] = entries[i].Name()
		}
		want := "kernel.spv,kernel.wgsl.gz,project.json,runtime.json"
		if verbose {
			want = "diagnostics,kernel.spv,kernel.wgsl.gz,project.json,runtime.json"
		}
		if got := strings.Join(names, ","); got != want {
			t.Fatalf("verbose=%v files = %s, want %s", verbose, got, want)
		}
		compressed, err := os.ReadFile(filepath.Join(output, "kernel.wgsl.gz"))
		if err != nil {
			t.Fatal(err)
		}
		reader, err := gzip.NewReader(bytes.NewReader(compressed))
		if err != nil {
			t.Fatal(err)
		}
		decompressed, err := io.ReadAll(reader)
		closeErr := reader.Close()
		if err != nil || closeErr != nil || string(decompressed) != result.WGSL {
			t.Fatalf("compressed WGSL did not round-trip: read=%v close=%v", err, closeErr)
		}
		if verbose {
			entries, err := os.ReadDir(filepath.Join(output, "diagnostics"))
			if err != nil {
				t.Fatal(err)
			}
			var diagnosticNames []string
			for _, entry := range entries {
				diagnosticNames = append(diagnosticNames, entry.Name())
			}
			if got, want := strings.Join(diagnosticNames, ","), "flow.ir,kernel.ir,kernel.spvasm,spirv.kernel.ir,spirv.plan.json,web.kernel.ir,web.plan.json"; got != want {
				t.Fatalf("diagnostics = %s, want %s", got, want)
			}
		}
	}
}

func TestBackendsOwnIndependentPhysicalIR(t *testing.T) {
	root := projectFixture(t, map[string]string{"core/scale.tach": baselineSource})
	result, err := Build(root, 2)
	if err != nil {
		t.Fatal(err)
	}
	web, native, logical := result.Web.KernelModule, result.Native.KernelModule, result.Module.Kernel
	if web == logical || native == logical || web == native || len(web.Functions) == 0 || len(native.Functions) == 0 || len(logical.Functions) == 0 {
		t.Fatal("backends did not produce independent physical IR modules")
	}
	web.Functions[0].Name = "mutated_web_entry"
	if native.Functions[0].Name == "mutated_web_entry" || logical.Functions[0].Name == "mutated_web_entry" {
		t.Fatal("Web physical IR aliases native or logical IR")
	}
}

func TestProjectDescriptionOwnsKernelsAndTargetNeutralABI(t *testing.T) {
	root := projectFixture(t, map[string]string{
		"data/types.tach":    `@docs(title("Values"), summary("Shared values.")); @docs(summary("A value."), field(x, "Coordinate.")) type Value = { x: float32, };`,
		"kernels/scale.tach": `@docs(title("Scale"), summary("Scaling.")); import "data/types"; @docs(summary("Scales."), coordinate(i, "Index."), param(values, "Values.")) export function scale[i](values: buffer<Value[]>) { if (i < values.length) { values[i].x *= 2.0; } }`,
	})
	result, err := Describe(root, 1)
	if err != nil {
		t.Fatal(err)
	}
	var value map[string]any
	if err := json.Unmarshal(result.Description, &value); err != nil {
		t.Fatal(err)
	}
	text := string(result.Description)
	for _, want := range []string{`"schema": 2`, `"package": "@test/fixture"`, `"identity": "data/types"`, `"role": "kernel"`, `"access": "readWrite"`} {
		if !strings.Contains(text, want) {
			t.Fatalf("description missing %s:\n%s", want, text)
		}
	}
	if strings.Contains(text, "ComputeBuffer") || strings.Contains(text, "TypeScript") {
		t.Fatalf("target syntax leaked into description:\n%s", text)
	}
}

func TestManifestIsStrict(t *testing.T) {
	valid := `{"name":"fixture","version":"0.1.0","javascript":{"package":"@test/fixture"},"docs":{"title":"Fixture","summary":"Fixture project."}}`
	for name, manifest := range map[string]string{
		"missing":    `{"name":"fixture"}`,
		"duplicate":  `{"name":"a","name":"b","version":"0.1.0","javascript":{"package":"x"},"docs":{"title":"x","summary":"x"}}`,
		"unknown":    strings.Replace(valid, `"name":"fixture"`, `"name":"fixture","modules":[]`, 1),
		"version":    strings.Replace(valid, "0.1.0", "v1", 1),
		"prerelease": strings.Replace(valid, "0.1.0", "1.0.0-01", 1),
		"package":    strings.Replace(valid, "@test/fixture", "INVALID PACKAGE", 1),
	} {
		t.Run(name, func(t *testing.T) {
			root := projectFixture(t, map[string]string{"tach.json": manifest, "m/a.tach": baselineSource})
			if _, err := loadProject(root, 1); err == nil {
				t.Fatal("invalid manifest accepted")
			}
		})
	}
	root := projectFixture(t, map[string]string{"tach.json": `{}`, "m/a.tach": baselineSource})
	if _, err := loadProject(root, 1); err == nil || !strings.Contains(err.Error(), "name is required") {
		t.Fatalf("required-field diagnostic = %v", err)
	}
}

func TestDiscoveryFromNestedDirectoryAndLayoutValidation(t *testing.T) {
	root := projectFixture(t, map[string]string{"module/kernel.tach": baselineSource})
	nested := filepath.Join(root, "unrelated", "nested")
	if err := os.MkdirAll(nested, 0o755); err != nil {
		t.Fatal(err)
	}
	project, err := loadProject(nested, 1)
	if err != nil || project.Root != root {
		t.Fatalf("nested discovery = %v, %v", project, err)
	}
	writeFixture(t, root, "root.tach", baselineSource)
	writeFixture(t, root, "module/deeper/invalid.tach", baselineSource)
	if _, err := loadProject(root, 1); err == nil || !strings.Contains(err.Error(), "root-level") || !strings.Contains(err.Error(), "exactly one") {
		t.Fatalf("layout diagnostics = %v", err)
	}
}

func TestDiscoveryIgnoresUnrelatedDirectoryNames(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("case-distinct sibling directories require a case-sensitive filesystem")
	}
	root := projectFixture(t, map[string]string{"module/kernel.tach": baselineSource})
	for _, name := range []string{"Unrelated", "unrelated"} {
		if err := os.Mkdir(filepath.Join(root, name), 0o755); err != nil {
			t.Fatal(err)
		}
	}
	if _, err := loadProject(root, 1); err != nil {
		t.Fatal(err)
	}
}

func TestDiscoveryRejectsDuplicatePhysicalKernel(t *testing.T) {
	root := projectFixture(t, map[string]string{"a/one.tach": baselineSource})
	if err := os.Mkdir(filepath.Join(root, "b"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.Link(filepath.Join(root, "a", "one.tach"), filepath.Join(root, "b", "two.tach")); err != nil {
		t.Skipf("hard links unavailable: %v", err)
	}
	if _, err := loadProject(root, 1); err == nil || !strings.Contains(err.Error(), "physical kernel is already included") {
		t.Fatalf("physical duplicate diagnostic = %v", err)
	}
}

func TestImportsEnforceShapeTargetsDuplicatesAndSelf(t *testing.T) {
	root := projectFixture(t, map[string]string{
		"a/one.tach": `import "bad"; import "@scope/package"; import "a/one"; import "b/missing"; import "b/two"; import "b/two"; function one(x: float32): float32 { return two(x); }`,
		"b/two.tach": `function two(x: float32): float32 { return x; }`,
	})
	_, err := loadProject(root, 1)
	if err == nil {
		t.Fatal("invalid imports accepted")
	}
	for _, want := range []string{"invalid import", "cannot import itself", "does not exist", "duplicate import"} {
		if !strings.Contains(err.Error(), want) {
			t.Errorf("diagnostics missing %q:\n%s", want, err)
		}
	}
}

func TestCanonicalImportShape(t *testing.T) {
	for _, target := range []string{"physics/integration", "my-module/my-kernel"} {
		if !validImport(target) {
			t.Errorf("valid import %q rejected", target)
		}
	}
	for _, target := range []string{"", "kernel", "/kernel", "module/", "./kernel", "../kernel", "module/.", "module/..", "module/deeper/kernel", `module\kernel`, "C:/kernel", "https://kernel", "module/kernel.tach", "@scope/package"} {
		if validImport(target) {
			t.Errorf("invalid import %q accepted", target)
		}
	}
}

func TestDiscoveryRejectsKernelIdentityThatCannotBeImported(t *testing.T) {
	root := projectFixture(t, map[string]string{"@scope/kernel.tach": baselineSource})
	if _, err := loadProject(root, 1); err == nil || !strings.Contains(err.Error(), `invalid kernel identity "@scope/kernel"`) {
		t.Fatalf("invalid discovered identity = %v", err)
	}
}

func TestVisibilityIsDirectAndImportsExposeAllDeclarationRoles(t *testing.T) {
	valid := projectFixture(t, map[string]string{
		"base/data.tach": `type Value = { x: float32, }; function helper(x: Value): float32 { return x.x; } function stage[i](out: buffer<float32[]>, x: Value) { if (i < out.length) { out[i] = helper(x); } } export function publicStage[i](out: buffer<float32[]>, x: Value) { if (i < out.length) { out[i] = helper(x); } }`,
		"app/main.tach":  `import "base/data"; function local[i](out: buffer<float32[]>, x: Value) { if (i < out.length) { out[i] = helper(x); } } export function execute(out: buffer<float32[]>, x: Value, count: uint32) { run stage(out, x) over count; run publicStage(out, x) over count; run local(out, x) over count; }`,
	})
	if _, err := Check(valid, 2); err != nil {
		t.Fatal(err)
	}
	for role, files := range map[string]map[string]string{
		"type": {
			"base/data.tach":   `type Hidden = { x: float32, };`,
			"middle/pass.tach": `import "base/data"; function read(x: Hidden): float32 { return x.x; }`,
			"app/main.tach":    `import "middle/pass"; export function use[i](out: buffer<float32[]>, x: Hidden) {}`,
		},
		"helper": {
			"base/data.tach":   `function hidden(x: float32): float32 { return x; }`,
			"middle/pass.tach": `import "base/data"; function pass(x: float32): float32 { return hidden(x); }`,
			"app/main.tach":    `import "middle/pass"; export function use[i](out: buffer<float32[]>) { if (i < out.length) { out[i] = hidden(out[i]); } }`,
		},
		"private stage": {
			"base/data.tach":   `function hidden[i](out: buffer<float32[]>) {}`,
			"middle/pass.tach": `import "base/data"; function pass[i](out: buffer<float32[]>) {}`,
			"app/main.tach":    `import "middle/pass"; export function use(out: buffer<float32[]>, count: uint32) { run hidden(out) over count; }`,
		},
		"public stage": {
			"base/data.tach":   `export function hidden[i](out: buffer<float32[]>) {}`,
			"middle/pass.tach": `import "base/data"; function pass[i](out: buffer<float32[]>) {}`,
			"app/main.tach":    `import "middle/pass"; export function use(out: buffer<float32[]>, count: uint32) { run hidden(out) over count; }`,
		},
		"explicit program": {
			"base/data.tach":   `function stage[i](out: buffer<float32[]>) {} export function hidden(out: buffer<float32[]>, count: uint32) { run stage(out) over count; }`,
			"middle/pass.tach": `import "base/data"; function pass[i](out: buffer<float32[]>) {}`,
			"app/main.tach":    `import "middle/pass"; export function use(out: buffer<float32[]>, count: uint32) { run hidden(out, count) over count; }`,
		},
	} {
		t.Run("non-transitive "+role, func(t *testing.T) {
			if _, err := Check(projectFixture(t, files), 2); err == nil || (!strings.Contains(err.Error(), "unknown") && !strings.Contains(err.Error(), "not an indexed stage")) {
				t.Fatalf("non-transitive %s visibility = %v", role, err)
			}
		})
	}
	roles := projectFixture(t, map[string]string{
		"base/data.tach": `function stage[i](out: buffer<float32[]>) {} export function program(out: buffer<float32[]>, count: uint32) { run stage(out) over count; }`,
		"app/main.tach":  `import "base/data"; export function execute(out: buffer<float32[]>, count: uint32) { run program(out, count) over count; }`,
	})
	if _, err := Check(roles, 2); err == nil || !strings.Contains(err.Error(), `public program "program" cannot be a run target`) {
		t.Fatalf("direct explicit-program role diagnostic = %v", err)
	}
}

func TestPublicProgramsRequireExternalBuffersAndFunctionRolesAreDiagnosed(t *testing.T) {
	for name, source := range map[string]string{
		"indexed buffer": `export function value[i](factor: float32) {}`,
		"program buffer": `function stage[i](out: buffer<float32[]>) {} export function value(count: uint32) { let scratch = transient<float32>(count); run stage(scratch) over count; }`,
		"stage call":     `function stage[i](out: buffer<float32[]>) {} export function value[i](out: buffer<float32[]>) { stage(out); }`,
		"program call":   `function stage[i](out: buffer<float32[]>) {} export function program(out: buffer<float32[]>, count: uint32) { run stage(out) over count; } export function value[i](out: buffer<float32[]>) { program(out, out.length); }`,
	} {
		t.Run(name, func(t *testing.T) {
			root := projectFixture(t, map[string]string{"m/forms.tach": source})
			_, err := Check(root, 1)
			want := "requires at least one buffer parameter"
			if strings.Contains(name, "call") {
				want = map[string]string{"stage call": "indexed stage", "program call": "public program"}[name]
			}
			if err == nil || !strings.Contains(err.Error(), want) || strings.Contains(err.Error(), "internal Flow IR verification failed") {
				t.Fatalf("diagnostic = %v, want %q", err, want)
			}
		})
	}
}

func TestFunctionFormsHaveExactHostExposure(t *testing.T) {
	root := projectFixture(t, map[string]string{
		"m/forms.tach": `type Value = { x: float32, };
function helper(x: float32): float32 { return x * 2.0; }
function privateStage[i](out: buffer<float32[]>) { if (i < out.length) { out[i] = helper(out[i]); } }
export function publicKernel[i](out: buffer<float32[]>) { if (i < out.length) { out[i] = helper(out[i]); } }
export function orchestrate(out: buffer<float32[]>, count: uint32) { run privateStage(out) over count; }`,
	})
	result, err := Build(root, 2)
	if err != nil {
		t.Fatal(err)
	}
	if got := result.Module.Programs; len(got) != 2 || got[0].Name != "publicKernel" || got[1].Name != "orchestrate" {
		t.Fatalf("public programs = %v", got)
	}
	for _, role := range []string{`"role": "helper"`, `"role": "stage"`, `"role": "kernel"`, `"role": "program"`} {
		if !bytes.Contains(result.Description, []byte(role)) {
			t.Errorf("description omits %s", role)
		}
	}
}

func TestGlobalNamespaceCollisionIncludesRelatedLocation(t *testing.T) {
	root := projectFixture(t, map[string]string{
		"a/one.tach": `type Same = { x: float32, };`,
		"b/two.tach": `function Same(x: float32): float32 { return x; }`,
	})
	_, err := loadProject(root, 1)
	if err == nil || !strings.Contains(err.Error(), `project declaration "Same" is already defined`) || !strings.Contains(err.Error(), "related a/one.tach") {
		t.Fatalf("collision diagnostic = %v", err)
	}
}

func TestGeneratedExportsRejectTypeScriptKeywords(t *testing.T) {
	root := projectFixture(t, map[string]string{
		"a/one.tach": `type interface = { x: float32, }; type Float32Array = { x: float32, }; function private(x: float32): float32 { return x; } export function await[i](out: buffer<float32[]>, arguments: float32) { if (i < out.length) { out[i] = private(out[i]) * arguments; } }`,
	})
	_, err := loadProject(root, 1)
	if err == nil || !strings.Contains(err.Error(), `"interface" is not a valid generated export`) || !strings.Contains(err.Error(), `"Float32Array" collides with a generated host collection type`) || !strings.Contains(err.Error(), `"await" is not a valid generated export`) || !strings.Contains(err.Error(), `parameter name: "arguments" is not a valid generated export`) {
		t.Fatalf("export diagnostics = %v", err)
	}
}

func TestKernelAndCollapsedModuleCyclesAreRejected(t *testing.T) {
	kernelCycle := projectFixture(t, map[string]string{
		"a/one.tach": `import "b/two"; function one(x: float32): float32 { return x; }`,
		"b/two.tach": `import "a/one"; function two(x: float32): float32 { return x; }`,
	})
	_, one := loadProject(kernelCycle, 1)
	_, many := loadProject(kernelCycle, runtime.GOMAXPROCS(0))
	if one == nil || many == nil || one.Error() != many.Error() || !strings.Contains(one.Error(), "kernel import cycle: a/one -> b/two -> a/one") || !strings.Contains(one.Error(), "module import cycle: a -> b -> a") || !strings.Contains(one.Error(), "a/one imports b/two") || !strings.Contains(one.Error(), "b/two imports a/one") {
		t.Fatalf("kernel cycle diagnostics differ or omit the complete chain:\nONE:\n%v\nMANY:\n%v", one, many)
	}
	moduleCycle := projectFixture(t, map[string]string{
		"a/one.tach":   `import "b/two"; function one(x: float32): float32 { return x; }`,
		"a/four.tach":  `function four(x: float32): float32 { return x; }`,
		"b/two.tach":   `function two(x: float32): float32 { return x; }`,
		"b/three.tach": `import "a/four"; function three(x: float32): float32 { return x; }`,
	})
	_, one = loadProject(moduleCycle, 1)
	_, many = loadProject(moduleCycle, runtime.GOMAXPROCS(0))
	if one == nil || many == nil || one.Error() != many.Error() || strings.Contains(one.Error(), "kernel import cycle") || !strings.Contains(one.Error(), "module import cycle: a -> b -> a") || !strings.Contains(one.Error(), "a imports b") || !strings.Contains(one.Error(), "b imports a") {
		t.Fatalf("module cycle diagnostics differ or omit the complete chain:\nONE:\n%v\nMANY:\n%v", one, many)
	}
}

func TestDisjointCyclesAreAllReportedInCanonicalOrder(t *testing.T) {
	root := projectFixture(t, map[string]string{
		"a/one.tach":   `import "b/two"; function one(): uint32 { return 1; }`,
		"b/two.tach":   `import "a/one"; function two(): uint32 { return 2; }`,
		"c/three.tach": `import "d/four"; function three(): uint32 { return 3; }`,
		"d/four.tach":  `import "c/three"; function four(): uint32 { return 4; }`,
	})
	_, one := loadProject(root, 1)
	_, many := loadProject(root, runtime.GOMAXPROCS(0))
	if one == nil || many == nil || one.Error() != many.Error() || strings.Count(one.Error(), "kernel import cycle:") != 2 || strings.Count(one.Error(), "module import cycle:") != 2 {
		t.Fatalf("disjoint cycle diagnostics are incomplete or nondeterministic:\nONE:\n%v\nMANY:\n%v", one, many)
	}
	if first, second := strings.Index(one.Error(), "a/one -> b/two -> a/one"), strings.Index(one.Error(), "c/three -> d/four -> c/three"); first < 0 || second < 0 || first >= second {
		t.Fatalf("cycle order is not canonical:\n%s", one)
	}
}

func TestRecoverableFrontendReportsIndependentFilesDeterministically(t *testing.T) {
	root := projectFixture(t, map[string]string{
		"a/one.tach": `function one( { return; } @ function broken() { return; }`,
		"b/two.tach": "function two() { const x = \"unterminated\n return; }",
	})
	_, first := loadProject(root, 1)
	_, many := loadProject(root, runtime.GOMAXPROCS(0))
	if first == nil || many == nil || first.Error() != many.Error() {
		t.Fatalf("diagnostics differ:\nONE:\n%v\nMANY:\n%v", first, many)
	}
	if !strings.Contains(first.Error(), "a/one.tach") || !strings.Contains(first.Error(), "b/two.tach") {
		t.Fatalf("missing cross-file diagnostics:\n%s", first)
	}
}

func TestDocumentationAndSemanticErrorsAccumulateAcrossFiles(t *testing.T) {
	root := projectFixture(t, map[string]string{
		"a/one.tach": `@docs(title("Missing summary")); function one(): float32 { return missing; }`,
		"b/two.tach": `@docs(summary("Wrong return.")) function two(): float32 { return true; }`,
	})
	_, err := Check(root, 1)
	_, many := Check(root, runtime.GOMAXPROCS(0))
	if err == nil || many == nil {
		t.Fatal("invalid project accepted")
	}
	if err.Error() != many.Error() {
		t.Fatalf("diagnostics differ:\nONE:\n%s\nMANY:\n%s", err, many)
	}
	for _, want := range []string{`@docs requires summary`, `unknown identifier "missing"`, `bool literal cannot be used as float32`} {
		if !strings.Contains(err.Error(), want) {
			t.Errorf("diagnostics missing %q:\n%s", want, err)
		}
	}
}
