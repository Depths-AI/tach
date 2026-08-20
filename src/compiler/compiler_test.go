package compiler

import (
	"bytes"
	"compress/gzip"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"runtime"
	"slices"
	"strings"
	"testing"

	"tach/src/lexer"
	"tach/src/parser"
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
		"program buffer": `function stage[i](out: buffer<float32[]>) {} export function value(count: uint32) { const scratch = transient<float32>(count); run stage(scratch) over count; }`,
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

func TestOneAndManyWorkerBuildsAreByteIdentical(t *testing.T) {
	root := projectFixture(t, map[string]string{
		"base/helper.tach":   `function twice(x: float32): float32 { return x * 2.0; }`,
		"kernels/scale.tach": `import "base/helper"; export function scale[i](values: buffer<float32[]>) { if (i < values.length) { values[i] = twice(values[i]); } }`,
	})
	one, err := Check(root, 1)
	if err != nil {
		t.Fatal(err)
	}
	many, err := Check(root, runtime.GOMAXPROCS(0))
	if err != nil {
		t.Fatal(err)
	}
	if one.WGSL != many.WGSL || !bytes.Equal(one.CompressedWGSL, many.CompressedWGSL) || !bytes.Equal(one.SPIRV, many.SPIRV) || !bytes.Equal(one.MetadataJSON, many.MetadataJSON) || !bytes.Equal(one.Description, many.Description) {
		t.Fatal("worker count changed artifacts")
	}
}

func TestWideProjectBuildIsCompleteAndDeterministic(t *testing.T) {
	files := map[string]string{}
	var imports, calls strings.Builder
	for i := range 32 {
		module := fmt.Sprintf("library%02d", i)
		typeName := fmt.Sprintf("Value%02d", i)
		helper := fmt.Sprintf("helper%02d", i)
		files[module+"/values.tach"] = fmt.Sprintf("type %s = { value: float32, }; function %s(value: float32): float32 { return value; }", typeName, helper)
		fmt.Fprintf(&imports, "import %q; ", module+"/values")
		fmt.Fprintf(&calls, "total += %s(float32(%d)); ", helper, i)
	}
	files["app/main.tach"] = imports.String() + `export function execute[i](out: buffer<float32[]>) { if (i < out.length) { let total = 0.0; ` + calls.String() + `out[i] = total; } }`
	root := projectFixture(t, files)
	one, err := Check(root, 1)
	if err != nil {
		t.Fatal(err)
	}
	many, err := Check(root, runtime.GOMAXPROCS(0))
	if err != nil {
		t.Fatal(err)
	}
	if one.WGSL != many.WGSL || !bytes.Equal(one.CompressedWGSL, many.CompressedWGSL) || !bytes.Equal(one.SPIRV, many.SPIRV) || !bytes.Equal(one.MetadataJSON, many.MetadataJSON) || !bytes.Equal(one.Description, many.Description) {
		t.Fatal("wide-project artifacts changed with worker count")
	}
	for i := range 32 {
		if name := fmt.Sprintf(`"name": "Value%02d"`, i); !strings.Contains(string(one.Description), name) {
			t.Fatalf("project description omits %s", name)
		}
	}
}

func TestWideProjectErrorsAreCompleteAndDeterministic(t *testing.T) {
	files := map[string]string{}
	for i := range 24 {
		files[fmt.Sprintf("module%02d/broken.tach", i)] = fmt.Sprintf("function broken%02d(): float32 { return missing%02d; }", i, i)
	}
	root := projectFixture(t, files)
	_, one := Check(root, 1)
	_, many := Check(root, runtime.GOMAXPROCS(0))
	if one == nil || many == nil || one.Error() != many.Error() || strings.Count(one.Error(), "unknown identifier") != len(files) {
		t.Fatalf("wide-project diagnostics are incomplete or nondeterministic:\nONE:\n%v\nMANY:\n%v", one, many)
	}
	for file := range files {
		if !strings.Contains(one.Error(), file) {
			t.Errorf("diagnostics omit %s", file)
		}
	}
}

func TestFormatterPreservesCommentsAndIsIdempotent(t *testing.T) {
	root := projectFixture(t, map[string]string{"m/a.tach": "// kept\r\nfunction helper(x:float32,):float32{return (-x);}\n@docs(summary(\"Scales.\"))@workgroup(64) export   function scale[i](values:buffer<float32[]>,factor:float32,){if(!false&&i<values.length){const negative=-64;const bits=~uint32(negative);const total=values[i]+values[i]+values[i]+values[i]+values[i]+values[i]+values[i]+values[i]+values[i]+values[i]; // trailing\r\nvalues[i]=!false?total*factor+float32(bits):-1.0;}}\n"})
	if err := Format(root, 1); err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(root, "m", "a.tach")
	first, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Contains(first, []byte("// kept")) || !bytes.Contains(first, []byte("; // trailing")) || !bytes.Contains(first, []byte("helper(x: float32): float32")) || !bytes.Contains(first, []byte("return (-x)")) || !bytes.Contains(first, []byte(")\n@workgroup(64)\nexport function")) || !bytes.Contains(first, []byte("if (!false &&")) || !bytes.Contains(first, []byte("negative = -64")) || !bytes.Contains(first, []byte("bits = ~uint32")) || !bytes.Contains(first, []byte("!false ? total * factor + float32(bits) : -1.0")) || bytes.Contains(first, []byte("helper(x: float32,)")) || bytes.ContainsAny(first, "\t\r") || !bytes.HasSuffix(first, []byte("\n")) {
		t.Fatalf("formatted source = %q", first)
	}
	for _, line := range strings.Split(string(first), "\n") {
		if len(line) > 100 {
			t.Fatalf("formatter produced %d columns: %s", len(line), line)
		}
	}
	if err := Format(root, 1); err != nil {
		t.Fatal(err)
	}
	second, _ := os.ReadFile(path)
	if !bytes.Equal(first, second) {
		t.Fatalf("formatter is not idempotent:\n%s\n---\n%s", first, second)
	}
}

func TestFormatterMakesNoPartialWritesOnSyntaxError(t *testing.T) {
	const original = `export   function valid[i](out:buffer<float32[]>){if(i<out.length){out[i]=1.0;}}`
	root := projectFixture(t, map[string]string{
		"m/a.tach": original,
		"m/b.tach": `export function broken[`,
	})
	if err := Format(root, 2); err == nil {
		t.Fatal("invalid project formatted")
	}
	content, err := os.ReadFile(filepath.Join(root, "m", "a.tach"))
	if err != nil {
		t.Fatal(err)
	}
	if string(content) != original {
		t.Fatalf("valid sibling changed to %q", content)
	}
}

func FuzzFormatterPreservesSyntaxAndStabilizes(f *testing.F) {
	for _, seed := range []string{
		`// comment
@docs(summary("Scales.")) @workgroup(64) export function scale[i](out: buffer<float32[]>, factor: float32,) { if (!false && i < out.length) { out[i] = factor < 0.0 ? -factor : factor; } }`,
		`import "base/data"; function value(x: float32): float32 { return (~uint32(x) & 3) == 0 ? -x : x; }`,
		`function fill[i](out: buffer<vec<float32, 4>[]>) { if (i < out.length) { out[i] = vec(1, 2, 3, 4); } } export function make() { const out = transient<vec<float32, 4>>(4); run fill(out) over 4; }`,
	} {
		f.Add(seed)
	}
	f.Fuzz(func(t *testing.T, input string) {
		// DECISION: bound one fuzz case at 64 KiB; project tests own larger-file pressure, and this cap can rise with formatter limits.
		if len(input) > 64<<10 {
			t.Skip()
		}
		if _, diagnostics := parser.ParseRecover("fuzz.tach", input); len(diagnostics) > 0 {
			return
		}
		formatted, err := formatSource("fuzz.tach", input)
		if err != nil {
			t.Fatal(err)
		}
		if _, diagnostics := parser.ParseRecover("fuzz.tach", formatted); len(diagnostics) > 0 {
			t.Fatalf("formatter produced invalid syntax:\n%s\n%v", formatted, diagnostics)
		}
		again, err := formatSource("fuzz.tach", formatted)
		if err != nil || again != formatted {
			t.Fatalf("formatter is not idempotent:\n%s\n---\n%s\n%v", formatted, again, err)
		}
		preserved := func(source string) []string {
			tokens, _ := lexer.LexRecover("fuzz.tach", source)
			var values []string
			for _, token := range tokens {
				for _, trivia := range token.Leading {
					values = append(values, "comment:"+strings.TrimRight(trivia.Text, " \t\r"))
				}
				if token.Kind == lexer.String {
					values = append(values, "string:"+token.Text)
				}
			}
			return values
		}
		if !slices.Equal(preserved(input), preserved(formatted)) {
			t.Fatal("formatter changed string or comment contents")
		}
		lexemes := func(source string) []string {
			tokens, _ := lexer.LexRecover("fuzz.tach", source)
			var values []string
			for i, token := range tokens[:len(tokens)-1] {
				if token.Kind == lexer.Comma && (tokens[i+1].Kind == lexer.RParen || tokens[i+1].Kind == lexer.RBracket || tokens[i+1].Kind == lexer.RBrace) {
					continue
				}
				values = append(values, fmt.Sprintf("%d:%s", token.Kind, token.Text))
			}
			return values
		}
		if !slices.Equal(lexemes(input), lexemes(formatted)) {
			t.Fatal("formatter changed the syntax token stream")
		}
	})
}

func TestMaintainedProjectsCompile(t *testing.T) {
	for _, path := range []string{filepath.Join("..", "..", "examples"), filepath.Join("..", "..", "showcase-ts")} {
		t.Run(filepath.Base(path), func(t *testing.T) {
			if _, err := Check(path, 0); err != nil {
				t.Fatal(err)
			}
			project, err := loadProject(path, 0)
			if err != nil {
				t.Fatal(err)
			}
			for _, kernel := range project.Kernels {
				formatted, err := formatSource(kernel.Identity+".tach", kernel.Source)
				if err != nil {
					t.Fatal(err)
				}
				if formatted != kernel.Source {
					t.Errorf("%s is not canonically formatted; run tach fmt", kernel.Identity)
				}
			}
		})
	}
}

func TestDocumentedTachSnippetsCompile(t *testing.T) {
	for _, name := range []string{"README.md", "docs/language.md", "docs/ir.md", "docs/abi.md", "tach-ts/README.md"} {
		data, err := os.ReadFile(filepath.Join("..", "..", filepath.FromSlash(name)))
		if err != nil {
			t.Fatal(err)
		}
		parts := strings.Split(strings.ReplaceAll(string(data), "\r\n", "\n"), "```tach\n")
		if len(parts) == 1 {
			t.Fatalf("%s contains no Tach fences", name)
		}
		for index, part := range parts[1:] {
			source, _, found := strings.Cut(part, "```")
			if !found {
				t.Fatalf("%s Tach fence %d is unterminated", name, index+1)
			}
			files := map[string]string{"module/snippet.tach": source}
			if strings.Contains(source, `import "data/particles"`) {
				files["data/particles.tach"] = ""
			}
			root := projectFixture(t, files)
			if _, err := Check(root, 1); err != nil {
				t.Errorf("%s Tach fence %d: %v", name, index+1, err)
			}
		}
	}
}
