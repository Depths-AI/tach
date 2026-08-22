package driver

import (
	"bytes"
	"fmt"
	"runtime"
	"strings"
	"testing"
)

func TestCompileTimeConstantsRespectImportsAndNeverReachHostArtifacts(t *testing.T) {
	root := projectFixture(t, map[string]string{
		"library/constants.tach": `const tileWidth: uint32 = 8;`,
		"app/main.tach": `import "library/constants";
@workgroup(tileWidth)
export function tiled[i](out: buffer<uint32[]>) {
  let scratch: shared<uint32[tileWidth * tileWidth]>;
  let lane = i % tileWidth;
  scratch[lane] = i;
  workgroupBarrier();
  if (i < out.length) { out[i] = scratch[lane]; }
}`,
	})
	result, err := Check(root, 2)
	if err != nil {
		t.Fatal(err)
	}
	if len(result.Diagnostics) != 0 {
		t.Fatalf("constant import warnings = %#v", result.Diagnostics)
	}
	for name, artifact := range map[string][]byte{
		"WGSL":        []byte(result.WGSL),
		"metadata":    result.MetadataJSON,
		"description": result.Description,
	} {
		if bytes.Contains(artifact, []byte("tileWidth")) {
			t.Errorf("%s leaked Tach-only constant name: %s", name, artifact)
		}
	}

	root = projectFixture(t, map[string]string{
		"library/constants.tach": `const privateWidth = 8;`,
		"app/main.tach":          `export function hidden[i](out: buffer<uint32[]>) { out[i] = privateWidth; }`,
	})
	if _, err := Check(root, 2); err == nil || !strings.Contains(err.Error(), `unknown identifier "privateWidth"`) {
		t.Fatalf("missing-import constant error = %v", err)
	}

	root = projectFixture(t, map[string]string{
		"app/main.tach": `const Value = 1; type Value = { item: uint32 };`,
	})
	if _, err := Check(root, 1); err == nil || !strings.Contains(err.Error(), `project declaration "Value" is already defined`) {
		t.Fatalf("constant declaration collision = %v", err)
	}

	root = projectFixture(t, map[string]string{
		"library/constants.tach": `const localWidth = 8;`,
		"app/main.tach":          `export function local[i](out: buffer<uint32[]>) { const localWidth = 4; if (i < out.length) { out[i] = localWidth; } }`,
	})
	result, err = Check(root, 1)
	if err != nil {
		t.Fatal(err)
	}
	found := false
	for _, diagnostic := range result.Diagnostics {
		found = found || diagnostic.Kind == "unused-constant"
	}
	if !found {
		t.Fatalf("unimported constant was incorrectly marked reachable: %#v", result.Diagnostics)
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
