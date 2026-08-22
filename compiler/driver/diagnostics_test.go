package driver

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"runtime"
	"tach/foundation"
	"testing"
)

func TestNegativeExamplesAttestStructuredErrorsAndWarnings(t *testing.T) {
	type expectation struct {
		fails bool
		codes []string
	}
	cases := map[string]expectation{
		"syntax-errors":        {true, []string{"parser"}},
		"lexer-errors":         {true, []string{"lexer"}},
		"manifest-error":       {true, []string{"manifest"}},
		"name-and-type-errors": {true, []string{"semantic"}},
		"control-flow-errors":  {true, []string{"semantic"}},
		"constant-errors":      {true, []string{"semantic"}},
		"constant-cycle":       {true, []string{"semantic"}},
		"documentation-errors": {true, []string{"semantic"}},
		"divergent-barrier":    {true, []string{"semantic"}},
		"missing-import":       {true, []string{"import"}},
		"name-collision":       {true, []string{"name"}},
		"dead-code":            {false, []string{"discarded-value", "unreachable-function", "unused-binding", "unused-constant"}},
		"launch-and-control":   {false, []string{"constant-condition", "zero-dispatch"}},
		"memory-access":        {false, []string{"constant-write-index", "strided-access"}},
		"no-effect-kernel":     {false, []string{"no-effect-kernel", "unused-binding"}},
	}
	root := filepath.Join("..", "..", "negative-examples")
	entries, err := os.ReadDir(root)
	if err != nil {
		t.Fatal(err)
	}
	found := 0
	for _, entry := range entries {
		expected, ok := cases[entry.Name()]
		if !entry.IsDir() || !ok {
			continue
		}
		found++
		t.Run(entry.Name(), func(t *testing.T) {
			path := filepath.Join(root, entry.Name())
			files, err := filepath.Glob(filepath.Join(path, "*", "*.tach"))
			if err != nil || len(files) != 1 {
				t.Fatalf("one-kernel project invariant: files=%v error=%v", files, err)
			}
			if !expected.fails {
				data, err := os.ReadFile(files[0])
				formatted, formatErr := formatSource(filepath.ToSlash(files[0]), string(data))
				if err != nil || formatErr != nil || formatted != string(data) {
					t.Fatalf("warning example is not canonically formatted: read=%v format=%v", err, formatErr)
				}
			}
			result, one := Check(path, 1)
			parallel, many := Check(path, runtime.GOMAXPROCS(0))
			if (one != nil) != expected.fails || (many != nil) != expected.fails {
				t.Fatalf("fails=%v: one=%v many=%v", expected.fails, one, many)
			}
			var diagnostics foundation.Diagnostics
			if expected.fails {
				var ok bool
				diagnostics, ok = ErrorDiagnostics(one)
				if !ok || one.Error() != many.Error() {
					t.Fatalf("unstructured or nondeterministic error: one=%v many=%v", one, many)
				}
			} else {
				diagnostics = result.Diagnostics
				if got := result.Diagnostics.Error(); got != parallel.Diagnostics.Error() {
					t.Fatalf("warnings differ by worker count:\nONE:\n%s\nMANY:\n%s", got, parallel.Diagnostics)
				}
			}
			codes := map[string]bool{}
			for _, diagnostic := range diagnostics {
				codes[diagnostic.Kind] = true
				severity := "warning"
				if expected.fails {
					severity = "error"
				}
				if diagnostic.Severity != severity || diagnostic.Span.File == "" || diagnostic.Source == "" {
					t.Errorf("incomplete diagnostic: %#v", diagnostic)
				}
			}
			for _, code := range expected.codes {
				if !codes[code] {
					t.Errorf("missing %s in %#v", code, diagnostics)
				}
			}
			encoded, err := json.Marshal(struct {
				Schema      int                    `json:"schema"`
				Diagnostics foundation.Diagnostics `json:"diagnostics"`
			}{1, diagnostics})
			if err != nil || !bytes.Contains(encoded, []byte(`"severity"`)) || !bytes.Contains(encoded, []byte(`"code"`)) || bytes.Contains(encoded, []byte(`"Kind"`)) {
				t.Fatalf("machine diagnostics = %s, %v", encoded, err)
			}
		})
	}
	if found != len(cases) {
		t.Fatalf("found %d negative projects, want %d", found, len(cases))
	}
}

func TestWarningsRespectImportsReachabilityAndControl(t *testing.T) {
	root := projectFixture(t, map[string]string{
		"library/math.tach": `type Factor = { value: float32, }; function twice(value: float32): float32 { return value * 2; }`,
		"app/main.tach":     `import "library/math"; export function apply[i](out: buffer<float32[]>, factor: Factor) { if (i == 0) { out[0] = twice(factor.value); } }`,
	})
	result, err := Check(root, 2)
	if err != nil {
		t.Fatal(err)
	}
	for _, diagnostic := range result.Diagnostics {
		if diagnostic.Kind == "unused-import" || diagnostic.Kind == "unreachable-function" || diagnostic.Kind == "constant-write-index" {
			t.Errorf("false positive: %#v", diagnostic)
		}
	}

	root = projectFixture(t, map[string]string{
		"library/math.tach": `type Factor = { value: float32, };`,
		"app/main.tach":     `import "library/math"; export function apply[i](out: buffer<float32[]>) { if (i < out.length) { out[i] = 1; } }`,
	})
	result, err = Check(root, 2)
	if err != nil {
		t.Fatal(err)
	}
	if len(result.Diagnostics) != 1 || result.Diagnostics[0].Kind != "unused-import" {
		t.Fatalf("unused import diagnostics = %#v", result.Diagnostics)
	}

	root = projectFixture(t, map[string]string{
		"app/main.tach": `@workgroup(4) export function sharedOnly[i](out: buffer<uint32[]>) { let scratch: shared<uint32[8]>; scratch[0] = out.length; scratch[i * 2] = 2; }`,
	})
	result, err = Check(root, 2)
	if err != nil {
		t.Fatal(err)
	}
	if len(result.Diagnostics) != 1 || result.Diagnostics[0].Kind != "no-effect-kernel" {
		t.Fatalf("shared-memory diagnostics = %#v", result.Diagnostics)
	}
}
