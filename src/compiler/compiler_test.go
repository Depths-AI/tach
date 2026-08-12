package compiler

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

const baselineSource = `export function scale[i](values: buffer<float32[]>, factor: float32) {
  if (i < values.length) { values[i] *= factor; }
}`

func TestBuildTargetsWriteExactArtifacts(t *testing.T) {
	tests := []struct {
		target BuildTarget
		files  string
	}{
		{TargetAll, "module.d.ts,module.js,module.spv,module.spvasm,module.tach.json,module.tir,module.wgsl"},
		{TargetWeb, "module.d.ts,module.js,module.wgsl"},
		{TargetSPIRV, "module.spv,module.tach.json"},
	}
	for _, test := range tests {
		directory := t.TempDir()
		result, err := CompileTarget("module.tach", baselineSource, test.target)
		if err != nil {
			t.Fatal(err)
		}
		if _, err := WriteDirectory(result, directory, "module"); err != nil {
			t.Fatal(err)
		}
		entries, err := os.ReadDir(directory)
		if err != nil {
			t.Fatal(err)
		}
		names := make([]string, len(entries))
		for i, entry := range entries {
			names[i] = entry.Name()
		}
		if got := strings.Join(names, ","); got != test.files {
			t.Fatalf("%s files = %s, want %s", test.target, got, test.files)
		}
	}
}

func TestUnknownBuildTargetIsRejected(t *testing.T) {
	if _, err := CompileTarget("module.tach", baselineSource, BuildTarget("metal")); err == nil {
		t.Fatal("accepted unknown target")
	}
}

func TestBaselineSugarProducesPublicProgramAndPrivateEntry(t *testing.T) {
	result, err := Compile("scale.tach", baselineSource)
	if err != nil {
		t.Fatal(err)
	}
	for name, text := range map[string]string{
		"IR": result.IR, "WGSL": result.WGSL, "SPIR-V": result.SPIRVAsm,
		"JavaScript": result.JavaScript, "TypeScript": result.TypeScript, "metadata": string(result.Metadata),
	} {
		if text == "" {
			t.Fatalf("empty %s", name)
		}
	}
	if !strings.Contains(result.IR, "program @scale") || !strings.Contains(result.IR, "workgroup(auto)") || !strings.Contains(result.WGSL, "fn _tach_k0") {
		t.Fatalf("unexpected compilation:\n%s\n%s", result.IR, result.WGSL)
	}
	if strings.Contains(result.WGSL, "fn scale") || !strings.Contains(result.TypeScript, "LaunchOptions<number>") {
		t.Fatalf("public/physical ABI mismatch:\n%s\n%s", result.WGSL, result.TypeScript)
	}
}

func TestFusionCorpusPlans(t *testing.T) {
	result, err := CompileFile(filepath.Join("..", "..", "examples", "fusion.tach"))
	if err != nil {
		t.Fatal(err)
	}
	var metadata struct {
		Schema  int `json:"schema"`
		Targets struct {
			Web struct {
				Programs []struct {
					Transients []any `json:"transients"`
					Steps      []struct {
						Kind string `json:"kind"`
					} `json:"steps"`
				} `json:"programs"`
			} `json:"web"`
			SPIRV struct {
				Programs []struct {
					Steps []struct {
						Kind string `json:"kind"`
					} `json:"steps"`
				} `json:"programs"`
			} `json:"spirv"`
		} `json:"targets"`
	}
	if err := json.Unmarshal(result.Metadata, &metadata); err != nil {
		t.Fatal(err)
	}
	if metadata.Schema != 1 || len(metadata.Targets.Web.Programs) != 2 {
		t.Fatalf("metadata = %s", result.Metadata)
	}
	if got := len(metadata.Targets.Web.Programs[0].Steps); got != 1 || len(metadata.Targets.Web.Programs[0].Transients) != 0 {
		t.Fatalf("fused plan = %#v", metadata.Targets.Web.Programs[0])
	}
	if got := len(metadata.Targets.Web.Programs[1].Steps); got != 2 {
		t.Fatalf("web fallback steps = %d", got)
	}
	steps := metadata.Targets.SPIRV.Programs[1].Steps
	if len(steps) != 3 || steps[1].Kind != "barrier" {
		t.Fatalf("SPIR-V fallback steps = %#v", steps)
	}
}

func TestCompilationIsDeterministic(t *testing.T) {
	a, err := Compile("scale.tach", baselineSource)
	if err != nil {
		t.Fatal(err)
	}
	b, err := Compile("scale.tach", baselineSource)
	if err != nil {
		t.Fatal(err)
	}
	if a.IR != b.IR || a.WGSL != b.WGSL || a.SPIRVAsm != b.SPIRVAsm || a.JavaScript != b.JavaScript || a.TypeScript != b.TypeScript || !bytes.Equal(a.SPIRV, b.SPIRV) || !bytes.Equal(a.Metadata, b.Metadata) {
		t.Fatal("identical compilations differ")
	}
}

func TestMaintainedExamplesCompile(t *testing.T) {
	paths, err := filepath.Glob(filepath.Join("..", "..", "examples", "*.tach"))
	if err != nil {
		t.Fatal(err)
	}
	for _, path := range paths {
		t.Run(filepath.Base(path), func(t *testing.T) {
			if _, err := CompileFile(path); err != nil {
				t.Fatal(err)
			}
		})
	}
}
