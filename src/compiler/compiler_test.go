package compiler

import (
	"bytes"
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
		{TargetAll, "module.d.ts,module.js,module.spv,module.spvasm,module.tir,module.wgsl"},
		{TargetWeb, "module.d.ts,module.js,module.wgsl"},
		{TargetSPIRV, "module.spv"},
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
