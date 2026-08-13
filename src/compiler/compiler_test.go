package compiler

import (
	"bytes"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
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

func TestDescribeEmitsTargetNeutralDocumentation(t *testing.T) {
	description, err := Describe("scale.tach", `
@docs(title("Scale"), summary("Scaling example."));
@docs(summary("Scales values."), coordinate(i, "Value index."), param(values, "Values."), param(factor, "Multiplier."))
export function scale[i](values: buffer<float32[]>, factor: float32) { if (i < values.length) { values[i] *= factor; } }
`)
	if err != nil {
		t.Fatal(err)
	}
	var value map[string]any
	if err := json.Unmarshal(description, &value); err != nil {
		t.Fatal(err)
	}
	text := string(description)
	for _, want := range []string{`"schema": 1`, `"title": "Scale"`, `"access": "readWrite"`, `"kind": "runtimeArray"`} {
		if !strings.Contains(text, want) {
			t.Fatalf("description missing %s:\n%s", want, text)
		}
	}
	if strings.Contains(text, "TypeScript") || strings.Contains(text, "ComputeBuffer") {
		t.Fatalf("target syntax leaked into description:\n%s", text)
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

func TestMaintainedDocumentationExamplesCompile(t *testing.T) {
	fence := regexp.MustCompile("(?s)```tach\\r?\\n(.*?)\\r?\\n```")
	paths := []string{
		filepath.Join("..", "..", "README.md"),
		filepath.Join("..", "..", "tach-ts", "README.md"),
		filepath.Join("..", "..", "docs", "language.md"),
		filepath.Join("..", "..", "docs", "abi.md"),
		filepath.Join("..", "..", "docs", "architecture.md"),
		filepath.Join("..", "..", "docs", "ir.md"),
	}
	total := 0
	for _, path := range paths {
		data, err := os.ReadFile(path)
		if err != nil {
			t.Fatal(err)
		}
		for index, match := range fence.FindAllSubmatch(data, -1) {
			name := fmt.Sprintf("%s#%d", filepath.ToSlash(path), index+1)
			t.Run(name, func(t *testing.T) {
				if _, err := Compile(name, string(match[1])); err != nil {
					t.Fatal(err)
				}
			})
			total++
		}
	}
	if total == 0 {
		t.Fatal("maintained documentation contains no Tach examples")
	}
}
