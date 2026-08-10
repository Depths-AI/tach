package compiler

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"tach/src/bindings"
	"tach/src/ir"
	"tach/src/opt"
	"tach/src/parser"
	"tach/src/sema"
	"tach/src/spirv"
	"tach/src/wgsl"
)

// Result is a complete Tach compilation. Every artifact in Result has already
// passed the validators owned by the Tach compiler.
type Result struct {
	SourceName string
	Module     *ir.Module
	IR         string
	WGSL       string
	SPIRV      []byte
	SPIRVAsm   string
	JavaScript string
	TypeScript string
	Metadata   []byte
}

// Compile runs the complete Tach pipeline: parsing, semantic analysis,
// structured SSA-ish IR verification, WGSL generation+validation, SPIR-V
// generation+binary validation, SPIR-V disassembly, and binding generation.
func Compile(sourceName, source string) (*Result, error) {
	astModule, err := parser.Parse(sourceName, source)
	if err != nil {
		return nil, err
	}
	module, err := sema.CheckAndLower(astModule)
	if err != nil {
		return nil, err
	}
	if err := opt.Run(module); err != nil {
		return nil, fmt.Errorf("IR optimization: %w", err)
	}
	if err := ir.Verify(module); err != nil {
		return nil, fmt.Errorf("IR verification: %w", err)
	}

	wgslSource, err := wgsl.Emit(module)
	if err != nil {
		return nil, fmt.Errorf("WGSL backend: %w", err)
	}
	// Emit currently validates too. Keeping this call explicit makes the
	// compiler pipeline invariant visible here and catches future emitter
	// refactors that might otherwise weaken it.
	if err := wgsl.Validate(wgslSource); err != nil {
		return nil, fmt.Errorf("WGSL validation: %w", err)
	}

	spv, err := spirv.Emit(module)
	if err != nil {
		return nil, fmt.Errorf("SPIR-V backend: %w", err)
	}
	if err := spirv.Validate(spv); err != nil {
		return nil, fmt.Errorf("SPIR-V validation: %w", err)
	}
	spvAsm, err := spirv.Disassemble(spv)
	if err != nil {
		return nil, fmt.Errorf("SPIR-V disassembly: %w", err)
	}

	generated, err := bindings.Generate(module, wgslSource)
	if err != nil {
		return nil, fmt.Errorf("bindings: %w", err)
	}

	return &Result{
		SourceName: sourceName,
		Module:     module,
		IR:         ir.Dump(module),
		WGSL:       wgslSource,
		SPIRV:      spv,
		SPIRVAsm:   spvAsm,
		JavaScript: generated.JavaScript,
		TypeScript: generated.Declarations,
		Metadata:   generated.MetadataJSON,
	}, nil
}

func CompileFile(path string) (*Result, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	return Compile(path, string(data))
}

// OutputNames returns the stable artifact filenames produced for a source base.
func OutputNames(base string) []string {
	return []string{
		base + ".tir",
		base + ".wgsl",
		base + ".spv",
		base + ".spvasm",
		base + ".js",
		base + ".d.ts",
		base + ".tach.json",
	}
}

// WriteDirectory writes a complete, already-validated compilation to dir.
func WriteDirectory(result *Result, dir, base string) ([]string, error) {
	if result == nil {
		return nil, fmt.Errorf("nil compilation result")
	}
	if base == "" {
		base = sourceBase(result.SourceName)
	}
	if dir == "" {
		dir = "."
	}
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return nil, err
	}

	artifacts := []struct {
		name string
		data []byte
	}{
		{base + ".tir", []byte(result.IR)},
		{base + ".wgsl", []byte(result.WGSL)},
		{base + ".spv", result.SPIRV},
		{base + ".spvasm", []byte(result.SPIRVAsm)},
		{base + ".js", []byte(result.JavaScript)},
		{base + ".d.ts", []byte(result.TypeScript)},
		{base + ".tach.json", result.Metadata},
	}
	paths := make([]string, 0, len(artifacts))
	for _, a := range artifacts {
		path := filepath.Join(dir, a.name)
		if err := os.WriteFile(path, a.data, 0o644); err != nil {
			return nil, fmt.Errorf("write %s: %w", path, err)
		}
		paths = append(paths, path)
	}
	return paths, nil
}

func sourceBase(path string) string {
	base := filepath.Base(path)
	if ext := filepath.Ext(base); ext != "" {
		base = strings.TrimSuffix(base, ext)
	}
	if base == "" || base == "." {
		return "module"
	}
	return base
}
