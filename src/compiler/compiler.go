package compiler

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"tach/src/backend"
	"tach/src/bindings"
	"tach/src/flow"
	"tach/src/ir"
	"tach/src/opt"
	"tach/src/parser"
	"tach/src/sema"
	"tach/src/spirv"
	"tach/src/wgsl"
)

type BuildTarget string

const (
	TargetWeb   BuildTarget = "web"
	TargetSPIRV BuildTarget = "spirv"
	TargetAll   BuildTarget = "all"
)

func ParseBuildTarget(name string) (BuildTarget, error) {
	target := BuildTarget(name)
	switch target {
	case TargetWeb, TargetSPIRV, TargetAll:
		return target, nil
	default:
		return "", fmt.Errorf("unknown build target %q (want web, spirv, or all)", name)
	}
}

// Result contains the validated artifacts requested from one Tach compilation.
type Result struct {
	SourceName string
	IR         string
	WGSL       string
	SPIRV      []byte
	SPIRVAsm   string
	JavaScript string
	TypeScript string
	Metadata   []byte
	target     BuildTarget
}

// Compile runs the complete Tach pipeline: parsing, semantic analysis,
// structured SSA-ish IR verification, WGSL generation+validation, SPIR-V
// generation+binary validation, SPIR-V disassembly, and binding generation.
func Compile(sourceName, source string) (*Result, error) {
	return CompileTarget(sourceName, source, TargetAll)
}

// CompileTarget runs only the backends and generators required by target.
func CompileTarget(sourceName, source string, target BuildTarget) (*Result, error) {
	if _, err := ParseBuildTarget(string(target)); err != nil {
		return nil, err
	}
	astModule, err := parser.Parse(sourceName, source)
	if err != nil {
		return nil, err
	}
	module, err := sema.CheckAndLower(astModule)
	if err != nil {
		return nil, err
	}
	if err := opt.OptimizeLogical(module); err != nil {
		return nil, fmt.Errorf("IR optimization: %w", err)
	}

	result := &Result{SourceName: sourceName, target: target}
	var webExecutable, spirvExecutable *backend.Executable
	if target == TargetWeb || target == TargetAll {
		webExecutable, err = wgsl.Lower(module)
		if err != nil {
			return nil, fmt.Errorf("Web executable planning: %w", err)
		}
		result.WGSL, err = wgsl.Emit(webExecutable)
		if err != nil {
			return nil, fmt.Errorf("WGSL backend: %w", err)
		}
	}
	if target == TargetSPIRV || target == TargetAll {
		spirvExecutable, err = spirv.Lower(module)
		if err != nil {
			return nil, fmt.Errorf("SPIR-V executable planning: %w", err)
		}
		result.SPIRV, err = spirv.Emit(spirvExecutable)
		if err != nil {
			return nil, fmt.Errorf("SPIR-V backend: %w", err)
		}
	}
	if target == TargetAll {
		result.IR = "=== optimized logical program ===\n" + flow.Dump(module) + "=== kernel templates ===\n" + ir.Dump(module.Kernel) + "=== web executable ===\n" + backend.Dump(webExecutable) + "=== spirv executable ===\n" + backend.Dump(spirvExecutable)
		result.SPIRVAsm, err = spirv.Disassemble(result.SPIRV)
		if err != nil {
			return nil, fmt.Errorf("SPIR-V disassembly: %w", err)
		}
	}
	generated, err := bindings.Generate(module, webExecutable, spirvExecutable, result.WGSL)
	if err != nil {
		return nil, fmt.Errorf("bindings: %w", err)
	}
	result.JavaScript, result.TypeScript, result.Metadata = generated.JavaScript, generated.Declarations, generated.MetadataJSON
	return result, nil
}

func CompileFile(path string) (*Result, error) {
	return CompileFileTarget(path, TargetAll)
}

func CompileFileTarget(path string, target BuildTarget) (*Result, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	return CompileTarget(path, string(data), target)
}

// WriteDirectory writes exactly the requested, already-validated artifact set.
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
	if _, err := ParseBuildTarget(string(result.target)); err != nil {
		return nil, fmt.Errorf("compilation result has invalid build target %q", result.target)
	}
	artifacts := []struct {
		name   string
		data   []byte
		target BuildTarget
	}{
		{base + ".tir", []byte(result.IR), TargetAll},
		{base + ".wgsl", []byte(result.WGSL), TargetWeb},
		{base + ".spv", result.SPIRV, TargetSPIRV},
		{base + ".spvasm", []byte(result.SPIRVAsm), TargetAll},
		{base + ".js", []byte(result.JavaScript), TargetWeb},
		{base + ".d.ts", []byte(result.TypeScript), TargetWeb},
	}
	paths := make([]string, 0, len(artifacts))
	for _, a := range artifacts {
		if result.target != TargetAll && result.target != a.target {
			continue
		}
		path := filepath.Join(dir, a.name)
		if err := os.WriteFile(path, a.data, 0o644); err != nil {
			return nil, fmt.Errorf("write %s: %w", path, err)
		}
		paths = append(paths, path)
	}
	for _, a := range artifacts {
		if result.target == TargetAll || result.target == a.target {
			continue
		}
		path := filepath.Join(dir, a.name)
		if err := os.Remove(path); err != nil && !errors.Is(err, os.ErrNotExist) {
			return nil, fmt.Errorf("remove stale %s: %w", path, err)
		}
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
