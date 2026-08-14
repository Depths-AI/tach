package compiler

import (
	"fmt"

	"tach/src/ast"
	"tach/src/backend"
	"tach/src/bindings"
	"tach/src/flow"
	"tach/src/opt"
	"tach/src/sema"
	"tach/src/spirv"
	"tach/src/wgsl"
)

type BuildTarget string

const (
	TargetWeb   BuildTarget = "web"
	TargetSPIRV BuildTarget = "spirv"
)

func ParseBuildTarget(name string) (BuildTarget, error) {
	target := BuildTarget(name)
	if target != TargetWeb && target != TargetSPIRV {
		return "", fmt.Errorf("unknown build target %q (want web or spirv)", name)
	}
	return target, nil
}

type Result struct {
	Module      *flow.Module
	WGSL        string
	SPIRV       []byte
	JavaScript  string
	TypeScript  string
	Metadata    []byte
	Description []byte
}

func Build(cwd string, target BuildTarget, workers int) (*Result, error) {
	if _, err := ParseBuildTarget(string(target)); err != nil {
		return nil, err
	}
	return compile(cwd, target == TargetWeb, target == TargetSPIRV, true, workers)
}

func Check(cwd string, workers int) (*Result, error) {
	return compile(cwd, true, true, true, workers)
}

func Describe(cwd string, workers int) (*Result, error) {
	return compile(cwd, false, false, false, workers)
}

func compile(cwd string, web, spv, optimize bool, workers int) (*Result, error) {
	project, err := loadProject(cwd, workers)
	if err != nil {
		return nil, err
	}
	modules := make([]*ast.Module, len(project.Kernels))
	for i := range project.Kernels {
		modules[i] = project.Kernels[i].AST
	}
	logical, documentation, err := sema.CheckAndLowerProject(modules, workers)
	if err != nil {
		return nil, project.semanticError(err)
	}
	for i := range project.Kernels {
		project.Kernels[i].Documentation = documentation[i]
	}
	if optimize {
		if err := opt.OptimizeLogical(logical); err != nil {
			return nil, fmt.Errorf("IR optimization: %w", err)
		}
	}
	result := &Result{Module: logical}
	var webExecutable, spirvExecutable *backend.Executable
	if web {
		webExecutable, err = wgsl.Lower(logical)
		if err == nil {
			result.WGSL, err = wgsl.Emit(webExecutable)
		}
		if err != nil {
			return nil, fmt.Errorf("web backend: %w", err)
		}
	}
	if spv {
		spirvExecutable, err = spirv.Lower(logical)
		if err == nil {
			result.SPIRV, err = spirv.Emit(spirvExecutable)
		}
		if err != nil {
			return nil, fmt.Errorf("SPIR-V backend: %w", err)
		}
	}
	if web || spv {
		generated, err := bindings.Generate(logical, webExecutable, spirvExecutable, result.WGSL)
		if err != nil {
			return nil, fmt.Errorf("bindings: %w", err)
		}
		result.JavaScript, result.TypeScript, result.Metadata = generated.JavaScript, generated.Declarations, generated.MetadataJSON
	}
	result.Description, err = bindings.DescribeProject(logical, project.description())
	if err != nil {
		return nil, fmt.Errorf("description: %w", err)
	}
	return result, nil
}
