package compiler

import (
	"bytes"
	"compress/gzip"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"

	"tach/src/ast"
	"tach/src/backend"
	"tach/src/bindings"
	"tach/src/flow"
	"tach/src/foundation"
	"tach/src/opt"
	"tach/src/sema"
	"tach/src/spirv"
	"tach/src/wgsl"
)

type Result struct {
	Module          *flow.Module
	Web             *backend.Executable
	SPIRVExecutable *backend.Executable
	WGSL            string
	CompressedWGSL  []byte
	SPIRV           []byte
	Metadata        *bindings.Metadata
	MetadataJSON    []byte
	Description     []byte
	Diagnostics     foundation.Diagnostics
}

func WriteNativeArtifacts(result *Result, output string, verbose bool) error {
	if result == nil || result.Module == nil || result.Web == nil || result.SPIRVExecutable == nil || result.Metadata == nil {
		return fmt.Errorf("nil project result")
	}
	root, err := filepath.Abs(output)
	if err != nil {
		return err
	}
	info, err := os.Lstat(root)
	if err != nil || !info.IsDir() || info.Mode()&os.ModeSymlink != 0 {
		return fmt.Errorf("output must be an existing non-symlink directory: %s", root)
	}
	files := map[string][]byte{
		"kernel.wgsl.gz": result.CompressedWGSL,
		"kernel.spv":     result.SPIRV,
		"project.json":   result.Description,
		"runtime.json":   result.MetadataJSON,
	}
	if verbose {
		for name, value := range map[string]any{
			"diagnostics/flow.ir":         result.Module,
			"diagnostics/kernel.ir":       result.Module.Kernel,
			"diagnostics/web.kernel.ir":   result.Web.KernelModule,
			"diagnostics/spirv.kernel.ir": result.SPIRVExecutable.KernelModule,
			"diagnostics/web.plan.json":   result.Metadata.Targets.Web,
			"diagnostics/spirv.plan.json": result.Metadata.Targets.SPIRV,
		} {
			files[name], err = diagnosticJSON(value)
			if err != nil {
				return fmt.Errorf("%s diagnostic: %w", name, err)
			}
		}
		disassembly, err := spirv.Disassemble(result.SPIRV)
		if err != nil {
			return fmt.Errorf("SPIR-V disassembly: %w", err)
		}
		files["diagnostics/kernel.spvasm"] = []byte(disassembly)
	}
	entries, err := os.ReadDir(root)
	if err != nil {
		return err
	}
	if len(entries) != 0 {
		return fmt.Errorf("output staging directory must be empty")
	}
	for name, data := range files {
		path := filepath.Join(root, filepath.FromSlash(name))
		if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
			return err
		}
		if err := os.WriteFile(path, data, 0o644); err != nil {
			return err
		}
	}
	return nil
}

func diagnosticJSON(value any) ([]byte, error) {
	data, err := json.MarshalIndent(value, "", "  ")
	return append(data, '\n'), err
}

func compressWGSL(source string) ([]byte, error) {
	var output bytes.Buffer
	compressed := gzip.NewWriter(&output)
	if _, err := io.WriteString(compressed, source); err != nil {
		return nil, err
	}
	if err := compressed.Close(); err != nil {
		return nil, err
	}
	return output.Bytes(), nil
}

func Build(cwd string, workers int) (*Result, error) {
	return compile(cwd, true, workers)
}

func Check(cwd string, workers int) (*Result, error) {
	return compile(cwd, true, workers)
}

func Describe(cwd string, workers int) (*Result, error) {
	return compile(cwd, false, workers)
}

func compile(cwd string, build bool, workers int) (*Result, error) {
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
	if build {
		if err := opt.OptimizeLogical(logical); err != nil {
			return nil, fmt.Errorf("IR optimization: %w", err)
		}
	}
	result := &Result{Module: logical, Diagnostics: warnings(project, logical)}
	if build {
		result.Web, err = wgsl.Lower(logical)
		if err == nil {
			result.WGSL, err = wgsl.Emit(result.Web)
		}
		if err != nil {
			return nil, fmt.Errorf("web backend: %w", err)
		}
		result.CompressedWGSL, err = compressWGSL(result.WGSL)
		if err != nil {
			return nil, fmt.Errorf("WGSL compression: %w", err)
		}
		result.SPIRVExecutable, err = spirv.Lower(logical)
		if err == nil {
			result.SPIRV, err = spirv.Emit(result.SPIRVExecutable)
		}
		if err != nil {
			return nil, fmt.Errorf("SPIR-V backend: %w", err)
		}
		generated, err := bindings.Generate(logical, result.Web, result.SPIRVExecutable)
		if err != nil {
			return nil, fmt.Errorf("bindings: %w", err)
		}
		result.Metadata, result.MetadataJSON = generated.Metadata, generated.MetadataJSON
	}
	result.Description, err = bindings.DescribeProject(logical, project.description())
	if err != nil {
		return nil, fmt.Errorf("description: %w", err)
	}
	return result, nil
}
