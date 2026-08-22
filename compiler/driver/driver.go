package driver

import (
	"bytes"
	"compress/gzip"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"

	"tach/foundation"
	"tach/ir"
	"tach/parser"
	"tach/semantics"
	"tach/spirv"
	"tach/wgsl"
)

type Result struct {
	Module         *ir.Module
	Web            *wgsl.Result
	Native         *spirv.Result
	WGSL           string
	CompressedWGSL []byte
	SPIRV          []byte
	Metadata       *Metadata
	MetadataJSON   []byte
	Description    []byte
	Diagnostics    foundation.Diagnostics
}

func WriteNativeArtifacts(result *Result, output string, verbose bool) error {
	if result == nil || result.Module == nil || result.Web == nil || result.Native == nil || result.Metadata == nil {
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
			"diagnostics/spirv.kernel.ir": result.Native.KernelModule,
			"diagnostics/web.plan.json":   json.RawMessage(result.Web.RuntimeJSON),
			"diagnostics/spirv.plan.json": json.RawMessage(result.Native.RuntimeJSON),
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
	modules := make([]*parser.File, len(project.Kernels))
	for i := range project.Kernels {
		modules[i] = project.Kernels[i].Syntax
	}
	var analyzed *semantics.Result
	if build {
		analyzed, err = semantics.Build(modules, workers)
	} else {
		analyzed, err = semantics.Describe(modules, workers)
	}
	if err != nil {
		return nil, project.semanticError(err)
	}
	for i := range project.Kernels {
		project.Kernels[i].Documentation = analyzed.Documentation[i]
	}
	logical := analyzed.Module
	result := &Result{Module: logical, Diagnostics: warnings(project, logical)}
	if build {
		result.Web, err = wgsl.Lower(logical)
		if err != nil {
			return nil, fmt.Errorf("web backend: %w", err)
		}
		result.WGSL = result.Web.Source
		result.CompressedWGSL, err = compressWGSL(result.WGSL)
		if err != nil {
			return nil, fmt.Errorf("WGSL compression: %w", err)
		}
		result.Native, err = spirv.Lower(logical)
		if err != nil {
			return nil, fmt.Errorf("SPIR-V backend: %w", err)
		}
		result.SPIRV = result.Native.Binary
		result.Metadata, result.MetadataJSON, err = generateMetadata(logical, result.Web.RuntimeJSON, result.Native.RuntimeJSON)
		if err != nil {
			return nil, fmt.Errorf("runtime metadata: %w", err)
		}
	}
	result.Description, err = DescribeProject(logical, project.description())
	if err != nil {
		return nil, fmt.Errorf("description: %w", err)
	}
	return result, nil
}
