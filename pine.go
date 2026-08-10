// Package pine exposes the embeddable Pine compiler.
package pine

import "pine/internal/compiler"

// Result is a complete compilation whose artifacts have passed Pine's owned
// semantic, IR, WGSL-subset, SPIR-V-binary, and binding-contract validators.
type Result = compiler.Result

func Compile(sourceName, source string) (*Result, error) {
	return compiler.Compile(sourceName, source)
}

func CompileFile(path string) (*Result, error) {
	return compiler.CompileFile(path)
}

func WriteDirectory(result *Result, dir, base string) ([]string, error) {
	return compiler.WriteDirectory(result, dir, base)
}
