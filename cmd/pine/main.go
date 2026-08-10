package main

import (
	"flag"
	"fmt"
	"os"
	"path/filepath"

	"pine/internal/compiler"
	"pine/internal/spirv"
)

const usageText = `Pine — lean typed GPGPU compiler

Usage:
  pine build [-o DIR] [-name BASE] FILE.pine
  pine check FILE.pine
  pine ir FILE.pine
  pine wgsl FILE.pine
  pine spirv-dis FILE.pine

Commands:
  build      compile and write .pir, .wgsl, .spv, .spvasm, .js, .d.ts, metadata
  check      run Pine's complete owned compilation and validation pipeline
  ir         print structured SSA-ish Pine IR
  wgsl       print generated WGSL
  spirv-dis  print Pine's disassembly of generated SPIR-V
`

func main() {
	if len(os.Args) < 2 {
		fmt.Fprint(os.Stderr, usageText)
		os.Exit(2)
	}
	var err error
	switch os.Args[1] {
	case "build":
		err = build(os.Args[2:])
	case "check":
		err = oneFile("check", os.Args[2:], func(r *compiler.Result) error {
			summary, e := spirv.Summary(r.SPIRV)
			if e != nil {
				return e
			}
			fmt.Printf("ok: %s\n", r.SourceName)
			fmt.Printf("  Pine IR: verified\n")
			fmt.Printf("  WGSL: %d bytes, validated\n", len(r.WGSL))
			fmt.Printf("  SPIR-V: %d bytes, %s\n", len(r.SPIRV), summary)
			fmt.Printf("  bindings: %d JS bytes, %d declaration bytes, validated\n", len(r.JavaScript), len(r.TypeScript))
			return nil
		})
	case "ir":
		err = oneFile("ir", os.Args[2:], func(r *compiler.Result) error { fmt.Print(r.IR); return nil })
	case "wgsl":
		err = oneFile("wgsl", os.Args[2:], func(r *compiler.Result) error { fmt.Print(r.WGSL); return nil })
	case "spirv-dis":
		err = oneFile("spirv-dis", os.Args[2:], func(r *compiler.Result) error { fmt.Print(r.SPIRVAsm); return nil })
	case "help", "-h", "--help":
		fmt.Print(usageText)
		return
	default:
		fmt.Fprintf(os.Stderr, "unknown command %q\n\n%s", os.Args[1], usageText)
		os.Exit(2)
	}
	if err != nil {
		fmt.Fprintln(os.Stderr, "pine:", err)
		os.Exit(1)
	}
}

func build(args []string) error {
	fs := flag.NewFlagSet("build", flag.ContinueOnError)
	fs.SetOutput(os.Stderr)
	out := fs.String("o", "build", "output directory")
	name := fs.String("name", "", "artifact base name")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if fs.NArg() != 1 {
		return fmt.Errorf("build expects exactly one .pine file")
	}
	path := fs.Arg(0)
	r, err := compiler.CompileFile(path)
	if err != nil {
		return err
	}
	base := *name
	if base == "" {
		base = filepath.Base(path)
		base = base[:len(base)-len(filepath.Ext(base))]
	}
	paths, err := compiler.WriteDirectory(r, *out, base)
	if err != nil {
		return err
	}
	fmt.Printf("built %s\n", path)
	for _, p := range paths {
		fmt.Printf("  %s\n", p)
	}
	return nil
}

func oneFile(command string, args []string, fn func(*compiler.Result) error) error {
	fs := flag.NewFlagSet(command, flag.ContinueOnError)
	fs.SetOutput(os.Stderr)
	if err := fs.Parse(args); err != nil {
		return err
	}
	if fs.NArg() != 1 {
		return fmt.Errorf("%s expects exactly one .pine file", command)
	}
	r, err := compiler.CompileFile(fs.Arg(0))
	if err != nil {
		return err
	}
	return fn(r)
}
