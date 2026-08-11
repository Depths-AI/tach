package main

import (
	"flag"
	"fmt"
	"os"
	"path/filepath"

	"tach/src/compiler"
	"tach/src/spirv"
)

const usageText = `Tach — lean typed GPGPU compiler

Usage:
  tach build [--target web|spirv|all] FILE.tach
  tach check [--target web|spirv|all] FILE.tach
  tach ir FILE.tach
  tach wgsl FILE.tach
  tach spirv-dis FILE.tach
  tach version

Commands:
  build      compile for WebGPU by default; use --target for SPIR-V or diagnostics
  check      validate the WebGPU pipeline by default; use --target for SPIR-V or all
  ir         print structured SSA-ish Tach IR
  wgsl       print generated WGSL
  spirv-dis  print Tach's disassembly of generated SPIR-V
  version    print the Tach CLI version
`

var version = "dev"

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
		err = check(os.Args[2:])
	case "ir":
		err = oneFile("ir", os.Args[2:], func(r *compiler.Result) error { fmt.Print(r.IR); return nil })
	case "wgsl":
		err = oneFile("wgsl", os.Args[2:], func(r *compiler.Result) error { fmt.Print(r.WGSL); return nil })
	case "spirv-dis":
		err = oneFile("spirv-dis", os.Args[2:], func(r *compiler.Result) error { fmt.Print(r.SPIRVAsm); return nil })
	case "version", "-v", "--version":
		fmt.Printf("tach %s\n", version)
		return
	case "help", "-h", "--help":
		fmt.Print(usageText)
		return
	default:
		fmt.Fprintf(os.Stderr, "unknown command %q\n\n%s", os.Args[1], usageText)
		os.Exit(2)
	}
	if err != nil {
		fmt.Fprintln(os.Stderr, "tach:", err)
		os.Exit(1)
	}
}

func build(args []string) error {
	target, path, err := targetFile("build", args)
	if err != nil {
		return err
	}
	r, err := compiler.CompileFileTarget(path, target)
	if err != nil {
		return err
	}
	base := filepath.Base(path)
	base = base[:len(base)-len(filepath.Ext(base))]
	paths, err := compiler.WriteDirectory(r, "build", base)
	if err != nil {
		return err
	}
	fmt.Printf("built %s\n", path)
	for _, p := range paths {
		fmt.Printf("  %s\n", p)
	}
	return nil
}

func check(args []string) error {
	target, path, err := targetFile("check", args)
	if err != nil {
		return err
	}
	r, err := compiler.CompileFileTarget(path, target)
	if err != nil {
		return err
	}
	fmt.Printf("ok: %s\n", r.SourceName)
	if target == compiler.TargetWeb || target == compiler.TargetAll {
		fmt.Printf("  WGSL: %d bytes, validated\n", len(r.WGSL))
		fmt.Printf("  bindings: %d JS bytes, %d declaration bytes, validated\n", len(r.JavaScript), len(r.TypeScript))
	}
	if target == compiler.TargetSPIRV || target == compiler.TargetAll {
		summary, err := spirv.Summary(r.SPIRV)
		if err != nil {
			return err
		}
		fmt.Printf("  SPIR-V: %d bytes, %s\n", len(r.SPIRV), summary)
	}
	if target == compiler.TargetAll {
		fmt.Printf("  diagnostics: Tach IR and SPIR-V disassembly generated\n")
	}
	return nil
}

func targetFile(command string, args []string) (compiler.BuildTarget, string, error) {
	flags := flag.NewFlagSet(command, flag.ContinueOnError)
	targetName := flags.String("target", "web", "target: web, spirv, or all")
	if err := flags.Parse(args); err != nil {
		return "", "", err
	}
	if flags.NArg() != 1 {
		return "", "", fmt.Errorf("%s expects exactly one .tach file", command)
	}
	target, err := compiler.ParseBuildTarget(*targetName)
	return target, flags.Arg(0), err
}

func oneFile(command string, args []string, fn func(*compiler.Result) error) error {
	if len(args) != 1 {
		return fmt.Errorf("%s expects exactly one .tach file", command)
	}
	r, err := compiler.CompileFile(args[0])
	if err != nil {
		return err
	}
	return fn(r)
}
