package semantics

import (
	"errors"
	"fmt"
	"maps"
	"runtime"
	"strings"
	"sync"
	"tach/foundation"
	"tach/ir"
	"tach/parser"
)

type analyzer struct {
	syntax        *parser.File
	kernel        *ir.KernelModule
	module        *ir.Module
	types         map[string]*foundation.Type
	consts        map[string]*constantDef
	funcs         map[string]*funcSig
	owners        map[string]string
	imports       map[string]map[string]bool
	workers       int
	constantStack []string
}

type constantDef struct {
	decl  *parser.ConstDecl
	value *foundation.ConstantValue
	err   error
	state uint8
}

type funcSig struct {
	name     string
	params   []namedType
	ret      *foundation.Type
	decl     *parser.FunctionDecl
	indexed  bool
	exported bool
	view     ir.ViewFormat
}
type namedType struct {
	name   string
	ty     *foundation.Type
	buffer bool
}

type symbol struct {
	ty        *foundation.Type
	value     ir.ValueID
	constant  *foundation.ConstantValue
	mutable   bool
	buffer    int // -1 unless this is a stage buffer place
	workgroup int // -1 unless this is a function workgroup place
}
type env struct{ syms map[string]symbol }

func newEnv() env        { return env{syms: map[string]symbol{}} }
func (e env) clone() env { return env{maps.Clone(e.syms)} }

type idAllocator struct {
	nextValue ir.ValueID
	nextPlace ir.PlaceID
}

type fnBuilder struct {
	fn       *ir.Function
	ids      *idAllocator
	block    *ir.Block
	top      bool
	loop     *loopContext
	comptime bool
}

type loopContext struct {
	names []string
	base  env
	post  parser.Stmt
}

func (b *fnBuilder) value() ir.ValueID {
	b.ids.nextValue++
	return b.ids.nextValue
}
func (b *fnBuilder) place() ir.PlaceID {
	b.ids.nextPlace++
	return b.ids.nextPlace
}
func (b *fnBuilder) emit(i ir.Instr) { b.block.Instrs = append(b.block.Instrs, i) }
func (b *fnBuilder) child(block *ir.Block) *fnBuilder {
	// Structured regions share one allocator. SSA/place IDs are function-global
	// identities even when definitions have region-scoped visibility.
	return &fnBuilder{fn: b.fn, ids: b.ids, block: block, loop: b.loop, comptime: b.comptime}
}

type Result struct {
	Module        *ir.Module
	Documentation []ir.Documentation
}

// Build resolves parsed Tach source into one verified, optimized logical program.
func Build(files []*parser.File, workers int) (*Result, error) {
	module, documentation, err := analyzeProject(files, workers)
	if err != nil {
		return nil, err
	}
	if err := optimize(module); err != nil {
		return nil, fmt.Errorf("IR optimization: %w", err)
	}
	return &Result{Module: module, Documentation: documentation}, nil
}

// Describe performs source semantics without optimization. Documentation
// generation needs the logical program but no backend work.
func Describe(files []*parser.File, workers int) (*Result, error) {
	module, documentation, err := analyzeProject(files, workers)
	if err != nil {
		return nil, err
	}
	return &Result{Module: module, Documentation: documentation}, nil
}

func analyze(file *parser.File) (*ir.Module, error) {
	module, _, err := analyzeProject([]*parser.File{file}, 0)
	return module, err
}

func analyzeProject(files []*parser.File, requestedWorkers int) (*ir.Module, []ir.Documentation, error) {
	workers := runtime.GOMAXPROCS(0)
	if requestedWorkers > 0 && requestedWorkers < workers {
		workers = requestedWorkers
	}
	merged := &parser.File{}
	documentation := ir.Documentation{Types: map[string]ir.TypeDocumentation{}, Functions: map[string]ir.FunctionDocumentation{}}
	documentationFiles := make([]ir.Documentation, 0, len(files))
	var documentationDiagnostics foundation.Diagnostics
	for _, file := range files {
		docs, err := checkDocumentation(file)
		if err != nil {
			documentationDiagnostics = appendError(documentationDiagnostics, err)
		}
		documentationFiles = append(documentationFiles, docs)
		if len(files) == 1 {
			documentation.Title, documentation.Summary = docs.Title, docs.Summary
		}
		for name, item := range docs.Types {
			documentation.Types[name] = item
		}
		for name, item := range docs.Functions {
			documentation.Functions[name] = item
		}
		merged.Decls = append(merged.Decls, file.Decls...)
	}
	kernel := &ir.KernelModule{}
	c := &analyzer{syntax: merged, kernel: kernel, module: &ir.Module{Kernel: kernel, Documentation: documentation}, types: map[string]*foundation.Type{}, consts: map[string]*constantDef{}, funcs: map[string]*funcSig{}, owners: map[string]string{}, imports: map[string]map[string]bool{}, workers: workers}
	for _, syntax := range files {
		file := strings.TrimSuffix(syntax.Path, ".tach")
		visible := map[string]bool{file: true}
		for _, item := range syntax.Imports {
			visible[item.Target] = true
		}
		c.imports[file] = visible
		for _, declaration := range syntax.Decls {
			switch item := declaration.(type) {
			case *parser.TypeDecl:
				c.owners[item.Name] = file
			case *parser.ConstDecl:
				c.owners[item.Name] = file
			case *parser.FunctionDecl:
				c.owners[item.Name] = file
			}
		}
	}
	for _, n := range []string{"void", "bool", "int32", "uint32", "float16", "float32"} {
		c.types[n] = foundation.ParseBuiltin(n)
	}
	var interfaceDiagnostics foundation.Diagnostics
	for _, check := range []func() error{c.collectTypes, c.collectConstants, c.resolveTypeFields, c.checkRuntimeArrayPlacement, c.checkTypeCycles, c.collectFunctions} {
		if err := check(); err != nil {
			interfaceDiagnostics = appendError(interfaceDiagnostics, err)
		}
	}
	if len(interfaceDiagnostics) > 0 {
		return nil, nil, append(documentationDiagnostics, interfaceDiagnostics...).Sorted()
	}
	if err := c.lowerFunctions(); err != nil {
		return nil, nil, appendError(documentationDiagnostics, err).Sorted()
	}
	inferBufferAccess(c.kernel)
	if err := checkRecursion(c.kernel); err != nil {
		return nil, nil, appendError(documentationDiagnostics, err).Sorted()
	}
	if err := ir.VerifyKernel(c.kernel); err != nil {
		var diagnostic *foundation.Diagnostic
		if errors.As(err, &diagnostic) {
			return nil, nil, appendError(documentationDiagnostics, diagnostic).Sorted()
		}
		return nil, nil, fmt.Errorf("internal IR verification failed: %w", err)
	}
	if err := c.lowerPrograms(); err != nil {
		return nil, nil, appendError(documentationDiagnostics, err).Sorted()
	}
	if err := ir.Verify(c.module); err != nil {
		return nil, nil, fmt.Errorf("internal Flow IR verification failed: %w", err)
	}
	if len(documentationDiagnostics) > 0 {
		return nil, nil, documentationDiagnostics.Sorted()
	}
	return c.module, documentationFiles, nil
}

func appendError(diagnostics foundation.Diagnostics, err error) foundation.Diagnostics {
	var list foundation.Diagnostics
	if errors.As(err, &list) {
		return append(diagnostics, list...)
	}
	var diagnostic *foundation.Diagnostic
	if errors.As(err, &diagnostic) {
		diagnostic.Kind = "semantic"
		return append(diagnostics, *diagnostic)
	}
	return append(diagnostics, foundation.Diagnostic{Kind: "semantic", Message: err.Error()})
}

func parallel(workers, count int, work func(int) error) []error {
	errors := make([]error, count)
	jobs := make(chan int)
	var wait sync.WaitGroup
	workers = min(workers, count)
	for range workers {
		wait.Add(1)
		go func() {
			defer wait.Done()
			for index := range jobs {
				errors[index] = work(index)
			}
		}()
	}
	for index := range count {
		jobs <- index
	}
	close(jobs)
	wait.Wait()
	return errors
}
