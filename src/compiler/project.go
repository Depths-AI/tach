package compiler

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"regexp"
	"runtime"
	"sort"
	"strings"
	"sync"

	"tach/src/ast"
	"tach/src/bindings"
	"tach/src/foundation"
	"tach/src/ir"
	"tach/src/parser"
	"tach/src/sema"
)

type manifest struct {
	Name       string `json:"name"`
	Version    string `json:"version"`
	JavaScript struct {
		Package string `json:"package"`
	} `json:"javascript"`
	Docs struct {
		Title   string `json:"title"`
		Summary string `json:"summary"`
	} `json:"docs"`
}

type kernel struct {
	Module        string
	Name          string
	Identity      string
	Path          string
	Source        string
	AST           *ast.Module
	Documentation ir.Documentation
}

type project struct {
	Root     string
	Manifest manifest
	Kernels  []kernel
	sources  map[string]string
}

var (
	semverPattern = regexp.MustCompile(`^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)(?:-(?:0|[1-9][0-9]*|[0-9]*[A-Za-z-][0-9A-Za-z-]*)(?:\.(?:0|[1-9][0-9]*|[0-9]*[A-Za-z-][0-9A-Za-z-]*))*)?(?:\+[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?$`)
	npmPattern    = regexp.MustCompile(`^(?:@[a-z0-9][a-z0-9._~-]*/)?[a-z0-9][a-z0-9._~-]*$`)
)

func findRoot(start string) (string, error) {
	root, err := filepath.Abs(start)
	if err != nil {
		return "", err
	}
	if info, statErr := os.Stat(root); statErr == nil && !info.IsDir() {
		root = filepath.Dir(root)
	}
	for {
		manifest := filepath.Join(root, "tach.json")
		if info, err := os.Stat(manifest); err == nil && !info.IsDir() {
			return filepath.EvalSymlinks(root)
		}
		parent := filepath.Dir(root)
		if parent == root {
			return "", fmt.Errorf("no tach.json found from %s", start)
		}
		root = parent
	}
}

func loadProject(cwd string, workers int) (*project, error) {
	root, err := findRoot(cwd)
	if err != nil {
		return nil, err
	}
	manifestPath := filepath.Join(root, "tach.json")
	manifest, manifestSource, err := readManifest(manifestPath)
	if err != nil {
		span := fileSpan("tach.json")
		var syntax *json.SyntaxError
		if errors.As(err, &syntax) {
			span.Start = position(manifestSource, int(syntax.Offset)-1)
			span.End = span.Start
			span.End.Column++
			span.End.Offset++
		}
		message := strings.TrimPrefix(err.Error(), "tach.json: ")
		return nil, newDiagnosticError(foundation.Diagnostics{{Kind: "manifest", Span: span, Message: message}}, map[string]string{"tach.json": manifestSource})
	}
	kernels, diagnostics := discover(root)
	if len(kernels) == 0 {
		diagnostics = append(diagnostics, foundation.Diagnostic{Kind: "layout", Span: fileSpan("tach.json"), Message: "project contains no module kernel files"})
	}
	project := &project{Root: root, Manifest: manifest, Kernels: kernels, sources: map[string]string{"tach.json": manifestSource}}
	project.parse(workers, &diagnostics)
	project.validate(&diagnostics)
	if len(diagnostics) > 0 {
		return nil, newDiagnosticError(diagnostics, project.sources)
	}
	return project, nil
}

func readManifest(path string) (manifest, string, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return manifest{}, "", err
	}
	source := string(data)
	if err := rejectDuplicateKeys(json.NewDecoder(strings.NewReader(source))); err != nil {
		return manifest{}, source, fmt.Errorf("tach.json: %w", err)
	}
	decoder := json.NewDecoder(strings.NewReader(source))
	decoder.DisallowUnknownFields()
	var result manifest
	if err := decoder.Decode(&result); err != nil {
		return manifest{}, source, fmt.Errorf("tach.json: %w", err)
	}
	if err := requireEOF(decoder); err != nil {
		return manifest{}, source, fmt.Errorf("tach.json: %w", err)
	}
	for _, field := range [][2]string{{"name", result.Name}, {"version", result.Version}, {"javascript.package", result.JavaScript.Package}, {"docs.title", result.Docs.Title}, {"docs.summary", result.Docs.Summary}} {
		if strings.TrimSpace(field[1]) == "" {
			return manifest{}, source, fmt.Errorf("tach.json: %s is required and must be non-empty", field[0])
		}
	}
	if !semverPattern.MatchString(result.Version) {
		return manifest{}, source, fmt.Errorf("tach.json: version %q is not semantic versioning", result.Version)
	}
	if !npmPattern.MatchString(result.JavaScript.Package) || len(result.JavaScript.Package) > 214 {
		return manifest{}, source, fmt.Errorf("tach.json: javascript.package %q is not a valid npm package name", result.JavaScript.Package)
	}
	return result, source, nil
}

func rejectDuplicateKeys(decoder *json.Decoder) error {
	var value func() error
	value = func() error {
		token, err := decoder.Token()
		if err != nil {
			return err
		}
		delimiter, ok := token.(json.Delim)
		if !ok {
			return nil
		}
		switch delimiter {
		case '{':
			seen := map[string]bool{}
			for decoder.More() {
				key, err := decoder.Token()
				if err != nil {
					return err
				}
				name := key.(string)
				if seen[name] {
					return fmt.Errorf("duplicate field %q", name)
				}
				seen[name] = true
				if err := value(); err != nil {
					return err
				}
			}
		case '[':
			for decoder.More() {
				if err := value(); err != nil {
					return err
				}
			}
		}
		_, err = decoder.Token()
		return err
	}
	if err := value(); err != nil {
		return err
	}
	return requireEOF(decoder)
}

func requireEOF(decoder *json.Decoder) error {
	var extra any
	if err := decoder.Decode(&extra); !errors.Is(err, io.EOF) {
		if err == nil {
			return fmt.Errorf("multiple JSON values")
		}
		return err
	}
	return nil
}

func discover(root string) ([]kernel, foundation.Diagnostics) {
	entries, err := os.ReadDir(root)
	if err != nil {
		return nil, foundation.Diagnostics{{Kind: "layout", Message: err.Error()}}
	}
	var kernels []kernel
	var diagnostics foundation.Diagnostics
	folded, foldedModules := map[string]string{}, map[string]string{}
	type physicalFile struct {
		identity string
		info     os.FileInfo
	}
	var physical []physicalFile
	for _, entry := range entries {
		name := entry.Name()
		if filepath.Ext(name) == ".tach" {
			diagnostics = append(diagnostics, foundation.Diagnostic{Kind: "layout", Span: fileSpan(filepath.ToSlash(name)), Message: "root-level .tach files are invalid; place kernels in a module directory"})
			continue
		}
		if strings.EqualFold(name, "build") || !entry.IsDir() || entry.Type()&os.ModeSymlink != 0 {
			continue
		}
		modulePath := filepath.Join(root, name)
		children, readErr := os.ReadDir(modulePath)
		if readErr != nil {
			diagnostics = append(diagnostics, foundation.Diagnostic{Kind: "layout", Span: fileSpan(filepath.ToSlash(name)), Message: readErr.Error()})
			continue
		}
		moduleSeen := false
		for _, child := range children {
			childPath := filepath.Join(modulePath, child.Name())
			if child.IsDir() && child.Type()&os.ModeSymlink == 0 {
				_ = filepath.WalkDir(childPath, func(path string, item os.DirEntry, walkErr error) error {
					if walkErr != nil {
						diagnostics = append(diagnostics, foundation.Diagnostic{Kind: "layout", Span: fileSpan(relative(root, path)), Message: walkErr.Error()})
						return nil
					}
					if item.Type()&os.ModeSymlink != 0 && item.IsDir() {
						return filepath.SkipDir
					}
					if !item.IsDir() && filepath.Ext(item.Name()) == ".tach" {
						diagnostics = append(diagnostics, foundation.Diagnostic{Kind: "layout", Span: fileSpan(relative(root, path)), Message: ".tach files must be exactly one module directory below tach.json"})
					}
					return nil
				})
				continue
			}
			if filepath.Ext(child.Name()) != ".tach" {
				continue
			}
			if !moduleSeen {
				foldedModule := strings.ToLower(name)
				if previous := foldedModules[foldedModule]; previous != "" && previous != name {
					diagnostics = append(diagnostics, foundation.Diagnostic{Kind: "layout", Span: fileSpan(filepath.ToSlash(name)), Message: fmt.Sprintf("module identity collides with %s under case folding", previous)})
				}
				foldedModules[foldedModule], moduleSeen = name, true
			}
			kernelName := strings.TrimSuffix(child.Name(), filepath.Ext(child.Name()))
			identity := filepath.ToSlash(filepath.Join(name, kernelName))
			if !validImport(identity) {
				diagnostics = append(diagnostics, foundation.Diagnostic{Kind: "layout", Span: fileSpan(identity + ".tach"), Message: fmt.Sprintf("invalid kernel identity %q; want <module>/<kernel> using canonical import spelling", identity)})
				continue
			}
			fold := strings.ToLower(identity)
			if previous := folded[fold]; previous != "" && previous != identity {
				diagnostics = append(diagnostics, foundation.Diagnostic{Kind: "layout", Span: fileSpan(identity + ".tach"), Message: fmt.Sprintf("kernel identity collides with %s under case folding", previous)})
			}
			folded[fold] = identity
			info, statErr := os.Stat(childPath)
			if statErr != nil {
				diagnostics = append(diagnostics, foundation.Diagnostic{Kind: "layout", Span: fileSpan(identity + ".tach"), Message: statErr.Error()})
				continue
			}
			// DECISION: os.SameFile is portable but O(n²); use platform file IDs if
			// physical-identity checks ever become measurable during discovery.
			duplicate := ""
			for _, previous := range physical {
				if os.SameFile(info, previous.info) {
					duplicate = previous.identity
					break
				}
			}
			if duplicate != "" {
				diagnostics = append(diagnostics, foundation.Diagnostic{Kind: "layout", Span: fileSpan(identity + ".tach"), Message: fmt.Sprintf("physical kernel is already included as %s", duplicate)})
				continue
			}
			physical = append(physical, physicalFile{identity, info})
			kernels = append(kernels, kernel{Module: name, Name: kernelName, Identity: identity, Path: childPath})
		}
	}
	sort.Slice(kernels, func(i, j int) bool { return kernels[i].Identity < kernels[j].Identity })
	return kernels, diagnostics
}

func (p *project) parse(workers int, diagnostics *foundation.Diagnostics) {
	if workers <= 0 || workers > runtime.GOMAXPROCS(0) {
		workers = runtime.GOMAXPROCS(0)
	}
	jobs := make(chan int)
	results := make([]foundation.Diagnostics, len(p.Kernels))
	var wait sync.WaitGroup
	for range workers {
		wait.Add(1)
		go func() {
			defer wait.Done()
			for index := range jobs {
				kernel := &p.Kernels[index]
				data, err := os.ReadFile(kernel.Path)
				if err != nil {
					results[index] = foundation.Diagnostics{{Kind: "source", Span: fileSpan(kernel.Identity + ".tach"), Message: err.Error()}}
					kernel.AST = &ast.Module{File: kernel.Identity}
					continue
				}
				kernel.Source = string(data)
				kernel.AST, results[index] = parser.ParseRecover(kernel.Identity+".tach", kernel.Source)
				kernel.AST.File = kernel.Identity
			}
		}()
	}
	for i := range p.Kernels {
		jobs <- i
	}
	close(jobs)
	wait.Wait()
	for i := range p.Kernels {
		p.sources[p.Kernels[i].Identity+".tach"] = p.Kernels[i].Source
		*diagnostics = append(*diagnostics, results[i]...)
	}
}

func (p *project) validate(diagnostics *foundation.Diagnostics) {
	byIdentity := map[string]*kernel{}
	owners := map[string]foundation.Span{}
	for i := range p.Kernels {
		kernel := &p.Kernels[i]
		byIdentity[kernel.Identity] = kernel
		for _, declaration := range kernel.AST.Decls {
			var name string
			exported := false
			validate := ir.ValidateExportName
			switch item := declaration.(type) {
			case *ast.TypeDecl:
				name = item.Name
				exported = true
				validate = ir.ValidateExportTypeName
			case *ast.ConstDecl:
				name = item.Name
			case *ast.FunctionDecl:
				name = item.Name
				exported = item.Exported
				if exported {
					for _, parameter := range item.Params {
						if err := ir.ValidateExportName(parameter.Name); err != nil {
							*diagnostics = append(*diagnostics, foundation.Diagnostic{Kind: "export", Span: parameter.Span, Message: fmt.Sprintf("parameter name: %v", err)})
						}
					}
				}
			}
			if sema.ReservedName(name) {
				*diagnostics = append(*diagnostics, foundation.Diagnostic{Kind: "name", Span: declaration.GetSpan(), Message: fmt.Sprintf("declaration name %q is reserved by Tach", name)})
			}
			if exported {
				if err := validate(name); err != nil {
					*diagnostics = append(*diagnostics, foundation.Diagnostic{Kind: "export", Span: declaration.GetSpan(), Message: err.Error()})
				}
			}
			if previous, exists := owners[name]; exists {
				*diagnostics = append(*diagnostics, foundation.Diagnostic{Kind: "name", Span: declaration.GetSpan(), Message: fmt.Sprintf("project declaration %q is already defined", name), Related: []foundation.RelatedDiagnostic{{Span: previous, Message: "first declaration"}}})
			} else {
				owners[name] = declaration.GetSpan()
			}
		}
	}
	for i := range p.Kernels {
		kernel := &p.Kernels[i]
		seen := map[string]bool{}
		for j := range kernel.AST.Imports {
			item := &kernel.AST.Imports[j]
			if !validImport(item.Target) {
				*diagnostics = append(*diagnostics, foundation.Diagnostic{Kind: "import", Span: item.Span, Message: fmt.Sprintf("invalid import %q; want \"<module>/<kernel>\" without .tach", item.Target)})
				continue
			}
			if item.Target == kernel.Identity {
				*diagnostics = append(*diagnostics, foundation.Diagnostic{Kind: "import", Span: item.Span, Message: "kernel cannot import itself"})
			}
			if seen[item.Target] {
				*diagnostics = append(*diagnostics, foundation.Diagnostic{Kind: "import", Span: item.Span, Message: fmt.Sprintf("duplicate import %q", item.Target)})
			}
			seen[item.Target] = true
			if byIdentity[item.Target] == nil {
				*diagnostics = append(*diagnostics, foundation.Diagnostic{Kind: "import", Span: item.Span, Message: fmt.Sprintf("import target %q does not exist", item.Target)})
			}
		}
	}
	*diagnostics = append(*diagnostics, graphDiagnostics(p.Kernels, false)...)
	*diagnostics = append(*diagnostics, graphDiagnostics(p.Kernels, true)...)
}

func validImport(target string) bool {
	if strings.Count(target, "/") != 1 || strings.HasPrefix(target, "@") || strings.ContainsAny(target, `\:`) || strings.HasSuffix(target, ".tach") {
		return false
	}
	parts := strings.Split(target, "/")
	return parts[0] != "" && parts[1] != "" && parts[0] != "." && parts[0] != ".." && parts[1] != "." && parts[1] != ".."
}

type graphEdge struct {
	to   string
	span foundation.Span
}

func graphDiagnostics(kernels []kernel, modules bool) foundation.Diagnostics {
	graph := map[string][]graphEdge{}
	known := map[string]bool{}
	for _, kernel := range kernels {
		known[kernel.Identity] = true
	}
	for _, kernel := range kernels {
		from := kernel.Identity
		if modules {
			from = kernel.Module
		}
		if graph[from] == nil {
			graph[from] = []graphEdge{}
		}
		for _, item := range kernel.AST.Imports {
			if !validImport(item.Target) || !known[item.Target] {
				continue
			}
			to := item.Target
			if modules {
				to = strings.Split(item.Target, "/")[0]
				if to == from {
					continue
				}
			}
			graph[from] = append(graph[from], graphEdge{to: to, span: item.Span})
		}
	}
	nodes := make([]string, 0, len(graph))
	for node := range graph {
		nodes = append(nodes, node)
		sort.Slice(graph[node], func(i, j int) bool { return graph[node][i].to < graph[node][j].to })
	}
	sort.Strings(nodes)
	state, positions := map[string]uint8{}, map[string]int{}
	var stack []string
	var edges []graphEdge
	var diagnostics foundation.Diagnostics
	var visit func(string)
	visit = func(node string) {
		state[node], positions[node] = 1, len(stack)
		stack = append(stack, node)
		for _, edge := range graph[node] {
			if graph[edge.to] == nil {
				continue
			}
			if state[edge.to] == 0 {
				edges = append(edges, edge)
				visit(edge.to)
				edges = edges[:len(edges)-1]
			} else if state[edge.to] == 1 {
				start := positions[edge.to]
				chain := append(append([]string(nil), stack[start:]...), edge.to)
				related := make([]foundation.RelatedDiagnostic, 0, len(chain)-1)
				for i := start; i < len(edges); i++ {
					related = append(related, foundation.RelatedDiagnostic{Span: edges[i].span, Message: fmt.Sprintf("%s imports %s", stack[i], stack[i+1])})
				}
				related = append(related, foundation.RelatedDiagnostic{Span: edge.span, Message: fmt.Sprintf("%s imports %s", node, edge.to)})
				kind := "kernel"
				if modules {
					kind = "module"
				}
				diagnostics = append(diagnostics, foundation.Diagnostic{Kind: kind + "-cycle", Span: edge.span, Message: fmt.Sprintf("%s import cycle: %s", kind, strings.Join(chain, " -> ")), Related: related})
			}
		}
		stack = stack[:len(stack)-1]
		delete(positions, node)
		state[node] = 2
	}
	for _, node := range nodes {
		if state[node] == 0 {
			visit(node)
		}
	}
	return diagnostics
}

type diagnosticError struct {
	diagnostics foundation.Diagnostics
}

func (e *diagnosticError) Error() string {
	var out strings.Builder
	for i, diagnostic := range e.diagnostics {
		if i > 0 {
			out.WriteByte('\n')
		}
		fmt.Fprintf(&out, "%s: %s", diagnostic.Span, diagnostic.Message)
		if diagnostic.Source != "" {
			fmt.Fprintf(&out, "\n  %s\n  %s%s", diagnostic.Source, strings.Repeat(" ", max(0, diagnostic.Span.Start.Column-1)), strings.Repeat("^", max(1, diagnostic.Span.End.Column-diagnostic.Span.Start.Column)))
		}
		for _, related := range diagnostic.Related {
			fmt.Fprintf(&out, "\n  related %s: %s", related.Span, related.Message)
		}
		if diagnostic.Help != "" {
			fmt.Fprintf(&out, "\n  help: %s", diagnostic.Help)
		}
	}
	return out.String()
}

func newDiagnosticError(diagnostics foundation.Diagnostics, sources map[string]string) *diagnosticError {
	return &diagnosticError{diagnostics: enrichDiagnostics(diagnostics, sources, "error")}
}

func enrichDiagnostics(diagnostics foundation.Diagnostics, sources map[string]string, severity string) foundation.Diagnostics {
	out := append(foundation.Diagnostics(nil), diagnostics...)
	for i := range out {
		if out[i].Severity == "" {
			out[i].Severity = severity
		}
		out[i].Source = sourceLine(sources[out[i].Span.File], out[i].Span.Start.Line)
		out[i].Related = append([]foundation.RelatedDiagnostic(nil), out[i].Related...)
		for j := range out[i].Related {
			related := &out[i].Related[j]
			related.Source = sourceLine(sources[related.Span.File], related.Span.Start.Line)
		}
	}
	return out.Sorted()
}

func ErrorDiagnostics(err error) (foundation.Diagnostics, bool) {
	var diagnostics *diagnosticError
	if !errors.As(err, &diagnostics) {
		return nil, false
	}
	return append(foundation.Diagnostics(nil), diagnostics.diagnostics...), true
}

func (p *project) semanticError(err error) error {
	var diagnostics foundation.Diagnostics
	if errors.As(err, &diagnostics) {
		return newDiagnosticError(diagnostics, p.sources)
	}
	var diagnostic *foundation.Diagnostic
	if errors.As(err, &diagnostic) {
		return newDiagnosticError(foundation.Diagnostics{*diagnostic}, p.sources)
	}
	return err
}

func sourceLine(text string, number int) string {
	lines := strings.Split(text, "\n")
	if number < 1 || number > len(lines) {
		return ""
	}
	return strings.TrimSuffix(lines[number-1], "\r")
}

func position(text string, offset int) foundation.Position {
	if offset < 0 {
		offset = 0
	}
	line, column := 1, 1
	for index, r := range text {
		if index >= offset {
			break
		}
		if r == '\n' {
			line, column = line+1, 1
		} else {
			column++
		}
	}
	return foundation.Position{Offset: offset, Line: line, Column: column}
}

func fileSpan(file string) foundation.Span {
	return foundation.Span{File: file, Start: foundation.Position{Line: 1, Column: 1}, End: foundation.Position{Line: 1, Column: 2}}
}

func relative(root, path string) string {
	value, err := filepath.Rel(root, path)
	if err != nil {
		return filepath.ToSlash(path)
	}
	return filepath.ToSlash(value)
}

func (p *project) description() bindings.ProjectInput {
	input := bindings.ProjectInput{Name: p.Manifest.Name, Version: p.Manifest.Version, Package: p.Manifest.JavaScript.Package, Title: p.Manifest.Docs.Title, Summary: p.Manifest.Docs.Summary}
	for _, kernel := range p.Kernels {
		item := bindings.KernelInput{Module: kernel.Module, Name: kernel.Name, Identity: kernel.Identity, Documentation: kernel.Documentation}
		for _, declaration := range kernel.AST.Decls {
			switch value := declaration.(type) {
			case *ast.TypeDecl:
				item.Types = append(item.Types, value.Name)
			case *ast.FunctionDecl:
				item.Functions = append(item.Functions, value.Name)
			}
		}
		input.Kernels = append(input.Kernels, item)
	}
	return input
}
