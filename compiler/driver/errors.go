package driver

import (
	"errors"
	"fmt"
	"strings"
	"tach/foundation"
)

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
