package source

import (
	"fmt"
	"sort"
	"strings"
)

type Pos struct {
	Offset int `json:"offset"`
	Line   int `json:"line"`
	Column int `json:"column"`
}

type Span struct {
	File  string `json:"file"`
	Start Pos    `json:"start"`
	End   Pos    `json:"end"`
}

func (s Span) String() string {
	if s.File == "" {
		return fmt.Sprintf("%d:%d", s.Start.Line, s.Start.Column)
	}
	return fmt.Sprintf("%s:%d:%d", s.File, s.Start.Line, s.Start.Column)
}

type Related struct {
	Span    Span   `json:"span"`
	Message string `json:"message"`
	Source  string `json:"source,omitempty"`
}

type Diagnostic struct {
	Severity string    `json:"severity"`
	Kind     string    `json:"code"`
	Span     Span      `json:"span"`
	Message  string    `json:"message"`
	Help     string    `json:"help,omitempty"`
	Source   string    `json:"source,omitempty"`
	Related  []Related `json:"related,omitempty"`
}

func (d Diagnostic) Error() string { return fmt.Sprintf("%s: %s", d.Span.String(), d.Message) }

type Diagnostics []Diagnostic

func (ds Diagnostics) Error() string {
	parts := make([]string, len(ds))
	for i := range ds {
		parts[i] = ds[i].Error()
	}
	return strings.Join(parts, "\n")
}

func (ds Diagnostics) Sorted() Diagnostics {
	out := append(Diagnostics(nil), ds...)
	sort.SliceStable(out, func(i, j int) bool {
		a, b := out[i], out[j]
		if a.Span.File != b.Span.File {
			return a.Span.File < b.Span.File
		}
		if a.Span.Start.Offset != b.Span.Start.Offset {
			return a.Span.Start.Offset < b.Span.Start.Offset
		}
		if a.Severity != b.Severity {
			return a.Severity != "warning"
		}
		return a.Kind < b.Kind
	})
	return out
}
