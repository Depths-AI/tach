package source

import (
	"fmt"
	"sort"
	"strings"
)

type Pos struct {
	Offset int
	Line   int
	Column int
}

type Span struct {
	File       string
	Start, End Pos
}

func (s Span) String() string {
	if s.File == "" {
		return fmt.Sprintf("%d:%d", s.Start.Line, s.Start.Column)
	}
	return fmt.Sprintf("%s:%d:%d", s.File, s.Start.Line, s.Start.Column)
}

type Related struct {
	Span    Span
	Message string
}

type Diagnostic struct {
	Kind    string
	Span    Span
	Message string
	Related []Related
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
		return a.Kind < b.Kind
	})
	return out
}
