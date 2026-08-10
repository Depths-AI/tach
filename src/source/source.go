package source

import "fmt"

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

type Error struct {
	Span    Span
	Message string
}

func (e *Error) Error() string { return fmt.Sprintf("%s: %s", e.Span.String(), e.Message) }
