package parser_test

import (
	"strings"
	"testing"

	"tach/parser"
)

func tokenize(t *testing.T, source string) []parser.Token {
	t.Helper()
	tokens, diagnostics := parser.Tokenize(t.Name()+".tach", source)
	if len(diagnostics) > 0 {
		t.Fatal(diagnostics)
	}
	return tokens
}

func TestNumericLiteralForms(t *testing.T) {
	tokens := tokenize(t, `0xff 0b1010 1.25e-3 2E+4 1_000_000`)
	want := []string{"0xff", "0b1010", "1.25e-3", "2E+4", "1_000_000"}
	for i, text := range want {
		if tokens[i].Kind != parser.Number || tokens[i].Text != text {
			t.Fatalf("token %d = %v %q, want number %q", i, tokens[i].Kind, tokens[i].Text, text)
		}
	}
}

func TestNumericLiteralErrors(t *testing.T) {
	for _, source := range []string{"0u", "0i", "1.0f", "0xffu"} {
		_, diagnostics := parser.Tokenize("suffix.tach", source)
		if len(diagnostics) != 1 || !strings.Contains(diagnostics.Error(), "numeric suffixes are not part of Tach") {
			t.Fatalf("Tokenize(%q) diagnostics = %v, want suffix diagnostic", source, diagnostics)
		}
	}
	for _, source := range []string{"0x", "0b", "1e", "1e+"} {
		if _, diagnostics := parser.Tokenize("digits.tach", source); len(diagnostics) != 1 {
			t.Fatalf("Tokenize(%q) diagnostics = %v, want one missing-digit diagnostic", source, diagnostics)
		}
	}
}

func TestCommentsStringsAndLongestOperators(t *testing.T) {
	tokens := tokenize(t, "// ignored\n@docs(summary(\"GPU \\\"work\\\".\")) <<= >>= ++ --")
	if tokens[0].Kind != parser.At || tokens[5].Kind != parser.String || tokens[5].Text != `"GPU \"work\"."` {
		t.Fatalf("tokens = %#v", tokens)
	}
	want := []parser.TokenKind{parser.ShiftLeftEq, parser.ShiftRightEq, parser.PlusPlus, parser.MinusMinus}
	for i, kind := range want {
		if got := tokens[8+i].Kind; got != kind {
			t.Fatalf("operator %d = %v, want %v", i, got, kind)
		}
	}
	block := tokenize(t, `/* not a comment */`)
	if block[0].Kind != parser.Slash {
		t.Fatalf("block comment was ignored: %#v", block)
	}
}

func TestTokenLocationsAndRecovery(t *testing.T) {
	tokens, diagnostics := parser.Tokenize("recovery.tach", "alpha @ # beta\n// tail")
	if len(diagnostics) != 1 || diagnostics[0].Kind != "lexer" || diagnostics[0].Span.Start.Offset != 8 {
		t.Fatalf("diagnostics = %#v", diagnostics)
	}
	if len(tokens) != 4 || tokens[0].Text != "alpha" || tokens[1].Kind != parser.At || tokens[2].Text != "beta" || tokens[3].Kind != parser.EOF {
		t.Fatalf("tokens = %#v", tokens)
	}
	if tokens[2].Span.Start.Line != 1 || tokens[3].LeadingComments[0].Text != "// tail" || tokens[3].Span.Start.Line != 2 {
		t.Fatalf("source locations = %#v", tokens)
	}
}

func FuzzTokenRecoverySpansStayBounded(f *testing.F) {
	for _, seed := range []string{"", "// comment\n@docs(summary(\"x\"))", "0xff 1e+ broken", string([]byte{0xff, '\n', 0xfe})} {
		f.Add(seed)
	}
	f.Fuzz(func(t *testing.T, input string) {
		// DECISION: bound one fuzz case at 64 KiB; project tests own larger-file pressure, and this cap can rise with parser limits.
		if len(input) > 64<<10 {
			t.Skip()
		}
		tokens, diagnostics := parser.Tokenize("fuzz.tach", input)
		if len(tokens) == 0 || tokens[len(tokens)-1].Kind != parser.EOF {
			t.Fatal("tokenization did not terminate with EOF")
		}
		last := 0
		for _, token := range tokens {
			if token.Span.Start.Offset < last || token.Span.End.Offset < token.Span.Start.Offset || token.Span.End.Offset > len(input) {
				t.Fatalf("invalid token span after byte %d: %#v", last, token)
			}
			last = token.Span.End.Offset
			for _, trivia := range token.LeadingComments {
				if trivia.Span.Start.Offset < 0 || trivia.Span.End.Offset < trivia.Span.Start.Offset || trivia.Span.End.Offset > len(input) {
					t.Fatalf("invalid trivia span: %#v", trivia)
				}
			}
		}
		for _, diagnostic := range diagnostics {
			if diagnostic.Span.Start.Offset < 0 || diagnostic.Span.End.Offset < diagnostic.Span.Start.Offset || diagnostic.Span.End.Offset > len(input) {
				t.Fatalf("invalid diagnostic span: %#v", diagnostic)
			}
		}
	})
}
