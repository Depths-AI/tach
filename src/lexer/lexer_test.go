package lexer

import (
	"strings"
	"testing"
)

func TestNumericLiteralForms(t *testing.T) {
	toks, err := Lex("numbers.tach", `0xff 0b1010 1.25e-3 2E+4 1_000_000`)
	if err != nil {
		t.Fatal(err)
	}
	want := []string{"0xff", "0b1010", "1.25e-3", "2E+4", "1_000_000"}
	for i, w := range want {
		if toks[i].Kind != Number || toks[i].Text != w {
			t.Fatalf("token %d = %v %q, want number %q", i, toks[i].Kind, toks[i].Text, w)
		}
	}
}

func TestNumericLiteralRejectsTypeSuffixes(t *testing.T) {
	for _, src := range []string{"0u", "0i", "1.0f", "0xffu"} {
		_, err := Lex("bad.tach", src)
		if err == nil || !strings.Contains(err.Error(), "numeric suffixes are not part of Tach") {
			t.Fatalf("Lex(%q) error = %v, want suffix diagnostic", src, err)
		}
	}
}

func TestNumericLiteralRejectsMissingDigits(t *testing.T) {
	for _, src := range []string{"0x", "0b", "1e", "1e+"} {
		if _, err := Lex("bad.tach", src); err == nil {
			t.Fatalf("Lex(%q) succeeded, want error", src)
		}
	}
}

func TestDocumentationStringsAndLineComments(t *testing.T) {
	tokens, err := Lex("docs.tach", `// ignored
@docs(summary("GPU \"work\"."))`)
	if err != nil {
		t.Fatal(err)
	}
	if tokens[0].Kind != At || tokens[5].Kind != String || tokens[5].Text != `"GPU \"work\"."` {
		t.Fatalf("tokens = %#v", tokens)
	}
	block, err := Lex("comments.tach", `/* not a comment */`)
	if err != nil || block[0].Kind != Slash {
		t.Fatalf("block comment was ignored: tokens=%#v error=%v", block, err)
	}
}

func FuzzLexerRecoverySpansStayBounded(f *testing.F) {
	for _, seed := range []string{"", "// comment\n@docs(summary(\"x\"))", "0xff 1e+ broken", string([]byte{0xff, '\n', 0xfe})} {
		f.Add(seed)
	}
	f.Fuzz(func(t *testing.T, input string) {
		// DECISION: bound one fuzz case at 64 KiB; project tests own larger-file pressure, and this cap can rise with lexer limits.
		if len(input) > 64<<10 {
			t.Skip()
		}
		tokens, diagnostics := LexRecover("fuzz.tach", input)
		if len(tokens) == 0 || tokens[len(tokens)-1].Kind != EOF {
			t.Fatal("lexer did not terminate with EOF")
		}
		last := 0
		for _, token := range tokens {
			if token.Span.Start.Offset < last || token.Span.End.Offset < token.Span.Start.Offset || token.Span.End.Offset > len(input) {
				t.Fatalf("invalid token span after byte %d: %#v", last, token)
			}
			last = token.Span.End.Offset
			for _, trivia := range token.Leading {
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
