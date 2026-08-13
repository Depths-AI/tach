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
