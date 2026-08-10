package lexer

import "testing"

func TestNumericLiteralForms(t *testing.T) {
	toks, err := Lex("numbers.tach", `0xffu 0b1010i 1.25e-3f 2E+4 1_000_000u`)
	if err != nil {
		t.Fatal(err)
	}
	want := []string{"0xffu", "0b1010i", "1.25e-3f", "2E+4", "1_000_000u"}
	for i, w := range want {
		if toks[i].Kind != Number || toks[i].Text != w {
			t.Fatalf("token %d = %v %q, want number %q", i, toks[i].Kind, toks[i].Text, w)
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
