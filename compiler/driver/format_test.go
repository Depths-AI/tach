package driver

import (
	"bytes"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"tach/parser"
	"testing"
)

func TestFormatterPreservesCommentsAndIsIdempotent(t *testing.T) {
	root := projectFixture(t, map[string]string{"m/a.tach": "// kept\r\nfunction helper(x:float32,):float32{return (-x);}\n@docs(summary(\"Scales.\"))@workgroup(64) export   function scale[i](values:buffer<float32[]>,factor:float32,){if(!false&&i<values.length){const negative=-64;const bits=~uint32(negative);const total=values[i]+values[i]+values[i]+values[i]+values[i]+values[i]+values[i]+values[i]+values[i]+values[i]; // trailing\r\nvalues[i]=!false?total*factor+float32(bits):-1.0;}}\n"})
	if err := Format(root, 1); err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(root, "m", "a.tach")
	first, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Contains(first, []byte("// kept")) || !bytes.Contains(first, []byte("; // trailing")) || !bytes.Contains(first, []byte("helper(x: float32): float32")) || !bytes.Contains(first, []byte("return (-x)")) || !bytes.Contains(first, []byte(")\n@workgroup(64)\nexport function")) || !bytes.Contains(first, []byte("if (!false &&")) || !bytes.Contains(first, []byte("negative = -64")) || !bytes.Contains(first, []byte("bits = ~uint32")) || !bytes.Contains(first, []byte("!false ? total * factor + float32(bits) : -1.0")) || bytes.Contains(first, []byte("helper(x: float32,)")) || bytes.ContainsAny(first, "\t\r") || !bytes.HasSuffix(first, []byte("\n")) {
		t.Fatalf("formatted source = %q", first)
	}
	for _, line := range strings.Split(string(first), "\n") {
		if len(line) > 100 {
			t.Fatalf("formatter produced %d columns: %s", len(line), line)
		}
	}
	if err := Format(root, 1); err != nil {
		t.Fatal(err)
	}
	second, _ := os.ReadFile(path)
	if !bytes.Equal(first, second) {
		t.Fatalf("formatter is not idempotent:\n%s\n---\n%s", first, second)
	}
}

func TestFormatterMakesNoPartialWritesOnSyntaxError(t *testing.T) {
	const original = `export   function valid[i](out:buffer<float32[]>){if(i<out.length){out[i]=1.0;}}`
	root := projectFixture(t, map[string]string{
		"m/a.tach": original,
		"m/b.tach": `export function broken[`,
	})
	if err := Format(root, 2); err == nil {
		t.Fatal("invalid project formatted")
	}
	content, err := os.ReadFile(filepath.Join(root, "m", "a.tach"))
	if err != nil {
		t.Fatal(err)
	}
	if string(content) != original {
		t.Fatalf("valid sibling changed to %q", content)
	}
}

func FuzzFormatterPreservesSyntaxAndStabilizes(f *testing.F) {
	for _, seed := range []string{
		`// comment
@docs(summary("Scales.")) @workgroup(64) export function scale[i](out: buffer<float32[]>, factor: float32,) { if (!false && i < out.length) { out[i] = factor < 0.0 ? -factor : factor; } }`,
		`import "base/data"; function value(x: float32): float32 { return (~uint32(x) & 3) == 0 ? -x : x; }`,
		`function fill[i](out: buffer<vec<float32, 4>[]>) { if (i < out.length) { out[i] = vec(1, 2, 3, 4); } } export function make() { let out = transient<vec<float32, 4>>(4); run fill(out) over 4; }`,
	} {
		f.Add(seed)
	}
	f.Fuzz(func(t *testing.T, input string) {
		// DECISION: bound one fuzz case at 64 KiB; project tests own larger-file pressure, and this cap can rise with formatter limits.
		if len(input) > 64<<10 {
			t.Skip()
		}
		if _, diagnostics := parser.ParseRecover("fuzz.tach", input); len(diagnostics) > 0 {
			return
		}
		formatted, err := formatSource("fuzz.tach", input)
		if err != nil {
			t.Fatal(err)
		}
		if _, diagnostics := parser.ParseRecover("fuzz.tach", formatted); len(diagnostics) > 0 {
			t.Fatalf("formatter produced invalid syntax:\n%s\n%v", formatted, diagnostics)
		}
		again, err := formatSource("fuzz.tach", formatted)
		if err != nil || again != formatted {
			t.Fatalf("formatter is not idempotent:\n%s\n---\n%s\n%v", formatted, again, err)
		}
		preserved := func(source string) []string {
			tokens, _ := parser.Tokenize("fuzz.tach", source)
			var values []string
			for _, token := range tokens {
				for _, trivia := range token.LeadingComments {
					values = append(values, "comment:"+strings.TrimRight(trivia.Text, " \t\r"))
				}
				if token.Kind == parser.String {
					values = append(values, "string:"+token.Text)
				}
			}
			return values
		}
		if !slices.Equal(preserved(input), preserved(formatted)) {
			t.Fatal("formatter changed string or comment contents")
		}
		lexemes := func(source string) []string {
			tokens, _ := parser.Tokenize("fuzz.tach", source)
			var values []string
			for i, token := range tokens[:len(tokens)-1] {
				if token.Kind == parser.Comma && (tokens[i+1].Kind == parser.RParen || tokens[i+1].Kind == parser.RBracket || tokens[i+1].Kind == parser.RBrace) {
					continue
				}
				values = append(values, fmt.Sprintf("%d:%s", token.Kind, token.Text))
			}
			return values
		}
		if !slices.Equal(lexemes(input), lexemes(formatted)) {
			t.Fatal("formatter changed the syntax token stream")
		}
	})
}
