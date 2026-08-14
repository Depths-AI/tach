// Package abi owns target-independent external names and ABI identifiers used
// by every Tach backend and generated host binding.
package abi

import (
	"fmt"
	"strings"
	"unicode"
)

func PrivateEntry(index int) string { return fmt.Sprintf("_tach_k%d", index) }

// Mangle maps a Tach identifier to a conservative ASCII identifier for
// compiler-private symbols.
func Mangle(s string) string {
	var b strings.Builder
	for _, r := range s {
		if r == '_' {
			b.WriteString("__")
		} else if r < 128 && (unicode.IsLetter(r) || unicode.IsDigit(r)) {
			b.WriteRune(r)
		} else {
			fmt.Fprintf(&b, "_u%x_", r)
		}
	}
	if b.Len() == 0 {
		return "x"
	}
	return b.String()
}

func ValidateExportName(name string) error {
	if !asciiIdentifier(name) || typeScriptKeywords[name] {
		return fmt.Errorf("%q is not a valid generated export", name)
	}
	return nil
}

func ValidateExportTypeName(name string) error {
	if err := ValidateExportName(name); err != nil {
		return err
	}
	if name == "Float32Array" || name == "Int32Array" || name == "Uint32Array" || name == "ReadonlyArray" {
		return fmt.Errorf("%q collides with a generated host collection type", name)
	}
	return nil
}

func asciiIdentifier(s string) bool {
	if s == "" {
		return false
	}
	for i, r := range s {
		if r > unicode.MaxASCII || !(r == '_' || r == '$' || unicode.IsLetter(r) || i > 0 && unicode.IsDigit(r)) {
			return false
		}
	}
	return true
}

var typeScriptKeywords = func() map[string]bool {
	keywords := map[string]bool{}
	for _, keyword := range strings.Fields("abstract any arguments as asserts async await bigint boolean break case catch class const constructor continue debugger declare default delete do else enum eval export extends false finally for from function get global if implements import in infer instanceof interface intrinsic is keyof let module namespace never new null number object of override package private protected public readonly require return satisfies set static string super switch symbol this throw true try type typeof undefined unique unknown using var void while with yield") {
		keywords[keyword] = true
	}
	return keywords
}()
