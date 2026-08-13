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
