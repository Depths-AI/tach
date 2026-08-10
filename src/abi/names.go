// Package abi owns target-independent external names and ABI identifiers used
// by every Tach backend and generated host binding.
package abi

import (
	"fmt"
	"strings"
	"unicode"
)

// KernelEntry is the exact externally-visible entry-point name emitted into
// both WGSL and SPIR-V. Exported Tach kernels keep their source spelling; host
// applications should never have to know or reconstruct a compiler mangling
// scheme.
func KernelEntry(name string) string { return name }

// Mangle maps a Tach identifier to a conservative ASCII identifier for
// compiler-private symbols. Public kernel entry points deliberately do not use
// it.
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
