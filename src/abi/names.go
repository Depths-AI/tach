// Package abi owns target-independent external names and ABI identifiers used
// by every Tach backend and generated host binding.
package abi

import (
	"fmt"
	"strings"
	"unicode"
)

// KernelEntry is the exact externally-visible entry-point name emitted into
// both WGSL and SPIR-V. Keeping this in the ABI package makes reflection
// backend-neutral.
func KernelEntry(name string) string { return "_tach_k_" + Mangle(name) }

// Mangle maps a Tach identifier to a conservative ASCII identifier. Tach source
// identifiers may be Unicode; target formats and generated host code receive a
// stable byte-for-byte spelling independent of locale or backend.
func Mangle(s string) string {
	var b strings.Builder
	for i, r := range s {
		if (r == '_' || unicode.IsLetter(r) || i > 0 && unicode.IsDigit(r)) && r < 128 {
			b.WriteRune(r)
			continue
		}
		fmt.Fprintf(&b, "x%x_", r)
	}
	if b.Len() == 0 {
		return "x"
	}
	return b.String()
}
