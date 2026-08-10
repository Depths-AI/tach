package abi

import (
	"regexp"
	"testing"
)

func TestMangleIsInjectiveAndPortable(t *testing.T) {
	identifier := regexp.MustCompile(`^[A-Za-z_][A-Za-z0-9_]*$`)
	seen := map[string]string{}
	for _, name := range []string{"alpha", "a_b", "a__b", "λ", "_u3bb_", "世界", "_u4e16__u754c_"} {
		mangled := Mangle(name)
		if !identifier.MatchString(mangled) {
			t.Fatalf("Mangle(%q) = %q, not a portable identifier", name, mangled)
		}
		if previous, exists := seen[mangled]; exists {
			t.Fatalf("Mangle(%q) and Mangle(%q) both equal %q", previous, name, mangled)
		}
		seen[mangled] = name
	}
}
