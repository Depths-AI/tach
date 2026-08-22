package ir

import (
	"regexp"
	"strings"
	"testing"

	"tach/foundation"
)

func TestMangleIdentifierIsInjectiveAndPortable(t *testing.T) {
	identifier := regexp.MustCompile(`^[A-Za-z_][A-Za-z0-9_]*$`)
	seen := map[string]string{}
	for _, name := range []string{"alpha", "a_b", "a__b", "λ", "_u3bb_", "世界", "_u4e16__u754c_"} {
		mangled := MangleIdentifier(name)
		if !identifier.MatchString(mangled) {
			t.Fatalf("MangleIdentifier(%q) = %q, not a portable identifier", name, mangled)
		}
		if previous, exists := seen[mangled]; exists {
			t.Fatalf("MangleIdentifier(%q) and MangleIdentifier(%q) both equal %q", previous, name, mangled)
		}
		seen[mangled] = name
	}
}

func TestPlanHostParametersUsesExplicitBindingAndCanonicalFields(t *testing.T) {
	f := &Function{Name: "stage", Kind: Stage, Params: []Param{{Name: "enabled", ID: 1, Type: foundation.BoolType}, {Name: "factor", ID: 2, Type: foundation.Float32Type}}}
	block, err := PlanHostParameters(f, 3)
	if err != nil {
		t.Fatal(err)
	}
	if block.Binding != 3 || block.Layout.Size != 16 || len(block.Fields) != 2 || block.Fields[0].Physical != foundation.Uint32Type || block.Fields[1].Offset != 4 {
		t.Fatalf("unexpected block: %+v", block)
	}
}

func TestPlanHostParametersOmitsEmptyBlock(t *testing.T) {
	block, err := PlanHostParameters(&Function{Kind: Stage}, 0)
	if err != nil || block != nil {
		t.Fatalf("block=%+v err=%v", block, err)
	}
}

func TestPlanHostParametersRejectsBooleanMasks(t *testing.T) {
	mask := foundation.VectorOf(foundation.BoolType, 2)
	for _, value := range []*foundation.Type{mask, {Kind: foundation.StructKind, Name: "Options", Fields: []foundation.TypeField{{Name: "mask", Type: mask}}}} {
		f := &Function{Name: "stage", Kind: Stage, Params: []Param{{Name: "value", Type: value}}}
		if _, err := PlanHostParameters(f, 0); err == nil || !strings.Contains(err.Error(), "cannot cross the host parameter ABI") {
			t.Fatalf("PlanHostParameters(%s) error = %v", value, err)
		}
	}
}
