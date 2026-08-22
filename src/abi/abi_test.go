package abi

import (
	"regexp"
	"strings"
	"tach/src/ir"
	"tach/src/types"
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

func TestPlanParametersUsesExplicitBindingAndCanonicalFields(t *testing.T) {
	f := &ir.Function{Name: "stage", Kind: ir.Stage, Params: []ir.Param{{Name: "enabled", ID: 1, Type: types.TBool}, {Name: "factor", ID: 2, Type: types.TF32}}}
	block, err := PlanParameters(f, 3)
	if err != nil {
		t.Fatal(err)
	}
	if block.Binding != 3 || block.Layout.Size != 16 || len(block.Fields) != 2 || block.Fields[0].Physical != types.TU32 || block.Fields[1].Offset != 4 {
		t.Fatalf("unexpected block: %+v", block)
	}
}

func TestPlanParametersOmitsEmptyBlock(t *testing.T) {
	block, err := PlanParameters(&ir.Function{Kind: ir.Stage}, 0)
	if err != nil || block != nil {
		t.Fatalf("block=%+v err=%v", block, err)
	}
}

func TestPlanParametersRejectsBooleanMasks(t *testing.T) {
	mask := types.Vec(types.TBool, 2)
	for _, value := range []*types.Type{mask, {Kind: types.Struct, Name: "Options", Fields: []types.Field{{Name: "mask", Type: mask}}}} {
		f := &ir.Function{Name: "stage", Kind: ir.Stage, Params: []ir.Param{{Name: "value", Type: value}}}
		if _, err := PlanParameters(f, 0); err == nil || !strings.Contains(err.Error(), "cannot cross the host parameter ABI") {
			t.Fatalf("PlanParameters(%s) error = %v", value, err)
		}
	}
}
