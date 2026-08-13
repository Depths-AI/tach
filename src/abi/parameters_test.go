package abi

import (
	"tach/src/ir"
	"tach/src/types"
	"testing"
)

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
