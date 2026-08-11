package ir

import (
	"strings"
	"testing"

	"tach/src/types"
)

func TestVerifyRejectsDuplicateSSAIDsAcrossRegions(t *testing.T) {
	m := &Module{Functions: []*Function{{
		Name:   "bad",
		Return: types.TVoid,
		Body: &Block{
			Instrs: []Instr{
				&Const{Result: 1, Type: types.TBool, Raw: "true"},
				&If{
					Cond:    1,
					Results: []Result{{ID: 2, Type: types.TBool}},
					Then: &Block{Instrs: []Instr{
						&Const{Result: 2, Type: types.TBool, Raw: "true"},
					}, Term: &Yield{Values: []ValueID{2}}},
					Else: &Block{Instrs: []Instr{
						&Const{Result: 3, Type: types.TBool, Raw: "false"},
					}, Term: &Yield{Values: []ValueID{3}}},
				},
			},
			Term: &Return{},
		},
	}}}
	if err := Verify(m); err == nil || !strings.Contains(err.Error(), "defined twice") {
		t.Fatalf("Verify() error = %v, want duplicate SSA definition", err)
	}
}

func TestVerifyRequiresEveryKernelValueMapping(t *testing.T) {
	module := &Module{
		Resources: []Resource{{Name: "out", Type: types.Runtime(types.TU32), Access: Mutable}},
		Functions: []*Function{{
			Name: "write", Compute: true, Return: types.TVoid, Workgroup: [3]uint32{1, 1, 1},
			Indices: []Param{{Name: "i", ID: 1, Type: types.TU32}},
			Params:  []Param{{Name: "value", ID: 2, Type: types.TU32}},
			KernelParams: []KernelParam{
				{Name: "out", Resource: 0},
				{Name: "value", Value: 2, Resource: -1},
			},
			Body: &Block{Term: &Return{}},
		}},
	}
	if err := Verify(module); err != nil {
		t.Fatal(err)
	}
	module.Functions[0].KernelParams = module.Functions[0].KernelParams[:1]
	if err := Verify(module); err == nil || !strings.Contains(err.Error(), "value parameter value is not mapped") {
		t.Fatalf("Verify() error = %v, want missing kernel value mapping", err)
	}
}
