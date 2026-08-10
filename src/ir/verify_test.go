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
