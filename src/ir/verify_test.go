package ir

import (
	"strings"
	"testing"

	"tach/src/types"
)

func validStage() *Module {
	return &Module{Functions: []*Function{{
		Name:         "write",
		Kind:         Stage,
		Indices:      []Param{{Name: "i", ID: 1, Type: types.TU32}},
		BufferParams: []BufferParam{{Name: "out", Type: types.Runtime(types.TU32), Access: Mutable}},
		SourceParams: []SourceParam{{Name: "out", Kind: SourceBuffer, Buffer: 0}},
		Return:       types.TVoid,
		Body: &Block{
			Instrs: []Instr{
				&PlaceRoot{Result: 1, Type: types.Runtime(types.TU32), Buffer: 0},
				&PlaceIndex{Result: 2, Type: types.TU32, Base: 1, Index: 1},
				&Store{Place: 2, Value: 1},
			},
			Term: &Return{},
		},
	}}}
}

func TestVerifyFunctionLocalBuffers(t *testing.T) {
	if err := Verify(validStage()); err != nil {
		t.Fatal(err)
	}
	m := validStage()
	m.Functions[0].SourceParams = nil
	if err := Verify(m); err == nil || !strings.Contains(err.Error(), "buffer") {
		t.Fatalf("err=%v", err)
	}
}

func TestVerifyScopeExit(t *testing.T) {
	m := validStage()
	m.Functions[0].Body = &Block{Instrs: []Instr{&Scope{Body: &Block{Term: &ExitScope{}}}}, Term: &Return{}}
	if err := Verify(m); err != nil {
		t.Fatal(err)
	}
	m.Functions[0].Body.Term = &ExitScope{}
	if err := Verify(m); err == nil {
		t.Fatal("accepted exit_scope outside scope")
	}
}

func TestVerifyUsesIntrinsicSignatureRules(t *testing.T) {
	function := &Function{Name: "value", Kind: Helper, Return: types.TI32, Body: &Block{Instrs: []Instr{
		&Const{Result: 1, Type: types.TI32, Raw: "0"},
		&Const{Result: 2, Type: types.TI32, Raw: "1"},
		&Intrinsic{Result: 3, Type: types.TI32, Kind: IntrinsicClamp, Args: []ValueID{1, 1, 2}},
	}, Term: &Return{Value: 3, HasValue: true}}}
	module := &Module{Functions: []*Function{function}}
	if err := Verify(module); err != nil {
		t.Fatal(err)
	}
	for _, instruction := range function.Body.Instrs[:2] {
		instruction.(*Const).Type = types.TF32
	}
	function.Return = types.TF32
	function.Body.Instrs[2].(*Intrinsic).Type = types.TF32
	if err := Verify(module); err == nil || !strings.Contains(err.Error(), "does not accept float32") {
		t.Fatalf("invalid intrinsic domain error = %v", err)
	}
}
