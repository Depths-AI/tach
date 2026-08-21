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
	if err := Verify(module); err != nil {
		t.Fatal(err)
	}
	for _, instruction := range function.Body.Instrs[:2] {
		instruction.(*Const).Type = types.TBool
	}
	function.Return = types.TBool
	function.Body.Instrs[2].(*Intrinsic).Type = types.TBool
	if err := Verify(module); err == nil || !strings.Contains(err.Error(), "does not accept bool") {
		t.Fatalf("invalid intrinsic domain error = %v", err)
	}
}

func TestVerifyMaskIntrinsicShapes(t *testing.T) {
	mask, value := types.Vec(types.TBool, 2), types.Vec(types.TF32, 2)
	selection := &Intrinsic{Result: 8, Type: value, Kind: IntrinsicSelect, Args: []ValueID{3, 6, 6}}
	function := &Function{Name: "mask", Kind: Helper, Return: value, Body: &Block{Instrs: []Instr{
		&Const{Result: 1, Type: types.TBool, Raw: "true"},
		&Const{Result: 2, Type: types.TBool, Raw: "false"},
		&Composite{Result: 3, Type: mask, Values: []ValueID{1, 2}},
		&Const{Result: 4, Type: types.TF32, Raw: "1.0"},
		&Const{Result: 5, Type: types.TF32, Raw: "2.0"},
		&Composite{Result: 6, Type: value, Values: []ValueID{4, 5}},
		&Intrinsic{Result: 7, Type: types.TBool, Kind: IntrinsicAll, Args: []ValueID{3}},
		selection,
	}, Term: &Return{Value: 8, HasValue: true}}}
	module := &Module{Functions: []*Function{function}}
	if err := Verify(module); err != nil {
		t.Fatal(err)
	}
	selection.Args[0] = 6
	if err := Verify(module); err == nil || !strings.Contains(err.Error(), "boolean-vector mask") {
		t.Fatalf("invalid select mask error = %v", err)
	}
}

func TestVerifyAtomicCompareExchangeShape(t *testing.T) {
	atomicArray := types.Runtime(types.AtomicOf(types.TU32))
	operation := &Atomic{Result: 4, Type: types.TU32, Op: AtomicCompareExchange, Place: 2, Expected: 2, Value: 3}
	function := &Function{
		Name:         "claim",
		Kind:         Stage,
		Indices:      []Param{{Name: "i", ID: 1, Type: types.TU32}},
		BufferParams: []BufferParam{{Name: "state", Type: atomicArray, Access: Mutable}},
		SourceParams: []SourceParam{{Name: "state", Kind: SourceBuffer, Buffer: 0}},
		Return:       types.TVoid,
		Body: &Block{Instrs: []Instr{
			&Const{Result: 2, Type: types.TU32, Raw: "0"},
			&Const{Result: 3, Type: types.TU32, Raw: "1"},
			&PlaceRoot{Result: 1, Type: atomicArray, Buffer: 0},
			&PlaceIndex{Result: 2, Type: types.AtomicOf(types.TU32), Base: 1, Index: 1},
			operation,
		}, Term: &Return{}},
	}
	module := &Module{Functions: []*Function{function}}
	if err := Verify(module); err != nil {
		t.Fatal(err)
	}
	operation.Expected = 0
	if err := Verify(module); err == nil || !strings.Contains(err.Error(), "result/operand shape") {
		t.Fatalf("invalid compare-exchange shape error = %v", err)
	}
}

func TestVerifyRejectsVectorRemainder(t *testing.T) {
	vector := types.Vec(types.TU32, 2)
	if err := verifyBinary(&Binary{Op: "%", Type: vector}, vector, vector); err == nil {
		t.Fatal("accepted vector remainder")
	}
}
