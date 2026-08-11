package abi

import (
	"fmt"
	"reflect"
	"strings"
	"testing"

	"tach/src/ir"
	"tach/src/parser"
	"tach/src/sema"
	"tach/src/types"
)

func parameterModule(t *testing.T, source string) *ir.Module {
	t.Helper()
	parsed, err := parser.Parse("parameters.tach", source)
	if err != nil {
		t.Fatal(err)
	}
	module, err := sema.CheckAndLower(parsed)
	if err != nil {
		t.Fatal(err)
	}
	return module
}

func TestParameterPlanFlattensValuesOncePerKernel(t *testing.T) {
	module := parameterModule(t, `
type Flags = { enabled: bool, scale: float32 };
type Params = { count: uint32, flags: Flags, lane: float32x2 };
export function first[i](out: buffer<uint32[]>, params: Params, bias: int32, ready: bool) {}
export function second[i](out: buffer<uint32[]>, gain: float32) {}
`)
	plan, err := PlanParameters(module)
	if err != nil {
		t.Fatal(err)
	}
	if len(plan.Blocks) != 2 {
		t.Fatalf("blocks = %d, want 2", len(plan.Blocks))
	}
	first := plan.For(module.Function("first"))
	if first == nil || first.Binding != 2 || first.Layout.Size != 32 {
		t.Fatalf("first block = %+v", first)
	}
	wantPaths := [][]string{{"count"}, {"flags", "enabled"}, {"flags", "scale"}, {"lane"}, {}, {}}
	wantOffsets := []uint32{0, 4, 8, 16, 24, 28}
	for index, field := range first.Fields {
		if !reflect.DeepEqual(field.Path, wantPaths[index]) || field.Offset != wantOffsets[index] {
			t.Fatalf("field %d = path %v offset %d, want %v at %d", index, field.Path, field.Offset, wantPaths[index], wantOffsets[index])
		}
	}
	if first.Fields[1].Logical != types.TBool || first.Fields[1].Physical != types.TU32 || first.Fields[5].Physical != types.TU32 {
		t.Fatal("bool parameters were not represented as physical uint32 fields")
	}
	second := plan.For(module.Function("second"))
	if second == nil || second.Binding != 3 || second.Layout.Size != 16 {
		t.Fatalf("second block = %+v", second)
	}
}

func TestParameterPlanOmitsBufferOnlyKernels(t *testing.T) {
	module := parameterModule(t, `export function clear[i](out: buffer<uint32[]>) {}`)
	plan, err := PlanParameters(module)
	if err != nil {
		t.Fatal(err)
	}
	if len(plan.Blocks) != 0 || plan.For(module.Function("clear")) != nil {
		t.Fatalf("buffer-only plan = %+v", plan.Blocks)
	}
}

func TestParameterPlanEnforcesPortableBlockLimit(t *testing.T) {
	large := &types.Type{Kind: types.Struct, Name: "Large"}
	for index := 0; index <= maxParameterBytes/4; index++ {
		large.Fields = append(large.Fields, types.Field{Name: fmt.Sprintf("f%d", index), Type: types.TU32})
	}
	function := &ir.Function{Name: "large", Compute: true, Params: []ir.Param{{Name: "value", ID: 1, Type: large}}}
	_, err := PlanParameters(&ir.Module{Functions: []*ir.Function{function}})
	if err == nil || !strings.Contains(err.Error(), "portable limit is 16384") {
		t.Fatalf("PlanParameters() error = %v, want portable-limit rejection", err)
	}
}
