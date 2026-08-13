package backend_test

import (
	"testing"

	"tach/src/backend"
	"tach/src/flow"
	"tach/src/ir"
	"tach/src/opt"
	"tach/src/parser"
	"tach/src/sema"
)

func lower(t *testing.T, source string, profile backend.Profile) *backend.Executable {
	t.Helper()
	a, err := parser.Parse("backend.tach", source)
	if err != nil {
		t.Fatal(err)
	}
	m, err := sema.CheckAndLower(a)
	if err != nil {
		t.Fatal(err)
	}
	if err := opt.OptimizeLogical(m); err != nil {
		t.Fatal(err)
	}
	executable, err := backend.Lower(m, profile)
	if err != nil {
		t.Fatal(err)
	}
	return executable
}

func TestPlannerInternalizesSafeRepeatAndPrunesDeadValues(t *testing.T) {
	executable := lower(t, `export function scale[i](data: buffer<float32[]>, used: float32, unused: float32) { data[i] *= used; }`, backend.WebProfile)
	plan := executable.Programs[0]
	if plan.Repeat != backend.RepeatInvocationLoop || len(plan.Steps) != 1 || plan.Steps[0].Parameters[len(plan.Steps[0].Parameters)-1].Kind != flow.ValueRepeat {
		t.Fatalf("plan = %#v", plan)
	}
	kernel := executable.PhysicalKernels[0]
	if len(kernel.Function.Params) != 2 || len(kernel.Parameters.Fields) != 2 {
		t.Fatalf("params = %#v; block = %#v", kernel.Function.Params, kernel.Parameters)
	}
	if _, ok := kernel.Function.Body.Instrs[1].(*ir.Loop); !ok {
		t.Fatalf("body = %#v", kernel.Function.Body)
	}
}

func TestSPIRVPlanKeepsCrossInvocationRepeatAndBarrier(t *testing.T) {
	executable := lower(t, `
function first[i](data: buffer<uint32[]>) { data[i] += uint32(1); }
function second[i](data: buffer<uint32[]>) { data[i] += data[i + uint32(1)]; }
export function graph(data: buffer<uint32[]>) { const count = data.length; run first(data) over count; run second(data) over count; }`, backend.SPIRVProfile)
	plan := executable.Programs[0]
	if plan.Repeat != backend.RepeatProgram || len(plan.Steps) != 3 || plan.Steps[1].Kind != backend.BarrierStepKind || len(plan.RepeatBarrier) == 0 {
		t.Fatalf("plan = %#v", plan)
	}
}
