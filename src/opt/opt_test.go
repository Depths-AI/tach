package opt_test

import (
	"strings"
	"testing"

	"tach/src/ir"
	"tach/src/opt"
	"tach/src/parser"
	"tach/src/sema"
)

func optimized(t *testing.T, name, source string) *ir.Module {
	t.Helper()
	a, err := parser.Parse(name, source)
	if err != nil {
		t.Fatal(err)
	}
	m, err := sema.CheckAndLower(a)
	if err != nil {
		t.Fatal(err)
	}
	if err := opt.OptimizeKernel(m.Kernel); err != nil {
		t.Fatal(err)
	}
	return m.Kernel
}

func TestDeadExpressionTreeIsRemoved(t *testing.T) {
	a, err := parser.Parse("dead.tach", `
@workgroup(1)
export function dead[i](out: buffer<float32[]>) {
  const unused = sin(2.0) * cos(3.0);
  if (i < out.length) { out[i] = 1.0; }
}`)
	if err != nil {
		t.Fatal(err)
	}
	m, err := sema.CheckAndLower(a)
	if err != nil {
		t.Fatal(err)
	}
	before := ir.Dump(m.Kernel)
	if !strings.Contains(before, "intrinsic sin") || !strings.Contains(before, "intrinsic cos") {
		t.Fatal("test setup produced no dead intrinsic tree")
	}
	if err := opt.OptimizeKernel(m.Kernel); err != nil {
		t.Fatal(err)
	}
	after := ir.Dump(m.Kernel)
	if strings.Contains(after, "intrinsic sin") || strings.Contains(after, "intrinsic cos") {
		t.Fatalf("dead intrinsic tree survived optimization:\n%s", after)
	}
}

func TestRepeatedPureValuesAreCanonicalized(t *testing.T) {
	kernel := optimized(t, "common.tach", `
@workgroup(64)
export function common[i](out: buffer<uint32[]>) {
  const a = i + 1;
  const b = i + 1;
  if (i < out.length) { out[i] = a + b; }
}`)
	dump := ir.Dump(kernel)
	if got := strings.Count(dump, " = + %1,"); got != 1 {
		t.Fatalf("repeated index expression was emitted %d times, want one:\n%s", got, dump)
	}
	if got := strings.Count(dump, "const uint32 1"); got != 1 {
		t.Fatalf("constant 1 was emitted %d times, want one:\n%s", got, dump)
	}
}

func TestImmutableMemoryValuesAreCanonicalized(t *testing.T) {
	kernel := optimized(t, "memory.tach", `
type Params = { scale: float32 };
@workgroup(1)
export function memory[i](values: buffer<float32[]>, source: buffer<float32[]>, params: Params) {
  const scale = params.scale + params.scale;
  const length = values.length + values.length;
  const immutable = source[0] + source[0];
  const before = values[0];
  values[0] = scale;
  const after = values[0];
  values[1] = before + after + immutable + float32(length);
}`)
	dump := ir.Dump(kernel)
	if got := strings.Count(dump, "load"); got != 3 {
		t.Fatalf("got %d loads, want one immutable-buffer and two ordered mutable-buffer loads:\n%s", got, dump)
	}
	if got := strings.Count(dump, "array_length"); got != 1 {
		t.Fatalf("array length was emitted %d times, want one:\n%s", got, dump)
	}
}

func TestUnusedPureMemoryAndCallResultsAreRemoved(t *testing.T) {
	kernel := optimized(t, "dead-memory.tach", `
function square(x: float32): float32 { return x * x; }
@workgroup(1)
export function dead[i](out: buffer<float32[]>) {
  const count = out.length;
  const value = out[0];
  square(value);
  out[0] = 1.0;
}`)
	dump := ir.Dump(kernel)
	for _, dead := range []string{"array.length", "load", "call @square"} {
		if strings.Contains(dump, dead) {
			t.Fatalf("dead %s survived:\n%s", dead, dump)
		}
	}
}

func TestLoopBufferValuesAreLazilyPromotedToRegisterCarriers(t *testing.T) {
	kernel := optimized(t, "loop-memory.tach", `
type Params = { dt: float32, count: uint32, steps: uint32 };
@workgroup(64)
export function integrate[i](
  positions: buffer<float32[]>,
  velocities: buffer<float32[]>,
  params: Params,
) {
  if (i < params.count) {
    for (let step = 0; step < params.steps; step++) {
      positions[i] += velocities[i] * params.dt;
    }
  }
}`)
	dump := ir.Dump(kernel)
	outer, ok := kernel.Functions[0].Body.Instrs[len(kernel.Functions[0].Body.Instrs)-1].(*ir.If)
	if !ok {
		t.Fatalf("test setup has no bounds guard:\n%s", dump)
	}
	var loop *ir.Loop
	for _, instruction := range outer.Then.Instrs {
		if candidate, ok := instruction.(*ir.Loop); ok {
			loop = candidate
			break
		}
	}
	if loop == nil {
		t.Fatalf("optimized loop is missing:\n%s", dump)
	}
	lazyLoads := 0
	for _, instruction := range loop.Body.Instrs {
		if _, ok := instruction.(*ir.Load); ok {
			t.Fatalf("loop performs an unconditional memory load:\n%s", dump)
		}
		branch, ok := instruction.(*ir.If)
		if ok && len(branch.Else.Instrs) == 1 {
			if _, ok := branch.Else.Instrs[0].(*ir.Load); ok {
				lazyLoads++
			}
		}
	}
	if lazyLoads != 2 {
		t.Fatalf("got %d lazy first-iteration loads, want position and velocity; dt is a kernel value:\n%s", lazyLoads, dump)
	}
	loopOffset := strings.Index(dump, "loop params=")
	if strings.Count(dump, "store &") != 1 || strings.Index(dump[loopOffset:], "store &") < 0 {
		t.Fatalf("promoted value was not stored exactly once after the loop:\n%s", dump)
	}
	tail := dump[loopOffset:]
	if guard, store := strings.Index(tail, "if %"), strings.Index(tail, "store &"); guard < 0 || store < guard {
		t.Fatalf("zero-trip loop store is not guarded:\n%s", dump)
	}
}

func TestLoopBufferPromotionStopsAtSynchronization(t *testing.T) {
	kernel := optimized(t, "synchronized-loop.tach", `
@workgroup(4)
export function synchronized[i](data: buffer<uint32[]>, steps: uint32) {
  for (let step = 0; step < steps; step++) {
    data[i % 4] += 1;
    workgroupBarrier();
  }
}`)
	dump := ir.Dump(kernel)
	loop := strings.Index(dump, "loop params=")
	if loop < 0 || !strings.Contains(dump[loop:], "store &") {
		t.Fatalf("synchronized memory update was incorrectly promoted:\n%s", dump)
	}
}
