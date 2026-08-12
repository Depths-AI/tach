package ir_test

import (
	"strings"
	"testing"

	"tach/src/ir"
	"tach/src/parser"
	"tach/src/sema"
)

func stage(t *testing.T, body string) *ir.Function {
	t.Helper()
	prefix := ""
	if strings.Contains(body, "Barrier") {
		prefix = "@workgroup(1) "
	}
	a, err := parser.Parse("access.tach", prefix+"export function access[i](data: buffer<uint32[]>) { "+body+" }")
	if err != nil {
		t.Fatal(err)
	}
	m, err := sema.CheckAndLower(a)
	if err != nil {
		t.Fatal(err)
	}
	return m.Kernel.Function("access")
}

func TestAccessSummaryRecognizesIdentityOffsetAndOpaque(t *testing.T) {
	identity := ir.AnalyzeAccess(stage(t, `data[i] = i;`)).Buffers[0]
	if !identity.CompleteWrite || len(identity.Accesses) != 1 || identity.Accesses[0].Indices[0].Coefficient != [3]int64{1, 0, 0} {
		t.Fatalf("identity = %#v", identity)
	}
	offset := ir.AnalyzeAccess(stage(t, `const value = data[i + uint32(2)]; data[i] = value;`)).Buffers[0]
	if got := offset.Accesses[0].Indices[0].Constant; got != 2 {
		t.Fatalf("offset = %d", got)
	}
	opaque := ir.AnalyzeAccess(stage(t, `const value = data[data[i]]; data[i] = value;`)).Buffers[0]
	if opaque.Accesses[1].Indices[0].Exact {
		t.Fatalf("opaque = %#v", opaque)
	}
}

func TestAccessSummaryTracksEffectsAndGuardedCompleteWrite(t *testing.T) {
	guarded := ir.AnalyzeAccess(stage(t, `if (i < data.length) { data[i] = i; }`))
	if !guarded.Buffers[0].CompleteWrite || !guarded.Effects.Memory {
		t.Fatalf("guarded = %#v", guarded)
	}
	barrier := ir.AnalyzeAccess(stage(t, `bufferBarrier(); data[i] = i;`))
	if !barrier.Effects.Barrier || !barrier.Effects.Workgroup {
		t.Fatalf("barrier = %#v", barrier)
	}
}
