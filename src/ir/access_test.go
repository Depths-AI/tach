package ir_test

import (
	"strings"
	"testing"

	"tach/src/ir"
	"tach/src/parser"
	"tach/src/sema"
)

func stage(t *testing.T, body string, indices ...string) *ir.Function {
	t.Helper()
	prefix := ""
	coordinates := "i"
	if len(indices) > 0 {
		coordinates = strings.Join(indices, ", ")
	}
	if strings.Contains(body, "Barrier") || strings.Contains(body, "shared<") {
		prefix = "@workgroup(1) "
	}
	a, err := parser.Parse("access.tach", prefix+"export function access["+coordinates+"](data: buffer<uint32[]>) { "+body+" }")
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
	if !identity.CoordinateWrite || len(identity.Accesses) != 1 || identity.Accesses[0].Indices[0].Coefficient != [3]int64{1, 0, 0} {
		t.Fatalf("identity = %#v", identity)
	}
	offset := ir.AnalyzeAccess(stage(t, `let value = data[i + uint32(2)]; data[i] = value;`)).Buffers[0]
	if got := offset.Accesses[0].Indices[0].Constant; got != 2 {
		t.Fatalf("offset = %d", got)
	}
	opaque := ir.AnalyzeAccess(stage(t, `let value = data[data[i]]; data[i] = value;`)).Buffers[0]
	if opaque.Accesses[1].Indices[0].Exact {
		t.Fatalf("opaque = %#v", opaque)
	}
}

func TestAccessSummaryTracksEffectsAndCoordinateWrite(t *testing.T) {
	guarded := ir.AnalyzeAccess(stage(t, `if (i < data.length) { data[i] = i; }`))
	if !guarded.Buffers[0].CoordinateWrite || !guarded.Effects.Memory {
		t.Fatalf("guarded = %#v", guarded)
	}
	partial := ir.AnalyzeAccess(stage(t, `if (i < 4) { data[i] = i; }`))
	if partial.Buffers[0].CoordinateWrite {
		t.Fatalf("fixed guard proved coordinate coverage = %#v", partial)
	}
	twoDimensional := ir.AnalyzeAccess(stage(t, `data[x] = x;`, "x", "y"))
	if twoDimensional.Buffers[0].CoordinateWrite {
		t.Fatalf("2D launch proved 1D coordinate coverage = %#v", twoDimensional)
	}
	earlyReturn := ir.AnalyzeAccess(stage(t, `if (i == 0) { return; } data[i] = i;`))
	if earlyReturn.Buffers[0].CoordinateWrite {
		t.Fatalf("early return proved coordinate coverage = %#v", earlyReturn)
	}
	barrier := ir.AnalyzeAccess(stage(t, `bufferBarrier(); data[i] = i;`))
	if !barrier.Effects.Barrier || !barrier.Effects.Workgroup {
		t.Fatalf("barrier = %#v", barrier)
	}
	shared := ir.AnalyzeAccess(stage(t, `let scratch: shared<uint32[4]>; scratch[0] = 7; let value = scratch[0];`))
	if !shared.Effects.Workgroup || shared.Effects.Memory || shared.Buffers[0].Read || shared.Buffers[0].Write || shared.Buffers[0].CoordinateWrite || len(shared.Buffers[0].Accesses) != 0 {
		t.Fatalf("shared memory polluted storage buffer summary = %#v", shared)
	}
}
