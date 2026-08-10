package opt_test

import (
	"strings"
	"testing"

	"pine/internal/ir"
	"pine/internal/opt"
	"pine/internal/parser"
	"pine/internal/sema"
)

func TestDeadExpressionTreeIsRemoved(t *testing.T) {
	a, err := parser.Parse("dead.pine", `
@workgroupSize(1)
export compute dead(out: storage<f32[], read_write>) {
  const unused = sin(2.0) * cos(3.0);
  if (globalId.x < out.length) { out[globalId.x] = 1.0; }
}`)
	if err != nil {
		t.Fatal(err)
	}
	m, err := sema.CheckAndLower(a)
	if err != nil {
		t.Fatal(err)
	}
	before := ir.Dump(m)
	if !strings.Contains(before, "intrinsic sin") || !strings.Contains(before, "intrinsic cos") {
		t.Fatal("test setup produced no dead intrinsic tree")
	}
	if err := opt.Run(m); err != nil {
		t.Fatal(err)
	}
	after := ir.Dump(m)
	if strings.Contains(after, "intrinsic sin") || strings.Contains(after, "intrinsic cos") {
		t.Fatalf("dead intrinsic tree survived optimization:\n%s", after)
	}
}
