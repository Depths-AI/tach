package parser_test

import (
	"strings"
	"testing"

	"tach/src/ast"
	"tach/src/parser"
)

func TestFunctionDeclarationFormsAndProgramRun(t *testing.T) {
	module, err := parser.Parse("forms.tach", `
function helper(x: uint32): uint32 { return x; }
function stage[i](out: buffer<uint32[]>) { out[i] = helper(i); }
export function kernel[i](out: buffer<uint32[]>) { out[i] = i; }
export function program(out: buffer<uint32[]>, count: uint32) {
  const scratch = transient<uint32>(count);
  run stage(scratch) over count;
  run stage(out) over count;
}`)
	if err != nil {
		t.Fatal(err)
	}
	if len(module.Decls) != 4 {
		t.Fatalf("declarations = %d", len(module.Decls))
	}
	program := module.Decls[3].(*ast.FunctionDecl)
	if !program.Exported || len(program.Indices) != 0 || len(program.Body.Stmts) != 3 {
		t.Fatalf("program = %#v", program)
	}
	if _, ok := program.Body.Stmts[1].(*ast.RunStmt); !ok {
		t.Fatalf("statement = %T", program.Body.Stmts[1])
	}
}

func TestEmptyIndexListIsRejected(t *testing.T) {
	_, err := parser.Parse("empty.tach", `export function empty[](out: buffer<uint32[]>) {}`)
	if err == nil || !strings.Contains(err.Error(), "requires 1 to 3") {
		t.Fatalf("error = %v", err)
	}
}
