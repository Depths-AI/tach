package parser_test

import (
	"reflect"
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
  let scratch = transient<uint32>(count);
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

func TestVectorTypeUsesOneStructuralSpelling(t *testing.T) {
	module, err := parser.Parse("vectors.tach", `
function stage[i](out: buffer<vec<float32, 4>[]>) { out[i] = vec(1, 2, 3, 4); }
export function vectors() {
  let out = transient<vec<float32, 4>>(4);
  run stage(out) over 4;
}`)
	if err != nil {
		t.Fatal(err)
	}
	parameter := module.Decls[0].(*ast.FunctionDecl).Params[0].Type.(*ast.GenericType)
	vector := parameter.Args[0].(*ast.RuntimeArrayType).Elem.(*ast.VectorType)
	if vector.Lanes != "4" || vector.Elem.(*ast.NamedType).Name != "float32" {
		t.Fatalf("vector type = %#v", vector)
	}
}

func TestCompileTimeConstantsHaveModuleAndLexicalForms(t *testing.T) {
	module, err := parser.Parse("constants.tach", `
const width: uint32 = 4 * 4;
export function constants[i](out: buffer<uint32[]>) {
  const area = width * width;
  let scratch: shared<uint32[area]>;
  out[i] = area;
}`)
	if err != nil {
		t.Fatal(err)
	}
	declaration, ok := module.Decls[0].(*ast.ConstDecl)
	if !ok || declaration.Name != "width" || declaration.Type == nil {
		t.Fatalf("module constant = %#v", module.Decls[0])
	}
	function := module.Decls[1].(*ast.FunctionDecl)
	constant, ok := function.Body.Stmts[0].(*ast.ConstStmt)
	if !ok || constant.Name != "area" {
		t.Fatalf("local constant = %#v", function.Body.Stmts[0])
	}
	shared := function.Body.Stmts[1].(*ast.WorkgroupStmt)
	count, ok := shared.Type.(*ast.FixedArrayType).Count.(*ast.IdentExpr)
	if !ok || count.Name != "area" {
		t.Fatalf("fixed-array count = %#v", shared.Type)
	}
	if _, err := parser.Parse("invalid.tach", `@workgroup(1) const width = 4;`); err == nil || !strings.Contains(err.Error(), "attributes are invalid on constants") {
		t.Fatalf("constant attribute error = %v", err)
	}
}

func TestDocumentationAttributesAttachToTheirContexts(t *testing.T) {
	module, err := parser.Parse("docs.tach", `
@docs(title("Particles"), summary("Simulation kernels."));
@docs(summary("Position and velocity."), field(position, "World position."))
type Particle = { position: vec<float32, 4> };
@docs(summary("Advance particles."), coordinate(i, "Particle index."), param(particles, "State."))
export function step[i](particles: buffer<Particle[]>) { }
`)
	if err != nil {
		t.Fatal(err)
	}
	if len(module.Attrs) != 1 || len(module.Decls[0].(*ast.TypeDecl).Attrs) != 1 || len(module.Decls[1].(*ast.FunctionDecl).Attrs) != 1 {
		t.Fatalf("documentation did not attach: %#v", module)
	}
}

func TestBreakAndContinueStatements(t *testing.T) {
	module, err := parser.Parse("control.tach", `
export function control[i](out: buffer<uint32[]>) {
  for (let step = 0; step < 8; step++) {
    if (step == 2) { continue; }
    if (step == 6) { break; }
  }
}`)
	if err != nil {
		t.Fatal(err)
	}
	body := module.Decls[0].(*ast.FunctionDecl).Body.Stmts[0].(*ast.ForStmt).Body
	if _, ok := body.Stmts[0].(*ast.IfStmt).Then.Stmts[0].(*ast.ContinueStmt); !ok {
		t.Fatalf("continue statement = %T", body.Stmts[0])
	}
	if _, ok := body.Stmts[1].(*ast.IfStmt).Then.Stmts[0].(*ast.BreakStmt); !ok {
		t.Fatalf("break statement = %T", body.Stmts[1])
	}
}

func TestModuleDocumentationMustComeFirst(t *testing.T) {
	_, err := parser.Parse("docs.tach", `type X = { value: uint32 }; @docs(summary("Too late."));`)
	if err == nil || !strings.Contains(err.Error(), "must precede declarations") {
		t.Fatalf("error = %v", err)
	}
	if _, err := parser.Parse("comments.tach", `/* removed */ export function k[i](out: buffer<uint32[]>) {}`); err == nil {
		t.Fatal("accepted a block comment")
	}
}

func FuzzParserRecoveryIsTotalAndDeterministic(f *testing.F) {
	for _, seed := range []string{
		`export function fill[i](out: buffer<uint32[]>) { if (i < out.length) { out[i] = i; } }`,
		`@docs(summary("x")) function broken( { import "late/path"; }`,
		"function x() { const value = \"unterminated\nreturn; }",
	} {
		f.Add(seed)
	}
	f.Fuzz(func(t *testing.T, input string) {
		// DECISION: bound one fuzz case at 64 KiB; project tests own larger-file pressure, and this cap can rise with parser limits.
		if len(input) > 64<<10 {
			t.Skip()
		}
		first, firstDiagnostics := parser.ParseRecover("fuzz.tach", input)
		second, secondDiagnostics := parser.ParseRecover("fuzz.tach", input)
		if first == nil || second == nil {
			t.Fatal("recoverable parser returned a nil module")
		}
		if !reflect.DeepEqual(first, second) || !reflect.DeepEqual(firstDiagnostics, secondDiagnostics) {
			t.Fatal("recoverable parse changed across identical runs")
		}
	})
}
