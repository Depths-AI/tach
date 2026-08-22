package parser_test

import (
	"reflect"
	"strings"
	"testing"

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
	program := module.Decls[3].(*parser.FunctionDecl)
	if !program.Exported || len(program.Indices) != 0 || len(program.Body.Stmts) != 3 {
		t.Fatalf("program = %#v", program)
	}
	if _, ok := program.Body.Stmts[1].(*parser.RunStmt); !ok {
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
	parameter := module.Decls[0].(*parser.FunctionDecl).Params[0].Type.(*parser.GenericType)
	vector := parameter.Args[0].(*parser.RuntimeArrayType).Elem.(*parser.VectorType)
	if vector.Lanes != "4" || vector.Elem.(*parser.NamedType).Name != "float32" {
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
	declaration, ok := module.Decls[0].(*parser.ConstDecl)
	if !ok || declaration.Name != "width" || declaration.Type == nil {
		t.Fatalf("module constant = %#v", module.Decls[0])
	}
	function := module.Decls[1].(*parser.FunctionDecl)
	constant, ok := function.Body.Stmts[0].(*parser.ConstStmt)
	if !ok || constant.Name != "area" {
		t.Fatalf("local constant = %#v", function.Body.Stmts[0])
	}
	shared := function.Body.Stmts[1].(*parser.WorkgroupStmt)
	count, ok := shared.Type.(*parser.FixedArrayType).Count.(*parser.IdentExpr)
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
	if len(module.Attrs) != 1 || len(module.Decls[0].(*parser.TypeDecl).Attrs) != 1 || len(module.Decls[1].(*parser.FunctionDecl).Attrs) != 1 {
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
	body := module.Decls[0].(*parser.FunctionDecl).Body.Stmts[0].(*parser.ForStmt).Body
	if _, ok := body.Stmts[0].(*parser.IfStmt).Then.Stmts[0].(*parser.ContinueStmt); !ok {
		t.Fatalf("continue statement = %T", body.Stmts[0])
	}
	if _, ok := body.Stmts[1].(*parser.IfStmt).Then.Stmts[0].(*parser.BreakStmt); !ok {
		t.Fatalf("break statement = %T", body.Stmts[1])
	}
}

func TestSyntaxTreePreservesPrecedencePostfixAndControlFlow(t *testing.T) {
	file, err := parser.Parse("surface.tach", `
import "shared/types";
type State = { history: vec<float32, 4>[8], values: uint32[] };
function exercise(flag: bool, data: buffer<uint32[]>): uint32 {
  let value = flag ? data[0] : 1 + 2 * 3;
  let component = normalize({ x: 1, y: 2 }).x;
  while (value < 8) {
    value += 1;
    if (value == 4) { data[0]++; } else { data[0]--; }
  }
  return value;
}`)
	if err != nil {
		t.Fatal(err)
	}
	if len(file.Imports) != 1 || file.Imports[0].Target != "shared/types" || file.Imports[0].Raw != `"shared/types"` {
		t.Fatalf("imports = %#v", file.Imports)
	}
	state := file.Decls[0].(*parser.TypeDecl)
	history := state.Fields[0].Type.(*parser.FixedArrayType)
	if history.Count.(*parser.NumberExpr).Raw != "8" || history.Elem.(*parser.VectorType).Lanes != "4" {
		t.Fatalf("fixed vector field = %#v", history)
	}
	body := file.Decls[1].(*parser.FunctionDecl).Body
	conditional := body.Stmts[0].(*parser.VarStmt).Value.(*parser.ConditionalExpr)
	addition := conditional.Else.(*parser.BinaryExpr)
	if _, ok := conditional.Then.(*parser.IndexExpr); !ok || addition.Op != "+" || addition.Right.(*parser.BinaryExpr).Op != "*" {
		t.Fatalf("conditional precedence = %#v", conditional)
	}
	member := body.Stmts[1].(*parser.VarStmt).Value.(*parser.MemberExpr)
	if member.Name != "x" {
		t.Fatalf("postfix chain = %#v", member)
	}
	loop := body.Stmts[2].(*parser.WhileStmt)
	branch := loop.Body.Stmts[1].(*parser.IfStmt)
	if _, ok := branch.Then.Stmts[0].(*parser.IncStmt); !ok || branch.Else == nil {
		t.Fatalf("control flow = %#v", loop)
	}
	if _, ok := body.Stmts[3].(*parser.ReturnStmt); !ok {
		t.Fatalf("return = %T", body.Stmts[3])
	}
}

func TestRecoveryCombinesLexicalAndGrammarDiagnostics(t *testing.T) {
	file, diagnostics := parser.ParseRecover("recovery.tach", `# function broken = 1;
export function good[i](out: buffer<uint32[]>) { out[i] = i; }`)
	if len(diagnostics) != 2 || diagnostics[0].Kind != "lexer" || diagnostics[1].Kind != "parser" {
		t.Fatalf("diagnostics = %#v", diagnostics)
	}
	if diagnostics[0].Span.Start.Offset >= diagnostics[1].Span.Start.Offset {
		t.Fatalf("diagnostics are not in source order: %#v", diagnostics)
	}
	if len(file.Decls) != 1 || file.Decls[0].(*parser.FunctionDecl).Name != "good" {
		t.Fatalf("recovered file = %#v", file)
	}
}

func TestKernelDocumentationMustComeFirst(t *testing.T) {
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
			t.Fatal("recoverable parser returned a nil syntax tree")
		}
		if !reflect.DeepEqual(first, second) || !reflect.DeepEqual(firstDiagnostics, secondDiagnostics) {
			t.Fatal("recoverable parse changed across identical runs")
		}
	})
}
