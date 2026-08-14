package ast

import "tach/src/source"

type Node interface{ GetSpan() source.Span }

type Module struct {
	File    string
	Attrs   []Attribute
	Imports []Import
	Decls   []Decl
}

type Import struct {
	Target string
	Raw    string
	Span   source.Span
}
type Decl interface {
	Node
	declNode()
}

type Attribute struct {
	Name string
	Args []Expr
	Span source.Span
}

func (a Attribute) GetSpan() source.Span { return a.Span }

type TypeDecl struct {
	Name   string
	Attrs  []Attribute
	Fields []Field
	Span   source.Span
}

func (*TypeDecl) declNode()              {}
func (d *TypeDecl) GetSpan() source.Span { return d.Span }

type Field struct {
	Name string
	Type TypeExpr
	Span source.Span
}

type FunctionDecl struct {
	Name     string
	Exported bool
	Attrs    []Attribute
	Indices  []Index
	Params   []Param
	Return   TypeExpr
	Body     *BlockStmt
	Span     source.Span
}

func (*FunctionDecl) declNode()              {}
func (d *FunctionDecl) GetSpan() source.Span { return d.Span }

type Index struct {
	Name string
	Span source.Span
}

type Param struct {
	Name string
	Type TypeExpr
	Span source.Span
}

type TypeExpr interface {
	Node
	typeNode()
}
type NamedType struct {
	Name string
	Span source.Span
}

func (*NamedType) typeNode()              {}
func (t *NamedType) GetSpan() source.Span { return t.Span }

type RuntimeArrayType struct {
	Elem TypeExpr
	Span source.Span
}

func (*RuntimeArrayType) typeNode()              {}
func (t *RuntimeArrayType) GetSpan() source.Span { return t.Span }

type FixedArrayType struct {
	Elem  TypeExpr
	Count string
	Span  source.Span
}

func (*FixedArrayType) typeNode()              {}
func (t *FixedArrayType) GetSpan() source.Span { return t.Span }

type GenericType struct {
	Name string
	Args []TypeExpr
	Span source.Span
}

func (*GenericType) typeNode()              {}
func (t *GenericType) GetSpan() source.Span { return t.Span }

type Stmt interface {
	Node
	stmtNode()
}
type BlockStmt struct {
	Stmts []Stmt
	Span  source.Span
}

func (*BlockStmt) stmtNode()              {}
func (s *BlockStmt) GetSpan() source.Span { return s.Span }

type VarStmt struct {
	Mutable bool
	Name    string
	Type    TypeExpr
	Value   Expr
	Span    source.Span
}

func (*VarStmt) stmtNode()              {}
func (s *VarStmt) GetSpan() source.Span { return s.Span }

type WorkgroupStmt struct {
	Name string
	Type TypeExpr
	Span source.Span
}

func (*WorkgroupStmt) stmtNode()              {}
func (s *WorkgroupStmt) GetSpan() source.Span { return s.Span }

type AssignStmt struct {
	Target Expr
	Op     string
	Value  Expr
	Span   source.Span
}

func (*AssignStmt) stmtNode()              {}
func (s *AssignStmt) GetSpan() source.Span { return s.Span }

type IncStmt struct {
	Target Expr
	Delta  int
	Span   source.Span
}

func (*IncStmt) stmtNode()              {}
func (s *IncStmt) GetSpan() source.Span { return s.Span }

type ExprStmt struct {
	Expr Expr
	Span source.Span
}

func (*ExprStmt) stmtNode()              {}
func (s *ExprStmt) GetSpan() source.Span { return s.Span }

type IfStmt struct {
	Cond Expr
	Then *BlockStmt
	Else *BlockStmt
	Span source.Span
}

func (*IfStmt) stmtNode()              {}
func (s *IfStmt) GetSpan() source.Span { return s.Span }

type WhileStmt struct {
	Cond Expr
	Body *BlockStmt
	Span source.Span
}

func (*WhileStmt) stmtNode()              {}
func (s *WhileStmt) GetSpan() source.Span { return s.Span }

type ForStmt struct {
	Init *VarStmt
	Cond Expr
	Post Stmt
	Body *BlockStmt
	Span source.Span
}

func (*ForStmt) stmtNode()              {}
func (s *ForStmt) GetSpan() source.Span { return s.Span }

type ReturnStmt struct {
	Value Expr
	Span  source.Span
}

func (*ReturnStmt) stmtNode()              {}
func (s *ReturnStmt) GetSpan() source.Span { return s.Span }

type RunStmt struct {
	Stage  string
	Args   []Expr
	Domain Domain
	Span   source.Span
}

func (*RunStmt) stmtNode()              {}
func (s *RunStmt) GetSpan() source.Span { return s.Span }

type Domain struct {
	Axes []Expr
	Span source.Span
}

func (d Domain) GetSpan() source.Span { return d.Span }

type Expr interface {
	Node
	exprNode()
}
type IdentExpr struct {
	Name string
	Span source.Span
}

func (*IdentExpr) exprNode()              {}
func (e *IdentExpr) GetSpan() source.Span { return e.Span }

type NumberExpr struct {
	Raw  string
	Span source.Span
}

func (*NumberExpr) exprNode()              {}
func (e *NumberExpr) GetSpan() source.Span { return e.Span }

type StringExpr struct {
	Value string
	Span  source.Span
}

func (*StringExpr) exprNode()              {}
func (e *StringExpr) GetSpan() source.Span { return e.Span }

type BoolExpr struct {
	Value bool
	Span  source.Span
}

func (*BoolExpr) exprNode()              {}
func (e *BoolExpr) GetSpan() source.Span { return e.Span }

type UnaryExpr struct {
	Op   string
	X    Expr
	Span source.Span
}

func (*UnaryExpr) exprNode()              {}
func (e *UnaryExpr) GetSpan() source.Span { return e.Span }

type BinaryExpr struct {
	Op          string
	Left, Right Expr
	Span        source.Span
}

func (*BinaryExpr) exprNode()              {}
func (e *BinaryExpr) GetSpan() source.Span { return e.Span }

type ConditionalExpr struct {
	Cond Expr
	Then Expr
	Else Expr
	Span source.Span
}

func (*ConditionalExpr) exprNode()              {}
func (e *ConditionalExpr) GetSpan() source.Span { return e.Span }

type CallExpr struct {
	Callee Expr
	Args   []Expr
	Span   source.Span
}

func (*CallExpr) exprNode()              {}
func (e *CallExpr) GetSpan() source.Span { return e.Span }

type MemberExpr struct {
	Base Expr
	Name string
	Span source.Span
}

func (*MemberExpr) exprNode()              {}
func (e *MemberExpr) GetSpan() source.Span { return e.Span }

type IndexExpr struct {
	Base, Index Expr
	Span        source.Span
}

func (*IndexExpr) exprNode()              {}
func (e *IndexExpr) GetSpan() source.Span { return e.Span }

type StructLiteralExpr struct {
	Fields []LiteralField
	Span   source.Span
}

func (*StructLiteralExpr) exprNode()              {}
func (e *StructLiteralExpr) GetSpan() source.Span { return e.Span }

type TransientExpr struct {
	Elem  TypeExpr
	Count Expr
	Span  source.Span
}

func (*TransientExpr) exprNode()              {}
func (e *TransientExpr) GetSpan() source.Span { return e.Span }

type LiteralField struct {
	Name  string
	Value Expr
	Span  source.Span
}
