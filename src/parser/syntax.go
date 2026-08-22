package parser

import "tach/src/foundation"

type Node interface{ GetSpan() foundation.Span }

type File struct {
	Path    string
	Attrs   []Attribute
	Imports []Import
	Decls   []Decl
}

type Import struct {
	Target string
	Raw    string
	Span   foundation.Span
}
type Decl interface {
	Node
	declNode()
}

type Attribute struct {
	Name string
	Args []Expr
	Span foundation.Span
}

func (a Attribute) GetSpan() foundation.Span { return a.Span }

type TypeDecl struct {
	Name   string
	Attrs  []Attribute
	Fields []Field
	Span   foundation.Span
}

func (*TypeDecl) declNode()                  {}
func (d *TypeDecl) GetSpan() foundation.Span { return d.Span }

type ConstDecl struct {
	Name  string
	Type  TypeExpr
	Value Expr
	Span  foundation.Span
}

func (*ConstDecl) declNode()                  {}
func (d *ConstDecl) GetSpan() foundation.Span { return d.Span }

type Field struct {
	Name string
	Type TypeExpr
	Span foundation.Span
}

type FunctionDecl struct {
	Name     string
	Exported bool
	Attrs    []Attribute
	Indices  []Index
	Params   []Param
	Return   TypeExpr
	Body     *BlockStmt
	Span     foundation.Span
}

func (*FunctionDecl) declNode()                  {}
func (d *FunctionDecl) GetSpan() foundation.Span { return d.Span }

type Index struct {
	Name string
	Span foundation.Span
}

type Param struct {
	Name string
	Type TypeExpr
	Span foundation.Span
}

type TypeExpr interface {
	Node
	typeNode()
}
type NamedType struct {
	Name string
	Span foundation.Span
}

func (*NamedType) typeNode()                  {}
func (t *NamedType) GetSpan() foundation.Span { return t.Span }

type RuntimeArrayType struct {
	Elem TypeExpr
	Span foundation.Span
}

func (*RuntimeArrayType) typeNode()                  {}
func (t *RuntimeArrayType) GetSpan() foundation.Span { return t.Span }

type FixedArrayType struct {
	Elem  TypeExpr
	Count Expr
	Span  foundation.Span
}

func (*FixedArrayType) typeNode()                  {}
func (t *FixedArrayType) GetSpan() foundation.Span { return t.Span }

type VectorType struct {
	Elem  TypeExpr
	Lanes string
	Span  foundation.Span
}

func (*VectorType) typeNode()                  {}
func (t *VectorType) GetSpan() foundation.Span { return t.Span }

type GenericType struct {
	Name string
	Args []TypeExpr
	Span foundation.Span
}

func (*GenericType) typeNode()                  {}
func (t *GenericType) GetSpan() foundation.Span { return t.Span }

type Stmt interface {
	Node
	stmtNode()
}
type BlockStmt struct {
	Stmts []Stmt
	Span  foundation.Span
}

func (*BlockStmt) stmtNode()                  {}
func (s *BlockStmt) GetSpan() foundation.Span { return s.Span }

type VarStmt struct {
	Name  string
	Type  TypeExpr
	Value Expr
	Span  foundation.Span
}

func (*VarStmt) stmtNode()                  {}
func (s *VarStmt) GetSpan() foundation.Span { return s.Span }

type ConstStmt struct {
	Name  string
	Type  TypeExpr
	Value Expr
	Span  foundation.Span
}

func (*ConstStmt) stmtNode()                  {}
func (s *ConstStmt) GetSpan() foundation.Span { return s.Span }

type WorkgroupStmt struct {
	Name string
	Type TypeExpr
	Span foundation.Span
}

func (*WorkgroupStmt) stmtNode()                  {}
func (s *WorkgroupStmt) GetSpan() foundation.Span { return s.Span }

type AssignStmt struct {
	Target Expr
	Op     string
	Value  Expr
	Span   foundation.Span
}

func (*AssignStmt) stmtNode()                  {}
func (s *AssignStmt) GetSpan() foundation.Span { return s.Span }

type IncStmt struct {
	Target Expr
	Delta  int
	Span   foundation.Span
}

func (*IncStmt) stmtNode()                  {}
func (s *IncStmt) GetSpan() foundation.Span { return s.Span }

type ExprStmt struct {
	Expr Expr
	Span foundation.Span
}

func (*ExprStmt) stmtNode()                  {}
func (s *ExprStmt) GetSpan() foundation.Span { return s.Span }

type IfStmt struct {
	Cond Expr
	Then *BlockStmt
	Else *BlockStmt
	Span foundation.Span
}

func (*IfStmt) stmtNode()                  {}
func (s *IfStmt) GetSpan() foundation.Span { return s.Span }

type WhileStmt struct {
	Cond Expr
	Body *BlockStmt
	Span foundation.Span
}

func (*WhileStmt) stmtNode()                  {}
func (s *WhileStmt) GetSpan() foundation.Span { return s.Span }

type ForStmt struct {
	Init *VarStmt
	Cond Expr
	Post Stmt
	Body *BlockStmt
	Span foundation.Span
}

func (*ForStmt) stmtNode()                  {}
func (s *ForStmt) GetSpan() foundation.Span { return s.Span }

type BreakStmt struct{ Span foundation.Span }

func (*BreakStmt) stmtNode()                  {}
func (s *BreakStmt) GetSpan() foundation.Span { return s.Span }

type ContinueStmt struct{ Span foundation.Span }

func (*ContinueStmt) stmtNode()                  {}
func (s *ContinueStmt) GetSpan() foundation.Span { return s.Span }

type ReturnStmt struct {
	Value Expr
	Span  foundation.Span
}

func (*ReturnStmt) stmtNode()                  {}
func (s *ReturnStmt) GetSpan() foundation.Span { return s.Span }

type RunStmt struct {
	Stage  string
	Args   []Expr
	Domain Domain
	Span   foundation.Span
}

func (*RunStmt) stmtNode()                  {}
func (s *RunStmt) GetSpan() foundation.Span { return s.Span }

type Domain struct {
	Axes []Expr
	Span foundation.Span
}

func (d Domain) GetSpan() foundation.Span { return d.Span }

type Expr interface {
	Node
	exprNode()
}
type IdentExpr struct {
	Name string
	Span foundation.Span
}

func (*IdentExpr) exprNode()                  {}
func (e *IdentExpr) GetSpan() foundation.Span { return e.Span }

type NumberExpr struct {
	Raw  string
	Span foundation.Span
}

func (*NumberExpr) exprNode()                  {}
func (e *NumberExpr) GetSpan() foundation.Span { return e.Span }

type StringExpr struct {
	Value string
	Span  foundation.Span
}

func (*StringExpr) exprNode()                  {}
func (e *StringExpr) GetSpan() foundation.Span { return e.Span }

type BoolExpr struct {
	Value bool
	Span  foundation.Span
}

func (*BoolExpr) exprNode()                  {}
func (e *BoolExpr) GetSpan() foundation.Span { return e.Span }

type UnaryExpr struct {
	Op   string
	X    Expr
	Span foundation.Span
}

func (*UnaryExpr) exprNode()                  {}
func (e *UnaryExpr) GetSpan() foundation.Span { return e.Span }

type BinaryExpr struct {
	Op          string
	Left, Right Expr
	Span        foundation.Span
}

func (*BinaryExpr) exprNode()                  {}
func (e *BinaryExpr) GetSpan() foundation.Span { return e.Span }

type ConditionalExpr struct {
	Cond Expr
	Then Expr
	Else Expr
	Span foundation.Span
}

func (*ConditionalExpr) exprNode()                  {}
func (e *ConditionalExpr) GetSpan() foundation.Span { return e.Span }

type CallExpr struct {
	Callee Expr
	Args   []Expr
	Span   foundation.Span
}

func (*CallExpr) exprNode()                  {}
func (e *CallExpr) GetSpan() foundation.Span { return e.Span }

type MemberExpr struct {
	Base Expr
	Name string
	Span foundation.Span
}

func (*MemberExpr) exprNode()                  {}
func (e *MemberExpr) GetSpan() foundation.Span { return e.Span }

type IndexExpr struct {
	Base, Index Expr
	Span        foundation.Span
}

func (*IndexExpr) exprNode()                  {}
func (e *IndexExpr) GetSpan() foundation.Span { return e.Span }

type StructLiteralExpr struct {
	Fields []LiteralField
	Span   foundation.Span
}

func (*StructLiteralExpr) exprNode()                  {}
func (e *StructLiteralExpr) GetSpan() foundation.Span { return e.Span }

type TransientExpr struct {
	Elem  TypeExpr
	Count Expr
	Span  foundation.Span
}

func (*TransientExpr) exprNode()                  {}
func (e *TransientExpr) GetSpan() foundation.Span { return e.Span }

type LiteralField struct {
	Name  string
	Value Expr
	Span  foundation.Span
}
