package parser

import (
	"fmt"

	"pine/internal/ast"
	"pine/internal/lexer"
	"pine/internal/source"
)

type Parser struct {
	toks []lexer.Token
	i    int
	file string
}

func Parse(file, src string) (*ast.Module, error) {
	toks, err := lexer.Lex(file, src)
	if err != nil {
		return nil, err
	}
	p := &Parser{toks: toks, file: file}
	return p.module()
}
func (p *Parser) cur() lexer.Token     { return p.toks[p.i] }
func (p *Parser) at(k lexer.Kind) bool { return p.cur().Kind == k }
func (p *Parser) text(s string) bool   { return p.cur().Kind == lexer.Ident && p.cur().Text == s }
func (p *Parser) take() lexer.Token {
	t := p.cur()
	if p.i < len(p.toks)-1 {
		p.i++
	}
	return t
}
func (p *Parser) err(t lexer.Token, f string, a ...any) error {
	return &source.Error{Span: t.Span, Message: fmt.Sprintf(f, a...)}
}
func (p *Parser) expect(k lexer.Kind) (lexer.Token, error) {
	if !p.at(k) {
		return lexer.Token{}, p.err(p.cur(), "expected %s, found %q", k, p.cur().Text)
	}
	return p.take(), nil
}
func (p *Parser) expectText(s string) (lexer.Token, error) {
	if !p.text(s) {
		return lexer.Token{}, p.err(p.cur(), "expected %q, found %q", s, p.cur().Text)
	}
	return p.take(), nil
}

func join(a, b source.Span) source.Span { a.End = b.End; return a }

func (p *Parser) module() (*ast.Module, error) {
	m := &ast.Module{File: p.file}
	for !p.at(lexer.EOF) {
		attrs, err := p.attrs()
		if err != nil {
			return nil, err
		}
		switch {
		case p.text("type"):
			if len(attrs) > 0 {
				return nil, p.err(p.cur(), "attributes are not supported on type declarations")
			}
			d, err := p.typeDecl()
			if err != nil {
				return nil, err
			}
			m.Decls = append(m.Decls, d)
		case p.text("fn"):
			if len(attrs) > 0 {
				return nil, p.err(p.cur(), "attributes are not supported on helper functions")
			}
			d, err := p.funcDecl()
			if err != nil {
				return nil, err
			}
			m.Decls = append(m.Decls, d)
		case p.text("export"):
			d, err := p.computeDecl(attrs)
			if err != nil {
				return nil, err
			}
			m.Decls = append(m.Decls, d)
		default:
			return nil, p.err(p.cur(), "expected type, fn, or export compute declaration")
		}
	}
	return m, nil
}
func (p *Parser) attrs() ([]ast.Attribute, error) {
	var out []ast.Attribute
	for p.at(lexer.At) {
		start := p.take().Span
		name, err := p.expect(lexer.Ident)
		if err != nil {
			return nil, err
		}
		a := ast.Attribute{Name: name.Text, Span: join(start, name.Span)}
		if p.at(lexer.LParen) {
			p.take()
			if !p.at(lexer.RParen) {
				for {
					e, err := p.expr(0)
					if err != nil {
						return nil, err
					}
					a.Args = append(a.Args, e)
					if !p.at(lexer.Comma) {
						break
					}
					p.take()
				}
			}
			r, err := p.expect(lexer.RParen)
			if err != nil {
				return nil, err
			}
			a.Span = join(start, r.Span)
		}
		out = append(out, a)
	}
	return out, nil
}
func (p *Parser) typeDecl() (*ast.TypeDecl, error) {
	start := p.take().Span
	name, err := p.expect(lexer.Ident)
	if err != nil {
		return nil, err
	}
	if _, err = p.expect(lexer.Assign); err != nil {
		return nil, err
	}
	if _, err = p.expect(lexer.LBrace); err != nil {
		return nil, err
	}
	d := &ast.TypeDecl{Name: name.Text}
	for !p.at(lexer.RBrace) {
		fn, err := p.expect(lexer.Ident)
		if err != nil {
			return nil, err
		}
		if _, err = p.expect(lexer.Colon); err != nil {
			return nil, err
		}
		ty, err := p.typeExpr()
		if err != nil {
			return nil, err
		}
		f := ast.Field{Name: fn.Text, Type: ty, Span: join(fn.Span, ty.GetSpan())}
		d.Fields = append(d.Fields, f)
		if p.at(lexer.Comma) || p.at(lexer.Semicolon) {
			p.take()
		} else if !p.at(lexer.RBrace) {
			return nil, p.err(p.cur(), "expected ',' or '}' after field")
		}
	}
	r := p.take()
	if p.at(lexer.Semicolon) {
		r = p.take()
	}
	d.Span = join(start, r.Span)
	return d, nil
}
func (p *Parser) params() ([]ast.Param, error) {
	if _, err := p.expect(lexer.LParen); err != nil {
		return nil, err
	}
	var out []ast.Param
	if !p.at(lexer.RParen) {
		for {
			attrs, err := p.attrs()
			if err != nil {
				return nil, err
			}
			n, err := p.expect(lexer.Ident)
			if err != nil {
				return nil, err
			}
			if _, err = p.expect(lexer.Colon); err != nil {
				return nil, err
			}
			ty, err := p.typeExpr()
			if err != nil {
				return nil, err
			}
			out = append(out, ast.Param{Name: n.Text, Attrs: attrs, Type: ty, Span: join(n.Span, ty.GetSpan())})
			if !p.at(lexer.Comma) {
				break
			}
			p.take()
			if p.at(lexer.RParen) {
				break
			}
		}
	}
	_, err := p.expect(lexer.RParen)
	return out, err
}
func (p *Parser) funcDecl() (*ast.FuncDecl, error) {
	start := p.take().Span
	n, err := p.expect(lexer.Ident)
	if err != nil {
		return nil, err
	}
	ps, err := p.params()
	if err != nil {
		return nil, err
	}
	var ret ast.TypeExpr
	if p.at(lexer.Colon) || p.at(lexer.Arrow) {
		p.take()
		ret, err = p.typeExpr()
		if err != nil {
			return nil, err
		}
	}
	body, err := p.block()
	if err != nil {
		return nil, err
	}
	return &ast.FuncDecl{Name: n.Text, Params: ps, Return: ret, Body: body, Span: join(start, body.Span)}, nil
}
func (p *Parser) computeDecl(attrs []ast.Attribute) (*ast.ComputeDecl, error) {
	start := p.take().Span
	if _, err := p.expectText("compute"); err != nil {
		return nil, err
	}
	n, err := p.expect(lexer.Ident)
	if err != nil {
		return nil, err
	}
	ps, err := p.params()
	if err != nil {
		return nil, err
	}
	b, err := p.block()
	if err != nil {
		return nil, err
	}
	return &ast.ComputeDecl{Name: n.Text, Attrs: attrs, Params: ps, Body: b, Span: join(start, b.Span)}, nil
}
func (p *Parser) typeExpr() (ast.TypeExpr, error) {
	n, err := p.expect(lexer.Ident)
	if err != nil {
		return nil, err
	}
	var t ast.TypeExpr = &ast.NamedType{Name: n.Text, Span: n.Span}
	if p.at(lexer.Less) {
		p.take()
		g := &ast.GenericType{Name: n.Text, Span: n.Span}
		for {
			if p.at(lexer.Ident) && (p.cur().Text == "read" || p.cur().Text == "read_write") {
				x := p.take()
				g.Args = append(g.Args, ast.TypeArg{Name: x.Text, IsName: true, Span: x.Span})
			} else {
				x, err := p.typeExpr()
				if err != nil {
					return nil, err
				}
				g.Args = append(g.Args, ast.TypeArg{Type: x, Span: x.GetSpan()})
			}
			if !p.at(lexer.Comma) {
				break
			}
			p.take()
		}
		r, err := p.expect(lexer.Greater)
		if err != nil {
			return nil, err
		}
		g.Span = join(n.Span, r.Span)
		t = g
	}
	if p.at(lexer.LBracket) {
		s := t.GetSpan()
		p.take()
		if p.at(lexer.RBracket) {
			r := p.take()
			t = &ast.RuntimeArrayType{Elem: t, Span: join(s, r.Span)}
		} else {
			n, err := p.expect(lexer.Number)
			if err != nil {
				return nil, err
			}
			r, err := p.expect(lexer.RBracket)
			if err != nil {
				return nil, err
			}
			t = &ast.FixedArrayType{Elem: t, Count: n.Text, Span: join(s, r.Span)}
		}
	}
	return t, nil
}
func (p *Parser) block() (*ast.BlockStmt, error) {
	l, err := p.expect(lexer.LBrace)
	if err != nil {
		return nil, err
	}
	b := &ast.BlockStmt{}
	for !p.at(lexer.RBrace) {
		if p.at(lexer.EOF) {
			return nil, p.err(p.cur(), "unterminated block")
		}
		s, err := p.stmt()
		if err != nil {
			return nil, err
		}
		b.Stmts = append(b.Stmts, s)
	}
	r := p.take()
	b.Span = join(l.Span, r.Span)
	return b, nil
}
func (p *Parser) stmt() (ast.Stmt, error) {
	switch {
	case p.text("workgroup"):
		start := p.take()
		n, err := p.expect(lexer.Ident)
		if err != nil {
			return nil, err
		}
		if _, err := p.expect(lexer.Colon); err != nil {
			return nil, err
		}
		ty, err := p.typeExpr()
		if err != nil {
			return nil, err
		}
		semi, err := p.expect(lexer.Semicolon)
		if err != nil {
			return nil, err
		}
		return &ast.WorkgroupStmt{Name: n.Text, Type: ty, Span: join(start.Span, semi.Span)}, nil
	case p.text("const") || p.text("let"):
		start := p.take()
		mut := start.Text == "let"
		n, err := p.expect(lexer.Ident)
		if err != nil {
			return nil, err
		}
		var ty ast.TypeExpr
		if p.at(lexer.Colon) {
			p.take()
			ty, err = p.typeExpr()
			if err != nil {
				return nil, err
			}
		}
		if _, err = p.expect(lexer.Assign); err != nil {
			return nil, err
		}
		v, err := p.expr(0)
		if err != nil {
			return nil, err
		}
		semi, err := p.expect(lexer.Semicolon)
		if err != nil {
			return nil, err
		}
		return &ast.VarStmt{Mutable: mut, Name: n.Text, Type: ty, Value: v, Span: join(start.Span, semi.Span)}, nil
	case p.text("if"):
		start := p.take().Span
		if _, err := p.expect(lexer.LParen); err != nil {
			return nil, err
		}
		c, err := p.expr(0)
		if err != nil {
			return nil, err
		}
		if _, err = p.expect(lexer.RParen); err != nil {
			return nil, err
		}
		th, err := p.block()
		if err != nil {
			return nil, err
		}
		var el *ast.BlockStmt
		end := th.Span
		if p.text("else") {
			p.take()
			if p.text("if") {
				nested, err := p.stmt()
				if err != nil {
					return nil, err
				}
				el = &ast.BlockStmt{Stmts: []ast.Stmt{nested}, Span: nested.GetSpan()}
				end = el.Span
				return &ast.IfStmt{Cond: c, Then: th, Else: el, Span: join(start, end)}, nil
			}
			el, err = p.block()
			if err != nil {
				return nil, err
			}
			end = el.Span
		}
		return &ast.IfStmt{Cond: c, Then: th, Else: el, Span: join(start, end)}, nil
	case p.text("while"):
		start := p.take().Span
		if _, err := p.expect(lexer.LParen); err != nil {
			return nil, err
		}
		c, err := p.expr(0)
		if err != nil {
			return nil, err
		}
		if _, err = p.expect(lexer.RParen); err != nil {
			return nil, err
		}
		b, err := p.block()
		if err != nil {
			return nil, err
		}
		return &ast.WhileStmt{Cond: c, Body: b, Span: join(start, b.Span)}, nil
	case p.text("for"):
		start := p.take().Span
		if _, err := p.expect(lexer.LParen); err != nil {
			return nil, err
		}
		if !p.text("const") && !p.text("let") {
			return nil, p.err(p.cur(), "for initializer must be a const/let declaration")
		}
		initStart := p.take()
		mut := initStart.Text == "let"
		n, err := p.expect(lexer.Ident)
		if err != nil {
			return nil, err
		}
		var ty ast.TypeExpr
		if p.at(lexer.Colon) {
			p.take()
			ty, err = p.typeExpr()
			if err != nil {
				return nil, err
			}
		}
		if _, err = p.expect(lexer.Assign); err != nil {
			return nil, err
		}
		iv, err := p.expr(0)
		if err != nil {
			return nil, err
		}
		semi, err := p.expect(lexer.Semicolon)
		if err != nil {
			return nil, err
		}
		init := &ast.VarStmt{Mutable: mut, Name: n.Text, Type: ty, Value: iv, Span: join(initStart.Span, semi.Span)}

		cond, err := p.expr(0)
		if err != nil {
			return nil, err
		}
		if _, err = p.expect(lexer.Semicolon); err != nil {
			return nil, err
		}

		postStart := p.cur().Span
		pe, err := p.expr(0)
		if err != nil {
			return nil, err
		}
		var post ast.Stmt
		if p.at(lexer.PlusPlus) || p.at(lexer.MinusMinus) {
			op := p.take()
			d := 1
			if op.Kind == lexer.MinusMinus {
				d = -1
			}
			post = &ast.IncStmt{Target: pe, Delta: d, Span: join(postStart, op.Span)}
		} else if p.at(lexer.Assign) || p.at(lexer.PlusEq) || p.at(lexer.MinusEq) || p.at(lexer.StarEq) || p.at(lexer.SlashEq) || p.at(lexer.PercentEq) || p.at(lexer.AmpEq) || p.at(lexer.PipeEq) || p.at(lexer.CaretEq) || p.at(lexer.ShiftLeftEq) || p.at(lexer.ShiftRightEq) {
			op := p.take()
			pv, err := p.expr(0)
			if err != nil {
				return nil, err
			}
			post = &ast.AssignStmt{Target: pe, Op: op.Text, Value: pv, Span: join(postStart, pv.GetSpan())}
		} else {
			return nil, p.err(p.cur(), "for update must be assignment, compound assignment, ++, or --")
		}
		if _, err = p.expect(lexer.RParen); err != nil {
			return nil, err
		}
		body, err := p.block()
		if err != nil {
			return nil, err
		}
		return &ast.ForStmt{Init: init, Cond: cond, Post: post, Body: body, Span: join(start, body.Span)}, nil
	case p.text("return"):
		start := p.take().Span
		var v ast.Expr
		var err error
		if !p.at(lexer.Semicolon) {
			v, err = p.expr(0)
			if err != nil {
				return nil, err
			}
		}
		s, err := p.expect(lexer.Semicolon)
		if err != nil {
			return nil, err
		}
		return &ast.ReturnStmt{Value: v, Span: join(start, s.Span)}, nil
	}
	start := p.cur().Span
	e, err := p.expr(0)
	if err != nil {
		return nil, err
	}
	if p.at(lexer.PlusPlus) || p.at(lexer.MinusMinus) {
		op := p.take()
		s, err := p.expect(lexer.Semicolon)
		if err != nil {
			return nil, err
		}
		d := 1
		if op.Kind == lexer.MinusMinus {
			d = -1
		}
		return &ast.IncStmt{Target: e, Delta: d, Span: join(start, s.Span)}, nil
	}
	if p.at(lexer.Assign) || p.at(lexer.PlusEq) || p.at(lexer.MinusEq) || p.at(lexer.StarEq) || p.at(lexer.SlashEq) || p.at(lexer.PercentEq) || p.at(lexer.AmpEq) || p.at(lexer.PipeEq) || p.at(lexer.CaretEq) || p.at(lexer.ShiftLeftEq) || p.at(lexer.ShiftRightEq) {
		op := p.take()
		v, err := p.expr(0)
		if err != nil {
			return nil, err
		}
		s, err := p.expect(lexer.Semicolon)
		if err != nil {
			return nil, err
		}
		return &ast.AssignStmt{Target: e, Op: op.Text, Value: v, Span: join(start, s.Span)}, nil
	}
	s, err := p.expect(lexer.Semicolon)
	if err != nil {
		return nil, err
	}
	return &ast.ExprStmt{Expr: e, Span: join(start, s.Span)}, nil
}

var prec = map[lexer.Kind]int{
	lexer.OrOr:       1,
	lexer.AndAnd:     2,
	lexer.Pipe:       3,
	lexer.Caret:      4,
	lexer.Amp:        5,
	lexer.EqEq:       6,
	lexer.NotEq:      6,
	lexer.Less:       7,
	lexer.LessEq:     7,
	lexer.Greater:    7,
	lexer.GreaterEq:  7,
	lexer.ShiftLeft:  8,
	lexer.ShiftRight: 8,
	lexer.Plus:       9,
	lexer.Minus:      9,
	lexer.Star:       10,
	lexer.Slash:      10,
	lexer.Percent:    10,
}

func (p *Parser) expr(min int) (ast.Expr, error) {
	x, err := p.prefix()
	if err != nil {
		return nil, err
	}
	for {
		pr, ok := prec[p.cur().Kind]
		if !ok || pr < min {
			break
		}
		op := p.take()
		r, err := p.expr(pr + 1)
		if err != nil {
			return nil, err
		}
		x = &ast.BinaryExpr{Op: op.Text, Left: x, Right: r, Span: join(x.GetSpan(), r.GetSpan())}
	}
	if min == 0 && p.at(lexer.Question) {
		p.take()
		thenExpr, err := p.expr(0)
		if err != nil {
			return nil, err
		}
		if _, err := p.expect(lexer.Colon); err != nil {
			return nil, err
		}
		elseExpr, err := p.expr(0)
		if err != nil {
			return nil, err
		}
		x = &ast.ConditionalExpr{Cond: x, Then: thenExpr, Else: elseExpr, Span: join(x.GetSpan(), elseExpr.GetSpan())}
	}
	return x, nil
}
func (p *Parser) prefix() (ast.Expr, error) {
	var x ast.Expr
	t := p.cur()
	switch t.Kind {
	case lexer.Number:
		p.take()
		x = &ast.NumberExpr{Raw: t.Text, Span: t.Span}
	case lexer.Ident:
		p.take()
		if t.Text == "true" || t.Text == "false" {
			x = &ast.BoolExpr{Value: t.Text == "true", Span: t.Span}
		} else {
			x = &ast.IdentExpr{Name: t.Text, Span: t.Span}
		}
	case lexer.Minus, lexer.Bang, lexer.Tilde:
		p.take()
		v, err := p.expr(11)
		if err != nil {
			return nil, err
		}
		x = &ast.UnaryExpr{Op: t.Text, X: v, Span: join(t.Span, v.GetSpan())}
	case lexer.LParen:
		p.take()
		v, err := p.expr(0)
		if err != nil {
			return nil, err
		}
		if _, err = p.expect(lexer.RParen); err != nil {
			return nil, err
		}
		x = v
	case lexer.LBrace:
		return p.structLiteral()
	default:
		return nil, p.err(t, "expected expression")
	}
	for {
		switch p.cur().Kind {
		case lexer.LParen:
			start := x.GetSpan()
			p.take()
			var args []ast.Expr
			if !p.at(lexer.RParen) {
				for {
					a, err := p.expr(0)
					if err != nil {
						return nil, err
					}
					args = append(args, a)
					if !p.at(lexer.Comma) {
						break
					}
					p.take()
				}
			}
			r, err := p.expect(lexer.RParen)
			if err != nil {
				return nil, err
			}
			x = &ast.CallExpr{Callee: x, Args: args, Span: join(start, r.Span)}
		case lexer.Dot:
			start := x.GetSpan()
			p.take()
			n, err := p.expect(lexer.Ident)
			if err != nil {
				return nil, err
			}
			x = &ast.MemberExpr{Base: x, Name: n.Text, Span: join(start, n.Span)}
		case lexer.LBracket:
			start := x.GetSpan()
			p.take()
			i, err := p.expr(0)
			if err != nil {
				return nil, err
			}
			r, err := p.expect(lexer.RBracket)
			if err != nil {
				return nil, err
			}
			x = &ast.IndexExpr{Base: x, Index: i, Span: join(start, r.Span)}
		default:
			return x, nil
		}
	}
}
func (p *Parser) structLiteral() (ast.Expr, error) {
	l := p.take()
	e := &ast.StructLiteralExpr{}
	if !p.at(lexer.RBrace) {
		for {
			n, err := p.expect(lexer.Ident)
			if err != nil {
				return nil, err
			}
			if _, err = p.expect(lexer.Colon); err != nil {
				return nil, err
			}
			v, err := p.expr(0)
			if err != nil {
				return nil, err
			}
			e.Fields = append(e.Fields, ast.LiteralField{Name: n.Text, Value: v, Span: join(n.Span, v.GetSpan())})
			if !p.at(lexer.Comma) {
				break
			}
			p.take()
			if p.at(lexer.RBrace) {
				break
			}
		}
	}
	r, err := p.expect(lexer.RBrace)
	if err != nil {
		return nil, err
	}
	e.Span = join(l.Span, r.Span)
	return e, nil
}
