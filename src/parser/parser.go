package parser

import (
	"encoding/json"
	"fmt"

	"tach/src/ast"
	"tach/src/lexer"
	"tach/src/source"
)

type parser struct {
	toks        []lexer.Token
	i           int
	file        string
	diagnostics source.Diagnostics
}

func Parse(file, src string) (*ast.Module, error) {
	m, diagnostics := ParseRecover(file, src)
	if len(diagnostics) > 0 {
		return nil, diagnostics
	}
	return m, nil
}

func ParseRecover(file, src string) (*ast.Module, source.Diagnostics) {
	toks, diagnostics := lexer.LexRecover(file, src)
	p := &parser{toks: toks, file: file}
	m, parsed := p.moduleRecover()
	return m, append(diagnostics, parsed...).Sorted()
}

func (p *parser) moduleRecover() (*ast.Module, source.Diagnostics) {
	m := &ast.Module{File: p.file}
	var diagnostics source.Diagnostics
	importsDone, declarations := false, false
	for !p.at(lexer.EOF) {
		start := p.i
		attrs, err := p.attrs()
		if err == nil && p.at(lexer.Semicolon) {
			if len(attrs) == 0 || declarations || len(m.Imports) > 0 || len(m.Attrs) > 0 {
				err = p.err(p.cur(), "kernel documentation must precede declarations and appear at most once before imports")
			} else {
				m.Attrs = attrs
				p.take()
				continue
			}
		}
		if err == nil && p.text("import") {
			if declarations || importsDone || len(attrs) > 0 {
				err = p.err(p.cur(), "imports must be contiguous and precede declarations")
			} else {
				var item ast.Import
				item, err = p.importDecl()
				if err == nil {
					m.Imports = append(m.Imports, item)
					continue
				}
			}
		} else if err == nil {
			importsDone, declarations = len(m.Imports) > 0, true
			var declaration ast.Decl
			switch {
			case p.text("type"):
				declaration, err = p.typeDecl(attrs)
			case p.text("function") || p.text("export"):
				declaration, err = p.functionDecl(attrs)
			default:
				err = p.err(p.cur(), "expected import, type, function, or export function declaration")
			}
			if err == nil {
				m.Decls = append(m.Decls, declaration)
				continue
			}
		}
		diagnostics = append(diagnostics, diagnostic(err))
		p.syncTop(start)
	}
	return m, append(diagnostics, p.diagnostics...).Sorted()
}

func diagnostic(err error) source.Diagnostic {
	if d, ok := err.(*source.Diagnostic); ok {
		out := *d
		out.Kind = "parser"
		return out
	}
	return source.Diagnostic{Kind: "parser", Message: err.Error()}
}

func (p *parser) syncTop(start int) {
	if p.i == start && !p.at(lexer.EOF) {
		p.take()
	}
	depth := 0
	for !p.at(lexer.EOF) {
		switch p.cur().Kind {
		case lexer.LBrace, lexer.LParen, lexer.LBracket:
			depth++
		case lexer.RBrace, lexer.RParen, lexer.RBracket:
			if depth > 0 {
				depth--
			}
		}
		if depth == 0 && (p.at(lexer.At) || p.text("import") || p.text("type") || p.text("function") || p.text("export")) {
			return
		}
		p.take()
	}
}

func (p *parser) syncStatement(start int) {
	if p.i == start && !p.at(lexer.EOF) {
		p.take()
	}
	depth := 0
	for !p.at(lexer.EOF) {
		if p.at(lexer.RBrace) && depth == 0 {
			return
		}
		switch p.cur().Kind {
		case lexer.LBrace:
			depth++
		case lexer.RBrace:
			depth--
		case lexer.Semicolon:
			p.take()
			if depth == 0 {
				return
			}
			continue
		}
		p.take()
	}
}

func (p *parser) importDecl() (ast.Import, error) {
	start := p.take().Span
	token, err := p.expect(lexer.String)
	if err != nil {
		return ast.Import{}, err
	}
	var target string
	if err := json.Unmarshal([]byte(token.Text), &target); err != nil {
		return ast.Import{}, p.err(token, "invalid import string: %v", err)
	}
	semi, err := p.expect(lexer.Semicolon)
	if err != nil {
		return ast.Import{}, err
	}
	return ast.Import{Target: target, Raw: token.Text, Span: join(start, semi.Span)}, nil
}
func (p *parser) cur() lexer.Token     { return p.toks[p.i] }
func (p *parser) at(k lexer.Kind) bool { return p.cur().Kind == k }
func (p *parser) text(s string) bool   { return p.cur().Kind == lexer.Ident && p.cur().Text == s }
func (p *parser) take() lexer.Token {
	t := p.cur()
	if p.i < len(p.toks)-1 {
		p.i++
	}
	return t
}
func (p *parser) err(t lexer.Token, f string, a ...any) error {
	return &source.Diagnostic{Span: t.Span, Message: fmt.Sprintf(f, a...)}
}
func (p *parser) expect(k lexer.Kind) (lexer.Token, error) {
	if !p.at(k) {
		return lexer.Token{}, p.err(p.cur(), "expected %s, found %q", k, p.cur().Text)
	}
	return p.take(), nil
}
func (p *parser) expectText(s string) (lexer.Token, error) {
	if !p.text(s) {
		return lexer.Token{}, p.err(p.cur(), "expected %q, found %q", s, p.cur().Text)
	}
	return p.take(), nil
}

func join(a, b source.Span) source.Span { a.End = b.End; return a }
func assignment(k lexer.Kind) bool {
	switch k {
	case lexer.Assign, lexer.PlusEq, lexer.MinusEq, lexer.StarEq, lexer.SlashEq, lexer.PercentEq, lexer.AmpEq, lexer.PipeEq, lexer.CaretEq, lexer.ShiftLeftEq, lexer.ShiftRightEq:
		return true
	}
	return false
}

func (p *parser) attrs() ([]ast.Attribute, error) {
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
					if p.at(lexer.RParen) {
						break
					}
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
func (p *parser) typeDecl(attrs []ast.Attribute) (*ast.TypeDecl, error) {
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
	d := &ast.TypeDecl{Name: name.Text, Attrs: attrs}
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
func (p *parser) params() ([]ast.Param, error) {
	if _, err := p.expect(lexer.LParen); err != nil {
		return nil, err
	}
	var out []ast.Param
	if !p.at(lexer.RParen) {
		for {
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
			out = append(out, ast.Param{Name: n.Text, Type: ty, Span: join(n.Span, ty.GetSpan())})
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
func (p *parser) functionDecl(attrs []ast.Attribute) (*ast.FunctionDecl, error) {
	start := p.cur().Span
	exported := false
	if p.text("export") {
		exported = true
		p.take()
		if _, err := p.expectText("function"); err != nil {
			return nil, err
		}
	} else {
		p.take()
	}
	n, err := p.expect(lexer.Ident)
	if err != nil {
		return nil, err
	}
	var indices []ast.Index
	if p.at(lexer.LBracket) {
		indices, err = p.indices()
		if err != nil {
			return nil, err
		}
	}
	ps, err := p.params()
	if err != nil {
		return nil, err
	}
	var ret ast.TypeExpr
	if p.at(lexer.Colon) {
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
	return &ast.FunctionDecl{Name: n.Text, Exported: exported, Attrs: attrs, Indices: indices, Params: ps, Return: ret, Body: body, Span: join(start, body.Span)}, nil
}

func (p *parser) indices() ([]ast.Index, error) {
	open, err := p.expect(lexer.LBracket)
	if err != nil {
		return nil, err
	}
	if p.at(lexer.RBracket) {
		return nil, p.err(open, "indexed function requires 1 to 3 logical indices")
	}
	var out []ast.Index
	for !p.at(lexer.RBracket) {
		n, err := p.expect(lexer.Ident)
		if err != nil {
			return nil, err
		}
		out = append(out, ast.Index{Name: n.Text, Span: n.Span})
		if !p.at(lexer.Comma) {
			break
		}
		p.take()
		if p.at(lexer.RBracket) {
			break
		}
	}
	if _, err := p.expect(lexer.RBracket); err != nil {
		return nil, err
	}
	return out, nil
}
func (p *parser) typeExpr() (ast.TypeExpr, error) {
	n, err := p.expect(lexer.Ident)
	if err != nil {
		return nil, err
	}
	var t ast.TypeExpr = &ast.NamedType{Name: n.Text, Span: n.Span}
	if p.at(lexer.Less) {
		p.take()
		g := &ast.GenericType{Name: n.Text, Span: n.Span}
		for {
			x, err := p.typeExpr()
			if err != nil {
				return nil, err
			}
			g.Args = append(g.Args, x)
			if !p.at(lexer.Comma) {
				break
			}
			p.take()
			if p.at(lexer.Greater) {
				break
			}
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
func (p *parser) block() (*ast.BlockStmt, error) {
	l, err := p.expect(lexer.LBrace)
	if err != nil {
		return nil, err
	}
	b := &ast.BlockStmt{}
	for !p.at(lexer.RBrace) {
		if p.at(lexer.EOF) {
			return nil, p.err(p.cur(), "unterminated block")
		}
		start := p.i
		s, err := p.stmt()
		if err != nil {
			p.diagnostics = append(p.diagnostics, diagnostic(err))
			p.syncStatement(start)
			continue
		}
		b.Stmts = append(b.Stmts, s)
	}
	r := p.take()
	b.Span = join(l.Span, r.Span)
	return b, nil
}
func (p *parser) stmt() (ast.Stmt, error) {
	switch {
	case p.text("run"):
		return p.runStmt()
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
		if !p.at(lexer.Assign) {
			shared, ok := ty.(*ast.GenericType)
			if !mut || !ok || shared.Name != "shared" || len(shared.Args) != 1 {
				return nil, p.err(p.cur(), "expected '=' or an uninitialized shared<T> declaration")
			}
			semi, err := p.expect(lexer.Semicolon)
			if err != nil {
				return nil, err
			}
			return &ast.WorkgroupStmt{Name: n.Text, Type: shared.Args[0], Span: join(start.Span, semi.Span)}, nil
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
		if !p.text("let") {
			return nil, p.err(p.cur(), "for initializer must be a let declaration")
		}
		initStart := p.take()
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
		init := &ast.VarStmt{Mutable: true, Name: n.Text, Type: ty, Value: iv, Span: join(initStart.Span, semi.Span)}

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
		} else if assignment(p.cur().Kind) {
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
	if assignment(p.cur().Kind) {
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

func (p *parser) runStmt() (ast.Stmt, error) {
	start := p.take().Span
	stage, err := p.expect(lexer.Ident)
	if err != nil {
		return nil, err
	}
	if _, err = p.expect(lexer.LParen); err != nil {
		return nil, err
	}
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
			if p.at(lexer.RParen) {
				break
			}
		}
	}
	if _, err = p.expect(lexer.RParen); err != nil {
		return nil, err
	}
	if _, err = p.expectText("over"); err != nil {
		return nil, err
	}
	domain := ast.Domain{}
	if p.at(lexer.LBracket) {
		left := p.take().Span
		for {
			axis, err := p.expr(0)
			if err != nil {
				return nil, err
			}
			domain.Axes = append(domain.Axes, axis)
			if !p.at(lexer.Comma) {
				break
			}
			p.take()
			if p.at(lexer.RBracket) {
				break
			}
		}
		right, err := p.expect(lexer.RBracket)
		if err != nil {
			return nil, err
		}
		domain.Span = join(left, right.Span)
	} else {
		axis, err := p.expr(0)
		if err != nil {
			return nil, err
		}
		domain.Axes = []ast.Expr{axis}
		domain.Span = axis.GetSpan()
	}
	semi, err := p.expect(lexer.Semicolon)
	if err != nil {
		return nil, err
	}
	return &ast.RunStmt{Stage: stage.Text, Args: args, Domain: domain, Span: join(start, semi.Span)}, nil
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

func (p *parser) expr(min int) (ast.Expr, error) {
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
func (p *parser) prefix() (ast.Expr, error) {
	var x ast.Expr
	t := p.cur()
	switch t.Kind {
	case lexer.Number:
		p.take()
		x = &ast.NumberExpr{Raw: t.Text, Span: t.Span}
	case lexer.String:
		p.take()
		var value string
		err := json.Unmarshal([]byte(t.Text), &value)
		if err != nil {
			return nil, p.err(t, "invalid string: %v", err)
		}
		x = &ast.StringExpr{Value: value, Span: t.Span}
	case lexer.Ident:
		p.take()
		if t.Text == "true" || t.Text == "false" {
			x = &ast.BoolExpr{Value: t.Text == "true", Span: t.Span}
		} else if t.Text == "transient" && p.at(lexer.Less) {
			p.take()
			elem, err := p.typeExpr()
			if err != nil {
				return nil, err
			}
			if _, err = p.expect(lexer.Greater); err != nil {
				return nil, err
			}
			if _, err = p.expect(lexer.LParen); err != nil {
				return nil, err
			}
			count, err := p.expr(0)
			if err != nil {
				return nil, err
			}
			right, err := p.expect(lexer.RParen)
			if err != nil {
				return nil, err
			}
			return &ast.TransientExpr{Elem: elem, Count: count, Span: join(t.Span, right.Span)}, nil
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
					if p.at(lexer.RParen) {
						break
					}
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
func (p *parser) structLiteral() (ast.Expr, error) {
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
