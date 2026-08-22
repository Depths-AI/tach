package parser

func (p *parser) block() (*BlockStmt, error) {
	l, err := p.expect(LBrace)
	if err != nil {
		return nil, err
	}
	b := &BlockStmt{}
	for !p.at(RBrace) {
		if p.at(EOF) {
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
func (p *parser) stmt() (Stmt, error) {
	switch {
	case p.text("run"):
		return p.runStmt()
	case p.text("const"):
		start := p.take()
		name, ty, value, end, err := p.binding()
		if err != nil {
			return nil, err
		}
		return &ConstStmt{Name: name.Text, Type: ty, Value: value, Span: join(start.Span, end)}, nil
	case p.text("let"):
		start := p.take()
		n, err := p.expect(Ident)
		if err != nil {
			return nil, err
		}
		var ty TypeExpr
		if p.at(Colon) {
			p.take()
			ty, err = p.typeExpr()
			if err != nil {
				return nil, err
			}
		}
		if !p.at(Assign) {
			shared, ok := ty.(*GenericType)
			if !ok || shared.Name != "shared" || len(shared.Args) != 1 {
				return nil, p.err(p.cur(), "expected '=' or an uninitialized shared<T> declaration")
			}
			semi, err := p.expect(Semicolon)
			if err != nil {
				return nil, err
			}
			return &WorkgroupStmt{Name: n.Text, Type: shared.Args[0], Span: join(start.Span, semi.Span)}, nil
		}
		if _, err = p.expect(Assign); err != nil {
			return nil, err
		}
		v, err := p.expr(0)
		if err != nil {
			return nil, err
		}
		semi, err := p.expect(Semicolon)
		if err != nil {
			return nil, err
		}
		return &VarStmt{Name: n.Text, Type: ty, Value: v, Span: join(start.Span, semi.Span)}, nil
	case p.text("if"):
		start := p.take().Span
		if _, err := p.expect(LParen); err != nil {
			return nil, err
		}
		c, err := p.expr(0)
		if err != nil {
			return nil, err
		}
		if _, err = p.expect(RParen); err != nil {
			return nil, err
		}
		th, err := p.block()
		if err != nil {
			return nil, err
		}
		var el *BlockStmt
		end := th.Span
		if p.text("else") {
			p.take()
			if p.text("if") {
				nested, err := p.stmt()
				if err != nil {
					return nil, err
				}
				el = &BlockStmt{Stmts: []Stmt{nested}, Span: nested.GetSpan()}
				end = el.Span
				return &IfStmt{Cond: c, Then: th, Else: el, Span: join(start, end)}, nil
			}
			el, err = p.block()
			if err != nil {
				return nil, err
			}
			end = el.Span
		}
		return &IfStmt{Cond: c, Then: th, Else: el, Span: join(start, end)}, nil
	case p.text("while"):
		start := p.take().Span
		if _, err := p.expect(LParen); err != nil {
			return nil, err
		}
		c, err := p.expr(0)
		if err != nil {
			return nil, err
		}
		if _, err = p.expect(RParen); err != nil {
			return nil, err
		}
		b, err := p.block()
		if err != nil {
			return nil, err
		}
		return &WhileStmt{Cond: c, Body: b, Span: join(start, b.Span)}, nil
	case p.text("for"):
		start := p.take().Span
		if _, err := p.expect(LParen); err != nil {
			return nil, err
		}
		if !p.text("let") {
			return nil, p.err(p.cur(), "for initializer must be a let declaration")
		}
		initStart := p.take()
		n, err := p.expect(Ident)
		if err != nil {
			return nil, err
		}
		var ty TypeExpr
		if p.at(Colon) {
			p.take()
			ty, err = p.typeExpr()
			if err != nil {
				return nil, err
			}
		}
		if _, err = p.expect(Assign); err != nil {
			return nil, err
		}
		iv, err := p.expr(0)
		if err != nil {
			return nil, err
		}
		semi, err := p.expect(Semicolon)
		if err != nil {
			return nil, err
		}
		init := &VarStmt{Name: n.Text, Type: ty, Value: iv, Span: join(initStart.Span, semi.Span)}

		cond, err := p.expr(0)
		if err != nil {
			return nil, err
		}
		if _, err = p.expect(Semicolon); err != nil {
			return nil, err
		}

		postStart := p.cur().Span
		pe, err := p.expr(0)
		if err != nil {
			return nil, err
		}
		var post Stmt
		if p.at(PlusPlus) || p.at(MinusMinus) {
			op := p.take()
			d := 1
			if op.Kind == MinusMinus {
				d = -1
			}
			post = &IncStmt{Target: pe, Delta: d, Span: join(postStart, op.Span)}
		} else if assignment(p.cur().Kind) {
			op := p.take()
			pv, err := p.expr(0)
			if err != nil {
				return nil, err
			}
			post = &AssignStmt{Target: pe, Op: op.Text, Value: pv, Span: join(postStart, pv.GetSpan())}
		} else {
			return nil, p.err(p.cur(), "for update must be assignment, compound assignment, ++, or --")
		}
		if _, err = p.expect(RParen); err != nil {
			return nil, err
		}
		body, err := p.block()
		if err != nil {
			return nil, err
		}
		return &ForStmt{Init: init, Cond: cond, Post: post, Body: body, Span: join(start, body.Span)}, nil
	case p.text("break") || p.text("continue"):
		keyword := p.take()
		semi, err := p.expect(Semicolon)
		if err != nil {
			return nil, err
		}
		span := join(keyword.Span, semi.Span)
		if keyword.Text == "break" {
			return &BreakStmt{Span: span}, nil
		}
		return &ContinueStmt{Span: span}, nil
	case p.text("return"):
		start := p.take().Span
		var v Expr
		var err error
		if !p.at(Semicolon) {
			v, err = p.expr(0)
			if err != nil {
				return nil, err
			}
		}
		s, err := p.expect(Semicolon)
		if err != nil {
			return nil, err
		}
		return &ReturnStmt{Value: v, Span: join(start, s.Span)}, nil
	}
	start := p.cur().Span
	e, err := p.expr(0)
	if err != nil {
		return nil, err
	}
	if p.at(PlusPlus) || p.at(MinusMinus) {
		op := p.take()
		s, err := p.expect(Semicolon)
		if err != nil {
			return nil, err
		}
		d := 1
		if op.Kind == MinusMinus {
			d = -1
		}
		return &IncStmt{Target: e, Delta: d, Span: join(start, s.Span)}, nil
	}
	if assignment(p.cur().Kind) {
		op := p.take()
		v, err := p.expr(0)
		if err != nil {
			return nil, err
		}
		s, err := p.expect(Semicolon)
		if err != nil {
			return nil, err
		}
		return &AssignStmt{Target: e, Op: op.Text, Value: v, Span: join(start, s.Span)}, nil
	}
	s, err := p.expect(Semicolon)
	if err != nil {
		return nil, err
	}
	return &ExprStmt{Expr: e, Span: join(start, s.Span)}, nil
}

func (p *parser) runStmt() (Stmt, error) {
	start := p.take().Span
	stage, err := p.expect(Ident)
	if err != nil {
		return nil, err
	}
	if _, err = p.expect(LParen); err != nil {
		return nil, err
	}
	var args []Expr
	if !p.at(RParen) {
		for {
			a, err := p.expr(0)
			if err != nil {
				return nil, err
			}
			args = append(args, a)
			if !p.at(Comma) {
				break
			}
			p.take()
			if p.at(RParen) {
				break
			}
		}
	}
	if _, err = p.expect(RParen); err != nil {
		return nil, err
	}
	if _, err = p.expectText("over"); err != nil {
		return nil, err
	}
	domain := Domain{}
	if p.at(LBracket) {
		left := p.take().Span
		for {
			axis, err := p.expr(0)
			if err != nil {
				return nil, err
			}
			domain.Axes = append(domain.Axes, axis)
			if !p.at(Comma) {
				break
			}
			p.take()
			if p.at(RBracket) {
				break
			}
		}
		right, err := p.expect(RBracket)
		if err != nil {
			return nil, err
		}
		domain.Span = join(left, right.Span)
	} else {
		axis, err := p.expr(0)
		if err != nil {
			return nil, err
		}
		domain.Axes = []Expr{axis}
		domain.Span = axis.GetSpan()
	}
	semi, err := p.expect(Semicolon)
	if err != nil {
		return nil, err
	}
	return &RunStmt{Stage: stage.Text, Args: args, Domain: domain, Span: join(start, semi.Span)}, nil
}
