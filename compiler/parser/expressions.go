package parser

import (
	"encoding/json"
)

var prec = map[TokenKind]int{
	OrOr:       1,
	AndAnd:     2,
	Pipe:       3,
	Caret:      4,
	Amp:        5,
	EqEq:       6,
	NotEq:      6,
	Less:       7,
	LessEq:     7,
	Greater:    7,
	GreaterEq:  7,
	ShiftLeft:  8,
	ShiftRight: 8,
	Plus:       9,
	Minus:      9,
	Star:       10,
	Slash:      10,
	Percent:    10,
}

func (p *parser) expr(min int) (Expr, error) {
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
		x = &BinaryExpr{Op: op.Text, Left: x, Right: r, Span: join(x.GetSpan(), r.GetSpan())}
	}
	if min == 0 && p.at(Question) {
		p.take()
		thenExpr, err := p.expr(0)
		if err != nil {
			return nil, err
		}
		if _, err := p.expect(Colon); err != nil {
			return nil, err
		}
		elseExpr, err := p.expr(0)
		if err != nil {
			return nil, err
		}
		x = &ConditionalExpr{Cond: x, Then: thenExpr, Else: elseExpr, Span: join(x.GetSpan(), elseExpr.GetSpan())}
	}
	return x, nil
}
func (p *parser) prefix() (Expr, error) {
	var x Expr
	t := p.cur()
	switch t.Kind {
	case Number:
		p.take()
		x = &NumberExpr{Raw: t.Text, Span: t.Span}
	case String:
		p.take()
		var value string
		err := json.Unmarshal([]byte(t.Text), &value)
		if err != nil {
			return nil, p.err(t, "invalid string: %v", err)
		}
		x = &StringExpr{Value: value, Span: t.Span}
	case Ident:
		p.take()
		if t.Text == "true" || t.Text == "false" {
			x = &BoolExpr{Value: t.Text == "true", Span: t.Span}
		} else if t.Text == "transient" && p.at(Less) {
			p.take()
			elem, err := p.typeExpr()
			if err != nil {
				return nil, err
			}
			if _, err = p.expect(Greater); err != nil {
				return nil, err
			}
			if _, err = p.expect(LParen); err != nil {
				return nil, err
			}
			count, err := p.expr(0)
			if err != nil {
				return nil, err
			}
			right, err := p.expect(RParen)
			if err != nil {
				return nil, err
			}
			return &TransientExpr{Elem: elem, Count: count, Span: join(t.Span, right.Span)}, nil
		} else {
			x = &IdentExpr{Name: t.Text, Span: t.Span}
		}
	case Minus, Bang, Tilde:
		p.take()
		v, err := p.expr(11)
		if err != nil {
			return nil, err
		}
		x = &UnaryExpr{Op: t.Text, X: v, Span: join(t.Span, v.GetSpan())}
	case LParen:
		p.take()
		v, err := p.expr(0)
		if err != nil {
			return nil, err
		}
		if _, err = p.expect(RParen); err != nil {
			return nil, err
		}
		x = v
	case LBrace:
		return p.structLiteral()
	default:
		return nil, p.err(t, "expected expression")
	}
	for {
		switch p.cur().Kind {
		case LParen:
			start := x.GetSpan()
			p.take()
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
			r, err := p.expect(RParen)
			if err != nil {
				return nil, err
			}
			x = &CallExpr{Callee: x, Args: args, Span: join(start, r.Span)}
		case Dot:
			start := x.GetSpan()
			p.take()
			n, err := p.expect(Ident)
			if err != nil {
				return nil, err
			}
			x = &MemberExpr{Base: x, Name: n.Text, Span: join(start, n.Span)}
		case LBracket:
			start := x.GetSpan()
			p.take()
			i, err := p.expr(0)
			if err != nil {
				return nil, err
			}
			r, err := p.expect(RBracket)
			if err != nil {
				return nil, err
			}
			x = &IndexExpr{Base: x, Index: i, Span: join(start, r.Span)}
		default:
			return x, nil
		}
	}
}
func (p *parser) structLiteral() (Expr, error) {
	l := p.take()
	e := &StructLiteralExpr{}
	if !p.at(RBrace) {
		for {
			n, err := p.expect(Ident)
			if err != nil {
				return nil, err
			}
			if _, err = p.expect(Colon); err != nil {
				return nil, err
			}
			v, err := p.expr(0)
			if err != nil {
				return nil, err
			}
			e.Fields = append(e.Fields, LiteralField{Name: n.Text, Value: v, Span: join(n.Span, v.GetSpan())})
			if !p.at(Comma) {
				break
			}
			p.take()
			if p.at(RBrace) {
				break
			}
		}
	}
	r, err := p.expect(RBrace)
	if err != nil {
		return nil, err
	}
	e.Span = join(l.Span, r.Span)
	return e, nil
}
