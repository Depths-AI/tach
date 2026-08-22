package parser

import (
	"encoding/json"
	"fmt"
	"tach/foundation"
)

type parser struct {
	toks        []Token
	i           int
	file        string
	diagnostics foundation.Diagnostics
}

func Parse(file, src string) (*File, error) {
	syntax, diagnostics := ParseRecover(file, src)
	if len(diagnostics) > 0 {
		return nil, diagnostics
	}
	return syntax, nil
}

func ParseRecover(file, src string) (*File, foundation.Diagnostics) {
	toks, diagnostics := Tokenize(file, src)
	p := &parser{toks: toks, file: file}
	syntax, parsed := p.parseFile()
	return syntax, append(diagnostics, parsed...).Sorted()
}

func (p *parser) parseFile() (*File, foundation.Diagnostics) {
	file := &File{Path: p.file}
	var diagnostics foundation.Diagnostics
	importsDone, declarations := false, false
	for !p.at(EOF) {
		start := p.i
		attrs, err := p.attrs()
		if err == nil && p.at(Semicolon) {
			if len(attrs) == 0 || declarations || len(file.Imports) > 0 || len(file.Attrs) > 0 {
				err = p.err(p.cur(), "kernel documentation must precede declarations and appear at most once before imports")
			} else {
				file.Attrs = attrs
				p.take()
				continue
			}
		}
		if err == nil && p.text("import") {
			if declarations || importsDone || len(attrs) > 0 {
				err = p.err(p.cur(), "imports must be contiguous and precede declarations")
			} else {
				var item Import
				item, err = p.importDecl()
				if err == nil {
					file.Imports = append(file.Imports, item)
					continue
				}
			}
		} else if err == nil {
			importsDone, declarations = len(file.Imports) > 0, true
			var declaration Decl
			switch {
			case p.text("type"):
				declaration, err = p.typeDecl(attrs)
			case p.text("const"):
				if len(attrs) > 0 {
					err = p.err(p.cur(), "attributes are invalid on constants")
				} else {
					declaration, err = p.constDecl()
				}
			case p.text("function") || p.text("export"):
				declaration, err = p.functionDecl(attrs)
			default:
				err = p.err(p.cur(), "expected import, const, type, function, or export function declaration")
			}
			if err == nil {
				file.Decls = append(file.Decls, declaration)
				continue
			}
		}
		diagnostics = append(diagnostics, diagnostic(err))
		p.syncTop(start)
	}
	return file, append(diagnostics, p.diagnostics...).Sorted()
}

func diagnostic(err error) foundation.Diagnostic {
	if d, ok := err.(*foundation.Diagnostic); ok {
		out := *d
		out.Kind = "parser"
		return out
	}
	return foundation.Diagnostic{Kind: "parser", Message: err.Error()}
}

func (p *parser) syncTop(start int) {
	if p.i == start && !p.at(EOF) {
		p.take()
	}
	depth := 0
	for !p.at(EOF) {
		switch p.cur().Kind {
		case LBrace, LParen, LBracket:
			depth++
		case RBrace, RParen, RBracket:
			if depth > 0 {
				depth--
			}
		}
		if depth == 0 && (p.at(At) || p.text("import") || p.text("const") || p.text("type") || p.text("function") || p.text("export")) {
			return
		}
		p.take()
	}
}

func (p *parser) syncStatement(start int) {
	if p.i == start && !p.at(EOF) {
		p.take()
	}
	depth := 0
	for !p.at(EOF) {
		if p.at(RBrace) && depth == 0 {
			return
		}
		switch p.cur().Kind {
		case LBrace:
			depth++
		case RBrace:
			depth--
		case Semicolon:
			p.take()
			if depth == 0 {
				return
			}
			continue
		}
		p.take()
	}
}

func (p *parser) importDecl() (Import, error) {
	start := p.take().Span
	token, err := p.expect(String)
	if err != nil {
		return Import{}, err
	}
	var target string
	if err := json.Unmarshal([]byte(token.Text), &target); err != nil {
		return Import{}, p.err(token, "invalid import string: %v", err)
	}
	semi, err := p.expect(Semicolon)
	if err != nil {
		return Import{}, err
	}
	return Import{Target: target, Raw: token.Text, Span: join(start, semi.Span)}, nil
}
func (p *parser) cur() Token          { return p.toks[p.i] }
func (p *parser) at(k TokenKind) bool { return p.cur().Kind == k }
func (p *parser) text(s string) bool  { return p.cur().Kind == Ident && p.cur().Text == s }
func (p *parser) take() Token {
	t := p.cur()
	if p.i < len(p.toks)-1 {
		p.i++
	}
	return t
}
func (p *parser) err(t Token, f string, a ...any) error {
	return &foundation.Diagnostic{Span: t.Span, Message: fmt.Sprintf(f, a...)}
}
func (p *parser) expect(k TokenKind) (Token, error) {
	if !p.at(k) {
		return Token{}, p.err(p.cur(), "expected %s, found %q", k, p.cur().Text)
	}
	return p.take(), nil
}
func (p *parser) expectText(s string) (Token, error) {
	if !p.text(s) {
		return Token{}, p.err(p.cur(), "expected %q, found %q", s, p.cur().Text)
	}
	return p.take(), nil
}

func (p *parser) typeGreater() (Token, error) {
	if p.at(Greater) {
		return p.take(), nil
	}
	if !p.at(ShiftRight) {
		return Token{}, p.err(p.cur(), "expected >, found %q", p.cur().Text)
	}
	t := p.cur()
	first := t
	first.Kind, first.Text, first.Span.End = Greater, ">", foundation.Position{Offset: t.Span.Start.Offset + 1, Line: t.Span.Start.Line, Column: t.Span.Start.Column + 1}
	p.toks[p.i] = Token{Kind: Greater, Text: ">", Span: foundation.Span{File: t.Span.File, Start: first.Span.End, End: t.Span.End}}
	return first, nil
}

func join(a, b foundation.Span) foundation.Span { a.End = b.End; return a }
func assignment(k TokenKind) bool {
	switch k {
	case Assign, PlusEq, MinusEq, StarEq, SlashEq, PercentEq, AmpEq, PipeEq, CaretEq, ShiftLeftEq, ShiftRightEq:
		return true
	}
	return false
}

func (p *parser) attrs() ([]Attribute, error) {
	var out []Attribute
	for p.at(At) {
		start := p.take().Span
		name, err := p.expect(Ident)
		if err != nil {
			return nil, err
		}
		a := Attribute{Name: name.Text, Span: join(start, name.Span)}
		if p.at(LParen) {
			p.take()
			if !p.at(RParen) {
				for {
					e, err := p.expr(0)
					if err != nil {
						return nil, err
					}
					a.Args = append(a.Args, e)
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
			a.Span = join(start, r.Span)
		}
		out = append(out, a)
	}
	return out, nil
}
func (p *parser) typeDecl(attrs []Attribute) (*TypeDecl, error) {
	start := p.take().Span
	name, err := p.expect(Ident)
	if err != nil {
		return nil, err
	}
	if _, err = p.expect(Assign); err != nil {
		return nil, err
	}
	if _, err = p.expect(LBrace); err != nil {
		return nil, err
	}
	d := &TypeDecl{Name: name.Text, Attrs: attrs}
	for !p.at(RBrace) {
		fn, err := p.expect(Ident)
		if err != nil {
			return nil, err
		}
		if _, err = p.expect(Colon); err != nil {
			return nil, err
		}
		ty, err := p.typeExpr()
		if err != nil {
			return nil, err
		}
		f := Field{Name: fn.Text, Type: ty, Span: join(fn.Span, ty.GetSpan())}
		d.Fields = append(d.Fields, f)
		if p.at(Comma) || p.at(Semicolon) {
			p.take()
		} else if !p.at(RBrace) {
			return nil, p.err(p.cur(), "expected ',' or '}' after field")
		}
	}
	r := p.take()
	if p.at(Semicolon) {
		r = p.take()
	}
	d.Span = join(start, r.Span)
	return d, nil
}

func (p *parser) constDecl() (*ConstDecl, error) {
	start := p.take().Span
	name, ty, value, end, err := p.binding()
	if err != nil {
		return nil, err
	}
	return &ConstDecl{Name: name.Text, Type: ty, Value: value, Span: join(start, end)}, nil
}

func (p *parser) binding() (Token, TypeExpr, Expr, foundation.Span, error) {
	name, err := p.expect(Ident)
	if err != nil {
		return Token{}, nil, nil, foundation.Span{}, err
	}
	var ty TypeExpr
	if p.at(Colon) {
		p.take()
		ty, err = p.typeExpr()
		if err != nil {
			return Token{}, nil, nil, foundation.Span{}, err
		}
	}
	if _, err = p.expect(Assign); err != nil {
		return Token{}, nil, nil, foundation.Span{}, err
	}
	value, err := p.expr(0)
	if err != nil {
		return Token{}, nil, nil, foundation.Span{}, err
	}
	semi, err := p.expect(Semicolon)
	return name, ty, value, semi.Span, err
}
func (p *parser) params() ([]Param, error) {
	if _, err := p.expect(LParen); err != nil {
		return nil, err
	}
	var out []Param
	if !p.at(RParen) {
		for {
			n, err := p.expect(Ident)
			if err != nil {
				return nil, err
			}
			if _, err = p.expect(Colon); err != nil {
				return nil, err
			}
			ty, err := p.typeExpr()
			if err != nil {
				return nil, err
			}
			out = append(out, Param{Name: n.Text, Type: ty, Span: join(n.Span, ty.GetSpan())})
			if !p.at(Comma) {
				break
			}
			p.take()
			if p.at(RParen) {
				break
			}
		}
	}
	_, err := p.expect(RParen)
	return out, err
}
func (p *parser) functionDecl(attrs []Attribute) (*FunctionDecl, error) {
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
	n, err := p.expect(Ident)
	if err != nil {
		return nil, err
	}
	var indices []Index
	if p.at(LBracket) {
		indices, err = p.indices()
		if err != nil {
			return nil, err
		}
	}
	ps, err := p.params()
	if err != nil {
		return nil, err
	}
	var ret TypeExpr
	if p.at(Colon) {
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
	return &FunctionDecl{Name: n.Text, Exported: exported, Attrs: attrs, Indices: indices, Params: ps, Return: ret, Body: body, Span: join(start, body.Span)}, nil
}

func (p *parser) indices() ([]Index, error) {
	open, err := p.expect(LBracket)
	if err != nil {
		return nil, err
	}
	if p.at(RBracket) {
		return nil, p.err(open, "indexed function requires 1 to 3 logical indices")
	}
	var out []Index
	for !p.at(RBracket) {
		n, err := p.expect(Ident)
		if err != nil {
			return nil, err
		}
		out = append(out, Index{Name: n.Text, Span: n.Span})
		if !p.at(Comma) {
			break
		}
		p.take()
		if p.at(RBracket) {
			break
		}
	}
	if _, err := p.expect(RBracket); err != nil {
		return nil, err
	}
	return out, nil
}
func (p *parser) typeExpr() (TypeExpr, error) {
	n, err := p.expect(Ident)
	if err != nil {
		return nil, err
	}
	var t TypeExpr = &NamedType{Name: n.Text, Span: n.Span}
	if p.at(Less) {
		p.take()
		if n.Text == "vec" {
			elem, err := p.typeExpr()
			if err != nil {
				return nil, err
			}
			if _, err = p.expect(Comma); err != nil {
				return nil, err
			}
			lanes, err := p.expect(Number)
			if err != nil {
				return nil, err
			}
			r, err := p.typeGreater()
			if err != nil {
				return nil, err
			}
			t = &VectorType{Elem: elem, Lanes: lanes.Text, Span: join(n.Span, r.Span)}
		} else {
			g := &GenericType{Name: n.Text, Span: n.Span}
			for {
				x, err := p.typeExpr()
				if err != nil {
					return nil, err
				}
				g.Args = append(g.Args, x)
				if !p.at(Comma) {
					break
				}
				p.take()
				if p.at(Greater) {
					break
				}
			}
			r, err := p.typeGreater()
			if err != nil {
				return nil, err
			}
			g.Span = join(n.Span, r.Span)
			t = g
		}
	}
	if p.at(LBracket) {
		s := t.GetSpan()
		p.take()
		if p.at(RBracket) {
			r := p.take()
			t = &RuntimeArrayType{Elem: t, Span: join(s, r.Span)}
		} else {
			count, err := p.expr(0)
			if err != nil {
				return nil, err
			}
			r, err := p.expect(RBracket)
			if err != nil {
				return nil, err
			}
			t = &FixedArrayType{Elem: t, Count: count, Span: join(s, r.Span)}
		}
	}
	return t, nil
}
