package wgsl

import (
	"fmt"
	"unicode"
)

type vk uint8

const (
	vEOF vk = iota
	vIdent
	vNumber
	vAt
	vLParen
	vRParen
	vLBrace
	vRBrace
	vLBracket
	vRBracket
	vLess
	vGreater
	vComma
	vColon
	vSemi
	vDot
	vAmp
	vBang
	vEq
	vOp
	vArrow
)

type vt struct {
	k   vk
	s   string
	off int
}

func lexWGSL(src string) ([]vt, error) {
	var out []vt
	for i := 0; i < len(src); {
		c := src[i]
		if c == '/' && i+1 < len(src) && src[i+1] == '/' {
			i += 2
			for i < len(src) && src[i] != '\n' {
				i++
			}
			continue
		}
		if unicode.IsSpace(rune(c)) {
			i++
			continue
		}
		start := i
		if unicode.IsLetter(rune(c)) || c == '_' {
			i++
			for i < len(src) && (unicode.IsLetter(rune(src[i])) || unicode.IsDigit(rune(src[i])) || src[i] == '_') {
				i++
			}
			out = append(out, vt{vIdent, src[start:i], start})
			continue
		}
		if unicode.IsDigit(rune(c)) {
			i++
			for i < len(src) && (unicode.IsDigit(rune(src[i])) || src[i] == '.' || src[i] == 'u' || src[i] == 'i' || src[i] == 'f') {
				i++
			}
			out = append(out, vt{vNumber, src[start:i], start})
			continue
		}
		one := func(k vk) { out = append(out, vt{k, string(c), start}); i++ }
		switch c {
		case '@':
			one(vAt)
		case '(':
			one(vLParen)
		case ')':
			one(vRParen)
		case '{':
			one(vLBrace)
		case '}':
			one(vRBrace)
		case '[':
			one(vLBracket)
		case ']':
			one(vRBracket)
		case '<':
			one(vLess)
		case '>':
			one(vGreater)
		case ',':
			one(vComma)
		case ':':
			one(vColon)
		case ';':
			one(vSemi)
		case '.':
			one(vDot)
		case '&':
			one(vAmp)
		case '!':
			one(vBang)
		case '=':
			one(vEq)
		case '+', '-', '*', '/', '%', '|', '^', '~':
			one(vOp)
		default:
			return nil, fmt.Errorf("WGSL byte %d: unexpected character %q", i, c)
		}
	}
	out = append(out, vt{vEOF, "", len(src)})
	return out, nil
}

type vp struct {
	t []vt
	i int
}

func (p *vp) cur() vt { return p.t[p.i] }
func (p *vp) take() vt {
	x := p.cur()
	if p.i < len(p.t)-1 {
		p.i++
	}
	return x
}
func (p *vp) want(k vk) error {
	if p.cur().k != k {
		return fmt.Errorf("WGSL byte %d: expected token %d, found %q", p.cur().off, k, p.cur().s)
	}
	p.take()
	return nil
}
func (p *vp) ident(s string) bool { return p.cur().k == vIdent && p.cur().s == s }

// Validate reparses the exact WGSL subset emitted by Tach. Its purpose is to
// catch backend syntax/structure regressions independently from IR validation.
func Validate(src string) error {
	ts, err := lexWGSL(src)
	if err != nil {
		return err
	}
	p := &vp{t: ts}
	for p.cur().k != vEOF {
		for p.cur().k == vAt {
			if err := p.attr(); err != nil {
				return err
			}
		}
		switch {
		case p.ident("struct"):
			if err := p.structDecl(); err != nil {
				return err
			}
		case p.ident("var"):
			if err := p.globalVar(); err != nil {
				return err
			}
		case p.ident("fn"):
			if err := p.fnDecl(); err != nil {
				return err
			}
		default:
			return fmt.Errorf("WGSL byte %d: expected struct, var, or fn; found %q", p.cur().off, p.cur().s)
		}
	}
	return nil
}
func (p *vp) attr() error {
	if err := p.want(vAt); err != nil {
		return err
	}
	if err := p.want(vIdent); err != nil {
		return err
	}
	if p.cur().k == vLParen {
		return p.skipBalanced(vLParen, vRParen)
	}
	return nil
}
func (p *vp) skipBalanced(l, r vk) error {
	if err := p.want(l); err != nil {
		return err
	}
	d := 1
	for d > 0 {
		if p.cur().k == vEOF {
			return fmt.Errorf("WGSL: unterminated balanced construct")
		}
		k := p.take().k
		if k == l {
			d++
		} else if k == r {
			d--
		}
	}
	return nil
}
func (p *vp) structDecl() error {
	p.take()
	if err := p.want(vIdent); err != nil {
		return err
	}
	if err := p.want(vLBrace); err != nil {
		return err
	}
	for p.cur().k != vRBrace {
		for p.cur().k == vAt {
			if err := p.attr(); err != nil {
				return err
			}
		}
		if err := p.want(vIdent); err != nil {
			return err
		}
		if err := p.want(vColon); err != nil {
			return err
		}
		if err := p.typeExpr(); err != nil {
			return err
		}
		if err := p.want(vComma); err != nil {
			return err
		}
	}
	p.take()
	return nil
}
func (p *vp) typeExpr() error {
	if err := p.want(vIdent); err != nil {
		return err
	}
	if p.cur().k == vLess {
		if err := p.skipBalanced(vLess, vGreater); err != nil {
			return err
		}
	}
	return nil
}
func (p *vp) globalVar() error {
	p.take()
	if p.cur().k == vLess {
		if err := p.skipBalanced(vLess, vGreater); err != nil {
			return err
		}
	}
	if err := p.want(vIdent); err != nil {
		return err
	}
	if err := p.want(vColon); err != nil {
		return err
	}
	if err := p.typeExpr(); err != nil {
		return err
	}
	return p.want(vSemi)
}
func (p *vp) fnDecl() error {
	p.take()
	if err := p.want(vIdent); err != nil {
		return err
	}
	if err := p.skipBalanced(vLParen, vRParen); err != nil {
		return err
	}
	if p.cur().k == vOp && p.cur().s == "-" {
		p.take()
		if p.cur().k != vGreater {
			return fmt.Errorf("WGSL byte %d: expected > in return arrow", p.cur().off)
		}
		p.take()
		if err := p.typeExpr(); err != nil {
			return err
		}
	}
	return p.block()
}
func (p *vp) block() error {
	if err := p.want(vLBrace); err != nil {
		return err
	}
	for p.cur().k != vRBrace {
		if p.cur().k == vEOF {
			return fmt.Errorf("WGSL: unterminated block")
		}
		if p.ident("if") {
			p.take()
			if err := p.skipBalanced(vLParen, vRParen); err != nil {
				return err
			}
			if err := p.block(); err != nil {
				return err
			}
			if p.ident("else") {
				p.take()
				if err := p.block(); err != nil {
					return err
				}
			}
			continue
		}
		if p.ident("loop") {
			p.take()
			if err := p.block(); err != nil {
				return err
			}
			continue
		}
		if p.ident("return") {
			p.take()
			for p.cur().k != vSemi {
				if p.cur().k == vEOF || p.cur().k == vRBrace {
					return fmt.Errorf("WGSL: malformed return")
				}
				p.take()
			}
			p.take()
			continue
		}
		if p.ident("break") {
			p.take()
			if err := p.want(vSemi); err != nil {
				return err
			}
			continue
		}
		if p.ident("continue") {
			p.take()
			if err := p.want(vSemi); err != nil {
				return err
			}
			continue
		} // declaration/assignment/call: consume one balanced statement
		// Angle brackets are deliberately not tracked here: WGSL uses the same
		// tokens for generic type arguments and comparisons. A semicolon cannot
		// occur inside a type argument list, so parenthesis/square-bracket depth
		// is sufficient and avoids misclassifying `a < b` as an open generic.
		depthP, depthS := 0, 0
		for {
			q := p.cur()
			if q.k == vEOF || q.k == vRBrace {
				return fmt.Errorf("WGSL byte %d: statement missing semicolon", q.off)
			}
			switch q.k {
			case vLParen:
				depthP++
			case vRParen:
				depthP--
			case vLBracket:
				depthS++
			case vRBracket:
				depthS--
			}
			p.take()
			if q.k == vSemi && depthP == 0 && depthS == 0 {
				break
			}
		}
	}
	p.take()
	return nil
}
