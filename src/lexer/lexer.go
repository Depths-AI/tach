package lexer

import (
	"fmt"
	"unicode"
	"unicode/utf8"

	"tach/src/source"
)

type Lexer struct {
	file string
	src  string
	off  int
	line int
	col  int
}

func New(file, src string) *Lexer { return &Lexer{file: file, src: src, line: 1, col: 1} }

func (l *Lexer) pos() source.Pos { return source.Pos{Offset: l.off, Line: l.line, Column: l.col} }
func (l *Lexer) span(start source.Pos) source.Span {
	return source.Span{File: l.file, Start: start, End: l.pos()}
}

func (l *Lexer) peek() (rune, int) {
	if l.off >= len(l.src) {
		return 0, 0
	}
	r, n := utf8.DecodeRuneInString(l.src[l.off:])
	return r, n
}
func (l *Lexer) peekN(bytes int) rune {
	if l.off+bytes >= len(l.src) {
		return 0
	}
	r, _ := utf8.DecodeRuneInString(l.src[l.off+bytes:])
	return r
}
func (l *Lexer) advance() rune {
	r, n := l.peek()
	if n == 0 {
		return 0
	}
	l.off += n
	if r == '\n' {
		l.line++
		l.col = 1
	} else {
		l.col++
	}
	return r
}

func (l *Lexer) skipSpaceAndComments() error {
	for {
		r, n := l.peek()
		if n == 0 {
			return nil
		}
		if unicode.IsSpace(r) {
			l.advance()
			continue
		}
		if r == '/' && l.peekN(n) == '/' {
			l.advance()
			l.advance()
			for {
				r, _ = l.peek()
				if r == 0 || r == '\n' {
					break
				}
				l.advance()
			}
			continue
		}
		return nil
	}
}

func (l *Lexer) Next() (Token, error) {
	if err := l.skipSpaceAndComments(); err != nil {
		return Token{}, err
	}
	start := l.pos()
	r, n := l.peek()
	if n == 0 {
		return Token{Kind: EOF, Span: l.span(start)}, nil
	}
	if unicode.IsLetter(r) || r == '_' {
		l.advance()
		for {
			r, _ = l.peek()
			if !(unicode.IsLetter(r) || unicode.IsDigit(r) || r == '_') {
				break
			}
			l.advance()
		}
		return Token{Kind: Ident, Text: l.src[start.Offset:l.off], Span: l.span(start)}, nil
	}
	if unicode.IsDigit(r) {
		// Tach literals carry values, not target-language type suffixes. Semantic
		// analysis supplies a concrete type before values enter Core IR.
		l.advance()
		if r == '0' {
			next, _ := l.peek()
			if next == 'x' || next == 'X' || next == 'b' || next == 'B' {
				baseChar := next
				l.advance()
				digits := 0
				for {
					c, _ := l.peek()
					valid := unicode.IsDigit(c)
					if baseChar == 'x' || baseChar == 'X' {
						valid = valid || c >= 'a' && c <= 'f' || c >= 'A' && c <= 'F'
					} else {
						valid = c == '0' || c == '1'
					}
					if valid {
						digits++
						l.advance()
						continue
					}
					if c == '_' {
						l.advance()
						continue
					}
					break
				}
				if digits == 0 {
					return Token{}, &source.Error{Span: l.span(start), Message: "base-prefixed integer literal requires digits"}
				}
				if c, _ := l.peek(); unicode.IsLetter(c) {
					return Token{}, &source.Error{Span: l.span(start), Message: "numeric suffixes are not part of Tach; use an explicit type constructor"}
				}
				return Token{Kind: Number, Text: l.src[start.Offset:l.off], Span: l.span(start)}, nil
			}
		}

		hasDot := false
		for {
			c, _ := l.peek()
			if unicode.IsDigit(c) || c == '_' {
				l.advance()
				continue
			}
			if c == '.' && !hasDot && unicode.IsDigit(l.peekN(1)) {
				hasDot = true
				l.advance()
				continue
			}
			break
		}
		c, _ := l.peek()
		if c == 'e' || c == 'E' {
			l.advance()
			c, _ = l.peek()
			if c == '+' || c == '-' {
				l.advance()
			}
			digits := 0
			for {
				c, _ = l.peek()
				if unicode.IsDigit(c) {
					digits++
					l.advance()
					continue
				}
				if c == '_' {
					l.advance()
					continue
				}
				break
			}
			if digits == 0 {
				return Token{}, &source.Error{Span: l.span(start), Message: "floating-point exponent requires digits"}
			}
		}
		if c, _ = l.peek(); unicode.IsLetter(c) {
			return Token{}, &source.Error{Span: l.span(start), Message: "numeric suffixes are not part of Tach; use an explicit type constructor"}
		}
		return Token{Kind: Number, Text: l.src[start.Offset:l.off], Span: l.span(start)}, nil
	}
	if r == '"' {
		l.advance()
		for {
			r, _ = l.peek()
			if r == 0 || r == '\n' {
				return Token{}, &source.Error{Span: l.span(start), Message: "unterminated string"}
			}
			l.advance()
			if r == '"' {
				return Token{Kind: String, Text: l.src[start.Offset:l.off], Span: l.span(start)}, nil
			}
			if r == '\\' {
				if next, _ := l.peek(); next == 0 || next == '\n' {
					return Token{}, &source.Error{Span: l.span(start), Message: "unterminated string"}
				}
				l.advance()
			}
		}
	}
	one := func(k Kind) (Token, error) {
		l.advance()
		return Token{Kind: k, Text: l.src[start.Offset:l.off], Span: l.span(start)}, nil
	}
	switch r {
	case '@':
		return one(At)
	case '(':
		return one(LParen)
	case ')':
		return one(RParen)
	case '{':
		return one(LBrace)
	case '}':
		return one(RBrace)
	case '[':
		return one(LBracket)
	case ']':
		return one(RBracket)
	case ',':
		return one(Comma)
	case ':':
		return one(Colon)
	case '?':
		return one(Question)
	case ';':
		return one(Semicolon)
	case '.':
		return one(Dot)
	case '%':
		l.advance()
		if r2, _ := l.peek(); r2 == '=' {
			l.advance()
			return Token{Kind: PercentEq, Text: "%=", Span: l.span(start)}, nil
		}
		return Token{Kind: Percent, Text: "%", Span: l.span(start)}, nil
	case '=':
		l.advance()
		if r2, _ := l.peek(); r2 == '=' {
			l.advance()
			return Token{Kind: EqEq, Text: "==", Span: l.span(start)}, nil
		}
		return Token{Kind: Assign, Text: "=", Span: l.span(start)}, nil
	case '!':
		l.advance()
		if r2, _ := l.peek(); r2 == '=' {
			l.advance()
			return Token{Kind: NotEq, Text: "!=", Span: l.span(start)}, nil
		}
		return Token{Kind: Bang, Text: "!", Span: l.span(start)}, nil
	case '<', '>':
		plain, equal, shift, shiftEqual := Less, LessEq, ShiftLeft, ShiftLeftEq
		if r == '>' {
			plain, equal, shift, shiftEqual = Greater, GreaterEq, ShiftRight, ShiftRightEq
		}
		l.advance()
		if r2, _ := l.peek(); r2 == r {
			l.advance()
			if r3, _ := l.peek(); r3 == '=' {
				l.advance()
				return Token{Kind: shiftEqual, Text: l.src[start.Offset:l.off], Span: l.span(start)}, nil
			}
			return Token{Kind: shift, Text: l.src[start.Offset:l.off], Span: l.span(start)}, nil
		} else if r2 == '=' {
			l.advance()
			return Token{Kind: equal, Text: l.src[start.Offset:l.off], Span: l.span(start)}, nil
		}
		return Token{Kind: plain, Text: l.src[start.Offset:l.off], Span: l.span(start)}, nil
	case '+', '-', '*', '/', '&', '|', '^':
		plain, equal, doubled := Star, StarEq, EOF
		switch r {
		case '+':
			plain, equal, doubled = Plus, PlusEq, PlusPlus
		case '-':
			plain, equal, doubled = Minus, MinusEq, MinusMinus
		case '/':
			plain, equal = Slash, SlashEq
		case '&':
			plain, equal, doubled = Amp, AmpEq, AndAnd
		case '|':
			plain, equal, doubled = Pipe, PipeEq, OrOr
		case '^':
			plain, equal = Caret, CaretEq
		}
		l.advance()
		if r2, _ := l.peek(); r2 == '=' {
			l.advance()
			return Token{Kind: equal, Text: l.src[start.Offset:l.off], Span: l.span(start)}, nil
		} else if r2 == r && doubled != EOF {
			l.advance()
			return Token{Kind: doubled, Text: l.src[start.Offset:l.off], Span: l.span(start)}, nil
		}
		return Token{Kind: plain, Text: l.src[start.Offset:l.off], Span: l.span(start)}, nil
	case '~':
		return one(Tilde)
	}
	l.advance()
	return Token{}, &source.Error{Span: l.span(start), Message: fmt.Sprintf("unexpected character %q", r)}
}

func Lex(file, src string) ([]Token, error) {
	l := New(file, src)
	var out []Token
	for {
		t, err := l.Next()
		if err != nil {
			return nil, err
		}
		out = append(out, t)
		if t.Kind == EOF {
			return out, nil
		}
	}
}
