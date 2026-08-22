package parser

import (
	"fmt"
	"unicode"
	"unicode/utf8"

	"tach/src/foundation"
)

type TokenKind uint16

const (
	EOF TokenKind = iota
	Ident
	Number
	String
	At
	LParen
	RParen
	LBrace
	RBrace
	LBracket
	RBracket
	Less
	Greater
	Comma
	Colon
	Question
	Semicolon
	Dot
	Assign
	Plus
	Minus
	Star
	Slash
	Percent
	Bang
	EqEq
	NotEq
	LessEq
	GreaterEq
	AndAnd
	OrOr
	Amp
	Pipe
	Caret
	Tilde
	ShiftLeft
	ShiftRight
	PlusEq
	MinusEq
	StarEq
	SlashEq
	PercentEq
	AmpEq
	PipeEq
	CaretEq
	ShiftLeftEq
	ShiftRightEq
	PlusPlus
	MinusMinus
)

type Token struct {
	Kind            TokenKind
	Text            string
	Span            foundation.Span
	LeadingComments []Comment
}

type Comment struct {
	Text string
	Span foundation.Span
}

func (k TokenKind) String() string {
	names := [...]string{
		"eof", "identifier", "number", "string", "@", "(", ")", "{", "}", "[", "]", "<", ">", ",", ":", "?", ";", ".", "=", "+", "-", "*", "/", "%", "!", "==", "!=", "<=", ">=", "&&", "||", "&", "|", "^", "~", "<<", ">>", "+=", "-=", "*=", "/=", "%=", "&=", "|=", "^=", "<<=", ">>=", "++", "--",
	}
	if int(k) < len(names) {
		return names[k]
	}
	return "token"
}

type scanner struct {
	file     string
	src      string
	off      int
	line     int
	col      int
	comments []Comment
}

func newScanner(file, src string) *scanner { return &scanner{file: file, src: src, line: 1, col: 1} }

func (s *scanner) pos() foundation.Position {
	return foundation.Position{Offset: s.off, Line: s.line, Column: s.col}
}
func (s *scanner) span(start foundation.Position) foundation.Span {
	return foundation.Span{File: s.file, Start: start, End: s.pos()}
}

func (s *scanner) peek() (rune, int) {
	if s.off >= len(s.src) {
		return 0, 0
	}
	r, n := utf8.DecodeRuneInString(s.src[s.off:])
	return r, n
}
func (s *scanner) peekN(bytes int) rune {
	if s.off+bytes >= len(s.src) {
		return 0
	}
	r, _ := utf8.DecodeRuneInString(s.src[s.off+bytes:])
	return r
}
func (s *scanner) advance() rune {
	r, n := s.peek()
	if n == 0 {
		return 0
	}
	s.off += n
	if r == '\n' {
		s.line++
		s.col = 1
	} else {
		s.col++
	}
	return r
}

func (s *scanner) skipSpaceAndComments() {
	for {
		r, n := s.peek()
		if n == 0 {
			return
		}
		if unicode.IsSpace(r) {
			s.advance()
			continue
		}
		if r == '/' && s.peekN(n) == '/' {
			start := s.pos()
			s.advance()
			s.advance()
			for {
				r, _ = s.peek()
				if r == 0 || r == '\n' {
					break
				}
				s.advance()
			}
			s.comments = append(s.comments, Comment{Text: s.src[start.Offset:s.off], Span: s.span(start)})
			continue
		}
		return
	}
}

func (s *scanner) next() (token Token, err error) {
	s.skipSpaceAndComments()
	start := s.pos()
	leading := s.comments
	s.comments = nil
	defer func() {
		if err == nil {
			token.LeadingComments = leading
		}
	}()
	r, n := s.peek()
	if n == 0 {
		return Token{Kind: EOF, Span: s.span(start)}, nil
	}
	if unicode.IsLetter(r) || r == '_' {
		s.advance()
		for {
			r, _ = s.peek()
			if !(unicode.IsLetter(r) || unicode.IsDigit(r) || r == '_') {
				break
			}
			s.advance()
		}
		return Token{Kind: Ident, Text: s.src[start.Offset:s.off], Span: s.span(start)}, nil
	}
	if unicode.IsDigit(r) {
		// Tach literals carry values, not target-language type suffixes. Semantic
		// analysis supplies a concrete type before values enter Core IR.
		s.advance()
		if r == '0' {
			next, _ := s.peek()
			if next == 'x' || next == 'X' || next == 'b' || next == 'B' {
				baseChar := next
				s.advance()
				digits := 0
				for {
					c, _ := s.peek()
					valid := unicode.IsDigit(c)
					if baseChar == 'x' || baseChar == 'X' {
						valid = valid || c >= 'a' && c <= 'f' || c >= 'A' && c <= 'F'
					} else {
						valid = c == '0' || c == '1'
					}
					if valid {
						digits++
						s.advance()
						continue
					}
					if c == '_' {
						s.advance()
						continue
					}
					break
				}
				if digits == 0 {
					return Token{}, &foundation.Diagnostic{Span: s.span(start), Message: "base-prefixed integer literal requires digits"}
				}
				if c, _ := s.peek(); unicode.IsLetter(c) {
					return Token{}, &foundation.Diagnostic{Span: s.span(start), Message: "numeric suffixes are not part of Tach; use an explicit type constructor"}
				}
				return Token{Kind: Number, Text: s.src[start.Offset:s.off], Span: s.span(start)}, nil
			}
		}

		hasDot := false
		for {
			c, _ := s.peek()
			if unicode.IsDigit(c) || c == '_' {
				s.advance()
				continue
			}
			if c == '.' && !hasDot && unicode.IsDigit(s.peekN(1)) {
				hasDot = true
				s.advance()
				continue
			}
			break
		}
		c, _ := s.peek()
		if c == 'e' || c == 'E' {
			s.advance()
			c, _ = s.peek()
			if c == '+' || c == '-' {
				s.advance()
			}
			digits := 0
			for {
				c, _ = s.peek()
				if unicode.IsDigit(c) {
					digits++
					s.advance()
					continue
				}
				if c == '_' {
					s.advance()
					continue
				}
				break
			}
			if digits == 0 {
				return Token{}, &foundation.Diagnostic{Span: s.span(start), Message: "floating-point exponent requires digits"}
			}
		}
		if c, _ = s.peek(); unicode.IsLetter(c) {
			return Token{}, &foundation.Diagnostic{Span: s.span(start), Message: "numeric suffixes are not part of Tach; use an explicit type constructor"}
		}
		return Token{Kind: Number, Text: s.src[start.Offset:s.off], Span: s.span(start)}, nil
	}
	if r == '"' {
		s.advance()
		for {
			r, _ = s.peek()
			if r == 0 || r == '\n' {
				return Token{}, &foundation.Diagnostic{Span: s.span(start), Message: "unterminated string"}
			}
			s.advance()
			if r == '"' {
				return Token{Kind: String, Text: s.src[start.Offset:s.off], Span: s.span(start)}, nil
			}
			if r == '\\' {
				if next, _ := s.peek(); next == 0 || next == '\n' {
					return Token{}, &foundation.Diagnostic{Span: s.span(start), Message: "unterminated string"}
				}
				s.advance()
			}
		}
	}
	one := func(k TokenKind) (Token, error) {
		s.advance()
		return Token{Kind: k, Text: s.src[start.Offset:s.off], Span: s.span(start)}, nil
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
		s.advance()
		if r2, _ := s.peek(); r2 == '=' {
			s.advance()
			return Token{Kind: PercentEq, Text: "%=", Span: s.span(start)}, nil
		}
		return Token{Kind: Percent, Text: "%", Span: s.span(start)}, nil
	case '=':
		s.advance()
		if r2, _ := s.peek(); r2 == '=' {
			s.advance()
			return Token{Kind: EqEq, Text: "==", Span: s.span(start)}, nil
		}
		return Token{Kind: Assign, Text: "=", Span: s.span(start)}, nil
	case '!':
		s.advance()
		if r2, _ := s.peek(); r2 == '=' {
			s.advance()
			return Token{Kind: NotEq, Text: "!=", Span: s.span(start)}, nil
		}
		return Token{Kind: Bang, Text: "!", Span: s.span(start)}, nil
	case '<', '>':
		plain, equal, shift, shiftEqual := Less, LessEq, ShiftLeft, ShiftLeftEq
		if r == '>' {
			plain, equal, shift, shiftEqual = Greater, GreaterEq, ShiftRight, ShiftRightEq
		}
		s.advance()
		if r2, _ := s.peek(); r2 == r {
			s.advance()
			if r3, _ := s.peek(); r3 == '=' {
				s.advance()
				return Token{Kind: shiftEqual, Text: s.src[start.Offset:s.off], Span: s.span(start)}, nil
			}
			return Token{Kind: shift, Text: s.src[start.Offset:s.off], Span: s.span(start)}, nil
		} else if r2 == '=' {
			s.advance()
			return Token{Kind: equal, Text: s.src[start.Offset:s.off], Span: s.span(start)}, nil
		}
		return Token{Kind: plain, Text: s.src[start.Offset:s.off], Span: s.span(start)}, nil
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
		s.advance()
		if r2, _ := s.peek(); r2 == '=' {
			s.advance()
			return Token{Kind: equal, Text: s.src[start.Offset:s.off], Span: s.span(start)}, nil
		} else if r2 == r && doubled != EOF {
			s.advance()
			return Token{Kind: doubled, Text: s.src[start.Offset:s.off], Span: s.span(start)}, nil
		}
		return Token{Kind: plain, Text: s.src[start.Offset:s.off], Span: s.span(start)}, nil
	case '~':
		return one(Tilde)
	}
	s.advance()
	return Token{}, &foundation.Diagnostic{Span: s.span(start), Message: fmt.Sprintf("unexpected character %q", r)}
}

func Tokenize(file, src string) ([]Token, foundation.Diagnostics) {
	s := newScanner(file, src)
	var out []Token
	var diagnostics foundation.Diagnostics
	for {
		t, err := s.next()
		if err != nil {
			diagnostic := *err.(*foundation.Diagnostic)
			diagnostic.Kind = "lexer"
			diagnostics = append(diagnostics, diagnostic)
			continue
		}
		out = append(out, t)
		if t.Kind == EOF {
			return out, diagnostics
		}
	}
}
