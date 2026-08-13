package lexer

import "tach/src/source"

type Kind uint16

const (
	EOF Kind = iota
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
	Kind Kind
	Text string
	Span source.Span
}

func (k Kind) String() string {
	names := [...]string{
		"eof", "identifier", "number", "string", "@", "(", ")", "{", "}", "[", "]", "<", ">", ",", ":", "?", ";", ".", "=", "+", "-", "*", "/", "%", "!", "==", "!=", "<=", ">=", "&&", "||", "&", "|", "^", "~", "<<", ">>", "+=", "-=", "*=", "/=", "%=", "&=", "|=", "^=", "<<=", ">>=", "++", "--",
	}
	if int(k) < len(names) {
		return names[k]
	}
	return "token"
}
