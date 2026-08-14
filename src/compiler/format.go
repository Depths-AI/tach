package compiler

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"tach/src/lexer"
)

type formatFile struct{ source, next, backup string }

func Format(cwd string, workers int) error {
	project, err := loadProject(cwd, workers)
	if err != nil {
		return err
	}
	files := make([]formatFile, 0, len(project.Kernels))
	for _, kernel := range project.Kernels {
		formatted, err := formatSource(kernel.Identity+".tach", kernel.Source)
		if err != nil {
			return err
		}
		temporary, err := os.CreateTemp(filepath.Dir(kernel.Path), ".tach-fmt-*")
		if err != nil {
			return err
		}
		name := temporary.Name()
		if _, err = temporary.WriteString(formatted); err == nil {
			info, statErr := os.Stat(kernel.Path)
			if statErr != nil {
				err = statErr
			} else {
				err = temporary.Chmod(info.Mode().Perm())
			}
		}
		closeErr := temporary.Close()
		if err == nil {
			err = closeErr
		}
		if err != nil {
			cleanupFormat(files, name)
			return err
		}
		backupFile, err := os.CreateTemp(filepath.Dir(kernel.Path), ".tach-fmt-backup-*")
		if err != nil {
			cleanupFormat(files, name)
			return err
		}
		backup := backupFile.Name()
		if err := backupFile.Close(); err != nil {
			cleanupFormat(files, name, backup)
			return err
		}
		if err := os.Remove(backup); err != nil {
			cleanupFormat(files, name, "")
			return err
		}
		files = append(files, formatFile{source: kernel.Path, next: name, backup: backup})
	}
	for i := range files {
		if err := os.Rename(files[i].source, files[i].backup); err != nil {
			rollbackFormat(files, i)
			return err
		}
	}
	for i := range files {
		if err := os.Rename(files[i].next, files[i].source); err != nil {
			for j := 0; j < i; j++ {
				_ = os.Remove(files[j].source)
			}
			rollbackFormat(files, len(files))
			return err
		}
	}
	for _, file := range files {
		if err := os.Remove(file.backup); err != nil {
			return fmt.Errorf("remove formatter backup %s: %w", file.backup, err)
		}
	}
	return nil
}

func cleanupFormat(files []formatFile, paths ...string) {
	for _, file := range files {
		_ = os.Remove(file.next)
	}
	for _, path := range paths {
		if path != "" {
			_ = os.Remove(path)
		}
	}
}

func rollbackFormat(files []formatFile, moved int) {
	for i := moved - 1; i >= 0; i-- {
		_ = os.Rename(files[i].backup, files[i].source)
	}
	for _, file := range files {
		_ = os.Remove(file.next)
	}
}

func formatSource(file, text string) (string, error) {
	tokens, diagnostics := lexer.LexRecover(file, text)
	if len(diagnostics) > 0 {
		return "", diagnostics
	}
	var out strings.Builder
	indent, parens, generics, line := 0, 0, 0, 0
	lineStart, blank := true, false
	topKind := ""
	type list struct {
		close     lexer.Kind
		multiline bool
		indented  bool
		attribute bool
	}
	var lists []list
	var questions []int
	write := func(value string) {
		if lineStart {
			out.WriteString(strings.Repeat("  ", indent))
			line = indent * 2
			lineStart = false
		}
		out.WriteString(value)
		line += len(value)
	}
	space := func() {
		if !lineStart && out.Len() > 0 {
			value := out.String()
			if value[len(value)-1] != ' ' && value[len(value)-1] != '\n' {
				out.WriteByte(' ')
				line++
			}
		}
	}
	newline := func(extra bool) {
		value := out.String()
		if len(value) > 0 && value[len(value)-1] != '\n' {
			out.WriteByte('\n')
		}
		if extra && !blank {
			out.WriteByte('\n')
			blank = true
		} else if !extra {
			blank = false
		}
		line, lineStart = 0, true
	}
	operator := func(kind lexer.Kind) bool {
		return kind >= lexer.Assign && kind <= lexer.ShiftRightEq || kind == lexer.Question
	}
	segmentWidth := func(start int) int {
		width, depth := 0, 0
		for _, token := range tokens[start:] {
			if depth == 0 && width > 0 && (operator(token.Kind) || token.Kind == lexer.Colon || token.Kind == lexer.Comma || token.Kind == lexer.Semicolon) {
				break
			}
			width += len(token.Text) + 1
			switch token.Kind {
			case lexer.LParen, lexer.LBracket, lexer.LBrace:
				depth++
			case lexer.RParen, lexer.RBracket, lexer.RBrace:
				if depth == 0 {
					return width
				}
				depth--
			}
		}
		return width
	}
	remainderWidth := func(start int) int {
		width := 0
		for _, token := range tokens[start:] {
			if token.Kind == lexer.Semicolon {
				break
			}
			width += len(token.Text) + 1
		}
		return width
	}
	// DECISION: a four-column margin covers spacing the lexical estimate omits; remove it if wrapping tracks rendered tail width exactly.
	var previous lexer.Token
	for i, token := range tokens {
		if token.Kind == lexer.EOF {
			for _, trivia := range token.Leading {
				if previous.Kind != lexer.EOF && trivia.Span.Start.Line == previous.Span.End.Line && !lineStart {
					space()
				} else {
					newline(false)
				}
				write(strings.TrimSuffix(trivia.Text, "\r"))
				newline(false)
			}
			break
		}
		for _, trivia := range token.Leading {
			if previous.Kind != lexer.EOF && trivia.Span.Start.Line == previous.Span.End.Line && !lineStart {
				space()
				write(strings.TrimSuffix(trivia.Text, "\r"))
				newline(false)
			} else {
				newline(false)
				write(strings.TrimSuffix(trivia.Text, "\r"))
				newline(false)
			}
		}
		next := lexer.Token{Kind: lexer.EOF}
		if i+1 < len(tokens) {
			next = tokens[i+1]
		}
		trailingComment := len(next.Leading) > 0 && next.Leading[0].Span.Start.Line == token.Span.End.Line
		if indent == 0 && lineStart && topKind == "" {
			if token.Kind == lexer.At {
				topKind = "docs"
			} else {
				topKind = token.Text
			}
		}
		switch token.Kind {
		case lexer.LBrace:
			_, comma := listWidth(tokens, i, lexer.RBrace)
			space()
			write("{")
			indent++
			lists = append(lists, list{close: lexer.RBrace, multiline: comma, indented: true})
			if trailingComment {
				space()
			} else {
				newline(false)
			}
		case lexer.RBrace:
			item := lists[len(lists)-1]
			lists = lists[:len(lists)-1]
			if item.multiline && previous.Kind != lexer.Comma && previous.Kind != lexer.LBrace {
				write(",")
			}
			indent--
			newline(false)
			write("}")
			if trailingComment {
				space()
			} else if next.Kind != lexer.Semicolon && next.Kind != lexer.Comma && next.Text != "else" {
				newline(indent == 0)
			}
		case lexer.Semicolon:
			write(";")
			if parens > 0 {
				space()
			} else if trailingComment {
				space()
			} else {
				newline(indent == 0 && (previous.Kind == lexer.RBrace || topKind == "docs" || topKind == "import" && next.Text != "import"))
				topKind = ""
			}
		case lexer.Comma:
			if len(lists) > 0 && next.Kind == lists[len(lists)-1].close && !lists[len(lists)-1].multiline {
				break
			}
			write(",")
			if trailingComment {
				space()
			} else if len(lists) > 0 && lists[len(lists)-1].multiline {
				newline(false)
			} else if line >= 96 {
				if len(lists) > 0 {
					lists[len(lists)-1].multiline = true
					if !lists[len(lists)-1].indented {
						indent++
						lists[len(lists)-1].indented = true
					}
				}
				newline(false)
			} else {
				space()
			}
		case lexer.Colon:
			if len(questions) > 0 && questions[len(questions)-1] == len(lists) {
				space()
				write(":")
				questions = questions[:len(questions)-1]
			} else {
				write(":")
			}
			space()
		case lexer.Dot:
			write(".")
		case lexer.LParen:
			if previous.Text == "if" || previous.Text == "while" || previous.Text == "for" || previous.Text == "return" {
				space()
			}
			write("(")
			parens++
			width, comma := listWidth(tokens, i, lexer.RParen)
			multiline := comma && line+width > 100
			attribute := i > 1 && tokens[i-1].Kind == lexer.Ident && tokens[i-2].Kind == lexer.At
			lists = append(lists, list{close: lexer.RParen, multiline: multiline, indented: multiline, attribute: attribute})
			if multiline {
				indent++
				newline(false)
			}
		case lexer.RParen:
			item := lists[len(lists)-1]
			lists = lists[:len(lists)-1]
			if item.multiline {
				if previous.Kind != lexer.Comma && previous.Kind != lexer.LParen {
					write(",")
				}
				indent--
				newline(false)
			}
			write(")")
			parens--
			if item.attribute && next.Kind != lexer.Semicolon && next.Kind != lexer.Comma {
				if trailingComment {
					space()
				} else {
					newline(false)
				}
			}
		case lexer.LBracket:
			write(token.Text)
			width, comma := listWidth(tokens, i, lexer.RBracket)
			multiline := comma && line+width > 100
			lists = append(lists, list{close: lexer.RBracket, multiline: multiline, indented: multiline})
			if multiline {
				indent++
				newline(false)
			}
		case lexer.RBracket:
			item := lists[len(lists)-1]
			lists = lists[:len(lists)-1]
			if item.multiline {
				if previous.Kind != lexer.Comma && previous.Kind != lexer.LBracket {
					write(",")
				}
				indent--
				newline(false)
			}
			write(token.Text)
		case lexer.Less:
			if previous.Text == "buffer" || previous.Text == "shared" || previous.Text == "atomic" || previous.Text == "transient" {
				write("<")
				generics++
			} else {
				if line >= 88 {
					indent++
					newline(false)
					write("<")
					indent--
					space()
					break
				}
				space()
				write("<")
				space()
			}
		case lexer.Greater:
			if generics > 0 {
				write(">")
				generics--
			} else {
				if line >= 88 {
					indent++
					newline(false)
					write(">")
					indent--
					space()
					break
				}
				space()
				write(">")
				space()
			}
		default:
			if operator(token.Kind) {
				if token.Kind == lexer.Question {
					questions = append(questions, len(lists))
				}
				unary := token.Kind == lexer.Bang || token.Kind == lexer.Tilde || token.Kind == lexer.Minus && (previous.Text == "return" || previous.Text == "over" || !endsExpression(previous.Kind))
				if unary {
					if previous.Text == "return" {
						space()
					}
					write(token.Text)
					break
				}
				if (line+segmentWidth(i) > 100 || token.Kind == lexer.Question && line+remainderWidth(i) > 96) && token.Kind != lexer.Assign {
					indent++
					newline(false)
					write(token.Text)
					indent--
					space()
					break
				}
				space()
				write(token.Text)
				space()
			} else {
				if token.Kind == lexer.At && previous.Kind == lexer.RParen {
					space()
				}
				word := token.Kind == lexer.Ident || token.Kind == lexer.Number || token.Kind == lexer.String
				previousWord := previous.Kind == lexer.Ident || previous.Kind == lexer.Number || previous.Kind == lexer.String || previous.Kind == lexer.RParen || previous.Kind == lexer.RBracket
				if word && previousWord {
					space()
				}
				write(token.Text)
			}
		}
		previous = token
	}
	formatted := strings.TrimRight(out.String(), " \t\r\n") + "\n"
	return formatted, nil
}

func listWidth(tokens []lexer.Token, start int, close lexer.Kind) (int, bool) {
	depth, width, comma := 0, 1, false
	for _, token := range tokens[start+1:] {
		switch token.Kind {
		case lexer.LParen, lexer.LBracket, lexer.LBrace:
			depth++
		case lexer.RParen, lexer.RBracket, lexer.RBrace:
			if depth == 0 && token.Kind == close {
				return width + 1, comma
			}
			depth--
		case lexer.Comma:
			comma = comma || depth == 0
		}
		width += len(token.Text) + 1
	}
	return width, comma
}

func endsExpression(kind lexer.Kind) bool {
	return kind == lexer.Ident || kind == lexer.Number || kind == lexer.String || kind == lexer.RParen || kind == lexer.RBracket || kind == lexer.RBrace || kind == lexer.PlusPlus || kind == lexer.MinusMinus
}
