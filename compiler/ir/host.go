package ir

import (
	"fmt"
	"strings"
	"unicode"

	"tach/foundation"
)

func PrivateEntryName(index int) string { return fmt.Sprintf("_tach_k%d", index) }

// MangleIdentifier maps a Tach identifier to a conservative ASCII identifier for
// compiler-private symbols.
func MangleIdentifier(s string) string {
	var b strings.Builder
	for _, r := range s {
		if r == '_' {
			b.WriteString("__")
		} else if r < 128 && (unicode.IsLetter(r) || unicode.IsDigit(r)) {
			b.WriteRune(r)
		} else {
			fmt.Fprintf(&b, "_u%x_", r)
		}
	}
	if b.Len() == 0 {
		return "x"
	}
	return b.String()
}

func ValidateExportName(name string) error {
	if !asciiIdentifier(name) || typeScriptKeywords[name] {
		return fmt.Errorf("%q is not a valid generated export", name)
	}
	return nil
}

func ValidateExportTypeName(name string) error {
	if err := ValidateExportName(name); err != nil {
		return err
	}
	if name == "Float16Array" || name == "Float32Array" || name == "Int32Array" || name == "Uint32Array" || name == "ReadonlyArray" {
		return fmt.Errorf("%q collides with a generated host collection type", name)
	}
	return nil
}

func asciiIdentifier(s string) bool {
	if s == "" {
		return false
	}
	for i, r := range s {
		if r > unicode.MaxASCII || !(r == '_' || r == '$' || unicode.IsLetter(r) || i > 0 && unicode.IsDigit(r)) {
			return false
		}
	}
	return true
}

var typeScriptKeywords = func() map[string]bool {
	keywords := map[string]bool{}
	for _, keyword := range strings.Fields("abstract any arguments as asserts async await bigint boolean break case catch class const constructor continue debugger declare default delete do else enum eval export extends false finally for from function get global if implements import in infer instanceof interface intrinsic is keyof let module namespace never new null number object of override package private protected public readonly require return satisfies set static string super switch symbol this throw true try type typeof undefined unique unknown using var void while with yield") {
		keywords[keyword] = true
	}
	return keywords
}()

const maxHostParameterBytes = 16 * 1024

// HostParameterBlock is the shared physical plan for immutable kernel values.
type HostParameterBlock struct {
	Function *Function
	Binding  uint32
	Type     *foundation.Type
	Layout   foundation.TypeLayout
	Fields   []HostParameterField
}

type HostParameterField struct {
	Parameter int
	Path      []string
	Name      string
	Logical   *foundation.Type
	Physical  *foundation.Type
	Offset    uint32
}

func PlanHostParameters(function *Function, binding uint32) (*HostParameterBlock, error) {
	if function == nil || function.Kind != Stage {
		return nil, fmt.Errorf("parameter planning requires a stage")
	}
	if len(function.Params) == 0 {
		return nil, nil
	}
	block := &HostParameterBlock{Function: function, Binding: binding, Type: &foundation.Type{Kind: foundation.StructKind, Name: "__tach_parameters_" + MangleIdentifier(function.Name)}}
	for parameter, value := range function.Params {
		if !foundation.IsHostParameter(value.Type) {
			return nil, fmt.Errorf("kernel %s parameter %s: type %s cannot cross the host parameter ABI", function.Name, value.Name, value.Type)
		}
		if err := flattenHostParameter(block, parameter, nil, value.Type); err != nil {
			return nil, fmt.Errorf("kernel %s parameter %s: %w", function.Name, value.Name, err)
		}
	}
	physical, err := foundation.LayoutOf(block.Type)
	if err != nil {
		return nil, fmt.Errorf("kernel %s parameter block: %w", function.Name, err)
	}
	if physical.Size > maxHostParameterBytes {
		return nil, fmt.Errorf("kernel %s parameter block is %d bytes; portable limit is %d", function.Name, physical.Size, maxHostParameterBytes)
	}
	block.Layout = physical
	for index := range block.Fields {
		block.Fields[index].Offset = physical.Fields[index].Offset
	}
	return block, nil
}

func flattenHostParameter(block *HostParameterBlock, parameter int, path []string, logical *foundation.Type) error {
	if logical == nil {
		return fmt.Errorf("missing type")
	}
	if logical.Kind == foundation.StructKind {
		for _, field := range logical.Fields {
			if err := flattenHostParameter(block, parameter, appendPath(path, field.Name), field.Type); err != nil {
				return err
			}
		}
		return nil
	}
	physical := logical
	if logical.Kind == foundation.BoolKind {
		physical = foundation.Uint32Type
	} else if !foundation.IsNumeric(logical) {
		return fmt.Errorf("type %s cannot cross the host parameter ABI", logical)
	}
	name := fmt.Sprintf("f%d", len(block.Fields))
	block.Type.Fields = append(block.Type.Fields, foundation.TypeField{Name: name, Type: physical})
	block.Fields = append(block.Fields, HostParameterField{Parameter: parameter, Path: append([]string{}, path...), Name: name, Logical: logical, Physical: physical})
	return nil
}

func appendPath(path []string, name string) []string {
	out := make([]string, len(path)+1)
	copy(out, path)
	out[len(path)] = name
	return out
}
