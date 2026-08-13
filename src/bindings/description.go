package bindings

import (
	"encoding/json"
	"path/filepath"

	"tach/src/flow"
	"tach/src/ir"
	"tach/src/types"
)

type Module struct {
	Schema    int        `json:"schema"`
	Source    string     `json:"source"`
	Title     string     `json:"title,omitempty"`
	Summary   string     `json:"summary,omitempty"`
	Types     []Type     `json:"types"`
	Functions []Function `json:"functions"`
}
type Type struct {
	Name    string  `json:"name"`
	Summary string  `json:"summary,omitempty"`
	Fields  []Field `json:"fields"`
}
type Field struct {
	Name        string  `json:"name"`
	Type        TypeRef `json:"type"`
	Description string  `json:"description,omitempty"`
}
type Function struct {
	Name        string       `json:"name"`
	Exported    bool         `json:"exported"`
	Summary     string       `json:"summary,omitempty"`
	Coordinates []Coordinate `json:"coordinates"`
	Parameters  []Parameter  `json:"parameters"`
	Returns     *Return      `json:"returns,omitempty"`
}
type Coordinate struct {
	Name        string `json:"name"`
	Description string `json:"description,omitempty"`
}
type Parameter struct {
	Name        string  `json:"name"`
	Type        TypeRef `json:"type"`
	Buffer      bool    `json:"buffer"`
	Access      string  `json:"access,omitempty"`
	Description string  `json:"description,omitempty"`
}
type Return struct {
	Type        TypeRef `json:"type"`
	Description string  `json:"description,omitempty"`
}
type TypeRef struct {
	Tach  string   `json:"tach"`
	Kind  string   `json:"kind"`
	Name  string   `json:"name,omitempty"`
	Elem  *TypeRef `json:"elem,omitempty"`
	Count uint32   `json:"count,omitempty"`
	Lanes int      `json:"lanes,omitempty"`
}

func Describe(module *flow.Module, sourceName string) ([]byte, error) {
	out := Module{Schema: 1, Source: filepath.Base(sourceName), Title: module.Documentation.Title, Summary: module.Documentation.Summary, Types: []Type{}, Functions: []Function{}}
	for _, t := range module.Kernel.Structs {
		if err := validateExportName(t.Name); err != nil {
			return nil, err
		}
		doc := module.Documentation.Types[t.Name]
		item := Type{Name: t.Name, Summary: doc.Summary, Fields: []Field{}}
		for _, field := range t.Fields {
			item.Fields = append(item.Fields, Field{Name: field.Name, Type: typeRef(field.Type), Description: doc.Fields[field.Name]})
		}
		out.Types = append(out.Types, item)
	}
	programs := map[string]*flow.Program{}
	for _, program := range module.Programs {
		if err := validateExportName(program.Name); err != nil {
			return nil, err
		}
		programs[program.Name] = program
	}
	for _, function := range module.Kernel.Functions {
		doc := module.Documentation.Functions[function.Name]
		program := programs[function.Name]
		if program != nil {
			continue
		}
		item := Function{Name: function.Name, Summary: doc.Summary, Coordinates: []Coordinate{}, Parameters: []Parameter{}}
		for _, coordinate := range function.Indices {
			item.Coordinates = append(item.Coordinates, Coordinate{Name: coordinate.Name, Description: doc.Coordinates[coordinate.Name]})
		}
		if function.Kind == ir.Helper {
			for _, parameter := range function.Params {
				item.Parameters = append(item.Parameters, Parameter{Name: parameter.Name, Type: typeRef(parameter.Type), Description: doc.Parameters[parameter.Name]})
			}
			if function.Return.Kind != types.Void {
				item.Returns = &Return{Type: typeRef(function.Return), Description: doc.Returns}
			}
		} else {
			summary := ir.AnalyzeAccess(function)
			valueParameter := 0
			for _, parameter := range function.SourceParams {
				if parameter.Kind == ir.SourceBuffer {
					buffer := function.BufferParams[parameter.Buffer]
					item.Parameters = append(item.Parameters, Parameter{Name: parameter.Name, Type: typeRef(buffer.Type), Buffer: true, Access: bufferAccess(summary.Buffers[parameter.Buffer]).String(), Description: doc.Parameters[parameter.Name]})
				} else {
					item.Parameters = append(item.Parameters, Parameter{Name: parameter.Name, Type: typeRef(function.Params[valueParameter].Type), Description: doc.Parameters[parameter.Name]})
					valueParameter++
				}
			}
		}
		out.Functions = append(out.Functions, item)
	}
	for _, program := range module.Programs {
		doc := module.Documentation.Functions[program.Name]
		item := Function{Name: program.Name, Exported: true, Summary: doc.Summary, Coordinates: []Coordinate{}, Parameters: []Parameter{}}
		if stage := module.Kernel.Function(program.Name); stage != nil {
			for _, coordinate := range stage.Indices {
				item.Coordinates = append(item.Coordinates, Coordinate{Name: coordinate.Name, Description: doc.Coordinates[coordinate.Name]})
			}
		}
		access := module.ProgramAccess(program)
		for _, parameter := range program.Parameters {
			item.Parameters = append(item.Parameters, Parameter{Name: parameter.Name, Type: typeRef(parameter.Type), Buffer: parameter.Kind == flow.BufferParameter, Access: access[parameter.Resource].String(), Description: doc.Parameters[parameter.Name]})
		}
		out.Functions = append(out.Functions, item)
	}
	encoded, err := json.MarshalIndent(out, "", "  ")
	return append(encoded, '\n'), err
}

func bufferAccess(buffer ir.BufferSummary) flow.ResourceAccess {
	if buffer.Atomic {
		return flow.AtomicAccess
	}
	if buffer.Read && buffer.Write {
		return flow.ReadWriteAccess
	}
	if buffer.Write {
		return flow.WriteAccess
	}
	return flow.ReadAccess
}

func typeRef(t *types.Type) TypeRef {
	kinds := [...]string{"invalid", "void", "bool", "i32", "u32", "f32", "vector", "struct", "atomic", "fixedArray", "runtimeArray"}
	out := TypeRef{Tach: t.String(), Kind: kinds[t.Kind]}
	switch t.Kind {
	case types.Struct:
		out.Name = t.Name
	case types.Vector:
		out.Lanes = t.Lanes
		item := typeRef(t.Elem)
		out.Elem = &item
	case types.Atomic, types.FixedArray, types.RuntimeArray:
		out.Count = t.Count
		item := typeRef(t.Elem)
		out.Elem = &item
	}
	return out
}
