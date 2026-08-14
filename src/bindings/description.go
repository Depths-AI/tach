package bindings

import (
	"encoding/json"

	"tach/src/flow"
	"tach/src/ir"
	"tach/src/types"
)

type ProjectInput struct {
	Name, Version, Package, Title, Summary string
	Kernels                                []KernelInput
}

type KernelInput struct {
	Module, Name, Identity string
	Types, Functions       []string
	Documentation          flow.Documentation
}

type ProjectDescription struct {
	Schema  int                 `json:"schema"`
	Name    string              `json:"name"`
	Version string              `json:"version"`
	Package string              `json:"package"`
	Title   string              `json:"title"`
	Summary string              `json:"summary"`
	Modules []ModuleDescription `json:"modules"`
}

type ModuleDescription struct {
	Name    string              `json:"name"`
	Kernels []KernelDescription `json:"kernels"`
}

type KernelDescription struct {
	Name      string     `json:"name"`
	Identity  string     `json:"identity"`
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
	Role        string       `json:"role"`
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

func DescribeProject(module *flow.Module, input ProjectInput) ([]byte, error) {
	out := ProjectDescription{Schema: 2, Name: input.Name, Version: input.Version, Package: input.Package, Title: input.Title, Summary: input.Summary, Modules: []ModuleDescription{}}
	typesByName := map[string]*types.Type{}
	for _, item := range module.Kernel.Structs {
		typesByName[item.Name] = item
	}
	functions := map[string]*ir.Function{}
	for _, item := range module.Kernel.Functions {
		functions[item.Name] = item
	}
	programs := map[string]*flow.Program{}
	for _, item := range module.Programs {
		programs[item.Name] = item
	}
	moduleIndex := map[string]int{}
	for _, source := range input.Kernels {
		index, exists := moduleIndex[source.Module]
		if !exists {
			index = len(out.Modules)
			moduleIndex[source.Module] = index
			out.Modules = append(out.Modules, ModuleDescription{Name: source.Module, Kernels: []KernelDescription{}})
		}
		kernel := KernelDescription{Name: source.Name, Identity: source.Identity, Title: source.Documentation.Title, Summary: source.Documentation.Summary, Types: []Type{}, Functions: []Function{}}
		for _, name := range source.Types {
			declaration := typesByName[name]
			doc := source.Documentation.Types[name]
			item := Type{Name: name, Summary: doc.Summary, Fields: []Field{}}
			for _, field := range declaration.Fields {
				item.Fields = append(item.Fields, Field{Name: field.Name, Type: typeRef(field.Type), Description: doc.Fields[field.Name]})
			}
			kernel.Types = append(kernel.Types, item)
		}
		for _, name := range source.Functions {
			doc := source.Documentation.Functions[name]
			if program := programs[name]; program != nil {
				kernel.Functions = append(kernel.Functions, describeProgram(module, program, functions[name], doc))
			} else if function := functions[name]; function != nil {
				kernel.Functions = append(kernel.Functions, describeFunction(function, doc))
			}
		}
		out.Modules[index].Kernels = append(out.Modules[index].Kernels, kernel)
	}
	encoded, err := json.MarshalIndent(out, "", "  ")
	return append(encoded, '\n'), err
}

func describeFunction(function *ir.Function, doc flow.FunctionDocumentation) Function {
	role := "helper"
	if function.Kind == ir.Stage {
		role = "stage"
	}
	item := Function{Name: function.Name, Role: role, Summary: doc.Summary, Coordinates: []Coordinate{}, Parameters: []Parameter{}}
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
		return item
	}
	access, value := ir.AnalyzeAccess(function), 0
	for _, parameter := range function.SourceParams {
		if parameter.Kind == ir.SourceBuffer {
			buffer := function.BufferParams[parameter.Buffer]
			item.Parameters = append(item.Parameters, Parameter{Name: parameter.Name, Type: typeRef(buffer.Type), Buffer: true, Access: bufferAccess(access.Buffers[parameter.Buffer]).String(), Description: doc.Parameters[parameter.Name]})
		} else {
			item.Parameters = append(item.Parameters, Parameter{Name: parameter.Name, Type: typeRef(function.Params[value].Type), Description: doc.Parameters[parameter.Name]})
			value++
		}
	}
	return item
}

func describeProgram(module *flow.Module, program *flow.Program, stage *ir.Function, doc flow.FunctionDocumentation) Function {
	role := "program"
	if stage != nil {
		role = "kernel"
	}
	item := Function{Name: program.Name, Role: role, Exported: true, Summary: doc.Summary, Coordinates: []Coordinate{}, Parameters: []Parameter{}}
	if stage != nil {
		for _, coordinate := range stage.Indices {
			item.Coordinates = append(item.Coordinates, Coordinate{Name: coordinate.Name, Description: doc.Coordinates[coordinate.Name]})
		}
	}
	access := module.ProgramAccess(program)
	for _, parameter := range program.Parameters {
		item.Parameters = append(item.Parameters, Parameter{Name: parameter.Name, Type: typeRef(parameter.Type), Buffer: parameter.Kind == flow.BufferParameter, Access: access[parameter.Resource].String(), Description: doc.Parameters[parameter.Name]})
	}
	return item
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
