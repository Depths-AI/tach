package abi

import (
	"fmt"

	"tach/src/ir"
	"tach/src/layout"
	"tach/src/types"
)

const maxParameterBytes = 16 * 1024

// ParameterPlan is the one cross-target physical plan for immutable kernel
// values. Core IR keeps those values as ordinary parameters; backends and host
// bindings share this plan when materializing one parameter block per kernel.
type ParameterPlan struct {
	Blocks     []*ParameterBlock
	byFunction map[*ir.Function]*ParameterBlock
}

type ParameterBlock struct {
	Function *ir.Function
	Binding  uint32
	Type     *types.Type
	Layout   layout.TypeLayout
	Fields   []ParameterField
}

type ParameterField struct {
	Parameter int
	Path      []string
	Name      string
	Logical   *types.Type
	Physical  *types.Type
	Offset    uint32
}

func (p *ParameterPlan) For(function *ir.Function) *ParameterBlock {
	if p == nil {
		return nil
	}
	return p.byFunction[function]
}

func PlanParameters(module *ir.Module) (*ParameterPlan, error) {
	plan := &ParameterPlan{Blocks: []*ParameterBlock{}, byFunction: map[*ir.Function]*ParameterBlock{}}
	for _, function := range module.Functions {
		if !function.Compute || len(function.Params) == 0 {
			continue
		}
		block := &ParameterBlock{
			Function: function,
			Binding:  uint32(len(module.Resources) + len(plan.Blocks)),
			Type:     &types.Type{Kind: types.Struct, Name: "__tach_parameters_" + Mangle(function.Name)},
		}
		for parameter, value := range function.Params {
			if err := flattenParameter(block, parameter, nil, value.Type); err != nil {
				return nil, fmt.Errorf("kernel %s parameter %s: %w", function.Name, value.Name, err)
			}
		}
		physical, err := layout.Of(block.Type)
		if err != nil {
			return nil, fmt.Errorf("kernel %s parameter block: %w", function.Name, err)
		}
		if physical.Size > maxParameterBytes {
			return nil, fmt.Errorf("kernel %s parameter block is %d bytes; portable limit is %d", function.Name, physical.Size, maxParameterBytes)
		}
		block.Layout = physical
		for index := range block.Fields {
			block.Fields[index].Offset = physical.Fields[index].Offset
		}
		plan.Blocks = append(plan.Blocks, block)
		plan.byFunction[function] = block
	}
	return plan, nil
}

func flattenParameter(block *ParameterBlock, parameter int, path []string, logical *types.Type) error {
	if logical == nil {
		return fmt.Errorf("missing type")
	}
	if logical.Kind == types.Struct {
		for _, field := range logical.Fields {
			if err := flattenParameter(block, parameter, appendPath(path, field.Name), field.Type); err != nil {
				return err
			}
		}
		return nil
	}
	physical := logical
	if logical.Kind == types.Bool {
		physical = types.TU32
	} else if !types.IsNumeric(logical) {
		return fmt.Errorf("type %s cannot cross the host parameter ABI", logical)
	}
	name := fmt.Sprintf("f%d", len(block.Fields))
	block.Type.Fields = append(block.Type.Fields, types.Field{Name: name, Type: physical})
	block.Fields = append(block.Fields, ParameterField{
		Parameter: parameter,
		Path:      append([]string{}, path...),
		Name:      name,
		Logical:   logical,
		Physical:  physical,
	})
	return nil
}

func appendPath(path []string, name string) []string {
	out := make([]string, len(path)+1)
	copy(out, path)
	out[len(path)] = name
	return out
}
