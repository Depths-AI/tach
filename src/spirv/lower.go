package spirv

import (
	"tach/src/backend"
	"tach/src/ir"
)

type inputKind uint8

const (
	inputGlobalIndex inputKind = iota + 1
	inputLocalIndex
	inputLocalLinear
)

type loweredFunction struct {
	source      *ir.Function
	coordinates *backend.Coordinates
}

type program struct {
	source    *ir.Module
	functions map[*ir.Function]*loweredFunction
}

func lower(m *ir.Module) (*program, error) {
	p := &program{source: m, functions: map[*ir.Function]*loweredFunction{}}
	for _, f := range m.Functions {
		coordinates, err := backend.LowerCoordinates(f)
		if err != nil {
			return nil, err
		}
		p.functions[f] = &loweredFunction{source: f, coordinates: coordinates}
	}
	return p, nil
}

func optimize(p *program) {
	for _, f := range p.functions {
		backend.OptimizeCoordinates(f.source, f.coordinates)
	}
}

func (f *loweredFunction) inputs() map[inputKind]bool {
	used := map[inputKind]bool{}
	for id, coordinate := range f.coordinates.Values {
		if f.coordinates.Uses[id] == 0 {
			continue
		}
		switch coordinate.Space {
		case backend.Global:
			used[inputGlobalIndex] = true
		case backend.Local:
			used[inputLocalIndex] = true
		case backend.LocalLinear:
			used[inputLocalLinear] = true
		}
	}
	if len(f.source.WorkgroupVars) > 0 {
		used[inputLocalLinear] = true
	}
	return used
}

func (f *loweredFunction) used(id ir.ValueID) bool { return f.coordinates.Uses[id] > 0 }
func (f *loweredFunction) replaced(id ir.ValueID) bool {
	return f.coordinates.Replaced[id]
}
func (f *loweredFunction) coordinate(id ir.ValueID) (inputKind, uint32) {
	coordinate := f.coordinates.Values[id]
	switch coordinate.Space {
	case backend.Global:
		return inputGlobalIndex, uint32(coordinate.Dimension)
	case backend.Local:
		return inputLocalIndex, uint32(coordinate.Dimension)
	case backend.LocalLinear:
		return inputLocalLinear, 0
	}
	panic("unknown lowered coordinate space")
}
