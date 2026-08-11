package wgsl

import (
	"tach/src/backend"
	"tach/src/ir"
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

func (f *loweredFunction) needs(space backend.CoordinateSpace) bool {
	for id, coordinate := range f.coordinates.Values {
		if coordinate.Space == space && f.coordinates.Uses[id] > 0 {
			return true
		}
	}
	return false
}

func (f *loweredFunction) needsGlobal() bool { return f.needs(backend.Global) }
func (f *loweredFunction) needsLocal() bool  { return f.needs(backend.Local) }
func (f *loweredFunction) needsLocalLinear() bool {
	return f.needs(backend.LocalLinear)
}
func (f *loweredFunction) replaced(id ir.ValueID) bool {
	return f.coordinates.Replaced[id]
}
func (f *loweredFunction) uses(id ir.ValueID) int { return f.coordinates.Uses[id] }

func (f *loweredFunction) expression(id ir.ValueID) (string, bool) {
	coordinate, ok := f.coordinates.Values[id]
	if !ok {
		return "", false
	}
	name := "_tach_global_index"
	suffix := "." + []string{"x", "y", "z"}[coordinate.Dimension]
	switch coordinate.Space {
	case backend.Local:
		name = "_tach_local_index"
	case backend.LocalLinear:
		name = "_tach_local_linear"
		suffix = ""
	}
	return name + suffix, true
}
