package wgsl

import (
	"tach/src/backend"
	"tach/src/flow"
	"tach/src/ir"
)

type loweredFunction struct {
	source      *ir.Function
	coordinates *backend.Coordinates
}

type program struct {
	executable *backend.Executable
	source     *ir.Module
	functions  map[*ir.Function]*loweredFunction
	kernels    map[*ir.Function]*backend.PhysicalKernel
}

func Lower(logical *flow.Module) (*backend.Executable, error) {
	return backend.Lower(logical, backend.WebProfile)
}

func lower(executable *backend.Executable) (*program, error) {
	p := &program{executable: executable, source: executable.KernelModule, functions: map[*ir.Function]*loweredFunction{}, kernels: map[*ir.Function]*backend.PhysicalKernel{}}
	for i := range executable.PhysicalKernels {
		kernel := &executable.PhysicalKernels[i]
		p.functions[kernel.Function] = &loweredFunction{source: kernel.Function, coordinates: kernel.Coordinates}
		p.kernels[kernel.Function] = kernel
	}
	for _, function := range p.source.Functions {
		if function.Kind == ir.Helper {
			uses, _, err := ir.UseCounts(function)
			if err != nil {
				return nil, err
			}
			p.functions[function] = &loweredFunction{source: function, coordinates: &backend.Coordinates{Values: map[ir.ValueID]backend.Coordinate{}, Replaced: map[ir.ValueID]bool{}, Uses: uses}}
		}
	}
	return p, nil
}

func (f *loweredFunction) needs(space backend.CoordinateSpace) bool {
	for id, coordinate := range f.coordinates.Values {
		if coordinate.Space == space && f.coordinates.Uses[id] > 0 {
			return true
		}
	}
	return false
}
func (f *loweredFunction) needsGlobal() bool           { return f.needs(backend.Global) }
func (f *loweredFunction) needsLocal() bool            { return f.needs(backend.Local) }
func (f *loweredFunction) needsLocalLinear() bool      { return f.needs(backend.LocalLinear) }
func (f *loweredFunction) replaced(id ir.ValueID) bool { return f.coordinates.Replaced[id] }
func (f *loweredFunction) uses(id ir.ValueID) int      { return f.coordinates.Uses[id] }
func (f *loweredFunction) expression(id ir.ValueID) (string, bool) {
	coordinate, ok := f.coordinates.Values[id]
	if !ok {
		return "", false
	}
	name, suffix := "_tach_global_index", "."+[]string{"x", "y", "z"}[coordinate.Dimension]
	if coordinate.Space == backend.Local {
		name = "_tach_local_index"
	} else if coordinate.Space == backend.LocalLinear {
		name, suffix = "_tach_local_linear", ""
	}
	return name + suffix, true
}
