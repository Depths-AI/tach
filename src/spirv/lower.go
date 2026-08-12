package spirv

import (
	"tach/src/backend"
	"tach/src/flow"
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
	executable *backend.Executable
	source     *ir.Module
	functions  map[*ir.Function]*loweredFunction
	kernels    map[*ir.Function]*backend.PhysicalKernel
}

func Lower(logical *flow.Module) (*backend.Executable, error) {
	return backend.Lower(logical, backend.SPIRVProfile)
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
func (f *loweredFunction) used(id ir.ValueID) bool     { return f.coordinates.Uses[id] > 0 }
func (f *loweredFunction) replaced(id ir.ValueID) bool { return f.coordinates.Replaced[id] }
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
