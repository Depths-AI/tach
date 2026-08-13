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

type program struct {
	executable *backend.Executable
	source     *ir.Module
	functions  map[*ir.Function]*backend.Coordinates
	kernels    map[*ir.Function]*backend.PhysicalKernel
}

func Lower(logical *flow.Module) (*backend.Executable, error) {
	return backend.Lower(logical, backend.SPIRVProfile)
}

func lower(executable *backend.Executable) (*program, error) {
	functions, kernels, err := executable.IndexFunctions()
	if err != nil {
		return nil, err
	}
	return &program{executable: executable, source: executable.KernelModule, functions: functions, kernels: kernels}, nil
}

func inputs(f *ir.Function, coordinates *backend.Coordinates) map[inputKind]bool {
	used := map[inputKind]bool{}
	for id, coordinate := range coordinates.Values {
		if coordinates.Uses[id] == 0 {
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
	if len(f.WorkgroupVars) > 0 {
		used[inputLocalLinear] = true
	}
	return used
}
func coordinate(f *backend.Coordinates, id ir.ValueID) (inputKind, uint32) {
	coordinate := f.Values[id]
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
