package wgsl

import (
	"tach/src/backend"
	"tach/src/flow"
	"tach/src/ir"
)

type program struct {
	executable *backend.Executable
	source     *ir.Module
	functions  map[*ir.Function]*backend.Coordinates
	kernels    map[*ir.Function]*backend.PhysicalKernel
}

func Lower(logical *flow.Module) (*backend.Executable, error) {
	return backend.Lower(logical, backend.WebProfile)
}

func lower(executable *backend.Executable) (*program, error) {
	functions, kernels, err := executable.IndexFunctions()
	if err != nil {
		return nil, err
	}
	return &program{executable: executable, source: executable.KernelModule, functions: functions, kernels: kernels}, nil
}

func needs(f *backend.Coordinates, space backend.CoordinateSpace) bool {
	for id, coordinate := range f.Values {
		if coordinate.Space == space && f.Uses[id] > 0 {
			return true
		}
	}
	return false
}
func expression(f *backend.Coordinates, id ir.ValueID) (string, bool) {
	coordinate, ok := f.Values[id]
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
