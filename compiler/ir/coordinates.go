package ir

import (
	"strconv"

	"tach/foundation"
)

type CoordinateSpace uint8

const (
	Global CoordinateSpace = iota + 1
	Local
	LocalLinear
)

type Coordinate struct {
	Space     CoordinateSpace
	Dimension int
}

type Coordinates struct {
	Values   map[ValueID]Coordinate
	Order    []ValueID
	Replaced map[ValueID]bool
	Uses     map[ValueID]int
}

func ResolveCoordinates(f *Function) (*Coordinates, error) {
	uses, _, err := UseCounts(f)
	if err != nil {
		return nil, err
	}
	coordinates := &Coordinates{
		Values:   map[ValueID]Coordinate{},
		Replaced: map[ValueID]bool{},
		Uses:     uses,
	}
	for dimension, index := range f.Indices {
		coordinates.Values[index.ID] = Coordinate{Space: Global, Dimension: dimension}
		coordinates.Order = append(coordinates.Order, index.ID)
	}
	return coordinates, nil
}

func ResolveFunctionCoordinates(module *KernelModule, known map[*Function]*Coordinates) (map[*Function]*Coordinates, error) {
	for _, function := range module.Functions {
		if known[function] != nil {
			continue
		}
		coordinates, err := ResolveCoordinates(function)
		if err != nil {
			return nil, err
		}
		known[function] = coordinates
	}
	return known, nil
}

func ChooseWorkgroup(function *Function, maximum [3]uint32, invocations uint32) ([3]uint32, bool) {
	if function.Workgroup.Explicit {
		size := function.Workgroup.Size
		product := uint64(1)
		for axis, value := range size {
			if value == 0 || value > maximum[axis] {
				return size, false
			}
			product *= uint64(value)
		}
		return size, product <= uint64(invocations)
	}
	size := [][3]uint32{{256, 1, 1}, {16, 16, 1}, {8, 8, 4}}[len(function.Indices)-1]
	for {
		valid := uint64(size[0])*uint64(size[1])*uint64(size[2]) <= uint64(invocations)
		for axis := range size {
			valid = valid && size[axis] <= maximum[axis]
		}
		if valid {
			return size, true
		}
		for axis := len(function.Indices) - 1; axis >= 0; axis-- {
			if size[axis] > 1 {
				size[axis] = (size[axis] + 1) / 2
				break
			}
		}
	}
}

func OptimizeCoordinates(f *Function, workgroup [3]uint32, coordinates *Coordinates) {
	// Coefficients use uint32 arithmetic, matching Core exactly. The row-major
	// [1, size.x, size.x*size.y] form is the portable local-linear coordinate.
	want := [3]uint32{}
	stride := uint32(1)
	for dimension := range f.Indices {
		want[dimension] = stride
		stride *= workgroup[dimension]
	}
	optimizeBlock(f, workgroup, coordinates, f.Body, map[ValueID]uint32{}, map[ValueID]*Binary{}, map[ValueID][3]uint32{}, want)
}

func optimizeBlock(f *Function, workgroup [3]uint32, coordinates *Coordinates, block *Block, constants map[ValueID]uint32, definitions map[ValueID]*Binary, forms map[ValueID][3]uint32, want [3]uint32) {
	for _, in := range block.Instrs {
		switch x := in.(type) {
		case *Const:
			if x.Type.Kind == foundation.Uint32Kind {
				if value, err := strconv.ParseUint(x.Raw, 10, 32); err == nil {
					constants[x.Result] = uint32(value)
				}
			}
		case *Binary:
			definitions[x.Result] = x
			origin, ok := coordinates.Values[x.Left]
			size, constant := constants[x.Right]
			if ok && constant && origin.Space == Global && x.Op == "%" && size == workgroup[origin.Dimension] {
				space := Local
				if len(f.Indices) == 1 {
					space = LocalLinear
				}
				coordinates.Values[x.Result] = Coordinate{Space: space, Dimension: origin.Dimension}
				coordinates.Order = append(coordinates.Order, x.Result)
				form := [3]uint32{}
				form[origin.Dimension] = 1
				forms[x.Result] = form
				replaceBinary(x, coordinates, definitions)
				continue
			}
			form, ok := affineForm(x, forms, constants)
			if !ok {
				continue
			}
			forms[x.Result] = form
			if form != want {
				continue
			}
			coordinates.Values[x.Result] = Coordinate{Space: LocalLinear}
			coordinates.Order = append(coordinates.Order, x.Result)
			replaceBinary(x, coordinates, definitions)
		case *If:
			optimizeBlock(f, workgroup, coordinates, x.Then, constants, definitions, forms, want)
			optimizeBlock(f, workgroup, coordinates, x.Else, constants, definitions, forms, want)
		case *Loop:
			optimizeBlock(f, workgroup, coordinates, x.Cond, constants, definitions, forms, want)
			optimizeBlock(f, workgroup, coordinates, x.Body, constants, definitions, forms, want)
		case *Scope:
			optimizeBlock(f, workgroup, coordinates, x.Body, constants, definitions, forms, want)
		}
	}
}

func affineForm(x *Binary, forms map[ValueID][3]uint32, constants map[ValueID]uint32) ([3]uint32, bool) {
	left, leftOK := forms[x.Left]
	right, rightOK := forms[x.Right]
	switch x.Op {
	case "+":
		if !leftOK || !rightOK {
			return [3]uint32{}, false
		}
		for i := range left {
			left[i] += right[i]
		}
		return left, true
	case "*":
		factor, constant := constants[x.Right]
		if leftOK && constant {
			for i := range left {
				left[i] *= factor
			}
			return left, true
		}
		factor, constant = constants[x.Left]
		if !rightOK || !constant {
			return [3]uint32{}, false
		}
		for i := range right {
			right[i] *= factor
		}
		return right, true
	}
	return [3]uint32{}, false
}

func replaceBinary(x *Binary, coordinates *Coordinates, definitions map[ValueID]*Binary) {
	// A lowered instruction stops using its Core operands exactly once.
	if coordinates.Replaced[x.Result] {
		return
	}
	coordinates.Replaced[x.Result] = true
	coordinates.Uses[x.Left]--
	coordinates.Uses[x.Right]--
	discardDeadBinary(x.Left, coordinates, definitions)
	discardDeadBinary(x.Right, coordinates, definitions)
}

func discardDeadBinary(id ValueID, coordinates *Coordinates, definitions map[ValueID]*Binary) {
	x := definitions[id]
	if x == nil || coordinates.Uses[id] != 0 || coordinates.Replaced[id] {
		return
	}
	replaceBinary(x, coordinates, definitions)
}
