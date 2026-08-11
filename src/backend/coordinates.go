package backend

import (
	"strconv"

	"tach/src/ir"
	"tach/src/types"
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
	Values   map[ir.ValueID]Coordinate
	Order    []ir.ValueID
	Replaced map[ir.ValueID]bool
	Uses     map[ir.ValueID]int
}

func LowerCoordinates(f *ir.Function) (*Coordinates, error) {
	uses, _, err := ir.UseCounts(f)
	if err != nil {
		return nil, err
	}
	coordinates := &Coordinates{
		Values:   map[ir.ValueID]Coordinate{},
		Replaced: map[ir.ValueID]bool{},
		Uses:     uses,
	}
	for dimension, index := range f.Indices {
		coordinates.Values[index.ID] = Coordinate{Space: Global, Dimension: dimension}
		coordinates.Order = append(coordinates.Order, index.ID)
	}
	return coordinates, nil
}

func OptimizeCoordinates(f *ir.Function, coordinates *Coordinates) {
	// Coefficients use u32 arithmetic, matching Core exactly. The row-major
	// [1, size.x, size.x*size.y] form is the portable local-linear coordinate.
	want := [3]uint32{}
	stride := uint32(1)
	for dimension := range f.Indices {
		want[dimension] = stride
		stride *= f.Workgroup[dimension]
	}
	optimizeBlock(f, coordinates, f.Body, map[ir.ValueID]uint32{}, map[ir.ValueID]*ir.Binary{}, map[ir.ValueID][3]uint32{}, want)
}

func optimizeBlock(f *ir.Function, coordinates *Coordinates, block *ir.Block, constants map[ir.ValueID]uint32, definitions map[ir.ValueID]*ir.Binary, forms map[ir.ValueID][3]uint32, want [3]uint32) {
	for _, in := range block.Instrs {
		switch x := in.(type) {
		case *ir.Const:
			if x.Type.Kind == types.U32 {
				if value, err := strconv.ParseUint(x.Raw, 10, 32); err == nil {
					constants[x.Result] = uint32(value)
				}
			}
		case *ir.Binary:
			definitions[x.Result] = x
			origin, ok := coordinates.Values[x.Left]
			size, constant := constants[x.Right]
			if ok && constant && origin.Space == Global && x.Op == "%" && size == f.Workgroup[origin.Dimension] {
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
		case *ir.If:
			optimizeBlock(f, coordinates, x.Then, constants, definitions, forms, want)
			optimizeBlock(f, coordinates, x.Else, constants, definitions, forms, want)
		case *ir.Loop:
			optimizeBlock(f, coordinates, x.Cond, constants, definitions, forms, want)
			optimizeBlock(f, coordinates, x.Body, constants, definitions, forms, want)
		}
	}
}

func affineForm(x *ir.Binary, forms map[ir.ValueID][3]uint32, constants map[ir.ValueID]uint32) ([3]uint32, bool) {
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

func replaceBinary(x *ir.Binary, coordinates *Coordinates, definitions map[ir.ValueID]*ir.Binary) {
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

func discardDeadBinary(id ir.ValueID, coordinates *Coordinates, definitions map[ir.ValueID]*ir.Binary) {
	x := definitions[id]
	if x == nil || coordinates.Uses[id] != 0 || coordinates.Replaced[id] {
		return
	}
	replaceBinary(x, coordinates, definitions)
}
