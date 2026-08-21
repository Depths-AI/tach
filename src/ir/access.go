package ir

import (
	"slices"
	"strconv"

	"tach/src/source"
)

type MemoryAccessKind uint8

const (
	MemoryRead MemoryAccessKind = iota + 1
	MemoryWrite
	MemoryAtomic
)

type Affine struct {
	Exact       bool
	Coefficient [3]int64
	Constant    int64
}

type MemoryAccess struct {
	Buffer    int
	FieldPath []int
	Kind      MemoryAccessKind
	Indices   []Affine
	Value     ValueID
	Span      source.Span
	complete  bool
}

type BufferSummary struct {
	Read            bool
	Write           bool
	Atomic          bool
	CoordinateWrite bool
	Accesses        []MemoryAccess
}

type EffectSummary struct {
	Memory    bool
	Atomic    bool
	Workgroup bool
	Barrier   bool
}

type AccessSummary struct {
	Buffers          []BufferSummary
	Effects          EffectSummary
	InstructionCount int
	PeakLive         int
	SharedBytes      uint32
}

type placeAccess struct {
	buffer int
	fields []int
	index  []ValueID
}

func AnalyzeAccess(function *Function) AccessSummary {
	summary := AccessSummary{Buffers: make([]BufferSummary, len(function.BufferParams))}
	definitions := map[ValueID]Instr{}
	coordinates := map[ValueID]int{}
	for dimension, parameter := range function.Indices {
		coordinates[parameter.ID] = dimension
	}
	places := map[PlaceID]placeAccess{}
	earlyReturn := false
	var walk func(*Block, []ValueID, bool)
	walk = func(block *Block, guards []ValueID, eligible bool) {
		if block == nil {
			return
		}
		if _, ok := block.Term.(*Return); block != function.Body && ok {
			earlyReturn = true
		}
		for _, instruction := range block.Instrs {
			summary.InstructionCount++
			if definition, ok := instruction.(ValueDef); ok && definition.ResultValue() != 0 {
				definitions[definition.ResultValue()] = instruction
			}
			switch x := instruction.(type) {
			case *PlaceRoot:
				places[x.Result] = placeAccess{buffer: x.Buffer}
			case *PlaceField:
				p := places[x.Base]
				p.fields = append(append([]int(nil), p.fields...), x.Field)
				places[x.Result] = p
			case *PlaceIndex:
				p := places[x.Base]
				p.index = append(append([]ValueID(nil), p.index...), x.Index)
				places[x.Result] = p
			case *Load:
				addMemoryAccess(&summary, places[x.Place], MemoryRead, 0, x.Span, false, definitions, coordinates)
			case *Store:
				place := places[x.Place]
				complete := eligible
				for _, guard := range guards {
					complete = complete && lengthGuard(guard, place, definitions, places, coordinates)
				}
				addMemoryAccess(&summary, place, MemoryWrite, x.Value, x.Span, complete, definitions, coordinates)
			case *Atomic:
				addMemoryAccess(&summary, places[x.Place], MemoryAtomic, x.Value, x.Span, false, definitions, coordinates)
				summary.Effects.Atomic = true
			case *Barrier:
				summary.Effects.Barrier = true
				summary.Effects.Workgroup = true
			case *PlaceWorkgroup:
				summary.Effects.Workgroup = true
				places[x.Result] = placeAccess{buffer: -1}
			case *If:
				walk(x.Then, append(append([]ValueID(nil), guards...), x.Cond), eligible)
				walk(x.Else, guards, false)
			case *Loop:
				walk(x.Cond, guards, false)
				walk(x.Body, guards, false)
			case *Scope:
				walk(x.Body, guards, eligible)
			}
		}
	}
	walk(function.Body, nil, true)
	for i := range summary.Buffers {
		buffer := &summary.Buffers[i]
		if len(buffer.Accesses) == 1 && buffer.Accesses[0].Kind == MemoryWrite && buffer.Accesses[0].complete && !earlyReturn && identityMap(buffer.Accesses[0].Indices, len(function.Indices)) {
			buffer.CoordinateWrite = true
		}
		for _, access := range buffer.Accesses {
			switch access.Kind {
			case MemoryRead:
				buffer.Read = true
			case MemoryWrite:
				buffer.Write = true
			case MemoryAtomic:
				buffer.Read = true
				buffer.Write = true
				buffer.Atomic = true
			}
		}
		if buffer.Read || buffer.Write {
			summary.Effects.Memory = true
		}
	}
	summary.PeakLive = summary.InstructionCount
	return summary
}

func addMemoryAccess(summary *AccessSummary, place placeAccess, kind MemoryAccessKind, value ValueID, span source.Span, complete bool, definitions map[ValueID]Instr, coordinates map[ValueID]int) {
	if place.buffer < 0 || place.buffer >= len(summary.Buffers) {
		return
	}
	access := MemoryAccess{Buffer: place.buffer, FieldPath: append([]int(nil), place.fields...), Kind: kind, Value: value, Span: span, complete: complete}
	for _, index := range place.index {
		access.Indices = append(access.Indices, affineValue(index, definitions, coordinates, map[ValueID]bool{}))
	}
	summary.Buffers[place.buffer].Accesses = append(summary.Buffers[place.buffer].Accesses, access)
}

func lengthGuard(id ValueID, target placeAccess, definitions map[ValueID]Instr, places map[PlaceID]placeAccess, coordinates map[ValueID]int) bool {
	comparison, ok := definitions[id].(*Binary)
	if !ok || comparison.Op != "<" || !identityMap([]Affine{affineValue(comparison.Left, definitions, coordinates, map[ValueID]bool{})}, 1) {
		return false
	}
	length, ok := definitions[comparison.Right].(*ArrayLength)
	if !ok {
		return false
	}
	limit, found := places[length.Place]
	return found && limit.buffer == target.buffer && slices.Equal(limit.fields, target.fields)
}

func affineValue(id ValueID, definitions map[ValueID]Instr, coordinates map[ValueID]int, active map[ValueID]bool) Affine {
	if dimension, ok := coordinates[id]; ok {
		out := Affine{Exact: true}
		out.Coefficient[dimension] = 1
		return out
	}
	if active[id] {
		return Affine{}
	}
	active[id] = true
	defer delete(active, id)
	switch x := definitions[id].(type) {
	case *Const:
		value, err := strconv.ParseUint(x.Raw, 0, 32)
		if err != nil {
			return Affine{}
		}
		return Affine{Exact: true, Constant: int64(value)}
	case *Binary:
		left := affineValue(x.Left, definitions, coordinates, active)
		right := affineValue(x.Right, definitions, coordinates, active)
		if !left.Exact || !right.Exact {
			return Affine{}
		}
		switch x.Op {
		case "+", "-":
			factor := int64(1)
			if x.Op == "-" {
				factor = -1
			}
			for i := range left.Coefficient {
				left.Coefficient[i] += factor * right.Coefficient[i]
			}
			left.Constant += factor * right.Constant
			return left
		case "*":
			if constantAffine(left) {
				return scaleAffine(right, left.Constant)
			}
			if constantAffine(right) {
				return scaleAffine(left, right.Constant)
			}
		}
	}
	return Affine{}
}

func constantAffine(value Affine) bool { return value.Exact && value.Coefficient == [3]int64{} }
func scaleAffine(value Affine, factor int64) Affine {
	for i := range value.Coefficient {
		value.Coefficient[i] *= factor
	}
	value.Constant *= factor
	return value
}
func identityMap(indices []Affine, rank int) bool {
	if len(indices) != 1 || rank != 1 {
		return false
	}
	want := Affine{Exact: true}
	want.Coefficient[0] = 1
	return indices[0] == want
}
