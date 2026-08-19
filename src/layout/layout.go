package layout

import (
	"fmt"

	"tach/src/types"
)

// Tach's portable host ABI uses one canonical layout for every host-visible
// structure. Backends adapt their declarations to these compiler-owned bytes.
type TypeLayout struct {
	Size    uint32
	Align   uint32
	Stride  uint32
	Fields  []FieldLayout
	Runtime bool
}

type FieldLayout struct {
	Name   string
	Offset uint32
	Layout TypeLayout
}

func checkedSize(value uint64) (uint32, error) {
	if value > uint64(^uint32(0)) {
		return 0, fmt.Errorf("host layout exceeds the 32-bit ABI size limit")
	}
	return uint32(value), nil
}

func roundUp(a, v uint32) (uint32, error) {
	if a == 0 {
		return v, nil
	}
	return checkedSize((uint64(v) + uint64(a) - 1) &^ uint64(a-1))
}

func Of(t *types.Type) (TypeLayout, error) {
	seen := map[string]bool{}
	return of(t, seen)
}

func of(t *types.Type, seen map[string]bool) (TypeLayout, error) {
	if t == nil {
		return TypeLayout{}, fmt.Errorf("nil type")
	}
	switch t.Kind {
	case types.F16:
		return TypeLayout{Size: 2, Align: 2}, nil
	case types.I32, types.U32, types.F32, types.Atomic:
		return TypeLayout{Size: 4, Align: 4}, nil
	case types.Bool:
		return TypeLayout{}, fmt.Errorf("bool is a value type and has no direct Tach buffer representation")
	case types.Vector:
		if !types.IsNumericScalar(t.Elem) {
			return TypeLayout{}, fmt.Errorf("vector element %s is not host-shareable", t.Elem)
		}
		element, err := of(t.Elem, seen)
		if err != nil {
			return TypeLayout{}, err
		}
		switch t.Lanes {
		case 2:
			return TypeLayout{Size: element.Size * 2, Align: element.Align * 2}, nil
		case 3:
			return TypeLayout{Size: element.Size * 3, Align: element.Align * 4}, nil
		case 4:
			return TypeLayout{Size: element.Size * 4, Align: element.Align * 4}, nil
		default:
			return TypeLayout{}, fmt.Errorf("unsupported vector width %d", t.Lanes)
		}
	case types.FixedArray:
		e, err := of(t.Elem, seen)
		if err != nil {
			return TypeLayout{}, err
		}
		if e.Runtime || t.Count == 0 {
			return TypeLayout{}, fmt.Errorf("invalid fixed array %s", t)
		}
		stride, err := roundUp(e.Align, e.Size)
		if err != nil {
			return TypeLayout{}, err
		}
		size, err := checkedSize(uint64(stride) * uint64(t.Count))
		if err != nil {
			return TypeLayout{}, err
		}
		return TypeLayout{Size: size, Align: e.Align, Stride: stride}, nil
	case types.RuntimeArray:
		e, err := of(t.Elem, seen)
		if err != nil {
			return TypeLayout{}, err
		}
		// Runtime arrays are buffer-only and use their element's natural stride.
		stride, err := roundUp(e.Align, e.Size)
		if err != nil {
			return TypeLayout{}, err
		}
		return TypeLayout{Align: e.Align, Stride: stride, Runtime: true}, nil
	case types.Struct:
		if seen[t.Name] {
			return TypeLayout{}, fmt.Errorf("recursive host type %s", t.Name)
		}
		seen[t.Name] = true
		defer delete(seen, t.Name)
		// 16-byte struct alignment ensures the type is directly usable in strict
		// uniform layouts. Member placement additionally reserves at least a
		// 16-byte multiple after nested structs.
		align := uint32(16)
		off := uint32(0)
		fields := make([]FieldLayout, 0, len(t.Fields))
		runtimeSeen := false
		for i, f := range t.Fields {
			fl, err := of(f.Type, seen)
			if err != nil {
				return TypeLayout{}, fmt.Errorf("%s.%s: %w", t.Name, f.Name, err)
			}
			if runtimeSeen {
				return TypeLayout{}, fmt.Errorf("runtime array in %s must be the final member", t.Name)
			}
			req := fl.Align
			if f.Type.Kind == types.Struct && req < 16 {
				req = 16
			}
			off, err = roundUp(req, off)
			if err != nil {
				return TypeLayout{}, err
			}
			fields = append(fields, FieldLayout{Name: f.Name, Offset: off, Layout: fl})
			if fl.Runtime {
				runtimeSeen = true
			} else {
				sz := fl.Size
				if f.Type.Kind == types.Struct {
					sz, err = roundUp(16, sz)
					if err != nil {
						return TypeLayout{}, err
					}
				}
				off, err = checkedSize(uint64(off) + uint64(sz))
				if err != nil {
					return TypeLayout{}, err
				}
			}
			if req > align {
				align = req
			}
			if fl.Runtime && i != len(t.Fields)-1 {
				return TypeLayout{}, fmt.Errorf("runtime array in %s must be the final member", t.Name)
			}
		}
		if runtimeSeen {
			return TypeLayout{Size: off, Align: align, Fields: fields, Runtime: true}, nil
		}
		size, err := roundUp(align, off)
		if err != nil {
			return TypeLayout{}, err
		}
		return TypeLayout{Size: size, Align: align, Fields: fields}, nil
	case types.Void:
		return TypeLayout{}, fmt.Errorf("void has no host layout")
	default:
		return TypeLayout{}, fmt.Errorf("unsupported host type %s", t)
	}
}
