package layout

import (
	"fmt"

	"tach/src/types"
)

// Tach's portable host ABI intentionally uses one layout for every host-visible
// structure. It is the strict superset of WGSL storage layout and the legacy
// uniform layout constraints, so the same bytes are valid for WebGPU storage,
// WebGPU uniform, and Vulkan block layouts.
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

func roundUp(a, v uint32) uint32 {
	if a == 0 {
		return v
	}
	return (v + a - 1) &^ (a - 1)
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
	case types.I32, types.U32, types.F32, types.Atomic:
		return TypeLayout{Size: 4, Align: 4}, nil
	case types.Bool:
		return TypeLayout{}, fmt.Errorf("bool is a control/value type and is not part of Tach's host ABI")
	case types.Vector:
		if t.Elem.Kind != types.I32 && t.Elem.Kind != types.U32 && t.Elem.Kind != types.F32 {
			return TypeLayout{}, fmt.Errorf("vector element %s is not host-shareable", t.Elem)
		}
		switch t.Lanes {
		case 2:
			return TypeLayout{Size: 8, Align: 8}, nil
		case 3:
			return TypeLayout{Size: 12, Align: 16}, nil
		case 4:
			return TypeLayout{Size: 16, Align: 16}, nil
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
		stride := roundUp(e.Align, e.Size)
		return TypeLayout{Size: stride * t.Count, Align: e.Align, Stride: stride}, nil
	case types.RuntimeArray:
		e, err := of(t.Elem, seen)
		if err != nil {
			return TypeLayout{}, err
		}
		// Runtime arrays are storage-only. Their natural stride remains valid for
		// both WGSL storage and Vulkan storage buffers.
		stride := roundUp(e.Align, e.Size)
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
			off = roundUp(req, off)
			fields = append(fields, FieldLayout{Name: f.Name, Offset: off, Layout: fl})
			if fl.Runtime {
				runtimeSeen = true
			} else {
				sz := fl.Size
				if f.Type.Kind == types.Struct {
					sz = roundUp(16, sz)
				}
				off += sz
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
		return TypeLayout{Size: roundUp(align, off), Align: align, Fields: fields}, nil
	case types.Void:
		return TypeLayout{}, fmt.Errorf("void has no host layout")
	default:
		return TypeLayout{}, fmt.Errorf("unsupported host type %s", t)
	}
}

func Field(l TypeLayout, index int) FieldLayout { return l.Fields[index] }
