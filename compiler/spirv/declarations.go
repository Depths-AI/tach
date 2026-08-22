package spirv

import (
	"fmt"
	"math"
	"strconv"
	"strings"
	"tach/foundation"
	"tach/ir"
)

func typeKey(t *foundation.Type) string {
	if t == nil {
		return "<nil>"
	}
	switch t.Kind {
	case foundation.VectorKind:
		return fmt.Sprintf("v%d:%s", t.Lanes, typeKey(t.Elem))
	case foundation.RuntimeArrayKind:
		return "ra:" + typeKey(t.Elem)
	case foundation.StructKind:
		return "s:" + t.Name
	default:
		return t.String()
	}
}

func normalizedTypeRole(t *foundation.Type, role typeRole) typeRole {
	if role == typeHostABI && t != nil {
		switch t.Kind {
		case foundation.FixedArrayKind, foundation.RuntimeArrayKind, foundation.StructKind:
			return typeHostABI
		}
	}
	return typeLogical
}

// typeID is the single SPIR-V type-lowering path. Logical types are used by
// SSA values and Workgroup memory. Host-ABI types exist only behind Uniform or
// StorageBuffer pointers and carry Tach's compiler-owned layout decorations.
func (b *builder) typeID(t *foundation.Type, role typeRole) (uint32, error) {
	role = normalizedTypeRole(t, role)
	key := fmt.Sprintf("%d:%s", role, typeKey(t))
	if id := b.types[key]; id != 0 {
		return id, nil
	}
	if t.Kind == foundation.AtomicKind {
		// SPIR-V atomicity is an operation property; the pointed-to object keeps
		// the underlying integer type. Resolve it before allocating an ID so the
		// module ID space remains dense and deterministic.
		elem, err := b.typeID(t.Elem, typeLogical)
		if err != nil {
			return 0, err
		}
		b.types[key] = elem
		return elem, nil
	}
	id := b.id()
	b.types[key] = id
	switch t.Kind {
	case foundation.VoidKind:
		emit(&b.typesGlobals, OpTypeVoid, id)
	case foundation.BoolKind:
		emit(&b.typesGlobals, OpTypeBool, id)
	case foundation.Int32Kind:
		emit(&b.typesGlobals, OpTypeInt, id, 32, 1)
	case foundation.Uint32Kind:
		emit(&b.typesGlobals, OpTypeInt, id, 32, 0)
	case foundation.Float16Kind:
		emit(&b.typesGlobals, OpTypeFloat, id, 16)
	case foundation.Float32Kind:
		emit(&b.typesGlobals, OpTypeFloat, id, 32)
	case foundation.VectorKind:
		elem, err := b.typeID(t.Elem, typeLogical)
		if err != nil {
			return 0, err
		}
		emit(&b.typesGlobals, OpTypeVector, id, elem, uint32(t.Lanes))
	case foundation.FixedArrayKind:
		elem, err := b.typeID(t.Elem, role)
		if err != nil {
			return 0, err
		}
		length, err := b.u32Constant(t.Count)
		if err != nil {
			return 0, err
		}
		emit(&b.typesGlobals, OpTypeArray, id, elem, length)
		if role == typeHostABI {
			l, err := foundation.LayoutOf(t)
			if err != nil {
				return 0, err
			}
			emit(&b.annotations, OpDecorate, id, DecorationArrayStride, l.Stride)
		}
	case foundation.RuntimeArrayKind:
		elem, err := b.typeID(t.Elem, role)
		if err != nil {
			return 0, err
		}
		emit(&b.typesGlobals, OpTypeRuntimeArray, id, elem)
		if role == typeHostABI {
			l, err := foundation.LayoutOf(t)
			if err != nil {
				return 0, err
			}
			emit(&b.annotations, OpDecorate, id, DecorationArrayStride, l.Stride)
		}
	case foundation.StructKind:
		members := []uint32{id}
		for _, f := range t.Fields {
			mid, err := b.typeID(f.Type, role)
			if err != nil {
				return 0, err
			}
			members = append(members, mid)
		}
		emit(&b.typesGlobals, OpTypeStruct, members...)
		if role == typeHostABI {
			l, err := foundation.LayoutOf(t)
			if err != nil {
				return 0, err
			}
			for i, fl := range l.Fields {
				emit(&b.annotations, OpMemberDecorate, id, uint32(i), DecorationOffset, fl.Offset)
			}
		}
	default:
		return 0, fmt.Errorf("unsupported SPIR-V type %s", t)
	}
	return id, nil
}

func typeRoleForStorage(storage uint32) typeRole {
	if storage == StorageUniform || storage == StorageStorageBuffer {
		return typeHostABI
	}
	return typeLogical
}

func (b *builder) pointerID(storage uint32, t *foundation.Type) (uint32, error) {
	pointee, err := b.typeID(t, typeRoleForStorage(storage))
	if err != nil {
		return 0, err
	}
	key := fmt.Sprintf("%d:%d", storage, pointee)
	if id := b.pointers[key]; id != 0 {
		return id, nil
	}
	id := b.id()
	b.pointers[key] = id
	emit(&b.typesGlobals, OpTypePointer, id, storage, pointee)
	return id, nil
}

func (b *builder) functionTypeID(ret *foundation.Type, params []ir.Param) (uint32, error) {
	rid, err := b.typeID(ret, typeLogical)
	if err != nil {
		return 0, err
	}
	ids := make([]uint32, len(params))
	var sb strings.Builder
	fmt.Fprintf(&sb, "%d", rid)
	for i, p := range params {
		ids[i], err = b.typeID(p.Type, typeLogical)
		if err != nil {
			return 0, err
		}
		fmt.Fprintf(&sb, ":%d", ids[i])
	}
	key := sb.String()
	if id := b.fnTypes[key]; id != 0 {
		return id, nil
	}
	id := b.id()
	b.fnTypes[key] = id
	ops := []uint32{id, rid}
	ops = append(ops, ids...)
	emit(&b.typesGlobals, OpTypeFunction, ops...)
	return id, nil
}

func (b *builder) constant(t *foundation.Type, raw string) (uint32, error) {
	tid, err := b.typeID(t, typeLogical)
	if err != nil {
		return 0, err
	}
	key := fmt.Sprintf("%d:%s", tid, raw)
	if id := b.constants[key]; id != 0 {
		return id, nil
	}
	id := b.id()
	switch t.Kind {
	case foundation.BoolKind:
		if raw == "true" {
			emit(&b.typesGlobals, OpConstantTrue, tid, id)
		} else if raw == "false" {
			emit(&b.typesGlobals, OpConstantFalse, tid, id)
		} else {
			return 0, fmt.Errorf("invalid bool constant %q", raw)
		}
	case foundation.Int32Kind:
		v, err := strconv.ParseInt(raw, 10, 32)
		if err != nil {
			return 0, err
		}
		emit(&b.typesGlobals, OpConstant, tid, id, uint32(int32(v)))
	case foundation.Uint32Kind:
		v, err := strconv.ParseUint(raw, 10, 32)
		if err != nil {
			return 0, err
		}
		emit(&b.typesGlobals, OpConstant, tid, id, uint32(v))
	case foundation.Float32Kind:
		v, err := strconv.ParseFloat(raw, 32)
		if err != nil {
			return 0, err
		}
		emit(&b.typesGlobals, OpConstant, tid, id, math.Float32bits(float32(v)))
	case foundation.Float16Kind:
		v, err := strconv.ParseFloat(raw, 64)
		bits, ok := foundation.Float16Bits(v)
		if err != nil || !ok {
			return 0, fmt.Errorf("invalid float16 constant %q", raw)
		}
		emit(&b.typesGlobals, OpConstant, tid, id, uint32(bits))
	default:
		return 0, fmt.Errorf("constant type %s is not scalar", t)
	}
	b.constants[key] = id
	return id, nil
}

func (b *builder) u32Constant(v uint32) (uint32, error) {
	return b.constant(foundation.Uint32Type, strconv.FormatUint(uint64(v), 10))
}

func (b *builder) nullConstant(t *foundation.Type) (uint32, error) {
	tid, err := b.typeID(t, typeLogical)
	if err != nil {
		return 0, err
	}
	key := fmt.Sprintf("%d:null", tid)
	if id := b.constants[key]; id != 0 {
		return id, nil
	}
	id := b.id()
	b.constants[key] = id
	emit(&b.typesGlobals, OpConstantNull, tid, id)
	return id, nil
}
