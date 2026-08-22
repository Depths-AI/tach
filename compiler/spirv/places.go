package spirv

import (
	"fmt"
	"tach/foundation"
	"tach/ir"
)

func (s *fnEmitter) value(id ir.ValueID) (uint32, error) {
	v := s.values[id]
	if v == 0 {
		return 0, fmt.Errorf("undefined IR value %%%d", id)
	}
	return v, nil
}

func (s *fnEmitter) place(id ir.PlaceID) (spvPlace, error) {
	p, ok := s.places[id]
	if !ok {
		return spvPlace{}, fmt.Errorf("undefined IR place &p%d", id)
	}
	return p, nil
}

func (s *fnEmitter) def(irID ir.ValueID, spvID uint32, t *foundation.Type) {
	if irID != 0 {
		s.values[irID] = spvID
		s.vtypes[irID] = t
	}
}

func (s *fnEmitter) accessField(base spvPlace, field int, t *foundation.Type) (spvPlace, error) {
	if base.ty == nil || base.ty.Kind != foundation.StructKind {
		return spvPlace{}, fmt.Errorf("field access base is not a struct place")
	}
	if field < 0 || field >= len(base.ty.Fields) || !foundation.Equal(base.ty.Fields[field].Type, t) {
		return spvPlace{}, fmt.Errorf("field %d does not match place type %s", field, base.ty)
	}
	ptrType, err := s.b.pointerID(base.storage, t)
	if err != nil {
		return spvPlace{}, err
	}
	idx, err := s.b.u32Constant(uint32(field))
	if err != nil {
		return spvPlace{}, err
	}
	id := s.b.id()
	emit(&s.b.functions, OpAccessChain, ptrType, id, base.ptr, idx)
	p := spvPlace{ptr: id, ty: t, storage: base.storage, resource: base.resource}
	if t.Kind == foundation.RuntimeArrayKind {
		p.arrayBase = base.ptr
		p.arrayMember = uint32(field)
		p.hasArrayLen = true
	}
	return p, nil
}

// loadPlace keeps physical host-layout aggregates behind descriptor pointers.
// A constructible resource struct is loaded field-by-field into its one logical
// SSA type; Workgroup and ordinary values use that logical type directly.
func (s *fnEmitter) loadPlace(p spvPlace, t *foundation.Type) (uint32, error) {
	if !foundation.IsConstructible(t) {
		return 0, fmt.Errorf("cannot load non-constructible place type %s", t)
	}
	role := typeRoleForStorage(p.storage)
	if role == typeHostABI && t.Kind == foundation.StructKind {
		tid, err := s.b.typeID(t, typeLogical)
		if err != nil {
			return 0, err
		}
		ops := []uint32{tid, s.b.id()}
		for i, f := range t.Fields {
			fp, err := s.accessField(p, i, f.Type)
			if err != nil {
				return 0, err
			}
			v, err := s.loadPlace(fp, f.Type)
			if err != nil {
				return 0, err
			}
			ops = append(ops, v)
		}
		emit(&s.b.functions, OpCompositeConstruct, ops...)
		return ops[1], nil
	}

	physical, err := s.b.typeID(t, role)
	if err != nil {
		return 0, err
	}
	logical, err := s.b.typeID(t, typeLogical)
	if err != nil {
		return 0, err
	}
	if physical != logical {
		return 0, fmt.Errorf("place type %s requires structural loading", t)
	}
	id := s.b.id()
	if err := s.emitLoad(logical, id, p.ptr, p.storage, t); err != nil {
		return 0, err
	}
	return id, nil
}

// storePlace is the exact inverse of loadPlace: logical resource structs are
// decomposed into host-layout fields, while logical Workgroup values are stored
// directly. No physical aggregate is admitted into the SSA value domain.
func (s *fnEmitter) storePlace(p spvPlace, value uint32) error {
	t := p.ty
	if !foundation.IsConstructible(t) {
		return fmt.Errorf("cannot store non-constructible place type %s", t)
	}
	role := typeRoleForStorage(p.storage)
	if role == typeHostABI && t.Kind == foundation.StructKind {
		for i, f := range t.Fields {
			fieldType, err := s.b.typeID(f.Type, typeLogical)
			if err != nil {
				return err
			}
			fieldValue := s.b.id()
			emit(&s.b.functions, OpCompositeExtract, fieldType, fieldValue, value, uint32(i))
			fp, err := s.accessField(p, i, f.Type)
			if err != nil {
				return err
			}
			if err := s.storePlace(fp, fieldValue); err != nil {
				return err
			}
		}
		return nil
	}

	physical, err := s.b.typeID(t, role)
	if err != nil {
		return err
	}
	logical, err := s.b.typeID(t, typeLogical)
	if err != nil {
		return err
	}
	if physical != logical {
		return fmt.Errorf("place type %s requires structural storage", t)
	}
	return s.emitStore(p.ptr, value, p.storage, t)
}
