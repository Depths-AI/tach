package spirv

import (
	"fmt"
	"tach/foundation"
)

func scalarKind(t *foundation.Type) foundation.TypeKind {
	if t != nil && t.Kind == foundation.VectorKind {
		return t.Elem.Kind
	}
	if t == nil {
		return foundation.InvalidKind
	}
	return t.Kind
}

func memoryAlignment(storage uint32, t *foundation.Type) (uint32, error) {
	if storage == StorageUniform || storage == StorageStorageBuffer {
		l, err := foundation.LayoutOf(t)
		if err != nil {
			return 0, err
		}
		return l.Align, nil
	}
	return logicalAlignment(t)
}

func logicalAlignment(t *foundation.Type) (uint32, error) {
	if t == nil {
		return 0, fmt.Errorf("nil type has no memory alignment")
	}
	switch t.Kind {
	case foundation.Float16Kind:
		return 2, nil
	case foundation.Int32Kind, foundation.Uint32Kind, foundation.Float32Kind, foundation.AtomicKind:
		return 4, nil
	case foundation.VectorKind:
		element, err := logicalAlignment(t.Elem)
		if err != nil {
			return 0, err
		}
		if t.Lanes == 2 {
			return element * 2, nil
		}
		return element * 4, nil
	case foundation.FixedArrayKind:
		return logicalAlignment(t.Elem)
	case foundation.StructKind:
		var align uint32
		for _, field := range t.Fields {
			fieldAlign, err := logicalAlignment(field.Type)
			if err != nil {
				return 0, err
			}
			if fieldAlign > align {
				align = fieldAlign
			}
		}
		if align == 0 {
			return 0, fmt.Errorf("struct %s has no aligned members", t)
		}
		return align, nil
	default:
		return 0, fmt.Errorf("type %s has no memory alignment", t)
	}
}

func (s *fnEmitter) memoryAccess(storage uint32, t *foundation.Type) (mask, align uint32, err error) {
	align, err = memoryAlignment(storage, t)
	if err != nil {
		return 0, 0, err
	}
	mask = MemoryAccessAligned
	if storage == StorageStorageBuffer || storage == StorageUniform || storage == StorageWorkgroup {
		mask |= MemoryAccessNonPrivatePointer
	}
	return mask, align, nil
}

func (s *fnEmitter) emitLoad(resultType, result, ptr, storage uint32, t *foundation.Type) error {
	mask, align, err := s.memoryAccess(storage, t)
	if err != nil {
		return err
	}
	emit(&s.b.functions, OpLoad, resultType, result, ptr, mask, align)
	return nil
}

func (s *fnEmitter) emitStore(ptr, value, storage uint32, t *foundation.Type) error {
	mask, align, err := s.memoryAccess(storage, t)
	if err != nil {
		return err
	}
	emit(&s.b.functions, OpStore, ptr, value, mask, align)
	return nil
}
