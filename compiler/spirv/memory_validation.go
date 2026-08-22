package spirv

import (
	"fmt"
)

func (v *validation) validateAtomic(in Instruction) error {
	a := in.Operands
	ctx := fmt.Sprintf("word %d %s", in.Offset, opName(in.Op))
	var resultType, resultID, ptrID, scopeID, semanticsID, unequalSemanticsID, valueID, comparatorID uint32
	switch in.Op {
	case OpAtomicLoad:
		resultType, resultID, ptrID, scopeID, semanticsID = a[0], a[1], a[2], a[3], a[4]
	case OpAtomicStore:
		ptrID, scopeID, semanticsID, valueID = a[0], a[1], a[2], a[3]
	case OpAtomicCompareExchange:
		resultType, resultID, ptrID, scopeID, semanticsID, unequalSemanticsID, valueID, comparatorID = a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7]
	default:
		resultType, resultID, ptrID, scopeID, semanticsID, valueID = a[0], a[1], a[2], a[3], a[4], a[5]
	}
	ptid, err := v.requireValue(ptrID, ctx)
	if err != nil {
		return err
	}
	pt := v.types[ptid]
	if pt == nil || pt.kind != typePointer || (pt.storage != StorageWorkgroup && pt.storage != StorageStorageBuffer) {
		return fmt.Errorf("%s pointer must be Workgroup or StorageBuffer", ctx)
	}
	pointee := v.types[pt.elem]
	if pointee == nil || pointee.kind != typeInt || pointee.width != 32 {
		return fmt.Errorf("%s pointer must point to a 32-bit integer", ctx)
	}
	if in.Op != OpAtomicStore {
		rt, err := v.requireType(resultType, ctx)
		if err != nil {
			return err
		}
		if rt.kind != typeInt || resultType != pt.elem {
			return fmt.Errorf("%s result type must equal the pointed-to integer type", ctx)
		}
		v.valueType[resultID] = resultType
	}
	if valueID != 0 {
		vt, err := v.requireValue(valueID, ctx)
		if err != nil {
			return err
		}
		if vt != pt.elem {
			return fmt.Errorf("%s value type does not match pointer", ctx)
		}
	}
	if comparatorID != 0 {
		comparatorType, err := v.requireValue(comparatorID, ctx)
		if err != nil {
			return err
		}
		if comparatorType != pt.elem {
			return fmt.Errorf("%s comparator type does not match pointer", ctx)
		}
	}
	scope, err := v.constantU32(scopeID, ctx+" scope")
	if err != nil {
		return err
	}
	wantScope := ScopeQueueFamily
	if pt.storage == StorageWorkgroup {
		wantScope = ScopeWorkgroup
	}
	if scope != wantScope {
		return fmt.Errorf("%s scope=%d, Tach requires %d for storage class %d", ctx, scope, wantScope, pt.storage)
	}
	sem, err := v.constantU32(semanticsID, ctx+" semantics")
	if err != nil {
		return err
	}
	if sem != MemorySemanticsRelaxed {
		return fmt.Errorf("%s memory semantics=0x%x, Tach atomics require Relaxed", ctx, sem)
	}
	if unequalSemanticsID != 0 {
		unequal, err := v.constantU32(unequalSemanticsID, ctx+" unequal semantics")
		if err != nil {
			return err
		}
		if unequal != MemorySemanticsRelaxed {
			return fmt.Errorf("%s unequal memory semantics=0x%x, Tach atomics require Relaxed", ctx, unequal)
		}
	}
	root := v.pointerRoot[ptrID]
	if root != 0 && v.decoration(root).nonWritable {
		return fmt.Errorf("%s accesses NonWritable resource %%%d", ctx, root)
	}
	if (in.Op == OpAtomicSMin || in.Op == OpAtomicSMax) && !pointee.signed {
		return fmt.Errorf("%s requires signed integer type", ctx)
	}
	if (in.Op == OpAtomicUMin || in.Op == OpAtomicUMax) && pointee.signed {
		return fmt.Errorf("%s requires unsigned integer type", ctx)
	}
	return nil
}

func (v *validation) validateMemoryAccess(ops []uint32, storage, pointee uint32) error {
	if len(ops) != 2 {
		return fmt.Errorf("memory access must be Aligned plus a power-of-two alignment")
	}
	mask, align := ops[0], ops[1]
	if mask&MemoryAccessAligned == 0 {
		return fmt.Errorf("memory access must be Aligned plus a power-of-two alignment")
	}
	want, err := v.accessAlignment(pointee, storage)
	if err != nil {
		return err
	}
	if align != want {
		return fmt.Errorf("aligned %d, want %d", align, want)
	}
	shared := storage == StorageStorageBuffer || storage == StorageUniform || storage == StorageWorkgroup
	if shared {
		if mask&MemoryAccessNonPrivatePointer == 0 {
			return fmt.Errorf("shared storage requires NonPrivatePointer")
		}
	} else if mask&MemoryAccessNonPrivatePointer != 0 {
		return fmt.Errorf("input access cannot use NonPrivatePointer")
	}
	if mask&^(MemoryAccessAligned|MemoryAccessNonPrivatePointer) != 0 {
		return fmt.Errorf("memory access mask 0x%x is outside Tach profile", mask)
	}
	return nil
}

func (v *validation) accessAlignment(id, storage uint32) (uint32, error) {
	return v.typeAlignment(id, storage == StorageUniform || storage == StorageStorageBuffer, map[uint32]bool{})
}

func (v *validation) typeAlignment(id uint32, host bool, seen map[uint32]bool) (uint32, error) {
	if seen[id] {
		return 0, fmt.Errorf("recursive type %%%d", id)
	}
	t := v.types[id]
	if t == nil {
		return 0, fmt.Errorf("%%%d is not a type", id)
	}
	seen[id] = true
	defer delete(seen, id)
	switch t.kind {
	case typeInt, typeFloat:
		return t.width / 8, nil
	case typeVector:
		element, err := v.typeAlignment(t.elem, host, seen)
		if err != nil {
			return 0, err
		}
		if t.lanes == 2 {
			return element * 2, nil
		}
		return element * 4, nil
	case typeArray, typeRuntimeArray:
		return v.typeAlignment(t.elem, host, seen)
	case typeStruct:
		if host {
			return 16, nil
		}
		var align uint32
		for _, member := range t.members {
			memberAlign, err := v.typeAlignment(member, host, seen)
			if err != nil {
				return 0, err
			}
			if memberAlign > align {
				align = memberAlign
			}
		}
		if align == 0 {
			return 0, fmt.Errorf("struct %%%d has no aligned members", id)
		}
		return align, nil
	default:
		return 0, fmt.Errorf("type %%%d has no memory alignment", id)
	}
}

func (v *validation) validateBarrier(in Instruction) error {
	a := in.Operands
	ctx := fmt.Sprintf("word %d OpControlBarrier", in.Offset)
	exec, err := v.constantU32(a[0], ctx+" execution scope")
	if err != nil {
		return err
	}
	mem, err := v.constantU32(a[1], ctx+" memory scope")
	if err != nil {
		return err
	}
	sem, err := v.constantU32(a[2], ctx+" semantics")
	if err != nil {
		return err
	}
	if exec != ScopeWorkgroup || mem != ScopeWorkgroup {
		return fmt.Errorf("%s requires Workgroup execution and memory scopes", ctx)
	}
	visible := MemorySemanticsAcquireRelease | MemorySemanticsMakeAvailable | MemorySemanticsMakeVisible
	wg := visible | MemorySemanticsWorkgroupMemory
	storage := visible | MemorySemanticsUniformMemory
	if sem != wg && sem != storage {
		return fmt.Errorf("%s semantics=0x%x outside Tach barrier profile", ctx, sem)
	}
	return nil
}

func (v *validation) validateCompositeConstruct(in Instruction) error {
	a := in.Operands
	ctx := fmt.Sprintf("word %d OpCompositeConstruct", in.Offset)
	t, err := v.requireType(a[0], ctx)
	if err != nil {
		return err
	}
	want := []uint32{}
	switch t.kind {
	case typeVector:
		for i := uint32(0); i < t.lanes; i++ {
			want = append(want, t.elem)
		}
	case typeStruct:
		want = t.members
	default:
		return fmt.Errorf("%s result type is not constructible composite", ctx)
	}
	if len(a[2:]) != len(want) {
		return fmt.Errorf("%s has %d constituents, want %d", ctx, len(a[2:]), len(want))
	}
	for i, id := range a[2:] {
		vt, err := v.requireValue(id, ctx)
		if err != nil {
			return err
		}
		if vt != want[i] {
			return fmt.Errorf("%s constituent %d type mismatch", ctx, i)
		}
	}
	v.valueType[a[1]] = a[0]
	return nil
}

func (v *validation) validateCompositeExtract(in Instruction) error {
	a := in.Operands
	ctx := fmt.Sprintf("word %d OpCompositeExtract", in.Offset)
	if _, err := v.requireType(a[0], ctx); err != nil {
		return err
	}
	bt, err := v.requireValue(a[2], ctx)
	if err != nil {
		return err
	}
	cur := bt
	for _, idx := range a[3:] {
		t := v.types[cur]
		if t == nil {
			return fmt.Errorf("%s indexes invalid type", ctx)
		}
		switch t.kind {
		case typeVector:
			if idx >= t.lanes {
				return fmt.Errorf("%s vector index out of range", ctx)
			}
			cur = t.elem
		case typeStruct:
			if int(idx) >= len(t.members) {
				return fmt.Errorf("%s struct index out of range", ctx)
			}
			cur = t.members[idx]
		default:
			return fmt.Errorf("%s cannot extract from this type", ctx)
		}
	}
	if cur != a[0] {
		return fmt.Errorf("%s result type mismatch", ctx)
	}
	v.valueType[a[1]] = a[0]
	return nil
}

func (v *validation) validateVectorExtractDynamic(in Instruction) error {
	a := in.Operands
	ctx := fmt.Sprintf("word %d OpVectorExtractDynamic", in.Offset)
	if _, err := v.requireType(a[0], ctx); err != nil {
		return err
	}
	bt, err := v.requireValue(a[2], ctx)
	if err != nil {
		return err
	}
	base := v.types[bt]
	if base == nil || base.kind != typeVector || base.elem != a[0] {
		return fmt.Errorf("%s base/result type mismatch", ctx)
	}
	it, err := v.requireValue(a[3], ctx)
	if err != nil {
		return err
	}
	index := v.types[it]
	if index == nil || index.kind != typeInt {
		return fmt.Errorf("%s index is not an integer scalar", ctx)
	}
	v.valueType[a[1]] = a[0]
	return nil
}
