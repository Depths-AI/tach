package spirv

import (
	"fmt"
)

func (v *validation) validateDecorationsAndABI() error {
	// Validate every explicit host-layout decoration that is present. Whether a
	// layout is required or forbidden is a property of the global variable that
	// reaches the type, not of arrays/structs globally.
	memo := map[uint32]abiLayout{}
	visiting := map[uint32]bool{}
	for id, t := range v.types {
		d := v.decoration(id)
		if d.block && t.kind != typeStruct {
			return fmt.Errorf("type %%%d has Block decoration but is not a struct", id)
		}
		if d.arrayStride != nil {
			if t.kind != typeArray && t.kind != typeRuntimeArray {
				return fmt.Errorf("type %%%d has ArrayStride but is not an array", id)
			}
			el, err := v.abiOf(t.elem, memo, visiting)
			if err != nil {
				return err
			}
			want := roundUp(el.align, el.size)
			if *d.arrayStride != want {
				return fmt.Errorf("array %%%d ArrayStride=%d, Tach ABI requires %d", id, *d.arrayStride, want)
			}
		}
		if t.kind == typeStruct && (d.block || len(d.offsets) > 0) {
			l, err := v.abiOf(id, memo, visiting)
			if err != nil {
				return err
			}
			_ = l
		}
	}

	pairs := map[[2]uint32][]uint32{}
	storage16, uniform16 := false, false
	for id, storage := range v.globalVars {
		d := v.decoration(id)
		vt := v.types[v.valueType[id]]
		if vt == nil || vt.kind != typePointer {
			return fmt.Errorf("global variable %%%d lacks pointer type", id)
		}
		switch storage {
		case StorageInput:
			if d.builtin == nil {
				return fmt.Errorf("input variable %%%d lacks BuiltIn decoration", id)
			}
			if d.binding != nil || d.set != nil {
				return fmt.Errorf("input builtin %%%d cannot have descriptor decorations", id)
			}
			switch *d.builtin {
			case BuiltInNumWorkgroups, BuiltInWorkgroupID, BuiltInLocalInvocationID, BuiltInGlobalInvocationID, BuiltInLocalInvocationIndex:
			default:
				return fmt.Errorf("input %%%d uses builtin %d outside Tach profile", id, *d.builtin)
			}
		case StorageUniform, StorageStorageBuffer:
			if d.binding == nil || d.set == nil {
				return fmt.Errorf("descriptor variable %%%d requires DescriptorSet and Binding", id)
			}
			pair := [2]uint32{*d.set, *d.binding}
			for _, prev := range pairs[pair] {
				for _, function := range v.functions {
					if v.functionUsesGlobal(function, prev) && v.functionUsesGlobal(function, id) {
						return fmt.Errorf("descriptor variables %%%d and %%%d share set=%d binding=%d in function %%%d", prev, id, pair[0], pair[1], function.id)
					}
				}
			}
			pairs[pair] = append(pairs[pair], id)
			st := v.types[vt.elem]
			if st == nil || st.kind != typeStruct || !v.decoration(vt.elem).block {
				return fmt.Errorf("descriptor variable %%%d must point to Block struct", id)
			}
			if err := v.requireHostABILayout(vt.elem, map[uint32]bool{}); err != nil {
				return fmt.Errorf("descriptor variable %%%d: %w", id, err)
			}
			if _, err := v.abiOf(vt.elem, memo, visiting); err != nil {
				return fmt.Errorf("descriptor variable %%%d: %w", id, err)
			}
			if storage == StorageUniform && containsRuntime(vt.elem, v.types, map[uint32]bool{}) {
				return fmt.Errorf("uniform descriptor %%%d contains runtime array", id)
			}
			if containsFloat16(vt.elem, v.types, map[uint32]bool{}) {
				if storage == StorageUniform {
					uniform16 = true
				} else {
					storage16 = true
				}
			}
		case StorageWorkgroup:
			if d.builtin != nil || d.binding != nil || d.set != nil || d.nonWritable {
				return fmt.Errorf("workgroup variable %%%d has invalid interface/descriptor decoration", id)
			}
			if err := v.rejectWorkgroupExplicitLayout(vt.elem, map[uint32]bool{}); err != nil {
				return fmt.Errorf("workgroup variable %%%d: %w", id, err)
			}
		default:
			return fmt.Errorf("global variable %%%d storage class %d outside Tach profile", id, storage)
		}
	}
	if storage16 != (v.capabilities[CapabilityStorageBuffer16BitAccess] == 1) || uniform16 != (v.capabilities[CapabilityUniformAndStorage16BitAccess] == 1) {
		return fmt.Errorf("16-bit storage capabilities do not match descriptor types")
	}
	return nil
}

func (v *validation) functionUsesGlobal(function *functionInfo, global uint32) bool {
	for _, label := range function.order {
		for _, index := range function.blocks[label].insts {
			for _, operand := range valueUses(v.m.Instructions[index]) {
				if v.pointerRoot[operand] == global {
					return true
				}
			}
		}
	}
	return false
}

func (v *validation) requireHostABILayout(id uint32, seen map[uint32]bool) error {
	if seen[id] {
		return nil
	}
	seen[id] = true
	t := v.types[id]
	if t == nil {
		return fmt.Errorf("host ABI references unknown type %%%d", id)
	}
	d := v.decoration(id)
	switch t.kind {
	case typeArray, typeRuntimeArray:
		if d.arrayStride == nil {
			return fmt.Errorf("host array type %%%d lacks ArrayStride", id)
		}
		return v.requireHostABILayout(t.elem, seen)
	case typeStruct:
		if len(d.offsets) != len(t.members) {
			return fmt.Errorf("host struct %%%d has %d Offset decorations for %d members", id, len(d.offsets), len(t.members))
		}
		for i, member := range t.members {
			if _, ok := d.offsets[uint32(i)]; !ok {
				return fmt.Errorf("host struct %%%d member %d lacks Offset", id, i)
			}
			if err := v.requireHostABILayout(member, seen); err != nil {
				return err
			}
		}
	}
	return nil
}

func (v *validation) rejectWorkgroupExplicitLayout(id uint32, seen map[uint32]bool) error {
	if seen[id] {
		return nil
	}
	seen[id] = true
	t := v.types[id]
	if t == nil {
		return fmt.Errorf("references unknown type %%%d", id)
	}
	d := v.decoration(id)
	if d.block {
		return fmt.Errorf("type %%%d carries Block explicit layout", id)
	}
	if d.arrayStride != nil {
		return fmt.Errorf("type %%%d carries ArrayStride explicit layout", id)
	}
	if len(d.offsets) > 0 {
		return fmt.Errorf("type %%%d carries Offset explicit layout", id)
	}
	switch t.kind {
	case typeArray, typeRuntimeArray, typeVector:
		return v.rejectWorkgroupExplicitLayout(t.elem, seen)
	case typeStruct:
		for _, member := range t.members {
			if err := v.rejectWorkgroupExplicitLayout(member, seen); err != nil {
				return err
			}
		}
	}
	return nil
}

type abiLayout struct {
	size, align, stride uint32
	runtime             bool
}

func roundUp(a, v uint32) uint32 {
	if a == 0 {
		return v
	}
	return (v + a - 1) &^ (a - 1)
}
func max32(a, b uint32) uint32 {
	if a > b {
		return a
	}
	return b
}
func (v *validation) abiOf(id uint32, memo map[uint32]abiLayout, vis map[uint32]bool) (abiLayout, error) {
	if l, ok := memo[id]; ok {
		return l, nil
	}
	if vis[id] {
		return abiLayout{}, fmt.Errorf("recursive SPIR-V type %%%d", id)
	}
	vis[id] = true
	defer delete(vis, id)
	t := v.types[id]
	if t == nil {
		return abiLayout{}, fmt.Errorf("ABI references unknown type %%%d", id)
	}
	var l abiLayout
	switch t.kind {
	case typeInt, typeFloat:
		l = abiLayout{size: t.width / 8, align: t.width / 8}
	case typeVector:
		e := v.types[t.elem]
		if e == nil || (e.kind != typeInt && e.kind != typeFloat) {
			return l, fmt.Errorf("vector %%%d has non-host element", id)
		}
		width := e.width / 8
		switch t.lanes {
		case 2:
			l = abiLayout{size: width * 2, align: width * 2}
		case 3:
			l = abiLayout{size: width * 3, align: width * 4}
		case 4:
			l = abiLayout{size: width * 4, align: width * 4}
		default:
			return l, fmt.Errorf("invalid vector width")
		}
	case typeRuntimeArray:
		e, err := v.abiOf(t.elem, memo, vis)
		if err != nil {
			return l, err
		}
		l = abiLayout{align: e.align, stride: roundUp(e.align, e.size), runtime: true}
	case typeArray:
		e, err := v.abiOf(t.elem, memo, vis)
		if err != nil {
			return l, err
		}
		count, ok := v.constants[t.length]
		if !ok || count == 0 {
			return l, fmt.Errorf("array %%%d has invalid length constant", id)
		}
		stride := roundUp(e.align, e.size)
		l = abiLayout{align: e.align, stride: stride, size: stride * uint32(count)}
	case typeStruct:
		d := v.decoration(id)
		align := uint32(16)
		off := uint32(0)
		runtime := false
		if len(d.offsets) != len(t.members) {
			return l, fmt.Errorf("struct %%%d has %d Offset decorations for %d members", id, len(d.offsets), len(t.members))
		}
		for i, mid := range t.members {
			ml, err := v.abiOf(mid, memo, vis)
			if err != nil {
				return l, err
			}
			if runtime {
				return l, fmt.Errorf("runtime array in struct %%%d is not final", id)
			}
			req := ml.align
			mt := v.types[mid]
			if mt.kind == typeStruct {
				req = max32(req, 16)
			}
			want := roundUp(req, off)
			got, ok := d.offsets[uint32(i)]
			if !ok || got != want {
				return l, fmt.Errorf("struct %%%d member %d Offset=%d, Tach ABI requires %d", id, i, got, want)
			}
			if ml.runtime {
				runtime = true
			} else {
				sz := ml.size
				if mt.kind == typeStruct {
					sz = roundUp(16, sz)
				}
				off = want + sz
			}
			align = max32(align, req)
		}
		l = abiLayout{align: align, runtime: runtime}
		if runtime {
			l.size = off
		} else {
			l.size = roundUp(align, off)
		}
	default:
		return l, fmt.Errorf("type %%%d is not in Tach host ABI", id)
	}
	memo[id] = l
	return l, nil
}
func containsRuntime(id uint32, ts map[uint32]*typeInfo, seen map[uint32]bool) bool {
	if seen[id] {
		return false
	}
	seen[id] = true
	t := ts[id]
	if t == nil {
		return false
	}
	if t.kind == typeRuntimeArray {
		return true
	}
	if t.kind == typeArray {
		return containsRuntime(t.elem, ts, seen)
	}
	if t.kind == typeStruct {
		for _, m := range t.members {
			if containsRuntime(m, ts, seen) {
				return true
			}
		}
	}
	return false
}

func containsFloat16(id uint32, ts map[uint32]*typeInfo, seen map[uint32]bool) bool {
	if seen[id] {
		return false
	}
	seen[id] = true
	t := ts[id]
	if t == nil {
		return false
	}
	if t.kind == typeFloat {
		return t.width == 16
	}
	if t.elem != 0 && containsFloat16(t.elem, ts, seen) {
		return true
	}
	for _, member := range t.members {
		if containsFloat16(member, ts, seen) {
			return true
		}
	}
	return false
}
