package spirv

import (
	"fmt"
	"maps"
	"sort"
	"strings"
)

func isTerminator(op Op) bool {
	switch op {
	case OpBranch, OpBranchConditional, OpReturn, OpReturnValue, OpUnreachable:
		return true
	}
	return false
}

func (v *validation) validateFunctions() error {
	// Reconstruct block instruction lists including labels' body operations, CFG,
	// phi predecessor sets, merge placement, function signatures, and SSA dominance.
	funcOfInst := map[int]*functionInfo{}
	blockOfInst := map[int]uint32{}
	var cur *functionInfo
	var block uint32
	for i, in := range v.m.Instructions {
		switch in.Op {
		case OpFunction:
			cur = v.functions[in.Operands[1]]
			block = 0
		case OpLabel:
			block = in.Operands[0]
			if cur != nil {
				funcOfInst[i] = cur
				blockOfInst[i] = block
			}
		case OpFunctionEnd:
			cur = nil
			block = 0
		default:
			if cur != nil {
				funcOfInst[i] = cur
				blockOfInst[i] = block
			}
		}
	}
	for _, f := range v.functions {
		ft := v.types[f.fnType]
		if ft == nil || ft.kind != typeFunction {
			return fmt.Errorf("function %%%d has invalid function type", f.id)
		}
		if ft.ret != f.ret || len(ft.params) != len(f.params) {
			return fmt.Errorf("function %%%d signature mismatches OpTypeFunction", f.id)
		}
		for i, p := range f.params {
			if v.valueType[p] != ft.params[i] {
				return fmt.Errorf("function %%%d parameter %d type mismatch", f.id, i)
			}
		}
		for _, label := range f.order {
			bl := f.blocks[label]
			if len(bl.insts) == 0 {
				return fmt.Errorf("function %%%d block %%%d is empty", f.id, label)
			}
			seenNonPhi := false
			terminated := false
			for pos, idx := range bl.insts {
				in := v.m.Instructions[idx]
				if terminated {
					return fmt.Errorf("block %%%d has instruction after terminator", label)
				}
				if in.Op == OpPhi {
					if seenNonPhi {
						return fmt.Errorf("block %%%d has OpPhi after non-phi instruction", label)
					}
				} else {
					seenNonPhi = true
				}
				if in.Op == OpSelectionMerge || in.Op == OpLoopMerge {
					if pos+1 >= len(bl.insts) {
						return fmt.Errorf("block %%%d merge instruction lacks following branch", label)
					}
					next := v.m.Instructions[bl.insts[pos+1]].Op
					if next != OpBranch && next != OpBranchConditional {
						return fmt.Errorf("block %%%d merge must immediately precede branch", label)
					}
				}
				if isTerminator(in.Op) {
					terminated = true
					switch in.Op {
					case OpBranch:
						bl.succ = []uint32{in.Operands[0]}
					case OpBranchConditional:
						bl.succ = []uint32{in.Operands[1], in.Operands[2]}
					}
				}
			}
			if !terminated {
				return fmt.Errorf("function %%%d block %%%d lacks terminator", f.id, label)
			}
		}
		for _, label := range f.order {
			for _, dst := range f.blocks[label].succ {
				db := f.blocks[dst]
				if db == nil {
					return fmt.Errorf("function %%%d branches to label %%%d outside function", f.id, dst)
				}
				db.pred = append(db.pred, label)
			}
		}
		for _, label := range f.order {
			bl := f.blocks[label]
			sort.Slice(bl.pred, func(i, j int) bool { return bl.pred[i] < bl.pred[j] })
			for _, idx := range bl.phis {
				a := v.m.Instructions[idx].Operands
				var incoming []uint32
				for i := 3; i < len(a); i += 2 {
					incoming = append(incoming, a[i])
				}
				sort.Slice(incoming, func(i, j int) bool { return incoming[i] < incoming[j] })
				if !equalU32(incoming, bl.pred) {
					return fmt.Errorf("function %%%d block %%%d phi predecessors %v do not match CFG predecessors %v", f.id, label, incoming, bl.pred)
				}
			}
		}
		if err := v.validateReturnTypes(f); err != nil {
			return err
		}
		if err := v.validateDominance(f, funcOfInst, blockOfInst); err != nil {
			return err
		}
	}
	return nil
}
func equalU32(a, b []uint32) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func (v *validation) validateReturnTypes(f *functionInfo) error {
	rt := v.types[f.ret]
	for _, label := range f.order {
		for _, idx := range f.blocks[label].insts {
			in := v.m.Instructions[idx]
			if in.Op == OpReturn && rt.kind != typeVoid {
				return fmt.Errorf("function %%%d returns void but return type is non-void", f.id)
			}
			if in.Op == OpReturnValue {
				if rt.kind == typeVoid {
					return fmt.Errorf("void function %%%d uses OpReturnValue", f.id)
				}
				vt := v.valueType[in.Operands[0]]
				if vt != f.ret {
					return fmt.Errorf("function %%%d return value type mismatch", f.id)
				}
			}
		}
	}
	return nil
}

func (v *validation) validateDominance(f *functionInfo, funcOfInst map[int]*functionInfo, blockOfInst map[int]uint32) error {
	if len(f.order) == 0 {
		return nil
	}
	entry := f.order[0]
	reachable := map[uint32]bool{}
	var walk func(uint32)
	walk = func(x uint32) {
		if reachable[x] {
			return
		}
		reachable[x] = true
		for _, y := range f.blocks[x].succ {
			walk(y)
		}
	}
	walk(entry)
	// Standard iterative dominator sets.
	dom := map[uint32]map[uint32]bool{}
	for _, l := range f.order {
		dom[l] = map[uint32]bool{}
		if l == entry {
			dom[l][l] = true
		} else if reachable[l] {
			for _, x := range f.order {
				if reachable[x] {
					dom[l][x] = true
				}
			}
		}
	}
	changed := true
	for changed {
		changed = false
		for _, l := range f.order {
			if l == entry || !reachable[l] {
				continue
			}
			preds := f.blocks[l].pred
			var next map[uint32]bool
			for _, p := range preds {
				if !reachable[p] {
					continue
				}
				if next == nil {
					next = cloneSet(dom[p])
				} else {
					for x := range next {
						if !dom[p][x] {
							delete(next, x)
						}
					}
				}
			}
			if next == nil {
				next = map[uint32]bool{}
			}
			next[l] = true
			if !setEq(next, dom[l]) {
				dom[l] = next
				changed = true
			}
		}
	}
	// Local definition location/order. Function parameters use block 0 and globals
	// are absent from this map, both dominating every function block.
	type loc struct {
		block uint32
		pos   int
	}
	defs := map[uint32]loc{}
	for _, p := range f.params {
		defs[p] = loc{}
	}
	for _, l := range f.order {
		for pos, idx := range f.blocks[l].insts {
			if id := resultID(v.m.Instructions[idx]); id != 0 {
				defs[id] = loc{l, pos}
			}
		}
	}
	// Labels and function id are not ordinary values.
	for _, l := range f.order {
		for pos, idx := range f.blocks[l].insts {
			in := v.m.Instructions[idx]
			for _, use := range valueUses(in) {
				d, local := defs[use]
				if !local {
					continue
				}
				if d.block == 0 {
					continue
				}
				if in.Op == OpPhi {
					continue
				}
				if d.block == l {
					if d.pos >= pos {
						return fmt.Errorf("function %%%d value %%%d does not precede its use in block %%%d", f.id, use, l)
					}
				} else if !dom[l][d.block] {
					return fmt.Errorf("function %%%d value %%%d defined in block %%%d does not dominate use in %%%d", f.id, use, d.block, l)
				}
			}
			if in.Op == OpPhi {
				a := in.Operands
				for j := 2; j < len(a); j += 2 {
					use, pred := a[j], a[j+1]
					d, local := defs[use]
					if !local || d.block == 0 {
						continue
					}
					if d.block == pred {
						continue
					}
					if !dom[pred][d.block] {
						return fmt.Errorf("function %%%d phi value %%%d does not dominate predecessor %%%d", f.id, use, pred)
					}
				}
			}
		}
	}
	return nil
}
func cloneSet(s map[uint32]bool) map[uint32]bool {
	return maps.Clone(s)
}
func setEq(a, b map[uint32]bool) bool {
	return maps.Equal(a, b)
}

// valueUses returns only ordinary SSA/object value operands; type ids, function
// ids, labels, and decoration targets are intentionally excluded.
func valueUses(in Instruction) []uint32 {
	a := in.Operands
	switch in.Op {
	case OpFunctionCall:
		return append([]uint32(nil), a[3:]...)
	case OpExtInst:
		return append([]uint32(nil), a[4:]...)
	case OpDot:
		return []uint32{a[2], a[3]}
	case OpLoad:
		return []uint32{a[2]}
	case OpStore:
		return []uint32{a[0], a[1]}
	case OpAtomicLoad:
		return []uint32{a[2], a[3], a[4]}
	case OpAtomicStore:
		return []uint32{a[0], a[1], a[2], a[3]}
	case OpAtomicExchange, OpAtomicIAdd, OpAtomicISub, OpAtomicSMin, OpAtomicUMin, OpAtomicSMax, OpAtomicUMax, OpAtomicAnd, OpAtomicOr, OpAtomicXor:
		return []uint32{a[2], a[3], a[4], a[5]}
	case OpAtomicCompareExchange:
		return []uint32{a[2], a[3], a[4], a[5], a[6], a[7]}
	case OpControlBarrier:
		return []uint32{a[0], a[1], a[2]}
	case OpAccessChain:
		return append([]uint32{a[2]}, a[3:]...)
	case OpArrayLength:
		return []uint32{a[2]}
	case OpCompositeConstruct:
		return append([]uint32(nil), a[2:]...)
	case OpVectorExtractDynamic:
		return []uint32{a[2], a[3]}
	case OpCompositeExtract:
		return []uint32{a[2]}
	case OpConvertFToU, OpConvertFToS, OpConvertSToF, OpConvertUToF, OpFConvert, OpBitcast, OpSNegate, OpFNegate, OpLogicalNot, OpNot, OpAny, OpAll:
		return []uint32{a[2]}
	case OpIAdd, OpFAdd, OpISub, OpFSub, OpIMul, OpFMul, OpUDiv, OpSDiv, OpFDiv, OpUMod, OpSRem, OpFRem, OpVectorTimesScalar, OpLogicalEqual, OpLogicalNotEqual, OpLogicalOr, OpLogicalAnd, OpIEqual, OpINotEqual, OpUGreaterThan, OpSGreaterThan, OpUGreaterThanEqual, OpSGreaterThanEqual, OpULessThan, OpSLessThan, OpULessThanEqual, OpSLessThanEqual, OpFOrdEqual, OpFOrdNotEqual, OpFOrdLessThan, OpFOrdGreaterThan, OpFOrdLessThanEqual, OpFOrdGreaterThanEqual, OpShiftRightLogical, OpShiftRightArithmetic, OpShiftLeftLogical, OpBitwiseOr, OpBitwiseXor, OpBitwiseAnd:
		return []uint32{a[2], a[3]}
	case OpSelect:
		return []uint32{a[2], a[3], a[4]}
	case OpBranchConditional:
		return []uint32{a[0]}
	case OpReturnValue:
		return []uint32{a[0]}
	}
	return nil
}

func (v *validation) validateEntryPoints() error {
	for fid, name := range v.entryPoints {
		f := v.functions[fid]
		if f == nil {
			return fmt.Errorf("entry point %q references non-function %%%d", name, fid)
		}
		rt := v.types[f.ret]
		if rt == nil || rt.kind != typeVoid {
			return fmt.Errorf("entry point %q must return void", name)
		}
		if len(f.params) != 0 {
			return fmt.Errorf("entry point %q cannot have function parameters", name)
		}
		if _, ok := v.localSize[fid]; !ok {
			return fmt.Errorf("entry point %q lacks LocalSize", name)
		}
		decl := map[uint32]bool{}
		for _, in := range v.m.Instructions {
			if in.Op == OpEntryPoint && in.Operands[1] == fid {
				_, next, _ := literalString(in.Operands, 2)
				var previous uint32
				for index, id := range in.Operands[next:] {
					if index > 0 && id <= previous {
						return fmt.Errorf("entry point %q interface ids are not unique and ascending", name)
					}
					decl[id] = true
					previous = id
				}
			}
		}
		used, err := v.entryGlobals(fid, map[uint32]bool{}, map[uint32]bool{})
		if err != nil {
			return fmt.Errorf("entry point %q: %w", name, err)
		}
		if !setEq(decl, used) {
			return fmt.Errorf("entry point %q interface %v does not exactly match statically used globals %v", name, keys(decl), keys(used))
		}
	}
	for fid := range v.localSize {
		if _, ok := v.entryPoints[fid]; !ok {
			return fmt.Errorf("LocalSize applied to non-entry function %%%d", fid)
		}
	}
	return nil
}

func (v *validation) entryGlobals(fid uint32, visiting, visited map[uint32]bool) (map[uint32]bool, error) {
	used := map[uint32]bool{}
	var walk func(uint32) error
	walk = func(functionID uint32) error {
		if visiting[functionID] {
			return fmt.Errorf("recursive static call graph at function %%%d", functionID)
		}
		if visited[functionID] {
			return nil
		}
		function := v.functions[functionID]
		if function == nil {
			return fmt.Errorf("static call graph references non-function %%%d", functionID)
		}
		visiting[functionID] = true
		for _, label := range function.order {
			for _, index := range function.blocks[label].insts {
				instruction := v.m.Instructions[index]
				for _, operand := range valueUses(instruction) {
					if root := v.pointerRoot[operand]; root != 0 {
						used[root] = true
					}
				}
				if instruction.Op == OpFunctionCall {
					if err := walk(instruction.Operands[2]); err != nil {
						return err
					}
				}
			}
		}
		delete(visiting, functionID)
		visited[functionID] = true
		return nil
	}
	return used, walk(fid)
}
func keys(m map[uint32]bool) []uint32 {
	r := make([]uint32, 0, len(m))
	for x := range m {
		r = append(r, x)
	}
	sort.Slice(r, func(i, j int) bool { return r[i] < r[j] })
	return r
}

// Summary returns a compact deterministic validation summary useful in CLI and
// tests without exposing validator internals.
func Summary(data []byte) (string, error) {
	m, err := Decode(data)
	if err != nil {
		return "", err
	}
	if err := Validate(data); err != nil {
		return "", err
	}
	entries := []string{}
	for _, in := range m.Instructions {
		if in.Op == OpEntryPoint {
			name, _, _ := literalString(in.Operands, 2)
			entries = append(entries, name)
		}
	}
	sort.Strings(entries)
	return fmt.Sprintf("SPIR-V 1.6: %d words, bound %d, entries [%s]", len(data)/4, m.Bound, strings.Join(entries, ", ")), nil
}

func Disassemble(data []byte) (string, error) {
	if err := Validate(data); err != nil {
		return "", err
	}
	m, _ := Decode(data)
	var out strings.Builder
	fmt.Fprintf(&out, "; SPIR-V 1.6\n; Bound %d\n", m.Bound)
	for _, instruction := range m.Instructions {
		fmt.Fprintf(&out, "%04d: %s", instruction.Offset, opName(instruction.Op))
		for _, operand := range instruction.Operands {
			fmt.Fprintf(&out, " %d", operand)
		}
		out.WriteByte('\n')
	}
	return out.String(), nil
}
